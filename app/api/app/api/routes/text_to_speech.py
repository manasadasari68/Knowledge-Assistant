from __future__ import annotations

import io
import wave
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from piper import PiperVoice
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/text-to-speech", tags=["speech"])

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VOICE_MODEL_PATH = PROJECT_ROOT / "en_US-lessac-medium.onnx"
VOICE_CONFIG_PATH = PROJECT_ROOT / "en_US-lessac-medium.onnx.json"


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1)


@lru_cache(maxsize=1)
def _get_voice() -> PiperVoice:
    if not VOICE_MODEL_PATH.is_file() or not VOICE_CONFIG_PATH.is_file():
        raise FileNotFoundError("The local Piper voice model or configuration is missing.")
    return PiperVoice.load(
        model_path=VOICE_MODEL_PATH,
        config_path=VOICE_CONFIG_PATH,
    )


def _synthesize_wav(text: str) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        _get_voice().synthesize_wav(text, wav_file)
    return output.getvalue()


@router.post("")
async def text_to_speech(request: TextToSpeechRequest) -> Response:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    try:
        audio = await run_in_threadpool(_synthesize_wav, text)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Local Piper text-to-speech is unavailable. Check the installed Piper voice model.",
        ) from exc

    return Response(content=audio, media_type="audio/wav")
