from __future__ import annotations

import io
import wave
from functools import lru_cache

import numpy as np
import whisper
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

router = APIRouter(prefix="/api/speech-to-text", tags=["speech"])

SAMPLE_RATE = 16_000
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class SpeechToTextResponse(BaseModel):
    text: str


@lru_cache(maxsize=1)
def _get_model():
    return whisper.load_model("base")


def _decode_wav(payload: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(payload), "rb") as audio_file:
            if (
                audio_file.getnchannels() != 1
                or audio_file.getsampwidth() != 2
                or audio_file.getframerate() != SAMPLE_RATE
                or audio_file.getcomptype() != "NONE"
            ):
                raise HTTPException(
                    status_code=415,
                    detail="Audio must be mono 16 kHz PCM WAV.",
                )
            audio = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype="<i2")
    except (wave.Error, EOFError) as exc:
        raise HTTPException(status_code=400, detail="The recorded audio could not be read.") from exc

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="No audio was recorded.")
    return audio.astype(np.float32) / 32768.0


def _transcribe(audio: np.ndarray) -> str:
    result = _get_model().transcribe(audio, fp16=False)
    return str(result.get("text", "")).strip()


@router.post("", response_model=SpeechToTextResponse)
async def speech_to_text(request: Request) -> SpeechToTextResponse:
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="No audio was received.")
    if len(payload) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="The recording is too large. Please record a shorter clip.")

    audio = _decode_wav(payload)
    try:
        transcription = await run_in_threadpool(_transcribe, audio)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Local speech transcription is unavailable. Ensure the Whisper base model is installed.",
        ) from exc
    return SpeechToTextResponse(text=transcription)
