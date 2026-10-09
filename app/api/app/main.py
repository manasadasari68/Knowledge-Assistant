from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.chat import router as chat_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.speech_to_text import router as speech_to_text_router
from app.api.routes.text_to_speech import router as text_to_speech_router

app = FastAPI(
    title="Utility Knowledge Assistant API",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
app.include_router(speech_to_text_router)
app.include_router(text_to_speech_router)
