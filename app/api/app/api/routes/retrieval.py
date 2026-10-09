from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.database import get_session_factory
from app.services.embeddings import EmbeddingError, EmbeddingService
from app.services.retrieval import retrieve_chunks

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


class RetrievalTestRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalTestResponse(BaseModel):
    question: str
    results: list[dict[str, Any]]


def _run_retrieval(question: str, top_k: int) -> RetrievalTestResponse:
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    try:
        embedding_service = EmbeddingService()
        with get_session_factory()() as session:
            results = retrieve_chunks(session, question, embedding_service, top_k)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RetrievalTestResponse(question=question, results=results)


@router.get("/test", response_model=RetrievalTestResponse)
def test_retrieval_get(
    question: str = Query(min_length=1), top_k: int = Query(default=5, ge=1, le=20)
) -> RetrievalTestResponse:
    return _run_retrieval(question, top_k)


@router.post("/test", response_model=RetrievalTestResponse)
def test_retrieval_post(request: RetrievalTestRequest) -> RetrievalTestResponse:
    return _run_retrieval(request.question, request.top_k)
