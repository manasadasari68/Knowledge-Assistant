"""Grounded answer generation using the existing vector retrieval service."""

from sqlalchemy.orm import Session

from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.retrieval import retrieve_chunks

INSUFFICIENT_CONTEXT_ANSWER = (
    "The documented knowledge base does not provide enough information to answer this question."
)
def _relevant_sources(chunks: list[dict[str, object]]) -> list[dict[str, str]]:
    """Return each source that contributed a selected context chunk once."""
    sources: list[dict[str, str]] = []
    seen_sources: set[tuple[str, str]] = set()
    for chunk in chunks:
        source = (str(chunk["title"]), str(chunk["filename"]))
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append({"title": source[0], "filename": source[1]})
    return sources


def answer_question(session: Session, question: str) -> dict[str, object]:
    embedding_service = EmbeddingService()
    chunks = retrieve_chunks(session, question, embedding_service)
    if not chunks:
        return {"answer": INSUFFICIENT_CONTEXT_ANSWER, "sources": []}

    answer = LLMService().generate_answer(question, chunks)
    return {"answer": answer, "sources": _relevant_sources(chunks)}
