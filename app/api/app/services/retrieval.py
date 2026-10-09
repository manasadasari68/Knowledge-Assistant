"""Vector search over stored document chunks."""

import math
import os
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.services.embeddings import EmbeddingService

DEFAULT_RETRIEVAL_TOP_K = 8
MAX_RETRIEVAL_TOP_K = 20
DEFAULT_RETRIEVAL_MIN_SIMILARITY = 0.25
DEFAULT_RETRIEVAL_MIN_KEYWORD_SCORE = 0.10
RRF_RANK_OFFSET = 60


def _configured_top_k() -> int:
    raw_value = os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_RETRIEVAL_TOP_K))
    try:
        top_k = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("RETRIEVAL_TOP_K must be an integer between 1 and 20") from exc
    if not 1 <= top_k <= MAX_RETRIEVAL_TOP_K:
        raise RuntimeError("RETRIEVAL_TOP_K must be an integer between 1 and 20")
    return top_k


def _configured_min_similarity() -> float:
    raw_value = os.getenv(
        "RETRIEVAL_MIN_SIMILARITY", str(DEFAULT_RETRIEVAL_MIN_SIMILARITY)
    )
    try:
        minimum = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("RETRIEVAL_MIN_SIMILARITY must be between -1 and 1") from exc
    if not math.isfinite(minimum) or not -1.0 <= minimum <= 1.0:
        raise RuntimeError("RETRIEVAL_MIN_SIMILARITY must be between -1 and 1")
    return minimum


def _configured_min_keyword_score() -> float:
    raw_value = os.getenv(
        "RETRIEVAL_MIN_KEYWORD_SCORE", str(DEFAULT_RETRIEVAL_MIN_KEYWORD_SCORE)
    )
    try:
        minimum = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("RETRIEVAL_MIN_KEYWORD_SCORE must be between 0 and 1") from exc
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise RuntimeError("RETRIEVAL_MIN_KEYWORD_SCORE must be between 0 and 1")
    return minimum


def retrieve_chunks(
    session: Session,
    question: str,
    embedding_service: EmbeddingService,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    result_limit = _configured_top_k() if top_k is None else top_k
    similarity_floor = (
        _configured_min_similarity() if min_similarity is None else min_similarity
    )
    keyword_floor = _configured_min_keyword_score()
    if not 1 <= result_limit <= MAX_RETRIEVAL_TOP_K:
        raise ValueError("top_k must be between 1 and 20")
    if not math.isfinite(similarity_floor) or not -1.0 <= similarity_floor <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")

    question_embedding = embedding_service.embed_texts([question])[0]
    distance = DocumentChunk.embedding.cosine_distance(question_embedding)
    base_conditions = (
        DocumentChunk.embedding.is_not(None),
        DocumentChunk.embedding_model == embedding_service.model,
    )
    candidate_limit = min(100, max(result_limit * 3, 20))
    semantic_statement = (
        select(DocumentChunk, Document, distance.label("cosine_distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(*base_conditions)
        .order_by(distance)
        .limit(candidate_limit)
    )

    candidates: dict[int, dict[str, Any]] = {}

    def record_candidate(
        row: Any,
        *,
        semantic_rank: int | None = None,
        keyword_rank: int | None = None,
        keyword_score: float | None = None,
    ) -> None:
        chunk, document, cosine_distance = row[:3]
        similarity = 1.0 - float(cosine_distance)
        candidate = candidates.setdefault(
            chunk.id,
            {
                "document_id": document.id,
                "filename": document.filename,
                "title": document.title,
                "document_metadata": document.document_metadata,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "chunk_metadata": chunk.chunk_metadata,
                "similarity_score": similarity,
                "semantic_rank": None,
                "keyword_rank": None,
                "keyword_score": 0.0,
            },
        )
        if semantic_rank is not None:
            candidate["semantic_rank"] = semantic_rank
        if keyword_rank is not None:
            candidate["keyword_rank"] = keyword_rank
            candidate["keyword_score"] = float(keyword_score or 0.0)

    for rank, row in enumerate(session.execute(semantic_statement), start=1):
        record_candidate(row, semantic_rank=rank)

    # PostgreSQL's English text-search config normalizes inflections. Build an
    # OR query from the question's generic word tokens so paraphrased semantic
    # matches and exact terminology can contribute independently.
    terms = re.findall(r"[^\W_]+", question.lower(), flags=re.UNICODE)
    lexical_query_text = " | ".join(term for term in terms if len(term) > 1)
    if lexical_query_text:
        text_vector = func.to_tsvector("english", DocumentChunk.chunk_text)
        text_query = func.to_tsquery("english", lexical_query_text)
        keyword_score = func.ts_rank_cd(text_vector, text_query, 32)
        keyword_statement = (
            select(
                DocumentChunk,
                Document,
                distance.label("cosine_distance"),
                keyword_score.label("keyword_score"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                *base_conditions,
                text_vector.op("@@")(text_query),
            )
            .order_by(keyword_score.desc(), distance)
            .limit(candidate_limit)
        )
        for rank, row in enumerate(session.execute(keyword_statement), start=1):
            score = float(row[3])
            if score >= keyword_floor:
                record_candidate(
                    row,
                    keyword_rank=rank,
                    keyword_score=score,
                )

    eligible = [
        candidate
        for candidate in candidates.values()
        if candidate["similarity_score"] >= similarity_floor
        or (
            candidate["keyword_rank"] is not None
            and candidate["keyword_score"] >= keyword_floor
        )
    ]

    def fusion_score(candidate: dict[str, Any]) -> float:
        score = 0.0
        if candidate["semantic_rank"] is not None:
            score += 1.0 / (RRF_RANK_OFFSET + candidate["semantic_rank"])
        if candidate["keyword_rank"] is not None:
            score += 1.0 / (RRF_RANK_OFFSET + candidate["keyword_rank"])
        return score

    eligible.sort(
        key=lambda candidate: (
            fusion_score(candidate),
            candidate["similarity_score"],
            candidate["keyword_score"],
        ),
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    seen_chunk_ids: set[int] = set()
    seen_chunk_texts: set[str] = set()
    for candidate in eligible:
        chunk_id = candidate["chunk_id"]
        canonical_text = re.sub(r"\s+", " ", candidate["chunk_text"]).strip().casefold()
        if chunk_id in seen_chunk_ids or canonical_text in seen_chunk_texts:
            continue
        seen_chunk_ids.add(chunk_id)
        seen_chunk_texts.add(canonical_text)
        results.append(
            {
                key: candidate[key]
                for key in (
                    "document_id",
                    "filename",
                    "title",
                    "document_metadata",
                    "chunk_id",
                    "chunk_index",
                    "chunk_text",
                    "chunk_metadata",
                    "similarity_score",
                )
            }
        )
        if len(results) >= result_limit:
            break
    return results
