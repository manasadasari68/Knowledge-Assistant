"""Generate and persist embeddings for stored document chunks."""

import argparse

from sqlalchemy import or_, select

from app.database import get_session_factory
from app.models import DocumentChunk
from app.services.embeddings import EmbeddingError, EmbeddingService

BATCH_SIZE = 64


def embed_stored_chunks(force: bool = False, batch_size: int = BATCH_SIZE) -> int:
    service = EmbeddingService()
    factory = get_session_factory()
    embedded = 0
    with factory() as session:
        statement = select(DocumentChunk.id).order_by(DocumentChunk.id)
        if not force:
            statement = statement.where(
                or_(
                    DocumentChunk.embedding.is_(None),
                    DocumentChunk.embedding_model.is_(None),
                    DocumentChunk.embedding_model != service.model,
                )
            )
        chunk_ids = list(session.scalars(statement))

        for start in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[start : start + batch_size]
            chunks = list(
                session.scalars(
                    select(DocumentChunk)
                    .where(DocumentChunk.id.in_(batch_ids))
                    .order_by(DocumentChunk.id)
                )
            )
            vectors = service.embed_texts([chunk.chunk_text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
                chunk.embedding_model = service.model
            session.commit()
            embedded += len(chunks)
            print(f"Embedded {embedded}/{len(chunk_ids)} chunks")

    print(f"Finished embedding {embedded} chunks with {service.model}")
    return embedded


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed document chunks in PostgreSQL")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all chunks, including those already using the configured model",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        embed_stored_chunks(force=args.force, batch_size=args.batch_size)
    except EmbeddingError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
