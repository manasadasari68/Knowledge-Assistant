"""Load all PDF files from the repository knowledge directory into PostgreSQL."""

import argparse
from pathlib import Path

from sqlalchemy import select

from app.database import get_session_factory
from app.models import Document, DocumentChunk
from app.services.pdf_ingestion import extract_pdf_chunks

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[4] / "knowledge"


def ingest_directory(knowledge_dir: Path) -> int:
    pdf_paths = sorted(knowledge_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found under {knowledge_dir}")
        return 0

    session_factory = get_session_factory()
    processed = 0
    with session_factory() as session:
        for pdf_path in pdf_paths:
            relative_path = pdf_path.relative_to(knowledge_dir).as_posix()
            chunks = extract_pdf_chunks(pdf_path)
            document = session.scalar(
                select(Document).where(Document.filename == relative_path)
            )

            if document is None:
                document = Document(
                    filename=relative_path,
                    title=pdf_path.stem.replace("_", " "),
                    document_metadata={"source_path": relative_path},
                )
                session.add(document)
                session.flush()
            else:
                document.title = pdf_path.stem.replace("_", " ")
                document.document_metadata = {"source_path": relative_path}
                session.query(DocumentChunk).filter_by(document_id=document.id).delete()

            for index, chunk in enumerate(chunks):
                session.add(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_text=chunk.text,
                        chunk_index=index,
                        chunk_metadata={
                            **chunk.metadata,
                            "source_filename": pdf_path.name,
                        },
                        embedding=None,
                    )
                )
            session.commit()
            processed += 1
            print(f"Ingested {relative_path}: {len(chunks)} chunks")

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF knowledge documents")
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_KNOWLEDGE_DIR,
        help="Directory to scan recursively (defaults to repository knowledge/)",
    )
    args = parser.parse_args()
    ingest_directory(args.directory.resolve())


if __name__ == "__main__":
    main()
