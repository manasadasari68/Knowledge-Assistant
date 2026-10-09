"""PDF text extraction with section-aware, paragraph-preserving chunks."""

from dataclasses import dataclass, field
from pathlib import Path
import re
import statistics
from typing import Any

from pypdf import PdfReader

CHUNK_SIZE = 850
CHUNK_OVERLAP = 120
MIN_CHUNK_SIZE = 180
HEADING_SIZE_MARGIN = 1.5


@dataclass(frozen=True)
class ExtractedChunk:
    text: str
    metadata: dict[str, int]


@dataclass
class _Section:
    heading: str
    page_start: int
    page_end: int
    fragments: list[str] = field(default_factory=list)


def _normalise_body(fragments: list[str]) -> str:
    raw_text = "".join(fragments)
    raw_text = re.sub(r"(?<!\n)\n(?!\n)", " ", raw_text)
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n+", raw_text)]
    return "\n\n".join(part for part in paragraphs if part)


def _split_words(text: str, limit: int) -> list[str]:
    """Split an unusually long sentence on word boundaries, never mid-word."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[index : index + limit] for index in range(0, len(word), limit))
        elif not current:
            current = word
        elif len(current) + len(word) + 1 <= limit:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _split_oversized_paragraph(paragraph: str, limit: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            pieces.append(sentence)
        else:
            pieces.extend(_split_words(sentence, limit))
    return pieces


def _split_section_body(body: str, limit: int) -> list[str]:
    """Pack whole paragraphs/sentences; overlap only at sentence boundaries."""
    units: list[tuple[str, bool]] = []
    for paragraph in body.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append((paragraph, True))
        else:
            units.extend((sentence, False) for sentence in _split_oversized_paragraph(paragraph, limit))

    chunks: list[str] = []
    current: list[tuple[str, bool]] = []
    current_size = 0

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(text for text, _ in current))

    for text, is_paragraph in units:
        separator_size = 2 if current and is_paragraph else 1 if current else 0
        if current and current_size + separator_size + len(text) > limit:
            flush()
            previous_text = current[-1][0]
            overlap = previous_text if len(previous_text) <= CHUNK_OVERLAP else ""
            current = [(overlap, False)] if overlap else []
            current_size = len(overlap)
            separator_size = 1 if current else 0
        current.append((text, is_paragraph))
        current_size += separator_size + len(text)
    flush()

    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_SIZE:
        combined = f"{chunks[-2]}\n\n{chunks[-1]}"
        if len(combined) <= limit:
            chunks[-2:] = [combined]
    return chunks


def _section_chunks(
    document_title: str, sections: list[_Section]
) -> list[ExtractedChunk]:
    chunks: list[ExtractedChunk] = []
    for section in sections:
        body = _normalise_body(section.fragments)
        if not body:
            continue

        heading = f"{document_title}\n\n{section.heading}".strip()
        body_limit = max(MIN_CHUNK_SIZE, CHUNK_SIZE - len(heading) - 2)
        body_chunks = (
            [body]
            if len(body) <= body_limit
            else _split_section_body(body, body_limit)
        )
        for body_chunk in body_chunks:
            chunks.append(
                ExtractedChunk(
                    text=f"{heading}\n{body_chunk}",
                    metadata={
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    },
                )
            )
    return chunks


def _extract_with_font_headings(pdf_path: Path) -> list[ExtractedChunk]:
    reader = PdfReader(str(pdf_path))
    sections: list[_Section] = []
    current: _Section | None = None
    document_title = ""

    def flush() -> None:
        nonlocal current
        if current is not None:
            sections.append(current)
            current = None

    for page_number, page in enumerate(reader.pages, start=1):
        spans: list[tuple[str, float]] = []
        page.extract_text(
            visitor_text=lambda text, _cm, _tm, _font, font_size: spans.append(
                (text, float(font_size or 0))
            )
        )
        body_sizes = [
            size
            for text, size in spans
            if len(re.sub(r"\s+", " ", text).strip()) >= 45 and size > 0
        ]
        if not body_sizes:
            continue
        body_size = statistics.median(body_sizes)
        heading_threshold = body_size + HEADING_SIZE_MARGIN

        for text, font_size in spans:
            cleaned = text.strip()
            if not cleaned or font_size < body_size - 0.5:
                continue

            if font_size > heading_threshold:
                heading = re.sub(r"\s+", " ", cleaned)
                if not document_title:
                    document_title = heading
                    continue
                if heading == document_title:
                    continue
                flush()
                current = _Section(
                    heading=heading,
                    page_start=page_number,
                    page_end=page_number,
                )
            elif current is not None:
                current.page_end = page_number
                current.fragments.append(text)

    flush()
    return _section_chunks(document_title or pdf_path.stem.replace("_", " "), sections)


def _extract_without_font_headings(pdf_path: Path) -> list[ExtractedChunk]:
    """Fallback for PDFs whose text has no useful font/style information."""
    reader = PdfReader(str(pdf_path))
    chunks: list[ExtractedChunk] = []
    document_title = pdf_path.stem.replace("_", " ")
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", page_text)
        ]
        text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
        for part in _split_section_body(text, CHUNK_SIZE - len(document_title) - 2):
            chunks.append(
                ExtractedChunk(
                    text=f"{document_title}\n{part}",
                    metadata={"page_start": page_number, "page_end": page_number},
                )
            )
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> list[ExtractedChunk]:
    """Extract PDF text into heading-aware chunks with page metadata."""
    chunks = _extract_with_font_headings(pdf_path)
    return chunks if chunks else _extract_without_font_headings(pdf_path)
