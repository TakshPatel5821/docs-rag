"""Corpus ingestion: read files -> chunk -> embed in batches -> persist."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.chunking import chunk_text
from app.config import get_settings
from app.db import delete_chunks, insert_chunks, upsert_document
from app.embeddings import Embedder

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass
class IngestReport:
    source: str
    document_id: int
    chunk_count: int
    seconds: float

    def __str__(self) -> str:
        rate = self.chunk_count / self.seconds if self.seconds else 0.0
        return (
            f"{self.source}: {self.chunk_count} chunks in {self.seconds:.1f}s "
            f"({rate:.1f} chunks/s)"
        )


def read_document(path: Path) -> str:
    """Extract plain text from a .txt/.md/.pdf file."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def derive_title(path: Path, text: str) -> str:
    """Use the first substantial line as the title, else the filename."""
    for line in text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 160:
            return line
    return path.stem


def iter_corpus(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def ingest_file(conn, embedder: Embedder, path: Path) -> IngestReport:
    """Ingest one file. Re-ingesting the same source replaces its chunks."""
    settings = get_settings()
    started = time.perf_counter()

    text = read_document(path)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

    # One encode() call per batch of 32, not one per chunk -- roughly 10x faster
    # on CPU because the model amortises the forward pass across the batch.
    vectors = embedder.embed_documents([c.content for c in chunks])
    if vectors and len(vectors[0]) != settings.embedding_dim:
        raise ValueError(
            f"Embedder returned {len(vectors[0])} dims but the schema expects "
            f"{settings.embedding_dim}. Update EMBEDDING_DIM and the VECTOR(n) column."
        )

    document_id = upsert_document(conn, source=str(path), title=derive_title(path, text))
    delete_chunks(conn, document_id)
    insert_chunks(
        conn,
        document_id,
        [(c.index, c.content, v) for c, v in zip(chunks, vectors)],
    )
    conn.commit()

    return IngestReport(
        source=str(path),
        document_id=document_id,
        chunk_count=len(chunks),
        seconds=time.perf_counter() - started,
    )


def ingest_directory(conn, embedder: Embedder, directory: Path) -> list[IngestReport]:
    reports = []
    for path in iter_corpus(directory):
        reports.append(ingest_file(conn, embedder, path))
    return reports
