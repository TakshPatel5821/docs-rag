"""Ingestion: text extraction, chunk persistence, dimension enforcement."""

from __future__ import annotations

import pytest

from app.ingest import derive_title, ingest_file, iter_corpus, read_document
from tests.conftest import FakeConnection

FILING = (
    "AAPL Form 10-K filed 2025-10-31 (accession 0000320193-25-000079)\n"
    "Source: https://www.sec.gov/Archives/edgar/data/320193/aapl.htm\n\n"
    + "\n\n".join(
        f"Item 1A. Risk Factor {i}. " + ("The Company depends on suppliers. " * 20)
        for i in range(6)
    )
)


@pytest.fixture
def filing(tmp_path):
    path = tmp_path / "AAPL_10-K_2025-10-31.txt"
    path.write_text(FILING, encoding="utf-8")
    return path


def test_reads_plain_text(filing):
    assert read_document(filing).startswith("AAPL Form 10-K filed 2025-10-31")


def test_rejects_unsupported_file_types(tmp_path):
    path = tmp_path / "filing.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_document(path)


def test_title_comes_from_the_first_substantial_line(filing):
    title = derive_title(filing, FILING)
    assert title.startswith("AAPL Form 10-K filed 2025-10-31")


def test_title_falls_back_to_the_filename(tmp_path):
    path = tmp_path / "unnamed.txt"
    assert derive_title(path, "\n\n") == "unnamed"


def test_corpus_iteration_picks_up_supported_files_only(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-")
    (tmp_path / "c.zip").write_bytes(b"PK")
    assert [p.name for p in iter_corpus(tmp_path)] == ["a.txt", "b.pdf"]


def test_ingest_writes_one_row_per_chunk(filing, embedder):
    conn = FakeConnection(rows=[(1,)])
    report = ingest_file(conn, embedder, filing)

    assert report.chunk_count > 1
    inserts = [
        params for sql, params in conn.statements if "INSERT INTO chunks" in sql
    ]
    assert len(inserts) == report.chunk_count
    assert conn.commits == 1


def test_ingested_rows_carry_index_content_and_vector(filing, embedder):
    conn = FakeConnection(rows=[(1,)])
    ingest_file(conn, embedder, filing)

    inserts = [p for sql, p in conn.statements if "INSERT INTO chunks" in sql]
    indexes = [params[1] for params in inserts]
    assert indexes == list(range(len(inserts)))
    for _, _, content, vector in inserts:
        assert content.strip()
        assert vector.startswith("[") and len(vector.split(",")) == embedder.dim


def test_reingesting_replaces_the_previous_chunks(filing, embedder):
    conn = FakeConnection(rows=[(1,)])
    ingest_file(conn, embedder, filing)
    assert "DELETE FROM chunks" in conn.sql_log()


def test_dimension_mismatch_fails_loudly(filing):
    from app.embeddings import HashEmbedder

    with pytest.raises(ValueError, match="dims but the schema expects"):
        ingest_file(FakeConnection(rows=[(1,)]), HashEmbedder(dim=128), filing)
