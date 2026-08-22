"""Real Postgres + pgvector round trip.

Skipped unless a DSN is provided, so the default `pytest` run needs no services:

    DOCS_RAG_TEST_DSN=postgresql://docsrag:docsrag@localhost:5432/docsrag pytest -m integration

It uses HashEmbedder, so it still needs no model download -- what is under test
here is the SQL, the vector column and the ordering, not semantic quality.
"""

from __future__ import annotations

import os

import pytest

from app.chunking import chunk_text
from app.db import build_ivfflat_index, delete_chunks, init_schema, insert_chunks, upsert_document
from app.retrieval import search

DSN = os.environ.get("DOCS_RAG_TEST_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="set DOCS_RAG_TEST_DSN to run"),
]

CORPUS = {
    "supply-chain.txt": "The Company depends on single-source suppliers concentrated "
    "in Asia and a disruption would materially affect operations.",
    "credit-risk.txt": "Concentrations of credit risk arise from exposures to "
    "counterparties in the same industry or geographic region.",
    "dividends.txt": "The board declared a quarterly cash dividend per common share "
    "payable to shareholders of record.",
}


@pytest.fixture
def conn(embedder):
    from app.db import connect

    with connect(DSN) as connection:
        init_schema(connection)
        for name, text in CORPUS.items():
            document_id = upsert_document(connection, f"test://{name}", name)
            delete_chunks(connection, document_id)
            chunks = chunk_text(text, 800, 150)
            insert_chunks(
                connection,
                document_id,
                [
                    (c.index, c.content, v)
                    for c, v in zip(
                        chunks, embedder.embed_documents([c.content for c in chunks])
                    )
                ],
            )
        connection.commit()
        yield connection
        connection.execute("DELETE FROM documents WHERE source LIKE 'test://%'")
        connection.commit()


def test_vector_search_returns_the_relevant_document_first(conn, embedder):
    results = search(conn, embedder, "single-source suppliers disruption", k=3)
    assert results
    assert "supply-chain.txt" in results[0].source
    assert 0.0 <= results[0].distance <= 2.0


def test_results_come_back_ordered_and_attributed(conn, embedder):
    results = search(conn, embedder, "credit risk concentration", k=3)
    assert [r.distance for r in results] == sorted(r.distance for r in results)
    assert all(r.source.startswith("test://") for r in results)


def test_ivfflat_index_can_be_built_and_queried(conn, embedder):
    lists = build_ivfflat_index(conn, lists=1)
    assert lists == 1
    row = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
        ("chunks_embedding_ivfflat_idx",),
    ).fetchone()
    assert row and "ivfflat" in row[0] and "vector_cosine_ops" in row[0]
    assert search(conn, embedder, "quarterly cash dividend", k=1)
