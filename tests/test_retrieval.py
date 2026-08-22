"""Retrieval: it is vector search, it returns k, and it carries attribution."""

from __future__ import annotations

from app.retrieval import RetrievedChunk, search

ROWS = [
    (1, 0, "closest chunk", "data/filings/AAPL_10-K.txt", "AAPL Form 10-K", 0.05),
    (2, 3, "next chunk", "data/filings/AAPL_10-K.txt", "AAPL Form 10-K", 0.21),
    (3, 9, "third chunk", "data/filings/JPM_10-K.txt", "JPM Form 10-K", 0.44),
    (4, 1, "fourth chunk", "data/filings/MSFT_10-K.txt", "MSFT Form 10-K", 0.68),
    (5, 2, "fifth chunk", "data/filings/BAC_10-K.txt", "BAC Form 10-K", 0.90),
]


def test_returns_k_results(embedder):
    from tests.conftest import FakeConnection

    conn = FakeConnection(rows=ROWS)
    results = search(conn, embedder, "supply chain risk", k=5)
    assert len(results) == 5
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_results_are_ordered_by_ascending_distance(embedder):
    from tests.conftest import FakeConnection

    results = search(FakeConnection(rows=ROWS), embedder, "question", k=5)
    distances = [r.distance for r in results]
    assert distances == sorted(distances)


def test_similarity_is_one_minus_cosine_distance(embedder):
    from tests.conftest import FakeConnection

    results = search(FakeConnection(rows=ROWS), embedder, "question", k=5)
    assert results[0].similarity == 1.0 - 0.05
    assert results[0].similarity > results[-1].similarity


def test_every_result_carries_its_source_document(embedder):
    from tests.conftest import FakeConnection

    results = search(FakeConnection(rows=ROWS), embedder, "question", k=5)
    assert all(r.source for r in results)
    assert {r.source for r in results} == {row[3] for row in ROWS}


def test_query_uses_cosine_vector_search_not_keyword_matching(embedder):
    from tests.conftest import FakeConnection

    conn = FakeConnection(rows=ROWS)
    search(conn, embedder, "supply chain risk", k=5)
    sql = conn.sql_log()
    assert "<=>" in sql, "must use pgvector's cosine distance operator"
    assert "ORDER BY" in sql
    assert "LIMIT" in sql
    assert "LIKE" not in sql.upper()


def test_k_is_passed_through_to_the_database(embedder):
    from tests.conftest import FakeConnection

    conn = FakeConnection(rows=ROWS[:2])
    search(conn, embedder, "question", k=2)
    params = [p for sql, p in conn.statements if p is not None]
    assert params and params[-1][-1] == 2


def test_the_query_is_embedded_and_sent_as_a_vector(embedder):
    from tests.conftest import FakeConnection

    conn = FakeConnection(rows=ROWS)
    search(conn, embedder, "supply chain risk", k=5)
    _, params = [(s, p) for s, p in conn.statements if p is not None][-1]
    literal = params[0]
    assert literal.startswith("[") and literal.endswith("]")
    assert len(literal.split(",")) == embedder.dim


def test_probes_are_set_for_the_ivfflat_scan(embedder):
    from tests.conftest import FakeConnection

    conn = FakeConnection(rows=ROWS)
    search(conn, embedder, "question", k=5, probes=7)
    assert "SET LOCAL ivfflat.probes = 7" in conn.sql_log()


def test_blank_question_short_circuits(embedder):
    from tests.conftest import FakeConnection

    conn = FakeConnection(rows=ROWS)
    assert search(conn, embedder, "   ", k=5) == []
    assert conn.statements == [], "no database round trip for an empty question"
