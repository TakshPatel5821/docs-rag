"""HTTP surface: request validation, response shape, error mapping."""

from __future__ import annotations

import contextlib

import pytest
from fastapi.testclient import TestClient

from app import api
from tests.conftest import FakeConnection, RecordingProvider

ROWS = [
    (
        1,
        4,
        "The Company depends on single-source component suppliers.",
        "data/filings/AAPL_10-K_2025-10-31.txt",
        "AAPL Form 10-K filed 2025-10-31",
        0.18,
    )
]


@pytest.fixture
def client(monkeypatch, embedder):
    @contextlib.contextmanager
    def fake_connect(dsn=None):
        yield FakeConnection(rows=ROWS)

    monkeypatch.setattr(api, "connect", fake_connect)
    monkeypatch.setattr(api, "get_embedder", lambda *a, **k: embedder)
    monkeypatch.setattr(api, "get_provider", lambda *a, **k: RecordingProvider())
    monkeypatch.setattr(api, "count_chunks", lambda conn: 4242)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_corpus_size(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 4242


def test_query_returns_answer_and_sources(client):
    response = client.post("/query", json={"question": "supply chain risk?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A grounded answer [1]."
    assert body["sources"][0]["source"].endswith("AAPL_10-K_2025-10-31.txt")
    assert body["sources"][0]["similarity"] == pytest.approx(0.82)
    assert body["provider"] == "recording"


def test_query_rejects_an_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_rejects_an_out_of_range_top_k(client):
    response = client.post("/query", json={"question": "valid?", "top_k": 99})
    assert response.status_code == 422


def test_generation_failure_maps_to_502(client, monkeypatch):
    from app.generation.base import GenerationError

    def boom(*args, **kwargs):
        raise GenerationError("upstream is down")

    monkeypatch.setattr(api, "answer_question", boom)
    response = client.post("/query", json={"question": "supply chain risk?"})
    assert response.status_code == 502
    assert "upstream is down" in response.json()["detail"]


def test_openapi_schema_documents_the_query_endpoint(client):
    schema = client.get("/openapi.json").json()
    assert "/query" in schema["paths"]
    assert "post" in schema["paths"]["/query"]
