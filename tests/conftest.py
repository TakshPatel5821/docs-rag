"""Shared fixtures and doubles.

No test in this suite talks to a real database or a real LLM. Postgres is
replaced by ``FakeConnection`` (which records the SQL it was handed and replays
canned rows), the model by ``HashEmbedder``, and generation by
``RecordingProvider``. The one test module that does need Postgres --
``test_integration_pgvector.py`` -- skips itself unless a DSN is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.embeddings import HashEmbedder
from app.generation.base import LLMProvider
from app.retrieval import RetrievedChunk


@dataclass
class FakeCursor:
    rows: list[tuple]
    statements: list[tuple[str, Any]]

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> "FakeCursor":
        self.statements.append((sql, params))
        return self

    def executemany(self, sql: str, params_seq: Any) -> "FakeCursor":
        for params in params_seq:
            self.statements.append((sql, params))
        return self

    def fetchall(self) -> list[tuple]:
        return list(self.rows)

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None


@dataclass
class FakeConnection:
    """Minimal stand-in for a psycopg connection."""

    rows: list[tuple] = field(default_factory=list)
    statements: list[tuple[str, Any]] = field(default_factory=list)
    commits: int = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.rows, self.statements)

    def execute(self, sql: str, params: Any = None) -> FakeCursor:
        return self.cursor().execute(sql, params)

    def commit(self) -> None:
        self.commits += 1

    def sql_log(self) -> str:
        return "\n".join(sql for sql, _ in self.statements)


class RecordingProvider(LLMProvider):
    """LLM double: records every prompt, returns a fixed answer."""

    name = "recording"

    def __init__(self, answer: str = "A grounded answer [1]."):
        self.answer = answer
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.answer


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=384)


@pytest.fixture
def provider() -> RecordingProvider:
    return RecordingProvider()


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=1,
            chunk_index=0,
            content="Apple relies on single-source suppliers for certain components.",
            source="data/filings/AAPL_10-K.txt",
            title="AAPL Form 10-K",
            distance=0.12,
        ),
        RetrievedChunk(
            chunk_id=2,
            chunk_index=7,
            content="JPMorgan discusses credit risk concentration in its loan book.",
            source="data/filings/JPM_10-K.txt",
            title="JPM Form 10-K",
            distance=0.31,
        ),
    ]
