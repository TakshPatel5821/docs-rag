"""PostgreSQL + pgvector access.

Everything here takes an explicit connection so callers (tests, CLI, API) decide
lifetime and transaction boundaries.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "001_schema.sql"


def to_pgvector(vector: Sequence[float]) -> str:
    """Render a Python sequence as a pgvector literal.

    Passed with an explicit ``::vector`` cast rather than relying on a driver
    adapter, so the same code works whether or not numpy is in the picture.
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Open a connection with the pgvector type adapters registered."""
    dsn = dsn or get_settings().database_url
    with psycopg.connect(dsn) as conn:
        _try_register_vector(conn)
        yield conn


def _try_register_vector(conn: psycopg.Connection) -> None:
    """Register pgvector's type adapters, tolerating a database where the
    extension does not exist yet -- ``init_schema`` creates it and re-registers."""
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.rollback()


def init_schema(conn: psycopg.Connection) -> None:
    """Apply the schema. Idempotent; the compose stack runs it at db init time,
    this exists for running against a Postgres you already had."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    _try_register_vector(conn)  # the type may not have existed at connect time


def upsert_document(conn: psycopg.Connection, source: str, title: str | None) -> int:
    """Insert (or fetch) a document row and return its id."""
    row = conn.execute(
        """
        INSERT INTO documents (source, title) VALUES (%s, %s)
        ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title
        RETURNING id
        """,
        (source, title),
    ).fetchone()
    return int(row[0])


def delete_chunks(conn: psycopg.Connection, document_id: int) -> None:
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    rows: Sequence[tuple[int, str, list[float]]],
) -> int:
    """Bulk-insert ``(chunk_index, content, embedding)`` rows."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (document_id, chunk_index)
            DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
            """,
            [
                (document_id, idx, content, to_pgvector(emb))
                for idx, content, emb in rows
            ],
        )
    return len(rows)


def count_chunks(conn: psycopg.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])


def build_ivfflat_index(conn: psycopg.Connection, lists: int | None = None) -> int:
    """(Re)build the IVFFlat cosine index, sizing ``lists`` from the row count.

    IVFFlat partitions the vectors into ``lists`` k-means cells and, at query
    time, scans only the ``ivfflat.probes`` nearest cells -- approximate nearest
    neighbour instead of a sequential scan over every row. It must be built on
    populated data, which is why it is a post-ingest step rather than part of
    the schema. pgvector's guidance: lists = rows / 1000 up to 1M rows.
    """
    rows = count_chunks(conn)
    if lists is None:
        if rows >= 1000:
            lists = min(1000, rows // 1000)
        else:
            # Below pgvector's rule-of-thumb range; sqrt(rows) keeps the cells
            # from being so few that a probe scans the whole table.
            lists = max(1, int(math.sqrt(max(rows, 1))))
    conn.execute("DROP INDEX IF EXISTS chunks_embedding_ivfflat_idx")
    conn.execute(
        f"CREATE INDEX chunks_embedding_ivfflat_idx ON chunks "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {int(lists)})"
    )
    conn.execute("ANALYZE chunks")
    conn.commit()
    return int(lists)
