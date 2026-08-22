"""Top-k retrieval by cosine similarity over pgvector.

Cosine, not L2, because ``all-MiniLM-L6-v2`` encodes meaning in the *direction*
of the vector; magnitude mostly tracks how much text went in. Under L2 a long
chunk and a short chunk about the same topic look far apart. Cosine ignores
that. (The vectors are L2-normalised at embedding time, which makes the two
metrics rank-equivalent -- keeping the operator honest costs nothing and keeps
the index correct if an un-normalised embedder is ever swapped in.)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.db import to_pgvector
from app.embeddings import Embedder

# `<=>` is pgvector's cosine *distance*: 0.0 identical, 1.0 orthogonal, 2.0 opposite.
SEARCH_SQL = """
SELECT c.id,
       c.chunk_index,
       c.content,
       d.source,
       d.title,
       c.embedding <=> %s::vector AS distance
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    chunk_index: int
    content: str
    source: str
    title: str | None
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [-1, 1]; 1.0 is identical."""
        return 1.0 - self.distance


def search(
    conn,
    embedder: Embedder,
    question: str,
    k: int = 5,
    probes: int = 10,
) -> list[RetrievedChunk]:
    """Embed ``question`` with the ingestion model and return the ``k`` nearest chunks."""
    if not question.strip():
        return []

    query_vector = to_pgvector(embedder.embed_query(question))

    with conn.cursor() as cur:
        # More probes = higher recall, slower query. Only affects IVFFlat scans.
        cur.execute(f"SET LOCAL ivfflat.probes = {int(probes)}")
        cur.execute(SEARCH_SQL, (query_vector, query_vector, k))
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_id=row[0],
            chunk_index=row[1],
            content=row[2],
            source=row[3],
            title=row[4],
            distance=float(row[5]),
        )
        for row in rows
    ]
