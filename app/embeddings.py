"""Embedding providers.

The rest of the service talks to the ``Embedder`` protocol, never to
sentence-transformers directly, for two reasons:

1. swapping in OpenAI ``text-embedding-3-small`` (or any other model) is a new
   class and an env var, not a rewrite of ingestion and retrieval; and
2. the test suite and the evaluation harness can run against ``HashEmbedder``
   without pulling ~2 GB of torch into CI.

Vectors are L2-normalised. pgvector's ``<=>`` returns cosine distance, which is
already scale-invariant, but normalising keeps the stored vectors comparable and
makes ``1 - distance`` a clean cosine similarity to report to callers.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

from app.config import get_settings


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class SentenceTransformerEmbedder:
    """Local ``all-MiniLM-L6-v2`` embeddings: 384 dims, free, no API key, CPU-fast."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None  # loaded on first use; importing torch is slow

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        # Batched on purpose: one forward pass per 32 chunks instead of per chunk.
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # Same model as ingestion -- a query embedded by a different model lands
        # in a different space and retrieval degrades to noise.
        return self.embed_documents([text])[0]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Test and CI double.

    Not semantic: it captures lexical overlap only. It exists so the pipeline can
    be exercised end-to-end (dimensions, batching, SQL, ordering) without a
    model download, and never as the production path.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embedder(name: str | None = None) -> Embedder:
    """Factory. ``hash`` selects the test double; anything else is the real model."""
    name = (name or "sentence-transformers").lower()
    if name == "hash":
        return HashEmbedder(get_settings().embedding_dim)
    return SentenceTransformerEmbedder()
