"""Embedding contract: dimension, determinism, normalisation, batching."""

from __future__ import annotations

import math

import pytest

from app.config import get_settings
from app.embeddings import Embedder, HashEmbedder, get_embedder


def test_hash_embedder_matches_the_configured_dimension(embedder):
    vector = embedder.embed_query("supply chain risk")
    assert len(vector) == 384 == get_settings().embedding_dim


def test_embeddings_are_unit_length(embedder):
    vector = embedder.embed_query("credit risk concentration")
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_embedding_is_deterministic(embedder):
    assert embedder.embed_query("same text") == embedder.embed_query("same text")


def test_query_and_document_paths_agree(embedder):
    text = "the same sentence embedded two ways"
    assert embedder.embed_query(text) == embedder.embed_documents([text])[0]


def test_batch_returns_one_vector_per_input(embedder):
    texts = [f"chunk number {i}" for i in range(37)]  # not a multiple of the batch size
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == embedder.dim for v in vectors)


def test_empty_batch_is_empty(embedder):
    assert embedder.embed_documents([]) == []


def test_related_text_scores_higher_than_unrelated(embedder):
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    query = embedder.embed_query("supply chain disruption risk")
    close = embedder.embed_documents(["supply chain disruption risk factors"])[0]
    far = embedder.embed_documents(["dividends declared per common share"])[0]
    assert cosine(query, close) > cosine(query, far)


def test_hash_embedder_satisfies_the_protocol(embedder):
    assert isinstance(embedder, Embedder)


def test_factory_selects_the_test_double():
    assert isinstance(get_embedder("hash"), HashEmbedder)


def test_factory_defaults_to_the_real_model_class():
    from app.embeddings import SentenceTransformerEmbedder

    # Constructed but not loaded -- no torch import, no model download.
    assert isinstance(get_embedder(), SentenceTransformerEmbedder)


@pytest.mark.integration
def test_real_model_returns_384_dimensions():
    """Runs only where sentence-transformers is installed."""
    pytest.importorskip("sentence_transformers")
    model = get_embedder("sentence-transformers")
    vector = model.embed_query("What risk factors relate to supply chain?")
    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-5)
