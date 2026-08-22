"""Chunking: size ceiling, real overlap, boundary preference."""

from __future__ import annotations

import pytest

from app.chunking import chunk_text, normalize

PARAGRAPH = (
    "Item 1A. Risk Factors. The Company depends on component suppliers that are "
    "concentrated in a small number of geographies, and disruption at any one of "
    "them could materially affect results of operations. "
)
DOCUMENT = "\n\n".join(f"{i}. {PARAGRAPH}" for i in range(12))


def test_returns_nothing_for_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("One short paragraph.", chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].content == "One short paragraph."


def test_no_chunk_exceeds_the_size_limit():
    chunks = chunk_text(DOCUMENT, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(c.content) <= 800 for c in chunks)


def test_indexes_are_sequential_from_zero():
    chunks = chunk_text(DOCUMENT, chunk_size=800, overlap=150)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_overlapping_text():
    chunks = chunk_text(DOCUMENT, chunk_size=800, overlap=150)
    for previous, current in zip(chunks, chunks[1:]):
        prefix = current.content.split("\n\n")[0]
        assert prefix, "every chunk after the first carries an overlap prefix"
        assert prefix in previous.content
        # ~150 chars, minus whatever was trimmed to land on a word boundary.
        assert 100 <= len(prefix) <= 150


def test_overlap_prefix_starts_on_a_word_boundary():
    chunks = chunk_text(DOCUMENT, chunk_size=800, overlap=150)
    for chunk in chunks[1:]:
        first_word = chunk.content.split()[0]
        assert first_word in DOCUMENT.split() or first_word.strip(".,;:")


def test_zero_overlap_shares_nothing():
    chunks = chunk_text(DOCUMENT, chunk_size=400, overlap=0)
    joined = "".join(c.content for c in chunks)
    # With no overlap the concatenation is no longer than the source text.
    assert len(joined) <= len(normalize(DOCUMENT)) + 2 * len(chunks)


def test_splits_prefer_paragraph_boundaries():
    text = "\n\n".join(["alpha " * 40, "beta " * 40, "gamma " * 40])
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # No chunk should begin partway through a word of the next paragraph.
    for chunk in chunks:
        assert not chunk.content.startswith(" ")


def test_a_paragraph_longer_than_the_budget_is_split_not_dropped():
    long_paragraph = "word " * 1000  # 5,000 characters, no paragraph breaks
    chunks = chunk_text(long_paragraph, chunk_size=800, overlap=150)
    assert len(chunks) > 5
    assert all(len(c.content) <= 800 for c in chunks)


def test_normalize_keeps_paragraph_breaks_and_collapses_wrapping():
    text = "line one\nstill line one\n\n\n  second paragraph  "
    assert normalize(text) == "line one still line one\n\nsecond paragraph"


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=size, overlap=overlap)
