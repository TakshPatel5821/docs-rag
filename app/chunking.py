"""Paragraph-aware text chunking.

Why 800 / 150 (the defaults, see ``Settings``):

* 800 characters is roughly 150-200 tokens -- large enough that a 10-K risk
  factor or accounting-policy paragraph survives intact, small enough that a
  single embedding still points at one topic. Embeddings are a mean over the
  sequence, so a 4,000-character chunk covering five subjects averages into a
  vector that is close to nothing in particular.
* 150 characters of overlap (~19%) exists so a fact that straddles a boundary
  is complete in at least one chunk. Sentences in filings run long; 150 chars
  covers about one of them.
* ``eval/run_eval.py`` measures recall@5 so these numbers can be argued from
  measurement rather than taste. See the README for the numbers this corpus
  produced.

Boundaries are respected in this order: paragraph, sentence, whitespace. A chunk
never exceeds ``chunk_size`` characters -- the overlap prefix is counted inside
the budget, not added on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str


def normalize(text: str) -> str:
    """Collapse runs of spaces and stray newlines while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # A single newline inside a paragraph is a wrap artefact, not a break.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = _WS_RE.sub(" ", text)
    text = _PARAGRAPH_RE.sub("\n\n", text)
    # No stray spaces hugging a paragraph break.
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _hard_split(text: str, limit: int) -> list[str]:
    """Split on whitespace so no piece exceeds ``limit`` characters."""
    pieces: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind(" ")
        if cut <= 0:  # a single unbroken token longer than the limit
            cut = limit
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def _units(text: str, limit: int) -> list[str]:
    """Break text into the largest pieces that still fit in ``limit``."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= limit:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, limit))
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Last ``overlap`` characters of ``chunk``, snapped to a word boundary."""
    if overlap <= 0 or not chunk:
        return ""
    tail = chunk[-overlap:]
    if len(chunk) > overlap:
        # Do not start the next chunk halfway through a word.
        space = tail.find(" ")
        tail = tail[space + 1 :] if space != -1 else tail
    return tail.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` characters.

    Each chunk after the first begins with the tail of its predecessor, so
    ``overlap`` characters of context are shared across the boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = normalize(text)
    if not text:
        return []

    # Units are capped so that a unit always fits alongside an overlap prefix.
    units = _units(text, chunk_size - overlap)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = _overlap_tail(current, overlap)
        current = f"{tail}\n\n{unit}" if tail else unit
    if current:
        chunks.append(current)

    return [Chunk(index=i, content=c) for i, c in enumerate(chunks)]
