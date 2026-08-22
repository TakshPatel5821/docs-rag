"""Prompt assembly for grounded answering."""

from __future__ import annotations

from typing import Sequence

from app.retrieval import RetrievedChunk

SYSTEM_INSTRUCTION = (
    "You are a financial-filings analyst. Answer the question using ONLY the "
    "context passages provided in the user message. Every claim must be supported "
    "by a passage. Cite the passages you used by their bracketed number, e.g. [2]. "
    "If the context does not contain the answer, reply exactly: "
    '"The provided context does not contain the answer to that question." '
    "Do not use outside knowledge and do not speculate."
)

NO_CONTEXT_ANSWER = "The provided context does not contain the answer to that question."

NO_CONTEXT_MARKER = "(no passages retrieved)"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Number the passages so the model has something concrete to cite."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk.title or chunk.source
        blocks.append(
            f"[{i}] source: {label} (chunk {chunk.chunk_index}, "
            f"cosine similarity {chunk.similarity:.3f})\n{chunk.content}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Retrieved context plus the question, as a single user message."""
    context = format_context(chunks) if chunks else NO_CONTEXT_MARKER
    return (
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Single-string form, for providers with no separate system slot."""
    return f"{SYSTEM_INSTRUCTION}\n\n{build_user_prompt(question, chunks)}"
