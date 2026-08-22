"""The RAG pipeline itself: retrieve -> ground -> generate -> attribute."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.embeddings import Embedder
from app.generation.base import LLMProvider
from app.prompt import NO_CONTEXT_ANSWER, SYSTEM_INSTRUCTION, build_user_prompt
from app.retrieval import RetrievedChunk, search


@dataclass
class Source:
    source: str
    title: str | None
    chunk_index: int
    similarity: float
    excerpt: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)


def _excerpt(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            source=c.source,
            title=c.title,
            chunk_index=c.chunk_index,
            similarity=round(c.similarity, 4),
            excerpt=_excerpt(c.content),
        )
        for c in chunks
    ]


def answer_from_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    provider: LLMProvider,
) -> Answer:
    """Ground, generate, attribute -- everything after retrieval.

    If retrieval came back empty the model is never called: there is nothing to
    ground an answer in, and a confident-sounding hallucination is the specific
    failure this whole design exists to prevent.
    """
    if not chunks:
        return Answer(question=question, answer=NO_CONTEXT_ANSWER, sources=[])

    prompt = build_user_prompt(question, chunks)
    answer = provider.generate(prompt, system=SYSTEM_INSTRUCTION)
    return Answer(question=question, answer=answer, sources=to_sources(chunks))


def answer_question(
    conn,
    embedder: Embedder,
    provider: LLMProvider,
    question: str,
    k: int = 5,
) -> Answer:
    """Answer ``question`` from the corpus, returning the passages used."""
    return answer_from_chunks(question, search(conn, embedder, question, k=k), provider)
