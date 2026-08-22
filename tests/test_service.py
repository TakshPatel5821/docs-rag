"""End-to-end pipeline behaviour with the database and the LLM mocked out."""

from __future__ import annotations

from app.prompt import NO_CONTEXT_ANSWER, SYSTEM_INSTRUCTION
from app.service import answer_question
from tests.conftest import FakeConnection

ROWS = [
    (
        1,
        4,
        "The Company depends on single-source component suppliers.",
        "data/filings/AAPL_10-K_2025-10-31.txt",
        "AAPL Form 10-K filed 2025-10-31",
        0.18,
    ),
    (
        2,
        11,
        "Concentrations of credit risk are monitored by the firm.",
        "data/filings/JPM_10-K_2026-02-13.txt",
        "JPM Form 10-K filed 2026-02-13",
        0.37,
    ),
]


def test_answer_carries_source_attribution(embedder, provider):
    result = answer_question(
        FakeConnection(rows=ROWS), embedder, provider, "supply chain risk?", k=2
    )
    assert result.answer == "A grounded answer [1]."
    assert len(result.sources) == 2
    assert result.sources[0].source.endswith("AAPL_10-K_2025-10-31.txt")
    assert result.sources[0].title == "AAPL Form 10-K filed 2025-10-31"
    assert result.sources[0].chunk_index == 4


def test_sources_report_similarity_in_retrieval_order(embedder, provider):
    result = answer_question(
        FakeConnection(rows=ROWS), embedder, provider, "question", k=2
    )
    similarities = [s.similarity for s in result.sources]
    assert similarities == sorted(similarities, reverse=True)
    assert similarities[0] == round(1.0 - 0.18, 4)


def test_the_model_receives_the_retrieved_context_and_the_grounding_rule(
    embedder, provider
):
    answer_question(FakeConnection(rows=ROWS), embedder, provider, "question", k=2)
    prompt, system = provider.calls[0]
    assert system == SYSTEM_INSTRUCTION
    assert "single-source component suppliers" in prompt
    assert "[1] source: AAPL Form 10-K filed 2025-10-31" in prompt


def test_empty_retrieval_refuses_without_calling_the_model(embedder, provider):
    result = answer_question(
        FakeConnection(rows=[]), embedder, provider, "Who won the 1998 World Cup?", k=5
    )
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert provider.calls == [], "no context means no generation call at all"


def test_excerpts_are_trimmed_for_the_response(embedder, provider):
    long_row = list(ROWS[0])
    long_row[2] = "word " * 400
    result = answer_question(
        FakeConnection(rows=[tuple(long_row)]), embedder, provider, "q", k=1
    )
    assert len(result.sources[0].excerpt) <= 301
