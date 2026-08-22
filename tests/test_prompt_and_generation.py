"""Prompt grounding and the provider-agnostic generation layer."""

from __future__ import annotations

import pytest

from app.generation import PROVIDERS, get_provider
from app.generation.base import GenerationError, LLMProvider
from app.generation.echo import EchoProvider
from app.prompt import (
    NO_CONTEXT_ANSWER,
    NO_CONTEXT_MARKER,
    SYSTEM_INSTRUCTION,
    build_prompt,
    build_user_prompt,
    format_context,
)


def test_system_instruction_forbids_ungrounded_answers():
    assert "ONLY the context" in SYSTEM_INSTRUCTION
    assert NO_CONTEXT_ANSWER in SYSTEM_INSTRUCTION
    assert "do not speculate" in SYSTEM_INSTRUCTION.lower()


def test_context_passages_are_numbered_for_citation(chunks):
    context = format_context(chunks)
    assert context.startswith("[1] source: AAPL Form 10-K")
    assert "[2] source: JPM Form 10-K" in context


def test_context_includes_every_retrieved_chunk(chunks):
    context = format_context(chunks)
    for chunk in chunks:
        assert chunk.content in context


def test_user_prompt_contains_context_and_question(chunks):
    prompt = build_user_prompt("What is the supply chain risk?", chunks)
    assert "--- CONTEXT ---" in prompt
    assert "Question: What is the supply chain risk?" in prompt


def test_empty_context_is_marked_explicitly():
    prompt = build_user_prompt("Anything?", [])
    assert NO_CONTEXT_MARKER in prompt


def test_single_string_prompt_carries_the_system_instruction(chunks):
    assert build_prompt("q", chunks).startswith(SYSTEM_INSTRUCTION)


def test_grounding_is_respected_when_the_context_is_empty():
    """The stated failure mode: no context must produce a refusal, not a guess."""
    prompt = build_prompt("Who won the 1998 World Cup?", [])
    assert EchoProvider().generate(prompt) == NO_CONTEXT_ANSWER


def test_provider_answers_from_context_when_context_exists(chunks):
    answer = EchoProvider().generate(build_prompt("supply chain?", chunks))
    assert NO_CONTEXT_ANSWER not in answer
    assert "single-source suppliers" in answer


def test_every_registered_provider_implements_the_interface():
    assert set(PROVIDERS) == {"anthropic", "openai", "ollama", "echo"}
    for provider_cls in PROVIDERS.values():
        assert issubclass(provider_cls, LLMProvider)


def test_provider_is_selected_by_name():
    assert isinstance(get_provider("echo"), EchoProvider)
    assert get_provider("ECHO ").name == "echo"


def test_unknown_provider_is_rejected():
    with pytest.raises(GenerationError, match="Unknown LLM_PROVIDER"):
        get_provider("gpt-from-a-dream")


def test_anthropic_provider_reports_a_missing_key_instead_of_calling_out():
    from app.generation.anthropic_provider import AnthropicProvider

    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(api_key="")._get_client()


def test_openai_provider_reports_a_missing_key_instead_of_calling_out():
    from app.generation.openai_provider import OpenAIProvider

    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key="")._get_client()


def test_anthropic_provider_sends_system_and_prompt_without_network(monkeypatch):
    """The vendor call is mocked -- no test in this suite hits a real API."""
    from app.generation.anthropic_provider import AnthropicProvider

    class FakeBlock:
        type = "text"
        text = "Grounded answer [1]."

    class FakeResponse:
        stop_reason = "end_turn"
        content = [FakeBlock()]

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    provider = AnthropicProvider(api_key="test-key", model="claude-opus-5")
    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    assert provider.generate("the prompt", system="the system") == "Grounded answer [1]."
    assert captured["model"] == "claude-opus-5"
    assert captured["system"] == "the system"
    assert captured["messages"] == [{"role": "user", "content": "the prompt"}]


def test_anthropic_refusal_becomes_a_generation_error(monkeypatch):
    from app.generation.anthropic_provider import AnthropicProvider

    class FakeResponse:
        stop_reason = "refusal"
        content = []

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return FakeResponse()

    provider = AnthropicProvider(api_key="test-key")
    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())
    with pytest.raises(GenerationError):
        provider.generate("prompt")
