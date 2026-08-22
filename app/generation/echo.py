"""Credential-free provider used for demos, CI and the evaluation harness.

It performs no generation. It reports what was retrieved and -- importantly --
honours the same grounding rule as a real model: no context, no answer. That
makes the "what happens when the answer is not in the corpus" path testable
without mocking a vendor SDK.
"""

from __future__ import annotations

from app.generation.base import LLMProvider
from app.prompt import NO_CONTEXT_ANSWER, NO_CONTEXT_MARKER


class EchoProvider(LLMProvider):
    name = "echo"

    def generate(self, prompt: str, system: str | None = None) -> str:
        if NO_CONTEXT_MARKER in prompt:
            return NO_CONTEXT_ANSWER
        _, _, context = prompt.partition("--- CONTEXT ---")
        context, _, _ = context.partition("--- END CONTEXT ---")
        return (
            "[echo provider -- no LLM configured; set LLM_PROVIDER to anthropic, "
            "openai or ollama for a generated answer]\n\n"
            "Retrieved context that would have been used:\n"
            f"{context.strip()}"
        )
