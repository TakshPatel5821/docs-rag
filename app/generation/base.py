"""The generation interface every provider implements.

One method, one contract: turn a prompt into text. Retrieval, prompt assembly
and the API layer never import a vendor SDK, so switching model providers is a
config change (``LLM_PROVIDER``) rather than a code change. Same
provider-agnostic shape as Auto-Apply-AI, rewritten in Python.
"""

from __future__ import annotations

import abc


class GenerationError(RuntimeError):
    """Raised when a provider is misconfigured or the upstream call fails."""


class LLMProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for ``prompt``."""
