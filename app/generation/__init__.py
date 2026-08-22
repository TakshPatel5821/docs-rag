"""Provider-agnostic generation layer, selected at runtime by ``LLM_PROVIDER``."""

from __future__ import annotations

from app.config import get_settings
from app.generation.anthropic_provider import AnthropicProvider
from app.generation.base import GenerationError, LLMProvider
from app.generation.echo import EchoProvider
from app.generation.ollama_provider import OllamaProvider
from app.generation.openai_provider import OpenAIProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "echo": EchoProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """Instantiate the configured provider. Vendor SDKs load lazily, inside it."""
    name = (name or get_settings().llm_provider).strip().lower()
    try:
        provider_cls = PROVIDERS[name]
    except KeyError as exc:
        raise GenerationError(
            f"Unknown LLM_PROVIDER {name!r}; expected one of {sorted(PROVIDERS)}"
        ) from exc
    return provider_cls()


__all__ = [
    "AnthropicProvider",
    "EchoProvider",
    "GenerationError",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "PROVIDERS",
    "get_provider",
]
