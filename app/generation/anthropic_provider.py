"""Anthropic (Claude) generation, via the official ``anthropic`` SDK."""

from __future__ import annotations

from app.config import get_settings
from app.generation.base import GenerationError, LLMProvider

MAX_TOKENS = 16000


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            # Configuration is checked before the import so a missing key gives a
            # useful message whether or not the optional SDK is installed.
            if not self.api_key:
                raise GenerationError("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise GenerationError(
                    "LLM_PROVIDER=anthropic requires the `anthropic` package"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise GenerationError("The model declined to answer this request.")
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
