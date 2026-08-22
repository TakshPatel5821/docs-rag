"""OpenAI generation, via the official ``openai`` SDK."""

from __future__ import annotations

from app.config import get_settings
from app.generation.base import GenerationError, LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model or settings.openai_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise GenerationError("OPENAI_API_KEY is not set")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise GenerationError(
                    "LLM_PROVIDER=openai requires the `openai` package"
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        return (response.choices[0].message.content or "").strip()
