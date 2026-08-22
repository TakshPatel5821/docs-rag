"""Self-hosted generation through Ollama's HTTP API.

Ollama has no first-party Python SDK we depend on, so this provider -- and only
this one -- speaks raw HTTP, via httpx.
"""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.generation.base import GenerationError, LLMProvider

TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate", json=payload, timeout=TIMEOUT
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        return str(response.json().get("response", "")).strip()
