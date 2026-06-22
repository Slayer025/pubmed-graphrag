"""LLM client implementations for the PubMed GraphRAG pipeline.

This module provides concrete ``LLMClient`` implementations that conform to the
protocol defined in ``src.rag_pipeline`` without modifying it:

* ``OpenAIClient`` — OpenAI-compatible chat completions API.
* ``OllamaClient`` — Local Ollama ``/api/generate`` endpoint.
* ``MockLLMClient`` — kept here as a re-export for convenience.

Configuration is read exclusively from environment variables; no secrets are
hard-coded.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.application.ports import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["LLMClient", "MockLLMClient", "OpenAIClient", "OllamaClient", "create_llm_client"]


class MockLLMClient:
    """Placeholder LLM that echoes the prompt context."""

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return (
            "[MOCK LLM] I would answer based on the retrieved context.\n\n"
            f"Prompt preview:\n{prompt[:self.max_chars]}..."
        )


class OpenAIClient:
    """OpenAI-compatible chat completion client.

    Reads ``OPENAI_API_KEY`` (required) and ``LLM_MODEL`` (optional, defaults to
    ``gpt-3.5-turbo``). ``OPENAI_BASE_URL`` can be set to target proxies or other
    OpenAI-compatible services.
    """

    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.3

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OpenAIClient requires OPENAI_API_KEY environment variable or api_key argument."
            )
        self.api_key = resolved_key
        self.model = model or os.environ.get("LLM_MODEL") or self.DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Optional import — only needed when this client is instantiated.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI client requested but 'openai' package is not installed. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Request a chat completion from the configured endpoint."""
        logger.info("Calling OpenAI-compatible model %s", self.model)
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = response.choices[0].message.content or ""
        logger.info("OpenAI response received (%d chars)", len(content))
        return content.strip()


class OllamaClient:
    """Local Ollama ``/api/generate`` client.

    Reads ``OLLAMA_URL`` (optional, defaults to ``http://localhost:11434``) and
    ``LLM_MODEL`` (required). Uses plain ``requests`` so no extra heavy
    dependencies are required.
    """

    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_OPTIONS: dict[str, Any] = {"temperature": 0.3, "num_predict": 512}

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.url = (url or os.environ.get("OLLAMA_URL") or self.DEFAULT_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL")
        if not self.model:
            raise RuntimeError(
                "OllamaClient requires LLM_MODEL environment variable or model argument."
            )
        self.options = options or self.DEFAULT_OPTIONS
        self._session = self._create_session()

    @staticmethod
    def _create_session():
        import requests

        return requests.Session()

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using the Ollama ``/api/generate`` endpoint."""
        logger.info("Calling Ollama model %s at %s", self.model, self.url)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": kwargs.get("options", self.options),
        }
        response = self._session.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=kwargs.get("timeout", 120),
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")
        logger.info("Ollama response received (%d chars)", len(content))
        return content.strip()


def _resolve_openai_api_key(api_key: str | None = None) -> str | None:
    resolved = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    return resolved or None


def create_llm_client(
    client_type: str = "mock",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    ollama_url: str | None = None,
) -> LLMClient:
    """Factory for selecting an LLM client by name.

    Args:
        client_type: One of ``mock``, ``openai``, or ``ollama``.
        api_key: Optional explicit API key for OpenAI.
        model: Optional explicit model name.
        base_url: Optional OpenAI-compatible base URL.
        ollama_url: Optional Ollama base URL.

    Returns:
        An ``LLMClient``-compatible instance. Never raises; falls back to mock.
    """
    client_type = client_type.lower().strip()
    try:
        if client_type == "openai":
            resolved_key = _resolve_openai_api_key(api_key)
            if not resolved_key:
                logger.warning(
                    "OpenAI selected but API key missing in Streamlit secrets. "
                    "OPENAI_API_KEY missing, falling back to mock"
                )
                return MockLLMClient()
            return OpenAIClient(api_key=resolved_key, model=model, base_url=base_url)
        if client_type == "ollama":
            return OllamaClient(url=ollama_url, model=model)
        if client_type == "mock":
            return MockLLMClient()
        logger.warning("Unknown LLM client type %r, falling back to mock", client_type)
        return MockLLMClient()
    except Exception as exc:
        logger.warning(
            "Failed to create LLM client %r (%s), falling back to mock",
            client_type,
            exc,
        )
        return MockLLMClient()


def main() -> int:
    """Quick smoke test for LLM client selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test an LLM client.")
    parser.add_argument(
        "--client",
        choices=["mock", "openai", "ollama"],
        default="mock",
        help="LLM client type",
    )
    parser.add_argument("--prompt", default="What is PubMedQA?", help="Prompt to send")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = create_llm_client(args.client)
    answer = client.complete(args.prompt)
    print("\nAnswer:\n", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
