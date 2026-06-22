"""Smoke tests for LLM client selection, mode reporting, and mock output."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.interfaces.streamlit import demo as streamlit_demo
from src.llm_client import (
    ANSWER_EVIDENCE_SUBTITLE,
    LLM_MODE_MOCK,
    LLM_MODE_OPENAI,
    UNABLE_TO_GENERATE_ANSWER,
    MockLLMClient,
    OpenAIClient,
    _build_extractive_answer,
    create_llm_client_with_mode,
    is_openai_package_installed,
    safe_llm_complete,
)


def _sample_prompt() -> str:
    return (
        "You are a biomedical research assistant. Answer the question using only the context below.\n"
        "Context:\n"
        "[1] chunk_id=c1 article_id=a1 combined_score=0.9000\n"
        "Family history of diabetes is associated with increased risk.\n"
        "[2] chunk_id=c2 article_id=a2 combined_score=0.8000\n"
        "Excess adiposity contributes to type 2 diabetes risk.\n"
        "\nQuestion: What are risk factors for type 2 diabetes?\n\nAnswer:"
    )


def test_openai_mode_when_key_and_package_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_openai_cls = MagicMock()
    with patch("src.llm_client.is_openai_package_installed", return_value=True), patch(
        "src.llm_client.OpenAIClient", mock_openai_cls
    ):
        result = create_llm_client_with_mode("openai")

    assert result.mode == LLM_MODE_OPENAI
    assert result.selected_mode == LLM_MODE_OPENAI
    assert result.fallback_reason is None
    mock_openai_cls.assert_called_once()


def test_missing_openai_package_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("src.llm_client.is_openai_package_installed", return_value=False):
        result = create_llm_client_with_mode("openai")

    assert result.mode == LLM_MODE_MOCK
    assert isinstance(result.client, MockLLMClient)
    assert result.fallback_reason is not None


def test_openai_init_failure_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("src.llm_client.is_openai_package_installed", return_value=True), patch(
        "src.llm_client.OpenAIClient",
        side_effect=RuntimeError("bad credentials"),
    ):
        result = create_llm_client_with_mode("openai")

    assert result.mode == LLM_MODE_MOCK
    assert result.fallback_reason is not None
    assert "initialization failed" in result.fallback_reason.lower()


def test_missing_openai_key_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = create_llm_client_with_mode("openai")

    assert result.mode == LLM_MODE_MOCK
    assert isinstance(result.client, MockLLMClient)
    assert "OPENAI_API_KEY" in (result.fallback_reason or "")


def test_streamlit_active_mode_matches_effective_mode() -> None:
    selection = create_llm_client_with_mode("mock")
    assert streamlit_demo._active_llm_mode(selection) == selection.mode


def test_mock_answer_has_no_debug_mode_banner() -> None:
    answer = MockLLMClient().complete(_sample_prompt())
    assert "MODE: RETRIEVAL-ONLY (NO LLM REASONING)" not in answer
    assert answer.startswith(ANSWER_EVIDENCE_SUBTITLE)


def test_mock_answer_formatting() -> None:
    answer = _build_extractive_answer(
        "What are risk factors for type 2 diabetes?",
        [
            ("c1", 0.9, "Family history of diabetes is associated with increased risk."),
            ("c2", 0.8, "Excess adiposity contributes to type 2 diabetes risk."),
        ],
    )
    assert ANSWER_EVIDENCE_SUBTITLE in answer
    assert "Answer:" in answer
    assert "Sources:" in answer
    assert "Risk factor:" in answer
    assert "Association:" in answer
    assert "(c1)" not in answer
    assert "- c1" in answer
    assert "MODE:" not in answer


def test_openai_complete_falls_back_to_mock_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.chat.completions.create.side_effect = Exception(
            "429 quota exceeded"
        )
        client = OpenAIClient(api_key="test-key")
        answer = client.complete(_sample_prompt())

    assert "MODE: RETRIEVAL-ONLY" not in answer
    assert ANSWER_EVIDENCE_SUBTITLE in answer
    assert "Risk factor:" in answer


def test_safe_llm_complete_returns_unable_message_when_all_fail() -> None:
    class _FailingClient:
        def complete(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("boom")

    with patch("src.llm_client.MockLLMClient") as mock_cls:
        mock_cls.return_value.complete.side_effect = RuntimeError("mock boom")
        answer = safe_llm_complete(_FailingClient(), _sample_prompt())

    assert answer == UNABLE_TO_GENERATE_ANSWER


@pytest.mark.skipif(not is_openai_package_installed(), reason="openai package not installed")
def test_openai_client_imports_when_package_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.OpenAI") as mock_openai:
        client = OpenAIClient(api_key="test-key")
        assert client.api_key == "test-key"
        mock_openai.assert_called_once()
