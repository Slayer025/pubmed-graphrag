"""Smoke tests for LLM client selection, mode reporting, and mock output."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.interfaces.streamlit import demo as streamlit_demo
from src.llm_client import (
    LLM_MODE_MOCK,
    LLM_MODE_OPENAI,
    MockLLMClient,
    OpenAIClient,
    _build_extractive_answer,
    create_llm_client_with_mode,
    is_openai_package_installed,
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


def test_mock_answer_has_no_mode_banner() -> None:
    answer = MockLLMClient().complete(_sample_prompt())
    assert "MODE: RETRIEVAL-ONLY (NO LLM REASONING)" not in answer
    assert answer.startswith("Answer:\n\n*")


def test_mock_answer_formatting() -> None:
    answer = _build_extractive_answer(
        "What are risk factors for type 2 diabetes?",
        [
            ("c1", 0.9, "Family history of diabetes is associated with increased risk."),
            ("c2", 0.8, "Excess adiposity contributes to type 2 diabetes risk."),
        ],
    )
    assert "Answer:\n\n" in answer
    assert "\n\nSources:\n\n" in answer
    assert "(c1)" not in answer
    assert "* c1" in answer or "* c2" in answer
    assert "MODE:" not in answer


@pytest.mark.skipif(not is_openai_package_installed(), reason="openai package not installed")
def test_openai_client_imports_when_package_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.OpenAI") as mock_openai:
        client = OpenAIClient(api_key="test-key")
        assert client.api_key == "test-key"
        mock_openai.assert_called_once()
