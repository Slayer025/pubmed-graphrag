"""Unit tests for query decomposition."""

from __future__ import annotations

from src.query_decomposer import DecomposerConfig, QueryDecomposer


class _FakeLLM:
    """Fake LLM client for testing."""

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, prompt: str, **kwargs) -> str:  # noqa: ARG002
        return self.response


class _FailingLLM:
    """LLM client that always raises."""

    def complete(self, prompt: str, **kwargs) -> str:  # noqa: ARG002
        raise RuntimeError("LLM failure")


def test_decompose_disabled_returns_original() -> None:
    llm = _FakeLLM('["sub1", "sub2"]')
    decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=False))
    assert decomposer.decompose("What is diabetes?") == ["What is diabetes?"]


def test_decompose_parses_json_array() -> None:
    llm = _FakeLLM('["What is diabetes?", "What are diabetes treatments?"]')
    decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
    result = decomposer.decompose("Tell me about diabetes and its treatments")
    assert result == ["What is diabetes?", "What are diabetes treatments?"]


def test_decompose_parses_markdown_fenced_json() -> None:
    llm = _FakeLLM('```json\n["a", "b"]\n```')
    decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
    assert decomposer.decompose("q") == ["a", "b"]


def test_decompose_parses_dict_with_known_key() -> None:
    llm = _FakeLLM('{"sub_questions": ["a", "b"]}')
    decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
    assert decomposer.decompose("q") == ["a", "b"]


def test_decompose_falls_back_on_llm_failure() -> None:
    decomposer = QueryDecomposer(llm=_FailingLLM(), config=DecomposerConfig(enabled=True))
    assert decomposer.decompose("What is diabetes?") == ["What is diabetes?"]


def test_decompose_falls_back_on_invalid_json() -> None:
    llm = _FakeLLM("not json")
    decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
    assert decomposer.decompose("What is diabetes?") == ["What is diabetes?"]


def test_decompose_trims_and_deduplicates_empty_items() -> None:
    llm = _FakeLLM('["a", "", "  ", "b"]')
    decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
    assert decomposer.decompose("q") == ["a", "b"]


def test_decompose_respects_max_sub_queries() -> None:
    llm = _FakeLLM('["a", "b", "c", "d", "e"]')
    decomposer = QueryDecomposer(
        llm=llm,
        config=DecomposerConfig(enabled=True, max_sub_queries=3),
    )
    assert decomposer.decompose("q") == ["a", "b", "c"]


def test_decompose_empty_query() -> None:
    decomposer = QueryDecomposer(llm=_FakeLLM('["x"]'), config=DecomposerConfig(enabled=True))
    assert decomposer.decompose("") == [""]
