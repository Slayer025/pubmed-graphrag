"""Unit tests for the graph reranker."""

from __future__ import annotations

from src.domain.entities.retrieval_result import RetrievalResult
from src.graph_reranker import GraphReranker, RerankConfig, create_graph_reranker


class _FakeIndex:
    """Minimal fake graph repository for reranker tests."""

    def __init__(self) -> None:
        self._entity_chunks: dict[str, set[str]] = {
            "e1": {"c1", "c2", "c3"},
            "e2": {"c1", "c3"},
            "e3": {"c2"},
        }
        self._chunk_entities: dict[str, set[str]] = {
            "c1": {"e1", "e2"},
            "c2": {"e1", "e3"},
            "c3": {"e1", "e2"},
        }
        self._entity_degrees: dict[str, int] = {
            "e1": 3,
            "e2": 2,
            "e3": 1,
        }

    def get_chunk_article(self, chunk_id: str) -> str:
        return "a1"

    def get_article_chunks(self, article_id: str) -> set[str]:
        return {"c1", "c2", "c3"}

    def get_chunk_entities(self, chunk_id: str) -> set[str]:
        return self._chunk_entities.get(chunk_id, set())

    def get_entity_chunks(self, entity_id: str) -> set[str]:
        return self._entity_chunks.get(entity_id, set())

    def get_entity_degree(self, entity_id: str) -> int:
        return self._entity_degrees.get(entity_id, 0)


def _make_result(
    chunk_id: str,
    combined_score: float,
    article_id: str = "a1",
    source: str = "vector",
    depth: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        article_id=article_id,
        text=f"text {chunk_id}",
        vector_score=combined_score,
        graph_score=0.5,
        combined_score=combined_score,
        depth=depth,
        source=source,
    )


def test_reranker_disabled_returns_input() -> None:
    index = _FakeIndex()
    reranker = GraphReranker(index=index, config=RerankConfig(enabled=False))
    results = [_make_result("c1", 0.9), _make_result("c2", 0.8)]
    assert reranker.rerank("query", results) is results


def test_reranker_empty_input() -> None:
    index = _FakeIndex()
    reranker = GraphReranker(index=index, config=RerankConfig(enabled=True))
    assert reranker.rerank("query", []) == []


def test_reranker_boosts_connected_chunks() -> None:
    index = _FakeIndex()
    reranker = GraphReranker(index=index, config=RerankConfig(enabled=True, beta=0.5))
    # c1 and c3 share more entities, so they should outrank c2 after reranking.
    results = [
        _make_result("c2", 0.95),
        _make_result("c1", 0.90),
        _make_result("c3", 0.85),
    ]
    reranked = reranker.rerank("query", results)
    chunk_ids = [r.chunk_id for r in reranked]
    # c1 and c3 have higher connectivity than c2.
    assert chunk_ids[0] in {"c1", "c3"}
    assert chunk_ids[-1] == "c2"


def test_reranker_preserves_all_results() -> None:
    index = _FakeIndex()
    reranker = GraphReranker(index=index, config=RerankConfig(enabled=True))
    results = [_make_result("c1", 0.9), _make_result("c2", 0.8), _make_result("c3", 0.7)]
    reranked = reranker.rerank("query", results)
    assert {r.chunk_id for r in reranked} == {"c1", "c2", "c3"}


def test_create_graph_reranker_factory() -> None:
    index = _FakeIndex()
    reranker = create_graph_reranker(index=index, enabled=True, beta=0.6)
    assert reranker.config.enabled is True
    assert reranker.config.beta == 0.6


def test_reranker_normalizes_signals() -> None:
    index = _FakeIndex()
    reranker = GraphReranker(index=index, config=RerankConfig(enabled=True, beta=0.0))
    results = [
        _make_result("c1", 0.9),
        _make_result("c2", 0.9),
        _make_result("c3", 0.9),
    ]
    reranked = reranker.rerank("query", results)
    # All original scores equal; order should reflect graph signal only.
    scores = [r.combined_score for r in reranked]
    assert scores[0] >= scores[-1]
    assert all(0.0 <= s <= 1.0 for s in scores)
