"""Domain entity representing one ranked retrieval result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked context chunk returned by the retriever."""

    chunk_id: str
    article_id: str
    text: str
    vector_score: float
    graph_score: float
    combined_score: float
    depth: int
    source: str  # "vector", "same_article", or "shared_entity"
