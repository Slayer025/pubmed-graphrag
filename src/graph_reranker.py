"""Graph-signal re-ranking application service.

Operates on a ``list[RetrievalResult]`` produced by retrieval and boosts results
using graph-derived signals.  This module contains no IO and no framework code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.application.dto.rerank_config import RerankConfig
from src.domain.entities.retrieval_result import RetrievalResult

if TYPE_CHECKING:
    from src.application.ports import GraphRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphSignal:
    """Graph-derived signals for a single retrieved chunk."""

    chunk_id: str
    shared_entity_count: float
    connected_chunk_count: float
    inverse_degree_score: float
    query_overlap_score: float

    @property
    def primary_score(self) -> float:
        """Aggregate graph connectivity signal in [0, 1]."""
        return (
            0.40 * self.shared_entity_count
            + 0.35 * self.connected_chunk_count
            + 0.20 * self.inverse_degree_score
            + 0.05 * self.query_overlap_score
        )


class GraphReranker:
    """Re-rank retrieval results using offline graph signals."""

    def __init__(
        self,
        index: GraphRepository,
        config: RerankConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or RerankConfig()

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Return results re-ranked by graph signals combined with the original score."""
        if not self.config.enabled or not results:
            return results

        if len(results) <= 1:
            return results

        logger.info("Graph reranking %d results (beta=%.2f).", len(results), self.config.beta)

        query_entities = self._extract_query_entities(query)
        result_chunk_ids = {r.chunk_id for r in results}

        signals = self._compute_signals(results, result_chunk_ids, query_entities)
        signals = self._normalize_signals(signals)
        return self._combine_scores(results, signals)

    def _compute_signals(
        self,
        results: list[RetrievalResult],
        result_chunk_ids: set[str],
        query_entities: set[str],
    ) -> dict[str, GraphSignal]:
        """Compute raw graph signals for every result chunk.

        Complexity:
            Let n = number of results, E = total entity mentions across results.
            This implementation builds an entity->result adjacency map in O(E)
            and then computes shared-entity counts in O(E) rather than O(n²).
        """
        signals: dict[str, GraphSignal] = {}

        # Build entity -> set of result chunks containing it.
        entity_to_results: dict[str, set[str]] = {}
        for result in results:
            for entity_id in self.index.get_chunk_entities(result.chunk_id):
                entity_to_results.setdefault(entity_id, set()).add(result.chunk_id)

        # Filter to entities that appear in at least two results.
        shared_entities = {
            entity_id: chunks
            for entity_id, chunks in entity_to_results.items()
            if len(chunks) > 1
        }

        for result in results:
            chunk_id = result.chunk_id
            chunk_entities = self.index.get_chunk_entities(chunk_id)

            shared_entity_count = 0
            connected_chunks: set[str] = set()

            # Count shared entities and connected chunks using adjacency map.
            for entity_id in chunk_entities:
                cooccurring = shared_entities.get(entity_id)
                if not cooccurring:
                    continue
                for other_id in cooccurring:
                    if other_id == chunk_id:
                        continue
                    connected_chunks.add(other_id)
                    shared_entity_count += 1

            # Same-article connections.
            article_id = self.index.get_chunk_article(chunk_id)
            if article_id:
                same_article_chunks = self.index.get_article_chunks(article_id)
                for other_id in same_article_chunks & result_chunk_ids:
                    if other_id != chunk_id:
                        connected_chunks.add(other_id)

            inverse_degree_score = 0.0
            if chunk_entities:
                degrees = [
                    self.index.get_entity_degree(entity_id)
                    for entity_id in chunk_entities
                ]
                inverse_degree_score = sum(1.0 / max(degree, 1) for degree in degrees) / len(
                    chunk_entities
                )

            query_overlap_score = 0.0
            if query_entities and chunk_entities:
                chunk_entity_names = {entity_id.lower() for entity_id in chunk_entities}
                overlap = len(query_entities & chunk_entity_names)
                query_overlap_score = overlap / len(query_entities)

            signals[chunk_id] = GraphSignal(
                chunk_id=chunk_id,
                shared_entity_count=float(shared_entity_count),
                connected_chunk_count=float(len(connected_chunks)),
                inverse_degree_score=float(inverse_degree_score),
                query_overlap_score=float(query_overlap_score),
            )

        return signals

    def _normalize_signals(self, signals: dict[str, GraphSignal]) -> dict[str, GraphSignal]:
        """Normalize each signal dimension to [0, 1] across the result set."""
        if not signals:
            return signals

        keys = list(signals.keys())
        fields = [
            "shared_entity_count",
            "connected_chunk_count",
            "inverse_degree_score",
            "query_overlap_score",
        ]
        arrays: dict[str, np.ndarray] = {}

        for field in fields:
            values = np.array([getattr(signals[k], field) for k in keys], dtype=np.float32)
            arrays[field] = self._min_max_normalize(values)

        normalized: dict[str, GraphSignal] = {}
        for idx, chunk_id in enumerate(keys):
            normalized[chunk_id] = GraphSignal(
                chunk_id=chunk_id,
                shared_entity_count=float(arrays["shared_entity_count"][idx]),
                connected_chunk_count=float(arrays["connected_chunk_count"][idx]),
                inverse_degree_score=float(arrays["inverse_degree_score"][idx]),
                query_overlap_score=float(arrays["query_overlap_score"][idx]),
            )

        return normalized

    @staticmethod
    def _min_max_normalize(values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1]; return zeros if range is zero."""
        min_val = float(values.min())
        max_val = float(values.max())
        if max_val <= min_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def _combine_scores(
        self,
        results: list[RetrievalResult],
        signals: dict[str, GraphSignal],
    ) -> list[RetrievalResult]:
        """Blend original combined_score with graph signal and re-rank."""
        beta = self.config.beta
        reranked: list[RetrievalResult] = []

        for result in results:
            signal = signals.get(result.chunk_id)
            graph_score = signal.primary_score if signal else 0.0
            new_combined = beta * result.combined_score + (1 - beta) * graph_score

            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    article_id=result.article_id,
                    text=result.text,
                    vector_score=result.vector_score,
                    graph_score=result.graph_score,
                    combined_score=float(new_combined),
                    depth=result.depth,
                    source=result.source,
                )
            )

        reranked.sort(key=lambda r: r.combined_score, reverse=True)
        return reranked

    def _extract_query_entities(self, query: str) -> set[str]:
        """Extract simple lowercased tokens/terms from the query."""
        if not query:
            return set()
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "of", "in", "on", "at", "to", "for", "with",
            "from", "by", "about", "and", "or", "but", "what", "which", "who",
            "when", "where", "why", "how",
        }
        terms = set()
        for token in re.split(r"[^a-zA-Z0-9]+", query.lower()):
            if len(token) > 2 and token not in stop_words:
                terms.add(token)
        return terms


def create_graph_reranker(
    index: GraphRepository,
    enabled: bool = False,
    beta: float = 0.7,
    use_pagerank: bool = False,
) -> GraphReranker:
    """Factory helper for building a graph reranker."""
    if use_pagerank:
        raise NotImplementedError("PageRank reranking is not supported in the clean architecture refactor.")
    config = RerankConfig(enabled=enabled, beta=beta, use_pagerank=False)
    return GraphReranker(index=index, config=config)
