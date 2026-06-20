"""Graph-aware re-ranking for retrieved PubMed chunks.

The reranker operates on the ``list[RetrievalResult]`` produced by
``src.retriever`` and boosts results using graph-derived signals.  It never
modifies the retriever itself.

Signals used (offline primary):
    * shared entity count with other retrieved chunks
    * number of retrieved chunks connected via shared entities or same article
    * inverse entity degree (rarer entities are more discriminative)
    * optional query/entity text overlap as a secondary feature

Optional enhancement:
    * If a Neo4j instance with the Graph Data Science (GDS) library is
      available and enabled in ``AppConfig.neo4j``, PageRank or degree
      centrality scores are fetched and blended into the ranking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.retriever import RetrievalResult

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.retriever import ArtifactIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphSignal:
    """Graph-derived signals for a single retrieved chunk."""

    chunk_id: str
    shared_entity_count: float
    connected_chunk_count: float
    inverse_degree_score: float
    query_overlap_score: float
    pagerank_score: float = 0.0

    @property
    def primary_score(self) -> float:
        """Aggregate graph connectivity signal in [0, 1]."""
        # Weighted combination of connectivity heuristics.
        return (
            0.40 * self.shared_entity_count
            + 0.35 * self.connected_chunk_count
            + 0.20 * self.inverse_degree_score
            + 0.05 * self.query_overlap_score
            + 0.00 * self.pagerank_score
        )


@dataclass(frozen=True)
class RerankConfig:
    """Configuration for the graph reranker."""

    enabled: bool = False
    beta: float = 0.7
    use_pagerank: bool = False


class GraphReranker:
    """Re-rank retrieval results using offline graph signals and optional GDS."""

    def __init__(
        self,
        index: ArtifactIndex,
        config: RerankConfig | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or RerankConfig()
        self.app_config = app_config
        self._pagerank: dict[str, float] | None = None

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

        if self.config.use_pagerank:
            signals = self._blend_pagerank(signals)

        return self._combine_scores(results, signals)

    def _compute_signals(
        self,
        results: list[RetrievalResult],
        result_chunk_ids: set[str],
        query_entities: set[str],
    ) -> dict[str, GraphSignal]:
        """Compute raw graph signals for every result chunk."""
        signals: dict[str, GraphSignal] = {}

        for result in results:
            chunk_id = result.chunk_id
            chunk_entities = self.index.chunk_entities.get(chunk_id, set())

            shared_entity_count = 0
            connected_chunks: set[str] = set()

            # Connectivity within the retrieved set.
            for other_id in result_chunk_ids:
                if other_id == chunk_id:
                    continue
                other_entities = self.index.chunk_entities.get(other_id, set())
                overlap = len(chunk_entities & other_entities)
                if overlap:
                    shared_entity_count += overlap
                    connected_chunks.add(other_id)

            # Same-article connections within the retrieved set.
            article_id = result.article_id
            if article_id:
                same_article_chunks = self.index.article_chunks.get(article_id, set())
                for other_id in same_article_chunks & result_chunk_ids:
                    if other_id != chunk_id:
                        connected_chunks.add(other_id)

            # Inverse entity degree: rarer entities contribute more.
            inverse_degree_score = 0.0
            if chunk_entities:
                degrees = [
                    self.index.entity_degrees.get(entity_id, 1)
                    for entity_id in chunk_entities
                ]
                # Sum of 1/degree, normalized by entity count.
                inverse_degree_score = sum(1.0 / max(degree, 1) for degree in degrees) / len(
                    chunk_entities
                )

            # Secondary query overlap using exact entity text match.
            query_overlap_score = 0.0
            if query_entities and chunk_entities:
                # Map entity ids to canonical lowercase names.
                chunk_entity_names = {
                    self._entity_name(entity_id).lower() for entity_id in chunk_entities
                }
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

    def _blend_pagerank(self, signals: dict[str, GraphSignal]) -> dict[str, GraphSignal]:
        """Blend optional Neo4j GDS PageRank scores into the signals."""
        pagerank = self._fetch_pagerank()
        if not pagerank:
            return signals

        max_pr = max(pagerank.values()) or 1.0
        blended: dict[str, GraphSignal] = {}
        for chunk_id, signal in signals.items():
            pr_score = pagerank.get(chunk_id, 0.0) / max_pr
            blended[chunk_id] = GraphSignal(
                chunk_id=chunk_id,
                shared_entity_count=signal.shared_entity_count,
                connected_chunk_count=signal.connected_chunk_count,
                inverse_degree_score=signal.inverse_degree_score,
                query_overlap_score=signal.query_overlap_score,
                pagerank_score=float(pr_score),
            )
        return blended

    def _fetch_pagerank(self) -> dict[str, float]:
        """Fetch PageRank scores from Neo4j GDS if available, else empty dict."""
        if self._pagerank is not None:
            return self._pagerank

        self._pagerank = {}
        if self.app_config is None or not self.app_config.neo4j.enabled:
            return self._pagerank

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            logger.warning("Neo4j driver unavailable; skipping PageRank. %s", exc)
            return self._pagerank

        uri = self.app_config.neo4j.uri
        auth = (self.app_config.neo4j.user, self.app_config.neo4j.password)
        database = self.app_config.neo4j.database

        try:
            driver = GraphDatabase.driver(uri, auth=auth)
            with driver.session(database=database) as session:
                # Fast existence check for GDS.
                gds_check = session.run(
                    "CALL gds.graph.exists('pubmed-graph') YIELD exists RETURN exists"
                ).single()
                if gds_check and gds_check["exists"]:
                    result = session.run(
                        "CALL gds.pageRank.stream('pubmed-graph') "
                        "YIELD nodeId, score RETURN gds.util.asNode(nodeId).chunk_id AS chunk_id, score"
                    )
                    for record in result:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            self._pagerank[str(chunk_id)] = float(record["score"])
                else:
                    logger.warning("Neo4j GDS graph 'pubmed-graph' not found; skipping PageRank.")
            driver.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch PageRank from Neo4j: %s", exc)

        return self._pagerank

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
        """Extract simple lowercased tokens/terms from the query.

        This is intentionally lightweight.  A full NER pass can be added later.
        """
        if not query:
            return set()
        # Drop common stop words and short tokens.
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "from",
            "by",
            "about",
            "and",
            "or",
            "but",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
        }
        terms = set()
        for token in re.split(r"[^a-zA-Z0-9]+", query.lower()):
            if len(token) > 2 and token not in stop_words:
                terms.add(token)
        return terms

    def _entity_name(self, entity_id: str) -> str:
        """Best-effort lookup of entity name; falls back to entity_id."""
        # The ArtifactIndex does not store entity names, so we use the id text.
        return entity_id


def create_graph_reranker(
    index: ArtifactIndex,
    enabled: bool = False,
    beta: float = 0.7,
    use_pagerank: bool = False,
    app_config: AppConfig | None = None,
) -> GraphReranker:
    """Factory helper for building a graph reranker."""
    config = RerankConfig(enabled=enabled, beta=beta, use_pagerank=use_pagerank)
    return GraphReranker(index=index, config=config, app_config=app_config)
