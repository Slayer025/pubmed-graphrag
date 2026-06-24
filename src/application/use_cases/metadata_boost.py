"""Application-layer adapter for metadata-aware boosting.

This module bridges the pure-domain ``boost_by_entity_labels`` service with the
graph repository. It lives in the application layer so it can depend on the
``GraphRepository`` port while remaining thin and testable.
"""

from __future__ import annotations

from src.application.ports import GraphRepository
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.metadata_boost_service import boost_by_entity_labels


class MetadataBoostService:
    """Thin adapter that applies entity-label boosting using a graph repository."""

    def __init__(self, graph_repository: GraphRepository) -> None:
        self.graph_repository = graph_repository

    def apply_boost(
        self,
        results: list[RetrievalResult],
        query: str,
        boost_factor: float,
    ) -> list[RetrievalResult]:
        """Boost ``combined_score`` when the query matches chunk entity labels."""
        if not results or boost_factor <= 1.0:
            return results

        entity_ids_by_chunk = {
            result.chunk_id: self.graph_repository.get_chunk_entities(result.chunk_id)
            for result in results
        }
        # The graph stores entity IDs (name::label). Map them back to labels.
        entity_labels_by_chunk: dict[str, list[str]] = {}
        for chunk_id, entity_ids in entity_ids_by_chunk.items():
            labels: list[str] = []
            for entity_id in entity_ids:
                # Entity IDs produced by entity_extraction.py are formatted as
                # "label:name". We need only the label for boosting.
                if ":" in entity_id:
                    labels.append(entity_id.split(":", 1)[0])
                else:
                    labels.append(entity_id)
            entity_labels_by_chunk[chunk_id] = labels

        return boost_by_entity_labels(
            results=results,
            query=query,
            entity_labels_by_chunk=entity_labels_by_chunk,
            boost_factor=boost_factor,
        )
