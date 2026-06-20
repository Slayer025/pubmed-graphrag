"""Graph expansion use case."""

from __future__ import annotations

from src.application.dto.search_config import SearchConfig
from src.application.ports import GraphRepository
from src.domain.services.graph_traversal_service import GraphTraversalService


class GraphExpandUseCase:
    """Expand seed chunks via graph traversal."""

    def __init__(self, graph_repository: GraphRepository) -> None:
        self.graph_repository = graph_repository
        self.traversal_service = GraphTraversalService()

    def execute(
        self,
        seed_chunk_ids: set[str],
        config: SearchConfig,
    ) -> dict[str, tuple[int, float, str]]:
        """Return expanded chunk map: chunk_id -> (depth, graph_score, source)."""
        return self.traversal_service.expand(
            seed_chunk_ids=seed_chunk_ids,
            graph=self.graph_repository,
            hyperparameters=config.to_hyperparameters(),
        )
