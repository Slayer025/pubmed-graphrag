"""Rerank use case."""

from __future__ import annotations

from src.application.dto.search_config import SearchConfig
from src.application.ports import ChunkRepository
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.score_fusion_service import ScoreFusionService


class RerankUseCase:
    """Fuse vector and graph scores into ranked retrieval results."""

    def __init__(self, chunk_repository: ChunkRepository) -> None:
        self.chunk_repository = chunk_repository
        self.fusion_service = ScoreFusionService()

    def execute(
        self,
        vector_results: list[tuple[str, float]],
        expanded: dict[str, tuple[int, float, str]],
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Combine vector and graph scores into ranked ``RetrievalResult`` objects."""
        all_chunk_ids = {chunk_id for chunk_id, _ in vector_results} | set(expanded.keys())
        raw_chunks = self.chunk_repository.get_chunks(all_chunk_ids)
        chunks = {
            chunk_id: Chunk(
                chunk_id=chunk_id,
                article_id=str(data.get("article_id", "")),
                text=str(data.get("text", "")),
            )
            for chunk_id, data in raw_chunks.items()
        }
        return self.fusion_service.fuse(
            vector_results, expanded, chunks, config.to_hyperparameters()
        )
