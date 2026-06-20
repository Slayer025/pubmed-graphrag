"""Retrieve documents use case."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.dto.search_config import SearchConfig
from src.application.ports import ChunkRepository, EmbeddingService, GraphRepository, VectorStore
from src.application.use_cases.graph_expand import GraphExpandUseCase
from src.application.use_cases.rerank import RerankUseCase
from src.application.use_cases.vector_search import VectorSearchUseCase
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.value_objects.query import Query


class RetrieveDocumentsUseCase:
    """End-to-end retrieval: vector search + graph expand + rerank."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        graph_repository: GraphRepository,
        chunk_repository: ChunkRepository,
    ) -> None:
        self.vector_search = VectorSearchUseCase(embedding_service, vector_store)
        self.graph_expand = GraphExpandUseCase(graph_repository)
        self.rerank = RerankUseCase(chunk_repository)

    def execute(self, query: Query, config: SearchConfig) -> list[RetrievalResult]:
        """Retrieve and rank context chunks for a query."""
        vector_results = self.vector_search.execute(query, config)
        seed_ids = {chunk_id for chunk_id, _ in vector_results}
        expanded = self.graph_expand.execute(seed_ids, config)
        return self.rerank.execute(vector_results, expanded, config)

    def retrieve_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Retrieve by a pre-computed query vector."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        vector_results = self.vector_search.search_by_vector(query_vector, config)
        seed_ids = {chunk_id for chunk_id, _ in vector_results}
        expanded = self.graph_expand.execute(seed_ids, config)
        return self.rerank.execute(vector_results, expanded, config)
