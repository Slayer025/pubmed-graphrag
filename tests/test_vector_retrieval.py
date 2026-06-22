"""Tests that pre-computed query vectors are not re-embedded during retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from src.application.dto.search_config import SearchConfig
from src.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from src.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from src.infrastructure.storage.chunk_repository import InMemoryChunkRepository
from src.infrastructure.vector_store.numpy_vector_store import NumpyVectorStore
from src.rag_pipeline import RAGPipeline


class _RaisingEmbeddingService:
    """Embedding service that fails if text embedding is attempted."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed() must not be called during retrieve_by_vector")

    def embed_query(self, query: str) -> list[float]:
        raise AssertionError("embed_query() must not be called during retrieve_by_vector")


def _build_use_case() -> RetrieveDocumentsUseCase:
    chunks = [
        {
            "chunk_id": "c0",
            "article_id": "a0",
            "text": "diabetes risk factors",
        },
        {
            "chunk_id": "c1",
            "article_id": "a1",
            "text": "hypertension treatment",
        },
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    graph_repository = InMemoryGraphRepository([], [], chunks)
    chunk_repository = InMemoryChunkRepository(chunks)
    vector_store = NumpyVectorStore(chunks, embeddings)
    return RetrieveDocumentsUseCase(
        embedding_service=_RaisingEmbeddingService(),
        vector_store=vector_store,
        graph_repository=graph_repository,
        chunk_repository=chunk_repository,
    )


def test_retrieve_by_vector_does_not_call_embed_query() -> None:
    use_case = _build_use_case()
    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    results = use_case.retrieve_by_vector(query_vector, config)

    assert len(results) >= 1
    assert results[0].chunk_id == "c0"


def test_rag_pipeline_retrieve_by_vector_does_not_call_embed_query() -> None:
    use_case = _build_use_case()
    pipeline = RAGPipeline(retrieve_documents=use_case)
    query_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    results = pipeline.retrieve_by_vector(query_vector, config)

    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
