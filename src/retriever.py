"""Backward-compatible retriever adapter (DEPRECATED).

This module exists ONLY to keep existing scripts and tests working while the
system migrates to Clean Architecture. New code should use
``src.bootstrap.bootstrap_pipeline`` or ``src.bootstrap.bootstrap_retriever``.

This adapter will be removed in a future release. No new functionality should be
added here. It currently delegates to ``src.bootstrap`` and
``RetrieveDocumentsUseCase`` for all retrieval behavior.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from src.application.dto.search_config import SearchConfig
from src.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from src.config import AppConfig, RetrievalConfig
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.value_objects.query import Query
from src.embeddings import create_embedding_model
from src.infrastructure.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from src.infrastructure.storage.artifact_loader import ArtifactLoader
from src.infrastructure.storage.chunk_repository import InMemoryChunkRepository
from src.infrastructure.vector_store.numpy_vector_store import NumpyVectorStore

logger = logging.getLogger(__name__)


_DEPRECATION_MESSAGE = (
    "Retriever is deprecated and will be removed in a future release. "
    "Use src.bootstrap.bootstrap_pipeline() instead."
)


def _warn_deprecation(message: str) -> None:
    warnings.warn(
        message,
        DeprecationWarning,
        stacklevel=3,
    )


class Retriever:
    """Deprecated facade exposing the legacy retrieval API.

    Internally delegates to ``RetrieveDocumentsUseCase``.  Do not use in new code.
    """

    def __init__(self, index: Any, config: AppConfig) -> None:
        """Initialize the deprecated retriever adapter.

        ``index`` is retained for compatibility but is not the source of truth
        for retrieval logic; the application use case performs retrieval.
        """
        _warn_deprecation(_DEPRECATION_MESSAGE)
        self.index = index
        self.config = config
        self.retrieval = config.retrieval
        self._model: Any | None = None
        self._retrieve_documents: RetrieveDocumentsUseCase | None = None

    def _get_model(self) -> Any:
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            self._model = create_embedding_model(self.config.embedding.model_name)
        return self._model

    def _get_retrieve_documents(self) -> RetrieveDocumentsUseCase:
        """Build the application use case on first use."""
        if self._retrieve_documents is None:
            artifacts = ArtifactLoader.load(self.config)
            model = self._get_model()
            embedding_service = SentenceTransformerEmbeddingService(
                model=model,
                batch_size=self.config.embedding.batch_size,
                normalize=self.config.embedding.normalize,
            )
            vector_store = NumpyVectorStore(artifacts.chunks, artifacts.embeddings)
            graph_repository = InMemoryGraphRepository(
                artifacts.mentions,
                artifacts.has_chunk,
                artifacts.chunks,
            )
            chunk_repository = InMemoryChunkRepository(artifacts.chunks)
            self._retrieve_documents = RetrieveDocumentsUseCase(
                embedding_service=embedding_service,
                vector_store=vector_store,
                graph_repository=graph_repository,
                chunk_repository=chunk_repository,
            )
        return self._retrieve_documents

    def embed_query(self, query: str) -> np.ndarray:
        """Embed and normalize a query string (legacy helper)."""
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed and normalize a batch of query strings (legacy helper)."""
        model = self._get_model()
        embedding_service = SentenceTransformerEmbeddingService(
            model=model,
            batch_size=self.config.embedding.batch_size,
            normalize=self.config.embedding.normalize,
        )
        vectors = embedding_service.embed(queries)
        return np.asarray(vectors, dtype=np.float32)

    def retrieve(
        self,
        query: str,
        retrieval_config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Run the full retrieval pipeline for a query string (deprecated)."""
        config = SearchConfig.from_retrieval_config(
            retrieval_config if retrieval_config is not None else self.retrieval
        )
        use_case = self._get_retrieve_documents()
        return use_case.execute(Query(query), config)

    def retrieve_by_vector(
        self,
        query_vector: np.ndarray,
        retrieval_config: RetrievalConfig | None = None,
        *,
        query_text: str = "",
    ) -> list[RetrievalResult]:
        """Legacy entry point: retrieve by pre-computed vector (deprecated)."""
        query = query_text or ""
        return self.retrieve(query, retrieval_config)


def create_retriever(config: AppConfig | None = None) -> Retriever:
    """Deprecated factory: load the artifact index and build a retriever adapter."""
    _warn_deprecation(_DEPRECATION_MESSAGE)
    if config is None:
        config = AppConfig.default()
    artifacts = ArtifactLoader.load(config)
    index = _build_index(artifacts)
    return Retriever(index, config)


def _build_index(artifacts: Any) -> Any:
    """Build a minimal legacy index object for compatibility."""
    graph_repository = InMemoryGraphRepository(
        artifacts.mentions,
        artifacts.has_chunk,
        artifacts.chunks,
    )
    chunk_repository = InMemoryChunkRepository(artifacts.chunks)

    class _Index:
        def __init__(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
            self.chunks = chunks
            self.embeddings = embeddings
            self.chunk_by_id = chunk_repository.get_chunks({str(c["chunk_id"]) for c in chunks})
            self.row_by_chunk_id = {
                str(chunk["chunk_id"]): row for row, chunk in enumerate(chunks)
            }
            self.article_chunks = graph_repository.article_chunks
            self.entity_chunks = graph_repository.entity_chunks
            self.chunk_entities = graph_repository.chunk_entities
            self.entity_degrees = graph_repository.entity_degrees

    return _Index(artifacts.chunks, artifacts.embeddings)
