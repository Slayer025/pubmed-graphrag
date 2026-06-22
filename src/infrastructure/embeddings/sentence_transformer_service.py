"""SentenceTransformer-based embedding service adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embeddings import embed_texts, normalize_embeddings


class SentenceTransformerEmbeddingService:
    """Infrastructure adapter wrapping a sentence-transformers model."""

    def __init__(self, model: Any, batch_size: int = 64, normalize: bool = True) -> None:
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings as plain Python lists."""
        if not texts:
            return []
        assert np is not None
        vectors = embed_texts(texts, self.model, batch_size=self.batch_size)
        if self.normalize:
            vectors = normalize_embeddings(vectors)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        vectors = self.embed([query])
        return vectors[0]


class LazySentenceTransformerEmbeddingService:
    """Embedding service that loads the HF model on first query (not at pipeline build)."""

    def __init__(
        self,
        model_name: str,
        hf_home: str,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._hf_home = hf_home
        self._batch_size = batch_size
        self._normalize = normalize
        self._delegate: SentenceTransformerEmbeddingService | None = None

    def _ensure_delegate(self) -> SentenceTransformerEmbeddingService:
        if self._delegate is None:
            from src.embeddings import create_embedding_model

            model = create_embedding_model(self._model_name, cache_folder=self._hf_home)
            self._delegate = SentenceTransformerEmbeddingService(
                model=model,
                batch_size=self._batch_size,
                normalize=self._normalize,
            )
        return self._delegate

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._ensure_delegate().embed(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._ensure_delegate().embed_query(query)
