"""SentenceTransformer-based embedding service adapter."""

from __future__ import annotations

from typing import Any

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
        vectors = embed_texts(texts, self.model, batch_size=self.batch_size)
        if self.normalize:
            vectors = normalize_embeddings(vectors)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        vectors = self.embed([query])
        return vectors[0]
