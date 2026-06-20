"""In-memory graph repository adapter."""

from __future__ import annotations

from typing import Any


class InMemoryGraphRepository:
    """Graph repository backed by dictionaries built from CSV relations."""

    def __init__(
        self,
        mentions: list[dict[str, str]],
        has_chunk: list[dict[str, str]],
        chunks: list[dict[str, Any]],
    ) -> None:
        self.chunk_article: dict[str, str] = {}
        self.article_chunks: dict[str, set[str]] = {}
        self.entity_chunks: dict[str, set[str]] = {}
        self.chunk_entities: dict[str, set[str]] = {}

        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            article_id = str(chunk.get("article_id", ""))
            self.chunk_article[chunk_id] = article_id
            self.article_chunks.setdefault(article_id, set()).add(chunk_id)

        for rel in has_chunk:
            article_id = str(rel["article_id"])
            chunk_id = str(rel["chunk_id"])
            self.article_chunks.setdefault(article_id, set()).add(chunk_id)
            self.chunk_article[chunk_id] = article_id

        for rel in mentions:
            entity_id = str(rel["entity_id"])
            chunk_id = str(rel["chunk_id"])
            self.entity_chunks.setdefault(entity_id, set()).add(chunk_id)
            self.chunk_entities.setdefault(chunk_id, set()).add(entity_id)

        self.entity_degrees: dict[str, int] = {
            entity_id: len(chunks) for entity_id, chunks in self.entity_chunks.items()
        }

    def get_chunk_article(self, chunk_id: str) -> str:
        return self.chunk_article.get(chunk_id, "")

    def get_article_chunks(self, article_id: str) -> set[str]:
        return self.article_chunks.get(article_id, set())

    def get_chunk_entities(self, chunk_id: str) -> set[str]:
        return self.chunk_entities.get(chunk_id, set())

    def get_entity_chunks(self, entity_id: str) -> set[str]:
        return self.entity_chunks.get(entity_id, set())

    def get_entity_degree(self, entity_id: str) -> int:
        return self.entity_degrees.get(entity_id, 0)
