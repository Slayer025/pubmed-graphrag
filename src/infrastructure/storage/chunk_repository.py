"""In-memory chunk repository adapter."""

from __future__ import annotations

from typing import Any


class InMemoryChunkRepository:
    """Chunk metadata repository backed by a dictionary."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = {str(chunk["chunk_id"]): chunk for chunk in chunks}

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
        return {chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks}
