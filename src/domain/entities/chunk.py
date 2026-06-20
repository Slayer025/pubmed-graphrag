"""Domain entity representing a text chunk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A semantic chunk of a PubMed article."""

    chunk_id: str
    article_id: str
    text: str
