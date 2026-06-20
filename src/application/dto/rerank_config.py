"""Application-layer reranking configuration value object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankConfig:
    """Optional graph re-ranking configuration."""

    enabled: bool = False
    beta: float = 0.7
    use_pagerank: bool = False
