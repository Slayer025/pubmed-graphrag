"""Domain value object for request-scoped retrieval hyperparameters.

This object is intentionally independent of ``src.config`` and
``src.application`` so that domain services can receive configuration without
violating Clean Architecture dependency rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHyperparameters:
    """Immutable retrieval parameters used by domain services."""

    expand_depth: int
    max_entity_degree: int
    max_expansion_per_entity: int
    max_expanded_nodes: int
    depth_scores: tuple[float, ...]
    alpha: float
    max_results: int
