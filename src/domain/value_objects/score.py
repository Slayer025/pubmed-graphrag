"""Domain value object representing a relevance score."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    """A normalized relevance score in [0, 1]."""

    value: float

    def __float__(self) -> float:
        return self.value
