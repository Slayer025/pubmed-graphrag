"""Domain value object representing graph traversal depth."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Depth:
    """Graph traversal depth starting from 0 for vector seeds."""

    value: int

    def __int__(self) -> int:
        return self.value
