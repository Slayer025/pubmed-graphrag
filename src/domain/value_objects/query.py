"""Domain value object representing a user query."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    """A user query string."""

    text: str

    def __str__(self) -> str:
        return self.text
