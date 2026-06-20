"""Domain entity representing a biomedical entity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """A biomedical entity extracted from the PubMed corpus."""

    entity_id: str
    name: str
    label: str
