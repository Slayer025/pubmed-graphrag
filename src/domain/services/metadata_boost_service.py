"""Pure-domain metadata-aware result boosting.

Boosts retrieval scores when a query explicitly asks for an entity type that a
chunk's extracted entity labels satisfy. This service intentionally uses simple
keyword/label matching and no heavy NLP or infrastructure imports.
"""

from __future__ import annotations

import re
from dataclasses import replace

from src.domain.entities.retrieval_result import RetrievalResult


# Maps common query keywords to the entity label that should trigger a boost.
# Keeping the mapping small and explicit matches the project's lightweight,
# deterministic style.
_KEYWORD_TO_LABEL: dict[str, str] = {
    "gene": "GENE",
    "genes": "GENE",
    "protein": "PROTEIN",
    "proteins": "PROTEIN",
    "drug": "DRUG",
    "drugs": "DRUG",
    "medication": "DRUG",
    "medications": "DRUG",
    "disease": "DISEASE",
    "diseases": "DISEASE",
    "disorder": "DISEASE",
    "disorders": "DISEASE",
    "symptom": "SYMPTOM",
    "symptoms": "SYMPTOM",
    "organism": "ORGANISM",
    "organisms": "ORGANISM",
    "cell": "CELL",
    "cells": "CELL",
    "chemical": "CHEMICAL",
    "chemicals": "CHEMICAL",
    "compound": "CHEMICAL",
    "compounds": "CHEMICAL",
    "tissue": "TISSUE",
    "tissues": "TISSUE",
    "pathway": "PATHWAY",
    "pathways": "PATHWAY",
    "mutation": "MUTATION",
    "mutations": "MUTATION",
    "variant": "VARIANT",
    "variants": "VARIANT",
}


def _extract_matching_labels(query: str) -> set[str]:
    """Return the set of labels the query explicitly asks for.

    Matches both the small keyword map and direct whole-word label names.
    """
    query_lower = query.lower()
    matched: set[str] = set()
    for keyword, label in _KEYWORD_TO_LABEL.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
            matched.add(label)
    # Also allow a label to match itself when the user names it directly.
    for label in set(_KEYWORD_TO_LABEL.values()):
        if re.search(r"\b" + re.escape(label.lower()) + r"\b", query_lower):
            matched.add(label)
    return matched


def boost_by_entity_labels(
    results: list[RetrievalResult],
    query: str,
    entity_labels_by_chunk: dict[str, list[str]],
    boost_factor: float = 1.5,
) -> list[RetrievalResult]:
    """Boost ``combined_score`` when the query matches a chunk's entity labels.

    Args:
        results: Retrieval results to boost. Each must have a ``chunk_id`` and
            a ``combined_score``.
        query: User query string used to detect desired entity types.
        entity_labels_by_chunk: Mapping from ``chunk_id`` to the list of entity
            labels attached to that chunk.
        boost_factor: Multiplicative boost applied to ``combined_score`` when at
            least one label matches. Must be greater than 1.0 to have an effect.

    Returns:
        A new list of ``RetrievalResult`` objects. Results whose labels match the
        query have their ``combined_score`` multiplied by ``boost_factor``;
        non-matching results are returned unchanged. If ``results`` is empty,
        the query matches no labels, or ``boost_factor`` is <= 1.0, the input is
        returned as-is.
    """
    if not results or boost_factor <= 1.0:
        return results

    desired_labels = _extract_matching_labels(query)
    if not desired_labels:
        return results

    boosted: list[RetrievalResult] = []
    for result in results:
        labels = entity_labels_by_chunk.get(result.chunk_id, [])
        has_match = any(label.upper() in desired_labels for label in labels)
        if has_match:
            boosted.append(replace(result, combined_score=result.combined_score * boost_factor))
        else:
            boosted.append(result)
    return boosted
