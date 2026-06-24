"""Tests for the entity-label metadata boosting service."""

from __future__ import annotations

import pytest

from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.metadata_boost_service import boost_by_entity_labels


def _make_result(
    chunk_id: str = "c1",
    combined_score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        article_id="a1",
        text="text",
        vector_score=0.5,
        graph_score=0.0,
        combined_score=combined_score,
        depth=0,
        source="vector",
    )


def test_boost_when_query_matches_entity_labels() -> None:
    results = [_make_result(chunk_id="c1", combined_score=1.0)]
    labels_by_chunk = {"c1": ["GENE"]}

    boosted = boost_by_entity_labels(
        results,
        "Which gene mutations cause breast cancer?",
        labels_by_chunk,
        boost_factor=1.5,
    )

    assert len(boosted) == 1
    assert boosted[0].combined_score == pytest.approx(1.5)


def test_no_boost_when_no_labels_match() -> None:
    results = [_make_result(chunk_id="c1", combined_score=1.0)]
    labels_by_chunk = {"c1": ["DATE", "ORG"]}

    boosted = boost_by_entity_labels(
        results,
        "Which gene mutations cause breast cancer?",
        labels_by_chunk,
        boost_factor=1.5,
    )

    assert len(boosted) == 1
    assert boosted[0].combined_score == pytest.approx(1.0)


def test_empty_results_returns_empty() -> None:
    boosted = boost_by_entity_labels(
        [],
        "Which gene mutations cause breast cancer?",
        {"c1": ["GENE"]},
        boost_factor=1.5,
    )

    assert boosted == []


def test_multiple_labels_per_chunk() -> None:
    results = [
        _make_result(chunk_id="c1", combined_score=1.0),
        _make_result(chunk_id="c2", combined_score=2.0),
    ]
    labels_by_chunk = {
        "c1": ["DATE", "ORG", "GENE"],
        "c2": ["DATE", "ORG"],
    }

    boosted = boost_by_entity_labels(
        results,
        "Find gene related studies",
        labels_by_chunk,
        boost_factor=1.5,
    )

    assert boosted[0].combined_score == pytest.approx(1.5)
    assert boosted[1].combined_score == pytest.approx(2.0)


def test_boost_factor_application() -> None:
    results = [_make_result(chunk_id="c1", combined_score=0.8)]
    labels_by_chunk = {"c1": ["DISEASE"]}

    boosted = boost_by_entity_labels(
        results,
        "What disease is associated with this gene?",
        labels_by_chunk,
        boost_factor=2.0,
    )

    assert boosted[0].combined_score == pytest.approx(1.6)


def test_no_boost_when_query_has_no_label_keywords() -> None:
    results = [_make_result(chunk_id="c1", combined_score=1.0)]
    labels_by_chunk = {"c1": ["GENE"]}

    boosted = boost_by_entity_labels(
        results,
        "Recent advances in cancer treatment",
        labels_by_chunk,
        boost_factor=1.5,
    )

    assert boosted[0].combined_score == pytest.approx(1.0)


def test_default_boost_factor_is_1_5() -> None:
    results = [_make_result(chunk_id="c1", combined_score=1.0)]
    labels_by_chunk = {"c1": ["GENE"]}

    boosted = boost_by_entity_labels(
        results,
        "gene",
        labels_by_chunk,
    )

    assert boosted[0].combined_score == pytest.approx(1.5)


def test_no_boost_when_factor_is_one_or_less() -> None:
    results = [_make_result(chunk_id="c1", combined_score=1.0)]
    labels_by_chunk = {"c1": ["GENE"]}

    assert boost_by_entity_labels(results, "gene", labels_by_chunk, boost_factor=1.0) is results
    assert boost_by_entity_labels(results, "gene", labels_by_chunk, boost_factor=0.5) is results


def test_missing_chunk_id_has_no_labels() -> None:
    results = [_make_result(chunk_id="c_missing", combined_score=1.0)]
    labels_by_chunk = {"c1": ["GENE"]}

    boosted = boost_by_entity_labels(
        results,
        "gene",
        labels_by_chunk,
        boost_factor=1.5,
    )

    assert boosted[0].combined_score == pytest.approx(1.0)


def test_case_insensitive_label_matching() -> None:
    results = [_make_result(chunk_id="c1", combined_score=1.0)]
    labels_by_chunk = {"c1": ["gene"]}

    boosted = boost_by_entity_labels(
        results,
        "GENE mutations",
        labels_by_chunk,
        boost_factor=1.5,
    )

    assert boosted[0].combined_score == pytest.approx(1.5)
