"""Retrieval evaluation metrics for the PubMed GraphRAG pipeline.

This module evaluates the Phase 3 retriever in two modes:

* **vector_only** — pure semantic vector search (``expand_depth=0, alpha=1.0``)
* **graph_rag** — graph-enhanced retrieval (default Phase 3 settings)

Gold evidence for each PubMedQA question is the set of semantic chunks belonging
to the matched article.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AppConfig, RetrievalConfig
from src.domain.entities.retrieval_result import RetrievalResult
from src.retriever import Retriever

logger = logging.getLogger(__name__)

DEFAULT_KS = (5, 10)
DEFAULT_EVALUATION_DIR = Path("data/evaluation")


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated retrieval metrics for a single method."""

    method: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    num_questions: int


@dataclass
class PerQuestionResult:
    """Retrieval result for one evaluation question."""

    question: str
    pubmed_id: str
    matched_article_id: str
    method: str
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    retrieved_chunks: list[str]
    gold_chunks: list[str] | None = None
    results: list[dict[str, Any]] | None = None


def recall_at_k(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@k: fraction of gold chunks found in top-k results."""
    if not gold_chunk_ids:
        return 0.0
    retrieved_top_k = set(retrieved_chunk_ids[:k])
    hits = len(retrieved_top_k & gold_chunk_ids)
    return hits / len(gold_chunk_ids)


def mean_reciprocal_rank(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: set[str],
) -> float:
    """Compute MRR: 1 / rank of the first relevant chunk, or 0 if none."""
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold_chunk_ids:
            return 1.0 / rank
    return 0.0


def _chunk_ids_for_article(
    article_id: str,
    chunks: list[dict[str, Any]],
) -> set[str]:
    """Return all semantic chunk IDs belonging to the matched article."""
    return {
        str(chunk["chunk_id"])
        for chunk in chunks
        if str(chunk.get("article_id", "")) == article_id
    }


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    retriever: Retriever,
    method_name: str,
    ks: tuple[int, ...] = DEFAULT_KS,
    query_vectors: np.ndarray | None = None,
) -> tuple[RetrievalMetrics, list[PerQuestionResult]]:
    """Evaluate one retriever configuration over the question set.

    Args:
        questions: PubMedQA evaluation records with ``question`` and
            ``matched_article_id`` keys.
        retriever: Configured ``Retriever`` instance.
        method_name: Label for this run (e.g. ``vector_only`` or ``graph_rag``).
        ks: Cut-off values for recall.
        query_vectors: Optional pre-computed query embedding matrix of shape
            ``(len(questions), embedding_dim)``. If provided, model loading is
            skipped entirely.

    Returns:
        ``(aggregated_metrics, per_question_results)``.
    """
    if not questions:
        raise ValueError("No evaluation questions provided.")

    chunks = retriever.index.chunks
    per_question: list[PerQuestionResult] = []
    recall_values: dict[int, list[float]] = {k: [] for k in ks}
    rr_values: list[float] = []

    if query_vectors is None:
        # Batch-embed all questions once to amortize model loading cost.
        query_texts = [str(q["question"]) for q in questions]
        query_vectors = retriever.embed_queries(query_texts)
    elif query_vectors.shape[0] != len(questions):
        raise ValueError(
            f"query_vectors rows ({query_vectors.shape[0]}) != question count ({len(questions)})"
        )

    for record, query_vector in zip(questions, query_vectors):
        question = str(record["question"])
        pubmed_id = str(record.get("pubmed_id", ""))
        article_id = str(record["matched_article_id"])
        gold_chunks = _chunk_ids_for_article(article_id, chunks)

        if not gold_chunks:
            logger.warning(
                "No gold chunks for pubmed_id=%s article_id=%s; skipping",
                pubmed_id,
                article_id,
            )
            continue

        results: list[RetrievalResult] = retriever.retrieve_by_vector(
            query_vector, query_text=question
        )
        retrieved_ids = [r.chunk_id for r in results]

        recalls = {k: recall_at_k(retrieved_ids, gold_chunks, k) for k in ks}
        rr = mean_reciprocal_rank(retrieved_ids, gold_chunks)

        for k, value in recalls.items():
            recall_values[k].append(value)
        rr_values.append(rr)

        per_question.append(
            PerQuestionResult(
                question=question,
                pubmed_id=pubmed_id,
                matched_article_id=article_id,
                method=method_name,
                recall_at_5=recalls[5],
                recall_at_10=recalls[10],
                reciprocal_rank=rr,
                retrieved_chunks=retrieved_ids,
            )
        )

    metrics = RetrievalMetrics(
        method=method_name,
        recall_at_5=float(np.mean(recall_values[5])),
        recall_at_10=float(np.mean(recall_values[10])),
        mrr=float(np.mean(rr_values)),
        num_questions=len(per_question),
    )
    return metrics, per_question


def build_vector_only_config(base_config: AppConfig | None = None) -> AppConfig:
    """Return a config that disables graph expansion and graph scoring."""
    if base_config is None:
        base_config = AppConfig.default()
    base_retrieval = base_config.retrieval
    vector_only_retrieval = RetrievalConfig(
        top_k=base_retrieval.top_k,
        expand_depth=0,
        max_entity_degree=base_retrieval.max_entity_degree,
        max_expansion_per_entity=base_retrieval.max_expansion_per_entity,
        max_expanded_nodes=base_retrieval.max_expanded_nodes,
        alpha=1.0,
        depth_scores=base_retrieval.depth_scores,
        max_results=base_retrieval.max_results,
    )
    return AppConfig(
        neo4j=base_config.neo4j,
        embedding=base_config.embedding,
        artifact=base_config.artifact,
        retrieval=vector_only_retrieval,
    )


def write_per_question_csv(
    results: list[PerQuestionResult],
    output_path: Path | str,
) -> Path:
    """Write per-question retrieval results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "pubmed_id",
                "matched_article_id",
                "method",
                "recall@5",
                "recall@10",
                "reciprocal_rank",
                "retrieved_chunks",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "question": r.question,
                    "pubmed_id": r.pubmed_id,
                    "matched_article_id": r.matched_article_id,
                    "method": r.method,
                    "recall@5": f"{r.recall_at_5:.4f}",
                    "recall@10": f"{r.recall_at_10:.4f}",
                    "reciprocal_rank": f"{r.reciprocal_rank:.4f}",
                    "retrieved_chunks": " ".join(r.retrieved_chunks),
                }
            )
    logger.info("Wrote per-question results to %s", path)
    return path


def write_comparison_csv(
    metrics: list[RetrievalMetrics],
    output_path: Path | str,
) -> Path:
    """Write aggregated comparison CSV (method vs metrics)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "recall@5", "recall@10", "mrr"],
        )
        writer.writeheader()
        for m in metrics:
            writer.writerow(
                {
                    "method": m.method,
                    "recall@5": f"{m.recall_at_5:.4f}",
                    "recall@10": f"{m.recall_at_10:.4f}",
                    "mrr": f"{m.mrr:.4f}",
                }
            )
    logger.info("Wrote comparison CSV to %s", path)
    return path


def load_questions(path: Path | str) -> list[dict[str, Any]]:
    """Load evaluation questions from a gzip JSONL file."""
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d evaluation questions from %s", len(records), path)
    return records


def precompute_query_embeddings(
    questions: list[dict[str, Any]],
    output_dir: Path | str = DEFAULT_EVALUATION_DIR,
    model_name: str | None = None,
) -> tuple[Path, Path]:
    """Embed all evaluation questions once and save vectors + index mapping.

    This amortizes the one-time cost of loading the sentence-transformers model
    across all Phase 4 evaluation runs. The saved artifacts are reused by
    ``scripts/run_evaluation.py`` if present.

    Args:
        questions: Evaluation records with a ``question`` field.
        output_dir: Directory where ``query_embeddings.npy`` and
            ``query_index.json`` are written.
        model_name: Override embedding model. Defaults to ``AppConfig`` default.

    Returns:
        ``(embeddings_path, index_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.embeddings import create_embedding_model, embed_texts

    if model_name is None:
        model_name = AppConfig.default().embedding.model_name

    query_texts = [str(q["question"]) for q in questions]
    model = create_embedding_model(model_name)
    embeddings = embed_texts(query_texts, model, batch_size=64)

    embeddings_path = output_dir / "query_embeddings.npy"
    np.save(embeddings_path, embeddings)

    index = [
        {
            "question": q["question"],
            "pubmed_id": q.get("pubmed_id", ""),
            "matched_article_id": q.get("matched_article_id", ""),
            "row": row,
        }
        for row, q in enumerate(questions)
    ]
    index_path = output_dir / "query_index.json"
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)

    logger.info(
        "Saved %d query embeddings to %s and index to %s",
        len(questions),
        embeddings_path,
        index_path,
    )
    return embeddings_path, index_path


def load_precomputed_query_embeddings(
    output_dir: Path | str = DEFAULT_EVALUATION_DIR,
) -> np.ndarray:
    """Load pre-computed query embeddings from ``output_dir``.

    Returns:
        Embedding matrix of shape ``(n_questions, embedding_dim)``.
    """
    output_dir = Path(output_dir)
    embeddings_path = output_dir / "query_embeddings.npy"
    embeddings = np.load(embeddings_path)
    logger.info("Loaded pre-computed query embeddings %s", embeddings.shape)
    return embeddings


def main() -> int:
    """CLI entry point for retrieval evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate vector-only vs GraphRAG retrieval."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/pubmedqa_filtered.jsonl.gz"),
        help="Path to evaluation questions (gzip JSONL)",
    )
    parser.add_argument(
        "--output-comparison",
        type=Path,
        default=Path("outputs/retrieval_comparison.csv"),
        help="Aggregated comparison CSV path",
    )
    parser.add_argument(
        "--output-results",
        type=Path,
        default=Path("outputs/retrieval_results.csv"),
        help="Per-question results CSV path",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit evaluation to the first N questions",
    )
    parser.add_argument(
        "--query-vectors",
        type=Path,
        default=DEFAULT_EVALUATION_DIR / "query_embeddings.npy",
        help="Pre-computed query embeddings .npy file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from src.retriever import create_retriever

    questions = load_questions(args.questions)
    if args.max_questions:
        questions = questions[: args.max_questions]

    query_vectors = None
    if args.query_vectors.is_file():
        query_vectors = load_precomputed_query_embeddings(args.query_vectors.parent)
        if args.max_questions and query_vectors.shape[0] > len(questions):
            query_vectors = query_vectors[: len(questions)]

    # GraphRAG run
    graph_config = AppConfig.default()
    graph_retriever = create_retriever(graph_config)
    graph_metrics, graph_results = evaluate_retrieval(
        questions, graph_retriever, method_name="graph_rag", query_vectors=query_vectors
    )
    logger.info("GraphRAG metrics: %s", graph_metrics)

    # Vector-only run
    vector_config = build_vector_only_config(graph_config)
    # Reuse the already-loaded index for speed.
    vector_retriever = Retriever(graph_retriever.index, vector_config)
    vector_metrics, vector_results = evaluate_retrieval(
        questions, vector_retriever, method_name="vector_only", query_vectors=query_vectors
    )
    logger.info("Vector-only metrics: %s", vector_metrics)

    write_comparison_csv([vector_metrics, graph_metrics], args.output_comparison)
    write_per_question_csv(vector_results + graph_results, args.output_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
