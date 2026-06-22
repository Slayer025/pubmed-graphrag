#!/usr/bin/env python3
"""End-to-end Phase 4 evaluation runner.

This script:

1. Loads the filtered PubMedQA evaluation dataset.
2. Loads (or pre-computes and caches) query embeddings.
3. Runs vector-only and GraphRAG retrieval.
4. Optionally generates answers with a real or mock LLM.
5. Computes ROUGE-L and BERTScore.
6. Writes CSV/JSON results to ``outputs/``.

Use ``--retrieval-only`` for a fast smoke test that skips generation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running the script directly from the scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.bootstrap import bootstrap_pipeline, bootstrap_retriever
from src.config import AppConfig
from src.evaluation import (
    build_vector_only_config,
    evaluate_retrieval,
    load_precomputed_query_embeddings,
    precompute_query_embeddings,
)
from src.generation_eval import evaluate_generation, write_generation_csv
from src.llm_client import create_llm_client
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PATH = Path("data/evaluation/pubmedqa_filtered.jsonl.gz")
DEFAULT_QUERY_EMBEDDINGS_DIR = Path("data/evaluation")
DEFAULT_OUTPUT_DIR = Path("outputs")


def _configure_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load evaluation questions from gzip JSONL."""
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d evaluation questions from %s", len(records), path)
    return records


def _gold_chunks_for_article(
    article_id: str, chunks: list[dict[str, Any]]
) -> list[str]:
    """Return chunk IDs belonging to the matched article."""
    return [
        str(chunk["chunk_id"])
        for chunk in chunks
        if str(chunk.get("article_id", "")) == article_id
    ]


def _get_or_build_query_vectors(
    questions: list[dict[str, Any]],
    cache_dir: Path,
    base_config: AppConfig,
) -> np.ndarray:
    """Load cached query embeddings or compute and cache them once."""
    embeddings_path = cache_dir / "query_embeddings.npy"
    if embeddings_path.is_file():
        logger.info("Using cached query embeddings from %s", embeddings_path)
        return load_precomputed_query_embeddings(cache_dir)

    logger.info("Query embedding cache not found; computing once.")
    _, _ = precompute_query_embeddings(questions, output_dir=cache_dir)
    return load_precomputed_query_embeddings(cache_dir)


def _write_retrieval_results(
    vector_results: list[Any],
    graph_results: list[Any],
    output_path: Path,
) -> None:
    """Write per-question retrieval results with the requested columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "method",
                "recall@5",
                "recall@10",
                "mrr",
                "retrieved_chunk_ids",
                "gold_chunk_ids",
            ],
        )
        writer.writeheader()
        for row_idx, (v, g) in enumerate(zip(vector_results, graph_results)):
            for result in (v, g):
                writer.writerow(
                    {
                        "question_id": row_idx,
                        "method": result.method,
                        "recall@5": f"{result.recall_at_5:.4f}",
                        "recall@10": f"{result.recall_at_10:.4f}",
                        "mrr": f"{result.reciprocal_rank:.4f}",
                        "retrieved_chunk_ids": " ".join(result.retrieved_chunks),
                        "gold_chunk_ids": " ".join(result.gold_chunks),
                    }
                )
    logger.info("Wrote retrieval results to %s", output_path)


def _write_retrieval_summary(
    vector_metrics: Any,
    graph_metrics: Any,
    output_path: Path,
) -> None:
    """Write aggregated retrieval summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "vector_only": {
            "method": vector_metrics.method,
            "recall@5": vector_metrics.recall_at_5,
            "recall@10": vector_metrics.recall_at_10,
            "mrr": vector_metrics.mrr,
            "num_questions": vector_metrics.num_questions,
        },
        "graph_rag": {
            "method": graph_metrics.method,
            "recall@5": graph_metrics.recall_at_5,
            "recall@10": graph_metrics.recall_at_10,
            "mrr": graph_metrics.mrr,
            "num_questions": graph_metrics.num_questions,
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Wrote retrieval summary to %s", output_path)


def _build_generation_records(
    questions: list[dict[str, Any]],
    pipeline: RAGPipeline,
    graph_results: list[Any],
    chunk_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate answers for each question using the GraphRAG top-10 context."""
    generation_records: list[dict[str, Any]] = []

    for question_record, retrieval_record in zip(questions, graph_results):
        question = question_record["question"]
        context: list[Any] = []
        for chunk_id in retrieval_record.retrieved_chunks[:10]:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            # Build a minimal context object compatible with RAGPipeline._build_prompt.
            context.append(
                type(
                    "_Result",
                    (),
                    {
                        "chunk_id": chunk_id,
                        "article_id": str(chunk.get("article_id", "")),
                        "text": str(chunk.get("text", "")),
                        "combined_score": 1.0,
                    },
                )()
            )

        response = pipeline.generate(question, context=context)
        generation_records.append(
            {
                "question": question,
                "pubmed_id": question_record.get("pubmed_id", ""),
                "generated_answer": response.answer,
                "reference_answer": question_record.get("long_answer", ""),
            }
        )
    return generation_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 retrieval and generation evaluation."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to evaluation questions (gzip JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV/JSON files",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit evaluation to the first N questions",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and generation metrics",
    )
    parser.add_argument(
        "--llm-client",
        choices=["mock", "openai", "ollama"],
        default="mock",
        help="LLM client for generation",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Alias for --llm-client openai (kept for compatibility)",
    )
    parser.add_argument(
        "--query-embeddings-dir",
        type=Path,
        default=DEFAULT_QUERY_EMBEDDINGS_DIR,
        help="Directory containing query_embeddings.npy cache",
    )
    args = parser.parse_args()

    _configure_logging()

    if args.use_llm:
        args.llm_client = "openai"

    # Load questions.
    questions = load_questions(args.questions)
    if args.max_questions:
        questions = questions[: args.max_questions]
        logger.info("Limited evaluation to first %d questions", len(questions))

    base_config = AppConfig.default()

    # Load or build query embeddings once.
    query_vectors = _get_or_build_query_vectors(
        questions, args.query_embeddings_dir, base_config
    )
    if args.max_questions and query_vectors.shape[0] > len(questions):
        query_vectors = query_vectors[: len(questions)]

    graph_retriever = bootstrap_retriever(base_config)
    chunks = graph_retriever.index.chunks
    chunk_by_id = graph_retriever.index.chunk_by_id
    vector_config = build_vector_only_config(base_config)
    vector_retriever = Retriever(graph_retriever.index, vector_config)

    graph_metrics, graph_per_question = evaluate_retrieval(
        questions,
        chunks=chunks,
        retrieve_by_vector=lambda query_vector: graph_retriever.retrieve_by_vector(query_vector),
        method_name="graph_rag",
        query_vectors=query_vectors,
    )
    vector_metrics, vector_per_question = evaluate_retrieval(
        questions,
        chunks=chunks,
        retrieve_by_vector=lambda query_vector: vector_retriever.retrieve_by_vector(query_vector),
        method_name="vector_only",
        query_vectors=query_vectors,
    )

    for per_q in (vector_per_question, graph_per_question):
        for r in per_q:
            r.gold_chunks = _gold_chunks_for_article(r.matched_article_id, chunks)
            r.results = [
                res
                for res in (chunk_by_id.get(cid) for cid in r.retrieved_chunks)
                if res is not None
            ]

    logger.info("Vector-only metrics: %s", vector_metrics)
    logger.info("GraphRAG metrics: %s", graph_metrics)

    # Write retrieval outputs.
    output_dir = args.output_dir
    _write_retrieval_results(
        vector_per_question,
        graph_per_question,
        output_dir / "retrieval_results.csv",
    )
    _write_retrieval_summary(
        vector_metrics,
        graph_metrics,
        output_dir / "retrieval_summary.json",
    )

    if args.retrieval_only:
        logger.info("Retrieval-only run complete; skipping generation.")
        return 0

    # Generation evaluation.
    llm = create_llm_client(args.llm_client)
    pipeline = bootstrap_pipeline(base_config, llm=llm)
    generation_records = _build_generation_records(
        questions,
        pipeline,
        graph_per_question,
        chunk_by_id,
    )

    gen_metrics, gen_per_question = evaluate_generation(
        generation_records,
        model_type="distilbert-base-uncased",
        device="cpu",
    )
    logger.info("Generation metrics: %s", gen_metrics)

    write_generation_csv(gen_per_question, output_dir / "generation_results.csv")

    # Combined summary.
    summary_path = output_dir / "evaluation_summary.json"
    summary = {
        "retrieval": {
            "vector_only": {
                "recall@5": vector_metrics.recall_at_5,
                "recall@10": vector_metrics.recall_at_10,
                "mrr": vector_metrics.mrr,
                "num_questions": vector_metrics.num_questions,
            },
            "graph_rag": {
                "recall@5": graph_metrics.recall_at_5,
                "recall@10": graph_metrics.recall_at_10,
                "mrr": graph_metrics.mrr,
                "num_questions": graph_metrics.num_questions,
            },
        },
        "generation": {
            "avg_rouge_l": gen_metrics.avg_rouge_l,
            "avg_bertscore_f1": gen_metrics.avg_bertscore_f1,
            "num_questions": gen_metrics.num_questions,
        },
        "config": {
            "llm_client": args.llm_client,
            "max_questions": args.max_questions,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Wrote evaluation summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
