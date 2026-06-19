"""Generation evaluation metrics for the PubMed GraphRAG pipeline.

Implements ROUGE-L (via ``rouge_score``) and BERTScore for comparing generated
answers against PubMedQA reference long answers.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BERTSCORE_MODEL = "distilbert-base-uncased"


@dataclass(frozen=True)
class GenerationMetrics:
    """Aggregated generation quality metrics."""

    avg_rouge_l: float
    avg_bertscore_f1: float
    num_questions: int


@dataclass(frozen=True)
class PerGenerationResult:
    """Generation metrics for a single question."""

    question: str
    pubmed_id: str
    generated_answer: str
    reference_answer: str
    rouge_l: float
    bertscore_f1: float


def compute_rouge_l(
    generated: str,
    reference: str,
) -> dict[str, float]:
    """Compute ROUGE-L for one generated/reference pair.

    Returns a dict with keys ``rouge-l`` containing ``precision``, ``recall``,
    and ``fmeasure``.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError(
            "ROUGE-L requested but 'rouge_score' is not installed. "
            "Install it with: pip install rouge-score"
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    rouge_l = scores["rougeL"]
    return {
        "precision": float(rouge_l.precision),
        "recall": float(rouge_l.recall),
        "fmeasure": float(rouge_l.fmeasure),
    }


def compute_bertscore(
    generated: list[str],
    references: list[str],
    model_type: str = DEFAULT_BERTSCORE_MODEL,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Compute BERTScore F1 for a batch of generated/reference pairs.

    Args:
        generated: List of generated answer strings.
        references: List of reference answer strings (same length).
        model_type: BERT model to use for scoring.
        device: ``cpu``/``cuda``/``None`` for auto.

    Returns:
        Dict with ``precision``, ``recall``, and ``f1`` lists.
    """
    if len(generated) != len(references):
        raise ValueError("generated and references must have the same length.")
    if not generated:
        return {"precision": [], "recall": [], "f1": []}

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore requested but 'bert_score' is not installed. "
            "Install it with: pip install bert-score"
        ) from exc

    logger.info(
        "Computing BERTScore for %d pairs (model=%s, device=%s)",
        len(generated),
        model_type,
        device or "auto",
    )
    precision, recall, f1 = bert_score(
        generated,
        references,
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return {
        "precision": [float(p) for p in precision.tolist()],
        "recall": [float(r) for r in recall.tolist()],
        "f1": [float(f) for f in f1.tolist()],
    }


def evaluate_generation(
    records: list[dict[str, Any]],
    model_type: str = DEFAULT_BERTSCORE_MODEL,
    device: str | None = None,
) -> tuple[GenerationMetrics, list[PerGenerationResult]]:
    """Evaluate a list of generated answers.

    Args:
        records: Dicts with ``question``, ``pubmed_id``, ``generated_answer``,
            and ``reference_answer`` keys.
        model_type: BERT model for BERTScore.
        device: Device for BERTScore.

    Returns:
        ``(aggregated_metrics, per_question_results)``.
    """
    if not records:
        raise ValueError("No generation records provided.")

    generated = [str(r["generated_answer"]) for r in records]
    references = [str(r["reference_answer"]) for r in records]

    # ROUGE-L per pair.
    rouge_fmeasures: list[float] = []
    for gen, ref in zip(generated, references):
        rouge = compute_rouge_l(gen, ref)
        rouge_fmeasures.append(rouge["fmeasure"])

    # BERTScore batch.
    bert_scores = compute_bertscore(generated, references, model_type=model_type, device=device)
    bert_f1s = bert_scores["f1"]

    per_question: list[PerGenerationResult] = []
    for record, rouge_f, bert_f1 in zip(records, rouge_fmeasures, bert_f1s):
        per_question.append(
            PerGenerationResult(
                question=str(record["question"]),
                pubmed_id=str(record.get("pubmed_id", "")),
                generated_answer=str(record["generated_answer"]),
                reference_answer=str(record["reference_answer"]),
                rouge_l=rouge_f,
                bertscore_f1=bert_f1,
            )
        )

    metrics = GenerationMetrics(
        avg_rouge_l=float(np.mean(rouge_fmeasures)),
        avg_bertscore_f1=float(np.mean(bert_f1s)),
        num_questions=len(records),
    )
    return metrics, per_question


def write_generation_csv(
    results: list[PerGenerationResult],
    output_path: Path | str,
) -> Path:
    """Write per-question generation results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "pubmed_id",
                "generated_answer",
                "reference_answer",
                "rouge_l",
                "bertscore_f1",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "question": r.question,
                    "pubmed_id": r.pubmed_id,
                    "generated_answer": r.generated_answer,
                    "reference_answer": r.reference_answer,
                    "rouge_l": f"{r.rouge_l:.4f}",
                    "bertscore_f1": f"{r.bertscore_f1:.4f}",
                }
            )
    logger.info("Wrote generation results to %s", path)
    return path


def main() -> int:
    """CLI smoke test using two hard-coded pairs."""
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test generation metrics.")
    parser.add_argument(
        "--bert-model",
        default=DEFAULT_BERTSCORE_MODEL,
        help="BERT model for BERTScore",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for BERTScore (cpu/cuda/auto)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    records = [
        {
            "question": "Q1",
            "pubmed_id": "1",
            "generated_answer": "The patient showed significant improvement after treatment.",
            "reference_answer": "The patient improved significantly following treatment.",
        },
        {
            "question": "Q2",
            "pubmed_id": "2",
            "generated_answer": "No evidence supports this hypothesis.",
            "reference_answer": "There is no supporting evidence for this hypothesis.",
        },
    ]
    metrics, per_question = evaluate_generation(records, model_type=args.bert_model, device=args.device)
    print("Metrics:", metrics)
    for r in per_question:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
