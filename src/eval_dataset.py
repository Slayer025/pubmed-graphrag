"""Build a filtered PubMedQA evaluation dataset matched to our PubMed subset.

This module loads the PubMedQA ``pqa_labeled`` split, matches each question's
context against the 5,000 abstracts in ``data/pubmed_5000.jsonl.gz`` using
simple token-overlap similarity, and writes a gzip-compressed JSONL file of
evaluation records.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

from src.storage import configure_hf_home, load_jsonl_gz, save_jsonl_gz

logger = logging.getLogger(__name__)

PUBMEDQA_DATASET_NAME = "pubmed_qa"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_RAW_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"
    "ori_pqal.json"
)
PUBMEDQA_HF_FALLBACK_NAME = "qiaojin/PubMedQA"

DEFAULT_EVALUATION_DIR = Path("data/evaluation")
DEFAULT_OUTPUT_PATH = DEFAULT_EVALUATION_DIR / "pubmedqa_filtered.jsonl.gz"
DEFAULT_ABSTRACTS_PATH = Path("data/pubmed_5000.jsonl.gz")
DEFAULT_MIN_JACCARD = 0.35


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, and return unique alphabetic tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return {token for token in text.split() if token.isalpha() and len(token) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _pubmedqa_context_to_text(context: Any) -> str:
    """Normalize PubMedQA context to a single string.

    PubMedQA contexts are either a dict of section -> sentences or already a
    string. This helper joins all sentences regardless of the representation.
    """
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, dict):
        parts: list[str] = []
        for section_text in context.values():
            if isinstance(section_text, list):
                parts.extend(str(sentence) for sentence in section_text)
            else:
                parts.append(str(section_text))
        return " ".join(part.strip() for part in parts if part.strip())
    if isinstance(context, list):
        return " ".join(str(part).strip() for part in context if str(part).strip())
    return str(context).strip()


def _normalize_answer(answer: Any) -> str:
    """Convert PubMedQA answer label to a string."""
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, (list, tuple)) and answer:
        return str(answer[0]).strip()
    return str(answer).strip()


def _extract_long_answer(record: dict[str, Any]) -> str:
    """Return the long answer if present, otherwise fall back to context."""
    long_answer = record.get("LONG_ANSWER") or record.get("final_decision") or record.get("long_answer")
    if isinstance(long_answer, str) and long_answer.strip():
        return long_answer.strip()
    return _pubmedqa_context_to_text(record.get("CONTEXTS", record.get("context", "")))


def load_our_abstracts(path: Path | str = DEFAULT_ABSTRACTS_PATH) -> dict[str, dict[str, Any]]:
    """Load our 5,000 abstract subset indexed by article_id."""
    path = Path(path)
    logger.info("Loading our abstracts from %s", path)
    records = load_jsonl_gz(path)
    indexed = {}
    for record in records:
        article_id = str(record.get("article_id", ""))
        abstract = str(record.get("abstract", ""))
        if article_id and abstract:
            indexed[article_id] = {
                "article_id": article_id,
                "abstract": abstract,
                "tokens": _tokenize(abstract),
            }
    logger.info("Loaded %d valid abstracts", len(indexed))
    return indexed


def _best_article_match(
    question_text: str,
    context_text: str,
    abstracts: dict[str, dict[str, Any]],
) -> tuple[str | None, float]:
    """Find the article in our subset that best matches the PubMedQA context.

    The query combines the question and the context. We use **query-overlap**
    (fraction of query tokens that appear in the abstract) rather than Jaccard
    because PubMedQA contexts are short summaries while our abstracts are full
    text; Jaccard is unfairly dominated by the large abstract denominator.

    Returns ``(article_id, overlap_similarity)``.
    """
    query_text = f"{question_text} {context_text}".strip()
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return None, 0.0

    best_article_id: str | None = None
    best_score = 0.0

    for article_id, record in abstracts.items():
        abstract_tokens = record["tokens"]
        intersection = len(query_tokens & abstract_tokens)
        score = intersection / len(query_tokens)
        if score > best_score:
            best_score = score
            best_article_id = article_id

    return best_article_id, best_score


def _load_pubmedqa_hf(
    dataset_name: str = PUBMEDQA_DATASET_NAME,
    config: str = PUBMEDQA_CONFIG,
) -> list[dict[str, Any]]:
    """Attempt to load PubMedQA from HuggingFace."""
    logger.info("Attempting HuggingFace load: %s/%s", dataset_name, config)
    configure_hf_home()
    try:
        dataset = load_dataset(
            dataset_name,
            config,
            trust_remote_code=True,
            streaming=False,
        )
    except Exception as exc:
        logger.warning("HuggingFace load failed for %s/%s: %s", dataset_name, config, exc)
        raise

    split = "train" if "train" in dataset else list(dataset.keys())[0]
    records = list(dataset[split])
    logger.info("Loaded %d PubMedQA records from %s/%s", len(records), dataset_name, config)
    return records


def _load_pubmedqa_fallback() -> list[dict[str, Any]]:
    """Fallback: try qiaojin/PubMedQA, then raw GitHub JSON."""
    # Try alternate HF name first (may work on some Python/datasets versions).
    try:
        return _load_pubmedqa_hf(PUBMEDQA_HF_FALLBACK_NAME, PUBMEDQA_CONFIG)
    except Exception as exc:
        logger.warning("Fallback HuggingFace load failed: %s", exc)

    # Final fallback: download the small raw JSON file from the PubMedQA repo.
    import urllib.request

    logger.info("Downloading raw PubMedQA JSON from %s", PUBMEDQA_RAW_URL)
    with urllib.request.urlopen(PUBMEDQA_RAW_URL, timeout=60) as response:
        raw_bytes = response.read()

    raw_data = json.loads(raw_bytes.decode("utf-8"))
    records: list[dict[str, Any]] = []
    for pubmed_id, value in raw_data.items():
        if not isinstance(value, dict):
            continue
        record = dict(value)
        record["pubmed_id"] = str(pubmed_id)
        record["question"] = record.get("QUESTION", "")
        record["answer"] = record.get("final_decision", "")
        record["long_answer"] = record.get("LONG_ANSWER", "")
        record["context"] = record.get("CONTEXTS", "")
        records.append(record)
    logger.info("Loaded %d PubMedQA records from raw JSON", len(records))
    return records


def load_pubmedqa_labeled() -> list[dict[str, Any]]:
    """Load PubMedQA pqa_labeled, falling back to raw JSON if HF fails."""
    try:
        return _load_pubmedqa_hf(PUBMEDQA_DATASET_NAME, PUBMEDQA_CONFIG)
    except Exception:
        logger.info("Primary HuggingFace load failed; using fallback loader")
        return _load_pubmedqa_fallback()


def build_evaluation_dataset(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    abstracts_path: Path | str = DEFAULT_ABSTRACTS_PATH,
    min_jaccard: float = DEFAULT_MIN_JACCARD,
) -> Path:
    """Build and save the filtered evaluation dataset.

    Args:
        output_path: Destination gzip JSONL file.
        abstracts_path: Our 5,000 abstract subset.
        min_jaccard: Minimum Jaccard similarity required to accept a match.

    Returns:
        Path to the saved evaluation dataset.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    abstracts = load_our_abstracts(abstracts_path)
    pubmedqa_records = load_pubmedqa_labeled()

    filtered: list[dict[str, Any]] = []
    match_scores: list[float] = []

    for record in pubmedqa_records:
        pubmed_id = str(record.get("pubmed_id", ""))
        question = str(record.get("QUESTION", record.get("question", ""))).strip()
        answer = _normalize_answer(record.get("final_decision", record.get("answer", "")))
        long_answer = _extract_long_answer(record)
        context_text = _pubmedqa_context_to_text(record.get("CONTEXTS", record.get("context", "")))

        if not question or not context_text:
            continue

        matched_article_id, score = _best_article_match(
            question, context_text, abstracts
        )
        if matched_article_id is None or score < min_jaccard:
            continue

        filtered.append(
            {
                "question": question,
                "answer": answer,
                "long_answer": long_answer,
                "pubmed_id": pubmed_id,
                "matched_article_id": matched_article_id,
                "match_score": round(score, 4),
            }
        )
        match_scores.append(score)

    if not filtered:
        logger.warning(
            "No PubMedQA questions matched the abstract subset at min_jaccard=%.2f",
            min_jaccard,
        )
    else:
        logger.info(
            "Matched %d PubMedQA questions (score min=%.3f, max=%.3f, mean=%.3f)",
            len(filtered),
            min(match_scores),
            max(match_scores),
            sum(match_scores) / len(match_scores),
        )

    save_jsonl_gz(filtered, output_path)
    return output_path


def main() -> int:
    """CLI entry point for building the evaluation dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the PubMedQA evaluation dataset matched to our abstracts."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output gzip JSONL path",
    )
    parser.add_argument(
        "--abstracts-path",
        type=Path,
        default=DEFAULT_ABSTRACTS_PATH,
        help="Path to our PubMed abstract subset",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=DEFAULT_MIN_JACCARD,
        help="Minimum Jaccard similarity to accept a match",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    build_evaluation_dataset(
        output_path=args.output_path,
        abstracts_path=args.abstracts_path,
        min_jaccard=args.min_jaccard,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
