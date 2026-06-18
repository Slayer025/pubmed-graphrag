"""Load and parse PubMed abstract data."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DownloadConfig, IterableDataset, load_dataset, load_dataset_builder
from datasets.exceptions import DatasetNotFoundError

from src.storage import (
    DEFAULT_DATA_PATH,
    PUBMED_STREAM_SUBSET,
    cleanup_hf_download_artifacts,
    cleanup_legacy_uncompressed_data,
    configure_hf_home,
    log_disk_estimate,
    save_jsonl_gz,
)

logger = logging.getLogger(__name__)

DATASET_CONFIG = "pubmed"
DATASET_CANDIDATES = ("scientific_papers", "armanc/scientific_papers")
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_PATH
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_RANDOM_SEED = 42
TRAIN_SPLIT = "train"


def parse_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw PubMed record into a standard schema.

    Args:
        raw_record: Unparsed record from the source JSONL format.

    Returns:
        A normalized record with ``article_id`` and ``abstract`` fields.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    try:
        article_id = raw_record["article_id"]
        abstract_text = raw_record["abstract_text"]
    except KeyError as exc:
        raise ValueError(f"Missing required field in raw record: {exc.args[0]}") from exc

    if not isinstance(abstract_text, list):
        raise ValueError("Field 'abstract_text' must be a list of strings.")

    abstract = "\n".join(str(part) for part in abstract_text)
    abstract = abstract.replace("<S>", "").replace("</S>", "").strip()

    return {
        "article_id": str(article_id),
        "abstract": abstract,
    }


def validate_abstract(record: dict[str, Any]) -> bool:
    """Check whether a record contains the minimum fields for processing.

    Args:
        record: A normalized abstract record.

    Returns:
        True if the record is valid for downstream pipeline steps.
    """
    article_id = record.get("article_id")
    abstract = record.get("abstract")

    return (
        isinstance(article_id, str)
        and bool(article_id.strip())
        and isinstance(abstract, str)
        and bool(abstract.strip())
    )


def _configure_logging() -> None:
    """Configure root logging when no handlers are present."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _resolve_dataset_name() -> str:
    """Return the first HuggingFace dataset identifier that can be loaded."""
    errors: list[str] = []

    for dataset_name in DATASET_CANDIDATES:
        try:
            load_dataset_builder(
                dataset_name,
                DATASET_CONFIG,
                trust_remote_code=True,
            )
            logger.info("Using HuggingFace dataset %s/%s", dataset_name, DATASET_CONFIG)
            return dataset_name
        except Exception as exc:  # noqa: BLE001 - collect and report all candidate failures
            errors.append(f"{dataset_name}: {exc}")

    raise DatasetNotFoundError(
        f"Could not load scientific_papers/pubmed from HuggingFace. Attempts: {'; '.join(errors)}"
    )


def _stream_example_to_raw(example: dict[str, Any], article_id: str) -> dict[str, Any]:
    """Adapt a HuggingFace streaming example to the raw JSONL schema.

    The published ``scientific_papers/pubmed`` schema exposes ``abstract`` as a
    string rather than ``abstract_text`` as a list, and does not include
    ``article_id`` in the streamed features.
    """
    abstract = example.get("abstract", "")
    if not isinstance(abstract, str):
        raise ValueError("Field 'abstract' must be a string.")

    return {
        "article_id": article_id,
        "abstract_text": abstract.split("\n") if abstract else [],
    }


def _collect_records_from_stream(
    stream: Iterable[dict[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    """Stream examples and collect the first ``target_count`` valid abstracts.

    Iteration stops immediately after ``target_count`` records so we never
    buffer the full 120k-record train split in memory or on disk.
    """
    records: list[dict[str, Any]] = []

    for stream_index, example in enumerate(stream):
        article_id = str(stream_index)
        try:
            raw_record = _stream_example_to_raw(example, article_id=article_id)
            record = parse_record(raw_record)
        except ValueError as exc:
            logger.warning("Skipping invalid streamed record at index %d: %s", stream_index, exc)
            continue

        if not validate_abstract(record):
            logger.debug("Skipping empty abstract at stream index %d", stream_index)
            continue

        records.append(record)
        if len(records) >= target_count:
            logger.info("Collected %d valid abstracts; stopping stream iteration", target_count)
            break

    return records


def load_abstracts(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    hf_home: Path | str | None = None,
    cleanup_cache: bool = True,
) -> pd.DataFrame:
    """Stream PubMed abstracts from HuggingFace, collect, persist, and return them.

    Storage behaviour:
    - Uses ``streaming=True`` so the full PubMed corpus is never loaded locally.
    - Writes only ``sample_size`` records to ``data/pubmed_5000.jsonl.gz``.
    - Pins HuggingFace caches via ``HF_HOME`` (see ``configure_hf_home``).
    - Optionally purges HuggingFace download artifacts after the subset is saved.

    Args:
        output_path: Destination path for the gzip JSONL file.
        sample_size: Number of valid abstracts to collect from the stream.
        random_seed: Retained for API compatibility; streaming uses dataset order.
        hf_home: HuggingFace cache root (``HF_HOME``). Defaults to ``outputs/.hf_cache``.
        cleanup_cache: When True, delete HuggingFace PubMed artifacts after saving.

    Returns:
        A DataFrame with columns ``article_id`` and ``abstract``.

    Raises:
        DatasetNotFoundError: If the HuggingFace dataset cannot be resolved.
        ValueError: If fewer than ``sample_size`` valid abstracts are available.
        RuntimeError: If the HuggingFace stream cannot be opened.
    """
    _configure_logging()

    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")

    if random_seed != DEFAULT_RANDOM_SEED:
        logger.debug(
            "random_seed=%d is ignored in streaming mode; collecting the first %d valid records",
            random_seed,
            sample_size,
        )

    output_path = Path(output_path)
    cache_root = configure_hf_home(hf_home)

    # disk02: estimate and log before any HuggingFace fetch.
    log_disk_estimate(PUBMED_STREAM_SUBSET)

    dataset_name = _resolve_dataset_name()

    # delete_extracted=True asks datasets to drop unpacked archives once the
    # stream finishes, reducing leftover disk use from the PubMed zip.
    download_config = DownloadConfig(
        resume_download=True,
        max_retries=3,
        delete_extracted=True,
    )

    try:
        logger.info(
            "Streaming %s/%s train split from HuggingFace (no full download)",
            dataset_name,
            DATASET_CONFIG,
        )
        stream = load_dataset(
            dataset_name,
            DATASET_CONFIG,
            split=TRAIN_SPLIT,
            streaming=True,
            trust_remote_code=True,
            download_config=download_config,
        )
        if not isinstance(stream, IterableDataset):
            raise RuntimeError("Expected an IterableDataset when streaming=True.")
    except DatasetNotFoundError:
        raise
    except Exception as exc:
        logger.exception("Failed to open stream for %s/%s", dataset_name, DATASET_CONFIG)
        raise RuntimeError(
            f"Failed to stream dataset {dataset_name}/{DATASET_CONFIG} from HuggingFace."
        ) from exc

    try:
        records = _collect_records_from_stream(stream, target_count=sample_size)
        if len(records) < sample_size:
            raise ValueError(
                f"Requested {sample_size} abstracts, but only {len(records)} valid records were streamed."
            )

        df = pd.DataFrame(records, columns=["article_id", "abstract"])
        logger.info("Collected %d abstracts from HuggingFace stream", len(df))

        # Persist only the gzip subset — no intermediate plain JSONL on disk.
        saved_path = save_jsonl_gz(df.to_dict(orient="records"), output_path)
        cleanup_legacy_uncompressed_data()

        if saved_path != output_path.resolve() and output_path.exists() and output_path.suffix != ".gz":
            output_path.unlink(missing_ok=True)

    finally:
        # Always attempt cache cleanup so failed runs do not leave multi-GB zips behind.
        if cleanup_cache:
            cleanup_hf_download_artifacts(hf_home=cache_root)

    return df
