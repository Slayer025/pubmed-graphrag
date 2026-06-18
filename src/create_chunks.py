"""Phase 1: build and persist chunk datasets from loaded PubMed abstracts."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from src.chunker import (
    STRATEGY_FIXED,
    STRATEGY_SEMANTIC,
    STRATEGY_SENTENCE,
    AbstractRecord,
    ChunkRecord,
    StrategyName,
    chunk_documents,
    load_abstract_records,
)
from src.storage import (
    DEFAULT_DATA_PATH,
    DiskUsageEstimate,
    WARN_THRESHOLD_BYTES,
    format_bytes,
    log_disk_estimate,
    save_jsonl_gz,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = DEFAULT_DATA_PATH
DEFAULT_OUTPUT_DIR = Path("data/chunks")

# Heuristic constants for pre-write size checks (disk02: abort if > 1 GB).
CHARS_PER_100_TOKENS = 450
CHUNK_JSON_OVERHEAD_BYTES = 140
GZIP_COMPRESSION_RATIO = 0.40
CHUNK_COUNT_SAFETY_FACTOR = 1.25

STRATEGY_OUTPUT_FILES: dict[StrategyName, str] = {
    STRATEGY_FIXED: "chunks_fixed.jsonl.gz",
    STRATEGY_SENTENCE: "chunks_sentence.jsonl.gz",
    STRATEGY_SEMANTIC: "chunks_semantic.jsonl.gz",
}


@dataclass(frozen=True)
class StrategyOutputEstimate:
    """Estimated on-disk size for one strategy output file."""

    strategy: StrategyName
    estimated_chunks: int
    estimated_bytes: int
    output_path: Path


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _abstract_char_length(record: AbstractRecord) -> int:
    return len(str(record.get("abstract", "")))


def estimate_chunk_count(record: AbstractRecord, strategy: StrategyName) -> int:
    """Upper-bound chunk count for one abstract without running chunkers."""
    chars = _abstract_char_length(record)
    if chars == 0:
        return 0

    chunks_at_100_tokens = math.ceil(chars / CHARS_PER_100_TOKENS * CHUNK_COUNT_SAFETY_FACTOR)
    if strategy == STRATEGY_FIXED:
        return max(1, chunks_at_100_tokens)

    # Sentence and semantic chunking typically produce fewer chunks than fixed windows.
    return max(1, math.ceil(chunks_at_100_tokens * 0.9))


def estimate_strategy_output_bytes(
    records: list[AbstractRecord],
    strategy: StrategyName,
) -> StrategyOutputEstimate:
    """Estimate gzip output size for a single strategy dataset."""
    total_chars = 0
    estimated_chunks = 0

    for record in records:
        chars = _abstract_char_length(record)
        total_chars += chars
        estimated_chunks += estimate_chunk_count(record, strategy)

    if estimated_chunks == 0:
        estimated_bytes = 0
    else:
        avg_text_per_chunk = total_chars / estimated_chunks
        uncompressed_bytes = estimated_chunks * (avg_text_per_chunk + CHUNK_JSON_OVERHEAD_BYTES)
        estimated_bytes = int(uncompressed_bytes * GZIP_COMPRESSION_RATIO)

    output_path = DEFAULT_OUTPUT_DIR / STRATEGY_OUTPUT_FILES[strategy]
    return StrategyOutputEstimate(
        strategy=strategy,
        estimated_chunks=estimated_chunks,
        estimated_bytes=estimated_bytes,
        output_path=output_path,
    )


def estimate_all_outputs(
    records: list[AbstractRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[int, list[StrategyOutputEstimate]]:
    """Estimate total gzip bytes for all three chunk datasets."""
    estimates = [
        StrategyOutputEstimate(
            strategy=estimate.strategy,
            estimated_chunks=estimate.estimated_chunks,
            estimated_bytes=estimate.estimated_bytes,
            output_path=output_dir / STRATEGY_OUTPUT_FILES[estimate.strategy],
        )
        for estimate in (
            estimate_strategy_output_bytes(records, STRATEGY_FIXED),
            estimate_strategy_output_bytes(records, STRATEGY_SENTENCE),
            estimate_strategy_output_bytes(records, STRATEGY_SEMANTIC),
        )
    ]
    total_bytes = sum(item.estimated_bytes for item in estimates)
    return total_bytes, estimates


def log_output_estimates(
    records: list[AbstractRecord],
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[int, list[StrategyOutputEstimate]]:
    """Log per-strategy and total output size estimates before writing."""
    input_size = input_path.stat().st_size if input_path.is_file() else 0
    total_bytes, estimates = estimate_all_outputs(records, output_dir=output_dir)

    logger.info(
        "Input: %d abstracts from %s (%s compressed)",
        len(records),
        input_path,
        format_bytes(input_size),
    )

    for estimate in estimates:
        logger.info(
            "Estimate [%s]: ~%d chunks, ~%s -> %s",
            estimate.strategy,
            estimate.estimated_chunks,
            format_bytes(estimate.estimated_bytes),
            estimate.output_path,
        )

    log_disk_estimate(
        DiskUsageEstimate(
            step="create_chunks_outputs",
            retained_bytes=total_bytes,
            peak_transient_bytes=total_bytes,
            uses_streaming=False,
            uses_compression=True,
            notes="Three gzip chunk datasets under data/chunks/.",
        )
    )

    return total_bytes, estimates


def ensure_within_size_limit(estimated_bytes: int, limit_bytes: int = WARN_THRESHOLD_BYTES) -> None:
    """Abort before writing when estimated output would exceed the disk budget."""
    if estimated_bytes > limit_bytes:
        raise RuntimeError(
            f"Estimated chunk output {format_bytes(estimated_bytes)} exceeds "
            f"limit of {format_bytes(limit_bytes)}; aborting without writing files."
        )
    logger.info(
        "Estimated total output %s is within limit %s",
        format_bytes(estimated_bytes),
        format_bytes(limit_bytes),
    )


def write_strategy_chunks(
    records: list[AbstractRecord],
    strategy: StrategyName,
    output_path: Path,
) -> Path:
    """Chunk abstracts with one strategy and write gzip JSONL."""
    logger.info("Building %s chunks for %d abstracts", strategy, len(records))
    chunks: list[ChunkRecord] = chunk_documents(records, strategies=(strategy,))
    saved_path = save_jsonl_gz(chunks, output_path)
    logger.info("Wrote %d %s chunks to %s", len(chunks), strategy, saved_path)
    return saved_path


def create_chunks(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[StrategyName, Path]:
    """Load abstracts, estimate output size, chunk, and save three gzip datasets.

    Args:
        input_path: Gzip JSONL produced by ``load_abstracts()``.
        output_dir: Directory for ``chunks_*.jsonl.gz`` files.

    Returns:
        Mapping of strategy name to written file path.

    Raises:
        RuntimeError: If estimated output exceeds 1 GB.
        FileNotFoundError: If ``input_path`` does not exist.
    """
    _configure_logging()

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_abstract_records(input_path)
    total_estimate, _ = log_output_estimates(records, input_path=input_path, output_dir=output_dir)
    ensure_within_size_limit(total_estimate)

    written_paths: dict[StrategyName, Path] = {}
    for strategy, filename in STRATEGY_OUTPUT_FILES.items():
        output_path = output_dir / filename
        written_paths[strategy] = write_strategy_chunks(records, strategy, output_path)

    logger.info("Phase 1 chunk saving complete: %s", written_paths)
    return written_paths


if __name__ == "__main__":
    create_chunks()
