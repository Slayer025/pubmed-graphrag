"""Disk-efficient storage helpers for the GraphRAG pipeline.

Storage policy (targets machines with <15 GB free space):
- Stream remote data; never materialize the full PubMed corpus locally.
- Persist only the 5000-record working subset, gzip-compressed.
- Keep HuggingFace caches in a project-local directory (``outputs/.hf_cache``)
  controlled via ``HF_HOME`` so large artifacts can be purged in one step.
- Avoid writing intermediate JSONL/Arrow copies when a compressed file suffices.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Steps that may retain or transiently use >= this many bytes log a warning (disk02).
WARN_THRESHOLD_BYTES = 1 * 1024**3


@dataclass(frozen=True)
class DiskUsageEstimate:
    """Expected disk footprint for a pipeline step that fetches external assets."""

    step: str
    retained_bytes: int
    peak_transient_bytes: int
    uses_streaming: bool
    uses_compression: bool
    notes: str = ""

    @property
    def worst_case_bytes(self) -> int:
        """Upper-bound bytes on disk at any point during the step."""
        return max(self.retained_bytes, self.peak_transient_bytes)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count for human-readable logs."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.2f} GB"


def log_disk_estimate(estimate: DiskUsageEstimate) -> None:
    """Log expected disk use and warn when a step may exceed 1 GB (disk02 policy)."""
    logger.info(
        "Disk estimate [%s]: retained=%s, peak transient=%s, streaming=%s, compressed=%s. %s",
        estimate.step,
        format_bytes(estimate.retained_bytes),
        format_bytes(estimate.peak_transient_bytes),
        estimate.uses_streaming,
        estimate.uses_compression,
        estimate.notes,
    )
    if estimate.worst_case_bytes >= WARN_THRESHOLD_BYTES:
        logger.warning(
            "DISK WARNING [%s]: this step may use up to %s (>= 1 GB). %s",
            estimate.step,
            format_bytes(estimate.worst_case_bytes),
            estimate.notes,
        )


# --- Known estimates for this project (update when adding new download steps) ---

PUBMED_STREAM_SUBSET = DiskUsageEstimate(
    step="pubmed_stream_subset",
    retained_bytes=3 * 1024**2,
    peak_transient_bytes=880 * 1024**2,
    uses_streaming=True,
    uses_compression=True,
    notes="Streams train split; retains only 5000-record gzip subset (~2 MB).",
)

PUBMED_FULL_DOWNLOAD = DiskUsageEstimate(
    step="pubmed_full_download",
    retained_bytes=3_500_000_000,
    peak_transient_bytes=7_000_000_000,
    uses_streaming=False,
    uses_compression=False,
    notes="Do not use — full PubMed corpus (~880 MB zip + ~2.5 GB Arrow).",
)

# Approximate HuggingFace Hub download sizes for common embedding models.
_MODEL_DISK_ESTIMATES: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 90 * 1024**2,
    "sentence-transformers/all-mpnet-base-v2": 420 * 1024**2,
    "sentence-transformers/all-MiniLM-L12-v2": 130 * 1024**2,
}


def estimate_model_download(model_name: str) -> DiskUsageEstimate:
    """Return a disk estimate for a HuggingFace embedding model download."""
    retained = _MODEL_DISK_ESTIMATES.get(model_name, 500 * 1024**2)
    return DiskUsageEstimate(
        step=f"model_download:{model_name}",
        retained_bytes=retained,
        peak_transient_bytes=retained,
        uses_streaming=False,
        uses_compression=False,
        notes=(
            f"Model weights cached under HF_HOME/hub. "
            f"{'Known size.' if model_name in _MODEL_DISK_ESTIMATES else 'Unknown model; assuming 500 MB.'}"
        ),
    )

# Project-local HF cache keeps multi-GB Hub/Datasets artifacts out of ~/.cache
# and makes cleanup a single ``cleanup_hf_download_artifacts()`` call.
DEFAULT_HF_HOME = Path("outputs/.hf_cache")

# Only this gzip subset (~few MB) is retained long-term under data/.
DEFAULT_DATA_PATH = Path("data/pubmed_5000.jsonl.gz")

# Scratch space for future pipeline stages; contents are safe to delete anytime.
TEMP_DIRS: tuple[Path, ...] = (Path("outputs/.tmp"),)

# Legacy uncompressed export from earlier iterations of load_data.py.
LEGACY_DATA_PATH = Path("data/pubmed_5000.jsonl")

# HuggingFace/Datasets artifacts tied to the PubMed loader (safe to delete after
# streaming the subset — we never need the full 880 MB zip or Arrow tables).
_HF_ARTIFACT_GLOBS = (
    "downloads/*.incomplete",
    "downloads/*pubmed*",
    "**/pubmed-dataset",
    "**/pubmed-dataset.zip",
    "**/armanc___scientific_papers",
    "**/scientific_papers",
)


def configure_hf_home(hf_home: Path | str | None = None) -> Path:
    """Pin HuggingFace caches to a configurable, project-local directory.

    Must be called before ``datasets.load_dataset`` so the library honours
    ``HF_HOME`` instead of defaulting to ``~/.cache/huggingface``.

    Args:
        hf_home: Cache root. Falls back to ``HF_HOME`` env var, then
            ``outputs/.hf_cache``.

    Returns:
        Resolved absolute cache root.
    """
    from src.infrastructure.storage.safety_guard import safe_mkdir

    cache_root = Path(hf_home or os.environ.get("HF_HOME", DEFAULT_HF_HOME)).resolve()
    safe_mkdir(cache_root)

    # Standard HuggingFace env vars — set explicitly so downstream libraries
    # agree on one location regardless of shell defaults.
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")

    safe_mkdir(cache_root / "datasets")
    safe_mkdir(cache_root / "hub")

    logger.info("HuggingFace cache directory: %s", cache_root)
    return cache_root


def resolve_hf_home() -> Path:
    """Return the active HuggingFace cache root."""
    return Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).resolve()


def save_jsonl_gz(records: Iterable[dict[str, Any]], output_path: Path | str) -> Path:
    """Write records to gzip-compressed JSON Lines (``.jsonl.gz``).

    Compression keeps the 5000-abstract subset on disk instead of a plain
    JSONL file (~3-5x smaller for repetitive scientific text).
    """
    path = Path(output_path)
    if not str(path).endswith(".jsonl.gz"):
        if path.suffix == ".jsonl":
            path = path.with_name(f"{path.name}.gz")
        else:
            path = path.with_suffix(".jsonl.gz")

    path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            record_count += 1

    logger.info("Saved %d records to compressed JSONL at %s", record_count, path)
    return path


def load_jsonl_gz(input_path: Path | str) -> list[dict[str, Any]]:
    """Load a gzip-compressed JSON Lines file into memory."""
    path = Path(input_path)
    records: list[dict[str, Any]] = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_number} of {path}") from exc

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def iter_jsonl_gz(input_path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records from a gzip-compressed JSON Lines file."""
    path = Path(input_path)

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def cleanup_hf_download_artifacts(
    hf_home: Path | str | None = None,
    *,
    dry_run: bool = False,
) -> int:
    """Delete HuggingFace PubMed download artifacts to reclaim disk space.

    Called after streaming completes so the 880 MB zip / Arrow tables are not
    kept once the 5000-record subset is saved.

    Args:
        hf_home: Cache root to clean. Defaults to active ``HF_HOME``.
        dry_run: Log planned deletions without removing files.

    Returns:
        Number of bytes freed (0 when ``dry_run=True``).
    """
    cache_root = Path(hf_home or resolve_hf_home())
    datasets_cache = cache_root / "datasets"
    if not datasets_cache.exists():
        logger.debug("No datasets cache at %s; nothing to clean", datasets_cache)
        return 0

    targets: list[Path] = []
    for pattern in _HF_ARTIFACT_GLOBS:
        targets.extend(datasets_cache.glob(pattern))

    targets = sorted(set(targets), key=lambda path: len(path.parts))
    unique_targets: list[Path] = []
    for target in targets:
        if any(target.is_relative_to(existing) for existing in unique_targets):
            continue
        unique_targets.append(target)

    bytes_freed = 0
    for target in sorted(unique_targets, key=lambda path: len(path.parts), reverse=True):
        size = _path_size(target)
        if dry_run:
            logger.info("[dry-run] Would remove %s (%d bytes)", target, size)
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink()
            else:
                continue
            bytes_freed += size
            logger.info("Removed HuggingFace artifact %s (%d bytes)", target, size)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", target, exc)

    logger.info("Freed %d bytes from HuggingFace cache", bytes_freed)
    return bytes_freed


def cleanup_temp_files(temp_dirs: Iterable[Path | str] = TEMP_DIRS, *, dry_run: bool = False) -> int:
    """Remove scratch files created during pipeline runs."""
    bytes_freed = 0

    for temp_dir in temp_dirs:
        path = Path(temp_dir)
        if not path.exists():
            continue

        for child in path.iterdir():
            size = _path_size(child)
            if dry_run:
                logger.info("[dry-run] Would remove temp path %s (%d bytes)", child, size)
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                bytes_freed += size
                logger.info("Removed temp path %s (%d bytes)", child, size)
            except OSError as exc:
                logger.warning("Could not remove temp path %s: %s", child, exc)

    return bytes_freed


def cleanup_legacy_uncompressed_data(data_path: Path | str = LEGACY_DATA_PATH, *, dry_run: bool = False) -> int:
    """Remove the legacy uncompressed JSONL export if present."""
    path = Path(data_path)
    if not path.is_file():
        return 0

    size = path.stat().st_size
    if dry_run:
        logger.info("[dry-run] Would remove legacy uncompressed data %s (%d bytes)", path, size)
        return 0

    path.unlink()
    logger.info("Removed legacy uncompressed data %s (%d bytes)", path, size)
    return size


def cleanup_all(
    *,
    hf_home: Path | str | None = None,
    remove_hf_cache: bool = True,
    dry_run: bool = False,
) -> int:
    """Run all storage cleanup utilities."""
    total = cleanup_temp_files(dry_run=dry_run)
    total += cleanup_legacy_uncompressed_data(dry_run=dry_run)
    if remove_hf_cache:
        total += cleanup_hf_download_artifacts(hf_home=hf_home, dry_run=dry_run)
    logger.info("Total bytes freed: %d", total)
    return total
