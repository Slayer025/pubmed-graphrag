"""Artifact loader for Phase 1/2 data files."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.config import AppConfig
from src.embeddings import normalize_embeddings
from src.infrastructure.storage.csv_loader import load_csv
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

MIN_VALID_SIZE_DEFAULT = 1024  # 1KB fallback
LOCK_RETRY_MS = 300
LOCK_RETRY_ATTEMPTS = 10

# Base URL for Streamlit Cloud / fresh-container bootstrap.
# Override via ARTIFACT_BASE_URL env var (no trailing slash required).
ARTIFACT_BASE_URL = os.environ.get("ARTIFACT_BASE_URL", "TODO_SET_THIS")

_ARTIFACT_REMOTE_NAMES: dict[str, str] = {
    "data/chunks/chunks_semantic.jsonl.gz": "chunks_semantic.jsonl.gz",
    "data/embeddings/semantic_embeddings.npy": "semantic_embeddings.npy",
    "data/graph/mentions.csv": "mentions.csv",
    "data/graph/entities.csv": "entities.csv",
    "data/graph/has_chunk.csv": "has_chunk.csv",
}

_MIN_VALID_SIZES: dict[str, int] = {
    "data/chunks/chunks_semantic.jsonl.gz": MIN_VALID_SIZE_DEFAULT,
    "data/embeddings/semantic_embeddings.npy": 1024 * 1024,
    "data/graph/mentions.csv": MIN_VALID_SIZE_DEFAULT,
    "data/graph/entities.csv": MIN_VALID_SIZE_DEFAULT,
    "data/graph/has_chunk.csv": MIN_VALID_SIZE_DEFAULT,
}


def _repo_root() -> Path:
    """Return repository root without relying on process cwd."""
    return Path(__file__).resolve().parents[3]


def _resolve_artifact_path(path: Path | str) -> Path:
    """Resolve artifact paths to a stable absolute path (no os.chdir)."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _repo_root() / candidate
    return candidate.resolve()


def _ready_path(path: Path) -> Path:
    return Path(f"{path}.ready")


def _process_lock_path(path: Path) -> Path:
    return Path(f"{path}.lock")


def _part_path(path: Path) -> Path:
    return Path(f"{path}.part")


def _effective_min_size(resolved: Path, min_size: int) -> int:
    rel = resolved.relative_to(_repo_root()).as_posix()
    return _MIN_VALID_SIZES.get(rel, min_size)


def _read_ready_content(ready_lock: Path) -> str | None:
    if not ready_lock.exists():
        return None
    try:
        return ready_lock.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _artifact_is_ready(path: Path | str, min_size: int = MIN_VALID_SIZE_DEFAULT) -> bool:
    """Return True only when file, size, and .ready marker are all valid."""
    resolved = _resolve_artifact_path(path)
    ready_lock = _ready_path(resolved)
    required_size = _effective_min_size(resolved, min_size)

    if not resolved.exists():
        return False
    if not resolved.is_file():
        return False

    size = os.path.getsize(resolved)
    if size <= 0 or size < required_size:
        return False

    if not ready_lock.exists():
        return False
    if _read_ready_content(ready_lock) != "ok":
        return False

    return True


def _log_skip_download(resolved: Path) -> None:
    print("SKIP DOWNLOAD", flush=True)
    logger.info("SKIP DOWNLOAD: artifact validated for %s", resolved)


def _remove_invalid_artifact(resolved: Path, min_size: int) -> None:
    """Remove corrupt or partial artifacts before a fresh download."""
    if _artifact_is_ready(resolved, min_size):
        return
    for stale in (_part_path(resolved), _ready_path(resolved), resolved):
        if stale.exists():
            try:
                stale.unlink()
            except OSError as exc:
                logger.warning("Failed to remove stale artifact %s: %s", stale, exc)


def _create_process_lock(lock: Path) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("downloading", encoding="utf-8")


def _remove_process_lock(lock: Path) -> None:
    try:
        lock.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove process lock %s: %s", lock, exc)


def _write_ready_lock(ready_lock: Path) -> None:
    ready_lock.parent.mkdir(parents=True, exist_ok=True)
    with open(ready_lock, "w", encoding="utf-8") as handle:
        handle.write("ok")
        handle.flush()
        os.fsync(handle.fileno())


def _wait_for_peer_lock(resolved: Path, min_size: int) -> bool:
    """Wait while a peer holds .lock; return False to skip download if still locked."""
    process_lock = _process_lock_path(resolved)

    for _ in range(LOCK_RETRY_ATTEMPTS):
        if _artifact_is_ready(resolved, min_size):
            return True
        if not process_lock.exists():
            return _artifact_is_ready(resolved, min_size)
        time.sleep(LOCK_RETRY_MS / 1000.0)

    if _artifact_is_ready(resolved, min_size):
        return True
    if process_lock.exists():
        return False
    return _artifact_is_ready(resolved, min_size)


def download_if_missing(url: str, path: Path | str) -> Path:
    """Download ``url`` to ``path`` when the artifact is not already ready."""
    resolved = _resolve_artifact_path(path)
    min_size = _effective_min_size(resolved, MIN_VALID_SIZE_DEFAULT)

    # Step 0 — early exit
    if _artifact_is_ready(resolved, min_size):
        _log_skip_download(resolved)
        return resolved

    process_lock = _process_lock_path(resolved)
    if process_lock.exists():
        if _wait_for_peer_lock(resolved, min_size):
            _log_skip_download(resolved)
            return resolved
        logger.info("Skipping download; peer lock still active for %s", resolved)
        if _artifact_is_ready(resolved, min_size):
            _log_skip_download(resolved)
            return resolved
        raise RuntimeError(f"Skipped download; artifact not ready: {resolved}")

    _remove_invalid_artifact(resolved, min_size)

    # Step 1 — acquire lock
    _create_process_lock(process_lock)

    part_path = _part_path(resolved)
    ready_lock = _ready_path(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    print("ARTIFACT DOWNLOAD", flush=True)
    logger.info("ARTIFACT DOWNLOAD: fetching %s -> %s", url, resolved)

    try:
        # Step 2 — download safely to .part
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()

        with open(part_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            # Step 3 — flush safety
            handle.flush()
            os.fsync(handle.fileno())

        # Step 4 — atomic move (only after fsync)
        os.replace(part_path, resolved)
        # Step 5 — mark ready (only after successful replace)
        _write_ready_lock(ready_lock)
        logger.info("Downloaded %s (%d bytes)", resolved, os.path.getsize(resolved))
        return resolved
    except Exception:
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass
        if ready_lock.exists():
            try:
                ready_lock.unlink()
            except OSError:
                pass
        raise
    finally:
        # Step 6 — cleanup lock (always)
        _remove_process_lock(process_lock)


def _ensure_artifact(path: Path | str) -> Path:
    """Ensure a pipeline artifact exists locally, downloading if configured."""
    resolved = _resolve_artifact_path(path)
    min_size = _effective_min_size(resolved, MIN_VALID_SIZE_DEFAULT)
    if _artifact_is_ready(resolved, min_size):
        _log_skip_download(resolved)
        return resolved

    rel = resolved.relative_to(_repo_root()).as_posix()
    remote_name = _ARTIFACT_REMOTE_NAMES.get(rel)
    if remote_name is None:
        raise FileNotFoundError(f"No remote mapping for artifact: {resolved}")

    base = ARTIFACT_BASE_URL.rstrip("/")
    if base == "TODO_SET_THIS":
        raise FileNotFoundError(
            f"Artifact missing: {resolved}. Set the ARTIFACT_BASE_URL environment variable "
            f"to a base URL hosting deployment artifacts, or generate data/ locally."
        )

    url = f"{base}/{remote_name}"
    return download_if_missing(url, resolved)


def _download_if_missing() -> tuple[str, ...]:
    """Ensure all deployment artifacts exist on disk (idempotent; no Streamlit)."""
    cfg = AppConfig.default()
    artifact = cfg.artifact
    paths = (
        artifact.chunks_path,
        artifact.embeddings_path,
        artifact.mentions_path,
        artifact.has_chunk_path,
        artifact.entities_path,
    )
    ensured: list[str] = []
    for path in paths:
        _ensure_artifact(path)
        ensured.append(str(_resolve_artifact_path(path)))
    return tuple(ensured)


@lru_cache(maxsize=1)
def ensure_deployment_artifacts() -> tuple[str, ...]:
    """Download missing deployment artifacts once per process (file-safe)."""
    return _download_if_missing()


@dataclass(frozen=True)
class LoadedArtifacts:
    """Container for all loaded pipeline artifacts."""

    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    mentions: list[dict[str, str]]
    has_chunk: list[dict[str, str]]
    entities: list[dict[str, str]]


class ArtifactLoader:
    """Load and validate chunks, embeddings, mentions, and graph edges."""

    @staticmethod
    def load(config: AppConfig) -> LoadedArtifacts:
        artifact = config.artifact

        ensure_deployment_artifacts()

        chunks_path = _resolve_artifact_path(artifact.chunks_path)
        embeddings_path = _resolve_artifact_path(artifact.embeddings_path)
        mentions_path = _resolve_artifact_path(artifact.mentions_path)
        has_chunk_path = _resolve_artifact_path(artifact.has_chunk_path)
        entities_path = _resolve_artifact_path(artifact.entities_path)

        chunks = list(iter_jsonl_gz(chunks_path))
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(chunks)})."
            )

        expected_dim = config.embedding.embedding_dim
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match config ({expected_dim})."
            )

        embeddings = normalize_embeddings(embeddings)

        mentions = load_csv(mentions_path, ["chunk_id", "entity_id"])
        has_chunk = load_csv(has_chunk_path, ["article_id", "chunk_id"])
        entities = load_csv(entities_path, ["entity_id", "name", "label"])

        ArtifactLoader._validate_mentions(chunks, mentions)

        return LoadedArtifacts(
            chunks=chunks,
            embeddings=embeddings,
            mentions=mentions,
            has_chunk=has_chunk,
            entities=entities,
        )

    @staticmethod
    def _validate_mentions(chunks: list[dict[str, Any]], mentions: list[dict[str, str]]) -> None:
        chunk_id_set = {str(chunk["chunk_id"]) for chunk in chunks}
        unknown_chunks = {rel["chunk_id"] for rel in mentions if rel["chunk_id"] not in chunk_id_set}
        if unknown_chunks:
            sample = sorted(unknown_chunks)[:5]
            raise ValueError(f"mentions.csv references unknown chunk_ids: {sample}")
