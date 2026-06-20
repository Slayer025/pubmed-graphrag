"""Phase-A artifact download (process start only; never during Streamlit runtime)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import requests

from src.config import AppConfig
from src.infrastructure.storage.safety_guard import (
    assert_no_repo_write,
    detect_repo_root,
    safe_mkdir,
    safe_write_file,
    verify_no_repo_writes,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.infrastructure.storage.artifact_loader import LoadedArtifacts

MIN_VALID_SIZE_DEFAULT = 1024
ARTIFACT_BASE_URL = os.environ.get("ARTIFACT_BASE_URL", "TODO_SET_THIS")

_ARTIFACT_LOGICAL_PATHS: tuple[str, ...] = (
    "data/chunks/chunks_semantic.jsonl.gz",
    "data/embeddings/semantic_embeddings.npy",
    "data/graph/mentions.csv",
    "data/graph/entities.csv",
    "data/graph/has_chunk.csv",
)

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

_streamlit_runtime = False
_bootstrap_complete = False
_downloading_allowed = False
_preloaded_artifacts: "LoadedArtifacts | None" = None


def mark_streamlit_runtime() -> None:
    """Mark that Streamlit has been imported; blocks further bootstrap calls."""
    global _streamlit_runtime
    _streamlit_runtime = True


def is_streamlit_runtime() -> bool:
    return _streamlit_runtime


def is_bootstrap_complete() -> bool:
    return _bootstrap_complete


@contextmanager
def _downloading_phase() -> Iterator[None]:
    global _downloading_allowed
    _downloading_allowed = True
    try:
        yield
    finally:
        _downloading_allowed = False


def assert_downloading_allowed() -> None:
    if not _downloading_allowed:
        raise RuntimeError("Artifact downloads are only allowed during bootstrap_artifacts()")


def default_cache_dir() -> str:
    env_dir = os.environ.get("ARTIFACT_CACHE_DIR", "").strip()
    return env_dir or "/tmp/pubmed-graphrag"


def artifact_paths(cache_dir: str) -> tuple[str, str, str, str, str]:
    """Return deterministic on-disk paths (no filesystem access)."""
    root = Path(cache_dir).resolve()
    return (
        str(root / "data/chunks/chunks_semantic.jsonl.gz"),
        str(root / "data/embeddings/semantic_embeddings.npy"),
        str(root / "data/graph/mentions.csv"),
        str(root / "data/graph/has_chunk.csv"),
        str(root / "data/graph/entities.csv"),
    )


def _repo_root() -> Path:
    return detect_repo_root()


def _cache_path(cache_dir: str, logical: str) -> Path:
    dest = (Path(cache_dir).resolve() / logical).resolve()
    assert_no_repo_write(str(dest))
    return dest


def _cache_hit(path: Path) -> bool:
    return path.is_file() and os.path.getsize(path) > 0


def _artifact_file_valid(path: Path, logical_key: str) -> bool:
    if not _cache_hit(path):
        return False
    return os.path.getsize(path) >= _MIN_VALID_SIZES.get(logical_key, MIN_VALID_SIZE_DEFAULT)


def _download_if_missing(url: str, dest: Path) -> None:
    assert_downloading_allowed()

    if _cache_hit(dest):
        logger.info("USING CACHED ARTIFACT: %s", dest)
        return

    safe_mkdir(dest.parent)
    part_path = Path(f"{dest}.part").resolve()
    assert_no_repo_write(str(part_path))

    logger.info("DOWNLOADING ARTIFACT: %s from %s", dest, url)

    try:
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()

        with safe_write_file(part_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        if not part_path.is_file() or os.path.getsize(part_path) == 0:
            raise OSError(f"Download produced empty file: {part_path}")

        os.replace(part_path, dest)
        logger.info("DOWNLOAD COMPLETED: %s (%d bytes)", dest, os.path.getsize(dest))
    except Exception:
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass
        raise


def _ensure_artifact(cache_dir: str, logical: str) -> Path:
    assert_downloading_allowed()
    logical_key = logical
    cache_path = _cache_path(cache_dir, logical)

    if _cache_hit(cache_path):
        logger.info("USING CACHED ARTIFACT: %s", cache_path)
        return cache_path

    repo_path = (_repo_root() / logical).resolve()
    if _artifact_file_valid(repo_path, logical_key):
        logger.info("Using existing repo artifact (read-only): %s", repo_path)
        return repo_path

    remote_name = _ARTIFACT_REMOTE_NAMES.get(logical_key)
    if remote_name is None:
        raise FileNotFoundError(f"No remote mapping for artifact: {logical_key}")

    base_url = ARTIFACT_BASE_URL.rstrip("/")
    if base_url == "TODO_SET_THIS":
        raise FileNotFoundError(
            f"Artifact missing at {cache_path}. Set ARTIFACT_BASE_URL or place files under "
            f"{repo_path} for local development."
        )

    _download_if_missing(f"{base_url}/{remote_name}", cache_path)
    return cache_path


def bootstrap_artifacts(cache_dir: str | None = None) -> None:
    """Download all deployment artifacts once at process start (before Streamlit cache)."""
    global _bootstrap_complete

    if _bootstrap_complete:
        return

    if _streamlit_runtime:
        raise RuntimeError(
            "bootstrap_artifacts() must run before Streamlit is imported; "
            "call it from scripts/demo.py at process start."
        )

    resolved_cache_dir = cache_dir or default_cache_dir()
    assert_no_repo_write(resolved_cache_dir)

    with _downloading_phase():
        for logical in _ARTIFACT_LOGICAL_PATHS:
            _ensure_artifact(resolved_cache_dir, logical)

    paths = list(artifact_paths(resolved_cache_dir))
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Bootstrap incomplete; artifact missing: {path}")

    verify_no_repo_writes(paths)

    global _preloaded_artifacts
    from src.infrastructure.storage.artifact_loader import ArtifactLoader

    cfg = AppConfig.default()
    _preloaded_artifacts = ArtifactLoader.load_from_paths(
        *paths,
        embedding_dim=cfg.embedding.embedding_dim,
    )

    _bootstrap_complete = True
    message = "ARTIFACT PHASE COMPLETE (ALL FILES LOCAL)"
    logger.info(message)
    print(message, flush=True)


def get_preloaded_artifacts() -> "LoadedArtifacts":
    """Return artifacts loaded during bootstrap (read-only after bootstrap)."""
    if _preloaded_artifacts is None:
        raise RuntimeError(
            "bootstrap_artifacts() must complete before pipeline build; "
            "call it at process start before importing Streamlit."
        )
    return _preloaded_artifacts
