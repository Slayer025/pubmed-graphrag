"""Artifact loader for Phase 1/2 data files."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.config import AppConfig
from src.embeddings import normalize_embeddings
from src.infrastructure.storage.csv_loader import load_csv
from src.infrastructure.storage.pure_build import assert_not_during_pure_build
from src.infrastructure.storage.safety_guard import (
    assert_no_repo_write,
    detect_repo_root,
    safe_mkdir,
    safe_write_file,
    verify_no_repo_writes,
)
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

MIN_VALID_SIZE_DEFAULT = 1024

# Remote artifact base URL (Streamlit Cloud). Override via ARTIFACT_BASE_URL.
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

_explicit_cache_root: Path | None = None


def set_artifact_cache_dir(cache_dir: str) -> None:
    """Pin artifact downloads to an explicit external cache directory."""
    assert_not_during_pure_build("artifact cache mutation")
    global _explicit_cache_root
    root = Path(cache_dir).resolve()
    assert_no_repo_write(str(root))
    _explicit_cache_root = root


def _repo_root() -> Path:
    """Return repository root without relying on process cwd."""
    return detect_repo_root()


def _normalize_relative(path: Path | str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.relative_to(_repo_root()).as_posix()
    return candidate.as_posix()


def get_cache_dir() -> Path:
    """Return the external artifact cache root (never under the repo directory)."""
    if _explicit_cache_root is not None:
        return _explicit_cache_root

    env_dir = os.environ.get("ARTIFACT_CACHE_DIR", "").strip()
    if env_dir:
        root = Path(env_dir).resolve()
    else:
        root = Path("/tmp/pubmed-graphrag").resolve()

    assert_no_repo_write(str(root))
    return root


def get_cache_path(relative_path: Path | str) -> Path:
    """Map a repo-relative artifact path to ``{CACHE_DIR}/data/...``."""
    rel = _normalize_relative(relative_path)
    dest = (get_cache_dir() / rel).resolve()
    assert_no_repo_write(str(dest))
    return dest


def _repo_artifact_path(logical: Path | str) -> Path:
    return (_repo_root() / _normalize_relative(logical)).resolve()


def _min_valid_size(logical_key: str) -> int:
    return _MIN_VALID_SIZES.get(logical_key, MIN_VALID_SIZE_DEFAULT)


def _cache_hit(path: Path) -> bool:
    """True when a cached artifact exists and is non-empty."""
    return path.is_file() and os.path.getsize(path) > 0


def _artifact_file_valid(path: Path, logical_key: str) -> bool:
    if not _cache_hit(path):
        return False
    return os.path.getsize(path) >= _min_valid_size(logical_key)


def resolve_artifact_path(logical: Path | str) -> Path:
    """Return the path to read an artifact from (cache first, then local repo copy)."""
    logical_key = _normalize_relative(logical)
    cache_path = get_cache_path(logical)
    if _artifact_file_valid(cache_path, logical_key):
        return cache_path

    repo_path = _repo_artifact_path(logical)
    if _artifact_file_valid(repo_path, logical_key):
        return repo_path

    return cache_path


def download_if_missing(url: str, logical: Path | str) -> Path:
    """Download to the external cache directory if the artifact file is missing."""
    assert_not_during_pure_build("artifact download")
    dest = get_cache_path(logical)

    if _cache_hit(dest):
        logger.info("USING CACHED ARTIFACT: %s", dest)
        return dest

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

        assert_no_repo_write(str(dest))
        os.replace(part_path, dest)
        logger.info("DOWNLOAD COMPLETED: %s (%d bytes)", dest, os.path.getsize(dest))
        return dest
    except Exception:
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass
        raise


def _ensure_artifact(logical: Path | str) -> Path:
    """Ensure artifact exists in external cache; never writes into the repo tree."""
    logical_key = _normalize_relative(logical)
    cache_path = get_cache_path(logical)

    if _cache_hit(cache_path):
        logger.info("USING CACHED ARTIFACT: %s", cache_path)
        return cache_path

    repo_path = _repo_artifact_path(logical)
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

    url = f"{base_url}/{remote_name}"
    return download_if_missing(url, logical)


def ensure_artifacts_present(cache_dir: str) -> tuple[str, str, str, str, str]:
    """Download missing deployment artifacts to external cache; return resolved paths."""
    set_artifact_cache_dir(cache_dir)
    cfg = AppConfig.default()
    artifact = cfg.artifact
    logical_paths = (
        artifact.chunks_path,
        artifact.embeddings_path,
        artifact.mentions_path,
        artifact.has_chunk_path,
        artifact.entities_path,
    )
    resolved = tuple(str(_ensure_artifact(path)) for path in logical_paths)
    verify_no_repo_writes(list(resolved))
    return resolved  # type: ignore[return-value]


def _download_if_missing() -> tuple[str, ...]:
    """Ensure all deployment artifacts exist (writes only to external cache)."""
    cfg = AppConfig.default()
    artifact = cfg.artifact
    paths = (
        artifact.chunks_path,
        artifact.embeddings_path,
        artifact.mentions_path,
        artifact.has_chunk_path,
        artifact.entities_path,
    )
    return tuple(str(_ensure_artifact(path)) for path in paths)


@lru_cache(maxsize=1)
def ensure_deployment_artifacts() -> tuple[str, ...]:
    """Download missing deployment artifacts once per process."""
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
    def load_from_paths(
        chunks_path: str,
        embeddings_path: str,
        mentions_path: str,
        has_chunk_path: str,
        entities_path: str,
        *,
        embedding_dim: int,
    ) -> LoadedArtifacts:
        """Load artifacts from existing on-disk paths (read-only, no downloads)."""
        for path in (chunks_path, embeddings_path, mentions_path, has_chunk_path, entities_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Artifact missing: {path}")

        verify_no_repo_writes(
            [chunks_path, embeddings_path, mentions_path, has_chunk_path, entities_path]
        )

        chunks = list(iter_jsonl_gz(Path(chunks_path)))
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(chunks)})."
            )

        if embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match config ({embedding_dim})."
            )

        embeddings = normalize_embeddings(embeddings)

        mentions = load_csv(Path(mentions_path), ["chunk_id", "entity_id"])
        has_chunk = load_csv(Path(has_chunk_path), ["article_id", "chunk_id"])
        entities = load_csv(Path(entities_path), ["entity_id", "name", "label"])

        ArtifactLoader._validate_mentions(chunks, mentions)

        return LoadedArtifacts(
            chunks=chunks,
            embeddings=embeddings,
            mentions=mentions,
            has_chunk=has_chunk,
            entities=entities,
        )

    @staticmethod
    def load(config: AppConfig) -> LoadedArtifacts:
        artifact = config.artifact

        ensure_deployment_artifacts()

        chunks_path = resolve_artifact_path(artifact.chunks_path)
        embeddings_path = resolve_artifact_path(artifact.embeddings_path)
        mentions_path = resolve_artifact_path(artifact.mentions_path)
        has_chunk_path = resolve_artifact_path(artifact.has_chunk_path)
        entities_path = resolve_artifact_path(artifact.entities_path)

        return ArtifactLoader.load_from_paths(
            str(chunks_path),
            str(embeddings_path),
            str(mentions_path),
            str(has_chunk_path),
            str(entities_path),
            embedding_dim=config.embedding.embedding_dim,
        )

    @staticmethod
    def _validate_mentions(chunks: list[dict[str, Any]], mentions: list[dict[str, str]]) -> None:
        chunk_id_set = {str(chunk["chunk_id"]) for chunk in chunks}
        unknown_chunks = {rel["chunk_id"] for rel in mentions if rel["chunk_id"] not in chunk_id_set}
        if unknown_chunks:
            sample = sorted(unknown_chunks)[:5]
            raise ValueError(f"mentions.csv references unknown chunk_ids: {sample}")
