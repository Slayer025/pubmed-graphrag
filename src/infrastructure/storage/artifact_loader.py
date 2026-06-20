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
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

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


def _repo_root() -> Path:
    """Return repository root without relying on process cwd."""
    return Path(__file__).resolve().parents[3]


def _resolve_artifact_path(path: Path | str) -> Path:
    """Resolve artifact paths against repo root when relative."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return _repo_root() / resolved


def download_if_missing(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` when the file is not already present."""
    path = _resolve_artifact_path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("ARTIFACT DOWNLOAD", flush=True)
    logger.info("ARTIFACT DOWNLOAD: fetching %s -> %s", url, path)
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    path.write_bytes(response.content)
    logger.info("Downloaded %s (%d bytes)", path, path.stat().st_size)
    return path


def _ensure_artifact(path: Path | str) -> Path:
    """Ensure a pipeline artifact exists locally, downloading if configured."""
    path = _resolve_artifact_path(path)
    if path.exists():
        return path

    rel = path.relative_to(_repo_root()).as_posix()
    remote_name = _ARTIFACT_REMOTE_NAMES.get(rel)
    if remote_name is None:
        raise FileNotFoundError(f"No remote mapping for artifact: {path}")

    base = ARTIFACT_BASE_URL.rstrip("/")
    if base == "TODO_SET_THIS":
        raise FileNotFoundError(
            f"Artifact missing: {path}. Set the ARTIFACT_BASE_URL environment variable "
            f"to a base URL hosting deployment artifacts, or generate data/ locally."
        )

    url = f"{base}/{remote_name}"
    return download_if_missing(url, path)


def _download_if_missing() -> tuple[str, ...]:
    """Download all deployment artifacts once (pure; no Streamlit)."""
    logger.info("Ensuring deployment artifacts are present on disk")
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
        ensured.append(str(_resolve_artifact_path(path).resolve()))
    return tuple(ensured)


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
