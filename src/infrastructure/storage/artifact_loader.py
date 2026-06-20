"""Read-only artifact loader (no downloads, no mkdir, no fallback fetch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AppConfig
from src.embeddings import normalize_embeddings
from src.infrastructure.storage.csv_loader import load_csv
from src.infrastructure.storage.safety_guard import verify_no_repo_writes
from src.storage import iter_jsonl_gz

_DOWNLOADING_ALLOWED = False


def assert_not_downloading() -> None:
    if _DOWNLOADING_ALLOWED:
        raise RuntimeError("ArtifactLoader must remain read-only; downloading is not allowed")


@dataclass(frozen=True)
class LoadedArtifacts:
    """Container for all loaded pipeline artifacts."""

    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    mentions: list[dict[str, str]]
    has_chunk: list[dict[str, str]]
    entities: list[dict[str, str]]


class ArtifactLoader:
    """Load and validate chunks, embeddings, mentions, and graph edges (read-only)."""

    @staticmethod
    def load(config: AppConfig) -> LoadedArtifacts:
        """Load artifacts after bootstrap download phase (CLI/scripts)."""
        from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts, default_cache_dir, get_preloaded_artifacts

        bootstrap_artifacts(default_cache_dir())
        return get_preloaded_artifacts()

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
        """Open and parse existing files only. Missing files raise immediately."""
        assert_not_downloading()

        paths = [chunks_path, embeddings_path, mentions_path, has_chunk_path, entities_path]
        verify_no_repo_writes(paths)

        try:
            chunks = list(iter_jsonl_gz(Path(chunks_path)))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Artifact missing: {chunks_path}") from exc

        try:
            embeddings = np.load(embeddings_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Artifact missing: {embeddings_path}") from exc

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(chunks)})."
            )

        if embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match config ({embedding_dim})."
            )

        embeddings = normalize_embeddings(embeddings)

        try:
            mentions = load_csv(Path(mentions_path), ["chunk_id", "entity_id"])
            has_chunk = load_csv(Path(has_chunk_path), ["article_id", "chunk_id"])
            entities = load_csv(Path(entities_path), ["entity_id", "name", "label"])
        except FileNotFoundError as exc:
            raise FileNotFoundError("Graph artifact missing during read-only load") from exc

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
