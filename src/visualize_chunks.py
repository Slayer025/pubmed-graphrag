"""Visualize semantic chunk embeddings in 2-D projection space."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.chunker import ChunkRecord
from src.embeddings import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, load_embeddings
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = DEFAULT_INPUT_PATH
DEFAULT_EMBEDDINGS_PATH = DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_IMAGE = Path("outputs/semantic_clusters.png")
MAX_POINTS_FOR_PROJECTION = 10_000
RANDOM_STATE = 42

_CLUSTER_LABEL_PATTERN = re.compile(r"^(?P<article_id>.+)_(?P<strategy>[a-z]+)_(?P<cluster_idx>\d+)$")


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def load_semantic_chunks(input_path: Path | str = DEFAULT_CHUNKS_PATH) -> list[ChunkRecord]:
    """Load semantic chunks for labeling the visualization."""
    path = Path(input_path)
    logger.info("Loading semantic chunks from %s", path)
    chunks = list(iter_jsonl_gz(path))
    logger.info("Loaded %d semantic chunks", len(chunks))
    return chunks  # type: ignore[return-value]


def extract_cluster_labels(chunks: list[ChunkRecord]) -> np.ndarray:
    """Derive semantic cluster labels from ``chunk_id`` values."""
    labels: list[int] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        match = _CLUSTER_LABEL_PATTERN.match(chunk_id)
        if match:
            article_id = match.group("article_id")
            cluster_idx = int(match.group("cluster_idx"))
            # Stable integer label per article-local semantic cluster.
            labels.append(hash(f"{article_id}:{cluster_idx}") % 10_000)
        else:
            labels.append(hash(chunk_id) % 10_000)
    return np.asarray(labels, dtype=np.int32)


def _subsample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_points: int = MAX_POINTS_FOR_PROJECTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample points to keep projection memory-light."""
    if len(embeddings) <= max_points:
        return embeddings, labels

    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(embeddings), size=max_points, replace=False)
    logger.info("Subsampled embeddings from %d to %d points for projection", len(embeddings), max_points)
    return embeddings[indices], labels[indices]


def project_embeddings_2d(
    embeddings: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Project embeddings to 2-D using UMAP, falling back to t-SNE."""
    try:
        import umap

        logger.info("Projecting embeddings with UMAP")
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        logger.warning("umap-learn not installed; falling back to t-SNE")
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            random_state=random_state,
            perplexity=min(30, max(5, len(embeddings) - 1)),
            max_iter=1000,
        )
        return reducer.fit_transform(embeddings)


def plot_semantic_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path | str = DEFAULT_OUTPUT_IMAGE,
) -> Path:
    """Save a 2-D scatter plot colored by semantic cluster label."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab20",
        s=8,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title("Semantic Chunk Embeddings (2-D projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(scatter, ax=ax, label="Semantic cluster label")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info("Saved semantic cluster visualization to %s", path)
    return path


def visualize_semantic_chunks(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path | str = DEFAULT_EMBEDDINGS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_IMAGE,
) -> Path:
    """Load semantic chunks and embeddings, project, and save visualization."""
    _configure_logging()

    chunks = load_semantic_chunks(chunks_path)
    embeddings = load_embeddings(embeddings_path)

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match embedding rows ({len(embeddings)})."
        )

    labels = extract_cluster_labels(chunks)
    sampled_embeddings, sampled_labels = _subsample(embeddings.astype(np.float32), labels)
    coords = project_embeddings_2d(sampled_embeddings)
    return plot_semantic_clusters(coords, sampled_labels, output_path=output_path)


if __name__ == "__main__":
    visualize_semantic_chunks()
