"""Visualization helpers for GraphRAG outputs."""

from pathlib import Path
from typing import Any

import numpy as np


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    output_path: Path | str | None = None,
) -> Any:
    """Visualize embeddings in a reduced 2-D space.

    Args:
        embeddings: 2-D array of shape ``(n_samples, embedding_dim)``.
        labels: Optional labels for each embedding point.
        output_path: Optional path to save the figure.

    Returns:
        A matplotlib or plotly figure object.
    """
    raise NotImplementedError


def plot_graph(
    graph: Any,
    output_path: Path | str | None = None,
) -> Any:
    """Render a knowledge graph built from PubMed entities and relations.

    Args:
        graph: Graph object (e.g. NetworkX graph) to visualize.
        output_path: Optional path to save the figure.

    Returns:
        A matplotlib or plotly figure object.
    """
    raise NotImplementedError


def save_visualization(figure: Any, output_path: Path | str) -> None:
    """Persist a visualization figure to disk.

    Args:
        figure: Figure object returned by a plotting function.
        output_path: Destination file path.
    """
    raise NotImplementedError
