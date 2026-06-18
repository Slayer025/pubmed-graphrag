"""Embedding generation for PubMed text chunks.

Disk policy (disk02): before downloading a model, call
``storage.estimate_model_download(model_name)`` and ``storage.log_disk_estimate()``.
Prefer small sentence-transformers checkpoints; warn if weights may exceed 1 GB.
"""

from typing import Any

import numpy as np

from src.storage import estimate_model_download, log_disk_estimate


def create_embedding_model(model_name: str) -> Any:
    """Initialize an embedding model.

    Args:
        model_name: Identifier or path of the embedding model to load.

    Returns:
        A loaded embedding model instance.
    """
    # disk02: log expected Hub download size before fetching weights.
    log_disk_estimate(estimate_model_download(model_name))
    raise NotImplementedError


def embed_texts(texts: list[str], model: Any) -> np.ndarray:
    """Generate embeddings for a list of text strings.

    Args:
        texts: Input strings to embed.
        model: Embedding model returned by ``create_embedding_model``.

    Returns:
        A 2-D array of shape ``(len(texts), embedding_dim)``.
    """
    raise NotImplementedError


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: Any,
    text_field: str = "text",
) -> list[dict[str, Any]]:
    """Attach embedding vectors to chunk records.

    Args:
        chunks: Chunk records produced by the chunking step.
        model: Embedding model returned by ``create_embedding_model``.
        text_field: Key used to read text from each chunk record.

    Returns:
        Chunk records augmented with an ``embedding`` field.
    """
    raise NotImplementedError
