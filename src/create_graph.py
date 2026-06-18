"""Phase 2 orchestrator: extract entities and export Neo4j-importable graph CSVs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.entity_extraction import extract_entities
from src.graph_loader import export_graph_to_csv

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/graph")
DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks_semantic.jsonl.gz")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings/semantic_embeddings.npy")
DEFAULT_ARTICLES_PATH = Path("data/pubmed_5000.jsonl.gz")
DEFAULT_ENTITIES_PATH = Path("data/graph/entities.jsonl.gz")


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def create_graph(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path | str = DEFAULT_EMBEDDINGS_PATH,
    articles_path: Path | str = DEFAULT_ARTICLES_PATH,
    entities_path: Path | str = DEFAULT_ENTITIES_PATH,
    *,
    skip_entity_extraction: bool = False,
) -> dict[str, Path]:
    """Run Phase 2 graph construction.

    Steps:
        1. Extract entities from semantic chunks (unless skipped).
        2. Export Article/Chunk/Entity nodes and relationships as CSVs.
        3. Write a Cypher import script.

    Args:
        output_dir: Directory for graph artifacts.
        chunks_path: Semantic chunk gzip JSONL.
        embeddings_path: Semantic embeddings `.npy` file.
        articles_path: Source abstracts gzip JSONL.
        entities_path: Entity mentions gzip JSONL (input/output).
        skip_entity_extraction: If True, reuse an existing entity file.

    Returns:
        Mapping of artifact names to written file paths.
    """
    _configure_logging()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_entity_extraction:
        extract_entities(
            input_path=chunks_path,
            output_path=entities_path,
        )
    else:
        logger.info("Skipping entity extraction; expecting %s", entities_path)

    paths = export_graph_to_csv(
        output_dir=output_dir,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        articles_path=articles_path,
        entities_path=entities_path,
    )

    logger.info("Phase 2 graph construction complete: %s", paths)
    return paths


if __name__ == "__main__":
    create_graph()
