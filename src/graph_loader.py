"""Build Neo4j-importable node/relationship tables from chunks, embeddings, and entities."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.graph_schema import (
    ARTICLE_LABEL,
    CHUNK_LABEL,
    ENTITY_LABEL,
    REL_HAS_CHUNK,
    REL_MENTIONS,
    GraphSchema,
)
from src.storage import (
    WARN_THRESHOLD_BYTES,
    format_bytes,
    iter_jsonl_gz,
    load_jsonl_gz,
    log_disk_estimate,
    DiskUsageEstimate,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/graph")
DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks_semantic.jsonl.gz")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings/semantic_embeddings.npy")
DEFAULT_ARTICLES_PATH = Path("data/pubmed_5000.jsonl.gz")
DEFAULT_ENTITIES_PATH = Path("data/graph/entities.jsonl.gz")

# CSV sizing heuristics.
_AVG_ARTICLE_ROW_BYTES = 700
_AVG_CHUNK_ROW_BYTES = 350
_AVG_ENTITY_ROW_BYTES = 80
_AVG_REL_ROW_BYTES = 40


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _embedding_to_csv_value(vector: np.ndarray) -> str:
    """Convert a float32 vector to a semicolon-separated string."""
    return ";".join(f"{value:.6f}" for value in vector)


def _chunk_id_to_article_id(chunk_id: str) -> str:
    """Derive article_id from chunk_id (format: article_id_strategy_index)."""
    # chunk_id pattern: {article_id}_{strategy}_{index:04d}
    parts = chunk_id.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    # Fallback: take everything before the last underscore.
    return chunk_id.rsplit("_", 1)[0]


def load_semantic_chunks_with_embeddings(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path | str = DEFAULT_EMBEDDINGS_PATH,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load semantic chunks and their corresponding embedding matrix."""
    chunks = list(iter_jsonl_gz(chunks_path))
    embeddings = np.load(embeddings_path)
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match embedding rows ({embeddings.shape[0]})."
        )
    logger.info("Loaded %d chunks and embeddings %s", len(chunks), embeddings.shape)
    return chunks, embeddings


def build_article_nodes(
    chunks: list[dict[str, Any]],
    articles_path: Path | str = DEFAULT_ARTICLES_PATH,
) -> list[dict[str, str]]:
    """Build deduplicated Article nodes, joining abstracts from the source file."""
    articles_by_id: dict[str, dict[str, str]] = {}

    # Start with article_ids observed in chunks.
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        article_id = _chunk_id_to_article_id(chunk_id)
        articles_by_id[article_id] = {"article_id": article_id, "abstract": ""}

    # Fill in abstracts where available.
    try:
        article_records = load_jsonl_gz(articles_path)
    except FileNotFoundError:
        logger.warning("Article source %s not found; Article nodes will have empty abstracts", articles_path)
        article_records = []

    for record in article_records:
        article_id = str(record.get("article_id", ""))
        if article_id in articles_by_id:
            abstract = str(record.get("abstract", ""))
            articles_by_id[article_id]["abstract"] = abstract

    return list(articles_by_id.values())


def build_chunk_nodes(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> list[dict[str, str]]:
    """Build Chunk nodes with semicolon-separated embeddings."""
    chunk_nodes: list[dict[str, str]] = []
    for chunk, vector in zip(chunks, embeddings, strict=True):
        chunk_id = str(chunk["chunk_id"])
        article_id = _chunk_id_to_article_id(chunk_id)
        chunk_nodes.append(
            {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "text": str(chunk.get("text", "")),
                "strategy": str(chunk.get("strategy", "semantic")),
                "embedding": _embedding_to_csv_value(vector),
            }
        )
    return chunk_nodes


def build_entity_nodes_and_mentions(
    entities_path: Path | str = DEFAULT_ENTITIES_PATH,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build deduplicated Entity nodes and Chunk->Entity MENTIONS relationships."""
    entity_nodes_by_id: dict[str, dict[str, str]] = {}
    mentions: list[dict[str, str]] = []

    try:
        records = load_jsonl_gz(entities_path)
    except FileNotFoundError:
        logger.warning("Entity file %s not found; graph will contain no entities", entities_path)
        return [], []

    for record in records:
        chunk_id = str(record.get("chunk_id", ""))
        for entity in record.get("entities", []):
            entity_id = str(entity["entity_id"])
            name = str(entity["name"])
            label = str(entity["label"])
            if entity_id not in entity_nodes_by_id:
                entity_nodes_by_id[entity_id] = {
                    "entity_id": entity_id,
                    "name": name,
                    "label": label,
                }
            mentions.append({"chunk_id": chunk_id, "entity_id": entity_id})

    return list(entity_nodes_by_id.values()), mentions


def build_has_chunk_relationships(chunk_nodes: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build Article->Chunk HAS_CHUNK relationships."""
    return [{"article_id": node["article_id"], "chunk_id": node["chunk_id"]} for node in chunk_nodes]


def estimate_graph_export_bytes(
    num_articles: int,
    num_chunks: int,
    num_entities: int,
    num_has_chunk_rels: int,
    num_mentions: int,
) -> int:
    """Estimate uncompressed CSV size for the full graph export."""
    return int(
        num_articles * _AVG_ARTICLE_ROW_BYTES
        + num_chunks * _AVG_CHUNK_ROW_BYTES
        + num_entities * _AVG_ENTITY_ROW_BYTES
        + (num_has_chunk_rels + num_mentions) * _AVG_REL_ROW_BYTES
    )


def _write_csv(
    rows: list[dict[str, str]],
    output_path: Path,
    fieldnames: list[str],
) -> Path:
    """Write a list of dicts to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows, %s)", output_path, len(rows), format_bytes(output_path.stat().st_size))
    return output_path


def export_graph_to_csv(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path | str = DEFAULT_EMBEDDINGS_PATH,
    articles_path: Path | str = DEFAULT_ARTICLES_PATH,
    entities_path: Path | str = DEFAULT_ENTITIES_PATH,
) -> dict[str, Path]:
    """Export Article/Chunk/Entity nodes and relationships as Neo4j CSV files.

    Args:
        output_dir: Destination directory for CSV files.
        chunks_path: Semantic chunk gzip JSONL.
        embeddings_path: Semantic embeddings `.npy` file.
        articles_path: Source abstracts gzip JSONL.
        entities_path: Entity mentions gzip JSONL.

    Returns:
        Mapping of artifact name to written file path.
    """
    _configure_logging()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks, embeddings = load_semantic_chunks_with_embeddings(chunks_path, embeddings_path)
    article_nodes = build_article_nodes(chunks, articles_path=articles_path)
    chunk_nodes = build_chunk_nodes(chunks, embeddings)
    entity_nodes, mentions = build_entity_nodes_and_mentions(entities_path)
    has_chunk_rels = build_has_chunk_relationships(chunk_nodes)

    estimated_bytes = estimate_graph_export_bytes(
        num_articles=len(article_nodes),
        num_chunks=len(chunk_nodes),
        num_entities=len(entity_nodes),
        num_has_chunk_rels=len(has_chunk_rels),
        num_mentions=len(mentions),
    )
    log_disk_estimate(
        DiskUsageEstimate(
            step="graph_csv_export",
            retained_bytes=estimated_bytes,
            peak_transient_bytes=estimated_bytes + len(chunks) * embeddings.shape[1] * 4,
            uses_streaming=False,
            uses_compression=False,
            notes="Neo4j CSV files under data/graph/ plus transient embedding matrix.",
        )
    )
    if estimated_bytes > WARN_THRESHOLD_BYTES:
        raise RuntimeError(
            f"Estimated graph export {format_bytes(estimated_bytes)} exceeds limit of 1 GB; aborting."
        )

    schema = GraphSchema(output_dir=str(output_dir))

    paths: dict[str, Path] = {}
    paths["articles"] = _write_csv(
        article_nodes,
        output_dir / schema.articles_csv,
        ["article_id", "abstract"],
    )
    paths["chunks"] = _write_csv(
        chunk_nodes,
        output_dir / schema.chunks_csv,
        ["chunk_id", "article_id", "text", "strategy", "embedding"],
    )
    paths["entities"] = _write_csv(
        entity_nodes,
        output_dir / schema.entities_csv,
        ["entity_id", "name", "label"],
    )
    paths["has_chunk"] = _write_csv(
        has_chunk_rels,
        output_dir / schema.has_chunk_csv,
        ["article_id", "chunk_id"],
    )
    paths["mentions"] = _write_csv(
        mentions,
        output_dir / schema.mentions_csv,
        ["chunk_id", "entity_id"],
    )

    cypher_path = output_dir / schema.cypher_file
    cypher_path.write_text(schema.cypher, encoding="utf-8")
    logger.info("Wrote Cypher import script to %s", cypher_path)
    paths["cypher"] = cypher_path

    return paths


if __name__ == "__main__":
    export_graph_to_csv()
