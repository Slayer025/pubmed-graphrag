"""Entity extraction for PubMed semantic chunks using spaCy with a regex fallback."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict

from src.storage import iter_jsonl_gz, save_jsonl_gz

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_SPACY_BATCH_SIZE = 128
DEFAULT_INPUT_PATH = Path("data/chunks/chunks_semantic.jsonl.gz")
DEFAULT_OUTPUT_PATH = Path("data/graph/entities.jsonl.gz")

# Fallback noun-phrase-ish pattern for environments without spaCy.
_FALLBACK_TOKEN_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[a-zA-Z]+){0,5}\b")


def _normalize_entity_name(name: str) -> str:
    """Return a trimmed, lower-cased entity name for stable keys."""
    return " ".join(name.lower().split())


def _make_entity_id(name: str, label: str) -> str:
    """Create a stable, filesystem-safe entity identifier."""
    normalized = _normalize_entity_name(name)
    safe = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return f"{label.lower()}:{safe}"


class EntitySpan(TypedDict):
    """One extracted entity mention."""

    entity_id: str
    name: str
    label: str


class EntityMentionRecord(TypedDict):
    """Entity mentions extracted from one chunk."""

    chunk_id: str
    entities: list[EntitySpan]


def _extract_with_spacy(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_SPACY_BATCH_SIZE,
) -> Iterator[list[EntitySpan]]:
    """Yield entity lists for each text using a spaCy pipeline."""
    import spacy

    logger.info("Loading spaCy model %s", model_name)
    nlp = spacy.load(model_name)
    logger.info("Processing %d texts with spaCy (batch_size=%d)", len(texts), batch_size)

    disabled = nlp.select_pipes(disable=["lemmatizer"])
    try:
        for doc in nlp.pipe(texts, batch_size=batch_size):
            seen: set[str] = set()
            entities: list[EntitySpan] = []

            # Named entities
            for ent in doc.ents:
                label = ent.label_
                name = ent.text
                entity_id = _make_entity_id(name, label)
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                entities.append(
                    {
                        "entity_id": entity_id,
                        "name": name,
                        "label": label,
                    }
                )

            # Noun chunks require parser; skip when parser is disabled.
            for noun_chunk in doc.noun_chunks:
                name = noun_chunk.text
                entity_id = _make_entity_id(name, "CONCEPT")
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                entities.append(
                    {
                        "entity_id": entity_id,
                        "name": name,
                        "label": "CONCEPT",
                    }
                )

            yield entities
    finally:
        disabled.restore()  # type: ignore[attr-defined]


def _extract_with_fallback(texts: list[str]) -> Iterator[list[EntitySpan]]:
    """Yield crude noun-phrase entities when spaCy is unavailable."""
    logger.warning("spaCy not available; using fallback entity extractor")
    for text in texts:
        seen: set[str] = set()
        entities: list[EntitySpan] = []
        for match in _FALLBACK_TOKEN_PATTERN.finditer(text):
            name = match.group(0)
            entity_id = _make_entity_id(name, "CONCEPT")
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entities.append(
                {
                    "entity_id": entity_id,
                    "name": name,
                    "label": "CONCEPT",
                }
            )
        yield entities


def extract_entities_for_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_SPACY_BATCH_SIZE,
) -> list[list[EntitySpan]]:
    """Extract entities from a list of chunk texts.

    Uses spaCy when installed, otherwise a lightweight regex fallback.
    """
    try:
        return list(_extract_with_spacy(texts, model_name=model_name, batch_size=batch_size))
    except ImportError:
        return list(_extract_with_fallback(texts))


def extract_entities_from_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_SPACY_BATCH_SIZE,
) -> Iterator[EntityMentionRecord]:
    """Extract entity mentions from an iterable of chunk records."""
    chunk_list = list(chunks)
    texts = [str(chunk.get("text", "")) for chunk in chunk_list]
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunk_list]

    entity_lists = extract_entities_for_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
    )

    for chunk_id, entities in zip(chunk_ids, entity_lists, strict=True):
        yield {"chunk_id": chunk_id, "entities": entities}


def extract_entities(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_SPACY_BATCH_SIZE,
) -> Path:
    """Extract entities from semantic chunks and save as gzip JSONL.

    Args:
        input_path: Semantic chunk gzip JSONL.
        output_path: Destination gzip JSONL.
        model_name: spaCy model name.
        batch_size: spaCy processing batch size.

    Returns:
        Path to the saved entity mentions file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    logger.info("Extracting entities from %s", input_path)
    chunks = iter_jsonl_gz(input_path)
    records = list(extract_entities_from_chunks(chunks, model_name=model_name, batch_size=batch_size))
    total_entities = sum(len(record["entities"]) for record in records)
    logger.info(
        "Extracted %d entity mentions across %d chunks",
        total_entities,
        len(records),
    )

    saved_path = save_jsonl_gz(records, output_path)
    return saved_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    extract_entities()
