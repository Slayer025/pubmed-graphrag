"""Text chunking utilities for PubMed abstracts (Phase 1)."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from transformers import PreTrainedTokenizerBase

from src.storage import DEFAULT_DATA_PATH, configure_hf_home, estimate_model_download, log_disk_estimate

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 100
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STRATEGY_FIXED: Literal["fixed"] = "fixed"
STRATEGY_SENTENCE: Literal["sentence"] = "sentence"
STRATEGY_SEMANTIC: Literal["semantic"] = "semantic"
StrategyName = Literal["fixed", "sentence", "semantic"]

# Matches load_abstracts() records: {"article_id": str, "abstract": str}
AbstractRecord = dict[str, Any]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class ChunkRecord(TypedDict):
    """Single chunk emitted by any chunking strategy."""

    article_id: str
    chunk_id: str
    text: str
    strategy: str


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def load_abstract_records(input_path: Path | str = DEFAULT_DATA_PATH) -> list[AbstractRecord]:
    """Load abstract records produced by ``load_abstracts()``."""
    from src.storage import load_jsonl_gz

    path = Path(input_path)
    logger.info("Loading abstracts from %s", path)
    records = load_jsonl_gz(path)
    logger.info("Loaded %d abstract records", len(records))
    return records


def _get_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    """Return a cached HuggingFace tokenizer aligned with the embedding model."""
    from transformers import AutoTokenizer

    configure_hf_home()
    if not hasattr(_get_tokenizer, "_cache"):
        _get_tokenizer._cache = {}  # type: ignore[attr-defined]
    cache: dict[str, PreTrainedTokenizerBase] = _get_tokenizer._cache  # type: ignore[attr-defined]
    if model_name not in cache:
        logger.info("Loading tokenizer for %s", model_name)
        cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return cache[model_name]


def _get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Return a cached sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    configure_hf_home()
    if not hasattr(_get_embedding_model, "_cache"):
        _get_embedding_model._cache = {}  # type: ignore[attr-defined]
    cache: dict[str, Any] = _get_embedding_model._cache  # type: ignore[attr-defined]
    if model_name not in cache:
        log_disk_estimate(estimate_model_download(model_name))
        logger.info("Loading embedding model %s", model_name)
        cache[model_name] = SentenceTransformer(model_name)
    return cache[model_name]


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count subword tokens in ``text``."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text: str) -> list[str]:
    """Split abstract text into sentences without breaking mid-sentence."""
    text = text.strip()
    if not text:
        return []

    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    return parts if parts else [text]


def _make_chunk_record(
    article_id: str,
    chunk_index: int,
    text: str,
    strategy: StrategyName,
) -> ChunkRecord:
    """Build a chunk record with a stable identifier."""
    return {
        "article_id": article_id,
        "chunk_id": f"{article_id}_{strategy}_{chunk_index:04d}",
        "text": text.strip(),
        "strategy": strategy,
    }


def fixed_token_chunking(
    text: str,
    article_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[ChunkRecord]:
    """Split text into fixed-size token windows with no overlap."""
    tokenizer = tokenizer or _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for start in range(0, len(token_ids), chunk_size):
        window = token_ids[start : start + chunk_size]
        chunk_text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if not chunk_text:
            continue
        chunks.append(_make_chunk_record(article_id, chunk_index, chunk_text, STRATEGY_FIXED))
        chunk_index += 1

    logger.debug(
        "fixed_token_chunking article_id=%s produced %d chunks",
        article_id,
        len(chunks),
    )
    return chunks


def sentence_chunking(
    text: str,
    article_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[ChunkRecord]:
    """Merge whole sentences into chunks of approximately ``chunk_size`` tokens."""
    tokenizer = tokenizer or _get_tokenizer()
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[ChunkRecord] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index, current_sentences, current_tokens
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences)
        chunks.append(_make_chunk_record(article_id, chunk_index, chunk_text, STRATEGY_SENTENCE))
        chunk_index += 1
        current_sentences = []
        current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, tokenizer)
        if current_sentences and current_tokens + sentence_tokens > chunk_size:
            flush()
        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    flush()

    logger.debug(
        "sentence_chunking article_id=%s produced %d chunks from %d sentences",
        article_id,
        len(chunks),
        len(sentences),
    )
    return chunks


def semantic_chunking(
    text: str,
    article_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_model: Any | None = None,
) -> list[ChunkRecord]:
    """Cluster sentence embeddings and merge each cluster into one ordered chunk."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return [_make_chunk_record(article_id, 0, sentences[0], STRATEGY_SEMANTIC)]

    model = embedding_model or _get_embedding_model(model_name)
    tokenizer = _get_tokenizer(model_name)

    logger.debug("Encoding %d sentences for semantic clustering (article_id=%s)", len(sentences), article_id)
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

    total_tokens = count_tokens(text, tokenizer)
    target_clusters = max(1, min(len(sentences), math.ceil(total_tokens / chunk_size)))

    clustering = AgglomerativeClustering(
        n_clusters=target_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        cluster_to_indices[int(label)].append(index)

    ordered_clusters = sorted(
        cluster_to_indices.values(),
        key=lambda indices: indices[0],
    )

    chunks: list[ChunkRecord] = []
    for chunk_index, indices in enumerate(ordered_clusters):
        ordered_indices = sorted(indices)
        chunk_text = " ".join(sentences[i] for i in ordered_indices)
        chunks.append(_make_chunk_record(article_id, chunk_index, chunk_text, STRATEGY_SEMANTIC))

    logger.debug(
        "semantic_chunking article_id=%s produced %d chunks from %d sentences (%d clusters)",
        article_id,
        len(chunks),
        len(sentences),
        target_clusters,
    )
    return chunks


def chunk_abstract(
    record: AbstractRecord,
    strategies: Sequence[StrategyName] = (STRATEGY_FIXED, STRATEGY_SENTENCE, STRATEGY_SEMANTIC),
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
    tokenizer: PreTrainedTokenizerBase | None = None,
    embedding_model: Any | None = None,
) -> list[ChunkRecord]:
    """Apply one or more chunking strategies to a single abstract record."""
    article_id = str(record["article_id"])
    text = str(record.get("abstract", "")).strip()
    if not text:
        logger.warning("Skipping empty abstract for article_id=%s", article_id)
        return []

    all_chunks: list[ChunkRecord] = []
    for strategy in strategies:
        if strategy == STRATEGY_FIXED:
            all_chunks.extend(
                fixed_token_chunking(text, article_id, chunk_size=chunk_size, tokenizer=tokenizer)
            )
        elif strategy == STRATEGY_SENTENCE:
            all_chunks.extend(
                sentence_chunking(text, article_id, chunk_size=chunk_size, tokenizer=tokenizer)
            )
        elif strategy == STRATEGY_SEMANTIC:
            all_chunks.extend(
                semantic_chunking(
                    text,
                    article_id,
                    chunk_size=chunk_size,
                    model_name=model_name,
                    embedding_model=embedding_model,
                )
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    return all_chunks


def chunk_documents(
    documents: Iterable[AbstractRecord],
    strategies: Sequence[StrategyName] = (STRATEGY_FIXED, STRATEGY_SENTENCE, STRATEGY_SEMANTIC),
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[ChunkRecord]:
    """Chunk a collection of abstract records using the requested strategies."""
    _configure_logging()

    documents_list = list(documents)
    logger.info(
        "Chunking %d documents with strategies=%s (chunk_size=%d)",
        len(documents_list),
        list(strategies),
        chunk_size,
    )

    tokenizer = _get_tokenizer(model_name)
    embedding_model = _get_embedding_model(model_name) if STRATEGY_SEMANTIC in strategies else None

    all_chunks: list[ChunkRecord] = []
    for index, record in enumerate(documents_list):
        chunks = chunk_abstract(
            record,
            strategies=strategies,
            chunk_size=chunk_size,
            model_name=model_name,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
        )
        all_chunks.extend(chunks)
        if (index + 1) % 500 == 0:
            logger.info("Processed %d / %d documents", index + 1, len(documents_list))

    logger.info("Produced %d total chunks", len(all_chunks))
    return all_chunks
