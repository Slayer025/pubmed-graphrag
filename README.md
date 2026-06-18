# GraphRAG for Scientific Literature

## Project Goal

This project builds a **GraphRAG** pipeline over PubMed scientific abstracts. The end-to-end vision:

- **PubMed GraphRAG** — ingest a reproducible subset of biomedical literature
- **Semantic chunking** — split abstracts into meaning-aware units for retrieval
- **Neo4j knowledge graph** — store entities, relations, and document structure
- **Graph-enhanced retrieval** — combine vector search with graph traversal
- **LLM generation** — answer questions grounded in retrieved evidence
- **Evaluation** — measure retrieval quality and answer faithfulness
- **Demo application** — interactive query interface over the graph

**Phase 1 (complete)** covers data loading, chunking, embedding, and visualization. **Phase 2 (complete)** covers entity extraction and Neo4j-importable graph export. Retrieval and generation are planned for later phases.

## Environment

**OS:** WSL2 Ubuntu

**Project location:** `/mnt/d/pubmed-graphrag`

**Symlink:** `~/projects/pubmed-graphrag`

**Virtual environment:** `/mnt/d/pubmed-graphrag/.venv`

**Python version:** 3.14.4

Activate the environment before running commands:

```bash
source /mnt/d/pubmed-graphrag/.venv/bin/activate
export HF_HOME=/mnt/d/hf_cache_backup
export PIP_CACHE_DIR=/mnt/d/pip_cache_backup
```

**HF cache:** `HF_HOME=/mnt/d/hf_cache_backup`

**Pip cache:** `PIP_CACHE_DIR=/mnt/d/pip_cache_backup`

## Dataset

**Dataset source:** HuggingFace `scientific_papers/pubmed` (`armanc/scientific_papers`)

**Sampling strategy:** Stream train split; collect first **5000** valid abstracts only

**Storage format:** gzip-compressed JSONL

**Input file:** `data/pubmed_5000.jsonl.gz` (~2.1 MB)

Each record:

```json
{"article_id": "0", "abstract": "..."}
```

## Current Architecture

### `src/load_data.py`

| | |
|---|---|
| **Purpose** | Stream PubMed abstracts from HuggingFace, sample 5000, save compressed subset |
| **Input** | HuggingFace `scientific_papers/pubmed` (streaming) |
| **Output** | `data/pubmed_5000.jsonl.gz` |

### `src/chunker.py`

| | |
|---|---|
| **Purpose** | Three chunking strategies over abstract text |
| **Input** | Abstract records (`article_id`, `abstract`) |
| **Output** | Chunk records (`article_id`, `chunk_id`, `text`, `strategy`) |

Strategies: `fixed` (100-token windows), `sentence` (~100-token sentence groups), `semantic` (embedding clustering).

### `src/create_chunks.py`

| | |
|---|---|
| **Purpose** | Orchestrate chunking and persist three strategy datasets |
| **Input** | `data/pubmed_5000.jsonl.gz` |
| **Output** | `data/chunks/chunks_{fixed,sentence,semantic}.jsonl.gz` |

Estimates output size before writing; aborts if estimate exceeds 1 GB.

### `src/embeddings.py`

| | |
|---|---|
| **Purpose** | Batch-encode semantic chunks with sentence-transformers |
| **Input** | `data/chunks/chunks_semantic.jsonl.gz` |
| **Output** | `data/embeddings/semantic_embeddings.npy` (L2-normalized float32) |

Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

### `src/visualize_chunks.py`

| | |
|---|---|
| **Purpose** | 2-D projection of semantic embeddings for quality inspection |
| **Input** | Semantic chunks + `semantic_embeddings.npy` |
| **Output** | `outputs/semantic_clusters.png` |

Uses UMAP when available; falls back to t-SNE. Subsamples to 10 000 points for memory efficiency.

### `src/graph_schema.py`

| | |
|---|---|
| **Purpose** | Define Neo4j node/relationship schema and generate import Cypher |
| **Input** | None (schema constants) |
| **Output** | `GraphSchema` object; `data/graph/schema.cypher` |

Schema: `:Article {article_id, abstract}`, `:Chunk {chunk_id, article_id, text, strategy, embedding}`, `:Entity {entity_id, name, label}`; relationships `:HAS_CHUNK` (Article → Chunk) and `:MENTIONS` (Chunk → Entity).

### `src/entity_extraction.py`

| | |
|---|---|
| **Purpose** | Extract named entities and noun phrases from semantic chunks |
| **Input** | `data/chunks/chunks_semantic.jsonl.gz` |
| **Output** | `data/graph/entities.jsonl.gz` |

Uses spaCy `en_core_web_sm` (~13 MB); falls back to a lightweight regex extractor if spaCy is unavailable.

### `src/graph_loader.py`

| | |
|---|---|
| **Purpose** | Build Article / Chunk / Entity nodes and HAS_CHUNK / MENTIONS edges as CSVs |
| **Input** | Semantic chunks, embeddings, source abstracts, and extracted entity mentions |
| **Output** | `data/graph/*.csv` and `data/graph/schema.cypher` |

Estimates output size before writing; aborts if estimate exceeds 1 GB.

### `src/create_graph.py`

| | |
|---|---|
| **Purpose** | Orchestrate Phase 2: entity extraction + graph CSV export |
| **Input** | Phase 1 artifacts |
| **Output** | All files under `data/graph/` |

### `src/storage.py`

Shared disk-budget utilities: gzip I/O, HF cache configuration, cleanup helpers, size estimation and 1 GB warnings (disk02 policy).

## Data Artifacts

| File | Size | Description |
|------|------|-------------|
| `data/pubmed_5000.jsonl.gz` | 2.1 MB | 5000 sampled abstracts |
| `data/chunks/chunks_fixed.jsonl.gz` | 2.2 MB | 15 582 fixed-token chunks |
| `data/chunks/chunks_sentence.jsonl.gz` | 2.2 MB | 17 449 sentence chunks |
| `data/chunks/chunks_semantic.jsonl.gz` | 2.2 MB | 15 556 semantic chunks |
| `data/embeddings/semantic_embeddings.npy` | 23 MB | Shape `(15556, 384)` |
| `outputs/semantic_clusters.png` | 712 KB | 2-D embedding visualization |
| `data/graph/entities.jsonl.gz` | 407 KB | Entity mentions per semantic chunk |
| `data/graph/articles.csv` | 5.9 MB | 5 000 Article nodes |
| `data/graph/chunks.csv` | 60.5 MB | 15 556 Chunk nodes with embeddings |
| `data/graph/entities.csv` | 7.5 MB | 137 219 deduplicated Entity nodes |
| `data/graph/has_chunk.csv` | 373 KB | 15 556 Article → Chunk edges |
| `data/graph/mentions.csv` | 10.6 MB | 258 464 Chunk → Entity edges |
| `data/graph/schema.cypher` | 1.8 KB | Neo4j constraints, indexes, and `LOAD CSV` script |

## Phase Progress

### Phase 0 — Project setup

**Status:** Complete

- Repository structure (`data/`, `outputs/`, `src/`, `notebooks/`)
- Storage-efficient loading policy (streaming, gzip, disk02 rules)
- Environment on D: drive with external HF/pip caches

### Phase 1 — Chunking, embeddings, visualization

**Status:** Complete

- [x] Load 5000 PubMed abstracts (`load_data.py`)
- [x] Implement three chunking strategies (`chunker.py`)
- [x] Save chunk datasets (`create_chunks.py`)
- [x] Encode semantic chunks (`embeddings.py`)
- [x] Visualize embedding clusters (`visualize_chunks.py`)

### Phase 2 — Neo4j graph construction

**Status:** Complete

- [x] Define graph schema (`graph_schema.py`)
- [x] Extract entities from semantic chunks (`entity_extraction.py`)
- [x] Build Neo4j-importable node/edge CSVs (`graph_loader.py`)
- [x] Orchestrate Phase 2 (`create_graph.py`)

### Phase 3 — Retrieval

**Status:** Not started

### Phase 4 — LLM generation & evaluation

**Status:** Not started

### Phase 5 — Demo application

**Status:** Not started

## Commands

Run the full Phase 1 pipeline from the project root:

```bash
cd /mnt/d/pubmed-graphrag
source .venv/bin/activate
export HF_HOME=/mnt/d/hf_cache_backup
export PIP_CACHE_DIR=/mnt/d/pip_cache_backup

# Step 1: Load abstracts (already done if pubmed_5000.jsonl.gz exists)
python -m src.load_data

# Step 2: Generate chunk datasets
python -m src.create_chunks

# Step 3: Encode semantic chunks
python -m src.embeddings

# Step 4: Visualize embeddings
python -m src.visualize_chunks

# Step 5: Build Neo4j-importable graph (Phase 2)
python -m src.create_graph
```

## Phase 2 import (after Neo4j is running)

Copy `data/graph/*.csv` and `data/graph/schema.cypher` to your Neo4j import directory, then run the Cypher script in Neo4j Browser or `cypher-shell`:

```cypher
:source schema.cypher
```

No Neo4j server is required to generate the CSV files — `create_graph.py` produces import-ready artifacts offline.

## Next Steps

1. **Neo4j setup** — Docker or local instance; load `data/graph/*.csv` with `schema.cypher`
2. **Vector index** — create a Neo4j vector index over `Chunk.embedding` (or keep an external FAISS/Annoy index)
3. **Graph-enhanced retrieval** — hybrid vector + graph traversal
4. **LLM integration** — grounded answer generation
5. **Evaluation** — retrieval metrics, faithfulness benchmarks
6. **Demo** — query UI over the graph

## Constraints

- Do not download the full PubMed dataset (~880 MB zip / ~2.5 GB Arrow)
- Prefer HuggingFace **streaming** for remote data
- Prefer **gzip-compressed** storage (`.jsonl.gz`, `.npy`)
- **Estimate disk usage** before large writes; **abort if > 1 GB**
- Keep modules separate with type hints and logging
- HuggingFace cache pinned via `HF_HOME`

## Handoff Notes

**Current state:** Phase 1 and Phase 2 are complete. The project has a 5000-abstract working set, three chunk strategies persisted as gzip JSONL, L2-normalized semantic embeddings, a cluster visualization, and a Neo4j-importable graph export (137k entities, 258k mentions) generated offline.

**Completed source files:**

- `src/load_data.py` — streaming data loader
- `src/chunker.py` — fixed / sentence / semantic chunking
- `src/create_chunks.py` — chunk dataset builder
- `src/embeddings.py` — semantic embedding encoder
- `src/visualize_chunks.py` — 2-D embedding plot
- `src/storage.py` — disk-budget and I/O helpers
- `src/graph_schema.py` — Neo4j graph schema and Cypher generator
- `src/entity_extraction.py` — spaCy NER + noun-phrase entity extractor
- `src/graph_loader.py` — Neo4j CSV node/relationship exporter
- `src/create_graph.py` — Phase 2 orchestrator

**Generated artifacts:** See [Data Artifacts](#data-artifacts) above.

**Next implementation target:** Phase 3 — Retrieval. Start by loading the Phase 2 CSV files into a running Neo4j instance, create a vector index on `Chunk.embedding`, and implement a hybrid retriever that combines vector similarity with graph traversal (e.g., expand retrieved chunks via `MENTIONS` to related entities and co-mentioned chunks).

**Do not yet:** start Neo4j in this phase unless explicitly requested; retrieval, evaluation, and demo are out of scope until the graph is loaded into a database.
