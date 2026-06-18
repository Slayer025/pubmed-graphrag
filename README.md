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

**Phase 1 (complete)** covers data loading, chunking, embedding, and visualization. Neo4j, retrieval, and generation are planned for later phases.

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

**Status:** Not started

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
```

## Next Steps

1. **Neo4j setup** — Docker or local instance; define connection config
2. **Entity extraction** — NER / relation extraction over semantic chunks
3. **Neo4j loader** — ingest chunks, entities, and edges
4. **Vector index** — attach embeddings to Neo4j nodes or external index
5. **Graph-enhanced retrieval** — hybrid vector + graph traversal
6. **LLM integration** — grounded answer generation
7. **Evaluation** — retrieval metrics, faithfulness benchmarks
8. **Demo** — query UI over the graph

## Constraints

- Do not download the full PubMed dataset (~880 MB zip / ~2.5 GB Arrow)
- Prefer HuggingFace **streaming** for remote data
- Prefer **gzip-compressed** storage (`.jsonl.gz`, `.npy`)
- **Estimate disk usage** before large writes; **abort if > 1 GB**
- Keep modules separate with type hints and logging
- HuggingFace cache pinned via `HF_HOME`

## Handoff Notes

**Current state:** Phase 1 is complete. The project has a 5000-abstract working set, three chunk strategies persisted as gzip JSONL, L2-normalized semantic embeddings, and a cluster visualization.

**Completed source files:**

- `src/load_data.py` — streaming data loader
- `src/chunker.py` — fixed / sentence / semantic chunking
- `src/create_chunks.py` — chunk dataset builder
- `src/embeddings.py` — semantic embedding encoder
- `src/visualize_chunks.py` — 2-D embedding plot
- `src/storage.py` — disk-budget and I/O helpers

**Generated artifacts:** See [Data Artifacts](#data-artifacts) above.

**Next implementation target:** Phase 2 — Neo4j setup, entity extraction, and graph loader. Start by defining a graph schema (Document → Chunk → Entity nodes) and a loader that reads `data/chunks/chunks_semantic.jsonl.gz` plus `data/embeddings/semantic_embeddings.npy`.

**Do not yet:** start Neo4j in this phase unless explicitly requested; retrieval, evaluation, and demo are out of scope until the graph exists.
