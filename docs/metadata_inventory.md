# Metadata Inventory

**Date:** 2026-06-24  
**Scope:** Phase 4 Step 1 — inventory only; no filtering/boosting logic implemented yet.

This document lists all metadata fields available on chunks, PubMed articles, and the derived graph, along with their storage location, example values, and candidate use for future metadata-aware retrieval (filtering / boosting).

---

## 1. Chunk Metadata

### Source files

| File | Format | Purpose |
|------|--------|---------|
| `data/chunks/chunks_semantic.jsonl.gz` | gzip JSONL | Semantic chunks used by the pipeline |
| `data/chunks/chunks_fixed.jsonl.gz` | gzip JSONL | Fixed-size chunks |
| `data/chunks/chunks_sentence.jsonl.gz` | gzip JSONL | Sentence-level chunks |
| `data/graph/chunks.csv` | CSV | Unified chunk table with embeddings |

### Fields

| Field | Type | Example | Available on all chunk files | Notes |
|-------|------|---------|------------------------------|-------|
| `article_id` | string | `"0"` | Yes | Foreign key to `pubmed_5000.jsonl.gz` and `articles.csv` / `has_chunk.csv` |
| `chunk_id` | string | `"0_semantic_0000"` | Yes | Unique identifier; encodes article, strategy, and sequence |
| `text` | string | chunk of abstract text | Yes | Retrieval content |
| `strategy` | string | `"semantic"`, `"fixed"` | Yes | Chunking strategy that produced the chunk |
| `embedding` | semicolon-delimited floats | `0.054182;0.088479;...` | `chunks.csv` only | Precomputed dense vector; dimensionality matches the configured encoder |

### Candidate retrieval uses (future phases)

- `strategy`: can be used to weight or filter by chunk granularity (e.g., prefer `semantic` chunks for relationship queries where boundaries matter).
- `article_id`: enables article-level deduplication, result grouping, and metadata join with the original PubMed record.
- `chunk_id`: positional metadata can be extracted (`<article>_<strategy>_<sequence>`) to boost earlier chunks in an abstract.

---

## 2. Original PubMed Article Metadata

### Source file

| File | Format | Purpose |
|------|--------|---------|
| `data/pubmed_5000.jsonl.gz` | gzip JSONL | Original PubMed records loaded into the system |

### Fields

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `article_id` | string | `"0"` | Primary key |
| `abstract` | string | full abstract text with `\n` separators | Source text for chunking |

### Observations

- Records appear to be minimal PubMed snapshots: only `article_id` and `abstract` are present in this corpus.
- No explicit fields observed for: title, authors, journal, MeSH terms, publication year, DOI, PMID, keywords, or article type.
- **Implication:** article-level structured metadata is currently limited. If title/date/MeSH are needed for filtering, they must be extracted from the abstract text or re-fetched from an external source.

### Candidate retrieval uses

- `abstract`: source for any NLP-extracted metadata such as dates, organisms, chemicals, P-values, study type phrases.

---

## 3. Graph Metadata

### Source files (under `data/graph/`)

| File | Columns | Purpose |
|------|---------|---------|
| `articles.csv` | `article_id`, `abstract` | Article master table |
| `chunks.csv` | `chunk_id`, `article_id`, `text`, `strategy`, `embedding` | Chunk master table with vectors |
| `entities.csv` | `entity_id`, `name`, `label` | Extracted named entities / concepts |
| `mentions.csv` | `chunk_id`, `entity_id` | Entity occurrences per chunk |
| `has_chunk.csv` | `article_id`, `chunk_id` | Article-to-chunk membership |

### Entity labels (`entities.csv`)

| Label | Count | Meaning | Candidate use |
|-------|-------|---------|-------------|
| `CONCEPT` | 119,813 | Domain concepts / key terms | Top candidate for entity-specific filtering and boosting |
| `CARDINAL` | 4,192 | Numbers / counts | Could support numeric comparison queries |
| `DATE` | 3,227 | Dates / years | Temporal filtering potential |
| `ORG` | 1,856 | Organizations | Institution / trial sponsor filtering |
| `PERSON` | 1,761 | People names | Author / investigator filtering (if names are authors) |
| `PERCENT` | 1,689 | Percentages | Numeric result filtering |
| `GPE` | 748 | Geopolitical entities | Geographic filtering |
| `QUANTITY` | 617 | Quantities with units | Dose / measurement filtering |
| `NORP` | 330 | Nationalities / religious / political groups | Population filtering |
| `TIME` | 194 | Time expressions | Temporal filtering |
| `PRODUCT` | 165 | Products / drugs / devices | Intervention / drug filtering |
| `MONEY` | 131 | Monetary values | Rare; unlikely useful |
| `LOC` | 74 | Locations | Geographic filtering |
| `ORDINAL` | 67 | Ordinals | Rare |
| `000"` | 58 | Data artifact / parsing error | Should be cleaned, not used |

### Graph edges

| Edge file | Columns | Notes |
|-----------|---------|-------|
| `mentions.csv` | `chunk_id`, `entity_id` | Many-to-many; allows entity-aware chunk retrieval |
| `has_chunk.csv` | `article_id`, `chunk_id` | Allows article-level result grouping |

### Candidate retrieval uses

- `entities.name`: exact or fuzzy entity match for entity-specific queries.
- `entities.label`: label-based boosting (e.g., boost `CONCEPT` matches, ignore `MONEY`).
- `mentions`: convert an entity query into a chunk filter (`WHERE chunk_id IN (SELECT chunk_id FROM mentions WHERE entity_id = ?)`).
- `has_chunk`: group results by `article_id` to reduce redundancy.

---

## 4. Missing / Extractable Metadata

The following common PubMed metadata fields are **not present as structured fields** in the current artifacts and would need to be extracted or sourced externally if required:

| Field | Extraction source | Feasibility |
|-------|-------------------|-------------|
| Publication year | Abstract text via regex (e.g., "between 2008 and 2009") | Medium |
| Study type | Abstract text phrases ("randomized", "case-control", "cohort") | Medium |
| Species / organism | Entity labels (`NORP`, `GPE`, `CONCEPT` names) | Medium |
| Intervention / drug | `CONCEPT` / `PRODUCT` entities | High |
| Outcome measure | `CONCEPT` entities + numeric labels | Medium |
| P-values / statistics | Abstract text regex | High |
| Author list | Not available | Low |
| Journal / source | Not available | Low |
| MeSH terms | Not available | Low |

---

## 5. Summary: Metadata Fields Useful for Filtering / Boosting

| Field | Source | Filtering | Boosting | Notes |
|-------|--------|-----------|----------|-------|
| `article_id` | chunks, PubMed records, graph | Yes (restrict to articles) | No | Join key |
| `chunk_id` prefix / sequence | chunks | Yes (position) | Yes (early chunks) | Parseable from ID |
| `strategy` | chunks | Yes | Yes | Semantic vs fixed vs sentence |
| `entities.name` | `entities.csv` | Yes (entity match) | Yes (entity relevance) | Requires join via `mentions.csv` |
| `entities.label` | `entities.csv` | Yes (label filter) | Yes (label weight) | `CONCEPT` is dominant |
| `mentions.chunk_id` | `mentions.csv` | Yes (chunk filter) | No | Bridge table |
| `has_chunk.chunk_id` | `has_chunk.csv` | Yes (dedup by article) | No | Bridge table |
| Abstract-derived dates | `pubmed_5000.jsonl.gz` | Yes (year range) | No | Requires regex extraction |
| Abstract-derived study type | `pubmed_5000.jsonl.gz` | Yes | No | Requires keyword matching |

---

## 6. Decisions Deferred to Phase 4 Implementation Steps

- Which entity labels to include/exclude in boosting.
- How to handle the `000"` entity-label artifact.
- Whether to extract and cache structured fields (year, study type, etc.) from abstracts.
- Whether to store metadata in the existing CSV files or introduce a lightweight metadata index.
- API surface for metadata filters in `RetrievalConfig` and the Streamlit UI.

No filtering or boosting logic has been implemented yet.
