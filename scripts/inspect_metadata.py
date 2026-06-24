#!/usr/bin/env python3
"""Dump metadata samples from chunks, graph, and original PubMed records."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sample_gz_jsonl(path: Path, n: int = 3) -> list[dict]:
    records: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


def _print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    # 1. Semantic chunks
    _print_section("SEMANTIC CHUNK SAMPLE (data/chunks/chunks_semantic.jsonl.gz)")
    for rec in _sample_gz_jsonl(ROOT / "data" / "chunks" / "chunks_semantic.jsonl.gz", n=2):
        print(json.dumps(rec, indent=2, ensure_ascii=False))

    # 2. Original PubMed article sample
    _print_section("ORIGINAL PUBMED ARTICLE SAMPLE (data/pubmed_5000.jsonl.gz)")
    for rec in _sample_gz_jsonl(ROOT / "data" / "pubmed_5000.jsonl.gz", n=2):
        print(json.dumps(rec, indent=2, ensure_ascii=False))

    # 3. Graph entity label distribution
    _print_section("GRAPH ENTITY LABEL DISTRIBUTION (data/graph/entities.csv)")
    labels: Counter = Counter()
    with open(ROOT / "data" / "graph" / "entities.csv", "r", encoding="utf-8") as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                labels[parts[2]] += 1
    for label, count in labels.most_common(15):
        print(f"  {label}: {count}")

    # 4. Chunks.csv columns/fields
    _print_section("CHUNKS.CSV HEADER (data/graph/chunks.csv)")
    with open(ROOT / "data" / "graph" / "chunks.csv", "r", encoding="utf-8") as fh:
        header = next(fh).strip()
        print(header)
        for i, line in enumerate(fh):
            if i >= 2:
                break
            print(line.strip())

    # 5. Fixed chunks sample (alternative chunking)
    _print_section("FIXED CHUNK SAMPLE (data/chunks/chunks_fixed.jsonl.gz)")
    for rec in _sample_gz_jsonl(ROOT / "data" / "chunks" / "chunks_fixed.jsonl.gz", n=1):
        print(json.dumps(rec, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
