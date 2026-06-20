"""Infrastructure CSV loading helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    """Load a CSV file and validate its header."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header")
        missing = set(expected_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        rows = [dict(row) for row in reader]
    return rows
