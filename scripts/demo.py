#!/usr/bin/env python3
"""Launcher for the Streamlit demo interface."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.environment import configure_environment

configure_environment()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

CACHE_DIR = os.environ.get("ARTIFACT_CACHE_DIR", "").strip() or "/tmp/pubmed-graphrag"

from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts

try:
    bootstrap_status = bootstrap_artifacts(CACHE_DIR)
except RuntimeError as exc:
    print(f"ARTIFACT BOOTSTRAP FAILED: {exc}", flush=True)
    raise SystemExit(1) from exc

print(f"ARTIFACT BOOTSTRAP STATUS: {json.dumps(bootstrap_status)}", flush=True)

from src.interfaces.streamlit.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
