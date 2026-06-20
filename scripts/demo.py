#!/usr/bin/env python3
"""Launcher for the Streamlit demo interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.environment import configure_environment

configure_environment()

CACHE_DIR = os.environ.get("ARTIFACT_CACHE_DIR", "").strip() or "/tmp/pubmed-graphrag"

from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts

bootstrap_artifacts(CACHE_DIR)

from src.interfaces.streamlit.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
