"""Guard against side effects during pure pipeline construction."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_pure_build_depth = 0


def is_pure_build_active() -> bool:
    return _pure_build_depth > 0


def assert_not_during_pure_build(operation: str) -> None:
    if _pure_build_depth > 0:
        raise RuntimeError(f"build_pipeline violated purity: {operation}")


@contextmanager
def pure_build_guard() -> Iterator[None]:
    """Reject env mutation and tracked side effects while building the pipeline."""
    global _pure_build_depth
    env_snapshot = dict(os.environ)
    _pure_build_depth += 1
    try:
        yield
    finally:
        _pure_build_depth -= 1

    if dict(os.environ) != env_snapshot:
        raise RuntimeError("build_pipeline violated purity: environment mutation")
