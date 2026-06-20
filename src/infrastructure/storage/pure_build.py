"""Guard against side effects during pure pipeline construction."""

from __future__ import annotations

import builtins
import os
from contextlib import contextmanager
from typing import Any, Iterator

_pure_build_depth = 0
_original_open = builtins.open
_filesystem_access_count = 0


def is_pure_build_active() -> bool:
    return _pure_build_depth > 0


def assert_not_during_pure_build(operation: str) -> None:
    if _pure_build_depth > 0:
        raise RuntimeError(f"build_pipeline violated purity: {operation}")


def _guarded_open(*args: Any, **kwargs: Any) -> Any:
    if _pure_build_depth > 0:
        global _filesystem_access_count
        _filesystem_access_count += 1
        raise RuntimeError("build_pipeline violated purity: filesystem access during build")
    return _original_open(*args, **kwargs)


@contextmanager
def pure_build_guard() -> Iterator[None]:
    """Reject env mutation and filesystem access while building the pipeline."""
    global _pure_build_depth, _filesystem_access_count
    env_snapshot = dict(os.environ)
    _filesystem_access_count = 0
    _pure_build_depth += 1
    builtins.open = _guarded_open  # type: ignore[assignment]
    try:
        yield
    finally:
        _pure_build_depth -= 1
        builtins.open = _original_open

    if dict(os.environ) != env_snapshot:
        raise RuntimeError("build_pipeline violated purity: environment mutation")

    if _filesystem_access_count > 0:
        raise RuntimeError(
            "build_pipeline violated purity: no_filesystem_access_occurred_during_build"
        )
