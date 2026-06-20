"""Filesystem write safety for Streamlit and deployment runtimes."""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_MOUNT_SRC_PREFIX = "/mount/src/"
_DEFAULT_HF_HOME = "/tmp/hf_cache"
_DEFAULT_TRANSFORMERS_CACHE = "/tmp/hf_cache/transformers"
_DEFAULT_TORCH_HOME = "/tmp/torch_cache"


@lru_cache(maxsize=1)
def detect_repo_root() -> Path:
    """Detect repository root via git, falling back to package layout or cwd."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        root = Path(result.stdout.strip()).resolve()
        if root.is_dir():
            return root
    except (OSError, subprocess.SubprocessError):
        pass

    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "src").is_dir():
        return package_root

    return Path.cwd().resolve()


def assert_no_repo_write(path: str) -> None:
    """Raise if ``path`` would write inside the watched repository tree."""
    resolved = str(Path(path).resolve())

    if resolved.startswith(_MOUNT_SRC_PREFIX):
        raise RuntimeError(f"Illegal write to repo detected: {resolved}")

    repo = str(detect_repo_root().resolve())
    if resolved == repo or resolved.startswith(f"{repo}{os.sep}"):
        raise RuntimeError(f"Illegal write to repo detected: {resolved}")


def safe_mkdir(path: str | Path) -> None:
    """Create a directory only when the target is outside the repository."""
    assert_no_repo_write(str(path))
    os.makedirs(path, exist_ok=True)


@contextmanager
def safe_write_file(path: str | Path, mode: str = "wb") -> Iterator[object]:
    """Open a file for writing after enforcing repository write protection."""
    assert_no_repo_write(str(path))
    with open(path, mode) as handle:
        yield handle


def configure_external_model_caches() -> None:
    """Redirect HuggingFace, Transformers, and Torch caches outside the repo."""
    os.environ["HF_HOME"] = _DEFAULT_HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = _DEFAULT_TRANSFORMERS_CACHE
    os.environ["TORCH_HOME"] = _DEFAULT_TORCH_HOME
    os.environ["HF_DATASETS_CACHE"] = str(Path(_DEFAULT_HF_HOME) / "datasets")
    os.environ["HF_HUB_CACHE"] = str(Path(_DEFAULT_HF_HOME) / "hub")

    safe_mkdir(_DEFAULT_HF_HOME)
    safe_mkdir(_DEFAULT_TRANSFORMERS_CACHE)
    safe_mkdir(_DEFAULT_TORCH_HOME)
    safe_mkdir(os.environ["HF_DATASETS_CACHE"])
    safe_mkdir(os.environ["HF_HUB_CACHE"])


def log_startup_diagnostics() -> None:
    """Log cache locations and write-protection status at process startup."""
    from src.infrastructure.storage.artifact_loader import get_cache_dir

    cache_dir = get_cache_dir()
    repo_root = detect_repo_root()

    lines = (
        f"CACHE_DIR={cache_dir}",
        f"HF_HOME={os.environ.get('HF_HOME', '')}",
        f"TORCH_HOME={os.environ.get('TORCH_HOME', '')}",
        f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE', '')}",
        f"REPO_ROOT={repo_root}",
        "WRITE PROTECTION ENABLED",
    )
    for line in lines:
        logger.info(line)
        print(line, flush=True)


def verify_no_repo_writes(paths: list[str]) -> None:
    """Fail fast when resolved artifact paths point at the Streamlit repo mount."""
    for path in paths:
        resolved = str(Path(path).resolve())
        if _MOUNT_SRC_PREFIX in resolved:
            raise RuntimeError(
                f"Streamlit restart risk detected: repo write path {resolved}"
            )
