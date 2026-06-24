"""Tests for the pure-build guard's framework exemption."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from src.infrastructure.storage.pure_build import pure_build_guard


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_application_open_is_blocked_during_pure_build(tmp_path: Path) -> None:
    """Project code reading a file during a pure build must still raise."""
    path = tmp_path / "secret.txt"
    path.write_text("secret")

    with pytest.raises(RuntimeError, match="build_pipeline violated purity"):
        with pure_build_guard():
            builtins.open(path)


def test_allowed_opens_do_not_count_as_violations(tmp_path: Path) -> None:
    """Opening a file outside a pure build is unaffected."""
    path = tmp_path / "plain.txt"
    path.write_text("hello")
    with builtins.open(path) as f:
        assert f.read() == "hello"


def test_site_packages_open_is_exempt(tmp_path: Path) -> None:
    """A caller that reports a site-packages frame must be allowed.

    This simulates Starlette/anyio static file serving by faking a frame whose
    filename lives under site-packages.
    """
    path = tmp_path / "asset.css"
    path.write_text("body {}")

    # Patch inspect.stack so _guarded_open sees a site-packages frame above us.
    fake_frame = type(
        "FrameInfo",
        (),
        {
            "filename": "/venv/lib/python3.10/site-packages/starlette/staticfiles.py",
            "function": "get_response",
        },
    )

    import src.infrastructure.storage.pure_build as pure_build_mod

    original_stack = pure_build_mod.inspect.stack
    pure_build_mod.inspect.stack = lambda context=1: [fake_frame]
    try:
        with pure_build_guard():
            with builtins.open(path) as f:
                assert f.read() == "body {}"
    finally:
        pure_build_mod.inspect.stack = original_stack
