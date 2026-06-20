#!/usr/bin/env python3
"""Architecture boundary checker.

Validates Clean Architecture dependency rules using AST parsing.

Rules:
- ``src/domain`` must not import ``src.application``, ``src.infrastructure``,
  ``src.interfaces``, or ``src.config``.
- ``src/application`` must not import ``src.interfaces``.
- Exceptions: ``__init__.py`` re-exports and imports from the same layer.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "src"


def get_imports(path: Path) -> list[str]:
    """Return all fully-qualified names imported by a Python module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports


def _layer_of(path: Path) -> str:
    """Return the Clean Architecture layer for a file under ``src/``."""
    rel = path.relative_to(ROOT)
    return rel.parts[0]


def check_domain() -> list[str]:
    """Verify domain layer does not depend on outer layers."""
    violations: list[str] = []
    forbidden_prefixes = (
        "src.application",
        "src.infrastructure",
        "src.interfaces",
        "src.config",
    )
    for path in sorted((ROOT / "domain").rglob("*.py")):
        for imp in get_imports(path):
            if imp.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(ROOT)} imports {imp}")
    return violations


def check_application() -> list[str]:
    """Verify application layer does not depend on interfaces."""
    violations: list[str] = []
    forbidden_prefix = "src.interfaces"
    for path in sorted((ROOT / "application").rglob("*.py")):
        for imp in get_imports(path):
            if imp.startswith(forbidden_prefix):
                violations.append(f"{path.relative_to(ROOT)} imports {imp}")
    return violations


def main() -> int:
    violations = check_domain() + check_application()
    if violations:
        print("Architecture violations detected:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print("Architecture check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
