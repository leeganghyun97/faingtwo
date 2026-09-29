#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Report repository-local Python imports for canonical Stage-1A roots."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"


def module_path(name: str) -> Path | None:
    candidate = SOURCE / Path(*name.split("."))
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path.resolve()
    return None


def module_name(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(SOURCE)
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_initializers(path: Path) -> set[Path]:
    """Return import-time ``__init__`` files for a resolved source module."""

    result: set[Path] = set()
    parent = path.resolve().parent
    while parent != SOURCE.parent:
        try:
            parent.relative_to(SOURCE)
        except ValueError:
            break
        initializer = parent / "__init__.py"
        if initializer.is_file():
            result.add(initializer.resolve())
        if parent == SOURCE:
            break
        parent = parent.parent
    return result


def imports(path: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    current = module_name(path)
    current_package = None
    if current:
        current_package = (
            current.split(".")
            if path.name == "__init__.py"
            else current.split(".")[:-1]
        )
    result: set[Path] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and current_package is not None:
                prefix = current_package[: max(0, len(current_package) - node.level + 1)]
                base = ".".join([*prefix, base] if base else prefix)
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names if base)
        for name in names:
            if not name.startswith("geniesim"):
                continue
            resolved = module_path(name)
            if resolved:
                result.add(resolved)
                result.update(package_initializers(resolved))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    pending = [(ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in args.roots]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        pending.extend(sorted(imports(path) - seen))
    relative = sorted(str(path.relative_to(ROOT)) for path in seen)
    print(json.dumps({"schema": "geniesim_dependency_closure_v1", "file_count": len(relative), "files": relative}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
