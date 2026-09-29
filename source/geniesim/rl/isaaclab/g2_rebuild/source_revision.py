# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exact source-tree identity for controlled-rebuild production artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


_NON_IDENTIFYING = frozenset({"", "unknown", "none", "null", "unset", "unspecified"})


def require_concrete_source_revision(value: str, *, prefix: str) -> str:
    """Return a normalized explicit revision or fail closed.

    The revision need not be a Git commit: a byte-manifest revision is valid
    for an uncommitted controlled rebuild.  Placeholder values are never valid
    production lineage.
    """

    if not isinstance(value, str) or value.strip().lower() in _NON_IDENTIFYING:
        raise ValueError(f"{prefix}_SOURCE_REVISION_NOT_IDENTIFYING")
    return value.strip()


def _controlled_files(repository_root: Path) -> tuple[Path, ...]:
    roots = (
        repository_root / "source/geniesim/rl/isaaclab/g2_rebuild",
        repository_root / "configs/g2_rebuild",
    )
    files = {
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    # The rebuild intentionally reuses the existing Isaac G2 environment,
    # camera, collision, quaternion and task authorities.  Those modules are
    # transitive runtime dependencies, not external libraries, so omitting
    # them would allow physics/control semantics to change without changing a
    # checkpoint's declared source revision.
    isaaclab_root = repository_root / "source/geniesim/rl/isaaclab"
    if isaaclab_root.is_dir():
        files.update(
            path
            for path in isaaclab_root.glob("g2_*.py")
            if path.is_file()
        )
    scripts = repository_root / "scripts"
    if scripts.is_dir():
        files.update(
            path
            for path in scripts.rglob("*")
            if path.is_file()
            and "g2" in path.name.lower()
            and path.suffix in {".py", ".sh"}
        )
    return tuple(sorted(files))


def controlled_rebuild_source_revision(repository_root: Path) -> str:
    """Bind a revision to the Git base and exact controlled source bytes."""

    root = Path(repository_root).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    git_revision = result.stdout.strip() if result.returncode == 0 else "nogit"
    digest = hashlib.sha256()
    files = _controlled_files(root)
    if not files:
        raise RuntimeError("G2_CONTROLLED_SOURCE_MANIFEST_EMPTY")
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"{git_revision}+g2rebuild:{digest.hexdigest()}"


__all__ = [
    "controlled_rebuild_source_revision",
    "require_concrete_source_revision",
]
