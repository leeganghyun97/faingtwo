#!/usr/bin/env python3
"""Create HANDOFF_MANIFEST.json for the documentation-only handoff package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDE = {"HANDOFF_MANIFEST.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDE:
            continue
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "package": "x2_cerf_locomotion_handoff_20261124",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "documentation-only; excludes checkpoints, raw traces, simulator assets, binaries, credentials, and private links",
        "source_snapshot": {
            "repository": "x2_locomotion_migration_20261124",
            "commit": "0dee161c976e29f4d9caa5770cb1d261c617ed8b",
        },
        "files": files,
    }
    (ROOT / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE_MANIFEST files={len(files)}")


if __name__ == "__main__":
    main()
