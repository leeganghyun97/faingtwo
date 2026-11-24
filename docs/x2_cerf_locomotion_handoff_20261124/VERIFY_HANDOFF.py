#!/usr/bin/env python3
"""Verify the small documentation-only X2 CReF handoff package.

This verifies only the manifest-listed documentation. It does not assert that
Isaac, an asset, checkpoint, or an original locomotion run is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "HANDOFF_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"MISSING_MANIFEST:{MANIFEST}")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    listed = {entry["path"] for entry in manifest.get("files", [])}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "HANDOFF_MANIFEST.json"
    }
    missing_from_manifest = sorted(actual - listed)
    missing_from_tree = sorted(listed - actual)
    for rel in missing_from_manifest:
        errors.append(f"UNLISTED_FILE:{rel}")
    for rel in missing_from_tree:
        errors.append(f"MISSING_MANIFEST_FILE:{rel}")
    for entry in manifest.get("files", []):
        path = ROOT / entry["path"]
        if not path.is_file():
            errors.append(f"MISSING:{entry['path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != entry["bytes"]:
            errors.append(f"SIZE_MISMATCH:{entry['path']} expected={entry['bytes']} actual={actual_size}")
        if actual_hash != entry["sha256"]:
            errors.append(f"SHA256_MISMATCH:{entry['path']} expected={entry['sha256']} actual={actual_hash}")
    if errors:
        print("HANDOFF_VERIFICATION_FAILED")
        print("\n".join(errors))
        return 1
    print("HANDOFF_VERIFICATION_PASS")
    print(f"files={len(manifest.get('files', []))}")
    print("scope=documentation-only; no simulator, checkpoint, asset, or raw-trace reproducibility is asserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
