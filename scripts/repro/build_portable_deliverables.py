#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build the standalone Keyboard-v3 source release and Drive transfer bundle.

The command never overwrites an existing destination. It copies only declared
inputs, emits per-file SHA-256 receipts, and keeps Isaac Sim/Lab, ROS2 runtime
trees, raw W&B output and credentials outside both deliverables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
KEYBOARD_SPEC = ROOT / "configs/reproducibility/keyboard_collection_release.json"
TRANSFER_SPEC = ROOT / "configs/reproducibility/g2_transfer_bundle.json"
SOURCE_MANIFEST = "SOURCE_MANIFEST.json"
BUNDLE_MANIFEST = "BUNDLE_MANIFEST.json"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SystemExit(f"UNSAFE_RELATIVE_PATH:{value}")
    return path


def file_rows(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude or "/.git/" in f"/{relative}/":
            continue
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def copy_file(relative: str, destination: Path) -> None:
    source = ROOT / safe_relative(relative)
    if not source.is_file():
        raise SystemExit(f"RELEASE_SOURCE_MISSING:{relative}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def dependency_closure(roots: list[str]) -> list[str]:
    command = [
        os.environ.get("GENIESIM_RELEASE_PYTHON", "python3"),
        str(ROOT / "scripts/repro/audit_dependency_closure.py"),
        *roots,
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    return [str(value) for value in payload["files"]]


def source_repository_receipt() -> dict[str, str]:
    """Bind a Drive bundle to the exact Git source that can consume it."""

    def query(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            raise SystemExit(f"GIT_SOURCE_RECEIPT_FAILED:{' '.join(arguments)}")
        return value

    return {
        "url": query("config", "--get", "remote.origin.url"),
        "branch": query("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": query("rev-parse", "HEAD"),
    }


KEYBOARD_README = """# G2 Keyboard-v3 collection (ROS2-free source release)

This repository contains the simulator-only G2 Keyboard-v3 collection and
validation path exported from Genie Sim. ROS2 workspaces, native teleoperation
binaries, datasets, checkpoints, Isaac Sim/Lab and credentials are excluded.

## Install

Install NVIDIA Isaac Sim 6.0.1 and Isaac Lab 6.1.14 on the target computer,
then install `requirements-minimal-training.txt` into that Isaac Python.

```bash
cp .env.example .env
${EDITOR:-nano} .env
set -a; source .env; set +a
python3 scripts/repro/keyboard_release_preflight.py --profile static
$GENIESIM_ISAAC_PYTHON scripts/repro/keyboard_release_preflight.py --profile live --collection
./scripts/static_test.sh
```

The G2 asset pack, Candidate-A USD and pregrasp source files are external and
must match `configs/reproducibility/external_assets.json`.

## Prepare and collect

```bash
$GENIESIM_ISAAC_PYTHON scripts/prepare_g2_keyboard_v3_collection.py \
  --output-root "$GENIESIM_DATA_ROOT/keyboard_v3"

./scripts/collect_data.sh --execute-live \
  --collection-root "$GENIESIM_DATA_ROOT/keyboard_v3" \
  --episode-id episode-000001 --target-residual-mm 20
```

## Validate

```bash
./scripts/validate_dataset.sh \
  --collection-root "$GENIESIM_DATA_ROOT/keyboard_v3" \
  --output "$GENIESIM_OUTPUT_ROOT/keyboard_v3_validation.json"
```

Collection is simulation-only. Student privileged input remains zero. The
latest G2 URDF/YAML/action-scale authority is included; mesh/USD/data payloads
remain external and hash-verified. See `SOURCE_MANIFEST.json` for provenance.
"""


def export_keyboard(args: argparse.Namespace) -> int:
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise SystemExit(f"OUTPUT_REFUSES_OVERWRITE:{destination}")
    destination.mkdir(parents=True)
    spec = load_object(KEYBOARD_SPEC)
    files = set(str(value) for value in spec["explicit_files"])
    files.update(dependency_closure([str(value) for value in spec["dependency_roots"]]))
    files.add("scripts/repro/audit_dependency_closure.py")
    for relative in sorted(files):
        components = set(Path(relative).parts)
        if components.intersection(set(spec["excluded_path_components"])):
            raise SystemExit(f"FORBIDDEN_RELEASE_COMPONENT:{relative}")
        copy_file(relative, destination)

    (destination / "README.md").write_text(KEYBOARD_README, encoding="utf-8")
    (destination / ".gitignore").write_text(
        ".env\n__pycache__/\n*.pyc\n*.h5\n*.hdf5\n*.pt\n*.pth\n*.ckpt\n"
        "output/\noutputs/\ndatasets/\nartifacts/\nwandb/\nbuild/\ninstall/\nlog/\n",
        encoding="utf-8",
    )
    preflight = destination / "scripts/preflight.sh"
    preflight.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
        "[[ ! -f \"${root}/.env\" ]] || { set -a; source \"${root}/.env\"; set +a; }\n"
        "exec \"${GENIESIM_ISAAC_PYTHON:-python3}\" \"${root}/scripts/repro/keyboard_release_preflight.py\" \"$@\"\n",
        encoding="utf-8",
    )
    preflight.chmod(0o755)
    validator = destination / "scripts/validate_dataset.sh"
    validator.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
        "[[ ! -f \"${root}/.env\" ]] || { set -a; source \"${root}/.env\"; set +a; }\n"
        "exec \"${GENIESIM_ISAAC_PYTHON:-python3}\" \"${root}/scripts/validate_g2_keyboard_v3_dataset.py\" \"$@\"\n",
        encoding="utf-8",
    )
    validator.chmod(0o755)
    static_test = destination / "scripts/static_test.sh"
    static_test.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
        "python_bin=\"${GENIESIM_ISAAC_PYTHON:-python3}\"\n"
        "export PYTHONPATH=\"${root}/source${PYTHONPATH:+:${PYTHONPATH}}\"\n"
        "\"${python_bin}\" \"${root}/scripts/repro/keyboard_release_preflight.py\" --profile static\n"
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \"${python_bin}\" -m pytest -q \"${root}/tests/test_g2_keyboard_v3_dataset.py\"\n",
        encoding="utf-8",
    )
    static_test.chmod(0o755)

    secret_patterns = (
        re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(rb"\bghp_[A-Za-z0-9]{20,}"),
        re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}"),
        re.compile(rb"\bWANDB_API_KEY\s*=\s*[^\s#]+"),
    )
    secret_hits: list[str] = []
    local_path_hits: list[str] = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        content = path.read_bytes()
        relative = path.relative_to(destination).as_posix()
        if any(pattern.search(content) for pattern in secret_patterns):
            secret_hits.append(relative)
        if re.search(rb"/(?:home|data)/fain(?:/|\b)", content):
            local_path_hits.append(relative)
    if secret_hits:
        raise SystemExit(f"SECRET_SCAN_FAILED:{','.join(secret_hits)}")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    rows = file_rows(destination, exclude={SOURCE_MANIFEST})
    manifest = {
        "schema": "g2_keyboard_source_manifest_v1",
        "source_repository_commit": commit,
        "source_worktree_dirty": bool(
            subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        ),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "ros2_included": False,
        "student_privileged_input_count": 0,
        "historical_machine_path_reference_files": local_path_hits,
        "files": rows,
    }
    (destination / SOURCE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"KEYBOARD_SOURCE_EXPORT": "PASS", **{k: manifest[k] for k in ("file_count", "total_size_bytes", "ros2_included", "historical_machine_path_reference_files")}}, indent=2, sort_keys=True))
    return 0


def _copy_entry(source: Path, target: Path, kind: str) -> None:
    if kind == "file":
        if not source.is_file():
            raise SystemExit(f"TRANSFER_FILE_MISSING:{source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    elif kind == "directory":
        if not source.is_dir():
            raise SystemExit(f"TRANSFER_DIRECTORY_MISSING:{source}")
        shutil.copytree(source, target)
    else:
        raise SystemExit(f"TRANSFER_KIND_INVALID:{kind}")


def export_transfer(args: argparse.Namespace) -> int:
    destination = args.output.expanduser().resolve()
    archive = destination.with_suffix(".tar.gz")
    if destination.exists() or archive.exists():
        raise SystemExit(f"OUTPUT_REFUSES_OVERWRITE:{destination}")
    destination.mkdir(parents=True)
    spec = load_object(TRANSFER_SPEC)
    source_repository = source_repository_receipt()
    included: list[dict[str, Any]] = []
    missing_optional: list[str] = []
    for entry in spec["entries"]:
        source_variable = str(entry.get("source_environment", ""))
        source_value = os.environ.get(source_variable, "") if source_variable else str(entry.get("source", ""))
        source = Path(source_value).expanduser().resolve() if source_value else Path("/__missing_transfer_source__")
        if not source.exists():
            if entry.get("required"):
                raise SystemExit(f"REQUIRED_TRANSFER_INPUT_MISSING:{entry['id']}:{source}")
            missing_optional.append(str(entry["id"]))
            continue
        expected = entry.get("expected_sha256")
        if expected and (not source.is_file() or sha256(source) != expected):
            raise SystemExit(f"TRANSFER_HASH_MISMATCH:{entry['id']}")
        target = destination / safe_relative(str(entry["target"]))
        _copy_entry(source, target, str(entry["kind"]))
        included.append({key: entry[key] for key in ("id", "target", "required", "roles")})

    readme = destination / "README_UPLOAD.md"
    readme.write_text(
        "# G2 Stage-1A data/model transfer bundle\n\n"
        "Upload the sibling `.tar.gz` and `.sha256` files to Google Drive. "
        "On the target machine, verify SHA-256 before extracting. This bundle "
        "contains authorized local-transfer artifacts and must not be committed to Git.\n\n"
        f"Source: `{source_repository['url']}` branch `{source_repository['branch']}` "
        f"commit `{source_repository['commit']}`.\n\n"
        "```bash\nsha256sum -c g2-stage1a-data-models.tar.gz.sha256\n"
        "tar -xzf g2-stage1a-data-models.tar.gz\n"
        "python3 scripts/repro/minimal_training_bundle.py verify --bundle <extracted>/runtime/minimal_bundle\n```\n",
        encoding="utf-8",
    )
    rows = file_rows(destination, exclude={BUNDLE_MANIFEST, "SHA256SUMS"})
    manifest = {
        "schema": "g2_stage1a_drive_bundle_manifest_v1",
        "bundle_name": spec["bundle_name"],
        "distribution_scope": spec["distribution_scope"],
        "source_repository": source_repository,
        "included_entries": included,
        "missing_optional_entries": missing_optional,
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "files": rows,
    }
    (destination / BUNDLE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_rows = file_rows(destination, exclude={"SHA256SUMS"})
    (destination / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in all_rows), encoding="utf-8"
    )
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(destination, arcname=destination.name)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"TRANSFER_BUNDLE": "PASS", "directory": str(destination), "archive": str(archive), "archive_sha256": sha256(archive), "file_count": manifest["file_count"], "total_size_bytes": manifest["total_size_bytes"], "missing_optional_entries": missing_optional}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    keyboard = sub.add_parser("keyboard-source")
    keyboard.add_argument("--output", type=Path, required=True)
    keyboard.set_defaults(handler=export_keyboard)
    transfer = sub.add_parser("transfer-bundle")
    transfer.add_argument("--output", type=Path, required=True)
    transfer.set_defaults(handler=export_transfer)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
