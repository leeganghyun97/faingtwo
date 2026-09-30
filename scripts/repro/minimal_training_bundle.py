#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Export, install, or verify the local-only Stage-1A training bundle.

The bundle deliberately excludes Isaac Sim/Lab.  It contains only the exact
Candidate-A layers, source-state evidence and learned initialization artifacts
required by the bounded 10-env/3K bootstrap route.  It is a local transfer
artifact and must not be committed or redistributed without asset permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/reproducibility/minimal_training_bundle.json"
BASELINE = ROOT / "artifacts/g2_candidate_a_left_arm_down_v2/BASELINE_MANIFEST.json"
BUNDLE_MANIFEST = "MINIMAL_TRAINING_BUNDLE_MANIFEST.json"
BUNDLE_SCHEMA = "geniesim_minimal_training_bundle_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SystemExit(f"UNSAFE_BUNDLE_PATH:{value}")
    return path


def _argument_path(args: argparse.Namespace, name: str) -> Path | None:
    value = getattr(args, name, None)
    return None if value is None else Path(value).expanduser().resolve()


def _resolve_source(entry: dict[str, Any], args: argparse.Namespace) -> Path:
    argument = entry.get("argument")
    if argument:
        supplied = _argument_path(args, str(argument))
        if supplied is not None:
            return supplied
    environment = entry.get("source_environment")
    if environment and os.environ.get(str(environment), "").strip():
        return Path(os.environ[str(environment)]).expanduser().resolve()
    source = entry.get("source")
    if source:
        return (ROOT / _safe_relative(str(source))).resolve()
    field = entry.get("manifest_field")
    if field:
        sample = _load_object(BASELINE).get("pregrasp_initial_state", {})
        if isinstance(sample, dict) and sample.get(str(field)):
            return Path(str(sample[str(field)])).expanduser().resolve()
    label = str(argument or entry.get("target", "unknown"))
    raise SystemExit(f"BUNDLE_SOURCE_UNRESOLVED:{label}")


def _copy_entry(source: Path, destination: Path, kind: str) -> None:
    if kind == "file":
        if not source.is_file():
            raise SystemExit(f"BUNDLE_SOURCE_FILE_MISSING:{source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    if kind == "directory":
        if not source.is_dir():
            raise SystemExit(f"BUNDLE_SOURCE_DIRECTORY_MISSING:{source}")
        shutil.copytree(source, destination)
        return
    raise SystemExit(f"BUNDLE_ENTRY_TYPE_INVALID:{kind}")


def _file_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == BUNDLE_MANIFEST:
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def export_bundle(args: argparse.Namespace) -> int:
    destination = args.bundle.resolve()
    if destination.exists():
        raise SystemExit(f"BUNDLE_OUTPUT_REFUSES_OVERWRITE:{destination}")
    destination.mkdir(parents=True)
    spec = _load_object(SPEC)
    for entry in spec["entries"]:
        source = _resolve_source(entry, args)
        expected = entry.get("expected_sha256")
        if expected is not None and (
            not source.is_file() or _sha256(source) != str(expected)
        ):
            raise SystemExit(f"BUNDLE_SOURCE_HASH_MISMATCH:{source}")
        target = destination / _safe_relative(str(entry["target"]))
        _copy_entry(source, target, str(entry["type"]))
    files = _file_rows(destination)
    payload = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": spec["bundle_id"],
        "distribution_scope": "LOCAL_AUTHORIZED_TRANSFER_ONLY",
        "isaac_included": False,
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
    }
    (destination / BUNDLE_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _verified_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / BUNDLE_MANIFEST
    payload = _load_object(manifest_path)
    if payload.get("schema") != BUNDLE_SCHEMA or not isinstance(payload.get("files"), list):
        raise SystemExit("BUNDLE_MANIFEST_INVALID")
    seen: set[str] = set()
    for row in payload["files"]:
        relative = _safe_relative(str(row.get("path", "")))
        if relative.as_posix() in seen:
            raise SystemExit("BUNDLE_MANIFEST_DUPLICATE_PATH")
        seen.add(relative.as_posix())
        path = bundle / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256(path) != row.get("sha256")
        ):
            raise SystemExit(f"BUNDLE_FILE_HASH_MISMATCH:{relative}")
    return payload


def _install_files(bundle: Path, payload: dict[str, Any]) -> None:
    for row in payload["files"]:
        relative = _safe_relative(str(row["path"]))
        source = bundle / relative
        destination = ROOT / relative
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != row["sha256"]:
                raise SystemExit(f"INSTALL_TARGET_CONFLICT:{destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_environment(*, isaac_python: Path, output_root: Path) -> None:
    if not isaac_python.is_file():
        raise SystemExit(f"ISAAC_PYTHON_MISSING:{isaac_python}")
    environment_path = ROOT / ".env"
    if environment_path.exists():
        raise SystemExit("ENV_REFUSES_OVERWRITE:.env")
    values = {
        "GENIESIM_REPO_ROOT": ROOT,
        "GENIESIM_DATA_ROOT": ROOT / "datasets",
        "GENIESIM_OUTPUT_ROOT": output_root,
        "GENIESIM_ISAAC_PYTHON": isaac_python,
        "GENIESIM_ASSET_ROOT": ROOT / "source/geniesim/assets",
        "GENIESIM_CANDIDATE_A_USD": ROOT
        / "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/"
        "A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda",
        "GENIESIM_GRU_CHECKPOINT": ROOT / "artifacts/external/BEST_HUMAN_GRASP_GRU_BC.pt",
        "GENIESIM_FAR_REACH_BC_CHECKPOINT": ROOT / "artifacts/external/far_reach_bc_best.pt",
        "GENIESIM_RESIDUAL_ACTOR_CHECKPOINT": ROOT / "artifacts/external/stage1a_residual_actor.pt",
        "GENIESIM_PREGRASP_HDF5": ROOT / "artifacts/external/stage1a_pregrasp_initial_states.hdf5",
        "GENIESIM_PREGRASP_REPORT": ROOT / "artifacts/external/stage1a_pregrasp_report.json",
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "geniesim-stage1a-minimal",
        "GENIESIM_ALLOW_REAL_ROBOT": "0",
    }
    rendered = ["# Generated by minimal_training_bundle.py; do not commit."]
    rendered.extend(f"{key}={value}" for key, value in values.items())
    environment_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def install_bundle(args: argparse.Namespace) -> int:
    bundle = args.bundle.resolve()
    payload = _verified_manifest(bundle)
    _install_files(bundle, payload)
    if args.write_env:
        if args.isaac_python is None:
            raise SystemExit("INSTALL_WRITE_ENV_REQUIRES_ISAAC_PYTHON")
        output_root = (
            args.output_root.expanduser().resolve()
            if args.output_root is not None
            else ROOT / "output"
        )
        _write_environment(
            isaac_python=args.isaac_python.expanduser().resolve(),
            output_root=output_root,
        )
    print(
        json.dumps(
            {
                "BUNDLE_INSTALL": "PASS",
                "file_count": payload["file_count"],
                "environment_written": bool(args.write_env),
                "next": "./scripts/preflight.sh --profile live --method A",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def verify_bundle(args: argparse.Namespace) -> int:
    payload = _verified_manifest(args.bundle.resolve())
    if not args.installed:
        print(
            json.dumps(
                {
                    "BUNDLE_VERIFY": "PASS",
                    "bundle_file_count": int(payload["file_count"]),
                    "total_size_bytes": int(payload["total_size_bytes"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    installed = []
    for row in payload["files"]:
        path = ROOT / _safe_relative(str(row["path"]))
        installed.append(path.is_file() and _sha256(path) == row["sha256"])
    result = {
        "BUNDLE_VERIFY": "PASS" if all(installed) else "FAIL",
        "bundle_file_count": len(installed),
        "installed_match_count": sum(installed),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(installed) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--far-reach-checkpoint", type=Path)
    export.add_argument("--pregrasp-hdf5", type=Path)
    export.add_argument("--pregrasp-report", type=Path)
    export.set_defaults(handler=export_bundle)
    install = subparsers.add_parser("install")
    install.add_argument("--bundle", type=Path, required=True)
    install.add_argument("--write-env", action="store_true")
    install.add_argument("--isaac-python", type=Path)
    install.add_argument("--output-root", type=Path)
    install.set_defaults(handler=install_bundle)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument(
        "--installed",
        action="store_true",
        help="also require every bundle file to be materialized in this clone",
    )
    verify.set_defaults(handler=verify_bundle)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
