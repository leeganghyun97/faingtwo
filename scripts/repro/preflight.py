#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Static/live reproducibility preflight without starting Isaac or a robot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
METHODS = ROOT / "configs/reproducibility/methods_a_to_g.json"
ASSETS = ROOT / "configs/reproducibility/external_assets.json"
FIXTURE = ROOT / "tests/fixtures/reproducibility/sample_dataset/manifest.json"

CANONICAL_FILES = (
    METHODS,
    ASSETS,
    ROOT / "scripts/repro/run_method.py",
    ROOT / "scripts/preflight.sh",
    ROOT / "scripts/smoke_test.sh",
    ROOT / "scripts/collect_data.sh",
    ROOT / "scripts/validate_dataset.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=False, capture_output=True
    )
    if result.returncode != 0:
        return []
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def static_checks() -> list[dict]:
    receipts: list[dict] = []
    for path in CANONICAL_FILES:
        receipts.append({"check": f"file:{path.relative_to(ROOT)}", "pass": path.is_file()})

    try:
        methods = load_json(METHODS)["methods"]
        method_ok = sorted(methods) == list("ABCDEFG")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        method_ok = False
    receipts.append({"check": "method_A_to_G_config", "pass": method_ok})

    try:
        fixture = load_json(FIXTURE)
        fixture_ok = fixture.get("synthetic") is True and fixture.get("frame") == "robot_root"
    except (OSError, ValueError, json.JSONDecodeError):
        fixture_ok = False
    receipts.append({"check": "synthetic_fixture", "pass": fixture_ok})

    absolute_hits: list[str] = []
    for path in CANONICAL_FILES:
        if path.is_file() and re.search(r"/(?:home|data)/fain(?:/|\b)", path.read_text(errors="replace")):
            absolute_hits.append(str(path.relative_to(ROOT)))
    receipts.append({"check": "canonical_user_absolute_paths", "pass": not absolute_hits, "paths": absolute_hits})

    secret_hits: list[str] = []
    forbidden_tracked: list[str] = []
    secret_patterns = (
        re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(rb"\bghp_[A-Za-z0-9]{20,}"),
        re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}"),
        re.compile(rb"\b(?:WANDB|OPENAI)_API_KEY\s*=\s*[^\s#'\"]+"),
    )
    forbidden_roots = {"artifacts", "wandb", "output", "outputs", "datasets", "data"}
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if relative.name == ".env" or (relative.parts and relative.parts[0] in forbidden_roots):
            forbidden_tracked.append(str(relative))
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if any(pattern.search(content) for pattern in secret_patterns):
            secret_hits.append(str(relative))
    receipts.append({"check": "tracked_secret_scan", "pass": not secret_hits, "paths": secret_hits})
    receipts.append({"check": "tracked_runtime_data", "pass": not forbidden_tracked, "paths": forbidden_tracked})

    broken: list[str] = []
    for path in tracked_files():
        if path.is_symlink() and not path.exists():
            broken.append(str(path.relative_to(ROOT)))
    # These upstream links depend on the optional teleop/native-SDK install and
    # are outside the Stage-1A canonical closure. Keep them visible instead of
    # pretending they resolve in a source-only clone.
    documented_external_links = {
        "source/geniesim/teleop/tasks",
        "source/teleop/app/share/genie_robot_description/urdf/G2/model.srdf",
        "source/teleop/app/share/genie_robot_description/urdf/G2/model.urdf",
        "source/teleop/app/vendors/lib/libpinocchio_casadi.so",
    }
    unexpected = sorted(set(broken) - documented_external_links)
    receipts.append({
        "check": "tracked_broken_symlinks",
        "pass": not unexpected,
        "unexpected_paths": unexpected,
        "documented_external_paths": sorted(set(broken) & documented_external_links),
        "classification": "PASS" if not broken else "PASS_WITH_EXTERNAL_TELEOP_EXCLUSIONS",
    })
    return receipts


def live_checks(method: str | None) -> list[dict]:
    receipts: list[dict] = []
    python = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", ""))
    receipts.append({"check": "isaac_python", "pass": python.is_file(), "path": str(python)})
    if python.is_file():
        probe = subprocess.run(
            [str(python), "-c", "import importlib.metadata as m; print(m.version('isaacsim')); print(m.version('isaaclab'))"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        receipts.append({"check": "isaac_packages", "pass": probe.returncode == 0, "versions": probe.stdout.splitlines()})

    manifest = load_json(ASSETS)
    required_for = {"all_live_methods", "isaac_scene"}
    if method:
        required_for.add(f"method_{method}")
    for item in manifest["assets"]:
        if not required_for.intersection(item["required_for"]):
            continue
        configured = os.environ.get(item["environment_variable"], "")
        path = Path(configured).expanduser() if configured else ROOT / item["legacy_relative_path"]
        exists = path.is_file() if item["sha256"] else path.is_dir()
        match = exists and (item["sha256"] is None or sha256(path) == item["sha256"])
        receipts.append({"check": f"asset:{item['id']}", "pass": bool(match), "path": str(path), "hash_checked": item["sha256"] is not None})

    # The source-freeze authority requires the artifact bundle in its original
    # repository-relative layout, not only the top-level USD.
    bundle_receipts = (
        "artifacts/g2_bounded_passive_range_qualification_20260921/CANDIDATE_A_MANIFEST.json",
        "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/A_source_min_q3_10deg_q4_11p25deg/bounded-range-contract.json",
        "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda",
    )
    for relative in bundle_receipts:
        receipts.append({"check": f"legacy_source_freeze:{relative}", "pass": (ROOT / relative).is_file()})
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("static", "live"), default="static")
    parser.add_argument("--method", choices=tuple("ABCDEFG"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipts = static_checks()
    if args.profile == "live":
        receipts.extend(live_checks(args.method))
    payload = {
        "schema": "geniesim_repro_preflight_v1",
        "profile": args.profile,
        "method": args.method,
        "status": "PASS" if all(row["pass"] for row in receipts) else "FAIL",
        "checks": receipts,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
