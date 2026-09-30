#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Verify the ROS2-free G2 Keyboard-v3 portable source release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
REQUIRED = (
    ROOT / "scripts/collect_data.sh",
    ROOT / "scripts/run_g2_keyboard_v3_terminal_collection.sh",
    ROOT / "scripts/prepare_g2_keyboard_v3_collection.py",
    ROOT / "scripts/validate_g2_keyboard_v3_dataset.py",
    ROOT / "configs/g2_policy_branch/keyboard_collection_v3.json",
    ROOT / "source/data_collection/config/robot_cfg/G2/G2_omnipicker_fixed_dual.urdf",
)
FORBIDDEN_DIRECTORIES = ("ros2", "build", "install", "log", "wandb", "artifacts", "datasets")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("static", "live"), default="static")
    parser.add_argument("--collection", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    for path in REQUIRED:
        checks.append({"check": f"required:{path.relative_to(ROOT)}", "pass": path.is_file()})

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    rows = manifest.get("files", []) if isinstance(manifest, dict) else []
    for row in rows:
        path = ROOT / str(row.get("path", ""))
        checks.append(
            {
                "check": f"hash:{row.get('path')}",
                "pass": path.is_file() and digest(path) == row.get("sha256"),
            }
        )

    present_forbidden = [name for name in FORBIDDEN_DIRECTORIES if (ROOT / name).exists()]
    checks.append(
        {"check": "ros2_and_runtime_outputs_excluded", "pass": not present_forbidden, "paths": present_forbidden}
    )
    config = ROOT / "configs/g2_policy_branch/keyboard_collection_v3.json"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        contract_ok = (
            payload.get("action_frame") == "robot_root"
            and payload.get("control_rate_hz") == 50
            and payload.get("physics_rate_hz") == 500
            and payload.get("rgbd_capture_rate_hz") == 25
            and payload.get("maximum_translation_norm_m") == 0.0045
        )
    except (OSError, ValueError, TypeError):
        contract_ok = False
    checks.append({"check": "keyboard_v3_contract", "pass": contract_ok})

    if args.profile == "live":
        executable = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", "")).expanduser()
        checks.append({"check": "isaac_python", "pass": executable.is_file(), "path": str(executable)})
        for variable in ("GENIESIM_ASSET_ROOT", "GENIESIM_CANDIDATE_A_USD", "GENIESIM_PREGRASP_HDF5"):
            value = Path(os.environ.get(variable, "")).expanduser()
            checks.append({"check": f"env:{variable}", "pass": value.exists(), "path": str(value)})
        checks.append({"check": "display", "pass": bool(os.environ.get("DISPLAY"))})
        checks.append({"check": "isaaclab_import_spec", "pass": importlib.util.find_spec("isaaclab") is not None})

    status = "PASS" if checks and all(bool(row["pass"]) for row in checks) else "FAIL"
    report = {
        "schema": "g2_keyboard_release_preflight_v1",
        "profile": args.profile,
        "collection": args.collection,
        "status": status,
        "checks": checks,
        "student_privileged_input_count": 0,
        "ros2_included": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
