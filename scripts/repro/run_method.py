#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve a canonical Stage-1A method and dry-run or execute its official supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reproducibility/methods_a_to_g.json"
ASSETS = ROOT / "configs/reproducibility/external_assets.json"
SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"
RUNNER = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"CONFIG_OBJECT_REQUIRED:{path}")
    return value


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def find_asset(asset_id: str) -> tuple[Path, str | None]:
    for item in load_json(ASSETS)["assets"]:
        if item["id"] != asset_id:
            continue
        configured = os.environ.get(item["environment_variable"], "")
        path = Path(configured).expanduser() if configured else ROOT / item["legacy_relative_path"]
        return path.resolve(), item["sha256"]
    raise SystemExit(f"UNKNOWN_EXTERNAL_ASSET:{asset_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple("ABCDEFG"), required=True)
    parser.add_argument("--accepted-transitions", type=int, choices=(6000, 7500), default=6000)
    parser.add_argument("--num-envs", type=int, choices=(25,), default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "geniesim-stage1a-repro"))
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group", default="stage1a-reset-fixed-a-to-g")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run the approved 25-env/100-transition update-free preflight")
    args = parser.parse_args()

    methods = load_json(CONFIG)["methods"]
    method = methods[args.method]
    budget = 7500 if args.smoke else args.accepted_transitions
    variant = method[f"runtime_variant_{budget}"]
    accepted = 100 if args.smoke else budget
    isaac_python = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", sys.executable)).expanduser().resolve()
    output = args.output_root
    if output is None:
        output_base = Path(os.environ.get("GENIESIM_OUTPUT_ROOT", ROOT / "output")).expanduser()
        suffix = "smoke" if args.smoke else f"{budget}"
        output = output_base / f"method_{args.method.lower()}_{suffix}_seed{args.seed}"
    output = output.resolve()
    run_name = args.wandb_run_name or output.name

    command = [
        str(isaac_python), str(SUPERVISOR),
        "--output-root", str(output),
        "--python", str(isaac_python),
        "--runner", str(RUNNER),
        "--seed", str(args.seed),
        "--num-envs", "25",
        "--accepted-transitions", str(accepted),
        "--runtime-variant", variant,
        "--wandb-project", args.wandb_project,
        "--wandb-run-name", run_name,
        "--wandb-group", args.wandb_group,
    ]
    if args.smoke:
        command.append("--preflight-only")

    advisory = bool(method["student_advisory"])
    student_path: Path | None = None
    expected_student_hash: str | None = None
    if advisory:
        student_path, expected_student_hash = find_asset("conditional_coral_frozen_student")
        command.extend([
            "--frozen-student-advisory-checkpoint", str(student_path),
            "--frozen-student-advisory-sha256", str(expected_student_hash),
        ])

    missing = [str(path) for path in (isaac_python, SUPERVISOR, RUNNER) if not path.is_file()]
    hash_mismatch = False
    if advisory and student_path is not None:
        if not student_path.is_file():
            missing.append(str(student_path))
        elif expected_student_hash is not None:
            hash_mismatch = digest(student_path) != expected_student_hash

    receipt = {
        "schema": "geniesim_method_launch_v1",
        "method": args.method,
        "name": method["name"],
        "purpose": method["purpose"],
        "runtime_variant": variant,
        "num_envs": 25,
        "accepted_transitions": accepted,
        "smoke": args.smoke,
        "execute_live": args.execute_live,
        "output_root": str(output),
        "privileged_runtime_authority": method["privileged_runtime_authority"],
        "student_advisory": advisory,
        "missing_inputs": missing,
        "checkpoint_hash_mismatch": hash_mismatch,
        "command": shlex.join(command),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not args.execute_live:
        return 0
    if missing or hash_mismatch:
        raise SystemExit("LIVE_METHOD_PREFLIGHT_FAILED")
    if output.exists():
        raise SystemExit(f"OUTPUT_REFUSES_OVERWRITE:{output}")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
