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
TRAINING_AUTHORITY = ROOT / "configs/reproducibility/g2_training_authority.json"
PREFLIGHT = ROOT / "scripts/preflight.sh"
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


def required_asset_receipts(method: str) -> list[dict[str, object]]:
    """Resolve every runtime input required by a canonical A--G method.

    Dry-run receipts must be as strict as the live preflight.  Previously the
    dry route only checked the frozen student used by F/G, which could report
    ``missing_inputs=[]`` while a shared BC policy or pregrasp dataset was
    absent.  Keep the same ``required_for`` semantics as the live preflight and
    hash every declared file before constructing the launch receipt.
    """

    scopes = {"isaac_scene", "all_live_methods", f"method_{method}"}
    receipts: list[dict[str, object]] = []
    for item in load_json(ASSETS)["assets"]:
        if not scopes.intersection(set(item["required_for"])):
            continue
        configured = os.environ.get(item["environment_variable"], "")
        path = (
            Path(configured).expanduser()
            if configured
            else ROOT / item["legacy_relative_path"]
        ).resolve()
        expected = item.get("sha256")
        exists = path.is_file() if expected else path.is_dir()
        actual = digest(path) if exists and expected else None
        hash_match = None if expected is None else bool(actual == expected)
        receipts.append(
            {
                "id": item["id"],
                "artifact_type": item.get("artifact_type", "unspecified"),
                "bundle_path": item.get("bundle_path"),
                "environment_variable": item["environment_variable"],
                "path": str(path),
                "path_source": "environment" if configured else "repository_legacy",
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_match": hash_match,
                "pass": bool(exists and hash_match is not False),
            }
        )
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple("ABCDEFG"), required=True)
    parser.add_argument("--accepted-transitions", type=int, choices=(6000, 7500), default=6000)
    parser.add_argument("--num-envs", type=int, choices=(10, 25), default=25)
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
    asset_receipts = required_asset_receipts(args.method)
    assets_by_id = {str(item["id"]): item for item in asset_receipts}
    training_authority = load_json(TRAINING_AUTHORITY)
    action_authority = training_authority["action_contract"]
    robot_authority = training_authority["robot_description"]
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
        "--num-envs", str(args.num_envs),
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
        student = assets_by_id["conditional_coral_frozen_student"]
        student_path = Path(str(student["path"]))
        expected_student_hash = str(student["expected_sha256"])
        command.extend([
            "--frozen-student-advisory-checkpoint", str(student_path),
            "--frozen-student-advisory-sha256", str(expected_student_hash),
        ])

    missing = [
        str(path)
        for path in (isaac_python, PREFLIGHT, SUPERVISOR, RUNNER, TRAINING_AUTHORITY)
        if not path.is_file()
    ]
    missing.extend(
        str(item["path"]) for item in asset_receipts if not bool(item["exists"])
    )
    missing = sorted(set(missing))
    hash_mismatches = [
        str(item["id"])
        for item in asset_receipts
        if bool(item["exists"]) and item["hash_match"] is False
    ]
    hash_mismatch = bool(hash_mismatches)

    receipt = {
        "schema": "geniesim_method_launch_v2",
        "method": args.method,
        "name": method["name"],
        "purpose": method["purpose"],
        "runtime_variant": variant,
        "num_envs": args.num_envs,
        "accepted_transitions": accepted,
        "smoke": args.smoke,
        "execute_live": args.execute_live,
        "output_root": str(output),
        "privileged_runtime_authority": method["privileged_runtime_authority"],
        "student_advisory": advisory,
        "missing_inputs": missing,
        "checkpoint_hash_mismatch": hash_mismatch,
        "asset_hash_mismatches": hash_mismatches,
        "required_asset_count": len(asset_receipts),
        "required_asset_contract_pass": all(
            bool(item["pass"]) for item in asset_receipts
        ),
        "required_assets": asset_receipts,
        "training_authority": {
            "manifest": str(TRAINING_AUTHORITY.relative_to(ROOT)),
            "manifest_sha256": digest(TRAINING_AUTHORITY),
            "training_urdf": robot_authority["training_urdf"],
            "training_urdf_sha256": robot_authority["training_urdf_sha256"],
            "teleop_translation_m_per_normalized": action_authority[
                "teleop_translation_m_per_normalized"
            ],
            "stage1a_final_xyz_max_norm_m": action_authority[
                "stage1a_final_xyz_max_norm_m"
            ],
            "residual_effective_max_norm_m": action_authority[
                "residual_effective_max_norm_m"
            ],
        },
        "live_preflight_command": shlex.join(
            [str(PREFLIGHT), "--profile", "live", "--method", args.method]
        ),
        "command": shlex.join(command),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not args.execute_live:
        return 0
    if missing or hash_mismatch:
        raise SystemExit("LIVE_METHOD_PREFLIGHT_FAILED")
    runtime_env = os.environ.copy()
    source_root = str(ROOT / "source")
    current_pythonpath = runtime_env.get("PYTHONPATH", "")
    runtime_env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else source_root
    )
    preflight = subprocess.run(
        [str(PREFLIGHT), "--profile", "live", "--method", args.method],
        cwd=ROOT,
        check=False,
        env=runtime_env,
    )
    if preflight.returncode != 0:
        raise SystemExit("LIVE_METHOD_PREFLIGHT_FAILED")
    if output.exists():
        raise SystemExit(f"OUTPUT_REFUSES_OVERWRITE:{output}")
    return subprocess.run(command, cwd=ROOT, check=False, env=runtime_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
