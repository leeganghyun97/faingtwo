#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run the G update-free OPEN smoke, then A--G reset-fixed 25-env 6K."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", sys.executable))
SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"
COMPARISON = ROOT / "scripts/run_g2_stage1a_reset_fixed_7p5k_comparison.py"
FROZEN_STUDENT = Path(
    os.environ.get(
        "GENIESIM_FROZEN_STUDENT_CHECKPOINT",
        ROOT / "artifacts/external/CONDITIONAL_CORAL_FROZEN_STUDENT.pt",
    )
)
FROZEN_STUDENT_SHA256 = (
    "1834e70db6fc3a56e77350248961c24d3606dc1ce9385ead462ee5349d31bff2"
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def completed_report(root: Path) -> dict[str, Any] | None:
    supervisor = read_json(root / "STARTUP_SUPERVISOR_REPORT.json") or {}
    if supervisor.get("status") != "TRAINING_COMPLETED":
        return None
    for attempt in reversed(supervisor.get("startup_attempts", [])):
        if not isinstance(attempt, dict) or attempt.get("result") != "PASS":
            continue
        report_path = attempt.get("runtime_report_path")
        if isinstance(report_path, str):
            report = read_json(Path(report_path))
            if report is not None:
                return report
    return None


def smoke_pass(report: dict[str, Any]) -> bool:
    reset = report.get("reset_open_restore")
    if not isinstance(reset, dict):
        return False
    attempts = [row for row in reset.get("attempts", []) if isinstance(row, dict)]
    initial = [row for row in attempts if int(row.get("episode_id", -1)) == 0]
    return bool(
        len(initial) == 25
        and all(bool(row.get("open_restore_pass", False)) for row in initial)
        and all(bool(row.get("fresh_geometry_receipt", False)) for row in initial)
        and all(bool(row.get("geometry_cache_fresh", False)) for row in initial)
        and int(report.get("accepted_transitions", -1)) == 100
        and int(report.get("sac_update_count", -1)) == 0
        and int(report.get("current_gru_privileged_update_count", -1)) == 0
        and int(report.get("RUNTIME_HARDSTOP", 0) or 0) == 0
        and int(report.get("FORBIDDEN_COLLISION", 0) or 0) == 0
        and int(report.get("ACTION_BOUND_VIOLATION", 0) or 0) == 0
        and int(report.get("GRIPPER_AUTHORITY_VIOLATION", 0) or 0) == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-prefix", required=True)
    parser.add_argument("--wandb-group", default="stage1a-reset-fixed-fair-6k")
    parser.add_argument("--constructor-timeout-s", type=float, default=45.0)
    parser.add_argument("--smoke-timeout-s", type=float, default=7200.0)
    parser.add_argument("--training-timeout-s", type=float, default=21600.0)
    args = parser.parse_args()

    report = completed_report(args.smoke_root)
    if report is None:
        if args.smoke_root.exists():
            raise SystemExit("G_SMOKE_OUTPUT_EXISTS_BUT_IS_NOT_COMPLETE")
        smoke_command = [
            str(PYTHON), str(SUPERVISOR),
            "--output-root", str(args.smoke_root),
            "--python", str(PYTHON),
            "--seed", str(args.seed),
            "--num-envs", "25",
            "--accepted-transitions", "100",
            "--runtime-variant",
            "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "--max-startup-attempts", "3",
            "--constructor-timeout-s", str(args.constructor_timeout_s),
            "--training-timeout-s", str(args.smoke_timeout_s),
            "--wandb-run-name", f"{args.wandb_prefix}-g-smoke",
            "--wandb-group", args.wandb_group,
            "--preflight-only",
            "--frozen-student-advisory-checkpoint", str(FROZEN_STUDENT),
            "--frozen-student-advisory-sha256", FROZEN_STUDENT_SHA256,
        ]
        result = subprocess.run(smoke_command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit("G_OPEN_RESTORE_SMOKE_FAILED")
        report = completed_report(args.smoke_root)
    if report is None or not smoke_pass(report):
        raise SystemExit("G_OPEN_RESTORE_SMOKE_CONTRACT_FAILED")
    if args.comparison_root.exists():
        raise SystemExit("COMPARISON_OUTPUT_REFUSES_OVERWRITE")
    comparison_command = [
        str(PYTHON), str(COMPARISON),
        "--output-root", str(args.comparison_root),
        "--python", str(PYTHON),
        "--seed", str(args.seed),
        "--accepted-transitions", "6000",
        "--constructor-timeout-s", str(args.constructor_timeout_s),
        "--training-timeout-s", str(args.training_timeout_s),
        "--wandb-prefix", args.wandb_prefix,
        "--wandb-group", args.wandb_group,
        "--frozen-student-checkpoint", str(FROZEN_STUDENT),
        "--frozen-student-sha256", FROZEN_STUDENT_SHA256,
    ]
    return subprocess.run(comparison_command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
