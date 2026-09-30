#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Sequential reset-fixed Stage-1A preflight and fair 6K/7.5K comparison.

Each method is launched through the existing fresh-top-level-process
supervisor.  Each full run performs physics smoke and measured-state reset
gating before replay/W&B/SAC.  The script never starts 15K/30K.  The CURRENT
GRU + Privileged method is the explicit signed-margin teacher / causal-GRU
auxiliary implementation; it does not reuse the excluded old distillation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"
DEFAULT_PYTHON = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", sys.executable))
DEFAULT_FROZEN_STUDENT = Path(
    os.environ.get(
        "GENIESIM_FROZEN_STUDENT_CHECKPOINT",
        ROOT / "artifacts/external/CONDITIONAL_CORAL_FROZEN_STUDENT.pt",
    )
)
DEFAULT_FROZEN_STUDENT_SHA256 = (
    "1834e70db6fc3a56e77350248961c24d3606dc1ce9385ead462ee5349d31bff2"
)


@dataclass(frozen=True)
class Method:
    label: str
    slug: str
    variant: str
    advisory: bool = False


METHODS = (
    Method("V3 CURRENT", "v3-current", "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV"),
    Method("V3.1", "v31", "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV"),
    Method(
        "V3.1 lateral-off",
        "v31-lateral-off",
        "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    ),
    Method("V3.2", "v32", "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV"),
    Method(
        "Privileged geometry hard-gate",
        "privileged-hard-gate",
        "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    ),
    Method(
        "FSM advisory",
        "fsm-advisory",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        advisory=True,
    ),
    Method(
        "CURRENT GRU + Privileged",
        "current-gru-privileged",
        "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        advisory=True,
    ),
)

METHODS_6K = (
    Method("V3 CURRENT", "v3-current", "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV"),
    Method("V3.1", "v31", "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV"),
    Method(
        "V3.1 lateral-off",
        "v31-lateral-off",
        "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV",
    ),
    Method("V3.2", "v32", "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV"),
    Method(
        "Privileged geometry hard-gate",
        "privileged-hard-gate",
        "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV",
    ),
    Method(
        "FSM advisory",
        "fsm-advisory",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV",
        advisory=True,
    ),
    Method(
        "CURRENT GRU + Privileged",
        "current-gru-privileged",
        "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV",
        advisory=True,
    ),
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def completed_runtime_report(run_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    supervisor = read_json(run_root / "STARTUP_SUPERVISOR_REPORT.json") or {}
    if supervisor.get("status") != "TRAINING_COMPLETED":
        return None, None
    for attempt in reversed(supervisor.get("startup_attempts", [])):
        if not isinstance(attempt, dict) or attempt.get("result") != "PASS":
            continue
        report_path = attempt.get("runtime_report_path")
        if not isinstance(report_path, str):
            continue
        path = Path(report_path)
        report = read_json(path)
        if report is not None:
            return path, report
    return None, None


def reset_pass(report: dict[str, Any], *, preflight: bool) -> tuple[bool, dict[str, Any]]:
    reset = report.get("reset_open_restore")
    if not isinstance(reset, dict):
        return False, {"reason": "RESET_RECEIPT_MISSING"}
    attempts = [row for row in reset.get("attempts", []) if isinstance(row, dict)]
    # A full run can stop while a subsequent reset is right-censored.  A row
    # is a failure only if it expired or carries an explicit failure reason;
    # merely being pending at the exact budget boundary is not a reset defect.
    failures = [
        row
        for row in attempts
        if bool(row.get("open_restore_expired", False))
        or row.get("failure_reason") not in (None, "")
    ]
    completed = [row for row in attempts if bool(row.get("open_restore_pass", False))]
    pass_value = (
        not failures
        and bool(completed)
        and all(bool(row.get("fresh_geometry_receipt", False)) for row in completed)
        and all(not bool(row.get("previous_episode_geometry_used", False)) for row in completed)
        and bool(reset.get("episode_clock_hold_pass", False))
        and int(reset.get("episode_timeout_during_open_restore_count", -1)) == 0
        and float(report.get("PREMATURE_PRE_CLOSE_CONTACT_RATE", 0.0) or 0.0) == 0.0
        and (not preflight or len(completed) >= 25)
    )
    return pass_value, {
        "completed_open_restores": len(completed),
        "right_censored_pending": len(attempts) - len(completed) - len(failures),
        "explicit_failures": len(failures),
        "previous_episode_geometry_used": sum(
            bool(row.get("previous_episode_geometry_used", False)) for row in completed
        ),
        "episode_clock_hold_pass": bool(
            reset.get("episode_clock_hold_pass", False)
        ),
        "episode_timeout_during_open_restore_count": int(
            reset.get("episode_timeout_during_open_restore_count", -1)
        ),
    }


def command(args: argparse.Namespace, method: Method, run_root: Path, *, preflight: bool) -> list[str]:
    target = 100 if preflight else int(args.accepted_transitions)
    cmd = [
        str(args.python),
        str(SUPERVISOR),
        "--output-root",
        str(run_root),
        "--python",
        str(args.python),
        "--seed",
        str(args.seed),
        "--num-envs",
        str(args.num_envs),
        "--accepted-transitions",
        str(target),
        "--runtime-variant",
        method.variant,
        "--max-startup-attempts",
        "3",
        "--constructor-timeout-s",
        str(args.constructor_timeout_s),
        "--training-timeout-s",
        str(args.training_timeout_s),
        "--wandb-run-name",
        f"{args.wandb_prefix}-{method.slug}-{target}",
        "--wandb-group",
        args.wandb_group,
    ]
    if preflight:
        cmd.append("--preflight-only")
    if method.advisory:
        cmd.extend(
            (
                "--frozen-student-advisory-checkpoint",
                str(args.frozen_student_checkpoint),
                "--frozen-student-advisory-sha256",
                args.frozen_student_sha256,
            )
        )
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, choices=(10, 25), default=25)
    parser.add_argument(
        "--accepted-transitions", type=int, choices=(6000, 7500), default=7500
    )
    parser.add_argument("--constructor-timeout-s", type=float, default=45.0)
    parser.add_argument("--training-timeout-s", type=float, default=21600.0)
    parser.add_argument("--wandb-prefix", required=True)
    parser.add_argument("--wandb-group", default="stage1a-reset-fixed-fair-7p5k")
    parser.add_argument("--frozen-student-checkpoint", type=Path, default=DEFAULT_FROZEN_STUDENT)
    parser.add_argument("--frozen-student-sha256", default=DEFAULT_FROZEN_STUDENT_SHA256)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("COMPARISON_OUTPUT_REFUSES_OVERWRITE")
    if not args.python.is_file() or not SUPERVISOR.is_file():
        raise SystemExit("COMPARISON_RUNTIME_MISSING")
    args.output_root.mkdir(parents=True, exist_ok=False)
    status_path = args.output_root / "COMPARISON_PROGRESS.json"
    method_results: dict[str, Any] = {}
    static_results: dict[str, Any] = {}

    def persist(status: str) -> None:
        atomic_json(
            status_path,
            {
                "schema": "g2_stage1a_reset_fixed_fair_7p5k_comparison_v1",
                "status": status,
                "common_contract": {
                    "num_envs": int(args.num_envs),
                    "accepted_transitions": int(args.accepted_transitions),
                    "transitions_per_env": (
                        int(args.accepted_transitions) // int(args.num_envs)
                    ),
                    "seed": args.seed,
                    "replay": "HER_FORCE",
                    "old_distillation_used": False,
                    "auto_long_run_started": False,
                },
                "methods": method_results,
                "static_results": static_results,
            },
        )

    # Every canonical runner executes the physics/finite-I/O smoke and then
    # blocks replay/W&B/SAC behind measured OPEN + fresh geometry.  Keeping
    # this preflight in the *same* fresh process avoids a second 150--200 step
    # OPEN restoration while still guaranteeing test-before-train ordering.
    persist("TRAINING_RUNNING")
    methods = METHODS_6K if args.accepted_transitions == 6000 else METHODS
    for method in methods:
        train_root = args.output_root / "train" / method.slug
        completed_path, completed = completed_runtime_report(train_root)
        returncode = 0
        if completed is None:
            returncode = subprocess.run(
                command(args, method, train_root, preflight=False),
                cwd=ROOT,
                check=False,
            ).returncode
            completed_path, completed = completed_runtime_report(train_root)
        entry: dict[str, Any] = {
            "variant": method.variant,
            "training_started": True,
        }
        method_results[method.label] = entry
        entry["training_returncode"] = returncode
        entry["training_report"] = str(completed_path) if completed_path else None
        if completed is None:
            entry["preflight"] = "FAIL"
            entry["training"] = "FAIL"
        else:
            reset_ok, reset_receipt = reset_pass(completed, preflight=False)
            training_pass = bool(
                reset_ok
                and completed.get("source_freeze_match") is True
                and completed.get("vector_contract_pass") is True
                and int(completed.get("accepted_transitions", -1))
                == int(args.accepted_transitions)
                and int(completed.get("RUNTIME_HARDSTOP", 0) or 0) == 0
                and int(completed.get("FORBIDDEN_COLLISION", 0) or 0) == 0
                and int(completed.get("ACTION_BOUND_VIOLATION", 0) or 0) == 0
                and int(completed.get("GRIPPER_AUTHORITY_VIOLATION", 0) or 0) == 0
            )
            entry["preflight"] = "PASS" if reset_ok else "FAIL"
            entry["preflight_order"] = (
                "PHYSICS_SMOKE_THEN_MEASURED_RESET_GATE_THEN_REPLAY_AND_SAC"
            )
            entry["training"] = "PASS" if training_pass else "FAIL"
            entry["reset"] = reset_receipt
            entry["grasp_evaluation"] = completed.get("GRASP_EVALUATION")
            entry["wandb"] = completed.get("wandb")
        persist("TRAINING_RUNNING")
        if entry["training"] != "PASS":
            # A missing method invalidates the fixed-seed A--G comparison.
            # Stop immediately instead of spending hours on later methods
            # whose results cannot form a complete fair table.
            static_results["fail_closed_method"] = method.label
            static_results["fail_closed_reason"] = (
                "METHOD_TRAINING_OR_RESET_CONTRACT_FAILED"
            )
            persist("METHOD_FAIL_CLOSED")
            return 2

    persist("COMPARISON_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
