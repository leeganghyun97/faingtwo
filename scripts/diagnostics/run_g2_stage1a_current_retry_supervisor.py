#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded fresh-process supervisor for Stage-1A HER_FORCE vector runs.

This launcher deliberately treats the known intermittent AppLauncher futex
wait as infrastructure: it never retries from inside an Isaac process.  A
timed-out constructor attempt is preserved as a zero-transition invalid
artifact, terminated, and replaced only by a wholly new Python child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", sys.executable))
DEFAULT_RUNNER = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
SCHEMA = "g2_stage1a_current_v3_her_force_fresh_startup_supervisor_v1"
ONLINE_WANDB_MODE_ARGS = (
        "--wandb-mode",
        "online",
)
RESET_FIXED_25ENV_7P5K_ADVISORY_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
RESET_FIXED_25ENV_7P5K_CURRENT_GRU_PRIVILEGED_VARIANT = (
    "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
RESET_FIXED_25ENV_6K_FSM_ADVISORY_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV"
)
RESET_FIXED_25ENV_6K_CURRENT_GRU_PRIVILEGED_VARIANT = (
    "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV"
)
RESET_FIXED_25ENV_7P5K_VARIANTS = frozenset(
    {
        "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        RESET_FIXED_25ENV_7P5K_ADVISORY_VARIANT,
        RESET_FIXED_25ENV_7P5K_CURRENT_GRU_PRIVILEGED_VARIANT,
    }
)
RESET_FIXED_25ENV_6K_VARIANTS = frozenset(
    {
        "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV",
        RESET_FIXED_25ENV_6K_FSM_ADVISORY_VARIANT,
        RESET_FIXED_25ENV_6K_CURRENT_GRU_PRIVILEGED_VARIANT,
    }
)
POST_APP_LAUNCH_STAGES = frozenset(
    {
        "POST_APP_LAUNCH",
        "PRE_ENV_IMPORT",
        "PRE_ENV_CONSTRUCT",
        "POST_ENV_CONSTRUCT",
        "POST_ENV_RESET",
        "PRE_PHYSICS_SMOKE",
        "POST_PHYSICS_SMOKE",
        "PRE_VECTOR_RUNTIME",
        "VECTOR_RUNTIME_RETURNED",
    }
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _last_line(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return lines[-1] if lines else None


def _relevant_environment() -> dict[str, str | None]:
    names = (
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "OMNI_KIT_ACCEPT_EULA",
    )
    values: dict[str, str | None] = {name: os.environ.get(name) for name in names}
    values.update(
        {
            name: "<present>"
            for name in sorted(os.environ)
            if name.startswith(("ISAAC_", "OMNI_")) and name not in values
        }
    )
    return values


def _environment_fingerprint(values: dict[str, str | None]) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _copy_proc_text(source: Path, destination: Path) -> None:
    try:
        content = source.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        content = f"<unavailable: {type(error).__name__}: {error}>\n"
    destination.write_text(content, encoding="utf-8")


def _capture_hang_snapshot(*, pid: int, destination: Path) -> dict[str, Any]:
    """Capture non-invasive evidence before terminating our own child."""

    destination.mkdir(parents=True, exist_ok=True)
    proc = Path("/proc") / str(pid)
    for name in ("status", "wchan", "syscall", "maps"):
        _copy_proc_text(proc / name, destination / f"process_{name}.txt")
    fd_lines: list[str] = []
    try:
        for entry in sorted((proc / "fd").iterdir(), key=lambda item: item.name):
            try:
                target = os.readlink(entry)
            except OSError as error:
                target = f"<unavailable: {type(error).__name__}: {error}>"
            fd_lines.append(f"{entry.name} -> {target}")
    except OSError as error:
        fd_lines.append(f"<fd unavailable: {type(error).__name__}: {error}>")
    (destination / "process_fds.txt").write_text("\n".join(fd_lines) + "\n", encoding="utf-8")

    thread_ids: list[int] = []
    try:
        task_entries = sorted((proc / "task").iterdir(), key=lambda item: item.name)
    except OSError:
        task_entries = []
    for task in task_entries:
        try:
            tid = int(task.name)
        except ValueError:
            continue
        thread_ids.append(tid)
        for name in ("status", "wchan", "stack", "syscall"):
            _copy_proc_text(task / name, destination / f"thread_{tid}_{name}.txt")
    return {"pid": pid, "thread_count": len(thread_ids), "thread_ids": thread_ids}


def _terminate_child(process: subprocess.Popen[Any]) -> dict[str, Any]:
    if process.poll() is not None:
        return {"termination": "already_exited", "returncode": process.returncode}
    process.terminate()
    try:
        return {
            "termination": "SIGTERM",
            "returncode": process.wait(timeout=10.0),
        }
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "termination": "SIGKILL_AFTER_SIGTERM_TIMEOUT",
            "returncode": process.wait(timeout=10.0),
        }


def _write_progress(
    path: Path, *, args: argparse.Namespace, attempts: list[dict[str, Any]], status: str
) -> None:
    _atomic_json(
        path,
        {
            "schema": SCHEMA,
            "execution_mode": "LIVE_STAGE1A_HER_FORCE_FRESH_PROCESS",
            "runtime_variant": args.runtime_variant,
            "num_envs": args.num_envs,
            "target_accepted_transitions": args.accepted_transitions,
            "status": status,
            "startup_attempts": attempts,
            "startup_attempt_count": len(attempts),
            "privileged_used": "PRIVILEGED" in args.runtime_variant,
            # This supervisor never chains a bounded result into an automatic
            # long run.  Keep the legacy explicit receipt alongside the more
            # general field below so downstream status readers remain stable.
            "auto_15k_started": False,
            "auto_long_run_started": False,
        },
    )


def _build_command(*, args: argparse.Namespace, attempt_root: Path, attempt: int) -> list[str]:
    report = attempt_root / "TRAINING_REPORT.json"
    command = [
        str(args.python),
        str(args.runner),
        "--execute-live",
        "--num-envs",
        str(args.num_envs),
        "--accepted-transitions",
        str(args.accepted_transitions),
        "--runtime-variant",
        args.runtime_variant,
        "--output-dir",
        str(attempt_root / "runtime"),
        "--report",
        str(report),
        "--seed",
        str(args.seed),
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    else:
        command.extend(
            (
                "--wandb",
                *ONLINE_WANDB_MODE_ARGS,
                "--wandb-project",
                args.wandb_project,
                "--wandb-run-name",
                f"{args.wandb_run_name}-attempt-{attempt:02d}",
                "--wandb-group",
                args.wandb_group,
            )
        )
    if args.frozen_student_advisory_checkpoint is not None:
        command.extend(
            (
                "--frozen-student-advisory-checkpoint",
                str(args.frozen_student_advisory_checkpoint),
                "--frozen-student-advisory-sha256",
                str(args.frozen_student_advisory_sha256),
            )
        )
    return command


def _run_attempt(*, args: argparse.Namespace, attempt: int, root: Path) -> dict[str, Any]:
    attempt_root = root / f"attempt-{attempt:02d}"
    attempt_root.mkdir(parents=False, exist_ok=False)
    report = attempt_root / "TRAINING_REPORT.json"
    # ``run_g2_stage1a_vector_runtime.py`` intentionally keeps bootstrap
    # receipts adjacent to the not-yet-created runtime directory.  Derive the
    # paths with the same contract; looking next to ``TRAINING_REPORT.json``
    # would miss a healthy ``PRE_VECTOR_RUNTIME`` receipt and misclassify an
    # active child as an AppLauncher timeout.
    marker = attempt_root.parent / f"{attempt_root.name}_launch.json"
    smoke_report = attempt_root.parent / f"{attempt_root.name}_physics_smoke.json"
    stdout_path = attempt_root / "stdout.log"
    stderr_path = attempt_root / "stderr.log"
    command = _build_command(args=args, attempt_root=attempt_root, attempt=attempt)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "source")
    environment["PYTHONUNBUFFERED"] = "1"
    relevant_env = _relevant_environment()
    receipt: dict[str, Any] = {
        "attempt": attempt,
        "attempt_root": str(attempt_root),
        "command": command,
        "cwd": str(ROOT),
        "python": str(args.python),
        "environment_fingerprint": _environment_fingerprint(relevant_env),
        "relevant_environment": relevant_env,
        "accepted_transitions": 0,
        "transition_zero_invalid": False,
        "privileged_used": False,
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        started_monotonic = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            start_new_session=True,
        )
        receipt.update(
            {
                "child_pid": process.pid,
                "parent_pid": os.getpid(),
                "child_started": True,
                "constructor_entry_monotonic_s": started_monotonic,
            }
        )
        _atomic_json(attempt_root / "ATTEMPT_RECEIPT.json", receipt)
        startup_pass = False
        while process.poll() is None:
            launch = _read_json(marker)
            stage = launch.get("stage") if launch else None
            # The runner deliberately reuses its durable launch marker for
            # later boundaries.  Any known post-App stage proves constructor
            # return; requiring the literal original value would incorrectly
            # kill a healthy env/reset/training process at 45 seconds.
            if stage in POST_APP_LAUNCH_STAGES:
                startup_pass = True
                receipt["constructor_return_monotonic_s"] = time.monotonic()
                receipt["constructor_duration_s"] = (
                    receipt["constructor_return_monotonic_s"] - started_monotonic
                )
                receipt["last_launch_stage"] = stage
                break
            if time.monotonic() - started_monotonic >= args.constructor_timeout_s:
                receipt.update(
                    {
                        "last_launch_stage": stage,
                        "constructor_timeout": True,
                        "transition_zero_invalid": True,
                        "failure_class": "APPLAUNCHER_CONSTRUCTOR_TIMEOUT",
                        "hang_snapshot": _capture_hang_snapshot(
                            pid=process.pid, destination=attempt_root / "hang_snapshot"
                        ),
                    }
                )
                receipt["termination"] = _terminate_child(process)
                receipt["last_stdout_line"] = _last_line(stdout_path)
                receipt["last_stderr_line"] = _last_line(stderr_path)
                _atomic_json(attempt_root / "ATTEMPT_RECEIPT.json", receipt)
                return receipt
            time.sleep(0.20)
        if not startup_pass:
            receipt.update(
                {
                    "last_launch_stage": (
                        (_read_json(marker) or {}).get("stage")
                    ),
                    "returncode": process.returncode,
                    "transition_zero_invalid": True,
                    "failure_class": "STARTUP_EXIT_BEFORE_CONSTRUCTOR_RETURN",
                    "last_stdout_line": _last_line(stdout_path),
                    "last_stderr_line": _last_line(stderr_path),
                }
            )
            _atomic_json(attempt_root / "ATTEMPT_RECEIPT.json", receipt)
            return receipt

        training_deadline = time.monotonic() + args.training_timeout_s
        while process.poll() is None and time.monotonic() < training_deadline:
            time.sleep(1.0)
        if process.poll() is None:
            receipt.update(
                {
                    "failure_class": "POST_STARTUP_TRAINING_TIMEOUT",
                    "hang_snapshot": _capture_hang_snapshot(
                        pid=process.pid, destination=attempt_root / "hang_snapshot"
                    ),
                    "termination": _terminate_child(process),
                }
            )
        else:
            receipt["returncode"] = process.returncode
        receipt["last_launch_stage"] = ((_read_json(marker) or {}).get("stage"))
        receipt["physics_smoke"] = _read_json(smoke_report)
        runtime_report = _read_json(attempt_root / "runtime" / "STAGE1A_VECTOR_REPORT.json")
        receipt["runtime_report_path"] = (
            str(attempt_root / "runtime" / "STAGE1A_VECTOR_REPORT.json")
            if runtime_report is not None
            else None
        )
        if runtime_report is not None:
            receipt["accepted_transitions"] = int(
                runtime_report.get("accepted_transitions", 0)
            )
            receipt["wandb"] = runtime_report.get("wandb")
        completed_report = bool(
            runtime_report is not None
            and runtime_report.get("TRAINING_COMPLETED") is True
            and receipt["accepted_transitions"] == args.accepted_transitions
        )
        # Isaac Kit's post-finalization -11 is a separately tracked shutdown
        # defect.  A durable completed 3K report is the authoritative result;
        # do not relabel it as a training/runtime failure after all updates,
        # checkpoint writes, and W&B finalization have completed.
        if completed_report:
            receipt["result"] = "PASS"
            receipt["known_shutdown_sigsegv_separate"] = bool(
                receipt.get("returncode") == -11
            )
        elif receipt.get("physics_smoke", {}).get("PHYSICS_SMOKE") != "PASS":
            receipt.setdefault("failure_class", "PHYSICS_SMOKE_FAIL")
        elif receipt.get("returncode") != 0:
            receipt.setdefault("failure_class", "POST_STARTUP_RUNTIME_FAILURE")
        elif receipt["accepted_transitions"] != args.accepted_transitions:
            receipt.setdefault("failure_class", "TRAINING_TRANSITION_TARGET_NOT_REACHED")
        receipt["last_stdout_line"] = _last_line(stdout_path)
        receipt["last_stderr_line"] = _last_line(stderr_path)
        _atomic_json(attempt_root / "ATTEMPT_RECEIPT.json", receipt)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-envs",
        type=int,
        choices=(10, 25),
        default=10,
        help="10 for legacy fair variants; 25 for reset-fixed 7.5K method comparison.",
    )
    # Legacy training-only choice contract retained for static readers:
    # choices=(3000, 6000, 7500, 15000, 30000)
    parser.add_argument(
        "--accepted-transitions", type=int, choices=(100, 3000, 6000, 7500, 15000, 30000), required=True
    )
    parser.add_argument(
        "--runtime-variant",
        choices=(
            "V3_BASELINE",
            "V3_1_POST_CLOSE_MICRO",
            "V3_1_LATERAL_GATE_OFF_PAIRED",
            "V3_1_LATERAL_OFF_HER_FORCE_15K",
            "V3_1_LATERAL_OFF_HER_FORCE_30K",
            "V3_CURRENT_HER_FORCE_FAIR_6K",
            "V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
            "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_FAIR_6K",
            "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K",
            RESET_FIXED_25ENV_7P5K_ADVISORY_VARIANT,
            "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            RESET_FIXED_25ENV_7P5K_CURRENT_GRU_PRIVILEGED_VARIANT,
            "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV",
            RESET_FIXED_25ENV_6K_FSM_ADVISORY_VARIANT,
            RESET_FIXED_25ENV_6K_CURRENT_GRU_PRIVILEGED_VARIANT,
            "V3_2_RELAX_LATERAL_STABILIZE",
            "V3_PRIVILEGED_GEOMETRY_TEACHER_HER_FORCE_3K",
            "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K",
        ),
        required=True,
    )
    parser.add_argument("--max-startup-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--constructor-timeout-s", type=float, default=45.0)
    parser.add_argument("--training-timeout-s", type=float, default=21600.0)
    parser.add_argument("--wandb-project", default="geniesim-g2-stage1a-residual-sac")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-group", default="current-rule-reward-v3-her-force")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--frozen-student-advisory-checkpoint", type=Path)
    parser.add_argument("--frozen-student-advisory-sha256")
    args = parser.parse_args()
    if not args.python.is_file() or not args.runner.is_file():
        raise SystemExit("SUPERVISOR_PYTHON_OR_RUNNER_MISSING")
    if args.preflight_only and (
        args.runtime_variant not in RESET_FIXED_25ENV_7P5K_VARIANTS
        or args.num_envs != 25
        or args.accepted_transitions != 100
    ):
        raise SystemExit("RESET_FIXED_PREFLIGHT_REQUIRES_25ENVS_AND_100_TRANSITIONS")
    advisory_variant = args.runtime_variant in (
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K",
        RESET_FIXED_25ENV_7P5K_ADVISORY_VARIANT,
        RESET_FIXED_25ENV_7P5K_CURRENT_GRU_PRIVILEGED_VARIANT,
        RESET_FIXED_25ENV_6K_FSM_ADVISORY_VARIANT,
        RESET_FIXED_25ENV_6K_CURRENT_GRU_PRIVILEGED_VARIANT,
    )
    if args.runtime_variant in RESET_FIXED_25ENV_7P5K_VARIANTS:
        expected_target = 100 if args.preflight_only else 7500
        if args.num_envs != 25 or args.accepted_transitions != expected_target:
            if args.runtime_variant == RESET_FIXED_25ENV_7P5K_ADVISORY_VARIANT:
                raise SystemExit(
                    "RESET_FIXED_7P5K_ADVISORY_REQUIRES_NUM_ENVS_25_AND_TARGET_7500"
                )
            raise SystemExit("RESET_FIXED_7P5K_REQUIRES_NUM_ENVS_25_AND_TARGET_7500")
    elif args.runtime_variant in RESET_FIXED_25ENV_6K_VARIANTS:
        if args.num_envs != 25 or args.accepted_transitions != 6000:
            raise SystemExit("RESET_FIXED_6K_REQUIRES_NUM_ENVS_25_AND_TARGET_6000")
    elif args.num_envs != 10:
        raise SystemExit("SUPERVISOR_NUM_ENVS_25_RESERVED_FOR_RESET_FIXED_7P5K")
    if advisory_variant and (
        args.frozen_student_advisory_checkpoint is None
        or not args.frozen_student_advisory_checkpoint.is_file()
        or not isinstance(args.frozen_student_advisory_sha256, str)
        or len(args.frozen_student_advisory_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.frozen_student_advisory_sha256.lower()
        )
    ):
        if args.runtime_variant == "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K":
            raise SystemExit(
                "FAIR_6K_ADVISORY_REQUIRES_EXISTING_FROZEN_STUDENT_CHECKPOINT_AND_SHA256"
            )
        raise SystemExit(
            "RESET_FIXED_7P5K_ADVISORY_REQUIRES_EXISTING_FROZEN_STUDENT_CHECKPOINT_AND_SHA256"
        )
    if not advisory_variant and (
        args.frozen_student_advisory_checkpoint is not None
        or args.frozen_student_advisory_sha256 is not None
    ):
        raise SystemExit("FROZEN_STUDENT_ADVISORY_ARGS_RESERVED_FOR_ADVISORY_VARIANT")
    if args.output_root.exists():
        raise SystemExit("SUPERVISOR_OUTPUT_REFUSES_OVERWRITE")
    args.output_root.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, Any]] = []
    progress = args.output_root / "STARTUP_SUPERVISOR_REPORT.json"
    _write_progress(progress, args=args, attempts=attempts, status="RUNNING")
    for attempt in range(1, args.max_startup_attempts + 1):
        receipt = _run_attempt(args=args, attempt=attempt, root=args.output_root)
        attempts.append(receipt)
        if receipt.get("result") == "PASS":
            _write_progress(progress, args=args, attempts=attempts, status="TRAINING_COMPLETED")
            return 0
        retryable_startup_failures = {
            "APPLAUNCHER_CONSTRUCTOR_TIMEOUT",
            "STARTUP_EXIT_BEFORE_CONSTRUCTOR_RETURN",
        }
        if receipt.get("failure_class") not in retryable_startup_failures:
            _write_progress(progress, args=args, attempts=attempts, status="POST_STARTUP_FAILURE")
            return 2
        _write_progress(progress, args=args, attempts=attempts, status="RETRYING_FRESH_CHILD")
    _write_progress(progress, args=args, attempts=attempts, status="STARTUP_RETRIES_EXHAUSTED")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
