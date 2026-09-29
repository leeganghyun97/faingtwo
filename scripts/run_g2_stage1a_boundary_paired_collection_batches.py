#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run bounded, update-free boundary-paired collection batches sequentially.

Each batch is a fresh top-level Isaac Python process.  It runs exactly one
3,000-transition collection-only vector job, then invokes the offline paired
audit across all completed batches.  No batch is a SAC/student training run;
only an incomplete/unsafe collection report or audit-process failure stops
the supervisor.  Isaac's known shutdown return code is not a data-validity
signal once the explicit collection receipt has passed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
AUDIT = ROOT / "scripts/audit_g2_stage1a_boundary_paired_dataset.py"
FINALIZER = ROOT / "scripts/finalize_g2_stage1a_boundary_teacher_student.py"
VARIANT = "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _paths(root: Path, index: int, attempt: int) -> tuple[Path, Path, Path, Path, Path]:
    """Return fresh-process paths; a startup retry can never overwrite an attempt."""

    batch = root / f"batch-{index:02d}-attempt-{attempt:02d}"
    return (
        batch,
        batch / "REPORT.json",
        root / "logs" / f"batch-{index:02d}-attempt-{attempt:02d}.stdout.log",
        root / "logs" / f"batch-{index:02d}-attempt-{attempt:02d}.stderr.log",
        root / f"batch-{index:02d}-attempt-{attempt:02d}_launch.json",
    )


def _terminate_process_group(child: subprocess.Popen[str]) -> int:
    """Stop only this fresh child session, preserving host caches and shared memory."""

    if child.poll() is not None:
        return int(child.returncode)
    os.killpg(child.pid, signal.SIGTERM)
    try:
        return child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        return child.wait()


def _launch_stage(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stage = payload.get("stage")
    return str(stage) if isinstance(stage, str) else None


def _valid_collection_payload(payload: dict[str, Any]) -> bool:
    """Accept only a fully completed, update-free, safety-clean collection receipt.

    The collection runtime deliberately has no SAC/actor/critic update, so its
    return value may be nonzero when training-only health checks are inapplicable.
    The explicit collection receipt is therefore the authority, not that return
    code nor Isaac's known shutdown status.
    """

    return bool(
        payload.get("COLLECTION_COMPLETED") is True
        and int(payload.get("accepted_transitions", -1)) == 3000
        and payload.get("PRECLOSE_COLLECTION_ONLY") is True
        and int(payload.get("sac_update_count", -1)) == 0
        and int(payload.get("student_optimizer_update_count", -1)) == 0
        and payload.get("vector_contract_pass") is True
        and payload.get("source_freeze_match") is True
        and payload.get("boundary_paired_collection", {}).get("enabled") is True
        and all(
            int(payload.get(name, -1)) == 0
            for name in (
                "ACTION_BOUND_VIOLATION",
                "GRIPPER_AUTHORITY_VIOLATION",
                "RUNTIME_HARDSTOP",
                "FORBIDDEN_COLLISION",
                "failure_event_count",
            )
        )
    )


def _recover_completed_batches(root: Path, *, batch_target: int, seed_base: int) -> list[dict[str, Any]]:
    """Recover only contiguous, report-attested logical batches for resume."""

    recovered: list[dict[str, Any]] = []
    for index in range(1, batch_target + 1):
        valid: list[tuple[int, Path, dict[str, Any]]] = []
        for report in sorted(root.glob(f"batch-{index:02d}-attempt-*/REPORT.json")):
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                attempt = int(report.parent.name.rsplit("-attempt-", 1)[1])
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if _valid_collection_payload(payload):
                valid.append((attempt, report, payload))
        if not valid:
            break
        if len(valid) != 1:
            raise RuntimeError(f"RESUME_AMBIGUOUS_VALID_BATCH:{index}")
        attempt, report, payload = valid[0]
        recovered.append({
            "batch": index,
            "seed": seed_base + index - 1,
            "accepted_transitions": int(payload["accepted_transitions"]),
            "collection_completed": True,
            "recovered": True,
            "output_dir": str(report.parent.resolve()),
            "report": str(report.resolve()),
            "attempts": [{
                "attempt": attempt,
                "report": str(report.resolve()),
                "output_dir": str(report.parent.resolve()),
                "collection_completed": True,
                "collection_receipt_valid": True,
                "recovered": True,
            }],
        })
    return recovered


def _run_child(
    command: list[str], *, stdout: Path, stderr: Path, timeout_s: int,
    launch_marker: Path, startup_timeout_s: int, completion_report: Path | None = None,
    post_completion_shutdown_grace_s: int = 30,
    env_construct_timeout_s: int | None = None,
) -> tuple[int, bool, bool, bool, str | None]:
    """Run one new top-level process and detect AppLauncher-only futex hangs early."""

    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        completed_receipt_seen_at: float | None = None
        while child.poll() is None:
            elapsed_s = time.monotonic() - started
            stage = _launch_stage(launch_marker)
            if stage == "PRE_APP_LAUNCH" and elapsed_s >= startup_timeout_s:
                return _terminate_process_group(child), False, True, False, stage
            # This optional bound is deliberately narrower than the global
            # runtime timeout.  It is used by diagnostics that add renderer
            # AOVs and need a fresh-child retry when Kit never leaves scene
            # construction.  Existing collection paths leave it disabled.
            if (
                stage == "PRE_ENV_CONSTRUCT"
                and env_construct_timeout_s is not None
                and elapsed_s >= env_construct_timeout_s
            ):
                return _terminate_process_group(child), False, True, False, stage
            if completion_report is not None and completion_report.is_file():
                try:
                    payload = json.loads(completion_report.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                if _valid_collection_payload(payload):
                    if completed_receipt_seen_at is None:
                        completed_receipt_seen_at = time.monotonic()
                    elif time.monotonic() - completed_receipt_seen_at >= post_completion_shutdown_grace_s:
                        # The result is already durable and contract-checked;
                        # only Kit shutdown remains stuck.  Terminate this
                        # child session, never global CUDA/IPC state.
                        return _terminate_process_group(child), False, False, True, stage
            if elapsed_s >= timeout_s:
                return _terminate_process_group(child), True, False, False, stage
            time.sleep(0.5)
        return int(child.returncode), False, False, False, _launch_stage(launch_marker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--boundary-paired-plan", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260928)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--startup-timeout-s", type=int, default=90)
    parser.add_argument("--startup-attempts", type=int, default=3)
    parser.add_argument("--post-completion-shutdown-grace-s", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path("/data/fain-data/test/genie_sim_isaaclab3_sim601/bin/python3.12"),
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    catalog = args.source_catalog.resolve()
    plan = args.boundary_paired_plan.resolve()
    python = args.runtime_python.resolve()
    if (
        args.batches != 10
        or args.timeout_s <= 0
        or args.startup_timeout_s <= 0
        or args.post_completion_shutdown_grace_s <= 0
        or args.startup_attempts != 3
        or (root.exists() and not args.resume)
        or (not root.exists() and args.resume)
        or not catalog.is_file()
        or not plan.is_file()
        or not python.is_file()
    ):
        raise SystemExit("BOUNDARY_BATCH_SUPERVISOR_CONTRACT_INVALID")
    if root.exists():
        (root / "logs").mkdir(exist_ok=True)
        completed = _recover_completed_batches(
            root, batch_target=args.batches, seed_base=args.seed_base
        )
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir()
        completed = []
    report_path = root / "SUPERVISOR_REPORT.json"
    for index in range(len(completed) + 1, args.batches + 1):
        record: dict[str, Any] = {
            "batch": index,
            "seed": args.seed_base + index - 1,
            "accepted_transitions": 0,
            "attempts": [],
        }
        prior_attempt_numbers: list[int] = []
        for path in root.glob(f"batch-{index:02d}-attempt-*"):
            try:
                prior_attempt_numbers.append(int(path.name.rsplit("-attempt-", 1)[1]))
            except ValueError:
                continue
        first_attempt = max(prior_attempt_numbers, default=0) + 1
        for attempt in range(first_attempt, args.startup_attempts + 1):
            batch, report, stdout, stderr, launch_marker = _paths(root, index, attempt)
            command = [
                str(python), str(RUNNER),
                "--execute-live",
                "--num-envs", "10",
                "--accepted-transitions", "3000",
                "--runtime-variant", VARIANT,
                "--preclose-collection-only",
                "--preclose-source-catalog", str(catalog),
                "--boundary-paired-plan", str(plan),
                "--output-dir", str(batch),
                "--report", str(report),
                "--seed", str(args.seed_base + index - 1),
            ]
            started = time.time()
            exit_code, timed_out, startup_timeout, shutdown_timeout, launch_stage = _run_child(
                command,
                stdout=stdout,
                stderr=stderr,
                timeout_s=args.timeout_s,
                launch_marker=launch_marker,
                startup_timeout_s=args.startup_timeout_s,
                completion_report=report,
                post_completion_shutdown_grace_s=args.post_completion_shutdown_grace_s,
            )
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "startup_timeout": startup_timeout,
                "post_completion_shutdown_timeout": shutdown_timeout,
                "launch_stage": launch_stage,
                "elapsed_s": time.time() - started,
                "output_dir": str(batch),
                "report": str(report),
                "stdout": str(stdout),
                "stderr": str(stderr),
                "launch_receipt": str(launch_marker),
            }
            if report.is_file():
                payload = json.loads(report.read_text(encoding="utf-8"))
                attempt_record["accepted_transitions"] = int(payload.get("accepted_transitions", 0))
                attempt_record["collection_completed"] = bool(payload.get("COLLECTION_COMPLETED", False))
                attempt_record["collection_receipt_valid"] = _valid_collection_payload(payload)
            record["attempts"].append(attempt_record)
            if attempt_record.get("collection_receipt_valid", False):
                record.update(attempt_record)
                record["accepted_transitions"] = int(attempt_record["accepted_transitions"])
                break
            # Only the known AppLauncher constructor hang is retryable.  A
            # failed run after startup remains fail-closed to protect dataset
            # provenance and mechanics contracts.
            if not startup_timeout:
                break
        completed.append(record)
        _atomic_json(report_path, {
            "schema": "g2_stage1a_boundary_paired_batch_supervisor_v1",
            "BATCH_TARGET": args.batches,
            "BATCHES_COMPLETED": len(completed),
            "TOTAL_TARGET_TRANSITIONS": args.batches * 3000,
            "TOTAL_ACCEPTED_TRANSITIONS": sum(item["accepted_transitions"] for item in completed),
            "SAC_UPDATE": 0,
            "STUDENT_OPTIMIZER_UPDATE": 0,
            "PRIVILEGED_HARD_GATE": "NO",
            "WANDB_TRAINING_RUN": "NO",
            "batches": completed,
            "STATUS": (
                "RUNNING" if record.get("collection_completed", False) and index < args.batches
                else "COMPLETED" if record.get("collection_completed", False)
                else "FAILED"
            ),
        })
        print(
            "[BOUNDARY_BATCH_PROGRESS] "
            f"batch={index}/{args.batches} accepted={record['accepted_transitions']}/3000 "
            f"total={sum(item['accepted_transitions'] for item in completed)}/{args.batches * 3000} "
            f"attempts={len(record['attempts'])}",
            flush=True,
        )
        if not record.get("collection_receipt_valid", False):
            return 2
        sidecars = [
            str(Path(str(item["output_dir"])) / "CLOSE_READINESS_PRE_CLOSE_ROWS.jsonl")
            for item in completed
        ]
        audit_output = root / f"BOUNDARY_PAIRED_AUDIT_AFTER_BATCH_{index:02d}.json"
        audit_command = [str(python), str(AUDIT)]
        for sidecar in sidecars:
            audit_command.extend(("--sidecar", sidecar))
        audit_command.extend(("--output", str(audit_output)))
        audit_exit, audit_timeout, _audit_startup_timeout, _audit_shutdown_timeout, _audit_stage = _run_child(
            audit_command,
            stdout=root / "logs" / f"audit-{index:02d}.stdout.log",
            stderr=root / "logs" / f"audit-{index:02d}.stderr.log",
            timeout_s=300,
            launch_marker=root / "audit-launch-marker-not-app.json",
            startup_timeout_s=301,
        )
        if audit_exit != 0 or audit_timeout:
            return 3
        try:
            audit_payload = json.loads(audit_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 3
        supervisor_payload = json.loads(report_path.read_text(encoding="utf-8"))
        supervisor_payload.update({
            "TOTAL_BATCHES": len(completed),
            "SOURCE_FAMILY_COUNT": audit_payload.get("SOURCE_FAMILY_COUNT"),
            "POSITIVE_COUNT": audit_payload.get("POSITIVE_COUNT"),
            "NEGATIVE_COUNT": audit_payload.get("NEGATIVE_COUNT"),
            "TRAIN_POS_NEG": audit_payload.get("TRAIN_POS_NEG"),
            "VALIDATION_POS_NEG": audit_payload.get("VALIDATION_POS_NEG"),
            "HELDOUT_POS_NEG": audit_payload.get("HELDOUT_POS_NEG"),
            "SOURCE_FAMILY_DISJOINT_SPLIT": audit_payload.get("SOURCE_FAMILY_DISJOINT_SPLIT"),
            "OFFLINE_CONTRACT_PASS": bool(audit_payload.get("OFFLINE_CONTRACT_PASS", False)),
            "3K_AUTHORIZED": audit_payload.get("3K_AUTHORIZED", "NO"),
            "LATEST_AUDIT": str(audit_output),
        })
        if bool(audit_payload.get("OFFLINE_CONTRACT_PASS", False)):
            frozen_dir = root / "FROZEN_STUDENT"
            finalize_command = [
                str(python), str(FINALIZER), "--audit", str(audit_output),
                "--output-dir", str(frozen_dir),
            ]
            for sidecar in sidecars:
                finalize_command.extend(("--sidecar", sidecar))
            finalize_exit, finalize_timeout, _finalize_startup_timeout, _finalize_shutdown_timeout, _finalize_stage = _run_child(
                finalize_command,
                stdout=root / "logs" / f"finalize-{index:02d}.stdout.log",
                stderr=root / "logs" / f"finalize-{index:02d}.stderr.log",
                timeout_s=300,
                launch_marker=root / "finalize-launch-marker-not-app.json",
                startup_timeout_s=301,
            )
            freeze_report = frozen_dir / "FREEZE_REPORT.json"
            try:
                freeze_payload = json.loads(freeze_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                freeze_payload = {}
            if (
                finalize_exit != 0
                or finalize_timeout
                or freeze_payload.get("STUDENT_HEAD_FROZEN") != "YES"
            ):
                supervisor_payload.update({
                    "STATUS": "QUALITY_PASS_FREEZE_FAILED",
                    "FROZEN_STUDENT_REPORT": str(freeze_report),
                })
                _atomic_json(report_path, supervisor_payload)
                return 4
            supervisor_payload.update({
                "STATUS": "CONTRACT_SATISFIED_EARLY_STOP",
                "STUDENT_HEAD_FROZEN": "YES",
                "FROZEN_STUDENT_REPORT": str(freeze_report),
                "COLLECTION_STOP_REASON": "QUALITY_PASS",
            })
            _atomic_json(report_path, supervisor_payload)
            print(
                f"[BOUNDARY_BATCH_STOP] contract=PASS batches={len(completed)}/{args.batches}",
                flush=True,
            )
            return 0
        _atomic_json(report_path, supervisor_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
