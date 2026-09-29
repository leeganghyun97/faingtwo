#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Collect exact privileged signed-margin boundary rows in fresh Isaac children.

This is deliberately *collection only*: every child is a fresh top-level
Python process, receives five source-family clone pairs from a frozen Candidate-A
catalog, and is allowed to emit only pre-CLOSE teacher rows.  The plan contains
geometry perturbations but no intended teacher label.  The unchanged live
oracle remains the only source of ``CLOSE_READY_BINARY`` and the physical
signed-margin target.

Three batches provide fifteen non-overlapping source families (five per 3K
batch, each replayed in two isolated clones) from the current 34-family catalog.  This is the smallest bounded plan
that adds real batch and provenance diversity without starting SAC, a student
optimizer, W&B training, or any runtime CLOSE gate experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from geniesim.rl.sac.stage1a_boundary_paired_collection import (
    SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
    build_signed_margin_paired_clone_plan,
    build_signed_margin_paired_clone_state_plan,
    build_signed_margin_paired_clone_state_wrist_plan,
    sha256,
)

# Reuse the process-group boundary and receipt validator used by the original
# V1 collection supervisor.  The helper is intentionally independent of Isaac
# imports; this module itself never launches Kit in-process.
from run_g2_stage1a_boundary_paired_collection_batches import (
    AUDIT,
    ROOT,
    RUNNER,
    _run_child,
    _valid_collection_payload,
)


VARIANT = "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K"
SCHEMA = "g2_stage1a_signed_margin_boundary_collection_supervisor_v3"
FAMILIES_PER_BATCH = 5
MAX_BATCHES = 3


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _catalog_family_ids(catalog: Path) -> tuple[str, ...]:
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    raw = payload.get("sources")
    if not isinstance(raw, list):
        raise ValueError("SIGNED_MARGIN_CATALOG_SOURCES_INVALID")
    ids = tuple(str(item.get("source_id")) for item in raw if isinstance(item, dict))
    if (
        len(ids) != len(raw)
        or len(set(ids)) != len(ids)
        or any(not item or item == "None" for item in ids)
        or len(ids) < FAMILIES_PER_BATCH * MAX_BATCHES
    ):
        raise ValueError("SIGNED_MARGIN_CATALOG_FAMILY_CAPACITY_INSUFFICIENT")
    catalog_hash = sha256(catalog)
    # Allocation is a deterministic function of frozen source identity only.
    # It cannot depend on live success/failure, a teacher target, or an actor.
    return tuple(sorted(
        ids,
        key=lambda item: hashlib.sha256(
            f"{catalog_hash}:{item}".encode("utf-8")
        ).hexdigest(),
    ))


def _allocation(catalog: Path, index: int) -> tuple[str, ...]:
    if not 1 <= index <= MAX_BATCHES:
        raise ValueError("SIGNED_MARGIN_BATCH_INDEX_OUT_OF_RANGE")
    ordered = _catalog_family_ids(catalog)
    start = (index - 1) * FAMILIES_PER_BATCH
    selected = ordered[start : start + FAMILIES_PER_BATCH]
    if len(selected) != FAMILIES_PER_BATCH:
        raise ValueError("SIGNED_MARGIN_BATCH_ALLOCATION_INSUFFICIENT")
    return selected


def _paths(root: Path, index: int, attempt: int) -> tuple[Path, Path, Path, Path, Path]:
    batch = root / f"batch-{index:02d}-attempt-{attempt:02d}"
    return (
        batch,
        batch / "REPORT.json",
        root / "logs" / f"batch-{index:02d}-attempt-{attempt:02d}.stdout.log",
        root / "logs" / f"batch-{index:02d}-attempt-{attempt:02d}.stderr.log",
        root / f"batch-{index:02d}-attempt-{attempt:02d}_launch.json",
    )


def _signed_receipt_valid(
    payload: dict[str, Any], *, family_ids: tuple[str, ...],
    causal_state_v4: bool = False, wrist_rgbd_v5: bool = False,
) -> bool:
    if causal_state_v4 and wrist_rgbd_v5:
        return False
    if not _valid_collection_payload(payload):
        return False
    boundary = payload.get("boundary_paired_collection")
    if not isinstance(boundary, dict):
        return False
    return bool(
        boundary.get("schema") == (
            SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA
            if wrist_rgbd_v5 else SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA
            if causal_state_v4 else SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA
        )
        and boundary.get("row_schema") == (
            SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA
            if wrist_rgbd_v5 else SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA
            if causal_state_v4 else SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA
        )
        and boundary.get("signed_margin_telemetry") is True
        and tuple(boundary.get("source_family_ids", ())) == tuple(
            family for family_id in family_ids for family in (family_id, family_id)
        )
        and boundary.get("paired_clone_allocation") is True
        and boundary.get("privileged_hard_gate") is False
        and boundary.get("student_privileged_input_count") == 0
        and bool(boundary.get("causal_student_state_receipt", False)) is (
            causal_state_v4 or wrist_rgbd_v5
        )
        and bool(boundary.get("wrist_rgbd_receipt", False)) is wrist_rgbd_v5
    )


def _recover(
    root: Path, *, seed_base: int, catalog: Path, batches: int,
    causal_state_v4: bool, wrist_rgbd_v5: bool,
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for index in range(1, batches + 1):
        expected = _allocation(catalog, index)
        valid: list[tuple[int, Path, dict[str, Any]]] = []
        for report in sorted(root.glob(f"batch-{index:02d}-attempt-*/REPORT.json")):
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                attempt = int(report.parent.name.rsplit("-attempt-", 1)[1])
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if _signed_receipt_valid(
                payload, family_ids=expected, causal_state_v4=causal_state_v4,
                wrist_rgbd_v5=wrist_rgbd_v5,
            ):
                valid.append((attempt, report, payload))
        if not valid:
            break
        if len(valid) != 1:
            raise RuntimeError(f"SIGNED_MARGIN_RESUME_AMBIGUOUS_VALID_BATCH:{index}")
        attempt, report, payload = valid[0]
        recovered.append({
            "batch": index,
            "seed": seed_base + index - 1,
            "accepted_transitions": int(payload["accepted_transitions"]),
            "source_family_ids": list(expected),
            "attempts": [{
                "attempt": attempt,
                "report": str(report.resolve()),
                "output_dir": str(report.parent.resolve()),
                "collection_receipt_valid": True,
                "recovered": True,
            }],
            "report": str(report.resolve()),
            "output_dir": str(report.parent.resolve()),
            "collection_receipt_valid": True,
            "recovered": True,
        })
    return recovered


def _run_audit(*, python: Path, root: Path, completed: list[dict[str, Any]], index: int) -> dict[str, Any]:
    sidecars = [
        str(Path(str(item["output_dir"])) / "CLOSE_READINESS_PRE_CLOSE_ROWS.jsonl")
        for item in completed
    ]
    output = root / f"SIGNED_MARGIN_BOUNDARY_AUDIT_AFTER_BATCH_{index:02d}.json"
    command = [str(python), str(AUDIT)]
    for sidecar in sidecars:
        command.extend(("--sidecar", sidecar))
    command.extend(("--output", str(output)))
    exit_code, timed_out, _startup, _shutdown, _stage = _run_child(
        command,
        stdout=root / "logs" / f"audit-{index:02d}.stdout.log",
        stderr=root / "logs" / f"audit-{index:02d}.stderr.log",
        timeout_s=300,
        launch_marker=root / "audit-launch-marker-not-app.json",
        startup_timeout_s=301,
    )
    if exit_code != 0 or timed_out or not output.is_file():
        raise RuntimeError("SIGNED_MARGIN_OFFLINE_AUDIT_FAILED")
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {"path": str(output.resolve()), "payload": payload}


def _write_supervisor_report(
    *, root: Path, catalog: Path, batches: int, completed: list[dict[str, Any]],
    status: str, latest_audit: dict[str, Any] | None, causal_state_v4: bool,
    wrist_rgbd_v5: bool,
) -> None:
    audit = (latest_audit or {}).get("payload", {})
    _atomic_json(root / "SUPERVISOR_REPORT.json", {
        "SCHEMA": "g2_stage1a_signed_margin_boundary_collection_supervisor_v5"
        if wrist_rgbd_v5 else "g2_stage1a_signed_margin_boundary_collection_supervisor_v4"
        if causal_state_v4 else SCHEMA,
        "COLLECTION_MODE": "AUTOMATIC_BOUNDARY_BALANCED_SIGNED_MARGIN_V5_PAIRED_CLONES_TIMESTAMP_ALIGNED_WRIST_RGBD"
        if wrist_rgbd_v5 else "AUTOMATIC_BOUNDARY_BALANCED_SIGNED_MARGIN_V4_PAIRED_CLONES_CAUSAL_STATE"
        if causal_state_v4 else "AUTOMATIC_BOUNDARY_BALANCED_SIGNED_MARGIN_V3_PAIRED_CLONES",
        "SOURCE_CATALOG": str(catalog),
        "SOURCE_CATALOG_SHA256": sha256(catalog),
        "BATCH_TARGET": batches,
        "FAMILIES_PER_BATCH": FAMILIES_PER_BATCH,
        "SOURCE_FAMILY_ALLOCATION": "FROZEN_CATALOG_HASH_DETERMINISTIC_NO_LABEL_LOOKUP",
        "BATCHES_COMPLETED": len(completed),
        "TOTAL_TARGET_TRANSITIONS": batches * 3000,
        "TOTAL_ACCEPTED_TRANSITIONS": sum(int(item["accepted_transitions"]) for item in completed),
        "SOURCE_FAMILY_COUNT_ALLOCATED": len({
            family for item in completed for family in item["source_family_ids"]
        }),
        "SAC_UPDATE": 0,
        "STUDENT_OPTIMIZER_UPDATE": 0,
        "PRIVILEGED_HARD_GATE": "NO",
        "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
        "WANDB_TRAINING_RUN": "NO",
        "CAUSAL_STUDENT_STATE_RECEIPT": "YES" if (causal_state_v4 or wrist_rgbd_v5) else "NO",
        "WRIST_RGBD_TIMESTAMP_ALIGNED_RECEIPT": "YES" if wrist_rgbd_v5 else "NO",
        "STUDENT_STATE_FORBIDDEN_INPUTS": [
            "cube_gt", "relative_pose_root_m", "privileged_geometry"
        ] if (causal_state_v4 or wrist_rgbd_v5) else [],
        "STATUS": status,
        "LATEST_AUDIT": (latest_audit or {}).get("path"),
        "CANONICAL_SIGNED_MARGIN_ROWS": audit.get("BOUNDARY_PAIRED_ROW_COUNT"),
        "POSITIVE_COUNT": audit.get("POSITIVE_COUNT"),
        "NEGATIVE_COUNT": audit.get("NEGATIVE_COUNT"),
        "USABLE_PAIRED_FAMILY_COUNT": audit.get("USABLE_PAIRED_FAMILY_COUNT"),
        "SOURCE_LABEL_CONFOUNDING": audit.get("SOURCE_LABEL_CONFOUNDING"),
        "SOURCE_FAMILY_DISJOINT_SPLIT": audit.get("SOURCE_FAMILY_DISJOINT_SPLIT"),
        "batches": completed,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--seed-base", type=int, default=20260929)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--startup-timeout-s", type=int, default=90)
    parser.add_argument("--startup-attempts", type=int, default=3)
    parser.add_argument("--post-completion-shutdown-grace-s", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--causal-state-v4", action="store_true",
        help="write V4 timestamp-aligned deployable robot-state receipts",
    )
    parser.add_argument(
        "--wrist-rgbd-v5", action="store_true",
        help="write V5 timestamp-aligned Wrist RGB-D HDF5 frame references",
    )
    parser.add_argument(
        "--runtime-python", type=Path,
        default=Path("/data/fain-data/test/genie_sim_isaaclab3_sim601/bin/python3.12"),
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    catalog = args.source_catalog.resolve()
    python = args.runtime_python.resolve()
    if (
        args.batches < 1 or args.batches > MAX_BATCHES
        or args.timeout_s <= 0 or args.startup_timeout_s <= 0
        or args.startup_attempts != 3 or args.post_completion_shutdown_grace_s <= 0
        or bool(args.causal_state_v4 and args.wrist_rgbd_v5)
        or not catalog.is_file() or not python.is_file()
        or (root.exists() and not args.resume)
        or (not root.exists() and args.resume)
    ):
        raise SystemExit("SIGNED_MARGIN_COLLECTION_SUPERVISOR_CONTRACT_INVALID")
    # Validate capacity and deterministic allocation before any directory is
    # created or child is launched.
    allocations = [_allocation(catalog, index) for index in range(1, args.batches + 1)]
    if len({family for allocation in allocations for family in allocation}) != args.batches * FAMILIES_PER_BATCH:
        raise SystemExit("SIGNED_MARGIN_SOURCE_FAMILY_REUSE_FORBIDDEN")
    if root.exists():
        (root / "logs").mkdir(exist_ok=True)
        (root / "plans").mkdir(exist_ok=True)
        completed = _recover(
            root, seed_base=args.seed_base, catalog=catalog, batches=args.batches,
            causal_state_v4=args.causal_state_v4,
            wrist_rgbd_v5=args.wrist_rgbd_v5,
        )
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir()
        (root / "plans").mkdir()
        completed = []
    latest_audit: dict[str, Any] | None = None
    if completed:
        latest_audit = _run_audit(
            python=python, root=root, completed=completed, index=len(completed)
        )
    for index in range(len(completed) + 1, args.batches + 1):
        family_ids = allocations[index - 1]
        plan_path = root / "plans" / f"batch-{index:02d}-SIGNED_MARGIN_PLAN.json"
        if plan_path.exists():
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            if tuple(plan_payload.get("source_family_unique_ids", ())) != family_ids:
                raise RuntimeError("SIGNED_MARGIN_EXISTING_PLAN_ALLOCATION_MISMATCH")
        else:
            comparison_probe = (
                "CONTAINMENT_ROOT_Y_PLUS_2MM"
                if index % 2 else "CONTAINMENT_ROOT_Y_MINUS_2MM"
            )
            plan_payload = (
                build_signed_margin_paired_clone_state_wrist_plan(
                    source_catalog=catalog,
                    source_family_ids=family_ids,
                    comparison_probe_variant=comparison_probe,
                ) if args.wrist_rgbd_v5 else build_signed_margin_paired_clone_state_plan(
                    source_catalog=catalog,
                    source_family_ids=family_ids,
                    comparison_probe_variant=comparison_probe,
                ) if args.causal_state_v4 else build_signed_margin_paired_clone_plan(
                    source_catalog=catalog,
                    source_family_ids=family_ids,
                    comparison_probe_variant=comparison_probe,
                )
            )
            plan_path.write_text(
                json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        record: dict[str, Any] = {
            "batch": index,
            "seed": args.seed_base + index - 1,
            "accepted_transitions": 0,
            "source_family_ids": list(family_ids),
            "plan": str(plan_path.resolve()),
            "plan_sha256": sha256(plan_path),
            "attempts": [],
        }
        existing_attempts: list[int] = []
        for existing in root.glob(f"batch-{index:02d}-attempt-*"):
            try:
                existing_attempts.append(int(existing.name.rsplit("-attempt-", 1)[1]))
            except ValueError:
                continue
        first_attempt = max(existing_attempts, default=0) + 1
        for attempt in range(first_attempt, args.startup_attempts + 1):
            batch, report, stdout, stderr, launch_marker = _paths(root, index, attempt)
            command = [
                str(python), str(RUNNER), "--execute-live",
                "--num-envs", "10", "--accepted-transitions", "3000",
                "--runtime-variant", VARIANT, "--preclose-collection-only",
                "--preclose-source-catalog", str(catalog),
                "--boundary-paired-plan", str(plan_path),
                "--output-dir", str(batch), "--report", str(report),
                "--seed", str(args.seed_base + index - 1),
            ]
            started = time.time()
            exit_code, timed_out, startup_timeout, shutdown_timeout, launch_stage = _run_child(
                command, stdout=stdout, stderr=stderr, timeout_s=args.timeout_s,
                launch_marker=launch_marker, startup_timeout_s=args.startup_timeout_s,
                completion_report=report,
                post_completion_shutdown_grace_s=args.post_completion_shutdown_grace_s,
            )
            attempt_record: dict[str, Any] = {
                "attempt": attempt, "exit_code": exit_code, "timed_out": timed_out,
                "startup_timeout": startup_timeout,
                "post_completion_shutdown_timeout": shutdown_timeout,
                "launch_stage": launch_stage,
                "elapsed_s": time.time() - started,
                "output_dir": str(batch.resolve()), "report": str(report.resolve()),
                "stdout": str(stdout.resolve()), "stderr": str(stderr.resolve()),
                "launch_receipt": str(launch_marker.resolve()),
            }
            if report.is_file():
                payload = json.loads(report.read_text(encoding="utf-8"))
                attempt_record["accepted_transitions"] = int(payload.get("accepted_transitions", 0))
                attempt_record["collection_receipt_valid"] = _signed_receipt_valid(
                    payload, family_ids=family_ids,
                    causal_state_v4=args.causal_state_v4,
                    wrist_rgbd_v5=args.wrist_rgbd_v5,
                )
            record["attempts"].append(attempt_record)
            if attempt_record.get("collection_receipt_valid", False):
                record.update(attempt_record)
                record["accepted_transitions"] = int(attempt_record["accepted_transitions"])
                break
            # A post-start failure is provenance-significant and must not be
            # retried into the same logical batch.  Only the known
            # pre-AppLauncher timeout gets a fresh-process retry.
            if not startup_timeout:
                break
        completed.append(record)
        if not record.get("collection_receipt_valid", False):
            _write_supervisor_report(
                root=root, catalog=catalog, batches=args.batches, completed=completed,
                status="FAILED", latest_audit=latest_audit,
                causal_state_v4=args.causal_state_v4,
                wrist_rgbd_v5=args.wrist_rgbd_v5,
            )
            print(f"[SIGNED_MARGIN_BATCH_FAILED] batch={index} attempts={len(record['attempts'])}", flush=True)
            return 2
        latest_audit = _run_audit(
            python=python, root=root, completed=completed, index=index
        )
        _write_supervisor_report(
            root=root, catalog=catalog, batches=args.batches, completed=completed,
            status="RUNNING" if index < args.batches else "COMPLETED",
            latest_audit=latest_audit,
            causal_state_v4=args.causal_state_v4,
            wrist_rgbd_v5=args.wrist_rgbd_v5,
        )
        audit = latest_audit["payload"]
        print(
            "[SIGNED_MARGIN_BATCH_PROGRESS] "
            f"batch={index}/{args.batches} accepted={record['accepted_transitions']}/3000 "
            f"families={len({family for item in completed for family in item['source_family_ids']})} "
            f"canonical={audit.get('BOUNDARY_PAIRED_ROW_COUNT')} "
            f"pos={audit.get('POSITIVE_COUNT')} neg={audit.get('NEGATIVE_COUNT')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
