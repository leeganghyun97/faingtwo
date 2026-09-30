#!/usr/bin/env python3
"""Collect an immutable, independent contact-free rollout set.

Each seed is executed in a fresh child process through the already reviewed
post-reset runtime-replan path.  The script deliberately does not retry a
failed seed: a partial set is reported as incomplete instead of being
silently promoted.  The child owns Isaac/action semantics; this parent only
owns process isolation, hashes, and aggregation.

The source reset authority currently varies cube XY only.  It does not add
robot-joint randomisation, and the manifest records that limitation explicitly
so a 30-seed result is not misreported as initial-state generalisation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py"
CANDIDATE = ROOT / "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
PRODUCTION = ROOT / "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
FROZEN_TRAJECTORY = ROOT / "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
BASELINE_REPORT = ROOT / "artifacts/g2_canonical_contact_free_bc_run003_20260922/TRAINING_REPORT.json"
EXPECTED_CANDIDATE = "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
EXPECTED_PRODUCTION = "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
EXPECTED_TRAJECTORY = "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"
POLICY_DT_S = 0.020
PHYSICS_DT_S = 0.002

OBSERVATION_SCHEMA_VERSION = "g2_canonical_contact_free_collection_v2"
ACTION_SCHEMA_VERSION = "g2_contact_free_metric_root_4d_v1"
FRAME_CONTRACT_VERSION = "g2_robot_root_xyzw_v1"
BASELINE_VARIANT = "LEGACY_CANDIDATE_A_LEFT_ARM_DOWN_V2"
BASELINE_MANIFEST = ROOT / "artifacts/g2_candidate_a_left_arm_down_v2/BASELINE_MANIFEST.json"
EXPECTED_BASELINE_MANIFEST = "cfd930feebb152c0a583b3bc418a5f4982111959814dea491c0720e57e2dfea3"
MONITORED_COLLECTION_SOURCES = (
    Path("scripts/run_g2_independent_contact_free_collection.py"),
    Path("scripts/diagnostics/run_g2_curobo_planner_live_smoke.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/canonical_contact_free_collection_v2.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/independent_contact_free_collection.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/contact_free_runtime_replan.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/curobo_planner_authority.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/contact_free_candidate_a_binding.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/training_env_factory.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/candidate_a_left_arm_down_v2.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/precontact_contract.py"),
    Path("source/geniesim/rl/isaaclab/g2_policy_branch/production_metric_adapter.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_source_freeze() -> dict[str, str]:
    """Hash every source that can change contact-free collection semantics."""

    result: dict[str, str] = {}
    for relative in MONITORED_COLLECTION_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"SOURCE_FREEZE_MISSING:{relative}")
        result[str(relative)] = sha256(path)
    return result


def git_revision() -> str:
    """Return origin revision without requiring a .git dir in a frozen snapshot."""

    supplied = os.environ.get("G2_ORIGIN_GIT_COMMIT", "").strip()
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def terminate_group(child: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    child.wait(timeout=10)


def _known_finalization(exit_code: int, report: Path, stdout: str) -> bool:
    return (
        exit_code == -11
        and report.is_file()
        and stdout.splitlines().count("REPORT_SAVED") == 1
        and stdout.splitlines().count("APP_CLOSED") == 1
        and stdout.splitlines().count("ATEXIT_RETURN") == 1
    )


def _recover_orphaned_completed_run(
    *, output_root: Path, episode_id: str, seed: int, capture_rate_hz: int
) -> dict[str, Any] | None:
    """Recover a completed child whose parent died before writing its receipt.

    This never retries the seed and never upgrades an unknown OS exit into a
    passing process verdict.  Functional publication is independently checked
    from the immutable report and HDF5 artifact.
    """

    run_dir = output_root / episode_id
    functional = run_dir / "FUNCTIONAL_RESULT_PRECONTACT_RUNTIME_REPLAN.json"
    dataset = run_dir / "CANONICAL_CONTACT_FREE_ROWS.hdf5"
    partial_safe_dataset = run_dir / "PARTIAL_SAFE_ROWS.hdf5"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    required = (functional, stdout_path, stderr_path)
    if not all(path.is_file() for path in required):
        return None
    try:
        payload = json.loads(functional.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    verdict = payload.get("verdict", {}) if isinstance(payload, dict) else {}
    collection = payload.get("canonical_contact_free_collection", {}) if isinstance(payload, dict) else {}
    if not isinstance(collection, dict):
        return None
    collection_state = collection.get("state")
    if collection_state == "PUBLISHED":
        if not dataset.is_file() or partial_safe_dataset.exists():
            return None
        published_artifact = dataset
    elif collection_state == "PARTIAL_SAFE_PUBLISHED":
        if not partial_safe_dataset.is_file() or dataset.exists():
            return None
        published_artifact = partial_safe_dataset
    else:
        return None
    if collection.get("sha256") != sha256(published_artifact):
        return None
    functional_pass = bool(
        verdict.get("CANDIDATE_A_CONTACT_FREE_SMOKE") == "PASS"
        and verdict.get("CANONICAL_CONTACT_FREE_COLLECTION") == "PASS"
        and verdict.get("FUNCTIONAL_VERDICT") == "PASS"
        and collection.get("state") == "PUBLISHED"
        and published_artifact == dataset
        and int(collection.get("policy_rate_hz", 0)) == 50
        and int(collection.get("capture_rate_hz", 0)) == capture_rate_hz
        and int(collection.get("capture_control_step_stride", 0)) == 50 // capture_rate_hz
        and payload.get("replay_close_row_count") == 0
        and payload.get("replay_contact_row_count") == 0
        and payload.get("replay_bc_row_count") == 0
    )
    receipt = {
        "schema": "g2_independent_contact_free_collection_run_v1",
        "episode_id": episode_id,
        "seed": seed,
        "command": ["RECOVERED_ORPHANED_CHILD_ARTIFACT"],
        "elapsed_seconds": None,
        "child_exit_code": None,
        "timed_out": False,
        "functional_pass": functional_pass,
        "process_verdict": "UNKNOWN_ORPHANED_EXIT",
        "recovery_provenance": "PARENT_DIED_AFTER_CHILD_ARTIFACT_PUBLISH",
        "functional_report": str(functional),
        "functional_report_sha256": sha256(functional),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset) if dataset.is_file() else None,
        "partial_safe_dataset": (
            str(partial_safe_dataset) if partial_safe_dataset.is_file() else None
        ),
        "partial_safe_dataset_sha256": (
            sha256(partial_safe_dataset) if partial_safe_dataset.is_file() else None
        ),
        "partial_safe_eligible": bool(
            collection_state == "PARTIAL_SAFE_PUBLISHED"
            and collection.get("partial_safe", {}).get("eligible") is True
        ),
        "sidecar": str(run_dir / "RUNTIME_REPLAN_PLAN_SIDECAR.json"),
        "sidecar_sha256": sha256(run_dir / "RUNTIME_REPLAN_PLAN_SIDECAR.json") if (run_dir / "RUNTIME_REPLAN_PLAN_SIDECAR.json").is_file() else None,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "row_count": int(collection.get("row_count", 0)),
        "max_action_norm_m": payload.get("maximum_final_metric_action_m"),
        "capture_rate_hz": capture_rate_hz,
        "control_rate_hz": 50,
        "capture_control_step_stride": 50 // capture_rate_hz,
        "contact_rows": payload.get("replay_contact_row_count"),
        "close_rows": payload.get("replay_close_row_count"),
        "source_freeze_complete": bool(payload.get("source_freeze_complete") is True),
    }
    atomic_json(run_dir / "RUN_RECEIPT.json", receipt)
    return receipt


def _run_one(*, output_root: Path, episode_id: str, seed: int, python: str, timeout_s: float, capture_rate_hz: int) -> dict[str, Any]:
    run_dir = output_root / episode_id
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    functional = run_dir / "FUNCTIONAL_RESULT_PRECONTACT_RUNTIME_REPLAN.json"
    dataset = run_dir / "CANONICAL_CONTACT_FREE_ROWS.hdf5"
    partial_safe_dataset = run_dir / "PARTIAL_SAFE_ROWS.hdf5"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = [
        python,
        str(CHILD),
        "--execute-live",
        "--seed", str(seed),
        "--output", str(functional),
        "--episode-horizon-steps", "817",
        "--ordinary-tracking-cap", "6",
        "--critical-tracking-cap", "20",
        "--final-settle-cap", "20",
        "--planner-active-velocity-limit-rad-s", "0.8",
        "--diagnostic-asset-variant", "custom",
        "--diagnostic-asset-path", str(CANDIDATE.resolve()),
        "--diagnostic-asset-sha256", EXPECTED_CANDIDATE,
        "--candidate-a-contact-free-runtime-replan",
        "--runtime-replan-canonical-contact-free-collection-output", str(dataset),
        "--canonical-capture-rate-hz", str(capture_rate_hz),
        "--independent-collection-episode-id", episode_id,
    ]
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(command, cwd=str(ROOT), stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            exit_code = child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(child)
            exit_code = child.returncode
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    payload: dict[str, Any] | None = None
    if functional.is_file():
        try:
            payload = json.loads(functional.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    verdict = payload.get("verdict", {}) if payload else {}
    collection = payload.get("canonical_contact_free_collection", {}) if payload else {}
    functional_pass = bool(
        payload
        and verdict.get("CANDIDATE_A_CONTACT_FREE_SMOKE") == "PASS"
        and verdict.get("CANONICAL_CONTACT_FREE_COLLECTION") == "PASS"
        and verdict.get("FUNCTIONAL_VERDICT") == "PASS"
        and collection.get("state") == "PUBLISHED"
        and dataset.is_file()
        and collection.get("sha256") == sha256(dataset)
        and int(collection.get("policy_rate_hz", 0)) == 50
        and int(collection.get("capture_rate_hz", 0)) == capture_rate_hz
        and int(collection.get("capture_control_step_stride", 0)) == 50 // capture_rate_hz
        and payload.get("replay_close_row_count") == 0
        and payload.get("replay_contact_row_count") == 0
        and payload.get("replay_bc_row_count") == 0
    )
    process_verdict = (
        "FAIL_TIMEOUT" if timed_out else
        "FAIL_KNOWN_ISAAC_FINALIZATION_SIGSEGV_-11" if _known_finalization(exit_code, functional, stdout_text) else
        "PASS" if exit_code == 0 else "FAIL_OTHER"
    )
    receipt = {
        "schema": "g2_independent_contact_free_collection_run_v1",
        "episode_id": episode_id,
        "seed": seed,
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "child_exit_code": exit_code,
        "timed_out": timed_out,
        "functional_pass": functional_pass,
        "process_verdict": process_verdict,
        "functional_report": str(functional),
        "functional_report_sha256": sha256(functional) if functional.is_file() else None,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset) if dataset.is_file() else None,
        "partial_safe_dataset": (
            str(partial_safe_dataset) if partial_safe_dataset.is_file() else None
        ),
        "partial_safe_dataset_sha256": (
            sha256(partial_safe_dataset) if partial_safe_dataset.is_file() else None
        ),
        "partial_safe_eligible": bool(
            isinstance(collection, dict)
            and collection.get("state") == "PARTIAL_SAFE_PUBLISHED"
            and collection.get("partial_safe", {}).get("eligible") is True
        ),
        "sidecar": str(run_dir / "RUNTIME_REPLAN_PLAN_SIDECAR.json"),
        "sidecar_sha256": sha256(run_dir / "RUNTIME_REPLAN_PLAN_SIDECAR.json") if (run_dir / "RUNTIME_REPLAN_PLAN_SIDECAR.json").is_file() else None,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "row_count": int(collection.get("row_count", 0)) if isinstance(collection, dict) else 0,
        "max_action_norm_m": payload.get("maximum_final_metric_action_m") if payload else None,
        "capture_rate_hz": capture_rate_hz,
        "control_rate_hz": 50,
        "capture_control_step_stride": 50 // capture_rate_hz,
        "contact_rows": payload.get("replay_contact_row_count") if payload else None,
        "close_rows": payload.get("replay_close_row_count") if payload else None,
        "source_freeze_complete": bool(payload and payload.get("source_freeze_complete") is True),
    }
    atomic_json(run_dir / "RUN_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--capture-rate-hz", type=int, choices=(25,), default=25)
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="complete every scheduled independent attempt while retaining failed attempts; never retries a seed",
    )
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=20.0,
        help="bounded pause between fresh Isaac child processes; prevents startup races after native finalization",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="required to launch child processes")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume an existing interrupted collection root; recorded episodes are never retried",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("COLLECTION_EXECUTION_REQUIRES_EXPLICIT_--execute")
    if args.count not in (30, 35):
        raise SystemExit("INDEPENDENT_COLLECTION_REQUIRES_30_OR_35_RUNS")
    if args.seed_base < 0:
        raise SystemExit("SEED_BASE_MUST_BE_NONNEGATIVE")
    for path, expected in ((CANDIDATE, EXPECTED_CANDIDATE), (PRODUCTION, EXPECTED_PRODUCTION), (FROZEN_TRAJECTORY, EXPECTED_TRAJECTORY), (BASELINE_MANIFEST, EXPECTED_BASELINE_MANIFEST), (CHILD, None)):
        if not path.is_file() or (expected is not None and sha256(path) != expected):
            raise SystemExit(f"SOURCE_FREEZE_FAIL:{path}")
    initial_collection_sources = collection_source_freeze()
    initial_git_revision = git_revision()
    source_snapshot_id = os.environ.get("G2_PREGRASP_SOURCE_SNAPSHOT_ID", "UNSNAPSHOTTED")
    root = args.output_root.expanduser().resolve()
    schedule = [{"episode_id": f"episode-{i:05d}", "seed": args.seed_base + i} for i in range(args.count)]
    if root.exists():
        if not args.resume_existing:
            raise SystemExit("OUTPUT_ROOT_MUST_BE_NEW_IMMUTABLE_DIRECTORY")
        manifest_path = root / "COLLECTION_MANIFEST.json"
        if not manifest_path.is_file():
            raise SystemExit("RESUME_MANIFEST_MISSING")
        manifest = json.loads(manifest_path.read_text())
        expected_source = {"candidate_sha256": sha256(CANDIDATE), "production_sha256": sha256(PRODUCTION), "frozen_trajectory_sha256": sha256(FROZEN_TRAJECTORY), "child_sha256": sha256(CHILD)}
        if manifest.get("source_freeze") != expected_source:
            raise SystemExit("RESUME_SOURCE_FREEZE_MISMATCH")
        if manifest.get("collection_source_freeze") != initial_collection_sources:
            raise SystemExit("RESUME_COLLECTION_SOURCE_FREEZE_MISMATCH")
        if manifest.get("git_commit") != initial_git_revision:
            raise SystemExit("RESUME_GIT_REVISION_MISMATCH")
        if manifest.get("source_snapshot_id") != source_snapshot_id:
            raise SystemExit("RESUME_SOURCE_SNAPSHOT_ID_MISMATCH")
        if int(manifest.get("capture_rate_hz", -1)) != int(args.capture_rate_hz):
            raise SystemExit("RESUME_CAPTURE_RATE_MISMATCH")
        if manifest.get("schedule") != schedule:
            raise SystemExit("RESUME_SCHEDULE_MISMATCH")
        recorded = {r.get("episode_id") for r in manifest.get("runs", [])}
        schedule = [item for item in schedule if item["episode_id"] not in recorded]
        # A host restart can orphan a completed Isaac child after it publishes
        # the report/HDF5 but before this parent appends RUN_RECEIPT.  Recover
        # only fully published artifacts; an incomplete directory remains a
        # fail-closed resume error rather than an overwrite/retry.
        remaining: list[dict[str, Any]] = []
        for item in schedule:
            run_dir = root / item["episode_id"]
            if not run_dir.exists():
                remaining.append(item)
                continue
            recovered = _recover_orphaned_completed_run(
                output_root=root,
                episode_id=item["episode_id"],
                seed=item["seed"],
                capture_rate_hz=args.capture_rate_hz,
            )
            if recovered is None:
                raise SystemExit(f"RESUME_UNRECORDED_INCOMPLETE_RUN:{run_dir}")
            manifest["runs"].append(recovered)
        schedule = remaining
        # The inter-child pause is operational metadata, not a dataset
        # schema/property.  Record the explicitly requested value on resume
        # so the manifest describes the cooldown used for every subsequent
        # child without rewriting any completed receipt.
        manifest["cooldown_s"] = float(args.cooldown_s)
        manifest["status"] = "RUNNING"
        manifest.pop("abort_reason", None)
        atomic_json(manifest_path, manifest)
    else:
        root.mkdir(parents=True)
        manifest = None
    if manifest is None:
        manifest = {
        "schema": "g2_independent_contact_free_collection_v1",
        "dataset_session_id": root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": initial_git_revision,
        "worktree_revision": f"{initial_git_revision}:{hashlib.sha256(json.dumps(initial_collection_sources, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
        "source_snapshot_id": source_snapshot_id,
        "baseline_variant": BASELINE_VARIANT,
        "baseline_manifest": str(BASELINE_MANIFEST),
        "baseline_manifest_sha256": EXPECTED_BASELINE_MANIFEST,
        "execution_mode": "SEQUENTIAL_FRESH_CHILD_NO_RETRY",
        "cooldown_s": float(args.cooldown_s),
        "capture_rate_hz": int(args.capture_rate_hz),
        "continue_after_failure": bool(args.continue_after_failure),
        "source_freeze": {"candidate_sha256": sha256(CANDIDATE), "production_sha256": sha256(PRODUCTION), "frozen_trajectory_sha256": sha256(FROZEN_TRAJECTORY), "child_sha256": sha256(CHILD)},
        "collection_source_freeze": initial_collection_sources,
        "contract": {"observation_schema_version": OBSERVATION_SCHEMA_VERSION, "action_schema_version": ACTION_SCHEMA_VERSION, "frame_contract_version": FRAME_CONTRACT_VERSION, "action_dim": 4, "action": "[dx,dy,dz,HOLD_OPEN=0]", "frame": "robot_root", "ee_frame": "gripper_r_center_link", "wrist_camera_frame": "right_wrist_camera_authored_frame", "cube_frame": "cube_center", "unit": "m", "depth_unit": "m", "angle_unit": "rad", "right_arm_state_order": [f"idx{index}_arm_r_joint{index - 60}" for index in range(61, 68)], "policy_dt_s": POLICY_DT_S, "policy_hz": 50, "capture_rate_hz": int(args.capture_rate_hz), "capture_control_step_stride": 50 // int(args.capture_rate_hz), "physics_dt_s": PHYSICS_DT_S, "physics_hz": 500, "camera_timestamp_source": "actual_acquisition_timestamp_preserved_on_reuse", "contact": False, "close": False, "no_cube_gt_in_actor_rows": True, "privileged_planner_state_scope": "SIDE_CAR_ONLY_NOT_STUDENT_OBSERVATION", "single_consumption": {"env_step": 1, "process_action": 1, "controller": 1}},
        "reset_distribution": {"authority": "source_owned_reset_object_position", "cube_xy_m": [[0.49, 0.51], [-0.24, -0.22]], "robot_initial_state": "LEGACY_CANDIDATE_A_LEFT_ARM_DOWN_V2_COMMON_RESET", "left_arm_baseline_source": "NEW_COMMON_BASELINE", "generalization_scope": "cube_xy_reset_seed_variation_only"},
        "baseline_reference": {
            "report": str(BASELINE_REPORT),
            "report_sha256": sha256(BASELINE_REPORT) if BASELINE_REPORT.is_file() else None,
            "validation_xyz_rmse_mm": 2.5589,
            "validation_direction_error_deg": 55.8358,
            "validation_split": "single_rollout_temporal_tail_90_10",
            "interpretation": "reference_only_not_independent_generalization",
        },
        "schedule": schedule,
        "runs": [],
        "status": "RUNNING",
        }
        atomic_json(root / "COLLECTION_MANIFEST.json", manifest)
    for item in schedule:
        if collection_source_freeze() != initial_collection_sources:
            manifest["status"] = "INCOMPLETE_SOURCE_FREEZE_MISMATCH"
            manifest["abort_reason"] = "COLLECTION_SOURCE_CHANGED_DURING_SESSION"
            atomic_json(root / "COLLECTION_MANIFEST.json", manifest)
            break
        receipt = _run_one(output_root=root, episode_id=item["episode_id"], seed=item["seed"], python=args.python, timeout_s=args.timeout_s, capture_rate_hz=args.capture_rate_hz)
        manifest["runs"].append(receipt)
        atomic_json(root / "COLLECTION_MANIFEST.json", manifest)
        if not receipt["functional_pass"] and not args.continue_after_failure:
            # A failed independent episode cannot be silently replaced or
            # skipped: the requested 30-run set must be complete, and a
            # partial set is not eligible for BC/CV.  Stop immediately so a
            # bootstrap failure does not waste GPU time on more invalid
            # attempts.
            manifest["status"] = "INCOMPLETE_FAIL_CLOSED"
            manifest["abort_reason"] = "INDEPENDENT_EPISODE_FUNCTIONAL_FAIL_NO_RETRY"
            break
        # Isaac/Kit has a known post-finalization native shutdown defect.  A
        # bounded inter-child cooldown is part of the collection contract so
        # the next child does not race residual driver/extension teardown.
        if args.cooldown_s > 0 and item is not schedule[-1]:
            time.sleep(args.cooldown_s)
    passed = [r for r in manifest["runs"] if r["functional_pass"]]
    if manifest.get("status") not in {
        "INCOMPLETE_FAIL_CLOSED",
        "INCOMPLETE_SOURCE_FREEZE_MISMATCH",
    }:
        if len(manifest["runs"]) == args.count:
            manifest["status"] = "PASS" if len(passed) == args.count else "COMPLETE_WITH_FAILURES"
        else:
            manifest["status"] = "INCOMPLETE_FAIL_CLOSED"
    manifest["functional_pass_count"] = len(passed)
    manifest["functional_fail_count"] = len(manifest["runs"]) - len(passed)
    manifest["known_process_finalization_count"] = sum(r["process_verdict"] == "FAIL_KNOWN_ISAAC_FINALIZATION_SIGSEGV_-11" for r in manifest["runs"])
    manifest["total_rows"] = sum(int(r.get("row_count") or 0) for r in manifest["runs"])
    manifest["success_rows"] = sum(
        int(r.get("row_count") or 0) for r in manifest["runs"] if r["functional_pass"]
    )
    manifest["partial_safe_episode_count"] = sum(
        bool(r.get("partial_safe_eligible")) for r in manifest["runs"]
    )
    manifest["partial_safe_rows"] = sum(
        int(r.get("row_count") or 0)
        for r in manifest["runs"]
        if r.get("partial_safe_eligible")
    )
    atomic_json(root / "COLLECTION_MANIFEST.json", manifest)
    return 0 if manifest["status"] in {"PASS", "COMPLETE_WITH_FAILURES"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
