#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded, actionless startup isolation for the Candidate-A OPEN audit.

The supervisor never imports Isaac.  It runs each live stage in one fresh
child process and records a durable, fsynced marker before every potentially
blocking boundary.  No stage submits an action or reaches the OPEN/CLOSE
controller.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_A_ASSET = ROOT / (
    "artifacts/g2_bounded_passive_range_qualification_20260921/"
    "candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
)
CANDIDATE_A_SHA256 = "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
SCHEMA = "g2_candidate_a_startup_isolation_v1"
STAGES = ("A_APP_LAUNCHER", "B_SIMULATION_CONTEXT", "C_PHYSICS_SCENE", "D_CANDIDATE_A_ARTICULATION", "E_OPEN_AUDIT_SCENE")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    """Convert diagnostic receipts without changing their runtime authority."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class _Journal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def mark(self, marker: str, **detail: Any) -> None:
        try:
            threads = len(list(Path(f"/proc/{os.getpid()}/task").iterdir()))
        except OSError:
            threads = threading.active_count()
        row = {
            "schema": SCHEMA,
            "marker": marker,
            "timestamp_utc": _utc(),
            "monotonic_s": time.monotonic(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "tid": threading.get_native_id(),
            "thread_count": threads,
            "cwd": os.getcwd(),
            "python": sys.executable,
            "argv": sys.argv,
            "detail": _jsonable(detail),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(marker, flush=True)


def _environment_snapshot() -> dict[str, Any]:
    selected = (
        "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX", "VIRTUAL_ENV",
        "DISPLAY", "XDG_SESSION_TYPE", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "OMNI_KIT_ACCEPT_EULA",
    )
    env = dict(os.environ)
    safe = {key: ("<REDACTED>" if any(word in key.upper() for word in ("TOKEN", "SECRET", "PASS", "KEY", "AUTH")) else value) for key, value in sorted(env.items())}
    limits: dict[str, list[int]] = {}
    for name in dir(resource):
        if name.startswith("RLIMIT_") and isinstance(getattr(resource, name), int):
            try:
                soft, hard = resource.getrlimit(getattr(resource, name))
                limits[name] = [int(soft), int(hard)]
            except (OSError, ValueError):
                pass
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":")).encode()
    return {
        "timestamp_utc": _utc(), "cwd": os.getcwd(), "python": sys.executable,
        "selected": {key: safe.get(key) for key in selected},
        "isaac_omni": {key: value for key, value in safe.items() if key.startswith(("ISAAC_", "OMNI_"))},
        "environment_fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "ulimit": limits,
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=5)
        return {"returncode": completed.returncode, "stdout": completed.stdout.splitlines(), "stderr": completed.stderr.strip()}
    except BaseException as error:
        return {"error": f"{type(error).__name__}:{error}"}


def _child(stage: str, output: Path) -> int:
    if output.exists():
        raise RuntimeError(f"CHILD_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=False)
    journal = _Journal(output / "MARKERS.jsonl")
    _atomic_json(output / "RUNTIME_ENVIRONMENT.json", _environment_snapshot())
    journal.mark("PROCESS_STARTED", stage=stage, gpu_before=_gpu_snapshot())
    app = sim = scene = env = None
    try:
        journal.mark("BEFORE_APPLAUNCHER_IMPORT")
        from isaaclab.app import AppLauncher
        journal.mark("AFTER_APPLAUNCHER_IMPORT")
        enable_cameras = stage == "E_OPEN_AUDIT_SCENE"
        journal.mark("BEFORE_APPLAUNCHER_CONSTRUCT", enable_cameras=enable_cameras)
        app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, fast_shutdown=False)
        app = app_launcher.app
        journal.mark("AFTER_APPLAUNCHER_CONSTRUCT", gpu_after_launcher=_gpu_snapshot())
        journal.mark("BEFORE_FIRST_APP_UPDATE")
        app.update()
        journal.mark("AFTER_FIRST_APP_UPDATE")
        if stage == "A_APP_LAUNCHER":
            return 0

        journal.mark("BEFORE_SIMULATION_CONTEXT_IMPORT")
        import isaaclab.sim as sim_utils
        journal.mark("AFTER_SIMULATION_CONTEXT_IMPORT")
        journal.mark("BEFORE_SIMULATION_CONTEXT_CREATE")
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device="cuda:0", render_interval=1))
        journal.mark("AFTER_SIMULATION_CONTEXT_CREATE")
        if stage == "B_SIMULATION_CONTEXT":
            return 0

        journal.mark("BEFORE_EMPTY_PHYSICS_RESET")
        sim.reset()
        journal.mark("AFTER_EMPTY_PHYSICS_RESET")
        journal.mark("BEFORE_FIRST_PHYSICS_STEP")
        sim.step(render=False)
        journal.mark("AFTER_FIRST_PHYSICS_STEP")
        if stage == "C_PHYSICS_SCENE":
            return 0

        if stage == "D_CANDIDATE_A_ARTICULATION":
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            from isaaclab.utils import configclass
            from geniesim.rl.isaaclab.g2_policy_branch.training_env_factory import (
                make_g2_candidate_a_left_arm_down_v2_training_env_cfg,
            )

            # Reuse the exact Candidate-A robot spawn configuration from the
            # audited full environment instead of approximating the USD's
            # articulation-root/collision-spawn semantics in this isolator.
            # This is an asset-load probe, not a new robot configuration.
            candidate_cfg, receipt, _contract = make_g2_candidate_a_left_arm_down_v2_training_env_cfg(num_envs=1)
            robot_cfg = candidate_cfg.scene.robot
            if Path(robot_cfg.spawn.usd_path).resolve() != CANDIDATE_A_ASSET.resolve():
                raise RuntimeError("CANDIDATE_A_FACTORY_AUTHORITY_MISMATCH")

            @configclass
            class CandidateASceneCfg(InteractiveSceneCfg):
                robot = robot_cfg

            journal.mark("BEFORE_CANDIDATE_A_SCENE_CREATE", asset_sha256=_sha256(CANDIDATE_A_ASSET), asset_receipt=receipt)
            scene = InteractiveScene(CandidateASceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True))
            journal.mark("AFTER_CANDIDATE_A_SCENE_CREATE")
            sim.reset()
            scene.reset()
            sim.step(render=False)
            scene.update(0.002)
            journal.mark("AFTER_CANDIDATE_A_FIRST_PHYSICS_STEP")
            return 0

        journal.mark("BEFORE_OPEN_AUDIT_SCENE_CONFIG")
        from isaaclab.envs import ManagerBasedRLEnv
        from geniesim.rl.isaaclab.g2_policy_branch.training_env_factory import make_g2_candidate_a_left_arm_down_v2_training_env_cfg
        cfg, receipt, contract = make_g2_candidate_a_left_arm_down_v2_training_env_cfg(num_envs=10)
        if Path(cfg.scene.robot.spawn.usd_path).resolve() != CANDIDATE_A_ASSET.resolve():
            raise RuntimeError("CANDIDATE_A_FACTORY_AUTHORITY_MISMATCH")
        journal.mark("AFTER_OPEN_AUDIT_SCENE_CONFIG", asset_receipt=receipt, contract=contract)
        journal.mark("BEFORE_OPEN_AUDIT_ENV_CREATE")
        env = ManagerBasedRLEnv(cfg)
        journal.mark("AFTER_OPEN_AUDIT_ENV_CREATE")
        journal.mark("BEFORE_OPEN_AUDIT_ENV_RESET")
        env.reset(seed=10043)
        journal.mark("AFTER_OPEN_AUDIT_ENV_RESET", gpu_final=_gpu_snapshot())
        return 0
    except BaseException as error:
        _atomic_json(output / "EXCEPTION.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        journal.mark("EXCEPTION", error_type=type(error).__name__)
        return 2
    finally:
        # All handles are deliberately local to the child.  No action manager
        # process/apply call, joint target write, or controller command exists.
        if env is not None:
            env.close()
        if app is not None:
            journal.mark("BEFORE_APP_CLOSE")
            app.close()
            journal.mark("AFTER_APP_CLOSE")


def _proc_snapshot(pid: int) -> dict[str, Any]:
    root = Path(f"/proc/{pid}")
    result: dict[str, Any] = {"pid": pid, "timestamp_utc": _utc()}
    try:
        result["status"] = (root / "status").read_text(errors="replace")
    except OSError as error:
        result["status_error"] = f"{type(error).__name__}:{error}"
    tasks: list[dict[str, Any]] = []
    try:
        for task in sorted((root / "task").iterdir()):
            if not task.name.isdigit():
                continue
            row = {"tid": int(task.name)}
            for name in ("wchan", "stack"):
                try:
                    row[name] = (task / name).read_text(errors="replace").strip()
                except OSError as error:
                    row[f"{name}_error"] = f"{type(error).__name__}:{error}"
            tasks.append(row)
    except OSError as error:
        result["tasks_error"] = f"{type(error).__name__}:{error}"
    result["tasks"] = tasks
    try:
        result["fd_count"] = len(list((root / "fd").iterdir()))
    except OSError as error:
        result["fd_count_error"] = f"{type(error).__name__}:{error}"
    return result


def _first_missing(marker_rows: list[dict[str, Any]], expected: list[str]) -> str | None:
    actual = {str(row.get("marker")) for row in marker_rows}
    return next((marker for marker in expected if marker not in actual), None)


def _read_markers(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _run_stage(stage: str, output: Path, timeout_s: float) -> dict[str, Any]:
    stage_dir = output / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--stage", stage, "--output", str(stage_dir / "child")]
    started = time.monotonic()
    with (stage_dir / "stdout.log").open("wb") as stdout, (stage_dir / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
        result: dict[str, Any] = {"stage": stage, "pid": child.pid, "command": command, "start_utc": _utc(), "timeout_s": timeout_s}
        try:
            exit_code = child.wait(timeout=timeout_s)
            result.update({"timed_out": False, "exit_code": exit_code})
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["hang_diagnostics"] = _proc_snapshot(child.pid)
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=10.0)
                result["exit_code_after_sigterm"] = child.returncode
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10.0)
                result["exit_code_after_sigkill"] = child.returncode
        result["end_utc"] = _utc()
        result["duration_s"] = time.monotonic() - started
        result["gpu_after"] = _gpu_snapshot()
    markers = _read_markers(stage_dir / "child" / "MARKERS.jsonl")
    marker_names = {str(row.get("marker")) for row in markers}
    required_marker = {
        "A_APP_LAUNCHER": "AFTER_FIRST_APP_UPDATE",
        "B_SIMULATION_CONTEXT": "AFTER_SIMULATION_CONTEXT_CREATE",
        "C_PHYSICS_SCENE": "AFTER_FIRST_PHYSICS_STEP",
        "D_CANDIDATE_A_ARTICULATION": "AFTER_CANDIDATE_A_FIRST_PHYSICS_STEP",
        "E_OPEN_AUDIT_SCENE": "AFTER_OPEN_AUDIT_ENV_RESET",
    }[stage]
    result["marker_count"] = len(markers)
    result["last_successful_checkpoint"] = markers[-1]["marker"] if markers else "NONE"
    result["first_missing_marker"] = _first_missing(markers, ["PROCESS_STARTED", "BEFORE_APPLAUNCHER_IMPORT", "AFTER_APPLAUNCHER_IMPORT", "BEFORE_APPLAUNCHER_CONSTRUCT", "AFTER_APPLAUNCHER_CONSTRUCT", "AFTER_FIRST_APP_UPDATE"])
    result["required_functional_marker"] = required_marker
    result["functional_stage_pass"] = required_marker in marker_names
    result["gpu_context_created"] = "AFTER_APPLAUNCHER_CONSTRUCT" in marker_names
    result["first_physics_step_reached"] = (
        "AFTER_FIRST_PHYSICS_STEP" in marker_names
        or "AFTER_CANDIDATE_A_FIRST_PHYSICS_STEP" in marker_names
        or "AFTER_OPEN_AUDIT_ENV_RESET" in marker_names
    )
    # Isaac's known post-finalization SIGSEGV occurs after a fully completed
    # functional stage.  It remains a clean-shutdown failure, but must never
    # be reported as a pre-AppLauncher futex/startup failure or prevent the
    # next fresh-child localization stage from running.
    result["known_shutdown_sigsegv"] = bool(
        result.get("exit_code") == -11
        and result["functional_stage_pass"]
        and "BEFORE_APP_CLOSE" in marker_names
        and "AFTER_APP_CLOSE" in marker_names
    )
    result["startup_blocking_failure"] = bool(result.get("timed_out") or not result["functional_stage_pass"])
    _atomic_json(stage_dir / "SUPERVISOR_RESULT.json", result)
    return result


def _known_good_comparison() -> list[dict[str, Any]]:
    candidates = (
        ROOT / "scripts/run_g2_stage1a_vector_runtime.py",
        ROOT / "scripts/run_g2_privileged_geometry_close_diagnostic.py",
        ROOT / "scripts/run_g2_contact_free_bc_regularized_sac_25env.py",
    )
    result = []
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        line = next((index + 1 for index, row in enumerate(text.splitlines()) if "AppLauncher(" in row), None)
        result.append({"path": str(path), "exists": path.is_file(), "app_launcher_line": line, "uses_enable_cameras_true": "enable_cameras=True" in text, "uses_headless_true": "headless=True" in text})
    return result


def _classify(stage_results: list[dict[str, Any]]) -> tuple[str, str | None, str | None]:
    failed = next((row for row in stage_results if row.get("startup_blocking_failure")), None)
    if failed is None:
        return "PASS", None, None
    stage = str(failed["stage"])
    mapping = {
        "A_APP_LAUNCHER": "APP_LAUNCHER_BOOTSTRAP",
        "B_SIMULATION_CONTEXT": "SIMULATION_CONTEXT_INIT",
        "C_PHYSICS_SCENE": "PHYSICS_CONTEXT_INIT",
        "D_CANDIDATE_A_ARTICULATION": "CANDIDATE_A_ASSET_LOAD",
        "E_OPEN_AUDIT_SCENE": "CAMERA_REPLICATOR_INIT",
    }
    return "FAIL", stage, mapping.get(stage, "UNRESOLVED")


def _supervisor(output: Path, timeout_s: float, full_repetitions: int) -> int:
    if output.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{output}")
    if not CANDIDATE_A_ASSET.is_file() or _sha256(CANDIDATE_A_ASSET) != CANDIDATE_A_SHA256:
        raise RuntimeError("CANDIDATE_A_SOURCE_FREEZE_MISMATCH")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "SUPERVISOR_ENVIRONMENT.json", _environment_snapshot())
    plan = list(STAGES[:-1]) + ["E_OPEN_AUDIT_SCENE" for _ in range(full_repetitions)]
    results: list[dict[str, Any]] = []
    for ordinal, stage in enumerate(plan, 1):
        name = stage if stage != "E_OPEN_AUDIT_SCENE" else f"E_OPEN_AUDIT_SCENE_RUN_{sum(row['stage'] == stage for row in results) + 1}"
        # The child receives the canonical stage name; the unique parent
        # directory represents the repetition.
        result = _run_stage(stage, output / f"{ordinal:02d}_{name}", timeout_s)
        result["ordinal"] = ordinal
        results.append(result)
        if result.get("startup_blocking_failure"):
            break
    status, first_hang_stage, root_cause = _classify(results)
    completed = [row["stage"] for row in results if row.get("functional_stage_pass")]
    known_shutdown_sigsegv = any(row.get("known_shutdown_sigsegv") for row in results)
    audit_status = "PARTIAL" if status == "PASS" and known_shutdown_sigsegv else status
    report = {
        "schema": SCHEMA,
        "mode": "LIVE_ACTIONLESS_STARTUP_ISOLATION",
        "candidate_a_asset": str(CANDIDATE_A_ASSET),
        "candidate_a_sha256": CANDIDATE_A_SHA256,
        "timeout_s_per_child": timeout_s,
        "full_bootstrap_repetitions_requested": full_repetitions,
        "stage_results": results,
        "known_good_bootstrap_static_comparison": _known_good_comparison(),
        "stale_processes_before": None,
        "STARTUP_AUDIT": audit_status,
        "LAST_SUCCESSFUL_STAGE": completed[-1] if completed else "NONE",
        "FIRST_HANG_STAGE": first_hang_stage,
        "FUTEX_WAIT_REPRODUCED": bool(first_hang_stage is not None and any("futex" in json.dumps(row.get("hang_diagnostics", {})).lower() for row in results)),
        "GPU_CONTEXT_CREATED": any(row.get("gpu_context_created") for row in results),
        "FIRST_PHYSICS_STEP_REACHED": any(row.get("first_physics_step_reached") for row in results),
        "ROOT_CAUSE": "KNOWN_SHUTDOWN_SIGSEGV_SEPARATE" if status == "PASS" and known_shutdown_sigsegv else root_cause,
        "KNOWN_SHUTDOWN_SIGSEGV": known_shutdown_sigsegv,
        "CANDIDATE_A_MECHANICS_TESTED": "NO",
        "OPEN_COMMAND_SUBMITTED": "NO",
        "PRIVILEGED_STARTED": "NO",
        "TRAINING_STARTED": "NO",
    }
    _atomic_json(output / "STARTUP_LADDER_REPORT.json", report)
    return 0 if status == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=75.0)
    parser.add_argument("--full-repetitions", type=int, default=3)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.child:
        if args.stage is None:
            raise SystemExit("CHILD_STAGE_REQUIRED")
        return _child(args.stage, args.output)
    if not args.execute_live:
        raise SystemExit("STARTUP_ISOLATION_REQUIRES_EXPLICIT_EXECUTE_LIVE")
    if args.timeout_s <= 0 or args.full_repetitions < 1:
        raise SystemExit("INVALID_BOUNDED_LIMIT")
    return _supervisor(args.output, args.timeout_s, args.full_repetitions)


if __name__ == "__main__":
    raise SystemExit(main())
