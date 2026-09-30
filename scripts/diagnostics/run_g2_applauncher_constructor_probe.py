#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fresh-child, AppLauncher-constructor-only intermittent futex probe.

This script deliberately never imports GenieSim, scene, simulation, assets,
camera/Replicator, policy, or controller code.  Each child only imports and
constructs Isaac Lab's AppLauncher with the r4 A-stage arguments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "g2_applauncher_constructor_probe_v1"
VOLATILE_ENV_KEYS = {"INVOCATION_ID", "JOURNAL_STREAM", "SYSTEMD_EXEC_PID", "SHLVL", "_"}
SENSITIVE_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "PASS", "KEY", "AUTH", "COOKIE", "CREDENTIAL")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _redacted_env() -> dict[str, str]:
    return {
        key: "<REDACTED>" if any(word in key.upper() for word in SENSITIVE_FRAGMENTS) else value
        for key, value in sorted(os.environ.items())
    }


def _stable_environment(redacted: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in redacted.items() if key not in VOLATILE_ENV_KEYS}


def _fd_snapshot(pid: int) -> dict[str, Any]:
    directory = Path(f"/proc/{pid}/fd")
    rows = []
    try:
        for entry in sorted(directory.iterdir(), key=lambda candidate: int(candidate.name)):
            try:
                target = os.readlink(entry)
            except OSError as error:
                target = f"<ERROR:{type(error).__name__}>"
            rows.append({"fd": int(entry.name), "target": target})
    except OSError as error:
        return {"error": f"{type(error).__name__}:{error}"}
    return {"count": len(rows), "entries": rows}


def _parent_snapshot() -> dict[str, Any]:
    pid = os.getppid()
    root = Path(f"/proc/{pid}")
    return {
        "pid": pid,
        "cmdline": [piece.decode("utf-8", errors="replace") for piece in (root / "cmdline").read_bytes().split(b"\0") if piece] if (root / "cmdline").exists() else [],
        "status": _read_text(root / "status"),
        "fd": _fd_snapshot(pid),
    }


def _runtime_snapshot() -> dict[str, Any]:
    redacted = _redacted_env()
    stable = _stable_environment(redacted)
    selected = (
        "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX", "VIRTUAL_ENV", "HOME",
        "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "DISPLAY",
        "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "OMNI_KIT_ACCEPT_EULA",
    )
    limits: dict[str, list[int]] = {}
    for name in dir(resource):
        value = getattr(resource, name)
        if name.startswith("RLIMIT_") and isinstance(value, int):
            try:
                limits[name] = [int(item) for item in resource.getrlimit(value)]
            except (OSError, ValueError):
                pass
    x11_socket = Path("/tmp/.X11-unix/X0")
    return {
        "schema": SCHEMA,
        "timestamp_utc": _utc(), "pid": os.getpid(), "ppid": os.getppid(),
        "cwd": os.getcwd(), "python": sys.executable, "argv": list(sys.argv),
        "selected_env": {name: redacted.get(name) for name in selected},
        "isaac_kit_omni_env": {key: value for key, value in redacted.items() if key.startswith(("ISAAC_", "KIT_", "OMNI_", "CARB_"))},
        "environment_redacted": redacted,
        "stable_environment_fingerprint_sha256": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "volatile_environment_keys_excluded": sorted(VOLATILE_ENV_KEYS),
        "resource_limits": limits,
        "parent": _parent_snapshot(),
        "inherited_fds": _fd_snapshot(os.getpid()),
        "display": {"x11_socket_exists": x11_socket.exists(), "x11_socket": str(x11_socket)},
        "kit_paths": {name: redacted.get(name) for name in ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR")},
    }


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=5, check=False,
        )
        return {"returncode": completed.returncode, "apps": completed.stdout.splitlines(), "stderr": completed.stderr.strip()}
    except BaseException as error:
        return {"error": f"{type(error).__name__}:{error}"}


class _Journal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def mark(self, marker: str, **detail: Any) -> None:
        try:
            count = len(list(Path(f"/proc/{os.getpid()}/task").iterdir()))
        except OSError:
            count = threading.active_count()
        row = {
            "schema": SCHEMA, "marker": marker, "timestamp_utc": _utc(), "monotonic_s": time.monotonic(),
            "pid": os.getpid(), "ppid": os.getppid(), "tid": threading.get_native_id(), "thread_count": count,
            "cwd": os.getcwd(), "python": sys.executable, "argv": list(sys.argv), "detail": detail,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(marker, flush=True)


def _child(output: Path) -> int:
    if output.exists():
        raise RuntimeError(f"CHILD_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=False)
    journal = _Journal(output / "MARKERS.jsonl")
    _atomic_json(output / "RUNTIME_SNAPSHOT.json", _runtime_snapshot())
    journal.mark("PROCESS_STARTED", gpu_before=_gpu_snapshot())
    app = None
    try:
        journal.mark("BEFORE_APPLAUNCHER_IMPORT")
        from isaaclab.app import AppLauncher
        journal.mark("AFTER_APPLAUNCHER_IMPORT", app_launcher_source=str(Path(inspect.getsourcefile(AppLauncher) or "").resolve()))
        journal.mark("BEFORE_APPLAUNCHER_CONSTRUCTOR", headless=True, enable_cameras=False, fast_shutdown=False)
        launcher = AppLauncher(headless=True, enable_cameras=False, fast_shutdown=False)
        app = launcher.app
        journal.mark("AFTER_APPLAUNCHER_CONSTRUCTOR", gpu_after=_gpu_snapshot())
        _atomic_json(output / "CONSTRUCTOR_RESULT.json", {"constructor_returned": True, "timestamp_utc": _utc(), "app_type": f"{type(app).__module__}.{type(app).__qualname__}"})
        return 0
    except BaseException as error:
        _atomic_json(output / "EXCEPTION.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        journal.mark("EXCEPTION", error_type=type(error).__name__)
        return 2
    finally:
        if app is not None:
            journal.mark("BEFORE_APP_CLOSE")
            app.close()
            journal.mark("AFTER_APP_CLOSE")


def _proc_snapshot(pid: int) -> dict[str, Any]:
    root = Path(f"/proc/{pid}")
    result: dict[str, Any] = {"pid": pid, "timestamp_utc": _utc(), "status": _read_text(root / "status"), "fds": _fd_snapshot(pid)}
    task_rows: list[dict[str, Any]] = []
    try:
        for task in sorted((root / "task").iterdir(), key=lambda candidate: int(candidate.name)):
            if not task.name.isdigit():
                continue
            row = {"tid": int(task.name)}
            for name in ("wchan", "stack"):
                try:
                    row[name] = (task / name).read_text(errors="replace").strip()
                except OSError as error:
                    row[f"{name}_error"] = f"{type(error).__name__}:{error}"
            task_rows.append(row)
    except OSError as error:
        result["task_error"] = f"{type(error).__name__}:{error}"
    result["tasks"] = task_rows
    try:
        maps = (root / "maps").read_text(errors="replace").splitlines()
        result["loaded_libraries_of_interest"] = [line for line in maps if any(term in line.lower() for term in ("isaac", "kit", "omni", "carb", "cuda"))]
    except OSError as error:
        result["maps_error"] = f"{type(error).__name__}:{error}"
    return result


def _marker_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _last_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-1] if lines else None


def _terminate_group(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        result["exit_code_after_sigterm"] = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        result["exit_code_after_sigkill"] = process.returncode
    return result


def _run_one(output: Path, ordinal: int, timeout_s: float, *, under_strace: bool = False) -> dict[str, Any]:
    run_dir = output / f"run_{ordinal:02d}{'_strace' if under_strace else ''}"
    run_dir.mkdir(parents=True, exist_ok=False)
    child_dir = run_dir / "child"
    child_cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--output", str(child_dir)]
    trace_prefix = run_dir / "futex.strace"
    command = (["strace", "-ff", "-tt", "-T", "-o", str(trace_prefix), "-e", "trace=futex"] + child_cmd) if under_strace else child_cmd
    started = time.monotonic()
    with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr, start_new_session=True)
        result: dict[str, Any] = {"ordinal": ordinal, "command": command, "child_command": child_cmd, "pid": process.pid, "start_utc": _utc(), "timeout_s": timeout_s, "strace": under_strace}
        try:
            result["exit_code"] = process.wait(timeout=timeout_s)
            result["timed_out"] = False
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["hang_diagnostics"] = _proc_snapshot(process.pid)
            result.update(_terminate_group(process))
        result["duration_s"] = time.monotonic() - started
        result["end_utc"] = _utc()
        result["gpu_after"] = _gpu_snapshot()
    markers = _marker_rows(child_dir / "MARKERS.jsonl")
    marker_names = {row.get("marker") for row in markers}
    result.update({
        "constructor_returned": "AFTER_APPLAUNCHER_CONSTRUCTOR" in marker_names,
        "constructor_entry_marker": "BEFORE_APPLAUNCHER_CONSTRUCTOR" in marker_names,
        "last_marker": markers[-1]["marker"] if markers else "NONE",
        "thread_count_at_last_marker": markers[-1].get("thread_count") if markers else None,
        "last_stdout_line": _last_line(run_dir / "stdout.log"),
        "last_stderr_line": _last_line(run_dir / "stderr.log"),
        "known_shutdown_sigsegv": bool(result.get("exit_code") == -11 and "AFTER_APP_CLOSE" in marker_names),
    })
    snapshot_path = child_dir / "RUNTIME_SNAPSHOT.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text())
        result["stable_environment_fingerprint_sha256"] = snapshot["stable_environment_fingerprint_sha256"]
        result["inherited_fd_count"] = snapshot["inherited_fds"].get("count")
        result["parent_pid"] = snapshot["parent"]["pid"]
        result["selected_env"] = snapshot["selected_env"]
    _atomic_json(run_dir / "RUN_RESULT.json", result)
    return result


def _material_differences(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("stable_environment_fingerprint_sha256", "inherited_fd_count", "parent_pid", "selected_env")
    result: dict[str, Any] = {}
    for key in keys:
        values = [run.get(key) for run in runs]
        unique = {json.dumps(value, sort_keys=True, default=str) for value in values}
        result[key] = {"different": len(unique) > 1, "values": values}
    return result


def _classify(runs: list[dict[str, Any]], differences: dict[str, Any]) -> str:
    passes = sum(bool(row.get("constructor_returned")) for row in runs)
    hangs = sum(bool(row.get("timed_out")) for row in runs)
    if passes and hangs:
        if any(item["different"] for item in differences.values()):
            return "INHERITED_PROCESS_STATE"
        return "INTERMITTENT_HOST_STATE"
    if hangs:
        return "APP_LAUNCHER_BOOTSTRAP"
    return "UNRESOLVED"


def _supervisor(output: Path, runs: int, timeout_s: float, strace_on_hang: bool) -> int:
    if output.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "SUPERVISOR_RUNTIME_SNAPSHOT.json", _runtime_snapshot())
    results = [_run_one(output, ordinal, timeout_s) for ordinal in range(1, runs + 1)]
    if strace_on_hang and any(row.get("timed_out") for row in results) and shutil.which("strace"):
        results.append(_run_one(output, runs + 1, min(timeout_s, 30.0), under_strace=True))
    differences = _material_differences(results[:runs])
    pass_count = sum(bool(row.get("constructor_returned")) for row in results[:runs])
    hang_count = sum(bool(row.get("timed_out")) for row in results[:runs])
    longest = 0
    current_streak = 0
    for row in results[:runs]:
        current_streak = current_streak + 1 if row.get("constructor_returned") else 0
        longest = max(longest, current_streak)
    report = {
        "schema": SCHEMA, "mode": "LIVE_APPLAUNCHER_CONSTRUCTOR_ONLY", "requested_runs": runs,
        "timeout_s": timeout_s, "runs": results, "PASS_COUNT": pass_count, "HANG_COUNT": hang_count,
        "INTERMITTENT_REPRODUCED": bool(pass_count and hang_count),
        "PASS_HANG_DIFFERENCE_FOUND": any(item["different"] for item in differences.values()),
        "PASS_HANG_DIFFERENCES": differences,
        "FUTEX_WCHAN": sorted({task.get("wchan") for row in results for task in row.get("hang_diagnostics", {}).get("tasks", []) if task.get("wchan")}),
        "ROOT_CAUSE": _classify(results[:runs], differences),
        "MINIMAL_FIX": "NOT_ESTABLISHED; fresh-child timeout isolation only; do not change Kit cache/locks without evidence",
        "APP_LAUNCHER_3X_CONSECUTIVE_PASS": longest >= 3,
        "PHYSICS_STARTED": "NO", "CANDIDATE_A_LOADED": "NO", "OPEN_COMMAND_SUBMITTED": "NO", "TRAINING_STARTED": "NO",
    }
    _atomic_json(output / "APPLAUNCHER_CONSTRUCTOR_REPORT.json", report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--strace-on-hang", action="store_true")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        return _child(args.output)
    if not args.execute_live or args.runs < 1 or args.timeout_s <= 0:
        raise SystemExit("EXPLICIT_BOUNDED_LIVE_EXECUTION_REQUIRED")
    return _supervisor(args.output, args.runs, args.timeout_s, args.strace_on_hang)


if __name__ == "__main__":
    raise SystemExit(main())
