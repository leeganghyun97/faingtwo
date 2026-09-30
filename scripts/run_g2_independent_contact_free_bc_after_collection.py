#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Start grouped contact-free BC/CV only after a complete 30-run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def _status(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-s", type=float, default=10.0)
    parser.add_argument("--capture-rate-hz", type=int, choices=(25,), default=25)
    parser.add_argument("--allow-partial-collection", action="store_true")
    parser.add_argument("--minimum-valid-episodes", type=int, default=5)
    parser.add_argument("--wait", action="store_true", help="required; never start on an absent manifest")
    args = parser.parse_args()
    if not args.wait:
        raise SystemExit("BC_AFTER_COLLECTION_REQUIRES_EXPLICIT_--wait")
    root = args.collection_root.resolve()
    manifest_path = root / "COLLECTION_MANIFEST.json"
    status_path = args.output.parent / (args.output.name + ".LAUNCH_STATUS.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        manifest = _status(manifest_path)
        status = str(manifest.get("status", "MISSING"))
        complete = len(manifest.get("runs", [])) == 30
        pass_gate = status == "PASS" and manifest.get("functional_pass_count") == 30
        partial_gate = (
            args.allow_partial_collection
            and status == "COMPLETE_WITH_FAILURES"
            and complete
            and int(manifest.get("functional_pass_count", 0)) >= args.minimum_valid_episodes
        )
        if pass_gate or partial_gate:
            break
        if status == "INCOMPLETE_FAIL_CLOSED":
            status_path.write_text(json.dumps({"status": "BLOCKED_INCOMPLETE_COLLECTION", "manifest": str(manifest_path)}, indent=2) + "\n", encoding="utf-8")
            return 2
        time.sleep(max(float(args.poll_s), 1.0))
    command = [
        args.python,
        str(Path(__file__).resolve().with_name("train_g2_independent_contact_free_bc.py")),
        "--collection-root", str(root),
        "--output", str(args.output.resolve()),
        "--epochs", "5",
        "--batch-size", "256",
        "--folds", "5",
        "--seed", "42",
        "--device", "cuda:0",
        "--baseline-checkpoint", str(args.baseline_checkpoint.resolve()),
        "--capture-rate-hz", str(args.capture_rate_hz),
        "--minimum-valid-episodes", str(args.minimum_valid_episodes),
    ]
    if args.allow_partial_collection:
        command.append("--allow-partial-collection")
    status_path.write_text(json.dumps({"status": "LAUNCHING", "command": command}, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[1]), check=False)
    status_path.write_text(json.dumps({"status": "COMPLETED" if result.returncode == 0 else "FAILED", "returncode": result.returncode, "command": command}, indent=2) + "\n", encoding="utf-8")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
