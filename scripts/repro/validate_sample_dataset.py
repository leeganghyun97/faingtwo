#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validate the tiny synthetic portability fixture; never accepts real data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.root / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    failures: list[str] = []
    if manifest.get("synthetic") is not True:
        failures.append("FIXTURE_MUST_BE_SYNTHETIC")
    if (manifest.get("frame"), manifest.get("translation_unit"), manifest.get("quaternion_order")) != ("robot_root", "m", "XYZW"):
        failures.append("FRAME_UNIT_QUATERNION_CONTRACT")
    if (manifest.get("control_hz"), manifest.get("rgbd_hz"), manifest.get("physics_hz")) != (50, 25, 500):
        failures.append("RATE_CONTRACT")
    if len(rows) != sum(int(item["rows"]) for item in manifest.get("episodes", [])):
        failures.append("ROW_COUNT")
    for index, row in enumerate(rows):
        values = [row.get("timestamp_s"), *row.get("action_xyz_m", [])]
        if len(values) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            failures.append(f"ROW_{index}_FINITE")
    print(json.dumps({"schema": "geniesim_sample_validation_v1", "status": "PASS" if not failures else "FAIL", "rows": len(rows), "failures": failures}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
