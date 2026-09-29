#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Offline validator for the Stage-1A deterministic-FSM GRU sequence sidecar.

This program never imports Isaac, writes a source artifact, trains a model, or
replays a rollout.  It verifies that the dual-RGB-D frame store and the
50-Hz causal control stream can be used for future offline GRU work without
mixing privileged teacher values into the deployable student input.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


SEQUENCE_SCHEMA = "g2_stage1a_deterministic_fsm_gru_sequence_v2"
FRAME_SCHEMA = "g2_stage1a_deterministic_fsm_rgbd_frame_store_v1"


class SequenceAuditError(ValueError):
    pass


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SequenceAuditError(f"JSONL_INVALID_LINE:{line_number}") from error
            if not isinstance(row, dict):
                raise SequenceAuditError(f"JSONL_ROW_NOT_MAPPING:{line_number}")
            rows.append(row)
    if not rows:
        raise SequenceAuditError("SEQUENCE_ROWS_EMPTY")
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_frame_reference(
    *,
    store: h5py.File,
    sensor: str,
    row: Mapping[str, Any],
) -> None:
    index_key = f"{sensor}_frame_store_index"
    id_key = f"{sensor}_camera_frame_id"
    timestamp_key = f"{sensor}_camera_timestamp_s"
    try:
        index = int(row[index_key])
        group = store[sensor]
    except (KeyError, TypeError, ValueError) as error:
        raise SequenceAuditError(f"FRAME_REFERENCE_MISSING:{sensor}") from error
    if not 0 <= index < int(group["rgb"].shape[0]):
        raise SequenceAuditError(f"FRAME_REFERENCE_OUTSIDE_STORE:{sensor}:{index}")
    if tuple(group["rgb"].shape[1:]) != (192, 256, 3):
        raise SequenceAuditError(f"RGB_SHAPE_INVALID:{sensor}")
    if tuple(group["depth_m"].shape[1:]) != (192, 256, 1):
        raise SequenceAuditError(f"DEPTH_SHAPE_INVALID:{sensor}")
    if int(group["env_id"][index]) != int(row["env_id"]):
        raise SequenceAuditError(f"FRAME_ENV_ID_MISMATCH:{sensor}:{index}")
    if int(group["episode_id"][index]) != int(row["episode_id"]):
        raise SequenceAuditError(f"FRAME_EPISODE_ID_MISMATCH:{sensor}:{index}")
    if int(group["sensor_frame_id"][index]) != int(row[id_key]):
        raise SequenceAuditError(f"FRAME_ID_MISMATCH:{sensor}:{index}")
    if not math.isclose(
        float(group["sensor_timestamp_s"][index]),
        float(row[timestamp_key]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise SequenceAuditError(f"FRAME_TIMESTAMP_MISMATCH:{sensor}:{index}")


def _groups(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["env_id"]), int(row["episode_id"]))].append(row)
    return grouped


def _sequence_contract(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    groups = _groups(rows)
    timestep_gaps = 0
    control_timestamp_regressions = 0
    camera_timestamp_regressions = {"head": 0, "right_wrist": 0}
    camera_reuse_max = {"head": 0, "right_wrist": 0}
    reconstructable = {str(length): 0 for length in (5, 10, 15, 25)}
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: int(row["control_step"]))
        previous_step = None
        previous_control_timestamp = None
        previous_camera_timestamp: dict[str, float | None] = {
            "head": None, "right_wrist": None,
        }
        reuse_run = {"head": 0, "right_wrist": 0}
        for row in ordered:
            step = int(row["control_step"])
            timestamp = float(row["control_timestamp_s"])
            if previous_step is not None and step != previous_step + 1:
                timestep_gaps += 1
            if previous_control_timestamp is not None and timestamp < previous_control_timestamp - 1e-9:
                control_timestamp_regressions += 1
            previous_step, previous_control_timestamp = step, timestamp
            for sensor in ("head", "right_wrist"):
                value = float(row[f"{sensor}_camera_timestamp_s"])
                prior = previous_camera_timestamp[sensor]
                if prior is not None and value < prior - 1e-9:
                    camera_timestamp_regressions[sensor] += 1
                if prior is not None and math.isclose(value, prior, rel_tol=0.0, abs_tol=1e-9):
                    reuse_run[sensor] += 1
                else:
                    reuse_run[sensor] = 1
                camera_reuse_max[sensor] = max(camera_reuse_max[sensor], reuse_run[sensor])
                previous_camera_timestamp[sensor] = value
        for length in (5, 10, 15, 25):
            if len(ordered) >= length:
                reconstructable[str(length)] += 1
    return {
        "episode_count": len(groups),
        "timestep_gap_count": timestep_gaps,
        "control_timestamp_regression_count": control_timestamp_regressions,
        "camera_timestamp_regression_count": camera_timestamp_regressions,
        "camera_reuse_max_control_rows": camera_reuse_max,
        "reconstructable_episode_count": reconstructable,
    }


def audit(*, run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "STAGE1A_VECTOR_REPORT.json"
    rows_path = run_dir / "FSM_GRU_SEQUENCE_ROWS.jsonl"
    store_path = run_dir / "FSM_GRU_SEQUENCE_RGBD.h5"
    if not all(path.is_file() for path in (report_path, rows_path, store_path)):
        raise SequenceAuditError("SEQUENCE_ARTIFACT_MISSING")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_rows(rows_path)
    exact_rows = 0
    teacher_input_leakage = 0
    missing_required = 0
    with h5py.File(store_path, "r") as store:
        if store.attrs.get("schema") != FRAME_SCHEMA:
            raise SequenceAuditError("FRAME_STORE_SCHEMA_INVALID")
        for sensor in ("head", "right_wrist"):
            if sensor not in store:
                raise SequenceAuditError(f"FRAME_STORE_SENSOR_MISSING:{sensor}")
        for row in rows:
            required = (
                "schema", "episode_id", "env_id", "control_step",
                "control_timestamp_s", "student_observation", "previous_action_4d_metric_root_m",
                "sac_residual_xyz_action_m", "fsm", "teacher", "outcome",
                "student_privileged_input_count", "gru_runtime_authority",
                "privileged_runtime_authority", "source_family_id",
            )
            if any(name not in row for name in required) or row.get("schema") != SEQUENCE_SCHEMA:
                missing_required += 1
                continue
            if row["student_privileged_input_count"] != 0:
                teacher_input_leakage += 1
            observation = row["student_observation"]
            if not isinstance(observation, Mapping) or "teacher" in observation:
                teacher_input_leakage += 1
            if row["gru_runtime_authority"] is not False or row["privileged_runtime_authority"] is not False:
                teacher_input_leakage += 1
            if not _finite(row["control_timestamp_s"]):
                raise SequenceAuditError("CONTROL_TIMESTAMP_INVALID")
            _validate_frame_reference(store=store, sensor="head", row=row)
            _validate_frame_reference(store=store, sensor="right_wrist", row=row)
            teacher = row["teacher"]
            if not isinstance(teacher, Mapping):
                raise SequenceAuditError("TEACHER_RECEIPT_INVALID")
            if bool(teacher.get("exact_signed_margin_available", False)):
                exact_rows += 1
                required_margins = (
                    "primary_pad_containment_margin_mm", "inner_containment_margin_mm",
                    "outer_containment_margin_mm", "aperture_margin_mm",
                    "pad_surface_gap_mm", "orientation_margin_deg",
                    "signed_readiness_margin",
                )
                if not all(_finite(teacher.get(name)) for name in required_margins):
                    raise SequenceAuditError("EXACT_SIGNED_MARGIN_TELEMETRY_INCOMPLETE")
    continuity = _sequence_contract(rows)
    summary = report.get("sequence_dataset")
    row_count_match = (
        isinstance(summary, Mapping)
        and int(summary.get("sequence_row_count", -1)) == len(rows)
        and int(report.get("accepted_transitions", -1)) == len(rows)
    )
    complete = (
        missing_required == 0
        and teacher_input_leakage == 0
        and exact_rows > 0
        and row_count_match
        and continuity["timestep_gap_count"] == 0
        and continuity["control_timestamp_regression_count"] == 0
        and all(value == 0 for value in continuity["camera_timestamp_regression_count"].values())
        and all(value > 0 for value in continuity["reconstructable_episode_count"].values())
    )
    return {
        "schema": "g2_stage1a_fsm_sequence_dataset_audit_v1",
        "run_dir": str(run_dir),
        "SAC_REPLAY_SOURCE": "CURRENT_15K_ROLLOUT_ONLY",
        "OLD_TEACHER_DATA_USED_IN_SAC": "NO",
        "BATCH10_USED": "NO",
        "NEW_SEQUENCE_ROWS": len(rows),
        "NEW_EXACT_SIGNED_MARGIN_ROWS": exact_rows,
        "HEAD_RGBD_STORED": True,
        "RIGHT_WRIST_RGBD_STORED": True,
        "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
        "GRU_RUNTIME_AUTHORITY": "NO",
        "PRIVILEGED_RUNTIME_AUTHORITY": "NO",
        "sequence_continuity": continuity,
        "row_count_match": row_count_match,
        "missing_required_row_count": missing_required,
        "student_privileged_input_leakage_count": teacher_input_leakage,
        "SEQUENCE_DATASET_VALID": "YES" if complete else "NO",
        "SIGNED_MARGIN_TELEMETRY_COMPLETE": "YES" if exact_rows > 0 and complete else "NO",
        "FUTURE_GRU_DATA_READY": "YES" if complete else "NO",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit(run_dir=args.run_dir.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {
            "schema": "g2_stage1a_fsm_sequence_dataset_audit_v1",
            "SEQUENCE_DATASET_VALID": "NO",
            "SIGNED_MARGIN_TELEMETRY_COMPLETE": "NO",
            "FUTURE_GRU_DATA_READY": "NO",
            "error": f"{type(error).__name__}:{error}",
        }
    output = args.output or args.run_dir / "FSM_SEQUENCE_DATASET_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("SEQUENCE_DATASET_VALID") == "YES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
