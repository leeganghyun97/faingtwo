#!/usr/bin/env python3
"""Grouped BC/CV for exact-contract cuRobo and keyboard-v2 rollouts.

This trainer is intentionally separate from the historical single-rollout
trainer.  Each HDF5 is one reset/seed episode and is kept intact for grouped
held-out evaluation; adjacent rows from the same 50-Hz rollout never cross a
split boundary.  The model, observation, and action contract are imported
from the canonical BC implementation, so this file cannot silently change
units or action dimensionality.  Human and planner rows feed the same student
only after their actor/action/state/cadence fingerprints compare exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "scripts/train_g2_canonical_contact_free_bc.py"
if not CANONICAL_PATH.is_file():
    raise RuntimeError("canonical BC implementation is missing")
_spec = importlib.util.spec_from_file_location("g2_canonical_bc_for_grouped_cv", CANONICAL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load canonical BC implementation")
canonical = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = canonical
_spec.loader.exec_module(canonical)

from geniesim.rl.isaaclab.g2_lift_methodology import RIGHT_ARM_JOINTS
from geniesim.rl.isaaclab.g2_policy_branch.canonical_contact_free_collection_v2 import (
    ACTOR_FIELDS,
    FORBIDDEN_FIELD_TOKENS,
)

POLICY_DT_S = 0.020
PHYSICS_DT_S = 0.002
ACTION_BOUND_M = 0.0045
KEYBOARD_V2_SCHEMA = "g2_keyboard_contact_free_v2"
KEYBOARD_V2_MANIFEST_SCHEMA = "g2_keyboard_contact_free_v2_manifest"
EXPECTED_CANDIDATE_A = (
    "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
)
KEYBOARD_SOURCE_DOMAIN = "HUMAN_KEYBOARD_V2"
CUROBO_SOURCE_DOMAIN = "CUROBO"
KEYBOARD_EXPORTED_ROW_DTYPES = {
    "source_row_index": ((), np.int64),
    "control_step": ((), np.int64),
    "timestamp_s": ((), np.float64),
    "right_wrist_rgb": ((192, 256, 3), np.uint8),
    "right_wrist_depth_m": ((192, 256, 1), np.float32),
    "right_wrist_depth_valid": ((192, 256, 1), np.bool_),
    "ee_pose_robot_root_m_xyzw": ((7,), np.float32),
    "right_arm_joint_position_rad": ((7,), np.float32),
    "right_arm_joint_velocity_rad_s": ((7,), np.float32),
    "right_arm_joint_acceleration_rad_s2": ((7,), np.float32),
    "gripper_state_open": ((1,), np.float32),
    "previous_policy_action_4d_metric_root_m": ((4,), np.float32),
    "policy_action_4d_metric_root_m": ((4,), np.float32),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _student_contract_fingerprint(
    manifest: dict[str, Any],
    *,
    expected_capture_rate_hz: int,
    require_action_order: bool,
) -> dict[str, Any]:
    """Return the exact shared actor/action surface or fail closed.

    Source-specific packet receipts and provenance legitimately differ between
    cuRobo and human keyboard episodes.  Every tensor consumed by the one
    student, its units/frame, and its capture/control cadence must not differ.
    """

    actor_fields = manifest.get("actor_fields")
    action = _mapping(manifest.get("action"), "student action")
    previous = _mapping(manifest.get("previous_action"), "student previous action")
    state = _mapping(manifest.get("state"), "student state")
    capture = _mapping(
        manifest.get("dataset_capture_contract"), "student capture contract"
    )
    if tuple(actor_fields or ()) != ACTOR_FIELDS or any(
        token in str(field).lower()
        for field in actor_fields or ()
        for token in FORBIDDEN_FIELD_TOKENS
    ):
        raise RuntimeError("student actor inventory mismatch")
    expected_action = {
        "field": "policy_action_4d_metric_root_m",
        "frame": "robot_root",
        "unit": "m,m,m,HOLD_OPEN=0",
        "max_norm_m": ACTION_BOUND_M,
    }
    if any(action.get(key) != value for key, value in expected_action.items()):
        raise RuntimeError("student action contract mismatch")
    if require_action_order and action.get("order") != [
        "dx",
        "dy",
        "dz",
        "HOLD_OPEN",
    ]:
        raise RuntimeError("student action order mismatch")
    expected_previous = {
        "field": "previous_policy_action_4d_metric_root_m",
        "frame": "robot_root",
        "unit": "m,m,m,HOLD_OPEN=0",
        "normalized_actor_input_forbidden": True,
    }
    if any(previous.get(key) != value for key, value in expected_previous.items()):
        raise RuntimeError("student previous-action contract mismatch")
    expected_state = {
        "ee": "robot_root,m,XYZW",
        "right_arm_order": list(RIGHT_ARM_JOINTS),
        "q_qd_qdd": "rad,rad/s,rad/s^2",
        "dtype": "float32",
        "physics_dt_s": PHYSICS_DT_S,
        "policy_dt_s": POLICY_DT_S,
    }
    if any(state.get(key) != value for key, value in expected_state.items()):
        raise RuntimeError("student state/unit contract mismatch")
    expected_capture = {
        "control_rate_hz": 50,
        "capture_rate_hz": expected_capture_rate_hz,
        "capture_control_step_stride": 50 // expected_capture_rate_hz,
    }
    if any(capture.get(key) != value for key, value in expected_capture.items()):
        raise RuntimeError("student capture/control cadence mismatch")
    return {
        "actor_fields": list(ACTOR_FIELDS),
        "action": expected_action,
        "action_order": ["dx", "dy", "dz", "HOLD_OPEN"],
        "previous_action": expected_previous,
        "state": expected_state,
        "capture": expected_capture,
        "cube_ground_truth_actor_input": False,
        "contact": False,
        "close": False,
        "lift": False,
        "her": False,
        "her_force": False,
    }


def _discover(
    root: Path,
    expected_capture_rate_hz: int,
    *,
    expected_attempts: int | None,
    allow_partial: bool,
    minimum_valid_episodes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "COLLECTION_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("COLLECTION_MANIFEST.json is missing")
    manifest = _load_json(manifest_path)
    status = str(manifest.get("status", ""))
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise RuntimeError("collection schedule is missing or empty")
    manifest_attempts = len(schedule)
    if expected_attempts is not None and manifest_attempts != expected_attempts:
        raise RuntimeError(
            f"collection schedule mismatch: {manifest_attempts} != {expected_attempts}"
        )
    expected_attempts = manifest_attempts
    if status == "PASS":
        if manifest.get("functional_pass_count") != expected_attempts:
            raise RuntimeError(
                "collection PASS does not contain every scheduled functional run"
            )
    elif not (allow_partial and status == "COMPLETE_WITH_FAILURES"):
        raise RuntimeError("collection set is not an authorized complete collection")
    contract = _mapping(manifest.get("contract"), "collection contract")
    expected = {
        "action_dim": 4,
        "action": "[dx,dy,dz,HOLD_OPEN=0]",
        "frame": "robot_root",
        "unit": "m",
        "policy_dt_s": POLICY_DT_S,
        "physics_dt_s": PHYSICS_DT_S,
        "capture_rate_hz": expected_capture_rate_hz,
        "capture_control_step_stride": 50 // expected_capture_rate_hz,
        "contact": False,
        "close": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"collection contract mismatch: {key}")
    records = manifest.get("runs")
    if not isinstance(records, list) or len(records) != expected_attempts:
        raise RuntimeError(
            f"manifest must contain exactly {expected_attempts} completed attempts"
        )
    collection_id = root.name
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record = _mapping(record, "run receipt")
        if not bool(record.get("functional_pass")):
            continue
        source_episode_id = str(record.get("episode_id"))
        episode_id = f"{collection_id}/{source_episode_id}"
        if episode_id in seen:
            raise RuntimeError(f"duplicate episode: {episode_id}")
        seen.add(episode_id)
        dataset = Path(str(record.get("dataset"))).resolve()
        report = Path(str(record.get("functional_report"))).resolve()
        sidecar = Path(str(record.get("sidecar"))).resolve()
        if not dataset.is_file() or not report.is_file() or not sidecar.is_file():
            raise RuntimeError(f"run artifact missing: {episode_id}")
        if record.get("dataset_sha256") != sha256(dataset):
            raise RuntimeError(f"dataset hash mismatch: {episode_id}")
        if record.get("functional_report_sha256") != sha256(report):
            raise RuntimeError(f"functional report hash mismatch: {episode_id}")
        if record.get("sidecar_sha256") != sha256(sidecar):
            raise RuntimeError(f"sidecar hash mismatch: {episode_id}")
        audit = canonical.audit_canonical_artifact(dataset=dataset, functional_report=report)
        with h5py.File(dataset, "r") as handle:
            manifest_json = json.loads(str(handle.attrs["manifest_json"]))
        contract_fingerprint = _student_contract_fingerprint(
            manifest_json,
            expected_capture_rate_hz=expected_capture_rate_hz,
            require_action_order=False,
        )
        independent = _mapping(manifest_json.get("independent_collection"), f"{episode_id}.independent_collection")
        if independent.get("episode_id") != source_episode_id or int(independent.get("reset_seed")) != int(record.get("seed")):
            raise RuntimeError(f"episode/seed binding mismatch: {episode_id}")
        result.append({
            "episode_id": episode_id,
            "source_episode_id": source_episode_id,
            "collection_id": collection_id,
            "source_domain": CUROBO_SOURCE_DOMAIN,
            "seed": int(record["seed"]),
            "dataset": dataset,
            "report": report,
            "sidecar": sidecar,
            "rows": int(audit.rows),
            "dataset_sha256": record["dataset_sha256"],
            "student_contract_fingerprint": contract_fingerprint,
        })
    result.sort(key=lambda item: item["episode_id"])
    if len(result) < minimum_valid_episodes:
        raise RuntimeError(f"insufficient functional episodes for grouped CV: {len(result)} < {minimum_valid_episodes}")
    return result, manifest


def _discover_keyboard_v2(
    root: Path, expected_capture_rate_hz: int = 25
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate one immutable HUMAN keyboard-v2 export without cuRobo receipts."""

    manifest_path = root / "KEYBOARD_V2_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("KEYBOARD_V2_MANIFEST.json is missing")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != KEYBOARD_V2_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("candidate_a_asset_sha256") != EXPECTED_CANDIDATE_A
        or manifest.get("source_domain") != KEYBOARD_SOURCE_DOMAIN
    ):
        raise RuntimeError("keyboard-v2 manifest authority mismatch")
    top_provenance = _mapping(
        manifest.get("provenance"), "keyboard-v2 top-level provenance"
    )
    if (
        top_provenance.get("immutable_source") is not True
        or top_provenance.get("source_domain") != KEYBOARD_SOURCE_DOMAIN
        or top_provenance.get("source_hdf_sha256")
        != manifest.get("source_hdf_sha256")
        or top_provenance.get("collection_summary_sha256")
        != manifest.get("collection_summary_sha256")
        or top_provenance.get("candidate_a_asset_sha256") != EXPECTED_CANDIDATE_A
    ):
        raise RuntimeError("keyboard-v2 top-level provenance mismatch")
    contract = _mapping(manifest.get("contract"), "keyboard-v2 contract")
    expected = {
        "action_dim": 4,
        "frame": "robot_root",
        "unit": "m,m,m,HOLD_OPEN=0",
        "maximum_action_norm_m": ACTION_BOUND_M,
        "control_rate_hz": 50,
        "capture_rate_hz": 25,
        "capture_control_step_stride": 2,
        "ee": "robot_root,m,XYZW",
        "right_arm_order": list(RIGHT_ARM_JOINTS),
        "q_qd_qdd": "rad,rad/s,rad/s^2",
        "previous_action": "robot_root,m,m,m,HOLD_OPEN=0",
        "gripper": "HOLD_OPEN=0",
        "contact": False,
        "close": False,
        "lift": False,
        "her": False,
        "her_force": False,
        "cube_ground_truth_actor_input": False,
    }
    if (
        expected_capture_rate_hz != 25
        or any(contract.get(key) != value for key, value in expected.items())
        or tuple(contract.get("actor_fields", ())) != ACTOR_FIELDS
        or contract.get("action_order") != ["dx", "dy", "dz", "HOLD_OPEN"]
    ):
        raise RuntimeError("keyboard-v2 contract mismatch")
    source_hdf = Path(str(manifest.get("source_hdf", ""))).resolve()
    if not source_hdf.is_file() or manifest.get("source_hdf_sha256") != sha256(source_hdf):
        raise RuntimeError("keyboard-v2 immutable source mismatch")
    summary_path = Path(str(manifest.get("collection_summary", ""))).resolve()
    if (
        not summary_path.is_file()
        or manifest.get("collection_summary_sha256") != sha256(summary_path)
    ):
        raise RuntimeError("keyboard-v2 immutable collection summary mismatch")
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise RuntimeError("keyboard-v2 episodes must be a list")
    artifacts: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for record in records:
        record = _mapping(record, "keyboard-v2 episode")
        source_episode_id = str(record.get("episode_id", ""))
        if not source_episode_id or source_episode_id in seen_episode_ids:
            raise RuntimeError("keyboard-v2 duplicate/missing source episode id")
        seen_episode_ids.add(source_episode_id)
        if record.get("source_domain") != KEYBOARD_SOURCE_DOMAIN:
            raise RuntimeError("keyboard-v2 record source domain mismatch")
        if (
            record.get("completion") not in {"complete", "partial_safe"}
            or int(record.get("source_rows", -1))
            != int(record.get("safe_prefix_rows_50hz", -2))
            + int(record.get("excluded_tail_rows_50hz", -2))
        ):
            raise RuntimeError("keyboard-v2 record prefix/completion mismatch")
        if record.get("eligible") is not True:
            if (
                record.get("dataset") is not None
                or record.get("dataset_sha256") is not None
                or record.get("rows") != 0
            ):
                raise RuntimeError("keyboard-v2 ineligible record publishes rows")
            continue
        episode_id = f"{root.name}/{source_episode_id}"
        dataset = Path(str(record.get("dataset", ""))).resolve()
        expected_dataset = (
            root / "episodes" / source_episode_id / "KEYBOARD_CONTACT_FREE_ROWS.hdf5"
        ).resolve()
        if dataset != expected_dataset:
            raise RuntimeError(f"keyboard-v2 dataset path mismatch: {episode_id}")
        if not dataset.is_file() or record.get("dataset_sha256") != sha256(dataset):
            raise RuntimeError(f"keyboard-v2 dataset mismatch: {episode_id}")
        with h5py.File(dataset, "r") as handle:
            if str(handle.attrs.get("schema", "")) != KEYBOARD_V2_SCHEMA:
                raise RuntimeError(f"keyboard-v2 schema mismatch: {episode_id}")
            stored = json.loads(str(handle.attrs["manifest_json"]))
            if (
                stored.get("schema") != KEYBOARD_V2_SCHEMA
                or stored.get("source") != KEYBOARD_SOURCE_DOMAIN
                or stored.get("source_domain") != KEYBOARD_SOURCE_DOMAIN
                or stored.get("source_hdf_sha256") != manifest["source_hdf_sha256"]
                or stored.get("contact_rows_included") is not False
                or stored.get("gt_actor_input") is not False
                or stored.get("source_episode_id") != source_episode_id
                or stored.get("source_role") != record.get("source_role")
                or stored.get("completion") != record.get("completion")
                or stored.get("stopped_reason") != record.get("stopped_reason")
            ):
                raise RuntimeError(f"keyboard-v2 episode manifest mismatch: {episode_id}")
            exclusions = _mapping(
                stored.get("training_exclusions"),
                f"{episode_id}.training_exclusions",
            )
            if any(
                exclusions.get(name) is not True
                for name in (
                    "close",
                    "contact",
                    "lift",
                    "her",
                    "her_force",
                    "cube_ground_truth_actor_input",
                )
            ):
                raise RuntimeError(f"keyboard-v2 training exclusion mismatch: {episode_id}")
            provenance = _mapping(
                stored.get("provenance"), f"{episode_id}.provenance"
            )
            if (
                provenance.get("immutable_source") is not True
                or provenance.get("source_domain") != KEYBOARD_SOURCE_DOMAIN
                or provenance.get("source_hdf_sha256")
                != manifest["source_hdf_sha256"]
                or provenance.get("collection_summary_sha256")
                != manifest["collection_summary_sha256"]
                or provenance.get("source_episode_id") != source_episode_id
                or provenance.get("source_role") != record.get("source_role")
                or provenance.get("source_row_index_field") != "source_row_index"
                or provenance.get("safe_prefix_rows_50hz")
                != record.get("safe_prefix_rows_50hz")
            ):
                raise RuntimeError(f"keyboard-v2 provenance mismatch: {episode_id}")
            contract_fingerprint = _student_contract_fingerprint(
                stored,
                expected_capture_rate_hz=expected_capture_rate_hz,
                require_action_order=True,
            )
            rows = handle.get("rows")
            if not isinstance(rows, h5py.Group):
                raise RuntimeError(f"keyboard-v2 rows missing: {episode_id}")
            if set(rows.keys()) != set(KEYBOARD_EXPORTED_ROW_DTYPES):
                raise RuntimeError(f"keyboard-v2 row inventory mismatch: {episode_id}")
            counts = set()
            for name, (trailing, dtype) in KEYBOARD_EXPORTED_ROW_DTYPES.items():
                value = rows.get(name)
                if (
                    not isinstance(value, h5py.Dataset)
                    or value.shape[1:] != trailing
                    or np.dtype(value.dtype) != np.dtype(dtype)
                ):
                    raise RuntimeError(f"keyboard-v2 field mismatch: {episode_id}:{name}")
                counts.add(int(value.shape[0]))
            if len(counts) != 1:
                raise RuntimeError(f"keyboard-v2 row cardinality mismatch: {episode_id}")
            row_count = counts.pop()
            if row_count < 2 or row_count != int(record.get("rows", -1)):
                raise RuntimeError(f"keyboard-v2 row count mismatch: {episode_id}")
            steps = np.asarray(rows["control_step"][:], dtype=np.int64)
            times = np.asarray(rows["timestamp_s"][:], dtype=np.float64)
            source_indices = np.asarray(rows["source_row_index"][:], dtype=np.int64)
            if (
                not np.all(np.diff(source_indices) == 2)
                or not np.array_equal(source_indices, steps)
                or not np.all(np.diff(steps) == 2)
                or not np.allclose(
                    times,
                    steps.astype(np.float64) * POLICY_DT_S,
                    atol=1.0e-8,
                    rtol=0.0,
                )
            ):
                raise RuntimeError(f"keyboard-v2 cadence mismatch: {episode_id}")
            action = np.asarray(rows["policy_action_4d_metric_root_m"][:], dtype=np.float32)
            previous = np.asarray(rows["previous_policy_action_4d_metric_root_m"][:], dtype=np.float32)
            expected_previous = np.concatenate((np.zeros((1, 4), np.float32), action[:-1]), axis=0)
            depth = np.asarray(rows["right_wrist_depth_m"][:], dtype=np.float32)
            depth_valid = np.asarray(rows["right_wrist_depth_valid"][:], dtype=np.bool_)
            pose = np.asarray(rows["ee_pose_robot_root_m_xyzw"][:], dtype=np.float32)
            state_values = [
                pose,
                np.asarray(rows["right_arm_joint_position_rad"][:], dtype=np.float32),
                np.asarray(rows["right_arm_joint_velocity_rad_s"][:], dtype=np.float32),
                np.asarray(rows["right_arm_joint_acceleration_rad_s2"][:], dtype=np.float32),
                previous,
            ]
            if (
                not np.isfinite(action).all()
                or not all(np.isfinite(value).all() for value in state_values)
                or np.any(action[:, 3] != 0.0)
                or np.any(np.linalg.norm(action[:, :3].astype(np.float64), axis=1) > ACTION_BOUND_M + 1.0e-8)
                or not np.allclose(previous, expected_previous, atol=1.0e-8, rtol=0.0)
                or not np.all(np.asarray(rows["gripper_state_open"][:]) == 1.0)
                or not np.isfinite(depth).all()
                or np.any((~depth_valid) & (depth != 0.0))
                or not np.allclose(
                    np.linalg.norm(pose[:, 3:7], axis=1),
                    1.0,
                    rtol=0.0,
                    atol=1.0e-4,
                )
            ):
                raise RuntimeError(f"keyboard-v2 action/open contract mismatch: {episode_id}")
        artifacts.append(
            {
                "episode_id": episode_id,
                "source_episode_id": source_episode_id,
                "collection_id": root.name,
                "source_domain": KEYBOARD_SOURCE_DOMAIN,
                "completion": record.get("completion"),
                "seed": int(record["seed"]),
                "dataset": dataset,
                "report": manifest_path,
                "sidecar": source_hdf,
                "rows": row_count,
                "dataset_sha256": record["dataset_sha256"],
                "student_contract_fingerprint": contract_fingerprint,
            }
        )
    if not artifacts:
        raise RuntimeError("keyboard-v2 export contains no eligible episodes")
    expected_counts = {
        "episode_count": len(records),
        "eligible_episode_count": sum(record.get("eligible") is True for record in records),
        "complete_episode_count": sum(
            record.get("eligible") is True and record.get("completion") == "complete"
            for record in records
        ),
        "partial_safe_episode_count": sum(
            record.get("eligible") is True
            and record.get("completion") == "partial_safe"
            for record in records
        ),
        "total_rows": sum(int(record.get("rows", 0)) for record in records),
    }
    if any(manifest.get(key) != value for key, value in expected_counts.items()):
        raise RuntimeError("keyboard-v2 manifest aggregate mismatch")
    return artifacts, manifest


class MultiFileDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, refs: list[tuple[Path, int]]) -> None:
        self.refs = refs
        self.handles: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def _rows(self, path: Path) -> h5py.Group:
        handle = self.handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self.handles[path] = handle
        return handle["rows"]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, row = self.refs[index]
        rows = self._rows(path)
        rgb = np.asarray(rows["right_wrist_rgb"][row]).transpose(2, 0, 1)
        depth = np.asarray(rows["right_wrist_depth_m"][row]).transpose(2, 0, 1)
        valid = np.asarray(rows["right_wrist_depth_valid"][row]).transpose(2, 0, 1)
        proprio = np.concatenate((
            np.asarray(rows["ee_pose_robot_root_m_xyzw"][row], dtype=np.float32),
            np.asarray(rows["right_arm_joint_position_rad"][row], dtype=np.float32),
            np.asarray(rows["right_arm_joint_velocity_rad_s"][row], dtype=np.float32),
            np.asarray(rows["right_arm_joint_acceleration_rad_s2"][row], dtype=np.float32),
            np.asarray(rows["gripper_state_open"][row], dtype=np.float32),
            np.asarray(rows["previous_policy_action_4d_metric_root_m"][row], dtype=np.float32),
        ))
        if proprio.shape != (canonical.PROPRIO_DIM,) or not np.isfinite(proprio).all():
            raise RuntimeError("invalid canonical proprioception")
        action = canonical.validate_serialized_policy_action_4d_metric_root_m(rows["policy_action_4d_metric_root_m"][row])
        return {"rgb": torch.from_numpy(rgb.copy()), "depth": torch.from_numpy(depth.copy()).float(), "depth_valid": torch.from_numpy(valid.copy()).float(), "proprio": torch.from_numpy(proprio.copy()), "action": torch.from_numpy(action.copy())}

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _metric_accumulator() -> dict[str, Any]:
    return {"rows": 0, "sq_coord_error": 0.0, "sq_vector_error": 0.0, "sum_norm_error": 0.0, "bound_violations": 0, "direction_deg": [], "target_zero": 0, "prediction_zero": 0, "finite": True}


def _update_metrics(acc: dict[str, Any], prediction: torch.Tensor, target: torch.Tensor) -> None:
    pred = prediction[:, :3].detach().float().cpu().numpy()
    truth = target[:, :3].detach().float().cpu().numpy()
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        acc["finite"] = False
        return
    error = pred - truth
    acc["rows"] += int(len(error))
    acc["sq_coord_error"] += float(np.square(error).sum())
    acc["sq_vector_error"] += float(np.square(error).sum(axis=1).sum())
    acc["sum_norm_error"] += float(np.abs(np.linalg.norm(pred, axis=1) - np.linalg.norm(truth, axis=1)).sum())
    acc["bound_violations"] += int(np.sum(np.linalg.norm(pred, axis=1) > ACTION_BOUND_M + 1.0e-9))
    truth_norm = np.linalg.norm(truth, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    acc["target_zero"] += int(np.sum(truth_norm <= 1.0e-7))
    acc["prediction_zero"] += int(np.sum(pred_norm <= 1.0e-7))
    valid = truth_norm > 1.0e-7
    if np.any(valid):
        cosine = np.sum(pred[valid] * truth[valid], axis=1) / np.maximum(pred_norm[valid] * truth_norm[valid], 1.0e-12)
        acc["direction_deg"].extend(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).tolist())


def _finish_metrics(acc: dict[str, Any]) -> dict[str, Any]:
    rows = int(acc["rows"])
    directions = np.asarray(acc["direction_deg"], dtype=np.float64)
    return {
        "rows": rows,
        "finite": bool(acc["finite"]),
        "xyz_coordinate_rmse_mm": math.sqrt(acc["sq_coord_error"] / max(rows * 3, 1)) * 1000.0,
        "vector_norm_rmse_mm": math.sqrt(acc["sq_vector_error"] / max(rows, 1)) * 1000.0,
        "action_norm_error_mm": acc["sum_norm_error"] / max(rows, 1) * 1000.0,
        "direction_valid_count": int(len(directions)),
        "target_zero_count": int(acc["target_zero"]),
        "prediction_zero_count": int(acc["prediction_zero"]),
        "direction_mean_deg": float(np.mean(directions)) if len(directions) else math.nan,
        "direction_p95_deg": float(np.percentile(directions, 95)) if len(directions) else math.nan,
        "4.5mm_bound_violation_count": int(acc["bound_violations"]),
    }


def _run_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    model.train(optimizer is not None)
    total_loss = 0.0
    count = 0
    acc = _metric_accumulator()
    for batch in loader:
        image = canonical._model_input(batch, device)
        proprio = batch["proprio"].to(device, non_blocking=True).float()
        target = batch["action"].to(device, non_blocking=True).float()
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(image, proprio)
            loss = F.mse_loss(prediction[:, :3], target[:, :3])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite BC loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        n = int(target.shape[0])
        total_loss += float(loss.detach().cpu()) * n
        count += n
        _update_metrics(acc, prediction, target)
    if count == 0:
        raise RuntimeError("empty grouped BC loader")
    result = _finish_metrics(acc)
    result["bc_loss"] = total_loss / count
    return result


def _refs(artifacts: list[dict[str, Any]]) -> list[tuple[Path, int]]:
    refs: list[tuple[Path, int]] = []
    for item in artifacts:
        with h5py.File(item["dataset"], "r") as handle:
            rows = int(handle["rows/policy_action_4d_metric_root_m"].shape[0])
        refs.extend((item["dataset"], index) for index in range(rows))
    return refs


def _evaluate_model(model: torch.nn.Module, artifacts: list[dict[str, Any]], device: torch.device, batch_size: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    model.eval()
    aggregate = _metric_accumulator()
    per_seed: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for item in artifacts:
            dataset = MultiFileDataset([(item["dataset"], i) for i in range(item["rows"])])
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
            local = _metric_accumulator()
            for batch in loader:
                prediction = model(canonical._model_input(batch, device), batch["proprio"].to(device).float())
                _update_metrics(local, prediction, batch["action"].to(device).float())
            dataset.close()
            local_result = _finish_metrics(local)
            per_seed[item["episode_id"]] = {"seed": item["seed"], **local_result}
            # Merge via synthetic arrays is unnecessary; add scalar fields and angles.
            for key in ("rows", "sq_coord_error", "sq_vector_error", "sum_norm_error", "bound_violations", "target_zero", "prediction_zero"):
                aggregate[key] += local[key]
            aggregate["finite"] = aggregate["finite"] and local["finite"]
            aggregate["direction_deg"].extend(local["direction_deg"])
    return _finish_metrics(aggregate), per_seed


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("action_dim") not in (None, 4):
        raise RuntimeError("baseline checkpoint is not a 4-D canonical BC checkpoint")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint state_dict missing")
    model.load_state_dict(state, strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--keyboard-v2-root",
        action="append",
        type=Path,
        default=[],
        help="repeatable g2_keyboard_contact_free_v2 export root",
    )
    parser.add_argument(
        "--expected-attempts",
        action="append",
        type=int,
        help="optional per-root schedule cardinality; repeat in collection-root order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--capture-rate-hz", type=int, choices=(25,), default=25)
    parser.add_argument("--allow-partial-collection", action="store_true")
    parser.add_argument("--minimum-valid-episodes", type=int, default=5)
    args = parser.parse_args()
    if not args.collection_root and not args.keyboard_v2_root:
        raise SystemExit("AT_LEAST_ONE_CUROBO_OR_KEYBOARD_V2_SOURCE_IS_REQUIRED")
    if args.epochs != 5 or args.batch_size != 256 or args.folds != 5:
        raise SystemExit("GROUPED_CONTACT_FREE_CV_REQUIRES_5_EPOCHS_BATCH_256_FOLDS_5")
    if args.output.exists():
        raise SystemExit("OUTPUT_MUST_BE_NEW_IMMUTABLE_DIRECTORY")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUESTED_BUT_UNAVAILABLE")
    if args.minimum_valid_episodes < args.folds:
        raise SystemExit("MINIMUM_VALID_EPISODES_MUST_COVER_ALL_FOLDS")
    expected_attempts = args.expected_attempts or [None] * len(args.collection_root)
    if len(expected_attempts) != len(args.collection_root):
        raise SystemExit("COLLECTION_ROOT_EXPECTED_ATTEMPTS_CARDINALITY_MISMATCH")
    discovered = [
        _discover(
            root.resolve(),
            args.capture_rate_hz,
            expected_attempts=expected,
            allow_partial=args.allow_partial_collection,
            minimum_valid_episodes=args.minimum_valid_episodes,
        )
        for root, expected in zip(
            args.collection_root, expected_attempts, strict=True
        )
    ]
    keyboard_discovered = [
        _discover_keyboard_v2(root.resolve(), args.capture_rate_hz)
        for root in args.keyboard_v2_root
    ]
    artifacts = [item for values, _manifest in discovered for item in values]
    artifacts.extend(
        item for values, _manifest in keyboard_discovered for item in values
    )
    collection_manifests = [manifest for _values, manifest in discovered]
    if len({item["episode_id"] for item in artifacts}) != len(artifacts):
        raise SystemExit("CROSS_COLLECTION_EPISODE_ID_COLLISION")
    if len(artifacts) < args.minimum_valid_episodes:
        raise SystemExit("INSUFFICIENT_COMBINED_FUNCTIONAL_EPISODES")
    fingerprints = {
        json.dumps(
            item["student_contract_fingerprint"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for item in artifacts
    }
    if len(fingerprints) != 1:
        raise SystemExit("CUROBO_KEYBOARD_STUDENT_CONTRACT_MISMATCH")
    shared_student_contract = artifacts[0]["student_contract_fingerprint"]
    ranked = sorted(artifacts, key=lambda item: hashlib.sha256(item["episode_id"].encode()).hexdigest())
    folds: list[list[dict[str, Any]]] = [[] for _ in range(args.folds)]
    for index, item in enumerate(ranked):
        folds[index % args.folds].append(item)
    device = torch.device(args.device)
    args.output.mkdir(parents=True)
    source_manifest_sha256 = {
        str(root.resolve()): sha256(root.resolve() / "COLLECTION_MANIFEST.json")
        for root in args.collection_root
    }
    keyboard_manifest_sha256 = {
        str(root.resolve()): sha256(root.resolve() / "KEYBOARD_V2_MANIFEST.json")
        for root in args.keyboard_v2_root
    }
    split_payload = {"folds": [[item["episode_id"] for item in fold] for fold in folds], "policy_dt_s": POLICY_DT_S, "physics_dt_s": PHYSICS_DT_S, "split_unit": "whole_rollout_episode", "source_manifest_sha256": source_manifest_sha256, "keyboard_v2_manifest_sha256": keyboard_manifest_sha256, "shared_student_contract": shared_student_contract, "student_count_per_fold": 1}
    atomic_json(args.output / "CV_SPLIT.json", split_payload)
    baseline_result = None
    if args.baseline_checkpoint is not None:
        baseline_model = canonical.CanonicalContactFreeVisualBC().to(device)
        _load_checkpoint(baseline_model, args.baseline_checkpoint.resolve(), device)
        baseline_result = {"checkpoint": str(args.baseline_checkpoint.resolve()), "heldout_by_fold": []}
        for fold in folds:
            aggregate, per_seed = _evaluate_model(baseline_model, fold, device, args.batch_size)
            baseline_result["heldout_by_fold"].append({"episodes": [i["episode_id"] for i in fold], "aggregate": aggregate, "per_seed": per_seed})
        del baseline_model
    fold_results: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(folds):
        heldout_ids = {item["episode_id"] for item in heldout}
        train_artifacts = [item for item in artifacts if item["episode_id"] not in heldout_ids]
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)
        model = canonical.CanonicalContactFreeVisualBC().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        train_loader = DataLoader(MultiFileDataset(_refs(train_artifacts)), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
        history: list[dict[str, Any]] = []
        best_loss = math.inf
        best_path = args.output / f"fold_{fold_index:02d}_best.pt"
        for epoch in range(1, args.epochs + 1):
            train_metric = _run_epoch(model, train_loader, device, optimizer)
            valid_metric, per_seed = _evaluate_model(model, heldout, device, args.batch_size)
            history.append({"epoch": epoch, "train": train_metric, "heldout": valid_metric})
            validation_score = float(valid_metric["xyz_coordinate_rmse_mm"])
            if validation_score < best_loss:
                best_loss = validation_score
                torch.save({"schema": "g2_independent_contact_free_bc_v1", "action_dim": 4, "action_contract": "[dx,dy,dz,HOLD_OPEN=0],robot_root,m", "policy_dt_s": POLICY_DT_S, "physics_dt_s": PHYSICS_DT_S, "state_dict": model.state_dict(), "epoch": epoch, "heldout_episode_ids": sorted(heldout_ids), "train_episode_ids": sorted(i["episode_id"] for i in train_artifacts)}, best_path)
        train_loader.dataset.close()  # type: ignore[attr-defined]
        # Reload the selected checkpoint in a fresh model before publishing
        # fold metrics.  A good in-memory score is not enough if the persisted
        # artifact is truncated or has an incompatible architecture.
        reloaded = canonical.CanonicalContactFreeVisualBC().to(device)
        checkpoint_payload = torch.load(best_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint_payload, dict) or checkpoint_payload.get("action_dim") != 4:
            raise RuntimeError("fold checkpoint action schema is not 4-D")
        reloaded.load_state_dict(checkpoint_payload["state_dict"], strict=True)
        final_metric, final_per_seed = _evaluate_model(reloaded, heldout, device, args.batch_size)
        fold_results.append({"fold": fold_index, "train_episode_ids": sorted(i["episode_id"] for i in train_artifacts), "heldout_episode_ids": sorted(heldout_ids), "history": history, "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256(best_path), "checkpoint_reload": "PASS", "heldout_aggregate": final_metric, "heldout_per_seed": final_per_seed})
        del model, optimizer
        del reloaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    failed_attempts: list[dict[str, Any]] = []
    collection_summaries: list[dict[str, Any]] = []
    for root, manifest in zip(args.collection_root, collection_manifests, strict=True):
        collection_summaries.append({
            "root": str(root.resolve()),
            "manifest_sha256": source_manifest_sha256[str(root.resolve())],
            "status": manifest.get("status"),
            "attempt_count": len(manifest.get("runs", [])),
            "functional_pass_count": int(manifest.get("functional_pass_count", 0)),
            "functional_fail_count": int(manifest.get("functional_fail_count", 0)),
        })
        failed_attempts.extend(
            {
                "collection_id": root.resolve().name,
                "episode_id": r.get("episode_id"),
                "seed": r.get("seed"),
                "process_verdict": r.get("process_verdict"),
                "row_count": r.get("row_count"),
                "functional_report": r.get("functional_report"),
            }
            for r in manifest.get("runs", [])
            if not r.get("functional_pass")
        )
    keyboard_summaries = [
        {
            "root": str(root.resolve()),
            "manifest_sha256": keyboard_manifest_sha256[str(root.resolve())],
            "eligible_episode_count": int(manifest.get("eligible_episode_count", 0)),
            "complete_episode_count": int(manifest.get("complete_episode_count", 0)),
            "partial_safe_episode_count": int(manifest.get("partial_safe_episode_count", 0)),
            "total_rows": int(manifest.get("total_rows", 0)),
        }
        for root, (_items, manifest) in zip(
            args.keyboard_v2_root, keyboard_discovered, strict=True
        )
    ]
    report = {"schema": "g2_independent_contact_free_bc_cv_v3", "collections": collection_summaries, "keyboard_v2_collections": keyboard_summaries, "collection_attempt_count": sum(value["attempt_count"] for value in collection_summaries), "collection_functional_pass_count": sum(value["functional_pass_count"] for value in collection_summaries), "collection_functional_fail_count": sum(value["functional_fail_count"] for value in collection_summaries), "failed_attempts": failed_attempts, "contract": {"action_dim": 4, "frame": "robot_root", "unit": "m", "gripper": "HOLD_OPEN=0", "policy_hz": 50, "capture_rate_hz": args.capture_rate_hz, "capture_control_step_stride": 50 // args.capture_rate_hz, "physics_dt_s": PHYSICS_DT_S, "max_delta_m": ACTION_BOUND_M, "contact_training": False, "close_training": False, "lift_training": False, "her_training": False, "her_force_training": False, "cube_ground_truth_actor_input": False}, "shared_student_contract": shared_student_contract, "student_count_per_fold": 1, "artifact_count": len(artifacts), "total_rows": sum(i["rows"] for i in artifacts), "source_domain_counts": {domain: sum(item.get("source_domain") == domain for item in artifacts) for domain in (CUROBO_SOURCE_DOMAIN, KEYBOARD_SOURCE_DOMAIN)}, "baseline_temporal_reference": {"single_rollout_val_xyz_rmse_mm": 2.5589, "single_rollout_val_direction_deg": 55.8358, "interpretation": "historical single-rollout temporal holdout only"}, "baseline_checkpoint_evaluation": baseline_result, "cross_validation": fold_results, "source_hashes": canonical._source_hashes(), "training_scope": "BC_ONLY_CONTACT_FREE; SAC/HER/HER_FORCE_DISABLED", "robot_initial_state_generalization": "NOT_TESTED_SOURCE_RESET_ARM_POSE_FIXED"}
    atomic_json(args.output / "CV_REPORT.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
