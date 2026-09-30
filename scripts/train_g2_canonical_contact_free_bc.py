#!/usr/bin/env python3
"""Offline five-epoch BC warm-start for canonical Candidate-A OPEN-only rows.

This entry point is deliberately narrower than the historical keyboard BC
trainer.  It accepts exactly one published ``g2_canonical_contact_free_collection_v2``
HDF5 artifact and its functional receipt, consumes Wrist RGB-D plus the
declared deployable robot state, and predicts only ``[dx, dy, dz, HOLD_OPEN]``.
It never imports Isaac or sends an action to a simulator/controller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# This executable is intentionally runnable from a fresh child process.  The
# canonical collection contract lives under the repository ``source`` tree,
# which is not an installed package in the frozen Isaac Python environment.
# Resolve it from this file rather than requiring caller cwd/PYTHONPATH state.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source"
if not SOURCE_ROOT.is_dir():
    raise RuntimeError(f"canonical BC source root unavailable: {SOURCE_ROOT}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geniesim.rl.isaaclab.g2_policy_branch.canonical_contact_free_collection_v2 import (
    ACTOR_FIELDS,
    CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA,
    CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA as COLLECTION_SCHEMA,
    CANONICAL_FRAME,
    CANONICAL_QUATERNION_ORDER,
    PacketEpochReceipt,
    SERIALIZED_FLOAT32_ACTION_ROUNDTRIP_TOLERANCE_M,
    validate_serialized_policy_action_4d_metric_root_m,
)
from geniesim.rl.isaaclab.g2_policy_branch.contact_free_training_contract import (
    CONTACT_FREE_MAX_DELTA_M,
)


SCHEMA = "g2_canonical_contact_free_visual_bc_v1"
LAUNCH_SEAL_SCHEMA = f"{SCHEMA}_launch_seal_v1"
CHECKPOINT_ATTESTATION_SCHEMA = f"{SCHEMA}_checkpoint_reload_attestation_v1"
FIRST_BC_EPOCHS = 5
FIRST_BC_BATCH_SIZE = 256
FIRST_BC_SEED = 42
FIRST_BC_DEVICE = "cuda:0"
TRAIN_FRACTION = 0.90
POLICY_DT_S = 0.020
PHYSICS_DT_S = 0.002
IMAGE_HW = (48, 64)
PROPRIO_DIM = 33  # EE XYZ+XYZW, arm q/qd/qdd, OPEN state, prior 4-D packet.
# The live policy contract remains exactly 4.5 mm.  The network emits float32,
# so use the immediately lower representable scalar for its closed output
# surface.  This avoids a harmless IEEE-754 overshoot from becoming a false
# acceptance-gate violation; it does not enlarge a command or alter the live
# action adapter.
MODEL_OUTPUT_MAX_DELTA_M = float(
    np.nextafter(np.float32(CONTACT_FREE_MAX_DELTA_M), np.float32(0.0))
)
REQUIRED_DATASETS = {
    "timestamp_s": ((None,), np.float64),
    "control_step": ((None,), np.int64),
    "policy_action_4d_metric_root_m": ((None, 4), np.float32),
    "previous_policy_action_4d_metric_root_m": ((None, 4), np.float32),
    "ee_pose_robot_root_m_xyzw": ((None, 7), np.float32),
    "right_arm_joint_position_rad": ((None, 7), np.float32),
    "right_arm_joint_velocity_rad_s": ((None, 7), np.float32),
    "right_arm_joint_acceleration_rad_s2": ((None, 7), np.float32),
    "gripper_state_open": ((None, 1), np.float32),
    "right_wrist_rgb": ((None, 192, 256, 3), np.uint8),
    "right_wrist_depth_m": ((None, 192, 256, 1), np.float32),
    "right_wrist_depth_valid": ((None, 192, 256, 1), np.bool_),
    "packet_receipt_json": ((None,), None),
}


class CanonicalBCContractError(ValueError):
    """The source artifact is not an eligible canonical OPEN-only dataset."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: dict[str, object]) -> str:
    """Persist one checkpoint under a new name and return its content hash.

    The caller has already rejected an existing output directory.  A checkpoint
    is therefore immutable from the trainer's point of view: every epoch gets
    an atomically replaced *new* representation, and its SHA-256 is recorded
    before it can be promoted.
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    digest = _sha256_file(temporary)
    os.replace(temporary, path)
    return digest


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "bc_entrypoint": Path(__file__).resolve(),
        "canonical_collection_schema": (
            root
            / "source/geniesim/rl/isaaclab/g2_policy_branch/"
            "canonical_contact_free_collection_v2.py"
        ),
        "training_contract": (
            root
            / "source/geniesim/rl/isaaclab/g2_policy_branch/"
            "contact_free_training_contract.py"
        ),
    }
    if not all(path.is_file() for path in paths.values()):
        raise CanonicalBCContractError("BC source provenance input missing")
    return {name: _sha256_file(path) for name, path in paths.items()}


@dataclass(frozen=True)
class CanonicalArtifactAudit:
    dataset: Path
    functional_report: Path
    dataset_sha256: str
    functional_report_sha256: str
    rows: int
    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]
    maximum_serialized_action_norm_m: float
    source_hashes: dict[str, str]

    def payload(self) -> dict[str, object]:
        return {
            "dataset": str(self.dataset),
            "functional_report": str(self.functional_report),
            "dataset_sha256": self.dataset_sha256,
            "functional_report_sha256": self.functional_report_sha256,
            "rows": self.rows,
            "train_rows": len(self.train_rows),
            "validation_rows": len(self.validation_rows),
            "maximum_serialized_action_norm_m": self.maximum_serialized_action_norm_m,
            "serialized_float32_roundtrip_tolerance_m": (
                SERIALIZED_FLOAT32_ACTION_ROUNDTRIP_TOLERANCE_M
            ),
            "source_hashes": dict(self.source_hashes),
        }


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CanonicalBCContractError(f"{label} must be a JSON object")
    return value


def _sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CanonicalBCContractError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise CanonicalBCContractError(f"{label} must be hexadecimal SHA-256")
    return normalized


def _resolve_frozen_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CanonicalBCContractError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise CanonicalBCContractError(f"{label} file is missing: {path}")
    return path


def _functional_freeze_inputs(functional_report: Path) -> dict[str, object]:
    """Read the contact-free functional receipt as immutable launch evidence.

    This deliberately binds only the inputs that still matter to the offline
    learner: Candidate-A/Production asset identities, frozen trajectory, and
    the complete before/after source-freeze verdict.  It does not claim that a
    historical live runner's source tree is still the current BC source tree;
    the latter is separately sealed by ``_source_hashes`` at preflight.
    """

    try:
        payload = json.loads(functional_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalBCContractError("functional receipt cannot be decoded for freeze") from error
    payload = _mapping(payload, "functional receipt")
    freeze_before = _mapping(payload.get("source_freeze_before"), "source_freeze_before")
    freeze_after = _mapping(payload.get("source_freeze_after"), "source_freeze_after")
    for label, freeze in (("source_freeze_before", freeze_before), ("source_freeze_after", freeze_after)):
        if freeze.get("SOURCE_FREEZE") != "PASS" or freeze.get("SOURCE_FREEZE_COMPLETE") is not True:
            raise CanonicalBCContractError(f"{label} is not a complete PASS freeze")

    asset_binding = _mapping(payload.get("asset_binding"), "asset_binding")
    candidate_path = _resolve_frozen_path(
        asset_binding.get("candidate_asset_path"), "Candidate-A asset"
    )
    production_path = _resolve_frozen_path(
        asset_binding.get("production_asset_path"), "Production USD"
    )
    candidate_hash = _sha256_value(
        asset_binding.get("candidate_asset_sha256"), "Candidate-A asset SHA-256"
    )
    production_hash = _sha256_value(
        asset_binding.get("production_asset_sha256"), "Production USD SHA-256"
    )

    def freeze_scope(label: str, freeze: dict[str, object]) -> tuple[dict[str, object], str]:
        scope = _mapping(freeze.get("freeze_scope"), f"{label}.freeze_scope")
        asset = _mapping(scope.get("asset"), f"{label}.freeze_scope.asset")
        if (
            _sha256_value(asset.get("candidate_a_composed_usd_sha256"), f"{label} Candidate-A SHA-256")
            != candidate_hash
            or _sha256_value(asset.get("production_usd_sha256"), f"{label} Production SHA-256")
            != production_hash
        ):
            raise CanonicalBCContractError(f"{label} asset freeze conflicts with asset binding")
        planner = _mapping(scope.get("planner"), f"{label}.freeze_scope.planner")
        trajectory_hash = _sha256_value(
            planner.get("trajectory_hash"), f"{label} frozen trajectory SHA-256"
        )
        if _sha256_value(freeze.get("frozen_trajectory_sha256"), f"{label} frozen trajectory") != trajectory_hash:
            raise CanonicalBCContractError(f"{label} frozen trajectory hash is inconsistent")
        return scope, trajectory_hash

    _, trajectory_before = freeze_scope("source_freeze_before", freeze_before)
    _, trajectory_after = freeze_scope("source_freeze_after", freeze_after)
    if trajectory_before != trajectory_after:
        raise CanonicalBCContractError("functional receipt trajectory freeze changed during run")

    frozen = _mapping(freeze_after.get("frozen"), "source_freeze_after.frozen")
    trajectory_paths = []
    for raw_path, receipt in frozen.items():
        item = _mapping(receipt, f"frozen receipt {raw_path}")
        expected = _sha256_value(item.get("expected"), f"frozen receipt {raw_path}.expected")
        actual = _sha256_value(item.get("actual"), f"frozen receipt {raw_path}.actual")
        if item.get("match") is not True or expected != actual:
            raise CanonicalBCContractError(f"functional source freeze contains a non-matching input: {raw_path}")
        if expected == trajectory_after:
            trajectory_paths.append(_resolve_frozen_path(raw_path, "frozen trajectory"))
    if len(trajectory_paths) != 1:
        raise CanonicalBCContractError("functional receipt does not identify exactly one frozen trajectory path")
    trajectory_path = trajectory_paths[0]

    actual_inputs = {
        "candidate_a": {"path": str(candidate_path), "sha256": _sha256_file(candidate_path)},
        "production_usd": {"path": str(production_path), "sha256": _sha256_file(production_path)},
        "frozen_trajectory": {"path": str(trajectory_path), "sha256": _sha256_file(trajectory_path)},
    }
    expected_inputs = {
        "candidate_a": {"path": str(candidate_path), "sha256": candidate_hash},
        "production_usd": {"path": str(production_path), "sha256": production_hash},
        "frozen_trajectory": {"path": str(trajectory_path), "sha256": trajectory_after},
    }
    if actual_inputs != expected_inputs:
        raise CanonicalBCContractError("functional asset/trajectory source freeze drifted before BC launch")
    return {
        "functional_report_source_freeze": "PASS_BEFORE_AND_AFTER",
        "inputs": expected_inputs,
    }


def build_launch_seal(*, audit: CanonicalArtifactAudit, learning_rate: float) -> dict[str, object]:
    """Create the immutable preflight artifact consumed by the actual launch."""

    if learning_rate <= 0.0:
        raise CanonicalBCContractError("learning rate must be positive")
    return {
        "schema": LAUNCH_SEAL_SCHEMA,
        "scope": "OFFLINE_CANONICAL_CANDIDATE_A_CONTACT_FREE_BC_ONLY",
        "launch_plan": {
            **first_launch_manifest(
                audit=audit, epochs=FIRST_BC_EPOCHS, batch_size=FIRST_BC_BATCH_SIZE
            ),
            "learning_rate": float(learning_rate),
            "device": FIRST_BC_DEVICE,
            "resume": "DISABLED_FRESH_OUTPUT_ONLY",
        },
        "input_freeze": {
            "dataset": str(audit.dataset),
            "dataset_sha256": audit.dataset_sha256,
            "functional_report": str(audit.functional_report),
            "functional_report_sha256": audit.functional_report_sha256,
            "current_bc_source_hashes": dict(audit.source_hashes),
            **_functional_freeze_inputs(audit.functional_report),
        },
    }


def verify_launch_seal(
    *, audit: CanonicalArtifactAudit, launch_seal: Path, learning_rate: float
) -> dict[str, object]:
    """Fail closed unless the current launch still equals the sealed preflight."""

    if not launch_seal.is_file():
        raise CanonicalBCContractError("BC launch seal is missing")
    try:
        observed = json.loads(launch_seal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalBCContractError("BC launch seal cannot be decoded") from error
    expected = build_launch_seal(audit=audit, learning_rate=learning_rate)
    if observed != expected:
        raise CanonicalBCContractError("BC launch seal differs from current input/source/asset/trajectory freeze")
    return expected


def cuda_zero_preflight_receipt() -> dict[str, object]:
    """Read-only CUDA:0 attestation, called only immediately before training."""

    if not torch.cuda.is_available():
        raise CanonicalBCContractError("CUDA:0 requested but torch.cuda.is_available() is false")
    count = int(torch.cuda.device_count())
    if count < 1:
        raise CanonicalBCContractError("CUDA:0 requested but no CUDA device is visible")
    capability = torch.cuda.get_device_capability(0)
    if not isinstance(capability, tuple) or len(capability) != 2:
        raise CanonicalBCContractError("CUDA:0 capability receipt is malformed")
    major, minor = (int(capability[0]), int(capability[1]))
    name = str(torch.cuda.get_device_name(0)).strip()
    if not name:
        raise CanonicalBCContractError("CUDA:0 device name is empty")
    return {
        "device": FIRST_BC_DEVICE,
        "visible_device_count": count,
        "device_name": name,
        "capability": [major, minor],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _expect_dataset(group: h5py.Group, name: str, expected_shape: tuple[int | None, ...], dtype: np.dtype | None) -> int:
    if name not in group:
        raise CanonicalBCContractError(f"canonical HDF5 missing dataset: {name}")
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset):
        raise CanonicalBCContractError(f"canonical HDF5 field is not a dataset: {name}")
    shape = dataset.shape
    if len(shape) != len(expected_shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(shape, expected_shape, strict=True)
    ):
        raise CanonicalBCContractError(
            f"canonical HDF5 shape mismatch for {name}: {shape} != {expected_shape}"
        )
    if dtype is not None and np.dtype(dataset.dtype) != np.dtype(dtype):
        raise CanonicalBCContractError(
            f"canonical HDF5 dtype mismatch for {name}: {dataset.dtype} != {dtype}"
        )
    return int(shape[0])


def _validate_functional_report(dataset: Path, report: Path, dataset_sha256: str) -> None:
    if not report.is_file():
        raise CanonicalBCContractError("functional receipt is missing")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalBCContractError("functional receipt cannot be decoded") from error
    verdict = payload.get("verdict", {})
    if (
        verdict.get("CANDIDATE_A_CONTACT_FREE_SMOKE") != "PASS"
        or verdict.get("CANONICAL_CONTACT_FREE_COLLECTION") != "PASS"
        or verdict.get("FUNCTIONAL_VERDICT") != "PASS"
        or verdict.get("TRAINING_AUTHORIZED") != "NO"
    ):
        raise CanonicalBCContractError("functional receipt is not a PASS contact-free collection")
    collection = payload.get("canonical_contact_free_collection")
    if not isinstance(collection, dict) or collection.get("state") != "PUBLISHED":
        raise CanonicalBCContractError("functional receipt lacks published collection authority")
    if Path(str(collection.get("path", ""))).resolve() != dataset.resolve():
        raise CanonicalBCContractError("functional receipt dataset path mismatch")
    if collection.get("sha256") != dataset_sha256:
        raise CanonicalBCContractError("functional receipt dataset hash mismatch")
    if int(collection.get("row_count", -1)) <= 0:
        raise CanonicalBCContractError("functional receipt has no canonical rows")
    if (
        payload.get("replay_close_row_count") != 0
        or payload.get("replay_contact_row_count") != 0
        or payload.get("replay_bc_row_count") != 0
    ):
        raise CanonicalBCContractError("functional receipt contains prohibited close/contact/BC rows")


def _temporal_tail_split(rows: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A transparent 90/10 chronological holdout for the single rollout.

    This is deliberately labelled as a temporal heldout, not an independent
    episode validation: Candidate-A currently supplies one accepted rollout.
    """

    validation = max(1, int(math.ceil(rows * (1.0 - TRAIN_FRACTION))))
    if rows - validation < 1:
        raise CanonicalBCContractError("canonical rollout is too short for train/validation")
    return tuple(range(0, rows - validation)), tuple(range(rows - validation, rows))


def audit_canonical_artifact(*, dataset: Path, functional_report: Path) -> CanonicalArtifactAudit:
    dataset = dataset.resolve()
    functional_report = functional_report.resolve()
    if not dataset.is_file():
        raise CanonicalBCContractError("canonical HDF5 dataset is missing")
    dataset_sha256 = _sha256_file(dataset)
    _validate_functional_report(dataset, functional_report, dataset_sha256)
    with h5py.File(dataset, "r") as handle:
        if str(handle.attrs.get("schema", "")) != COLLECTION_SCHEMA:
            raise CanonicalBCContractError("canonical HDF5 schema mismatch")
        try:
            manifest = json.loads(str(handle.attrs["manifest_json"]))
        except (KeyError, json.JSONDecodeError) as error:
            raise CanonicalBCContractError("canonical HDF5 manifest missing/corrupt") from error
        if (
            manifest.get("schema") != CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA
            or tuple(manifest.get("actor_fields", ())) != ACTOR_FIELDS
            or manifest.get("action", {}).get("field")
            != "policy_action_4d_metric_root_m"
            or manifest.get("action", {}).get("frame") != CANONICAL_FRAME
            or manifest.get("state", {}).get("policy_dt_s") != POLICY_DT_S
            or manifest.get("state", {}).get("physics_dt_s") != PHYSICS_DT_S
            or manifest.get("state", {}).get("ee") != "robot_root,m,XYZW"
        ):
            raise CanonicalBCContractError("canonical HDF5 manifest contract mismatch")
        group = handle.get("rows")
        if not isinstance(group, h5py.Group):
            raise CanonicalBCContractError("canonical HDF5 rows group missing")
        counts = {
            name: _expect_dataset(group, name, shape, dtype)
            for name, (shape, dtype) in REQUIRED_DATASETS.items()
        }
        if len(set(counts.values())) != 1 or not counts:
            raise CanonicalBCContractError("canonical HDF5 row cardinality mismatch")
        rows = next(iter(counts.values()))
        if rows <= 1:
            raise CanonicalBCContractError("canonical HDF5 needs at least two rows")
        capture_contract = manifest.get("dataset_capture_contract", {})
        if not isinstance(capture_contract, dict):
            raise CanonicalBCContractError("dataset capture contract must be an object")
        capture_rate_hz = int(capture_contract.get("capture_rate_hz", -1))
        control_rate_hz = int(capture_contract.get("control_rate_hz", 50))
        capture_stride = int(capture_contract.get("capture_control_step_stride", 1))
        if (
            capture_rate_hz != 25
            or control_rate_hz != 50
            or capture_stride != 2
        ):
            raise CanonicalBCContractError("unsupported capture/control cadence contract")
        expected_steps = np.arange(rows, dtype=np.int64) * capture_stride
        steps = np.asarray(group["control_step"][:], dtype=np.int64)
        timestamps = np.asarray(group["timestamp_s"][:], dtype=np.float64)
        if not np.array_equal(steps, expected_steps) or not np.allclose(
            timestamps,
            expected_steps.astype(np.float64) * POLICY_DT_S,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise CanonicalBCContractError("canonical HDF5 control timebase mismatch")
        maximum_norm = 0.0
        previous_expected = np.zeros(4, dtype=np.float32)
        for start in range(0, rows, 8):
            stop = min(rows, start + 8)
            action = np.asarray(group["policy_action_4d_metric_root_m"][start:stop])
            previous = np.asarray(
                group["previous_policy_action_4d_metric_root_m"][start:stop]
            )
            depth = np.asarray(group["right_wrist_depth_m"][start:stop])
            valid = np.asarray(group["right_wrist_depth_valid"][start:stop])
            gripper = np.asarray(group["gripper_state_open"][start:stop])
            if not np.isfinite(depth).all() or not np.all(depth[~valid] == 0.0):
                raise CanonicalBCContractError("canonical HDF5 depth validity contract mismatch")
            if not np.all(gripper == 1.0):
                raise CanonicalBCContractError("canonical HDF5 contains non-OPEN gripper state")
            expected_previous = np.concatenate((previous_expected[None], action[:-1]), axis=0)
            if not np.allclose(previous, expected_previous, rtol=0.0, atol=1.0e-7):
                raise CanonicalBCContractError("canonical prior-action chain mismatch")
            previous_expected = action[-1]
            for row, raw_receipt in zip(action, group["packet_receipt_json"][start:stop], strict=True):
                stored = validate_serialized_policy_action_4d_metric_root_m(row)
                receipt = PacketEpochReceipt(**json.loads(raw_receipt))
                expected_metric = np.asarray(receipt.full8_packet[:3], dtype=np.float64) * 0.0225
                if not np.allclose(
                    stored[:3].astype(np.float64), expected_metric, rtol=0.0, atol=1.0e-9
                ):
                    raise CanonicalBCContractError("canonical label/packet scale-once mismatch")
                maximum_norm = max(
                    maximum_norm,
                    float(np.linalg.norm(stored[:3].astype(np.float64))),
                )
        train_rows, validation_rows = _temporal_tail_split(rows)
    return CanonicalArtifactAudit(
        dataset=dataset,
        functional_report=functional_report,
        dataset_sha256=dataset_sha256,
        functional_report_sha256=_sha256_file(functional_report),
        rows=rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        maximum_serialized_action_norm_m=maximum_norm,
        source_hashes=_source_hashes(),
    )


class CanonicalHDF5Dataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy reader limited to canonical actor inputs and 4-D OPEN labels."""

    def __init__(self, dataset: Path, indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self._handle: h5py.File | None = None

    def _rows(self) -> h5py.Group:
        if self._handle is None:
            self._handle = h5py.File(self.dataset, "r")
        return self._handle["rows"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        rows = self._rows()
        index = self.indices[item]
        rgb = np.asarray(rows["right_wrist_rgb"][index]).transpose(2, 0, 1)
        depth = np.asarray(rows["right_wrist_depth_m"][index]).transpose(2, 0, 1)
        valid = np.asarray(rows["right_wrist_depth_valid"][index]).transpose(2, 0, 1)
        proprio = np.concatenate(
            (
                np.asarray(rows["ee_pose_robot_root_m_xyzw"][index], dtype=np.float32),
                np.asarray(rows["right_arm_joint_position_rad"][index], dtype=np.float32),
                np.asarray(rows["right_arm_joint_velocity_rad_s"][index], dtype=np.float32),
                np.asarray(rows["right_arm_joint_acceleration_rad_s2"][index], dtype=np.float32),
                np.asarray(rows["gripper_state_open"][index], dtype=np.float32),
                np.asarray(
                    rows["previous_policy_action_4d_metric_root_m"][index],
                    dtype=np.float32,
                ),
            )
        )
        if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
            raise CanonicalBCContractError("canonical actor proprioception invalid")
        action = validate_serialized_policy_action_4d_metric_root_m(
            rows["policy_action_4d_metric_root_m"][index]
        )
        return {
            "rgb": torch.from_numpy(rgb.copy()),
            "depth": torch.from_numpy(depth.copy()).float(),
            "depth_valid": torch.from_numpy(valid.copy()).float(),
            "proprio": torch.from_numpy(proprio.copy()),
            "action": torch.from_numpy(action.copy()),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _model_input(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    rgb = batch["rgb"].to(device, non_blocking=True).float().div_(255.0)
    depth = batch["depth"].to(device, non_blocking=True).clamp_(0.0, 2.0).div_(2.0)
    valid = batch["depth_valid"].to(device, non_blocking=True).clamp_(0.0, 1.0)
    return F.interpolate(torch.cat((rgb, depth, valid), dim=1), size=IMAGE_HW, mode="bilinear", align_corners=False)


class CanonicalContactFreeVisualBC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(96 + PROPRIO_DIM), nn.Linear(96 + PROPRIO_DIM, 128),
            nn.ReLU(), nn.Linear(128, 3),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        xyz = torch.tanh(self.head(torch.cat((self.vision(image), proprio), dim=-1)))
        # Component-wise tanh alone permits a diagonal vector with norm
        # ``sqrt(3) * bound``.  Project every output onto the same Euclidean
        # 4.5-mm ball used by the live 4-D contract, then retain one float32
        # ulp of representation margin below that boundary.
        norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
        xyz = xyz * torch.clamp(
            MODEL_OUTPUT_MAX_DELTA_M / norm.clamp_min(1.0e-12), max=1.0
        )
        return torch.cat((xyz, torch.zeros_like(xyz[:, :1])), dim=-1)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction[:, :3] - target[:, :3]
    nonzero = target[:, :3].norm(dim=1) > 1.0e-7
    direction_error = math.nan
    if bool(nonzero.any()):
        cosine = F.cosine_similarity(prediction[nonzero, :3], target[nonzero, :3], dim=1)
        direction_error = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).mean().item()
    return {
        "xyz_rmse_mm": torch.sqrt(error.square().mean()).item() * 1000.0,
        "action_direction_error_deg": direction_error,
        "action_norm_error_mm": (
            prediction[:, :3].norm(dim=1) - target[:, :3].norm(dim=1)
        ).abs().mean().item() * 1000.0,
        "4.5mm_bound_violation_count": int(
            (prediction[:, :3].norm(dim=1) > CONTACT_FREE_MAX_DELTA_M + 1.0e-9)
            .sum()
            .item()
        ),
    }


def _run_epoch(*, model: CanonicalContactFreeVisualBC, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_rows = total_loss = 0.0
    aggregate = {"xyz_rmse_mm": 0.0, "action_direction_error_deg": 0.0, "action_norm_error_mm": 0.0, "4.5mm_bound_violation_count": 0.0}
    direction_rows = 0
    for batch in loader:
        image = _model_input(batch, device)
        proprio = batch["proprio"].to(device, non_blocking=True).float()
        target = batch["action"].to(device, non_blocking=True).float()
        with torch.set_grad_enabled(training):
            prediction = model(image, proprio)
            loss = F.mse_loss(prediction[:, :3], target[:, :3])
            if not torch.isfinite(loss):
                raise CanonicalBCContractError("BC loss is NaN/Inf")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        count = float(target.shape[0])
        total_rows += count
        total_loss += float(loss.detach().item()) * count
        metric = _metrics(prediction.detach(), target)
        for name, value in metric.items():
            if name == "action_direction_error_deg" and not math.isfinite(value):
                continue
            aggregate[name] += value if name == "4.5mm_bound_violation_count" else value * count
        direction_rows += int((target[:, :3].norm(dim=1) > 1.0e-7).sum().item())
    if total_rows <= 0:
        raise CanonicalBCContractError("empty training loader")
    return {
        "bc_loss": total_loss / total_rows,
        "xyz_rmse_mm": aggregate["xyz_rmse_mm"] / total_rows,
        "action_direction_error_deg": (
            aggregate["action_direction_error_deg"] / total_rows
            if direction_rows else math.nan
        ),
        "action_norm_error_mm": aggregate["action_norm_error_mm"] / total_rows,
        "4.5mm_bound_violation_count": aggregate["4.5mm_bound_violation_count"],
    }


def first_launch_manifest(*, audit: CanonicalArtifactAudit, epochs: int, batch_size: int) -> dict[str, object]:
    if epochs != FIRST_BC_EPOCHS or batch_size != FIRST_BC_BATCH_SIZE:
        raise CanonicalBCContractError("FIRST_BC_PLAN_REQUIRES_EXACTLY_5_EPOCHS_AND_BATCH_256")
    return {
        "schema": f"{SCHEMA}_first_launch_manifest_v1",
        "scope": "OFFLINE_CANONICAL_CANDIDATE_A_CONTACT_FREE_BC_ONLY",
        "epochs": FIRST_BC_EPOCHS,
        "batch_size": FIRST_BC_BATCH_SIZE,
        "seed": FIRST_BC_SEED,
        "dataset_audit": audit.payload(),
        "actor_input": {
            "vision": "right_wrist_rgb,right_wrist_depth_m,right_wrist_depth_valid",
            "proprio": (
                "EE_XYZW,right_arm_q_qd_qdd,gripper_open,"
                "previous_policy_action_4d_metric_root_m"
            ),
            "frame": "robot_root",
            "privileged_cube_contact_phase_input": False,
        },
        "label": "[dx_m,dy_m,dz_m,HOLD_OPEN=0],robot_root",
        "loss": "MSE_XYZ_ONLY",
        "model_output_max_delta_m": MODEL_OUTPUT_MAX_DELTA_M,
        "split": "chronological_tail_90_10_single_rollout_temporal_heldout",
        "sac_first": False,
        "her": "DISABLED",
        "her_force": "DISABLED",
        "contact": "DISABLED",
    }


def _checkpoint_payload(
    *,
    model: CanonicalContactFreeVisualBC,
    optimizer: torch.optim.Optimizer,
    audit: CanonicalArtifactAudit,
    epoch: int,
    seed: int,
    history: list[dict[str, object]],
    launch_seal_sha256: str,
    cuda_receipt: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed,
        "history": history,
        "proprio_dim": PROPRIO_DIM,
        "image_hw": IMAGE_HW,
        "action_dim": 4,
        "action_contract": "[dx,dy,dz,HOLD_OPEN=0],robot_root,m",
        "model_output_max_delta_m": MODEL_OUTPUT_MAX_DELTA_M,
        "audit": audit.payload(),
        "source_hashes": audit.source_hashes,
        "launch_seal_sha256": launch_seal_sha256,
        "cuda_receipt": cuda_receipt,
        "contact_training_authorized": False,
        "sac_started": False,
        "her": "DISABLED",
        "her_force": "DISABLED",
    }


def _same_checkpoint_value(observed: object, expected: object) -> bool:
    """Compare checkpoint metadata while treating an optional NaN metric as equal.

    Direction error is intentionally NaN when an all-zero heldout has no
    direction.  That metric remains non-promotable as a number, but it must not
    make an otherwise byte-identical checkpoint impossible to reload.
    """

    if isinstance(observed, float) and isinstance(expected, float):
        return observed == expected or (math.isnan(observed) and math.isnan(expected))
    if isinstance(observed, dict) and isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _same_checkpoint_value(observed[key], expected[key]) for key in observed
        )
    if isinstance(observed, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(observed) == len(expected) and all(
            _same_checkpoint_value(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _verify_checkpoint_reload(
    *,
    path: Path,
    expected_sha256: str,
    audit: CanonicalArtifactAudit,
    expected_epoch: int,
    expected_seed: int,
    expected_history: list[dict[str, object]],
    launch_seal_sha256: str,
    cuda_receipt: dict[str, object],
) -> dict[str, object]:
    """Strict CPU reload check before a checkpoint can be promoted.

    This does not resume training.  It validates that a future explicit
    initializer can reconstruct exactly the current visual-BC architecture and
    optimizer state without accepting a differently shaped action/data schema.
    """

    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise CanonicalBCContractError("checkpoint SHA-256 changed after atomic write")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise CanonicalBCContractError("checkpoint cannot be reloaded") from error
    payload = _mapping(payload, "checkpoint")
    required = {
        "schema": SCHEMA,
        "epoch": expected_epoch,
        "seed": expected_seed,
        "history": expected_history,
        "proprio_dim": PROPRIO_DIM,
        "image_hw": IMAGE_HW,
        "action_dim": 4,
        "action_contract": "[dx,dy,dz,HOLD_OPEN=0],robot_root,m",
        "model_output_max_delta_m": MODEL_OUTPUT_MAX_DELTA_M,
        "audit": audit.payload(),
        "source_hashes": audit.source_hashes,
        "launch_seal_sha256": launch_seal_sha256,
        "cuda_receipt": cuda_receipt,
        "contact_training_authorized": False,
        "sac_started": False,
        "her": "DISABLED",
        "her_force": "DISABLED",
    }
    if any(not _same_checkpoint_value(payload.get(key), value) for key, value in required.items()):
        raise CanonicalBCContractError("checkpoint compatibility metadata mismatch")
    state = payload.get("state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict) or not isinstance(optimizer_state, dict):
        raise CanonicalBCContractError("checkpoint model/optimizer state is missing")
    model = CanonicalContactFreeVisualBC()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    try:
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    except (RuntimeError, ValueError, KeyError) as error:
        raise CanonicalBCContractError("checkpoint strict architecture/optimizer reload failed") from error
    return {
        "schema": CHECKPOINT_ATTESTATION_SCHEMA,
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "epoch": expected_epoch,
        "strict_model_reload": True,
        "optimizer_reload": True,
        "resume": "NOT_IMPLEMENTED_FRESH_OUTPUT_ONLY",
    }


def train(
    *,
    audit: CanonicalArtifactAudit,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    launch_seal: Path,
) -> dict[str, object]:
    if device_name != FIRST_BC_DEVICE:
        raise CanonicalBCContractError(f"first canonical BC run requires device={FIRST_BC_DEVICE}")
    seal = verify_launch_seal(audit=audit, launch_seal=launch_seal, learning_rate=learning_rate)
    launch_seal_sha256 = _sha256_file(launch_seal)
    cuda_receipt = cuda_zero_preflight_receipt()
    if output.exists():
        raise CanonicalBCContractError("refusing to overwrite BC run output")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    train_set = CanonicalHDF5Dataset(audit.dataset, audit.train_rows)
    validation_set = CanonicalHDF5Dataset(audit.dataset, audit.validation_rows)
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "TRAINING_STATUS.json"
    base = {
        "schema": f"{SCHEMA}_status_v1",
        "scope": "OFFLINE_CANONICAL_CANDIDATE_A_CONTACT_FREE_BC_ONLY",
        "audit": audit.payload(),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "cuda_preflight": cuda_receipt,
        "launch_seal": {"path": str(launch_seal.resolve()), "sha256": launch_seal_sha256},
        "launch_seal_schema": seal["schema"],
        "action_contract": "[dx,dy,dz,HOLD_OPEN],robot_root,m; 4.5mm strict live bound",
        "model_output_max_delta_m": MODEL_OUTPUT_MAX_DELTA_M,
        "contact_training_authorized": False,
        "sac_started": False,
        "resume": "NOT_IMPLEMENTED_FRESH_OUTPUT_ONLY",
    }
    _atomic_json(status_path, {**base, "status": "RUNNING", "last_completed_epoch": 0})
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = CanonicalContactFreeVisualBC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = math.inf
    history: list[dict[str, object]] = []
    best_path = output / "best_validation.pt"
    last_path = output / "last.pt"
    best_attestation: dict[str, object] | None = None
    last_attestation: dict[str, object] | None = None
    try:
        for epoch in range(1, epochs + 1):
            train_metric = _run_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
            validation_metric = _run_epoch(model=model, loader=validation_loader, device=device, optimizer=None)
            if not math.isfinite(train_metric["bc_loss"]) or not math.isfinite(validation_metric["bc_loss"]):
                raise CanonicalBCContractError("non-finite train/validation loss")
            record = {"epoch": epoch, "train": train_metric, "validation": validation_metric}
            history.append(record)
            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                audit=audit,
                epoch=epoch,
                seed=seed,
                history=history,
                launch_seal_sha256=launch_seal_sha256,
                cuda_receipt=cuda_receipt,
            )
            last_sha256 = _atomic_torch(last_path, checkpoint)
            last_attestation = _verify_checkpoint_reload(
                path=last_path,
                expected_sha256=last_sha256,
                audit=audit,
                expected_epoch=epoch,
                expected_seed=seed,
                expected_history=history,
                launch_seal_sha256=launch_seal_sha256,
                cuda_receipt=cuda_receipt,
            )
            if validation_metric["bc_loss"] < best:
                best = float(validation_metric["bc_loss"])
                best_sha256 = _atomic_torch(best_path, checkpoint)
                best_attestation = _verify_checkpoint_reload(
                    path=best_path,
                    expected_sha256=best_sha256,
                    audit=audit,
                    expected_epoch=epoch,
                    expected_seed=seed,
                    expected_history=history,
                    launch_seal_sha256=launch_seal_sha256,
                    cuda_receipt=cuda_receipt,
                )
            _atomic_json(status_path, {
                **base,
                "status": "RUNNING",
                "last_completed_epoch": epoch,
                "best_validation_loss": best,
                "best_checkpoint_exists": best_path.is_file(),
                "last_checkpoint_exists": last_path.is_file(),
                "last_checkpoint_reload_attestation": last_attestation,
                "best_checkpoint_reload_attestation": best_attestation,
            })
    except BaseException as error:
        _atomic_json(status_path, {**base, "status": "FAILED_OR_INTERRUPTED", "last_completed_epoch": len(history), "exception_type": type(error).__name__, "exception": str(error), "best_checkpoint_exists": best_path.is_file(), "last_checkpoint_exists": last_path.is_file()})
        raise
    finally:
        train_set.close()
        validation_set.close()
    report = {
        "schema": SCHEMA,
        "scope": "OFFLINE_CANONICAL_CANDIDATE_A_CONTACT_FREE_BC_ONLY",
        "audit": audit.payload(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "history": history,
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        # A SHA-verified strict reload proves checkpoint integrity only.  This
        # dataset is one temporal split from one rollout, so it cannot by
        # itself promote a BC actor to deployment or SAC initialization.
        "checkpoint_integrity_attested": bool(last_attestation and best_attestation),
        "checkpoint_promotion_authorized": False,
        "checkpoint_promotion_blocker": (
            "INDEPENDENT_GENERALIZATION_AND_FRESH_RUNTIME_SMOKE_REQUIRED"
        ),
        "checkpoint_reload_attestations": {"last": last_attestation, "best": best_attestation},
        "contact_training_authorized": False,
        "sac_started": False,
        "her": "DISABLED",
        "her_force": "DISABLED",
        "validation_interpretation": "SINGLE_ROLLOUT_TEMPORAL_HELDOUT_NOT_INDEPENDENT_GENERALIZATION",
    }
    _atomic_json(output / "TRAINING_REPORT.json", report)
    _atomic_json(status_path, {
        **base,
        "status": "COMPLETED",
        "last_completed_epoch": epochs,
        "best_validation_loss": best,
        "best_checkpoint_exists": best_path.is_file(),
        "last_checkpoint_exists": last_path.is_file(),
        "last_checkpoint_reload_attestation": last_attestation,
        "best_checkpoint_reload_attestation": best_attestation,
        "training_report_exists": True,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--functional-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=FIRST_BC_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FIRST_BC_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=FIRST_BC_SEED)
    parser.add_argument("--device", choices=(FIRST_BC_DEVICE,), default=FIRST_BC_DEVICE)
    parser.add_argument(
        "--launch-seal",
        type=Path,
        help="immutable preflight seal; required for the actual five-epoch BC launch",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.epochs != FIRST_BC_EPOCHS or args.batch_size != FIRST_BC_BATCH_SIZE or args.seed != FIRST_BC_SEED:
        raise CanonicalBCContractError("first canonical BC run requires epochs=5, batch=256, seed=42")
    if args.learning_rate <= 0.0:
        raise CanonicalBCContractError("learning rate must be positive")
    audit = audit_canonical_artifact(dataset=args.dataset, functional_report=args.functional_report)
    if args.preflight_only:
        if args.launch_seal is not None:
            raise CanonicalBCContractError("preflight creates a seal; it does not consume one")
        if args.output.exists():
            raise CanonicalBCContractError("refusing to overwrite BC preflight output")
        manifest = build_launch_seal(audit=audit, learning_rate=args.learning_rate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.launch_seal is None:
        raise CanonicalBCContractError("actual BC launch requires --launch-seal from a matching preflight")
    report = train(
        audit=audit,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        launch_seal=args.launch_seal,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
