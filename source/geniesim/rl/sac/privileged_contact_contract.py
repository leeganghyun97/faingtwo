# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Simulation-only grasp outcome semantics for the G2 OmniPicker pipeline.

Nothing in this module is a deployable observation.  The physical OmniPicker
has no fingertip touch sensor, so these values may be recorded in simulation
and consumed by offline labels, rewards, replay sampling or an explicitly
asymmetric critic, but never by the student actor.

The contract deliberately separates raw measurements from derived labels:

``CONTACT = left_contact OR right_contact``
``BILATERAL_CONTACT = left_contact AND right_contact``
``STABLE = bilateral AND low relative speed AND dwell``
``GRASP_SUCCESS = STABLE AND retained AND (lifted when required)``

Boolean contact is the primary privileged learning authority.  Force remains
raw diagnostic/safety evidence and produces the independent ``FORCE_OK``
label, but it cannot create CONTACT, BILATERAL_CONTACT or STABLE.  Force
thresholds are simulation-analysis thresholds.  They are not the OEM 30 N
whole-gripper product limit and cannot be promoted to production without an
immutable distribution receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SIM_PRIVILEGED_CONTACT_SCHEMA = "g2_sim_privileged_contact_raw_v1"
SIM_PRIVILEGED_LABEL_SCHEMA = "g2_sim_privileged_grasp_labels_v1"
SIM_THRESHOLD_RECEIPT_SCHEMA = "g2_sim_contact_threshold_receipt_v1"
SIM_FEASIBILITY_SCHEMA = "g2_sim_privileged_feasibility_geometry_v1"
CLOSE_WINDOW_SCHEMA = "g2_keyboard_close_window_ms_v1"
CONTROL_HZ = 50
CONTROL_DT_S = 1.0 / CONTROL_HZ

DEPLOYABLE_ACTOR_FIELDS = (
    "right_wrist_rgb",
    "right_wrist_depth_m",
    "right_wrist_depth_valid",
    "robot_state",
    "previous_action_4d",
    "gripper_state",
)
PROHIBITED_DEPLOY_TOKENS = (
    "touch",
    "contact",
    "force",
    "cube_pose",
    "cube_gt",
    "pad_pose",
    "pad_distance",
    "stable",
    "grasp_success",
)


class PrivilegedContactContractError(ValueError):
    """Raised before invalid simulation telemetry can enter a learner."""


class ContactReplayClass(IntEnum):
    NO_CONTACT = 0
    FIRST_CONTACT = 1
    BILATERAL = 2
    STABLE = 3
    LIFT = 4


def _array(name: str, value: Any, shape: tuple[int | None, ...], dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != len(shape) or any(
        expected is not None and result.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise PrivilegedContactContractError(
            f"{name} shape {result.shape} does not match {shape}"
        )
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise PrivilegedContactContractError(f"{name} contains NaN/Inf")
    return result


def _bool_vector(name: str, value: Any, rows: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or (rows is not None and raw.shape != (rows,)):
        raise PrivilegedContactContractError(f"{name} must be bool [N]")
    if raw.dtype != np.bool_ and not np.all(np.isin(raw, (0, 1))):
        raise PrivilegedContactContractError(f"{name} must be binary")
    return raw.astype(np.bool_, copy=False)


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SimulationContactRaw:
    """One episode of measured simulation telemetry, all arrays row-aligned."""

    timestamp_s: np.ndarray
    control_step: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    left_normal_force_n: np.ndarray
    right_normal_force_n: np.ndarray
    left_pad_pose_root_m_xyzw: np.ndarray
    right_pad_pose_root_m_xyzw: np.ndarray
    cube_pose_root_m_xyzw: np.ndarray
    relative_pad_cube_velocity_root_m_s: np.ndarray
    measurement_valid: np.ndarray
    lifted: np.ndarray

    def __post_init__(self) -> None:
        timestamp = _array("timestamp_s", self.timestamp_s, (None,), np.float64)
        rows = int(timestamp.shape[0])
        if rows <= 0 or np.any(np.diff(timestamp) <= 0.0):
            raise PrivilegedContactContractError(
                "timestamp_s must be nonempty and strictly monotonic"
            )
        step = _array("control_step", self.control_step, (rows,), np.int64)
        if rows > 1 and not np.array_equal(
            np.diff(step), np.ones(rows - 1, dtype=np.int64)
        ):
            raise PrivilegedContactContractError("control_step must be contiguous")
        if rows > 1 and not np.allclose(
            np.diff(timestamp), CONTROL_DT_S, rtol=0.0, atol=1.0e-6
        ):
            raise PrivilegedContactContractError("contact raw rows must be 50 Hz")
        booleans = {
            "left_contact": _bool_vector("left_contact", self.left_contact, rows),
            "right_contact": _bool_vector("right_contact", self.right_contact, rows),
            "measurement_valid": _bool_vector(
                "measurement_valid", self.measurement_valid, rows
            ),
            "lifted": _bool_vector("lifted", self.lifted, rows),
        }
        forces = {
            "left_normal_force_n": _array(
                "left_normal_force_n", self.left_normal_force_n, (rows,), np.float64
            ),
            "right_normal_force_n": _array(
                "right_normal_force_n", self.right_normal_force_n, (rows,), np.float64
            ),
        }
        if any(np.any(value < 0.0) for value in forces.values()):
            raise PrivilegedContactContractError("normal force cannot be negative")
        for name in (
            "left_pad_pose_root_m_xyzw",
            "right_pad_pose_root_m_xyzw",
            "cube_pose_root_m_xyzw",
        ):
            pose = _array(name, getattr(self, name), (rows, 7), np.float64)
            if np.any(np.abs(np.linalg.norm(pose[:, 3:7], axis=1) - 1.0) > 1.0e-3):
                raise PrivilegedContactContractError(f"{name} quaternion is not unit")
        _array(
            "relative_pad_cube_velocity_root_m_s",
            self.relative_pad_cube_velocity_root_m_s,
            (rows, 3),
            np.float64,
        )
        for name, value in {**booleans, **forces}.items():
            frozen = np.ascontiguousarray(value.copy())
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)
        for name in (
            "timestamp_s",
            "control_step",
            "left_pad_pose_root_m_xyzw",
            "right_pad_pose_root_m_xyzw",
            "cube_pose_root_m_xyzw",
            "relative_pad_cube_velocity_root_m_s",
        ):
            frozen = np.ascontiguousarray(np.asarray(getattr(self, name)).copy())
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)

    @property
    def rows(self) -> int:
        return int(self.timestamp_s.shape[0])

    @property
    def total_normal_force_n(self) -> np.ndarray:
        return self.left_normal_force_n + self.right_normal_force_n

    @property
    def first_contact_step(self) -> int | None:
        rows = np.flatnonzero(
            self.measurement_valid & (self.left_contact | self.right_contact)
        )
        return int(self.control_step[int(rows[0])]) if rows.size else None

    @property
    def first_contact_timestamp_s(self) -> float | None:
        rows = np.flatnonzero(
            self.measurement_valid & (self.left_contact | self.right_contact)
        )
        return None if not rows.size else float(self.timestamp_s[int(rows[0])])


@dataclass(frozen=True)
class SimulationContactThresholds:
    """Provisional thresholds backed by a successful-data receipt."""

    minimum_total_force_n: float
    maximum_total_force_n: float
    maximum_relative_speed_m_s: float
    stable_dwell_s: float
    distribution_receipt_sha256: str
    production_threshold_locked: bool = False

    def __post_init__(self) -> None:
        values = (
            self.minimum_total_force_n,
            self.maximum_total_force_n,
            self.maximum_relative_speed_m_s,
            self.stable_dwell_s,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise PrivilegedContactContractError("thresholds must be finite/nonnegative")
        if self.maximum_total_force_n < self.minimum_total_force_n:
            raise PrivilegedContactContractError("force band is inverted")
        if self.stable_dwell_s <= 0.0:
            raise PrivilegedContactContractError("stable dwell must be positive")
        receipt = self.distribution_receipt_sha256
        if len(receipt) != 64 or any(value not in "0123456789abcdef" for value in receipt):
            raise PrivilegedContactContractError("threshold receipt must be SHA-256")
        if self.production_threshold_locked:
            raise PrivilegedContactContractError(
                "simulation distributions cannot lock a production threshold"
            )

    @property
    def stable_dwell_rows(self) -> int:
        return max(1, int(math.ceil(self.stable_dwell_s * CONTROL_HZ - 1.0e-12)))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SIM_THRESHOLD_RECEIPT_SCHEMA,
            "minimum_total_force_n": self.minimum_total_force_n,
            "maximum_total_force_n": self.maximum_total_force_n,
            "maximum_relative_speed_m_s": self.maximum_relative_speed_m_s,
            "stable_dwell_s": self.stable_dwell_s,
            "stable_dwell_rows_at_50hz": self.stable_dwell_rows,
            "distribution_receipt_sha256": self.distribution_receipt_sha256,
            "production_threshold_locked": False,
            "force_semantics": "SIM_CLASSIFICATION_BAND_NOT_OEM_30N_GATE",
            "boolean_contact_primary_authority": True,
            "force_required_for_contact": False,
            "force_required_for_stable": False,
        }


@dataclass(frozen=True)
class PrivilegedFeasibilityThresholds:
    """Provisional geometry teacher thresholds, never deployment inputs."""

    maximum_pad_to_cube_distance_m: float
    maximum_lateral_alignment_error_m: float
    maximum_orientation_error_rad: float
    maximum_relative_speed_m_s: float
    distribution_receipt_sha256: str
    production_threshold_locked: bool = False

    def __post_init__(self) -> None:
        values = (
            self.maximum_pad_to_cube_distance_m,
            self.maximum_lateral_alignment_error_m,
            self.maximum_orientation_error_rad,
            self.maximum_relative_speed_m_s,
        )
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in values
        ):
            raise PrivilegedContactContractError(
                "feasibility thresholds must be finite/nonnegative"
            )
        receipt = self.distribution_receipt_sha256
        if len(receipt) != 64 or any(
            value not in "0123456789abcdef" for value in receipt
        ):
            raise PrivilegedContactContractError(
                "feasibility threshold receipt must be SHA-256"
            )
        if self.production_threshold_locked:
            raise PrivilegedContactContractError(
                "simulation feasibility thresholds cannot lock production"
            )


def derive_privileged_feasibility(
    *,
    pad_to_cube_distance_m: Sequence[float] | np.ndarray,
    lateral_alignment_error_m: Sequence[float] | np.ndarray,
    orientation_error_rad: Sequence[float] | np.ndarray,
    relative_speed_m_s: Sequence[float] | np.ndarray,
    valid: Sequence[bool] | np.ndarray,
    thresholds: PrivilegedFeasibilityThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a geometry-only auxiliary target and its explicit valid mask."""

    distance = _array(
        "pad_to_cube_distance_m", pad_to_cube_distance_m, (None,), np.float64
    )
    rows = int(distance.shape[0])
    lateral = _array(
        "lateral_alignment_error_m",
        lateral_alignment_error_m,
        (rows,),
        np.float64,
    )
    orientation = _array(
        "orientation_error_rad", orientation_error_rad, (rows,), np.float64
    )
    speed = _array("relative_speed_m_s", relative_speed_m_s, (rows,), np.float64)
    valid_mask = _bool_vector("feasibility_valid", valid, rows)
    if any(np.any(value < 0.0) for value in (distance, lateral, orientation, speed)):
        raise PrivilegedContactContractError(
            "feasibility geometry values must be nonnegative"
        )
    feasible = (
        valid_mask
        & (distance <= thresholds.maximum_pad_to_cube_distance_m)
        & (lateral <= thresholds.maximum_lateral_alignment_error_m)
        & (orientation <= thresholds.maximum_orientation_error_rad)
        & (speed <= thresholds.maximum_relative_speed_m_s)
    )
    return feasible, valid_mask


@dataclass(frozen=True)
class PrivilegedGraspLabels:
    contact: np.ndarray
    bilateral_contact: np.ndarray
    force_ok: np.ndarray
    stable: np.ndarray
    bilateral_dwell_s: np.ndarray
    cube_retained: np.ndarray
    grasp_success: np.ndarray
    replay_class: np.ndarray
    valid: np.ndarray
    lift_required: bool

    def taxonomy_counts(self) -> dict[str, int]:
        return {
            item.name: int(np.count_nonzero(self.replay_class == int(item)))
            for item in ContactReplayClass
        }


def derive_privileged_grasp_labels(
    raw: SimulationContactRaw,
    *,
    thresholds: SimulationContactThresholds,
    cube_retained: Sequence[bool] | np.ndarray,
    cube_retained_valid: Sequence[bool] | np.ndarray,
    require_lift: bool = False,
) -> PrivilegedGraspLabels:
    """Derive the single canonical CONTACT/STABLE/SUCCESS definition."""

    retained = _bool_vector("cube_retained", cube_retained, raw.rows)
    retained_valid = _bool_vector(
        "cube_retained_valid", cube_retained_valid, raw.rows
    )
    valid = raw.measurement_valid & retained_valid
    # Invalid sensor rows can never create a learning milestone.  Raw booleans
    # remain stored in ``SimulationContactRaw`` for forensic inspection.
    contact = valid & (raw.left_contact | raw.right_contact)
    bilateral = valid & raw.left_contact & raw.right_contact
    total = raw.total_normal_force_n
    force_ok = valid & (total >= thresholds.minimum_total_force_n) & (
        total <= thresholds.maximum_total_force_n
    )
    relative_speed = np.linalg.norm(
        raw.relative_pad_cube_velocity_root_m_s, axis=1
    )
    # Boolean bilateral contact owns STABLE.  Force is intentionally not in
    # this conjunction: FORCE_OK remains an orthogonal diagnostic/safety tag.
    stable_candidate = (
        valid
        & bilateral
        & (relative_speed <= thresholds.maximum_relative_speed_m_s)
    )
    stable = np.zeros(raw.rows, dtype=np.bool_)
    bilateral_dwell_s = np.zeros(raw.rows, dtype=np.float64)
    consecutive = 0
    for index, candidate in enumerate(stable_candidate):
        consecutive = consecutive + 1 if bool(candidate) else 0
        bilateral_dwell_s[index] = consecutive * CONTROL_DT_S
        stable[index] = consecutive >= thresholds.stable_dwell_rows
    success = stable & retained
    if require_lift:
        success &= raw.lifted
    taxonomy = np.full(raw.rows, int(ContactReplayClass.NO_CONTACT), dtype=np.uint8)
    taxonomy[contact] = int(ContactReplayClass.FIRST_CONTACT)
    taxonomy[bilateral] = int(ContactReplayClass.BILATERAL)
    taxonomy[stable] = int(ContactReplayClass.STABLE)
    taxonomy[stable & raw.lifted] = int(ContactReplayClass.LIFT)
    return PrivilegedGraspLabels(
        contact=contact,
        bilateral_contact=bilateral,
        force_ok=force_ok,
        stable=stable,
        bilateral_dwell_s=bilateral_dwell_s,
        cube_retained=retained,
        grasp_success=success,
        replay_class=taxonomy,
        valid=valid,
        lift_required=bool(require_lift),
    )


def close_window_mask_ms(
    *,
    timestamp_s: Sequence[float] | np.ndarray,
    close_edge: Sequence[bool] | np.ndarray,
    before_ms: float = 200.0,
    after_ms: float = 200.0,
) -> np.ndarray:
    """Return the physical-time CLOSE window; steps are never the authority."""

    timestamp = _array("timestamp_s", timestamp_s, (None,), np.float64)
    edge = _bool_vector("close_edge", close_edge, int(timestamp.shape[0]))
    indices = np.flatnonzero(edge)
    if indices.size != 1:
        raise PrivilegedContactContractError("CLOSE window requires exactly one edge")
    if before_ms < 0.0 or after_ms < 0.0:
        raise PrivilegedContactContractError("CLOSE window widths must be nonnegative")
    center = float(timestamp[int(indices[0])])
    return (timestamp >= center - before_ms / 1000.0 - 1.0e-12) & (
        timestamp <= center + after_ms / 1000.0 + 1.0e-12
    )


def human_privileged_confusion(
    human_close: Sequence[bool] | np.ndarray,
    privileged_feasible: Sequence[bool] | np.ndarray,
    valid: Sequence[bool] | np.ndarray,
) -> dict[str, int]:
    human = _bool_vector("human_close", human_close)
    privileged = _bool_vector("privileged_feasible", privileged_feasible, human.size)
    mask = _bool_vector("valid", valid, human.size)
    return {
        "HUMAN_PRIVILEGED_00_COUNT": int(np.count_nonzero(mask & ~human & ~privileged)),
        "HUMAN_PRIVILEGED_01_COUNT": int(np.count_nonzero(mask & ~human & privileged)),
        "HUMAN_PRIVILEGED_10_COUNT": int(np.count_nonzero(mask & human & ~privileged)),
        "HUMAN_PRIVILEGED_11_COUNT": int(np.count_nonzero(mask & human & privileged)),
    }


def distribution_summary(values: Mapping[str, Sequence[float] | np.ndarray]) -> dict[str, Any]:
    """Create a receipt input; it proposes no production threshold."""

    percentiles = (5, 25, 50, 75, 95)
    fields: dict[str, Any] = {}
    for name, raw in values.items():
        array = np.asarray(raw, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size == 0:
            raise PrivilegedContactContractError(f"no finite samples for {name}")
        fields[name] = {
            f"p{percentile:02d}": float(np.percentile(array, percentile))
            for percentile in percentiles
        }
        fields[name]["count"] = int(array.size)
    payload = {
        "schema": SIM_THRESHOLD_RECEIPT_SCHEMA,
        "statistics": fields,
        "production_threshold_locked": False,
    }
    payload["receipt_sha256"] = _sha256_payload(payload)
    return payload


def deployment_guard(actor_input_fields: Sequence[str]) -> dict[str, Any]:
    lowered = tuple(str(value).lower() for value in actor_input_fields)
    leaks = sorted(
        field
        for field in lowered
        if any(token in field for token in PROHIBITED_DEPLOY_TOKENS)
    )
    return {
        "student_privileged_input_count": len(leaks),
        "deploy_sim_only_signal_count": len(leaks),
        "leaking_fields": leaks,
        "passed": not leaks,
    }


__all__ = [
    "CLOSE_WINDOW_SCHEMA",
    "CONTROL_DT_S",
    "CONTROL_HZ",
    "ContactReplayClass",
    "DEPLOYABLE_ACTOR_FIELDS",
    "PROHIBITED_DEPLOY_TOKENS",
    "PrivilegedContactContractError",
    "PrivilegedGraspLabels",
    "PrivilegedFeasibilityThresholds",
    "SIM_PRIVILEGED_CONTACT_SCHEMA",
    "SIM_PRIVILEGED_LABEL_SCHEMA",
    "SIM_THRESHOLD_RECEIPT_SCHEMA",
    "SIM_FEASIBILITY_SCHEMA",
    "SimulationContactRaw",
    "SimulationContactThresholds",
    "close_window_mask_ms",
    "deployment_guard",
    "derive_privileged_grasp_labels",
    "derive_privileged_feasibility",
    "distribution_summary",
    "human_privileged_confusion",
]
