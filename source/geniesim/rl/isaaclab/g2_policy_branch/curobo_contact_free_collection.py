# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Static collection contract for cuRobo OPEN-only contact-free rows.

This module deliberately does *not* import Isaac, cuRobo, h5py, or a
controller.  It is the small adapter a future, separately authorised live
runner must call *after* its canonical ``env.step`` receipt is known.  Keeping
it pure makes all unit/frame/action checks testable without starting PhysX.

The controlled-rebuild HDF5 schema predates the four-dimensional policy
branch.  A row therefore carries both:

* the historical controller-compatible 7-D/8-D normalised records; and
* branch-native 4-D normalised and metre-in-``robot_root`` records.

The latter is the only action label that contact-free BC/SAC may consume.
The legacy records are compatibility/audit material and cannot be silently
projected into a learner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_interface import (
    AbstractGripperIntent,
    CartesianControlFrame,
    CartesianResidualScale,
)
from .contact_free_training_contract import (
    CONTACT_FREE_ACTION_DIM,
    CONTACT_FREE_MAX_DELTA_M,
    validate_contact_free_action,
)
from .curobo_4d_handoff import CuroboCanonical4DHandoffReceipt
from .precontact_contract import CONTACT_FREE_EXECUTION_MODE, PrecontactPhase
from geniesim.rl.isaaclab.g2_rebuild.data_contract import G2_RUNTIME_JOINT_ORDER


CUROBO_CONTACT_FREE_COLLECTION_SCHEMA = "g2_curobo_contact_free_collection_v1"
POLICY_ACTION_4D_NORMALIZED_FIELD = "policy_action_4d_normalized"
POLICY_ACTION_4D_METRIC_FIELD = "policy_action_4d_metric_robot_root_m"
PREVIOUS_POLICY_ACTION_4D_METRIC_FIELD = "previous_policy_action_4d_metric_root_m"
COLLECTION_ALLOWED_PHASES = frozenset(
    {
        PrecontactPhase.REACH.value,
        PrecontactPhase.PREGRASP.value,
        PrecontactPhase.FINE_APPROACH.value,
        PrecontactPhase.OPEN_HANDOFF.value,
    }
)
DEPLOYABLE_ACTOR_FIELD_NAMES = frozenset(
    {
        "right_wrist_rgb",
        "right_wrist_depth_m",
        "right_wrist_depth_valid",
        "ee_pose_robot_root_xyzw",
        "robot_joint_position_rad",
        "robot_joint_velocity_rad_s",
        "gripper_state",
        PREVIOUS_POLICY_ACTION_4D_METRIC_FIELD,
    }
)
FORBIDDEN_ACTOR_FIELD_NAMES = frozenset(
    {
        "cube_gt_pose_robot_root_xyzw",
        "head_cube_gt_pose_at_capture_robot_root_xyzw",
        "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw",
        "phase_label",
        "planner_target_root_m",
        "planned_segment_minimum_pad_object_distance_m",
        "observed_pad_object_distance_m",
        "contact_guard_distance_m",
        "contact_count",
        "gripper_close_count",
        "forbidden_collision_count",
        "hard_limit_event_count",
        "safety_reject_count",
    }
)


class CuroboContactFreeCollectionError(ValueError):
    """Raised before an unsafe/ambiguous row can enter a collection payload."""


def assert_deployable_actor_input_inventory(names: Sequence[str]) -> tuple[str, ...]:
    """Fail closed if a trainer tries to bind labels/GT into its actor input."""
    inventory = tuple(str(name) for name in names)
    forbidden = sorted(set(inventory).intersection(FORBIDDEN_ACTOR_FIELD_NAMES))
    unknown = sorted(set(inventory).difference(DEPLOYABLE_ACTOR_FIELD_NAMES))
    if forbidden or unknown:
        raise CuroboContactFreeCollectionError(
            "actor input inventory is not deployable: "
            f"forbidden={forbidden},unknown={unknown}"
        )
    if set(inventory) != set(DEPLOYABLE_ACTOR_FIELD_NAMES):
        missing = sorted(DEPLOYABLE_ACTOR_FIELD_NAMES.difference(inventory))
        raise CuroboContactFreeCollectionError(
            f"actor input inventory is incomplete: missing={missing}"
        )
    return inventory


def _finite_array(name: str, value: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise CuroboContactFreeCollectionError(f"{name} must have shape {shape}; got {array.shape}")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise CuroboContactFreeCollectionError(f"{name} contains NaN/Inf")
    return array


def _unit_xyzw_pose(name: str, value: Any) -> np.ndarray:
    pose = _finite_array(name, value, (7,), np.float32)
    norm = float(np.linalg.norm(pose[3:]))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3):
        raise CuroboContactFreeCollectionError(f"{name} quaternion must be unit XYZW")
    return pose


def _rgb(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8)
    if result.shape != (192, 256, 3):
        raise CuroboContactFreeCollectionError("right_wrist_rgb must be [192,256,3] uint8")
    return result


def _depth(value: Any, *, boolean: bool) -> np.ndarray:
    dtype = np.bool_ if boolean else np.float32
    result = np.asarray(value, dtype=dtype)
    if result.shape == (192, 256):
        result = result[..., None]
    if result.shape != (192, 256, 1):
        raise CuroboContactFreeCollectionError("right_wrist depth must be [192,256,1]")
    if not boolean and (not np.isfinite(result).all() or np.any(result < 0.0)):
        raise CuroboContactFreeCollectionError("right_wrist_depth_m must be finite non-negative metres")
    return result


@dataclass(frozen=True)
class CuroboContactFreeCollectionRow:
    """One post-consumption, OPEN-only transition ready for canonical HDF5.

    ``cube``/contact GT is intentionally absent from the payload.  The caller
    may use protected simulator geometry to enforce the pre-submit clearance
    guard, but the resulting training row contains only deployable RGB-D,
    proprioception, previous action, and action/transition receipts.
    """

    timestamp_s: float
    control_step: int
    phase: str
    right_wrist_rgb: Any
    right_wrist_depth_m: Any
    right_wrist_depth_valid: Any
    robot_joint_position_rad: Any
    robot_joint_velocity_rad_s: Any
    ee_pose_robot_root_xyzw: Any
    controller_target_ee_pose_robot_root_xyzw: Any
    gripper_state_open: float
    previous_policy_action_4d_metric_root_m: Sequence[float]
    handoff: CuroboCanonical4DHandoffReceipt
    action_manager_process_action_count: int
    controller_consumption_count: int
    contact_count: int
    gripper_close_count: int
    forbidden_collision_count: int
    hard_limit_event_count: int
    safety_reject_count: int
    planned_segment_minimum_pad_object_distance_m: float
    observed_pad_object_distance_m: float
    contact_guard_distance_m: float

    def validate(self, *, scale: CartesianResidualScale = CartesianResidualScale()) -> None:
        if self.phase not in COLLECTION_ALLOWED_PHASES:
            raise CuroboContactFreeCollectionError("phase is not collectable in OPEN-only scope")
        if not math.isfinite(float(self.timestamp_s)) or float(self.timestamp_s) < 0.0:
            raise CuroboContactFreeCollectionError("timestamp_s must be finite non-negative seconds")
        if type(self.control_step) is not int or self.control_step < 0:
            raise CuroboContactFreeCollectionError("control_step must be a non-negative int")
        _rgb(self.right_wrist_rgb)
        depth = _depth(self.right_wrist_depth_m, boolean=False)
        valid = _depth(self.right_wrist_depth_valid, boolean=True)
        if np.any((~valid) & (depth != 0.0)):
            # The live camera adapter zero-fills invalid depth.  Requiring this
            # prevents a dataset-only masking convention from diverging from
            # the runtime encoder path.
            raise CuroboContactFreeCollectionError("invalid wrist depth must be zero-filled")
        joint_shape = (len(G2_RUNTIME_JOINT_ORDER),)
        _finite_array("robot_joint_position_rad", self.robot_joint_position_rad, joint_shape, np.float32)
        _finite_array("robot_joint_velocity_rad_s", self.robot_joint_velocity_rad_s, joint_shape, np.float32)
        _unit_xyzw_pose("ee_pose_robot_root_xyzw", self.ee_pose_robot_root_xyzw)
        _unit_xyzw_pose(
            "controller_target_ee_pose_robot_root_xyzw",
            self.controller_target_ee_pose_robot_root_xyzw,
        )
        if float(self.gripper_state_open) != 1.0:
            raise CuroboContactFreeCollectionError("contact-free collection requires measured OPEN state=1")
        previous = _finite_array(
            PREVIOUS_POLICY_ACTION_4D_METRIC_FIELD,
            self.previous_policy_action_4d_metric_root_m,
            (CONTACT_FREE_ACTION_DIM,),
            np.float64,
        )
        if previous[3] != 0.0 or float(np.linalg.norm(previous[:3])) > CONTACT_FREE_MAX_DELTA_M + 1.0e-12:
            raise CuroboContactFreeCollectionError(
                "previous action must be metric robot_root [dx_m,dy_m,dz_m,HOLD_OPEN=0] "
                "with norm <= 0.0045 m"
            )
        if self.handoff.frame is not CartesianControlFrame.ROBOT_ROOT:
            raise CuroboContactFreeCollectionError("handoff frame must be robot_root")
        if self.handoff.gripper_intent is not AbstractGripperIntent.OPEN:
            raise CuroboContactFreeCollectionError("handoff must carry OPEN")
        action = np.asarray(self.handoff.normalized_action, dtype=np.float64)
        if action.shape != (4,) or action[3] != 0.0:
            raise CuroboContactFreeCollectionError("handoff action must be normalized [dx,dy,dz,OPEN=0]")
        metric = np.asarray(self.handoff.metric_delta_root_m, dtype=np.float64)
        if not np.allclose(metric, action[:3] * scale.translation_m_per_normalized, rtol=0.0, atol=1e-12):
            raise CuroboContactFreeCollectionError("metric/normalized scale mismatch or double scaling")
        validate_contact_free_action(np.concatenate((metric, np.zeros(1))))
        if self.action_manager_process_action_count != 1 or self.controller_consumption_count != 1:
            raise CuroboContactFreeCollectionError("row lacks exact single action consumption")
        event_counts = (
            self.contact_count,
            self.gripper_close_count,
            self.forbidden_collision_count,
            self.hard_limit_event_count,
            self.safety_reject_count,
        )
        if any(type(value) is not int or value != 0 for value in event_counts):
            raise CuroboContactFreeCollectionError("contact-free row contains prohibited event/reject")
        for name, value in (
            ("planned_segment_minimum_pad_object_distance_m", self.planned_segment_minimum_pad_object_distance_m),
            ("observed_pad_object_distance_m", self.observed_pad_object_distance_m),
            ("contact_guard_distance_m", self.contact_guard_distance_m),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise CuroboContactFreeCollectionError(f"{name} must be positive metres")
        if not (
            self.planned_segment_minimum_pad_object_distance_m > self.contact_guard_distance_m
            and self.observed_pad_object_distance_m > self.contact_guard_distance_m
        ):
            raise CuroboContactFreeCollectionError("row enters contact guard or lacks strict clearance")

    def fields(self) -> dict[str, np.ndarray]:
        """Return one-row fields for ``G2DemonstrationWriter.write_episode``.

        Registered fields retain their historic names.  Four branch-native
        fields have deliberately explicit names/units and must be written with
        ``allow_unregistered_fields=True`` until a schema-versioned HDF5
        extension is approved.
        """
        self.validate()
        normalized = np.asarray(self.handoff.normalized_action, dtype=np.float32)
        metric = np.concatenate((np.asarray(self.handoff.metric_delta_root_m, dtype=np.float32), np.zeros(1, dtype=np.float32)))
        compatibility7 = np.concatenate((normalized[:3], np.zeros(3, dtype=np.float32), np.ones(1, dtype=np.float32)))
        compatibility8 = np.concatenate((normalized[:3], np.zeros(4, dtype=np.float32), np.ones(1, dtype=np.float32)))
        return {
            "timestamp_s": np.asarray([[self.timestamp_s]], dtype=np.float32),
            "control_step": np.asarray([[self.control_step]], dtype=np.int64),
            "right_wrist_rgb": _rgb(self.right_wrist_rgb)[None],
            "right_wrist_depth_m": _depth(self.right_wrist_depth_m, boolean=False)[None],
            "right_wrist_depth_valid": _depth(self.right_wrist_depth_valid, boolean=True)[None],
            "robot_joint_position_rad": _finite_array("robot_joint_position_rad", self.robot_joint_position_rad, (len(G2_RUNTIME_JOINT_ORDER),), np.float32)[None],
            "robot_joint_velocity_rad_s": _finite_array("robot_joint_velocity_rad_s", self.robot_joint_velocity_rad_s, (len(G2_RUNTIME_JOINT_ORDER),), np.float32)[None],
            "ee_pose_robot_root_xyzw": _unit_xyzw_pose("ee_pose_robot_root_xyzw", self.ee_pose_robot_root_xyzw)[None],
            "controller_target_ee_pose_robot_root_xyzw": _unit_xyzw_pose("controller_target_ee_pose_robot_root_xyzw", self.controller_target_ee_pose_robot_root_xyzw)[None],
            "gripper_command": np.asarray([[1.0]], dtype=np.float32),
            "gripper_state": np.asarray([[1.0]], dtype=np.float32),
            "operator_action_8d": compatibility8[None],
            "canonical_policy_action_7d": compatibility7[None],
            POLICY_ACTION_4D_NORMALIZED_FIELD: normalized[None],
            POLICY_ACTION_4D_METRIC_FIELD: metric[None],
            PREVIOUS_POLICY_ACTION_4D_METRIC_FIELD: np.asarray(
                self.previous_policy_action_4d_metric_root_m, dtype=np.float32
            )[None],
            "phase_label": np.asarray([self.phase.encode("ascii")]),
            "contact_free_execution_mode": np.asarray([CONTACT_FREE_EXECUTION_MODE.encode("ascii")]),
            "action_manager_process_action_count": np.asarray([[1]], dtype=np.int64),
            "controller_consumption_count": np.asarray([[1]], dtype=np.int64),
            "planned_segment_minimum_pad_object_distance_m": np.asarray([[self.planned_segment_minimum_pad_object_distance_m]], dtype=np.float32),
            "observed_pad_object_distance_m": np.asarray([[self.observed_pad_object_distance_m]], dtype=np.float32),
            "contact_guard_distance_m": np.asarray([[self.contact_guard_distance_m]], dtype=np.float32),
        }

    @staticmethod
    def field_metadata() -> Mapping[str, Mapping[str, str]]:
        return {
            POLICY_ACTION_4D_NORMALIZED_FIELD: {
                "schema": "g2_ee_xyz_gripper_v1",
                "unit": "normalized; normalized_xyz_times_0.0225m; gripper_probability_0_open",
                "frame": "robot_root",
                "source": "post_env_step_canonical_receipt",
                "actor_allowed": "true",
            },
            POLICY_ACTION_4D_METRIC_FIELD: {
                "schema": CUROBO_CONTACT_FREE_COLLECTION_SCHEMA,
                "unit": "m,m,m,HOLD_OPEN=0",
                "frame": "robot_root",
                "source": "post_env_step_canonical_receipt",
                "actor_allowed": "false; action_label_only",
            },
            PREVIOUS_POLICY_ACTION_4D_METRIC_FIELD: {
                "schema": "g2_policy_branch_observation_v2",
                "unit": "m,m,m,HOLD_OPEN=0",
                "frame": "robot_root",
                "source": "previous_accepted_canonical_receipt",
                "actor_allowed": "true",
            },
        }


def build_collection_manifest(*, candidate_a_sha256: str, seed: int) -> dict[str, Any]:
    """Pure manifest used to bind a later writer without running a collection."""
    if len(candidate_a_sha256) != 64 or any(c not in "0123456789abcdef" for c in candidate_a_sha256.lower()):
        raise CuroboContactFreeCollectionError("candidate asset hash must be SHA-256")
    if seed != 42:
        raise CuroboContactFreeCollectionError("contact-free Candidate-A collection seed must be 42")
    return {
        "schema": CUROBO_CONTACT_FREE_COLLECTION_SCHEMA,
        "execution_mode": CONTACT_FREE_EXECUTION_MODE,
        "candidate_a_sha256": candidate_a_sha256,
        "seed": seed,
        "action": {
            "native_field": POLICY_ACTION_4D_NORMALIZED_FIELD,
            "metric_label_field": POLICY_ACTION_4D_METRIC_FIELD,
            "dimension": 4,
            "frame": "robot_root",
            "translation_m_per_normalized": 0.0225,
            "maximum_metric_translation_m": CONTACT_FREE_MAX_DELTA_M,
            "gripper": "HOLD_OPEN=0; compatibility7=+1_OPEN",
            "orientation": "fixed_zero",
            "elbow": "fixed_zero",
            "scale_count": 1,
        },
        "previous_policy_action": {
            "field": PREVIOUS_POLICY_ACTION_4D_METRIC_FIELD,
            "representation": "[dx_m,dy_m,dz_m,HOLD_OPEN=0]",
            "frame": "robot_root",
            "maximum_metric_translation_m": CONTACT_FREE_MAX_DELTA_M,
            "normalization": "FORBIDDEN_IN_POLICY_OBSERVATION",
            "source": "previous_accepted_post_env_step_metric_receipt",
        },
        "actor_inputs": list(sorted(DEPLOYABLE_ACTOR_FIELD_NAMES)),
        "actor_joint_projection": {
            "field": "robot_joint_position_rad / robot_joint_velocity_rad_s",
            "layout": "idx61_arm_r_joint1 through idx67_arm_r_joint7",
            "unit": "rad / rad/s",
        },
        "privileged_actor_inputs_forbidden": [
            "cube_gt_pose_robot_root_xyzw",
            "cube visibility labels",
            "contact labels",
            "phase_label",
            "planner target",
            "clearance geometry",
        ],
        "row_acceptance": {
            "exact_env_step_consumption": 1,
            "controller_consumption": 1,
            "contact_count": 0,
            "gripper_close_count": 0,
            "forbidden_collision_count": 0,
            "hard_limit_event_count": 0,
            "safety_reject_count": 0,
            "strict_clearance": "planned_and_observed > contact_guard_distance_m",
        },
    }


def build_contact_free_episode_payload(
    rows: Sequence[CuroboContactFreeCollectionRow],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Stack rows after proving temporal action and 50 Hz alignment.

    The first previous action is explicitly the reset OPEN zero packet.  Every
    later row must carry the preceding *accepted* 4-D receipt, not the latest
    cuRobo waypoint, an unconsumed router packet, or a controller joint
    target.  This is the training/runtime temporal contract.
    """
    if not rows:
        raise CuroboContactFreeCollectionError("episode requires at least one row")
    all_fields: list[dict[str, np.ndarray]] = []
    prior = np.zeros(4, dtype=np.float64)
    prior_timestamp: float | None = None
    prior_step: int | None = None
    for row in rows:
        row.validate()
        observed_previous = np.asarray(row.previous_policy_action_4d_metric_root_m, dtype=np.float64)
        if not np.allclose(observed_previous, prior, rtol=0.0, atol=1.0e-7):
            raise CuroboContactFreeCollectionError(
                "previous_policy_action does not equal prior accepted 4-D receipt"
            )
        if prior_timestamp is not None:
            if not math.isclose(row.timestamp_s - prior_timestamp, 0.02, rel_tol=0.0, abs_tol=1.0e-5):
                raise CuroboContactFreeCollectionError("collection timestamps must be consecutive 50 Hz rows")
            if row.control_step != prior_step + 1:
                raise CuroboContactFreeCollectionError("collection control_step must increase by one")
        all_fields.append(row.fields())
        prior = np.concatenate(
            (np.asarray(row.handoff.metric_delta_root_m, dtype=np.float64), np.zeros(1, dtype=np.float64))
        )
        prior_timestamp, prior_step = row.timestamp_s, row.control_step
    names = tuple(all_fields[0])
    if any(tuple(item) != names for item in all_fields[1:]):
        raise CuroboContactFreeCollectionError("row field inventory mismatch")
    payload = {name: np.concatenate([item[name] for item in all_fields], axis=0) for name in names}
    validity = {name: np.ones(len(rows), dtype=np.bool_) for name in names}
    return payload, validity


def write_contact_free_episode_hdf5(
    destination: str | Path,
    *,
    episode_id: str,
    rows: Sequence[CuroboContactFreeCollectionRow],
    candidate_a_sha256: str,
    seed: int,
) -> str:
    """Write one immutable, canonical-HDF5 *scoped* POLICY episode.

    This is intentionally a writer adapter rather than a new collection
    format.  It does not make the episode a full CONTACT/LIFT demonstration:
    the standard contract will mark missing unrelated event/GT fields, while
    the metadata seals the allowed OPEN-only branch fields and their actor
    eligibility.  A later trainer must bind this manifest, not infer policy
    eligibility from generic HDF5 field names.
    """
    from geniesim.rl.isaaclab.g2_rebuild.data_contract import DatasetSource, EpisodeRole
    from geniesim.rl.isaaclab.g2_rebuild.demonstration import G2DemonstrationWriter

    manifest = build_collection_manifest(candidate_a_sha256=candidate_a_sha256, seed=seed)
    payload, validity = build_contact_free_episode_payload(rows)
    with G2DemonstrationWriter(destination, dataset_source=DatasetSource.POLICY) as writer:
        location = writer.write_episode(
            episode_id,
            role=EpisodeRole.RECOVERY,
            payload=payload,
            field_validity=validity,
            field_metadata=CuroboContactFreeCollectionRow.field_metadata(),
            metadata={
                "collection_manifest": manifest,
                "terminal_reason": "OPEN_HANDOFF_STOP_CONTACT_FORBIDDEN",
                "contact_training_eligible": False,
                "contact_free_bc_eligible_only_after_live_collection_gate": False,
            },
            allow_unregistered_fields=True,
        )
    return location.hdf5_path


__all__ = [
    "COLLECTION_ALLOWED_PHASES",
    "CUROBO_CONTACT_FREE_COLLECTION_SCHEMA",
    "CuroboContactFreeCollectionError",
    "CuroboContactFreeCollectionRow",
    "DEPLOYABLE_ACTOR_FIELD_NAMES",
    "FORBIDDEN_ACTOR_FIELD_NAMES",
    "POLICY_ACTION_4D_METRIC_FIELD",
    "POLICY_ACTION_4D_NORMALIZED_FIELD",
    "assert_deployable_actor_input_inventory",
    "build_contact_free_episode_payload",
    "build_collection_manifest",
    "write_contact_free_episode_hdf5",
]
