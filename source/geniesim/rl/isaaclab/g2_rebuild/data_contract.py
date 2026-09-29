# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""One executable data contract for the controlled G2 grasp rebuild.

This module deliberately has no Isaac Lab or Kit imports.  Dataset tools,
sensor acquisition, observation construction and learning stages all import
the same definitions from here.  Unit/frame/schema mismatches are rejected;
they are never repaired silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


class SchemaVersion(str, Enum):
    CONTROLLED_REBUILD_V1 = "g2_controlled_rebuild_v1"
    CONTROLLED_REBUILD_V2 = "g2_controlled_rebuild_v2"


class Unit(str, Enum):
    NONE = "none"
    NORMALIZED = "normalized"
    PROBABILITY = "probability_0_1"
    METER = "m"
    METER_PER_SECOND = "m/s"
    METER_PER_SECOND_SQUARED = "m/s^2"
    RADIAN = "rad"
    RADIAN_PER_SECOND = "rad/s"
    RADIAN_PER_SECOND_SQUARED = "rad/s^2"
    NEWTON = "N"
    JOULE = "J"
    SECOND = "s"
    CONTROL_STEP = "control_step"
    UINT8_RGB = "uint8_rgb_0_255"
    BINARY = "binary_0_1"
    QUATERNION_XYZW = "unit_quaternion_xyzw"
    POSE_XYZ_M_QUATERNION_XYZW = "xyz_m+unit_quaternion_xyzw"


class CoordinateFrame(str, Enum):
    NONE = "none"
    WORLD = "world"
    ENVIRONMENT = "environment"
    ROBOT_ROOT = "robot_root"
    HEAD_OPTICAL = "head_optical"
    RIGHT_WRIST_OPTICAL = "right_wrist_optical"
    ROBOT_JOINT = "robot_joint"
    RIGHT_GRIPPER_INNER_OUTER = "right_gripper_inner_outer"
    ACTION = "action"


class TensorSource(str, Enum):
    SENSOR = "sensor"
    LIVE_FK = "live_fk"
    VISION_PREDICTION = "vision_prediction"
    DERIVED_DEPLOYABLE = "derived_deployable"
    POLICY = "policy"
    HUMAN = "human"
    MIGRATED = "migrated"
    PRIVILEGED_GT = "privileged_gt"
    PHYSICAL_EVENT = "physical_event"
    ANNOTATION = "annotation"


class DatasetSource(str, Enum):
    HUMAN = "HUMAN"
    MIGRATED = "MIGRATED"
    MIMIC_SYNTHETIC = "MIMIC_SYNTHETIC"
    POLICY = "POLICY"


class EpisodeRole(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RECOVERY = "recovery"


class GateOutcome(str, Enum):
    PASS = "PASS"
    HARD_STOP = "HARD_STOP"
    STAGE_BLOCK = "STAGE_BLOCK"
    ADVISORY_CONTINUE = "ADVISORY_CONTINUE"


# Ordered live PhysX articulation contract from the validated G2 runtime USD
# (asset SHA-256 7751a265...dc032132).  Dataset joint state is the full
# articulation state; the 8 controlled right-arm/master joints remain a
# separate action/controller concept and must never be mislabeled as "robot".
G2_RUNTIME_JOINT_ORDER = (
    "idx01_body_joint1", "idx02_body_joint2",
    "idx111_chassis_lwheel_front_joint1", "idx121_chassis_lwheel_rear_joint1",
    "idx131_chassis_rwheel_front_joint1", "idx141_chassis_rwheel_rear_joint1",
    "idx03_body_joint3", "idx112_chassis_lwheel_front_joint2",
    "idx122_chassis_lwheel_rear_joint2", "idx132_chassis_rwheel_front_joint2",
    "idx142_chassis_rwheel_rear_joint2", "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx12_head_joint2", "idx21_arm_l_joint1",
    "idx61_arm_r_joint1", "idx13_head_joint3", "idx22_arm_l_joint2",
    "idx62_arm_r_joint2", "idx23_arm_l_joint3", "idx63_arm_r_joint3",
    "idx24_arm_l_joint4", "idx64_arm_r_joint4", "idx25_arm_l_joint5",
    "idx65_arm_r_joint5", "idx26_arm_l_joint6", "idx66_arm_r_joint6",
    "idx27_arm_l_joint7", "idx67_arm_r_joint7", "idx31_gripper_l_inner_joint1",
    "idx41_gripper_l_outer_joint1", "idx71_gripper_r_inner_joint1",
    "idx81_gripper_r_outer_joint1", "idx32_gripper_l_inner_joint3",
    "idx42_gripper_l_outer_joint3", "idx72_gripper_r_inner_joint3",
    "idx82_gripper_r_outer_joint3", "idx33_gripper_l_inner_joint4",
    "idx43_gripper_l_outer_joint4", "idx73_gripper_r_inner_joint4",
    "idx83_gripper_r_outer_joint4", "idx39_gripper_l_inner_joint0",
    "idx49_gripper_l_outer_joint0", "idx79_gripper_r_inner_joint0",
    "idx89_gripper_r_outer_joint0",
)


@dataclass(frozen=True)
class GateRecord:
    outcome: GateOutcome
    code: str
    message: str
    metrics: Mapping[str, float | int | bool | str] = field(default_factory=dict)

    @property
    def permits_process(self) -> bool:
        return self.outcome is not GateOutcome.HARD_STOP

    @property
    def permits_next_stage(self) -> bool:
        return self.outcome in (GateOutcome.PASS, GateOutcome.ADVISORY_CONTINUE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "code": self.code,
            "message": self.message,
            "metrics": dict(self.metrics),
            "permits_process": self.permits_process,
            "permits_next_stage": self.permits_next_stage,
        }


def _dtype_name(value: Any) -> str:
    text = str(getattr(value, "dtype", ""))
    for prefix in ("torch.", "numpy."):
        text = text.removeprefix(prefix)
    return text


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("contract tensor must expose shape")
    return tuple(int(dimension) for dimension in shape)


def _all_finite(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
    except ImportError:
        pass
    return bool(np.isfinite(np.asarray(value)).all())


def _minimum_maximum(value: Any) -> tuple[float, float]:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.min().item()), float(value.max().item())
    except ImportError:
        pass
    array = np.asarray(value)
    return float(np.min(array)), float(np.max(array))


@dataclass(frozen=True)
class TensorSpec:
    """Metadata and validation rules for one semantic sample tensor.

    ``shape`` is the per-sample trailing shape.  Batch and sequence leading
    dimensions are allowed, which lets the same specification validate a live
    packet, an HDF5 episode and a recurrent minibatch.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    unit: Unit
    frame: CoordinateFrame
    normalization: str
    timestamp: str
    source: TensorSource
    schema_version: SchemaVersion = SchemaVersion.CONTROLLED_REBUILD_V2
    actor_allowed: bool = False
    finite_required: bool = True
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name is required")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"{self.name}: sample dimensions must be positive")
        if not self.dtype or not self.normalization or not self.timestamp:
            raise ValueError(f"{self.name}: dtype/normalization/timestamp are required")
        if self.value_range is not None:
            low, high = self.value_range
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(f"{self.name}: invalid value range")
        if self.unit in (
            Unit.METER,
            Unit.METER_PER_SECOND,
            Unit.METER_PER_SECOND_SQUARED,
            Unit.POSE_XYZ_M_QUATERNION_XYZW,
        ) and self.frame is CoordinateFrame.NONE:
            raise ValueError(f"{self.name}: spatial quantity requires a frame")

    def validate(self, value: Any) -> None:
        observed_shape = _shape(value)
        if len(observed_shape) < len(self.shape) or observed_shape[-len(self.shape) :] != self.shape:
            raise ValueError(
                f"{self.name}: trailing shape {self.shape} required; got {observed_shape}"
            )
        observed_dtype = _dtype_name(value)
        aliases = {
            "float": {"float32", "float64"},
            # Binary semantics are part of the canonical schema.  Accepting
            # uint8 here makes a legacy storage convention look like a valid
            # canonical boolean without an explicit migration step.
            "bool": {"bool"},
        }
        accepted = aliases.get(self.dtype, {self.dtype})
        if observed_dtype not in accepted:
            raise TypeError(
                f"{self.name}: dtype {self.dtype} required; got {observed_dtype}"
            )
        if self.finite_required and not _all_finite(value):
            raise ValueError(f"{self.name}: NaN/Inf is forbidden")
        if self.value_range is not None:
            minimum, maximum = _minimum_maximum(value)
            low, high = self.value_range
            if minimum < low or maximum > high:
                raise ValueError(
                    f"{self.name}: values [{minimum},{maximum}] outside [{low},{high}]"
                )

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result["unit"] = self.unit.value
        result["frame"] = self.frame.value
        result["source"] = self.source.value
        result["schema_version"] = self.schema_version.value
        return result


G2_GOAL_CONDITIONING_SCHEMA = "g2_controlled_rebuild_goal_conditioning_v1"


@dataclass(frozen=True)
class G2GoalConditioningContract:
    """Opt-in M9 goal extension without changing the fixed sensor schema.

    The base data contract intentionally remains byte-for-byte compatible with
    previously signed M0--M9 artifacts.  HER-enabled *fresh* replay artifacts
    bind this extension hash separately.  Goals are three-dimensional metres
    in ``ROBOT_ROOT``.  They are never direct actor inputs: the only deployable
    actor feature is ``desired_goal - achieved_goal``, carried by the existing
    ``relative_grasp_xyz_robot_root_m`` input.
    """

    schema: str
    achieved_goal: TensorSpec
    next_achieved_goal: TensorSpec
    desired_goal: TensorSpec

    @classmethod
    def canonical(cls) -> "G2GoalConditioningContract":
        common = dict(
            shape=(3,),
            dtype="float32",
            unit=Unit.METER,
            frame=CoordinateFrame.ROBOT_ROOT,
            normalization="identity_meter",
            timestamp="timestamp_s",
            schema_version=SchemaVersion.CONTROLLED_REBUILD_V2,
            actor_allowed=False,
        )
        return cls(
            schema=G2_GOAL_CONDITIONING_SCHEMA,
            achieved_goal=TensorSpec(
                "achieved_goal_robot_root_m",
                source=TensorSource.LIVE_FK,
                **common,
            ),
            next_achieved_goal=TensorSpec(
                "next_achieved_goal_robot_root_m",
                source=TensorSource.LIVE_FK,
                **common,
            ),
            desired_goal=TensorSpec(
                "desired_goal_robot_root_m",
                # Original replay derives this from the deployable vision
                # prediction; HER derives it from a same-episode future FK
                # achieved goal.  Per-row provenance distinguishes the two.
                source=TensorSource.DERIVED_DEPLOYABLE,
                **common,
            ),
        )

    def validate(
        self,
        *,
        achieved_goal: Any,
        next_achieved_goal: Any,
        desired_goal: Any,
    ) -> None:
        self.achieved_goal.validate(achieved_goal)
        self.next_achieved_goal.validate(next_achieved_goal)
        self.desired_goal.validate(desired_goal)
        require_same_frame(
            self.achieved_goal, self.next_achieved_goal, self.desired_goal
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "actor_direct_goal_input": False,
            "actor_goal_feature": (
                "relative_grasp_xyz_robot_root_m = "
                "desired_goal_robot_root_m - achieved_goal_robot_root_m"
            ),
            "critic_goal_feature": "same deployable relative feature",
            "ground_truth_goal_allowed": False,
            "original_desired_goal_source": "vision_prediction",
            "her_desired_goal_source": "same_episode_future_live_fk",
            "tensors": {
                "achieved_goal": self.achieved_goal.metadata(),
                "next_achieved_goal": self.next_achieved_goal.metadata(),
                "desired_goal": self.desired_goal.metadata(),
            },
        }

    def sha256(self) -> str:
        return canonical_json_sha256(self.metadata())


def require_same_frame(*specs: TensorSpec) -> CoordinateFrame:
    if not specs:
        raise ValueError("at least one tensor specification is required")
    frames = {spec.frame for spec in specs}
    if len(frames) != 1 or CoordinateFrame.NONE in frames:
        detail = {spec.name: spec.frame.value for spec in specs}
        raise ValueError(f"G2_FRAME_MISMATCH:{detail}")
    return specs[0].frame


ACTOR_ALLOWED_INPUTS = frozenset(
    {
        "head_rgb",
        "head_depth_m",
        "head_depth_valid",
        "right_wrist_rgb",
        "right_wrist_depth_m",
        "right_wrist_depth_valid",
        "head_camera_pose_robot_root_xyzw",
        "right_wrist_camera_pose_robot_root_xyzw",
        "robot_joint_position_rad",
        "robot_joint_velocity_rad_s",
        "ee_pose_robot_root_xyzw",
        "predicted_cube_xyz_robot_root_m",
        "predicted_cube_velocity_robot_root_m_s",
        "predicted_head_visibility_probability",
        "predicted_right_wrist_visibility_probability",
        "predicted_fused_confidence",
        "distal_pad_midpoint_robot_root_m",
        "relative_grasp_xyz_robot_root_m",
        "ee_linear_velocity_robot_root_m_s",
        "relative_velocity_robot_root_m_s",
        "gripper_state",
        "previous_action",
        "camera_frame_age_s",
    }
)

PRIVILEGED_FIELDS = frozenset(
    {
        "cube_gt_pose_robot_root_xyzw",
        "head_cube_gt_pose_at_capture_robot_root_xyzw",
        "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw",
        "head_cube_visible",
        "right_wrist_cube_visible",
        "head_cube_visibility_valid",
        "right_wrist_cube_visibility_valid",
        "cube_center_height_world_m",
        "exact_inner_contact",
        "exact_outer_contact",
        "exact_bilateral_contact",
        "stable_grasp",
        "physical_lift_start",
        "success",
        "failure",
        "forbidden_collision",
        "safety_violation",
        "future_contact_step",
        "future_stable_step",
        "simulation_object_state",
    }
)


def _canonical_specs() -> tuple[TensorSpec, ...]:
    v = SchemaVersion.CONTROLLED_REBUILD_V2
    sample_time = "timestamp_s"
    camera_time = "camera_timestamp_s"
    return (
        TensorSpec("timestamp_s", (1,), "float", Unit.SECOND, CoordinateFrame.NONE, "identity", sample_time, TensorSource.SENSOR, v),
        TensorSpec("control_step", (1,), "int64", Unit.CONTROL_STEP, CoordinateFrame.NONE, "identity", sample_time, TensorSource.SENSOR, v),
        TensorSpec("head_rgb", (192, 256, 3), "uint8", Unit.UINT8_RGB, CoordinateFrame.HEAD_OPTICAL, "divide_255_for_network", camera_time, TensorSource.SENSOR, v, True, False, (0.0, 255.0)),
        TensorSpec("head_depth_m", (192, 256, 1), "float32", Unit.METER, CoordinateFrame.HEAD_OPTICAL, "identity_meter_with_valid_mask", camera_time, TensorSource.SENSOR, v, True),
        TensorSpec("head_depth_valid", (192, 256, 1), "bool", Unit.BINARY, CoordinateFrame.HEAD_OPTICAL, "identity", camera_time, TensorSource.SENSOR, v, True, False, (0.0, 1.0)),
        TensorSpec("right_wrist_rgb", (192, 256, 3), "uint8", Unit.UINT8_RGB, CoordinateFrame.RIGHT_WRIST_OPTICAL, "divide_255_for_network", camera_time, TensorSource.SENSOR, v, True, False, (0.0, 255.0)),
        TensorSpec("right_wrist_depth_m", (192, 256, 1), "float32", Unit.METER, CoordinateFrame.RIGHT_WRIST_OPTICAL, "identity_meter_with_valid_mask", camera_time, TensorSource.SENSOR, v, True),
        TensorSpec("right_wrist_depth_valid", (192, 256, 1), "bool", Unit.BINARY, CoordinateFrame.RIGHT_WRIST_OPTICAL, "identity", camera_time, TensorSource.SENSOR, v, True, False, (0.0, 1.0)),
        TensorSpec("head_camera_pose_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "camera_optical_to_robot_root_at_exact_head_capture; quaternion_order=xyzw", "camera_timestamp_s[head]", TensorSource.LIVE_FK, v, True),
        TensorSpec("right_wrist_camera_pose_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "camera_optical_to_robot_root_at_exact_right_wrist_capture; quaternion_order=xyzw", "camera_timestamp_s[right_wrist]", TensorSource.LIVE_FK, v, True),
        TensorSpec("camera_timestamp_s", (2,), "float32", Unit.SECOND, CoordinateFrame.NONE, "camera_order=head,right_wrist", sample_time, TensorSource.SENSOR, v),
        TensorSpec("camera_sequence_id", (2,), "int64", Unit.NONE, CoordinateFrame.NONE, "camera_order=head,right_wrist; exact_sensor_frame_identity", camera_time, TensorSource.SENSOR, v),
        TensorSpec("camera_frame_age_s", (2,), "float32", Unit.SECOND, CoordinateFrame.NONE, "camera_order=head,right_wrist", sample_time, TensorSource.DERIVED_DEPLOYABLE, v, True, True, (0.0, 10.0)),
        TensorSpec("robot_joint_position_rad", (len(G2_RUNTIME_JOINT_ORDER),), "float32", Unit.RADIAN, CoordinateFrame.ROBOT_JOINT, "ordered_full_articulation; joint_order_in_contract_metadata", sample_time, TensorSource.SENSOR, v, True),
        TensorSpec("robot_joint_velocity_rad_s", (len(G2_RUNTIME_JOINT_ORDER),), "float32", Unit.RADIAN_PER_SECOND, CoordinateFrame.ROBOT_JOINT, "ordered_full_articulation; joint_order_in_contract_metadata", sample_time, TensorSource.SENSOR, v, True),
        TensorSpec("ee_pose_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "identity; quaternion_order=xyzw", sample_time, TensorSource.LIVE_FK, v, True),
        TensorSpec("controller_target_ee_pose_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "identity; quaternion_order=xyzw; exact_controller_target", sample_time, TensorSource.POLICY, v),
        TensorSpec("distal_link_origin_midpoint_environment_m", (3,), "float32", Unit.METER, CoordinateFrame.ENVIRONMENT, "identity", sample_time, TensorSource.LIVE_FK, v),
        TensorSpec("distal_pad_midpoint_robot_root_m", (3,), "float32", Unit.METER, CoordinateFrame.ROBOT_ROOT, "identity", sample_time, TensorSource.LIVE_FK, v, True),
        TensorSpec("predicted_cube_xyz_robot_root_m", (3,), "float32", Unit.METER, CoordinateFrame.ROBOT_ROOT, "identity", sample_time, TensorSource.VISION_PREDICTION, v, True),
        TensorSpec("predicted_cube_velocity_robot_root_m_s", (3,), "float32", Unit.METER_PER_SECOND, CoordinateFrame.ROBOT_ROOT, "finite_difference_or_temporal_vision", sample_time, TensorSource.VISION_PREDICTION, v, True),
        TensorSpec("predicted_head_visibility_probability", (1,), "float32", Unit.PROBABILITY, CoordinateFrame.HEAD_OPTICAL, "sigmoid_visibility_head; deployable_rgbd_only", sample_time, TensorSource.VISION_PREDICTION, v, True, True, (0.0, 1.0)),
        TensorSpec("predicted_right_wrist_visibility_probability", (1,), "float32", Unit.PROBABILITY, CoordinateFrame.RIGHT_WRIST_OPTICAL, "sigmoid_visibility_head; deployable_rgbd_only", sample_time, TensorSource.VISION_PREDICTION, v, True, True, (0.0, 1.0)),
        TensorSpec("predicted_fused_confidence", (1,), "float32", Unit.PROBABILITY, CoordinateFrame.NONE, "one_minus_product_of_camera_invisibility; threshold_bound_to_signed_model_config", sample_time, TensorSource.VISION_PREDICTION, v, True, True, (0.0, 1.0)),
        TensorSpec("vision_prediction_timestamp_s", (1,), "float32", Unit.SECOND, CoordinateFrame.NONE, "identity", sample_time, TensorSource.VISION_PREDICTION, v),
        TensorSpec("vision_prediction_valid", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "identity", sample_time, TensorSource.VISION_PREDICTION, v, False, False, (0.0, 1.0)),
        TensorSpec("relative_grasp_xyz_robot_root_m", (3,), "float32", Unit.METER, CoordinateFrame.ROBOT_ROOT, "identity", sample_time, TensorSource.DERIVED_DEPLOYABLE, v, True),
        TensorSpec("ee_linear_velocity_robot_root_m_s", (3,), "float32", Unit.METER_PER_SECOND, CoordinateFrame.ROBOT_ROOT, "identity", sample_time, TensorSource.DERIVED_DEPLOYABLE, v, True),
        TensorSpec("relative_velocity_robot_root_m_s", (3,), "float32", Unit.METER_PER_SECOND, CoordinateFrame.ROBOT_ROOT, "identity", sample_time, TensorSource.DERIVED_DEPLOYABLE, v, True),
        TensorSpec("gripper_command", (1,), "float32", Unit.NORMALIZED, CoordinateFrame.ACTION, "minus1_close_plus1_open", sample_time, TensorSource.HUMAN, v, False, True, (-1.0, 1.0)),
        TensorSpec("gripper_state", (1,), "float32", Unit.NORMALIZED, CoordinateFrame.ROBOT_JOINT, "closed_0_open_1", sample_time, TensorSource.SENSOR, v, True, True, (0.0, 1.0)),
        TensorSpec("gripper_master_raw_rad", (1,), "float32", Unit.RADIAN, CoordinateFrame.ROBOT_JOINT, "identity", sample_time, TensorSource.SENSOR, v),
        TensorSpec("right_gripper_inner_outer_contact_force_n", (2,), "float32", Unit.NEWTON, CoordinateFrame.RIGHT_GRIPPER_INNER_OUTER, "identity", sample_time, TensorSource.SENSOR, v),
        TensorSpec("right_gripper_inner_outer_contact_valid", (2,), "bool", Unit.BINARY, CoordinateFrame.RIGHT_GRIPPER_INNER_OUTER, "identity", sample_time, TensorSource.SENSOR, v, False, False, (0.0, 1.0)),
        TensorSpec("keyboard_physical_command", (8,), "float32", Unit.NONE, CoordinateFrame.ACTION, "xyz_meter_rotvec_radian_elbow_radian_gripper_sign; scale_exactly_once", sample_time, TensorSource.HUMAN, v),
        TensorSpec("operator_action_8d", (8,), "float32", Unit.NORMALIZED, CoordinateFrame.ACTION, "xyz_rotvec_elbow_gripper", sample_time, TensorSource.HUMAN, v, False, True, (-1.0, 1.0)),
        TensorSpec("canonical_policy_action_7d", (7,), "float32", Unit.NORMALIZED, CoordinateFrame.ACTION, "xyz_rotvec_gripper", sample_time, TensorSource.POLICY, v, False, True, (-1.0, 1.0)),
        TensorSpec("applied_actuator_joint_target_rad", (8,), "float32", Unit.RADIAN, CoordinateFrame.ROBOT_JOINT, "right_arm_7_plus_right_gripper_master; ordered_names_in_action_contract", sample_time, TensorSource.POLICY, v),
        TensorSpec("previous_action", (7,), "float32", Unit.NORMALIZED, CoordinateFrame.ACTION, "xyz_rotvec_gripper", sample_time, TensorSource.DERIVED_DEPLOYABLE, v, True, True, (-1.0, 1.0)),
        TensorSpec("cube_gt_pose_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "identity; quaternion_order=xyzw", sample_time, TensorSource.PRIVILEGED_GT, v),
        TensorSpec("head_cube_gt_pose_at_capture_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "renderer_capture_time_gt_only; no interpolation; quaternion_order=xyzw", "camera_timestamp_s[head]", TensorSource.PRIVILEGED_GT, v),
        TensorSpec("right_wrist_cube_gt_pose_at_capture_robot_root_xyzw", (7,), "float32", Unit.POSE_XYZ_M_QUATERNION_XYZW, CoordinateFrame.ROBOT_ROOT, "renderer_capture_time_gt_only; no interpolation; quaternion_order=xyzw", "camera_timestamp_s[right_wrist]", TensorSource.PRIVILEGED_GT, v),
        TensorSpec("head_cube_visible", (1,), "bool", Unit.BINARY, CoordinateFrame.HEAD_OPTICAL, "explicit_object_specific_rendered_rgbd_visibility_at_capture", "camera_timestamp_s[head]", TensorSource.PRIVILEGED_GT, v, False, False, (0.0, 1.0)),
        TensorSpec("right_wrist_cube_visible", (1,), "bool", Unit.BINARY, CoordinateFrame.RIGHT_WRIST_OPTICAL, "explicit_object_specific_rendered_rgbd_visibility_at_capture", "camera_timestamp_s[right_wrist]", TensorSource.PRIVILEGED_GT, v, False, False, (0.0, 1.0)),
        TensorSpec("head_cube_visibility_valid", (1,), "bool", Unit.BINARY, CoordinateFrame.HEAD_OPTICAL, "rendered_rgbd_visibility_measurement_valid_at_capture", "camera_timestamp_s[head]", TensorSource.PRIVILEGED_GT, v, False, False, (0.0, 1.0)),
        TensorSpec("right_wrist_cube_visibility_valid", (1,), "bool", Unit.BINARY, CoordinateFrame.RIGHT_WRIST_OPTICAL, "rendered_rgbd_visibility_measurement_valid_at_capture", "camera_timestamp_s[right_wrist]", TensorSource.PRIVILEGED_GT, v, False, False, (0.0, 1.0)),
        TensorSpec("exact_inner_contact", (1,), "bool", Unit.BINARY, CoordinateFrame.RIGHT_GRIPPER_INNER_OUTER, "identity", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("exact_outer_contact", (1,), "bool", Unit.BINARY, CoordinateFrame.RIGHT_GRIPPER_INNER_OUTER, "identity", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("exact_bilateral_contact", (1,), "bool", Unit.BINARY, CoordinateFrame.RIGHT_GRIPPER_INNER_OUTER, "identity", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("stable_grasp", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "physical_contact_plus_low_slip_hold", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("cube_center_height_world_m", (1,), "float32", Unit.METER, CoordinateFrame.WORLD, "measured_post_action_object_height", sample_time, TensorSource.PHYSICAL_EVENT, v),
        TensorSpec("physical_lift_start", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "measured_cube_height_rising_event", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("success", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "physical_success_event", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("failure", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "terminal_failure_event", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("forbidden_collision", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "safety_authority", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
        TensorSpec("safety_violation", (1,), "bool", Unit.BINARY, CoordinateFrame.NONE, "collision_force_velocity_acceleration_authority", sample_time, TensorSource.PHYSICAL_EVENT, v, False, False, (0.0, 1.0)),
    )


@dataclass(frozen=True)
class G2DataContract:
    schema_version: SchemaVersion
    control_hz: float
    dt_s: float
    tensors: Mapping[str, TensorSpec]
    actor_allowed_inputs: frozenset[str]
    privileged_fields: frozenset[str]

    @classmethod
    def canonical(cls) -> "G2DataContract":
        specs = _canonical_specs()
        return cls(
            schema_version=SchemaVersion.CONTROLLED_REBUILD_V2,
            control_hz=50.0,
            dt_s=0.02,
            tensors={spec.name: spec for spec in specs},
            actor_allowed_inputs=ACTOR_ALLOWED_INPUTS,
            privileged_fields=PRIVILEGED_FIELDS,
        ).validated()

    def validated(self) -> "G2DataContract":
        if not math.isclose(self.control_hz * self.dt_s, 1.0, abs_tol=1.0e-12):
            raise ValueError("control_hz and dt_s are inconsistent")
        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("duplicate tensor names")
        for name, spec in self.tensors.items():
            if name != spec.name or spec.schema_version is not self.schema_version:
                raise ValueError(f"invalid tensor registration: {name}")
        if not self.actor_allowed_inputs.issubset(self.tensors):
            raise ValueError("actor allowlist contains unknown tensor")
        if self.actor_allowed_inputs.intersection(self.privileged_fields):
            raise ValueError("privileged field leaked into actor allowlist")
        for name in self.actor_allowed_inputs:
            if not self.tensors[name].actor_allowed:
                raise ValueError(f"actor allowlist/spec mismatch: {name}")
        return self

    def spec(self, name: str) -> TensorSpec:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"G2_UNKNOWN_TENSOR:{name}") from error

    def validate_payload(
        self, payload: Mapping[str, Any], names: Sequence[str] | None = None
    ) -> None:
        selected = tuple(payload) if names is None else tuple(names)
        missing = [name for name in selected if name not in payload]
        if missing:
            raise KeyError(f"G2_PAYLOAD_MISSING:{missing}")
        for name in selected:
            spec = self.spec(name)
            spec.validate(payload[name])
            if spec.unit is Unit.POSE_XYZ_M_QUATERNION_XYZW:
                value = payload[name]
                try:
                    import torch

                    if isinstance(value, torch.Tensor):
                        norms = torch.linalg.vector_norm(value[..., 3:7], dim=-1)
                        valid = bool(
                            torch.isfinite(norms).all()
                            and torch.all(torch.abs(norms - 1.0) <= 1.0e-4)
                        )
                    else:
                        norms = np.linalg.norm(np.asarray(value)[..., 3:7], axis=-1)
                        valid = bool(
                            np.isfinite(norms).all()
                            and np.all(np.abs(norms - 1.0) <= 1.0e-4)
                        )
                except ImportError:
                    norms = np.linalg.norm(np.asarray(value)[..., 3:7], axis=-1)
                    valid = bool(
                        np.isfinite(norms).all()
                        and np.all(np.abs(norms - 1.0) <= 1.0e-4)
                    )
                if not valid:
                    raise ValueError(f"{name}: quaternion must be unit XYZW")

    def assert_actor_payload(self, payload: Mapping[str, Any]) -> None:
        keys = set(payload)
        if not keys:
            raise RuntimeError("G2_ACTOR_INPUT_EMPTY")
        forbidden = keys.intersection(self.privileged_fields)
        unknown = keys.difference(self.actor_allowed_inputs)
        if forbidden:
            raise RuntimeError(f"G2_GT_RUNTIME_LEAKAGE:{sorted(forbidden)}")
        if unknown:
            raise RuntimeError(f"G2_ACTOR_INPUT_NOT_ALLOWLISTED:{sorted(unknown)}")
        self.validate_payload(payload)

    def require_same_frame(self, *names: str) -> CoordinateFrame:
        return require_same_frame(*(self.spec(name) for name in names))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version.value,
            "control_hz": self.control_hz,
            "dt_s": self.dt_s,
            "actor_allowed_inputs": sorted(self.actor_allowed_inputs),
            "privileged_fields": sorted(self.privileged_fields),
            "robot_joint_order": list(G2_RUNTIME_JOINT_ORDER),
            "tensors": {
                name: self.tensors[name].metadata() for name in sorted(self.tensors)
            },
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_schema(self, value: str | SchemaVersion) -> None:
        observed = value.value if isinstance(value, SchemaVersion) else str(value)
        if observed != self.schema_version.value:
            raise RuntimeError(
                f"G2_SCHEMA_MISMATCH:{observed}!={self.schema_version.value}"
            )


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class G2ArtifactLineage:
    """Immutable lineage shared by every M0--M9 artifact."""

    stage: str
    contract_schema: SchemaVersion
    contract_sha256: str
    config_sha256: str
    source_revision: str
    seed: int
    control_hz: float
    parent_artifact_sha256: tuple[str, ...] = ()
    dataset_sha256: tuple[str, ...] = ()
    wandb_run_identity: str | None = None

    def __post_init__(self) -> None:
        from .source_revision import require_concrete_source_revision

        if not self.stage.strip():
            raise ValueError("G2_ARTIFACT_STAGE_AND_SOURCE_REVISION_REQUIRED")
        require_concrete_source_revision(
            self.source_revision, prefix="G2_ARTIFACT"
        )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("G2_ARTIFACT_SEED_MUST_BE_NONNEGATIVE_INTEGER")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("G2_ARTIFACT_CONTROL_RATE_INVALID")
        hashes = {
            "contract_sha256": (self.contract_sha256,),
            "config_sha256": (self.config_sha256,),
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "dataset_sha256": self.dataset_sha256,
        }
        for kind, values in hashes.items():
            for value in values:
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"G2_ARTIFACT_HASH_INVALID:{kind}:{value!r}")
        if self.wandb_run_identity is not None and not self.wandb_run_identity.strip():
            raise ValueError("G2_ARTIFACT_WANDB_IDENTITY_EMPTY")

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        resolved_config: Mapping[str, Any],
        source_revision: str,
        seed: int,
        parent_artifact_sha256: Sequence[str] = (),
        dataset_sha256: Sequence[str] = (),
        wandb_run_identity: str | None = None,
        contract: G2DataContract | None = None,
    ) -> "G2ArtifactLineage":
        authority = contract or G2DataContract.canonical()
        if not stage or not source_revision or seed < 0:
            raise ValueError("stage/source revision/nonnegative seed are required")
        return cls(
            stage=stage,
            contract_schema=authority.schema_version,
            contract_sha256=authority.sha256(),
            config_sha256=canonical_json_sha256(resolved_config),
            source_revision=source_revision,
            seed=int(seed),
            control_hz=authority.control_hz,
            parent_artifact_sha256=tuple(parent_artifact_sha256),
            dataset_sha256=tuple(dataset_sha256),
            wandb_run_identity=wandb_run_identity,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "G2ArtifactLineage":
        """Parse serialized lineage without weakening dataclass validation."""

        if not isinstance(value, Mapping):
            raise TypeError("G2_ARTIFACT_LINEAGE_MUST_BE_MAPPING")
        required = {
            "stage",
            "contract_schema",
            "contract_sha256",
            "config_sha256",
            "source_revision",
            "seed",
            "control_hz",
            "parent_artifact_sha256",
            "dataset_sha256",
            "wandb_run_identity",
        }
        if set(value) != required:
            raise ValueError("G2_ARTIFACT_LINEAGE_FIELDS_MISMATCH")
        return cls(
            stage=str(value["stage"]),
            contract_schema=SchemaVersion(value["contract_schema"]),
            contract_sha256=str(value["contract_sha256"]),
            config_sha256=str(value["config_sha256"]),
            source_revision=str(value["source_revision"]),
            seed=int(value["seed"]),
            control_hz=float(value["control_hz"]),
            parent_artifact_sha256=tuple(value["parent_artifact_sha256"]),
            dataset_sha256=tuple(value["dataset_sha256"]),
            wandb_run_identity=(
                None
                if value["wandb_run_identity"] is None
                else str(value["wandb_run_identity"])
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["contract_schema"] = self.contract_schema.value
        return result

    def assert_compatible(
        self, *, contract: G2DataContract, config: Mapping[str, Any]
    ) -> None:
        contract.assert_schema(self.contract_schema)
        if self.contract_sha256 != contract.sha256():
            raise RuntimeError("G2_ARTIFACT_CONTRACT_HASH_MISMATCH")
        if self.config_sha256 != canonical_json_sha256(config):
            raise RuntimeError("G2_ARTIFACT_CONFIG_HASH_MISMATCH")
        if not math.isclose(self.control_hz, contract.control_hz, abs_tol=1.0e-12):
            raise RuntimeError("G2_ARTIFACT_CONTROL_RATE_MISMATCH")


__all__ = [
    "ACTOR_ALLOWED_INPUTS",
    "CoordinateFrame",
    "DatasetSource",
    "EpisodeRole",
    "G2ArtifactLineage",
    "G2DataContract",
    "G2GoalConditioningContract",
    "G2_GOAL_CONDITIONING_SCHEMA",
    "G2_RUNTIME_JOINT_ORDER",
    "GateOutcome",
    "GateRecord",
    "PRIVILEGED_FIELDS",
    "SchemaVersion",
    "TensorSource",
    "TensorSpec",
    "Unit",
    "canonical_json_sha256",
    "require_same_frame",
]
