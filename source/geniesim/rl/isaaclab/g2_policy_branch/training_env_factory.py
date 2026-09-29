# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Authoritative config factory for future 4-D G2 policy training.

The public policy remains ``[dx, dy, dz, gripper]``.  The already-qualified
P0 adapter expands that into the existing 8-D redundancy-controller packet
with zero roll/pitch/yaw and elbow fields.  This module only constructs the
existing RGB-D task config, restores termination ownership that the keyboard
recorder deliberately disables, and binds the exact M2-qualified simulation
candidate before an environment is created.

It does not create ``ManagerBasedRLEnv``, start Isaac, or run training.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
from typing import Any

from isaaclab.managers import TerminationTermCfg as DoneTerm

from .. import g2_lift_task_mdp
from ..g2_camera_visibility import precontact_camera_visibility_failure
from ..g2_keyboard_pose import (
    G2_KEYBOARD_CUBE_CENTER_WORLD_M,
    G2_KEYBOARD_TABLE_CENTER_WORLD_M,
)
from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from ..g2_redundancy_teleop_env_cfg import G2RedundancyTeleopEnvCfg
from .runtime_asset_binding import (
    M2QualifiedAssetBindingReceipt,
    bind_m2_qualified_candidate,
)
from .contact_free_candidate_a_binding import (
    ContactFreeCandidateABindingReceipt,
    bind_contact_free_candidate_a,
)
from .candidate_a_left_arm_down_v2 import apply_left_arm_down_v2_to_cfg
from geniesim.rl.sac.keyboard_grasp_contract import (
    canonical_keyboard_grasp_contract,
)


POLICY_4D_TRAINING_ENV_CONTRACT_SCHEMA = "g2_policy_4d_training_env_contract_v2"
POLICY_4D_TASK_GEOMETRY_RUNTIME_BINDING_SCHEMA = (
    "g2_policy_4d_task_geometry_runtime_binding_v1"
)
POLICY_4D_ACTION_COMPONENTS = ("dx", "dy", "dz", "gripper")
CANONICAL_INTERNAL_ACTION_COMPONENTS = (
    "dx",
    "dy",
    "dz",
    "rx",
    "ry",
    "rz",
    "elbow",
    "gripper",
)

# This exact surface was captured by the passing P0-C artifact before P0
# disabled task scoring.  Zero-weight terms remain visible because their
# presence is part of the Manager config contract even though they contribute
# no reward.
EXPECTED_TASK_REWARD_TERMS = (
    "action_rate",
    "any_finger_touch",
    "bilateral_contact",
    "bilateral_contact_maintenance",
    "controlled_joint_acceleration",
    "cube_goal_progress_pbrs",
    "ee_cube_progress_pbrs",
    "grasp_orientation",
    "grasp_slip_excess",
    "joint_vel",
    "lifting_object",
    "object_goal_tracking",
    "object_goal_tracking_fine_grained",
    "outside_certified_region",
    "partial_grasp_terminal_credit",
    "persistent_push",
    "push_recovery_pbrs",
    "reaching_object",
    "stable_grasp",
    "stable_grasp_maintenance",
)
EXPECTED_TASK_CURRICULUM_TERMS = ("action_rate", "joint_vel")
EXPECTED_TASK_TERMINATION_TERMS = (
    "cube_left_table_persistently",
    "fixed_torso_drift",
    "forbidden_collision",
    "object_dropping",
    "object_reached_goal",
    "precontact_camera_visibility",
    "time_out",
)


class Policy4DTrainingEnvContractError(RuntimeError):
    """Raised rather than constructing a task with drifted semantics."""


def _bind_keyboard_v3_time_contract(cfg: G2RedundancyTeleopEnvCfg) -> dict[str, Any]:
    """Bind the frozen 500/50/25-Hz clocks on the Candidate-A branch only."""

    contract = canonical_keyboard_grasp_contract()
    if not math.isclose(float(cfg.sim.dt), contract.physics_dt_s, rel_tol=0.0, abs_tol=1.0e-12):
        raise Policy4DTrainingEnvContractError("G2_POLICY_PHYSICS_DT_NOT_0P002_S")
    if int(cfg.decimation) != contract.physics_hz // contract.control_hz:
        raise Policy4DTrainingEnvContractError("G2_POLICY_CONTROL_DECIMATION_NOT_10")
    cfg.sim.render_interval = contract.physics_hz // contract.rgbd_acquisition_hz
    for camera_name in ("head_camera", "right_wrist_camera"):
        camera_cfg = getattr(cfg.scene, camera_name, None)
        if camera_cfg is None:
            raise Policy4DTrainingEnvContractError(
                f"G2_POLICY_REQUIRED_CAMERA_MISSING:{camera_name}"
            )
        camera_cfg.update_period = contract.rgbd_dt_s
    return {
        "control_hz": contract.control_hz,
        "control_dt_s": contract.policy_dt_s,
        "physics_hz": contract.physics_hz,
        "physics_dt_s": contract.physics_dt_s,
        "rgbd_hz": contract.rgbd_acquisition_hz,
        "rgbd_dt_s": contract.rgbd_dt_s,
        "render_interval_physics_steps": int(cfg.sim.render_interval),
        "camera_update_period_s": contract.rgbd_dt_s,
        "rgbd_timestamp_source": "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE",
    }


def _finite_xyz(value: Any, *, label: str) -> tuple[float, float, float]:
    try:
        xyz = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise Policy4DTrainingEnvContractError(
            f"G2_POLICY_{label}_NOT_XYZ"
        ) from error
    if len(xyz) != 3 or not all(math.isfinite(component) for component in xyz):
        raise Policy4DTrainingEnvContractError(f"G2_POLICY_{label}_NOT_FINITE_XYZ")
    return xyz


def _configured_names(manager_cfg: Any) -> tuple[str, ...]:
    if manager_cfg is None:
        return ()
    names = {
        config_field.name for config_field in fields(manager_cfg)
    } if is_dataclass(manager_cfg) else set()
    names.update(name for name in vars(manager_cfg) if not name.startswith("_"))
    return tuple(sorted(name for name in names if getattr(manager_cfg, name) is not None))


def _restore_training_owned_terminations(cfg: G2RedundancyTeleopEnvCfg) -> None:
    # G2RedundancyTeleopEnvCfg disables these because the keyboard recorder
    # must flush the terminal row before reset.  A training environment has no
    # such external owner, so use the already-existing task authorities.
    cfg.terminations.forbidden_collision = DoneTerm(
        func=g2_lift_task_mdp.forbidden_collision,
        params={"force_threshold_n": 1.0e-6},
    )
    cfg.terminations.precontact_camera_visibility = DoneTerm(
        func=precontact_camera_visibility_failure,
    )


def _validate_contract(cfg: G2RedundancyTeleopEnvCfg) -> dict[str, Any]:
    reward_terms = _configured_names(cfg.rewards)
    curriculum_terms = _configured_names(cfg.curriculum)
    termination_terms = _configured_names(cfg.terminations)
    if reward_terms != tuple(sorted(EXPECTED_TASK_REWARD_TERMS)):
        raise Policy4DTrainingEnvContractError(
            f"G2_POLICY_TASK_REWARD_SURFACE_MISMATCH:{reward_terms}"
        )
    if curriculum_terms != tuple(sorted(EXPECTED_TASK_CURRICULUM_TERMS)):
        raise Policy4DTrainingEnvContractError(
            f"G2_POLICY_TASK_CURRICULUM_SURFACE_MISMATCH:{curriculum_terms}"
        )
    if termination_terms != tuple(sorted(EXPECTED_TASK_TERMINATION_TERMS)):
        raise Policy4DTrainingEnvContractError(
            f"G2_POLICY_TASK_TERMINATION_SURFACE_MISMATCH:{termination_terms}"
        )
    arm_scale = tuple(float(value) for value in cfg.actions.arm_action.scale)
    if len(arm_scale) != 7:
        raise Policy4DTrainingEnvContractError(
            f"G2_POLICY_REDUNDANCY_ARM_ACTION_WIDTH_MISMATCH:{len(arm_scale)}"
        )
    expected_xyz_scale = (G2_TRANSLATION_ACTION_SCALE_M,) * 3
    if arm_scale[:3] != expected_xyz_scale:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_TRANSLATION_SCALE_NOT_0P0225_M_PER_NORMALIZED"
        )
    controller = cfg.actions.arm_action.controller
    if (
        getattr(controller, "command_type", None) != "pose"
        or getattr(controller, "use_relative_mode", None) is not True
        or getattr(controller, "ik_method", None) != "dls"
    ):
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_ROBOT_ROOT_RELATIVE_DLS_CONTRACT_MISMATCH"
        )
    if float(cfg.rewards.grasp_orientation.weight) != 0.0:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_UNCONTROLLABLE_ORIENTATION_REWARD_IS_NONZERO"
        )
    table_center_world_m = _finite_xyz(
        cfg.scene.table.init_state.pos, label="TABLE_CENTER_WORLD"
    )
    cube_center_world_m = _finite_xyz(
        cfg.scene.object.init_state.pos, label="CUBE_CENTER_WORLD"
    )
    robot_center_world_m = _finite_xyz(
        cfg.scene.robot.init_state.pos, label="ROBOT_CENTER_WORLD"
    )
    if table_center_world_m != G2_KEYBOARD_TABLE_CENTER_WORLD_M:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_TABLE_CENTER_NOT_KEYBOARD_CANONICAL"
        )
    if cube_center_world_m != G2_KEYBOARD_CUBE_CENTER_WORLD_M:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_CUBE_CENTER_NOT_KEYBOARD_CANONICAL"
        )
    if robot_center_world_m != (0.0, 0.0, 0.0):
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_ROBOT_ROOT_NOT_KEYBOARD_CANONICAL"
        )
    return {
        "schema": POLICY_4D_TRAINING_ENV_CONTRACT_SCHEMA,
        "public_policy_action_components": list(POLICY_4D_ACTION_COMPONENTS),
        "canonical_internal_action_components": list(CANONICAL_INTERNAL_ACTION_COMPONENTS),
        "internal_zero_components_owned_by_existing_p0_adapter": [
            "rx",
            "ry",
            "rz",
            "elbow",
        ],
        "translation_m_per_normalized": G2_TRANSLATION_ACTION_SCALE_M,
        "translation_scale_count": 1,
        "cartesian_command_frame": "robot_root",
        "controller_command_type": "relative_pose_dls",
        "reward_terms": list(reward_terms),
        "zero_weight_reward_terms": ["reaching_object", "grasp_orientation"],
        "curriculum_terms": list(curriculum_terms),
        "termination_terms": list(termination_terms),
        "keyboard_pose_and_task_geometry": True,
        "table_center_world_m": list(table_center_world_m),
        "cube_center_world_m": list(cube_center_world_m),
        "robot_center_world_m": list(robot_center_world_m),
        "task_geometry_runtime_binding_required_before_reset_or_step": True,
        "task_reward_target_requires_runtime_calibration_before_first_step": True,
        "alternate_asset_fallback": "NONE",
    }


def bind_g2_policy_4d_training_task_geometry_runtime(env: Any) -> dict[str, Any]:
    """Publish the scene-authored table center used by task termination.

    ``cube_left_table_persistently`` evaluates cube position in the robot-root
    frame.  The keyboard task deliberately moves its table from the shared
    sandbox Y coordinate, so using the shared fallback would classify the
    untouched reset cube as outside.  The keyboard collector already
    publishes this same authority.  Training binds it from the concrete
    environment config, verifies exact keyboard geometry, and does not alter
    the predicate, persistence interval, table extents, or any safety term.
    """

    cfg = getattr(env, "cfg", None)
    scene = getattr(cfg, "scene", None)
    if scene is None:
        raise Policy4DTrainingEnvContractError("G2_POLICY_RUNTIME_SCENE_CFG_MISSING")
    table_world = _finite_xyz(
        scene.table.init_state.pos, label="RUNTIME_TABLE_CENTER_WORLD"
    )
    cube_world = _finite_xyz(
        scene.object.init_state.pos, label="RUNTIME_CUBE_CENTER_WORLD"
    )
    robot_world = _finite_xyz(
        scene.robot.init_state.pos, label="RUNTIME_ROBOT_CENTER_WORLD"
    )
    if table_world != G2_KEYBOARD_TABLE_CENTER_WORLD_M:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_RUNTIME_TABLE_CENTER_NOT_KEYBOARD_CANONICAL"
        )
    if cube_world != G2_KEYBOARD_CUBE_CENTER_WORLD_M:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_RUNTIME_CUBE_CENTER_NOT_KEYBOARD_CANONICAL"
        )
    if robot_world != (0.0, 0.0, 0.0):
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_RUNTIME_ROBOT_ROOT_NOT_KEYBOARD_CANONICAL"
        )
    table_root = tuple(
        table_component - robot_component
        for table_component, robot_component in zip(table_world, robot_world, strict=True)
    )
    env._g2_task_table_center_root_m = table_root
    return {
        "schema": POLICY_4D_TASK_GEOMETRY_RUNTIME_BINDING_SCHEMA,
        "source": "env.cfg.scene.table.init_state.pos_minus_robot.init_state.pos",
        "table_center_world_m": list(table_world),
        "cube_center_world_m": list(cube_world),
        "robot_center_world_m": list(robot_world),
        "table_center_root_m": list(table_root),
        "keyboard_demonstration_geometry_exact": True,
        "cube_left_table_persistently_preserved": True,
    }


def attest_g2_policy_4d_training_task_geometry_runtime(env: Any) -> dict[str, Any]:
    """Read back reset cube/root geometry before the first policy step."""

    center = getattr(env, "_g2_task_table_center_root_m", None)
    if center is None:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_RUNTIME_TABLE_CENTER_NOT_BOUND"
        )
    table_root = _finite_xyz(center, label="BOUND_TABLE_CENTER_ROOT")

    def first_xyz(value: Any, *, label: str) -> tuple[float, float, float]:
        try:
            row = value.detach().to("cpu").reshape(-1, 3)[0].tolist()
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
            raise Policy4DTrainingEnvContractError(
                f"G2_POLICY_{label}_READBACK_FAILED"
            ) from error
        return _finite_xyz(row, label=label)

    robot_world = first_xyz(
        env.scene["robot"].data.root_pos_w, label="ROBOT_ROOT_WORLD_READBACK"
    )
    cube_world = first_xyz(
        env.scene["object"].data.root_pos_w, label="CUBE_WORLD_READBACK"
    )
    cube_root = tuple(
        cube_component - robot_component
        for cube_component, robot_component in zip(cube_world, robot_world, strict=True)
    )
    table_size = _finite_xyz(
        env.cfg.scene.table.spawn.size, label="TABLE_SIZE"
    )
    cube_size = _finite_xyz(
        env.cfg.scene.object.spawn.size, label="CUBE_SIZE"
    )
    half_x = table_size[0] / 2.0 - cube_size[0] / 2.0
    half_y = table_size[1] / 2.0 - cube_size[1] / 2.0
    offset_xy = (
        cube_root[0] - table_root[0],
        cube_root[1] - table_root[1],
    )
    inside = abs(offset_xy[0]) <= half_x and abs(offset_xy[1]) <= half_y
    if not inside:
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_RESET_CUBE_OUTSIDE_BOUND_TABLE"
        )
    return {
        "schema": POLICY_4D_TASK_GEOMETRY_RUNTIME_BINDING_SCHEMA,
        "robot_root_world_m": list(robot_world),
        "cube_center_world_m": list(cube_world),
        "cube_center_root_m": list(cube_root),
        "table_center_root_m": list(table_root),
        "cube_minus_table_xy_m": list(offset_xy),
        "allowed_center_half_extent_xy_m": [half_x, half_y],
        "cube_inside_table": True,
    }


def make_g2_policy_4d_training_env_cfg(
    *, num_envs: int | None = None
) -> tuple[G2RedundancyTeleopEnvCfg, M2QualifiedAssetBindingReceipt, dict[str, Any]]:
    """Build, bind, and statically validate the future training config.

    No environment is instantiated.  Callers must retain the receipt in the
    run manifest and must configure the source-owned grasp reward target after
    environment creation but before the first step.
    """

    if num_envs is not None and (isinstance(num_envs, bool) or num_envs <= 0):
        raise Policy4DTrainingEnvContractError("G2_POLICY_NUM_ENVS_MUST_BE_POSITIVE")
    cfg = G2RedundancyTeleopEnvCfg()
    _restore_training_owned_terminations(cfg)
    # The 4-D policy has no orientation action.  This follows the existing
    # G2LiftVisualSACEnvCfg authority: orientation remains in telemetry/safety,
    # but an uncontrollable reward is not optimized.
    cfg.rewards.grasp_orientation.weight = 0.0
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    receipt = bind_m2_qualified_candidate(cfg)
    contract = _validate_contract(cfg)
    return cfg, receipt, contract


def make_g2_contact_free_candidate_a_training_env_cfg(
    *, num_envs: int | None = None
) -> tuple[
    G2RedundancyTeleopEnvCfg,
    ContactFreeCandidateABindingReceipt,
    dict[str, Any],
]:
    """Build the exact Candidate-A OPEN-only pre-contact training config.

    This dedicated entry point prevents a caller from first receiving an E1
    qualification receipt and then replacing its USD path with Candidate A.
    The selected asset is bound before environment construction and the task
    contract publishes the same identity.  It grants no CLOSE, contact,
    contact-training, hardware, Production-promotion, or M2-contact authority.
    """

    if num_envs is not None and (isinstance(num_envs, bool) or num_envs <= 0):
        raise Policy4DTrainingEnvContractError("G2_POLICY_NUM_ENVS_MUST_BE_POSITIVE")
    cfg = G2RedundancyTeleopEnvCfg()
    _restore_training_owned_terminations(cfg)
    cfg.rewards.grasp_orientation.weight = 0.0
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    time_contract = _bind_keyboard_v3_time_contract(cfg)
    receipt = bind_contact_free_candidate_a(cfg)
    contract = {
        **_validate_contract(cfg),
        "selected_asset_binding": {
            "schema": receipt.schema,
            "binding_id": receipt.binding_id,
            "classification": receipt.classification,
            "scope": receipt.scope,
            "candidate_asset_path": receipt.candidate_asset_path,
            "candidate_asset_sha256": receipt.candidate_asset_sha256,
            "dependency_manifest_sha256": receipt.dependency_manifest_sha256,
            "m2_contact_verdict": receipt.m2_contact_verdict,
            "close_authorized": receipt.close_authorized,
            "contact_authorized": receipt.contact_authorized,
            "contact_training_authorized": receipt.contact_training_authorized,
            "hardware_authority_claim": receipt.hardware_authority_claim,
            "production_promotion_claim": receipt.production_promotion_claim,
            "alternate_asset_fallback": receipt.alternate_asset_fallback,
        },
        "time_contract": time_contract,
    }
    selected = contract["selected_asset_binding"]
    if (
        selected["candidate_asset_sha256"] != receipt.candidate_asset_sha256
        or selected["candidate_asset_path"] != receipt.candidate_asset_path
        or any(
            selected[name] is not False
            for name in (
                "close_authorized",
                "contact_authorized",
                "contact_training_authorized",
                "hardware_authority_claim",
                "production_promotion_claim",
            )
        )
        or selected["m2_contact_verdict"] != "FAIL_CONTACT_MECHANICS"
        or selected["alternate_asset_fallback"] != "NONE"
    ):
        raise Policy4DTrainingEnvContractError(
            "G2_POLICY_CONTACT_FREE_CANDIDATE_A_AUTHORITY_MISMATCH"
        )
    return cfg, receipt, contract


def make_g2_candidate_a_left_arm_down_v2_training_env_cfg(
    *, num_envs: int | None = None
) -> tuple[
    G2RedundancyTeleopEnvCfg,
    ContactFreeCandidateABindingReceipt,
    dict[str, Any],
]:
    """Build the new common Candidate-A V2 reset baseline.

    The old Candidate-A factory remains unchanged for historical replay.  V2
    changes only the seven left-arm reset coordinates after the exact legacy
    Candidate-A asset/time/task contracts have been bound.
    """

    cfg, receipt, contract = make_g2_contact_free_candidate_a_training_env_cfg(
        num_envs=num_envs
    )
    baseline = apply_left_arm_down_v2_to_cfg(cfg)
    return cfg, receipt, {**contract, "reset_baseline": baseline}


__all__ = [
    "CANONICAL_INTERNAL_ACTION_COMPONENTS",
    "EXPECTED_TASK_CURRICULUM_TERMS",
    "EXPECTED_TASK_REWARD_TERMS",
    "EXPECTED_TASK_TERMINATION_TERMS",
    "POLICY_4D_ACTION_COMPONENTS",
    "POLICY_4D_TASK_GEOMETRY_RUNTIME_BINDING_SCHEMA",
    "POLICY_4D_TRAINING_ENV_CONTRACT_SCHEMA",
    "Policy4DTrainingEnvContractError",
    "attest_g2_policy_4d_training_task_geometry_runtime",
    "bind_g2_policy_4d_training_task_geometry_runtime",
    "make_g2_contact_free_candidate_a_training_env_cfg",
    "make_g2_candidate_a_left_arm_down_v2_training_env_cfg",
    "make_g2_policy_4d_training_env_cfg",
]
