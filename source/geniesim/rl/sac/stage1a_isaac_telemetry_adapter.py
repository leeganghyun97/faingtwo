# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed Isaac contact telemetry adapter for the Stage-1A reward.

This module is deliberately pure: it does not import or start Isaac, step an
environment, or mutate a contact sensor.  The live runner supplies ten records
for a normal 50-Hz action or exactly the real prefix ``1..k`` ending at a
structured 500-Hz RuntimeHardstop sample.

The physical channels are the right OmniPicker's ``inner`` and ``outer`` pads.
Stage-1A's historical ``[left, right]`` names are explicit aliases here:
``left := right_inner`` and ``right := right_outer``.  They are not the robot's
left and right hands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..isaaclab.g2_rebuild.sensor_packet import (
    world_points_to_root,
    world_point_velocity_to_root,
    world_quaternions_to_root,
    world_vectors_to_root,
)
from .stage1a_grasp_reward import (
    PHYSICS_DT_S,
    PHYSICS_SUBSTEPS_PER_CONTROL,
    Stage1AGraspRewardError,
    Stage1ARewardInputs,
)


STAGE1A_ISAAC_TELEMETRY_ADAPTER_SCHEMA = (
    "g2_stage1a_isaac_telemetry_adapter_v1"
)
STAGE1A_SIDE_CHANNELS = ("right_inner", "right_outer")
STAGE1A_LEGACY_SIDE_ALIASES = ("left", "right")
EXPECTED_OBJECT_PARTNER_SUFFIX = "/Object"


class Stage1AIsaacTelemetryAdapterError(Stage1AGraspRewardError):
    """Raised before incomplete or ambiguous runtime evidence enters replay."""


@dataclass(frozen=True)
class Stage1ARuntimeTransitionState:
    """Non-contact state paired with one accepted 50-Hz/partial transition.

    Root states are provided for every physics substep because transforming a
    contact point with only the final root pose is incorrect for a moving base.
    Framework-native quaternions must be converted to XYZW by the caller.
    """

    current_ee_position_root_m: Any
    next_ee_position_root_m: Any
    nominal_grasp_position_root_m: Any
    nominal_approach_axis_root: Any
    stage1a_active: bool
    grasp_decision_phase: bool
    phase_reset: bool
    episode_reset: bool
    root_position_world_m_by_substep: Any
    root_quat_world_xyzw_by_substep: Any
    root_linear_velocity_world_m_s_by_substep: Any
    root_angular_velocity_world_rad_s_by_substep: Any
    cube_center_world_m: Any
    cube_quat_world_xyzw: Any
    cube_linear_velocity_world_m_s: Any
    cube_angular_velocity_world_rad_s: Any
    cube_half_extents_m: Any
    effective_residual_root_m: Any
    previous_effective_residual_root_m: Any
    final_action_xyz_root_m: Any
    contract_grasp_state_valid: bool
    safety_violation: bool
    authoritative_safety_penalty: float
    non_pad_gripper_cube_contact: bool
    close_command: bool
    consumed_substeps: int = PHYSICS_SUBSTEPS_PER_CONTROL
    runtime_hardstop: bool = False


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise Stage1AIsaacTelemetryAdapterError(
            f"{name} must be finite with shape {shape}"
        )
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise Stage1AIsaacTelemetryAdapterError(f"{name} must be bool")
    return bool(value)


def _raw_side(record: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    key = "raw_inner" if side == "right_inner" else "raw_outer"
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise Stage1AIsaacTelemetryAdapterError(f"missing {key} telemetry")
    return value


def _contact_count(raw: Mapping[str, Any], label: str) -> int:
    value = raw.get("count")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Stage1AIsaacTelemetryAdapterError(f"{label}.count must be integer")
    count = int(value)
    if not 0 <= count <= 32:
        raise Stage1AIsaacTelemetryAdapterError(f"{label}.count outside [0,32]")
    return count


def _contact_field(
    raw: Mapping[str, Any], name: str, count: int, width: int | None = None
) -> np.ndarray:
    value = raw.get(name)
    if value is None:
        raise Stage1AIsaacTelemetryAdapterError(f"missing contact field {name}")
    array = np.asarray(value, dtype=np.float64)
    expected_tail = () if width is None else (width,)
    if array.ndim != 1 + len(expected_tail) or array.shape[1:] != expected_tail:
        raise Stage1AIsaacTelemetryAdapterError(
            f"contact field {name} has invalid shape {array.shape}"
        )
    selected = array[:count]
    if not np.isfinite(selected).all():
        raise Stage1AIsaacTelemetryAdapterError(
            f"contact field {name} contains NaN/Inf in active rows"
        )
    return selected


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    consumed_substeps: int,
    runtime_hardstop: bool,
) -> int:
    if type(consumed_substeps) is not int or not 1 <= consumed_substeps <= PHYSICS_SUBSTEPS_PER_CONTROL:
        raise Stage1AIsaacTelemetryAdapterError(
            "consumed_substeps must be an integer in [1,10]"
        )
    if len(records) != consumed_substeps:
        raise Stage1AIsaacTelemetryAdapterError(
            "physics record count differs from consumed_substeps; normal transition requires exactly 10"
        )
    if not runtime_hardstop and consumed_substeps != PHYSICS_SUBSTEPS_PER_CONTROL:
        raise Stage1AIsaacTelemetryAdapterError(
            "partial physics records require runtime_hardstop"
        )
    expected_substeps = list(range(consumed_substeps))
    observed_substeps = [record.get("physics_substep") for record in records]
    if observed_substeps != expected_substeps:
        raise Stage1AIsaacTelemetryAdapterError(
            "physics substeps must be consecutive "
            f"0..{consumed_substeps - 1}, got {observed_substeps}"
        )
    policy_steps = {record.get("policy_step") for record in records}
    if len(policy_steps) != 1 or next(iter(policy_steps)) is None:
        raise Stage1AIsaacTelemetryAdapterError(
            "physics records do not belong to one policy packet"
        )
    samples = [record.get("global_physics_sample") for record in records]
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in samples):
        raise Stage1AIsaacTelemetryAdapterError("global physics sample identity missing")
    if [int(value) for value in samples] != list(
        range(int(samples[0]), int(samples[0]) + consumed_substeps)
    ):
        raise Stage1AIsaacTelemetryAdapterError("physics samples are not consecutive")
    for record in records:
        dt_s = record.get("dt_s")
        if not isinstance(dt_s, (int, float)) or not math.isclose(
            float(dt_s), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise Stage1AIsaacTelemetryAdapterError("physics dt is not 0.002 s")
    return consumed_substeps


def _validate_partner_paths(paths: Sequence[str]) -> None:
    if len(paths) != 2 or any(
        not isinstance(path, str) or not path.endswith(EXPECTED_OBJECT_PARTNER_SUFFIX)
        for path in paths
    ):
        raise Stage1AIsaacTelemetryAdapterError(
            "pad contact sensors must be explicitly filtered to Object"
        )


def build_stage1a_reward_inputs(
    records: Sequence[Mapping[str, Any]],
    state: Stage1ARuntimeTransitionState,
    *,
    contact_partner_actor_paths: Sequence[str],
    device: torch.device | str = "cpu",
) -> Stage1ARewardInputs:
    """Convert one real full or hard-stop-prefix interval into reward tensors.

    The strongest normal contact point is selected independently for each
    pad/substep.  Native normal impulse is used only when it is available for
    every active contact on a side across the complete interval.  Otherwise
    that entire side uses the force fallback; native and fallback authority are
    never mixed within a side/interval.
    """

    if not isinstance(state, Stage1ARuntimeTransitionState):
        raise Stage1AIsaacTelemetryAdapterError("runtime transition state is required")
    runtime_hardstop = _strict_bool(state.runtime_hardstop, "runtime_hardstop")
    s = _validate_records(
        records,
        consumed_substeps=state.consumed_substeps,
        runtime_hardstop=runtime_hardstop,
    )
    _validate_partner_paths(contact_partner_actor_paths)
    root_position = _finite_array(
        state.root_position_world_m_by_substep, (s, 3), "root position by substep"
    )
    root_quaternion = _finite_array(
        state.root_quat_world_xyzw_by_substep, (s, 4), "root quaternion by substep"
    )
    root_linear_velocity = _finite_array(
        state.root_linear_velocity_world_m_s_by_substep,
        (s, 3),
        "root linear velocity by substep",
    )
    root_angular_velocity = _finite_array(
        state.root_angular_velocity_world_rad_s_by_substep,
        (s, 3),
        "root angular velocity by substep",
    )
    quaternion_norm = np.linalg.norm(root_quaternion, axis=1)
    if np.any(np.abs(quaternion_norm - 1.0) > 1.0e-3):
        raise Stage1AIsaacTelemetryAdapterError("root quaternion is not unit XYZW")

    contact_valid = np.zeros((2, s), dtype=np.bool_)
    geometry_valid = np.zeros_like(contact_valid)
    normal_impulse = np.zeros((2, s), dtype=np.float64)
    normal_force = np.zeros((2, s), dtype=np.float64)
    points_world = np.zeros((2, s, 3), dtype=np.float64)
    normals_world = np.zeros_like(points_world)
    tangential_speed = np.zeros((2, s), dtype=np.float64)
    normal_speed = np.zeros((2, s), dtype=np.float64)
    native_complete = np.ones(2, dtype=np.bool_)

    # Decide interval authority before selecting any point.  If even one active
    # substep lacks native impulse, point selection and integration both use
    # force for the complete side/interval.
    for side_index, side in enumerate(STAGE1A_SIDE_CHANNELS):
        for substep, record in enumerate(records):
            raw = _raw_side(record, side)
            count = _contact_count(raw, f"{side}[{substep}]")
            if count == 0:
                continue
            native_value = raw.get("native_normal_impulse_ns")
            if native_value is None:
                native_complete[side_index] = False
                break
            candidate = np.asarray(native_value, dtype=np.float64)
            if (
                candidate.ndim != 1
                or candidate.shape[0] < count
                or not np.isfinite(candidate[:count]).all()
                or np.any(candidate[:count] < 0.0)
            ):
                native_complete[side_index] = False
                break

    for side_index, side in enumerate(STAGE1A_SIDE_CHANNELS):
        for substep, record in enumerate(records):
            raw = _raw_side(record, side)
            count = _contact_count(raw, f"{side}[{substep}]")
            if count == 0:
                continue
            force = _contact_field(raw, "normal_force_n", count)
            if np.any(force < 0.0):
                raise Stage1AIsaacTelemetryAdapterError("normal force cannot be negative")
            native: np.ndarray | None = None
            native_value = raw.get("native_normal_impulse_ns")
            if native_complete[side_index] and native_value is not None:
                candidate = np.asarray(native_value, dtype=np.float64)
                if candidate.ndim == 1 and candidate.shape[0] >= count:
                    candidate = candidate[:count]
                    if np.isfinite(candidate).all() and np.all(candidate >= 0.0):
                        native = candidate
            strength = force if native is None else native
            strongest = int(np.argmax(strength))
            point = _contact_field(raw, "point_world_m", count, 3)[strongest]
            normal = _contact_field(raw, "normal_world", count, 3)[strongest]
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-8:
                raise Stage1AIsaacTelemetryAdapterError("contact normal has zero norm")
            slip = _contact_field(raw, "slip_speed_m_s", count)[strongest]
            relative_normal = _contact_field(
                raw, "relative_normal_velocity_m_s", count
            )[strongest]
            if slip < 0.0:
                raise Stage1AIsaacTelemetryAdapterError("slip speed cannot be negative")
            contact_valid[side_index, substep] = True
            geometry_valid[side_index, substep] = True
            normal_force[side_index, substep] = force[strongest]
            normal_impulse[side_index, substep] = (
                0.0 if native is None else native[strongest]
            )
            points_world[side_index, substep] = point
            normals_world[side_index, substep] = normal / normal_norm
            tangential_speed[side_index, substep] = slip
            normal_speed[side_index, substep] = abs(float(relative_normal))

    # An interval-wide force fallback must contain valid force for every active
    # contact.  Native values are discarded for that whole side to prevent
    # mixed authority inside the reward's integration interval.
    for side_index in range(2):
        if not native_complete[side_index]:
            normal_impulse[side_index] = 0.0

    torch_device = torch.device(device)
    root_position_t = torch.as_tensor(root_position, dtype=torch.float32, device=torch_device)
    root_quaternion_t = torch.as_tensor(root_quaternion, dtype=torch.float32, device=torch_device)
    points_flat = torch.as_tensor(
        points_world.transpose(1, 0, 2).reshape(s * 2, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    normals_flat = torch.as_tensor(
        normals_world.transpose(1, 0, 2).reshape(s * 2, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    repeated_root_position = root_position_t.repeat_interleave(2, dim=0)
    repeated_root_quaternion = root_quaternion_t.repeat_interleave(2, dim=0)
    points_root = world_points_to_root(
        points_flat, repeated_root_position, repeated_root_quaternion
    ).reshape(s, 2, 3).transpose(0, 1)
    normals_root = world_vectors_to_root(
        normals_flat, repeated_root_quaternion
    ).reshape(s, 2, 3).transpose(0, 1)
    valid_t = torch.as_tensor(contact_valid, dtype=torch.bool, device=torch_device)
    points_root = torch.where(valid_t[..., None], points_root, torch.zeros_like(points_root))
    normals_root = torch.where(valid_t[..., None], normals_root, torch.zeros_like(normals_root))

    final_root_position = root_position_t[-1:]
    final_root_quaternion = root_quaternion_t[-1:]
    final_root_linear_velocity = torch.as_tensor(
        root_linear_velocity[-1:], dtype=torch.float32, device=torch_device
    )
    final_root_angular_velocity = torch.as_tensor(
        root_angular_velocity[-1:], dtype=torch.float32, device=torch_device
    )
    cube_center_world = torch.as_tensor(
        _finite_array(state.cube_center_world_m, (3,), "cube center world"),
        dtype=torch.float32,
        device=torch_device,
    )[None]
    cube_quaternion_world = torch.as_tensor(
        _finite_array(state.cube_quat_world_xyzw, (4,), "cube quaternion world"),
        dtype=torch.float32,
        device=torch_device,
    )[None]
    cube_linear_world = torch.as_tensor(
        _finite_array(
            state.cube_linear_velocity_world_m_s, (3,), "cube linear velocity world"
        ),
        dtype=torch.float32,
        device=torch_device,
    )[None]
    cube_angular_world = torch.as_tensor(
        _finite_array(
            state.cube_angular_velocity_world_rad_s,
            (3,),
            "cube angular velocity world",
        ),
        dtype=torch.float32,
        device=torch_device,
    )[None]
    cube_center_root = world_points_to_root(
        cube_center_world, final_root_position, final_root_quaternion
    )
    cube_quaternion_root = world_quaternions_to_root(
        cube_quaternion_world, final_root_quaternion
    )
    cube_linear_root = world_point_velocity_to_root(
        cube_center_world,
        cube_linear_world,
        final_root_position,
        final_root_quaternion,
        final_root_linear_velocity,
        final_root_angular_velocity,
    )
    cube_angular_root = world_vectors_to_root(
        cube_angular_world - final_root_angular_velocity,
        final_root_quaternion,
    )

    def vector(name: str, value: Any) -> torch.Tensor:
        return torch.as_tensor(
            _finite_array(value, (3,), name), dtype=torch.float32, device=torch_device
        )[None]

    safety_violation = _strict_bool(state.safety_violation, "safety_violation")
    safety_penalty = float(state.authoritative_safety_penalty)
    if not math.isfinite(safety_penalty) or safety_penalty > 0.0:
        raise Stage1AIsaacTelemetryAdapterError(
            "authoritative safety penalty must be finite and nonpositive"
        )
    if not safety_violation and safety_penalty != 0.0:
        raise Stage1AIsaacTelemetryAdapterError(
            "safety penalty without safety authority violation"
        )

    return Stage1ARewardInputs(
        current_ee_position_root_m=vector(
            "current EE root", state.current_ee_position_root_m
        ),
        next_ee_position_root_m=vector("next EE root", state.next_ee_position_root_m),
        nominal_grasp_position_root_m=vector(
            "nominal grasp root", state.nominal_grasp_position_root_m
        ),
        nominal_approach_axis_root=vector(
            "nominal approach axis root", state.nominal_approach_axis_root
        ),
        stage1a_active=torch.tensor(
            [_strict_bool(state.stage1a_active, "stage1a_active")],
            dtype=torch.bool,
            device=torch_device,
        ),
        grasp_decision_phase=torch.tensor(
            [_strict_bool(state.grasp_decision_phase, "grasp_decision_phase")],
            dtype=torch.bool,
            device=torch_device,
        ),
        phase_reset=torch.tensor(
            [_strict_bool(state.phase_reset, "phase_reset")],
            dtype=torch.bool,
            device=torch_device,
        ),
        episode_reset=torch.tensor(
            [_strict_bool(state.episode_reset, "episode_reset")],
            dtype=torch.bool,
            device=torch_device,
        ),
        pad_cube_contact_valid=valid_t[None],
        normal_impulse_ns=torch.as_tensor(
            normal_impulse, dtype=torch.float32, device=torch_device
        )[None],
        normal_impulse_available=torch.as_tensor(
            native_complete, dtype=torch.bool, device=torch_device
        )[None],
        normal_force_n=torch.as_tensor(
            normal_force, dtype=torch.float32, device=torch_device
        )[None],
        contact_geometry_valid=torch.as_tensor(
            geometry_valid, dtype=torch.bool, device=torch_device
        )[None],
        contact_point_root_m=points_root[None],
        contact_normal_root=normals_root[None],
        tangential_relative_speed_m_s=torch.as_tensor(
            tangential_speed, dtype=torch.float32, device=torch_device
        )[None],
        normal_relative_speed_m_s=torch.as_tensor(
            normal_speed, dtype=torch.float32, device=torch_device
        )[None],
        cube_center_root_m=cube_center_root,
        cube_quat_root_xyzw=cube_quaternion_root,
        cube_half_extents_m=vector("cube half extents", state.cube_half_extents_m),
        cube_linear_velocity_root_m_s=cube_linear_root,
        cube_angular_velocity_root_rad_s=cube_angular_root,
        effective_residual_root_m=vector(
            "effective residual root", state.effective_residual_root_m
        ),
        previous_effective_residual_root_m=vector(
            "previous effective residual root",
            state.previous_effective_residual_root_m,
        ),
        final_action_xyz_root_m=vector(
            "final action XYZ root", state.final_action_xyz_root_m
        ),
        contract_grasp_state_valid=torch.tensor(
            [
                _strict_bool(
                    state.contract_grasp_state_valid,
                    "contract_grasp_state_valid",
                )
            ],
            dtype=torch.bool,
            device=torch_device,
        ),
        safety_violation=torch.tensor(
            [safety_violation], dtype=torch.bool, device=torch_device
        ),
        authoritative_safety_penalty=torch.tensor(
            [safety_penalty], dtype=torch.float32, device=torch_device
        ),
        non_pad_gripper_cube_contact=torch.tensor(
            [
                _strict_bool(
                    state.non_pad_gripper_cube_contact,
                    "non_pad_gripper_cube_contact",
                )
            ],
            dtype=torch.bool,
            device=torch_device,
        ),
        close_command=torch.tensor(
            [_strict_bool(state.close_command, "close_command")],
            dtype=torch.bool,
            device=torch_device,
        ),
        consumed_substeps=torch.tensor(
            [s], dtype=torch.int64, device=torch_device
        ),
        runtime_hardstop=torch.tensor(
            [runtime_hardstop], dtype=torch.bool, device=torch_device
        ),
    )


__all__ = [
    "STAGE1A_ISAAC_TELEMETRY_ADAPTER_SCHEMA",
    "STAGE1A_LEGACY_SIDE_ALIASES",
    "STAGE1A_SIDE_CHANNELS",
    "Stage1AIsaacTelemetryAdapterError",
    "Stage1ARuntimeTransitionState",
    "build_stage1a_reward_inputs",
]
