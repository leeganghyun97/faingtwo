# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from geniesim.rl.sac.stage1a_grasp_reward import Stage1AGraspReward
from geniesim.rl.sac.stage1a_isaac_telemetry_adapter import (
    STAGE1A_LEGACY_SIDE_ALIASES,
    STAGE1A_SIDE_CHANNELS,
    Stage1AIsaacTelemetryAdapterError,
    Stage1ARuntimeTransitionState,
    build_stage1a_reward_inputs,
)


OBJECT_PATHS = ("/World/envs/env_0/Object", "/World/envs/env_0/Object")


def _raw(*, force=(0.0,), impulse=(0.0,), points=None, normals=None, native=True):
    count = len(force)
    points = points if points is not None else [(1.0, 3.0, 0.0)] * count
    normals = normals if normals is not None else [(0.0, 1.0, 0.0)] * count
    result = {
        "count": count,
        "normal_force_n": np.asarray(force, dtype=np.float64),
        "point_world_m": np.asarray(points, dtype=np.float64).reshape(count, 3),
        "normal_world": np.asarray(normals, dtype=np.float64).reshape(count, 3),
        "slip_speed_m_s": np.full(count, 0.004, dtype=np.float64),
        "relative_normal_velocity_m_s": np.full(count, -0.03, dtype=np.float64),
    }
    if native:
        result["native_normal_impulse_ns"] = np.asarray(impulse, dtype=np.float64)
    return result


def _records():
    rows = []
    for index in range(10):
        if index < 3:
            # The second contact is strongest and must own point/normal evidence.
            inner = _raw(
                force=(2.0, 5.0),
                impulse=(0.004, 0.010),
                points=((1.0, 2.5, 0.0), (1.0, 3.0, 0.0)),
                normals=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            )
            outer = _raw(
                force=(4.0,),
                impulse=(0.008,),
                points=((1.0, 1.0, 0.0),),
                normals=((0.0, -1.0, 0.0),),
            )
        else:
            inner = _raw(force=(), impulse=(), points=(), normals=())
            outer = _raw(force=(), impulse=(), points=(), normals=())
        rows.append(
            {
                "policy_step": 7,
                "global_physics_sample": 100 + index,
                "physics_substep": index,
                "dt_s": 0.002,
                "raw_inner": inner,
                "raw_outer": outer,
            }
        )
    return rows


def _state():
    half = math.sqrt(0.5)
    root_quaternion = np.tile([0.0, 0.0, half, half], (10, 1))
    return Stage1ARuntimeTransitionState(
        current_ee_position_root_m=(-0.020, 0.0, 0.0),
        next_ee_position_root_m=(-0.019, 0.0, 0.0),
        nominal_grasp_position_root_m=(0.0, 0.0, 0.0),
        nominal_approach_axis_root=(1.0, 0.0, 0.0),
        stage1a_active=True,
        grasp_decision_phase=True,
        phase_reset=False,
        episode_reset=False,
        root_position_world_m_by_substep=np.tile([1.0, 2.0, 0.0], (10, 1)),
        root_quat_world_xyzw_by_substep=root_quaternion,
        root_linear_velocity_world_m_s_by_substep=np.zeros((10, 3)),
        root_angular_velocity_world_rad_s_by_substep=np.zeros((10, 3)),
        cube_center_world_m=(1.0, 2.0, 0.0),
        cube_quat_world_xyzw=(0.0, 0.0, half, half),
        cube_linear_velocity_world_m_s=(0.0, 1.0, 0.0),
        cube_angular_velocity_world_rad_s=(0.0, 0.0, 0.2),
        cube_half_extents_m=(0.02, 0.02, 0.02),
        effective_residual_root_m=(0.0001, 0.0, 0.0),
        previous_effective_residual_root_m=(0.0, 0.0, 0.0),
        final_action_xyz_root_m=(0.001, 0.0, 0.0),
        contract_grasp_state_valid=False,
        safety_violation=False,
        authoritative_safety_penalty=0.0,
        non_pad_gripper_cube_contact=False,
        close_command=True,
    )


def test_real_substep_adapter_maps_inner_outer_and_full_root_transform():
    assert STAGE1A_SIDE_CHANNELS == ("right_inner", "right_outer")
    assert STAGE1A_LEGACY_SIDE_ALIASES == ("left", "right")
    inputs = build_stage1a_reward_inputs(
        _records(), _state(), contact_partner_actor_paths=OBJECT_PATHS
    )

    assert inputs.pad_cube_contact_valid.shape == (1, 2, 10)
    assert inputs.pad_cube_contact_valid[0, 0, :3].all()
    assert inputs.normal_impulse_available.tolist() == [[True, True]]
    assert inputs.normal_impulse_ns[0, 0, 0].item() == pytest.approx(0.010)
    assert inputs.normal_force_n[0, 0, 0].item() == pytest.approx(5.0)
    # root yaw is +90 degrees: world +Y becomes root +X.
    assert inputs.contact_point_root_m[0, 0, 0].tolist() == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-6
    )
    assert inputs.contact_normal_root[0, 0, 0].tolist() == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-6
    )
    assert inputs.normal_relative_speed_m_s[0, 0, 0].item() == pytest.approx(0.03)
    assert inputs.cube_center_root_m[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert inputs.cube_quat_root_xyzw[0].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0], abs=1.0e-6
    )
    assert inputs.cube_linear_velocity_root_m_s[0].tolist() == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-6
    )
    result = Stage1AGraspReward(1).step(inputs)
    assert result.left_contact_boolean.item()
    assert result.right_contact_boolean.item()


@pytest.mark.parametrize("consumed_substeps", (1, 3, 5, 10))
def test_runtime_hardstop_prefix_accepts_exact_consecutive_real_samples(
    consumed_substeps: int,
):
    base = _state()
    partial = replace(
        base,
        root_position_world_m_by_substep=np.asarray(
            base.root_position_world_m_by_substep
        )[:consumed_substeps],
        root_quat_world_xyzw_by_substep=np.asarray(
            base.root_quat_world_xyzw_by_substep
        )[:consumed_substeps],
        root_linear_velocity_world_m_s_by_substep=np.asarray(
            base.root_linear_velocity_world_m_s_by_substep
        )[:consumed_substeps],
        root_angular_velocity_world_rad_s_by_substep=np.asarray(
            base.root_angular_velocity_world_rad_s_by_substep
        )[:consumed_substeps],
        safety_violation=True,
        consumed_substeps=consumed_substeps,
        runtime_hardstop=True,
    )
    inputs = build_stage1a_reward_inputs(
        _records()[:consumed_substeps],
        partial,
        contact_partner_actor_paths=OBJECT_PATHS,
    )
    assert inputs.pad_cube_contact_valid.shape == (1, 2, consumed_substeps)
    assert inputs.consumed_substeps.tolist() == [consumed_substeps]
    assert inputs.runtime_hardstop.tolist() == [True]


def test_interval_does_not_mix_native_impulse_and_force_fallback():
    rows = _records()
    rows[0]["raw_inner"] = _raw(
        force=(10.0, 5.0),
        impulse=(0.004, 0.010),
        points=((1.0, 2.5, 0.0), (1.0, 3.0, 0.0)),
        normals=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    )
    rows[1]["raw_inner"] = _raw(force=(5.0,), native=False)
    inputs = build_stage1a_reward_inputs(
        rows, _state(), contact_partner_actor_paths=OBJECT_PATHS
    )
    assert inputs.normal_impulse_available.tolist() == [[False, True]]
    assert inputs.normal_impulse_ns[0, 0].eq(0.0).all()
    assert inputs.normal_force_n[0, 0, 0].item() == pytest.approx(10.0)
    assert inputs.contact_point_root_m[0, 0, 0].tolist() == pytest.approx(
        [0.5, 0.0, 0.0], abs=1.0e-6
    )
    assert inputs.normal_force_n[0, 0, 1].item() == pytest.approx(5.0)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda rows: rows.pop(), "exactly 10"),
        (
            lambda rows: rows[5].__setitem__("physics_substep", 6),
            "consecutive 0..9",
        ),
        (
            lambda rows: rows[5].__setitem__("global_physics_sample", 999),
            "not consecutive",
        ),
        (lambda rows: rows[5].__setitem__("dt_s", 0.004), "not 0.002"),
    ],
)
def test_substep_contract_fails_closed(mutator, match):
    rows = _records()
    mutator(rows)
    with pytest.raises(Stage1AIsaacTelemetryAdapterError, match=match):
        build_stage1a_reward_inputs(
            rows, _state(), contact_partner_actor_paths=OBJECT_PATHS
        )


def test_wrong_partner_and_active_nan_fail_closed():
    with pytest.raises(Stage1AIsaacTelemetryAdapterError, match="filtered to Object"):
        build_stage1a_reward_inputs(
            _records(),
            _state(),
            contact_partner_actor_paths=("/World/Table", "/World/Object"),
        )
    rows = _records()
    rows[0]["raw_inner"]["point_world_m"][0, 0] = np.nan
    with pytest.raises(Stage1AIsaacTelemetryAdapterError, match="NaN/Inf"):
        build_stage1a_reward_inputs(
            rows, _state(), contact_partner_actor_paths=OBJECT_PATHS
        )


def test_moving_root_velocity_and_safety_authority_are_not_silently_ignored():
    state = _state()
    root_linear = np.tile([0.0, 1.0, 0.0], (10, 1))
    moving = replace(
        state,
        root_linear_velocity_world_m_s_by_substep=root_linear,
        authoritative_safety_penalty=-1.0,
    )
    with pytest.raises(
        Stage1AIsaacTelemetryAdapterError,
        match="without safety authority violation",
    ):
        build_stage1a_reward_inputs(
            _records(), moving, contact_partner_actor_paths=OBJECT_PATHS
        )
    safe = replace(moving, authoritative_safety_penalty=0.0)
    inputs = build_stage1a_reward_inputs(
        _records(), safe, contact_partner_actor_paths=OBJECT_PATHS
    )
    assert inputs.cube_linear_velocity_root_m_s[0].tolist() == pytest.approx(
        [0.0, 0.0, 0.0], abs=1.0e-6
    )
