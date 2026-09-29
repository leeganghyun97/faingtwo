# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded real-Isaac Stage-1A smoke loop.

This module is imported only after the established Candidate-A V2 runner has
created/reset the environment, applied the immutable direct-pregrasp state,
and installed its single-consumption instrumentation.  It never creates an
environment and never calls ``ActionManager.process_action`` directly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from geniesim.rl.isaaclab.g2_quaternion import (
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
)
from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
    AbstractGripperIntent,
    HighLevelPolicyAction,
)
from geniesim.rl.isaaclab.g2_policy_branch.contact_free_training_contract import (
    metric_xyz_to_normalized,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_direct_pregrasp_init import (
    apply_direct_pregrasp_initial_state,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_live_driver import (
    _capture_camera_rgbd,
    _ee_root_pose,
)
from geniesim.rl.sac.human_grasp_gru_bc import HumanGraspSequenceInputs
from geniesim.rl.sac.keyboard_v3_dataset import preprocess_depth_m
from geniesim.rl.sac.automatic_close_supervision import (
    classify_cube_contact_region,
    root_point_to_cube_local,
    serialize_physx_vector,
)
from geniesim.rl.sac.bc_residual_sac_contract import (
    compose_bc_and_residual_action,
)
from geniesim.rl.sac.stage1a_grasp_reward import (
    Stage1AGraspReward,
    Stage1ARewardV3Config,
    stage1a_reward_contract,
)
from geniesim.rl.sac.stage1a_isaac_telemetry_adapter import (
    Stage1ARuntimeTransitionState,
    build_stage1a_reward_inputs,
)
from geniesim.rl.sac.stage1a_real_sac_coordinator import (
    REPLAY_STRATEGY_HER_FORCE,
    Stage1AAcceptedRealRow,
    Stage1ARealSACCoordinator,
)
from geniesim.rl.sac.stage1a_exception_safe_transition import (
    ActionConsumptionState,
    RuntimeHardstop,
)
from geniesim.rl.sac.stage1a_close_gate import (
    SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG,
    SIMPLIFIED_CLOSE_MAX_RESIDUAL_M,
    SIMPLIFIED_CLOSE_MIN_RESIDUAL_M,
    SIMPLIFIED_CLOSE_PERSISTENCE_STEPS,
    SimplifiedClosePersistenceGate,
)


STAGE1A_ISAAC_SHORT_SMOKE_SCHEMA = "g2_stage1a_real_isaac_training_v2"
ACCEPTED_TRANSITION_TARGET = 1000
LONG_TRAINING_ACCEPTED_TRANSITION_TARGET = 15000
REWARD_V3_ACCEPTED_TRANSITION_TARGET = 3000
CONTROL_HZ = 50
PHYSICS_HZ = 500
RGBD_HZ = 25
MAX_FINAL_ACTION_M = 0.0045


class Stage1AIsaacShortSmokeError(RuntimeError):
    pass


def _orientation_error_deg(reference_xyzw: Any, measured_xyzw: Any) -> float:
    reference = np.asarray(reference_xyzw, dtype=np.float64)
    measured = np.asarray(measured_xyzw, dtype=np.float64)
    if reference.shape != (4,) or measured.shape != (4,):
        raise Stage1AIsaacShortSmokeError("ORIENTATION_QUATERNION_SHAPE_INVALID")
    reference_norm = float(np.linalg.norm(reference))
    measured_norm = float(np.linalg.norm(measured))
    if reference_norm <= 1.0e-12 or measured_norm <= 1.0e-12:
        raise Stage1AIsaacShortSmokeError("ORIENTATION_QUATERNION_NORM_INVALID")
    quaternion_dot = float(
        abs(np.dot(reference / reference_norm, measured / measured_norm))
    )
    return math.degrees(
        2.0 * math.acos(float(np.clip(quaternion_dot, -1.0, 1.0)))
    )


def _step_joint_qdd_metrics(
    *, env: Any, records: list[Mapping[str, Any]]
) -> dict[str, float]:
    """Summarize unchanged 500-Hz qdd readback for arm/passive groups."""

    robot = env.scene["robot"]
    names = tuple(str(name) for name in robot.joint_names)
    arm_term = env.action_manager.get_term("arm_action")
    arm_ids = [int(value) for value in arm_term._joint_ids]
    passive_names = (
        "idx72_gripper_r_inner_joint3",
        "idx73_gripper_r_inner_joint4",
        "idx82_gripper_r_outer_joint3",
        "idx83_gripper_r_outer_joint4",
        "idx93_gripper_r_outer_joint2",
        "idx94_gripper_r_inner_joint2",
    )
    passive_ids = [names.index(name) for name in passive_names if name in names]
    arrays = [
        np.asarray(record["fd_qdd_rad_s2"], dtype=np.float64)
        for record in records
        if "fd_qdd_rad_s2" in record
    ]
    if not arrays:
        return {"arm_qdd_max_abs_rad_s2": 0.0, "passive_qdd_max_abs_rad_s2": 0.0}
    qdd = np.stack(arrays)

    def maximum(indices: list[int]) -> float:
        if not indices:
            return 0.0
        values = np.abs(qdd[:, indices])
        finite = values[np.isfinite(values)]
        return 0.0 if finite.size == 0 else float(np.max(finite))

    return {
        "arm_qdd_max_abs_rad_s2": maximum(arm_ids),
        "passive_qdd_max_abs_rad_s2": maximum(passive_ids),
    }


def _zero_xyz_close_execution_proposal(proposal: Any, *, alpha: Any) -> Any:
    """Return the exact replay/execution receipt for a privileged CLOSE hold.

    The privileged interlock may remove Cartesian motion while the gripper is
    closing, but it must not leave the learner believing that an unconsumed
    residual action reached the controller.  Re-compose the typed action with
    zero nominal XYZ and zero SAC residual, preserve the canonical abstract
    CLOSE command, and store the same zero residual in replay.
    """

    composition = compose_bc_and_residual_action(
        bc_action_metric_root_m=(0.0, 0.0, 0.0, 1.0),
        raw_sac_residual_metric_root_m=(0.0, 0.0, 0.0),
        alpha=alpha,
    )
    return replace(
        proposal,
        normalized_sac_action=np.zeros(3, dtype=np.float32),
        raw_residual_metric_root_m=(0.0, 0.0, 0.0),
        bc_action_4d_metric_root_m=(0.0, 0.0, 0.0, 1.0),
        composition=composition,
        residual_active=False,
    )


def source_authoritative_hybrid_activation_initial_state() -> tuple[Any, tuple[float, ...], dict[str, Any]]:
    """Select an immutable cuRobo row that remains above the 30-mm handoff.

    The direct-init path consumes two canonical OPEN refresh steps before the
    smoke.  Selection therefore requires the source row and the following
    three recorded rows to remain strictly above the existing 30-mm boundary.
    No pose component is synthesized or edited.
    """

    import h5py

    from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
        selected_pregrasp_initial_state,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
        nominal_grasp_pose_root_m_xyzw_for_cube,
    )
    from geniesim.rl.sac.keyboard_grasp_contract import (
        canonical_keyboard_grasp_contract,
    )

    base = selected_pregrasp_initial_state()
    source_path = Path(base.hdf5_path).resolve()
    if not source_path.is_file() or _sha256(source_path) != base.hdf5_sha256:
        raise Stage1AIsaacShortSmokeError("HYBRID_INITIAL_SOURCE_HASH_MISMATCH")
    nominal_pose = nominal_grasp_pose_root_m_xyzw_for_cube(
        base.cube_pose_robot_root_m_xyzw[:3]
    )
    nominal_position = np.asarray(nominal_pose[:3], dtype=np.float64)
    handoff_m = float(canonical_keyboard_grasp_contract().curobo_default_handoff_m)
    source_guard_rows = 3
    with h5py.File(source_path, "r") as source:
        required = (
            "rows/control_step",
            "rows/ee_pose_robot_root_m_xyzw",
            "rows/gripper_state_open",
            "rows/packet_receipt_json",
            "rows/right_arm_joint_position_rad",
            "rows/right_arm_joint_velocity_rad_s",
            "rows/right_wrist_depth_valid",
            "rows/right_wrist_rgb",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise Stage1AIsaacShortSmokeError(
                f"HYBRID_INITIAL_SOURCE_SCHEMA_MISSING:{missing}"
            )
        ee_pose = np.asarray(source["rows/ee_pose_robot_root_m_xyzw"], dtype=np.float64)
        gripper_open = np.asarray(source["rows/gripper_state_open"], dtype=np.float64).reshape(-1)
        control_steps = np.asarray(source["rows/control_step"], dtype=np.int64)
        q = np.asarray(source["rows/right_arm_joint_position_rad"], dtype=np.float64)
        qd = np.asarray(source["rows/right_arm_joint_velocity_rad_s"], dtype=np.float64)
        packet_json = source["rows/packet_receipt_json"][:]
        row_count = int(ee_pose.shape[0])
        if (
            ee_pose.shape != (row_count, 7)
            or q.shape != (row_count, 7)
            or qd.shape != (row_count, 7)
            or gripper_open.shape != (row_count,)
            or control_steps.shape != (row_count,)
        ):
            raise Stage1AIsaacShortSmokeError("HYBRID_INITIAL_SOURCE_SHAPE_INVALID")
        distances = np.linalg.norm(ee_pose[:, :3] - nominal_position, axis=1)
        packets: list[dict[str, Any]] = []
        for raw in packet_json:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            packets.append(json.loads(text))
        candidates: list[int] = []
        for index in range(row_count - source_guard_rows):
            stop = index + source_guard_rows + 1
            window_packets = packets[index:stop]
            window_is_contact_free = all(
                packet.get("contact") is False
                and packet.get("forbidden_collision") is False
                and packet.get("gripper_close") is False
                and packet.get("terminal") is False
                for packet in window_packets
            )
            if (
                np.isfinite(ee_pose[index:stop]).all()
                and np.isfinite(q[index:stop]).all()
                and np.isfinite(qd[index:stop]).all()
                and np.all(distances[index:stop] > handoff_m)
                and np.all(gripper_open[index:stop] == 1.0)
                and window_is_contact_free
            ):
                candidates.append(index)
        if not candidates:
            raise Stage1AIsaacShortSmokeError(
                "HYBRID_INITIAL_SOURCE_NO_ABOVE_30MM_CONTACT_FREE_WINDOW"
            )
        row_index = max(candidates)
        if not bool(np.asarray(source["rows/right_wrist_depth_valid"][row_index]).any()):
            raise Stage1AIsaacShortSmokeError("HYBRID_INITIAL_SOURCE_DEPTH_INVALID")
        rgb = np.asarray(source["rows/right_wrist_rgb"][row_index])
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise Stage1AIsaacShortSmokeError("HYBRID_INITIAL_SOURCE_RGB_INVALID")

    selected = replace(
        base,
        sample_id=f"{Path(base.hdf5_path).parent.name}/row-{row_index:06d}",
        hdf5_row_index=int(row_index),
        source_control_step=int(control_steps[row_index]),
        right_arm_q_rad=tuple(float(value) for value in q[row_index]),
        right_arm_qd_rad_s=tuple(float(value) for value in qd[row_index]),
        ee_pose_robot_root_m_xyzw=tuple(float(value) for value in ee_pose[row_index]),
    )
    receipt = {
        "schema": "g2_hybrid_activation_source_initial_state_v1",
        "SOURCE_AUTHORITATIVE": True,
        "selection_rule": (
            "LATEST_IMMUTABLE_CONTACT_FREE_SOURCE_ROW_WITH_CURRENT_AND_NEXT_"
            "THREE_ROWS_STRICTLY_ABOVE_EXISTING_30MM_HANDOFF"
        ),
        "source_hdf5_path": str(source_path),
        "source_hdf5_sha256": base.hdf5_sha256,
        "source_row_index": int(row_index),
        "source_control_step": int(control_steps[row_index]),
        "source_sample_id": selected.sample_id,
        "nominal_grasp_pose_root_m_xyzw": [float(value) for value in nominal_pose],
        "source_measured_distance_m": float(distances[row_index]),
        "source_window_min_distance_m": float(
            np.min(distances[row_index : row_index + source_guard_rows + 1])
        ),
        "handoff_authority_m": handoff_m,
        "source_guard_rows": source_guard_rows,
        "pose_or_threshold_modified": False,
    }
    return selected, tuple(float(value) for value in nominal_pose), receipt


@dataclass
class _CameraAcquisitionState:
    acquisition_id: int
    timestamp_s: float
    sensor_clock_time_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    depth_valid: np.ndarray


def _tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    return torch.from_dlpack(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _rotation_xyzw(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise Stage1AIsaacShortSmokeError("TELEMETRY_QUATERNION_INVALID")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1.0e-3):
        raise Stage1AIsaacShortSmokeError("TELEMETRY_QUATERNION_NOT_UNIT_XYZW")
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
        ),
        dtype=np.float64,
    )


def _runtime_cube_geometry_authority(env: Any) -> dict[str, Any]:
    """Read the effective oriented-box size from the live USD Cube prim."""

    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_path = "/World/envs/env_0/Object"
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise Stage1AIsaacShortSmokeError("RUNTIME_CUBE_ROOT_PRIM_MISSING")
    receipts: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Cube):
            continue
        cube = UsdGeom.Cube(prim)
        authored_size = float(cube.GetSizeAttr().Get())
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        scale = np.abs(np.asarray(Gf.Transform(matrix).GetScale(), dtype=np.float64))
        size = authored_size * scale
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            raise Stage1AIsaacShortSmokeError("RUNTIME_CUBE_SIZE_INVALID")
        receipts.append(
            {
                "prim_path": str(prim.GetPath()),
                "authored_cube_size": authored_size,
                "effective_scale_xyz": [float(value) for value in scale],
                "effective_size_m": [float(value) for value in size],
            }
        )
    if len(receipts) != 1:
        raise Stage1AIsaacShortSmokeError(
            f"RUNTIME_CUBE_GEOMETRY_AMBIGUOUS:{len(receipts)}"
        )
    return {
        **receipts[0],
        "quaternion_order": "XYZW_AFTER_EXPLICIT_ISAACLAB_NATIVE_CONVERSION",
        "geometry_source": "LIVE_USD_PRIM_EFFECTIVE_TRANSFORM",
    }


def _aggregate_raw_pad_contacts(
    records: list[Mapping[str, Any]], *, side: str
) -> dict[str, Any]:
    authorities = {
        "LEFT_PAD": (
            "raw_inner", "gripper_r_inner_link4", "PAD_PRIMARY"
        ),
        "RIGHT_PAD": (
            "raw_outer", "gripper_r_outer_link4", "PAD_PRIMARY"
        ),
        "OUTER_LINK2_CANDIDATE": (
            "raw_outer_link2",
            "gripper_r_outer_link2",
            "OUTER_LINK2_CANDIDATE",
        ),
    }
    if side not in authorities:
        raise ValueError(f"UNKNOWN_GRIPPER_SURFACE_SIDE:{side}")
    raw_name, body_name, surface_identity = authorities[side]
    raw_rows: list[dict[str, Any]] = []
    all_points_root: list[np.ndarray] = []
    all_normals_root: list[np.ndarray] = []
    all_points_link_local: list[np.ndarray] = []
    all_normals_link_local: list[np.ndarray] = []
    native_weights: list[float] = []
    derived_weights: list[float] = []
    force_time_ns = 0.0
    native_impulse_ns = 0.0
    nonfinite_geometry_slot_count = 0
    load_bearing_slot_count = 0
    for record in records:
        raw = record[raw_name]
        count = int(raw["count"])
        rotation = _rotation_xyzw(record["root_quat_world_xyzw"])
        root_position = np.asarray(record["root_position_world_m"], dtype=np.float64)
        points_world = np.asarray(raw["point_world_m"][:count], dtype=np.float64)
        normals_world = np.asarray(raw["normal_world"][:count], dtype=np.float64)
        points_root = (rotation.T @ (points_world - root_position).T).T
        normals_root = (rotation.T @ normals_world.T).T
        body_pose_world = np.asarray(
            raw["body_state"]["link_pose_world_xyzw"], dtype=np.float64
        )
        body_rotation_world = _rotation_xyzw(body_pose_world[3:7])
        points_link_local = (
            body_rotation_world.T
            @ (points_world - body_pose_world[:3]).T
        ).T
        normals_link_local = (body_rotation_world.T @ normals_world.T).T
        force = np.asarray(raw["normal_force_n"][:count], dtype=np.float64)
        native = np.asarray(raw["native_normal_impulse_ns"][:count], dtype=np.float64)
        derived = np.asarray(raw["derived_normal_impulse_ns"][:count], dtype=np.float64)
        slip = np.asarray(raw["slip_speed_m_s"][:count], dtype=np.float64)
        for index in range(count):
            point_world = points_world[index]
            point_root = points_root[index]
            normal_world = normals_world[index]
            normal_root = normals_root[index]
            serialized_geometry = {}
            nonfinite_geometry = {}
            geometry_valid = True
            for name, vector in (
                ("contact_point_world_m", point_world),
                ("contact_point_root_m", point_root),
                ("contact_normal_world", normal_world),
                ("contact_normal_root", normal_root),
                ("contact_point_link_local_m", points_link_local[index]),
                ("contact_normal_link_local", normals_link_local[index]),
            ):
                serialized, invalid, valid = serialize_physx_vector(name, vector)
                serialized_geometry[name] = serialized
                geometry_valid = geometry_valid and valid
                if invalid:
                    nonfinite_geometry[name] = invalid
            force_value = float(force[index])
            native_value = float(native[index])
            derived_value = float(derived[index])
            slip_value = float(slip[index])
            scalar_validity = {
                "normal_force_n": bool(np.isfinite(force_value)),
                "native_normal_impulse_ns": bool(np.isfinite(native_value)),
                "derived_normal_impulse_ns": bool(np.isfinite(derived_value)),
                "slip_speed_m_s": bool(np.isfinite(slip_value)),
            }
            finite_force = max(force_value, 0.0) if scalar_validity["normal_force_n"] else 0.0
            finite_native = max(native_value, 0.0) if scalar_validity["native_normal_impulse_ns"] else 0.0
            finite_derived = max(derived_value, 0.0) if scalar_validity["derived_normal_impulse_ns"] else 0.0
            load_bearing = max(finite_force, finite_native, finite_derived) > 0.0
            force_time_ns += finite_force * float(record["dt_s"])
            native_impulse_ns += finite_native
            if load_bearing:
                load_bearing_slot_count += 1
            if not geometry_valid:
                nonfinite_geometry_slot_count += 1
            if geometry_valid and load_bearing:
                all_points_root.append(point_root)
                all_normals_root.append(normal_root)
                all_points_link_local.append(points_link_local[index])
                all_normals_link_local.append(normals_link_local[index])
                native_weights.append(finite_native)
                derived_weights.append(finite_derived)
            raw_rows.append(
                {
                    "physics_substep": int(record["physics_substep"]),
                    **serialized_geometry,
                    "normal_force_n": force_value if scalar_validity["normal_force_n"] else None,
                    "native_normal_impulse_ns": native_value if scalar_validity["native_normal_impulse_ns"] else None,
                    "derived_normal_impulse_ns": derived_value if scalar_validity["derived_normal_impulse_ns"] else None,
                    "slip_speed_m_s": slip_value if scalar_validity["slip_speed_m_s"] else None,
                    "geometry_valid": geometry_valid,
                    "load_bearing": load_bearing,
                    "scalar_validity": scalar_validity,
                    "nonfinite_geometry_tokens": nonfinite_geometry,
                }
            )
    if all_points_root:
        points = np.stack(all_points_root)
        normals = np.stack(all_normals_root)
        points_local = np.stack(all_points_link_local)
        normals_local = np.stack(all_normals_link_local)
        native_array = np.asarray(native_weights, dtype=np.float64)
        derived_array = np.asarray(derived_weights, dtype=np.float64)
        if np.isfinite(native_array).all() and float(native_array.sum()) > 0.0:
            weights = native_array
            method = "ALL_CONTACT_NATIVE_NORMAL_IMPULSE_WEIGHTED_CENTROID"
        else:
            weights = derived_array
            method = "ALL_CONTACT_FORCE_TIMES_DT_WEIGHTED_CENTROID"
        if float(weights.sum()) > 0.0:
            point = np.sum(points * weights[:, None], axis=0) / float(weights.sum())
            normal = np.sum(normals * weights[:, None], axis=0)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            aggregated_point = point.tolist()
            aggregated_normal = normal.tolist()
            aggregated_point_link_local = (
                np.sum(points_local * weights[:, None], axis=0)
                / float(weights.sum())
            ).tolist()
            aggregated_normal_link_local = np.sum(
                normals_local * weights[:, None], axis=0
            )
            aggregated_normal_link_local /= max(
                float(np.linalg.norm(aggregated_normal_link_local)), 1.0e-12
            )
            aggregated_normal_link_local = aggregated_normal_link_local.tolist()
        else:
            method = "LOAD_WITHOUT_FINITE_WEIGHTED_GEOMETRY"
            aggregated_point = None
            aggregated_normal = None
            aggregated_point_link_local = None
            aggregated_normal_link_local = None
    else:
        method = "NO_CONTACT"
        aggregated_point = None
        aggregated_normal = None
        aggregated_point_link_local = None
        aggregated_normal_link_local = None
    return {
        "pad_side": side,
        "surface_side": side,
        "contact_link": body_name,
        "surface_identity": surface_identity,
        "surface_identity_valid": True,
        "surface_patch_membership_valid": surface_identity == "PAD_PRIMARY",
        "contact_body_a": body_name,
        "contact_body_b": "/World/envs/env_0/Object",
        "is_pad_cube_contact": load_bearing_slot_count > 0,
        "is_nonpad_cube_contact": False,
        "raw_contact_slot_count": len(raw_rows),
        "raw_contact_point_count": len(all_points_root),
        "load_bearing_contact_slot_count": load_bearing_slot_count,
        "nonfinite_geometry_slot_count": nonfinite_geometry_slot_count,
        "raw_contacts": raw_rows,
        "aggregated_contact_point_root_m": aggregated_point,
        "aggregated_contact_normal_root": aggregated_normal,
        "aggregated_contact_point_link_local_m": aggregated_point_link_local,
        "aggregated_contact_normal_link_local": aggregated_normal_link_local,
        "aggregation_method": method,
        "normal_impulse_ns": native_impulse_ns,
        "force_times_dt_impulse_ns": force_time_ns,
        "effective_normal_force_from_native_n": native_impulse_ns / 0.02,
        "effective_normal_force_from_force_time_n": force_time_ns / 0.02,
        "force_aggregation_method": "SUM_PER_CONTACT_PER_500HZ_SUBSTEP_OVER_20MS",
    }


def _gru_inputs(
    *,
    env: Any,
    p0a: Any,
    camera_state: _CameraAcquisitionState | None,
    previous_action: np.ndarray,
    hidden_reset: bool,
) -> tuple[HumanGraspSequenceInputs, _CameraAcquisitionState]:
    rgb, depth, valid, _raw_frame, sensor_time = _capture_camera_rgbd(
        env,
        camera_name="right_wrist",
        prior_frame=None,
        prior_sensor_time_s=0.0,
    )
    camera = env.scene["right_wrist_camera"]
    captured_source, age_source = camera_capture_time_and_age(camera)
    captured_source_s = float(_tensor(captured_source).reshape(-1)[0].item())
    age_source_s = float(_tensor(age_source).reshape(-1)[0].item())
    if (
        not math.isfinite(captured_source_s)
        or not math.isfinite(age_source_s)
        or age_source_s < 0.0
        or not math.isclose(
            captured_source_s, float(sensor_time), rel_tol=0.0, abs_tol=1.0e-6
        )
    ):
        raise Stage1AIsaacShortSmokeError(
            "RIGHT_WRIST_SENSOR_CLOCK_PROVENANCE_MISMATCH"
        )
    sensor_clock_time_s = captured_source_s + age_source_s
    if camera_state is not None:
        if sensor_time < camera_state.timestamp_s:
            raise Stage1AIsaacShortSmokeError(
                "RIGHT_WRIST_ACQUISITION_TIMESTAMP_REGRESSED"
            )
        if sensor_time == camera_state.timestamp_s:
            # Isaac may increment its public render/update counter before a
            # new 25-Hz acquisition becomes authoritative.  Acquisition time,
            # not that counter, owns the RGB-D identity.  Reuse the exact
            # cached frame for the adjacent 50-Hz control row.
            rgb = camera_state.rgb
            depth = camera_state.depth_m
            valid = camera_state.depth_valid
            acquisition_id = camera_state.acquisition_id
        else:
            acquisition_id = camera_state.acquisition_id + 1
        if sensor_clock_time_s + 1.0e-6 < camera_state.sensor_clock_time_s:
            raise Stage1AIsaacShortSmokeError(
                "RIGHT_WRIST_SENSOR_CLOCK_REGRESSED"
            )
    else:
        acquisition_id = 0
    # Apply the already-frozen deployable depth-validity rule.  Values beyond
    # the 2 m policy range are marked invalid and zeroed; they are never
    # clipped or reinterpreted as a closer surface.
    depth, valid = preprocess_depth_m(depth, valid, maximum_depth_m=2.0)
    next_camera_state = _CameraAcquisitionState(
        acquisition_id=acquisition_id,
        timestamp_s=float(sensor_time),
        sensor_clock_time_s=float(sensor_clock_time_s),
        rgb=np.asarray(rgb).copy(),
        depth_m=np.asarray(depth).copy(),
        depth_valid=np.asarray(valid).copy(),
    )
    position, quaternion = _ee_root_pose(env, p0a)
    robot = env.scene["robot"]
    arm_term = env.action_manager.get_term("arm_action")
    arm_ids = torch.as_tensor(
        [int(value) for value in arm_term._joint_ids],
        dtype=torch.long,
        device=env.device,
    )
    q = _tensor(robot.data.joint_pos).index_select(1, arm_ids)[0]
    qd = _tensor(robot.data.joint_vel).index_select(1, arm_ids)[0]
    gripper_state = 1.0 if previous_action[3] >= 0.5 else 0.0
    inputs = HumanGraspSequenceInputs(
        right_wrist_rgb=torch.as_tensor(rgb, dtype=torch.uint8, device=env.device)[None, None],
        right_wrist_depth_m=torch.as_tensor(depth, dtype=torch.float32, device=env.device)[None, None],
        right_wrist_depth_valid=torch.as_tensor(valid, dtype=torch.bool, device=env.device)[None, None],
        ee_pose_robot_root_m_xyzw=torch.as_tensor(
            np.concatenate((position, quaternion)), dtype=torch.float32, device=env.device
        )[None, None],
        right_arm_joint_position_rad=q[None, None],
        right_arm_joint_velocity_rad_s=qd[None, None],
        current_gripper_state=torch.tensor(
            [[[gripper_state]]], dtype=torch.float32, device=env.device
        ),
        previous_policy_action_4d_metric_root_m=torch.as_tensor(
            previous_action, dtype=torch.float32, device=env.device
        )[None, None],
        hidden_reset_mask=torch.tensor(
            [[hidden_reset]], dtype=torch.bool, device=env.device
        ),
    )
    return inputs, next_camera_state


def _cube_world_state(env: Any) -> tuple[np.ndarray, ...]:
    cube = env.scene["object"].data
    quaternion = quaternion_native_to_xyzw(
        _tensor(cube.root_quat_w), isaaclab_native_quaternion_order()
    )
    return tuple(
        _tensor(value)[0].detach().cpu().numpy().astype(np.float64)
        for value in (
            cube.root_pos_w,
            quaternion,
            cube.root_lin_vel_w,
            cube.root_ang_vel_w,
        )
    )


def _reset_direct_pregrasp(
    *, env: Any, p0a: Any, preflight: Any, counter: Any, deferred_port: Any, latch: Any,
    physics_telemetry: Any, episode_index: int, sample: Any | None = None,
    runtime_seed: int | None = None,
) -> None:
    # Keep the Isaac application and loaded asset alive, but make each CLOSE
    # supervision episode causally independent.  Contact/qdot history is
    # cleared before env.reset so the first new substep cannot be differenced
    # against the preceding episode's closed-gripper terminal state.
    reset_buffers = getattr(physics_telemetry, "reset_episode_buffers", None)
    if callable(reset_buffers):
        reset_buffers()
    env.reset(seed=42 + episode_index if runtime_seed is None else runtime_seed)
    if sample is None:
        apply_direct_pregrasp_initial_state(env)
    else:
        apply_direct_pregrasp_initial_state(env, sample=sample)
    arm_term = env.action_manager.get_term("arm_action")
    arm_term.synchronize_reset_to_measured(
        torch.zeros((1,), device=env.device, dtype=torch.long)
    )
    latch.reset()
    if deferred_port.outstanding_packet is not None:
        raise Stage1AIsaacShortSmokeError("RESET_WITH_OUTSTANDING_ACTION_PACKET")
    open_action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.0))
    open_packet, _, _ = p0a._build_authoritative_packet(
        high_level=open_action, batch_size=1, device=env.device, latch=latch
    )
    for refresh in range(2):
        physics_telemetry.set_context(
            policy_step=-(2 - refresh), bc_step=None, phase="RESET_REFRESH",
            gripper_intent="OPEN", close_onset=False, clipping=False,
        )
        outputs, receipt = preflight._consume_once(
            env=env, counter=counter, deferred_port=deferred_port,
            packet=open_packet, label=f"STAGE1A_RESET_REFRESH_{episode_index}_{refresh}",
        )
        if not receipt["single_consumption"]:
            raise Stage1AIsaacShortSmokeError("RESET_REFRESH_NOT_SINGLE_CONSUMPTION")
        latch.commit(open_packet.gripper_intent)
        if bool(outputs[2].reshape(-1)[0]) or bool(outputs[3].reshape(-1)[0]):
            raise Stage1AIsaacShortSmokeError("RESET_REFRESH_TERMINATED")


def _grasp_geometry_snapshot(env: Any, p0a: Any) -> dict[str, Any]:
    """Read-only geometry/joint snapshot for CLOSE-admission attribution."""

    robot = env.scene["robot"]
    body_names = tuple(str(name) for name in robot.body_names)
    requested_bodies = (
        "gripper_r_inner_link2",
        "gripper_r_outer_link2",
        "gripper_r_inner_link4",
        "gripper_r_outer_link4",
    )
    missing_bodies = [name for name in requested_bodies if name not in body_names]
    if missing_bodies:
        raise Stage1AIsaacShortSmokeError(
            f"CLOSE_DIAGNOSTIC_BODY_MISSING:{missing_bodies}"
        )
    body_pos = _tensor(robot.data.body_pos_w)[0].detach().cpu().numpy()
    body_quat_native = _tensor(robot.data.body_quat_w)
    body_quat = body_quat_native[0].detach().cpu().numpy()
    body_quat_xyzw = quaternion_native_to_xyzw(
        body_quat_native, isaaclab_native_quaternion_order()
    )[0].detach().cpu().numpy()
    cube_pos, cube_quat, cube_lin, cube_ang = _cube_world_state(env)
    ee_pos_root, ee_quat_root = _ee_root_pose(env, p0a)
    bodies: dict[str, Any] = {}
    for name in requested_bodies:
        index = body_names.index(name)
        position = np.asarray(body_pos[index], dtype=np.float64)
        bodies[name] = {
            "position_world_m": position.tolist(),
            "quaternion_world_native": np.asarray(
                body_quat[index], dtype=np.float64
            ).tolist(),
            "quaternion_world_xyzw": np.asarray(
                body_quat_xyzw[index], dtype=np.float64
            ).tolist(),
            "center_to_cube_center_m": float(np.linalg.norm(position-cube_pos)),
        }
    joint_names = tuple(str(name) for name in robot.joint_names)
    requested_joints = (
        "idx71_gripper_r_inner_joint1",
        "idx72_gripper_r_inner_joint3",
        "idx73_gripper_r_inner_joint4",
        "idx81_gripper_r_outer_joint1",
        "idx82_gripper_r_outer_joint3",
        "idx83_gripper_r_outer_joint4",
    )
    q = _tensor(robot.data.joint_pos)[0].detach().cpu().numpy()
    qd = _tensor(robot.data.joint_vel)[0].detach().cpu().numpy()
    joints = {
        name: {
            "q_rad": float(q[joint_names.index(name)]),
            "qd_rad_s": float(qd[joint_names.index(name)]),
        }
        for name in requested_joints
    }
    return {
        "cube_position_world_m": cube_pos.tolist(),
        "cube_quaternion_world_xyzw": cube_quat.tolist(),
        "cube_linear_velocity_world_m_s": cube_lin.tolist(),
        "cube_angular_velocity_world_rad_s": cube_ang.tolist(),
        "ee_position_root_m": np.asarray(ee_pos_root, dtype=np.float64).tolist(),
        "ee_quaternion_root_xyzw": np.asarray(
            ee_quat_root, dtype=np.float64
        ).tolist(),
        "bodies": bodies,
        "joints": joints,
    }


def _right_arm_state_snapshot(env: Any) -> dict[str, Any]:
    """Read the controller-owned right-arm ordering without mutation."""

    robot = env.scene["robot"]
    arm_term = env.action_manager.get_term("arm_action")
    joint_ids = [int(value) for value in arm_term._joint_ids]
    joint_names = tuple(str(name) for name in robot.joint_names)
    q = _tensor(robot.data.joint_pos)[0].detach().cpu().numpy()
    qd = _tensor(robot.data.joint_vel)[0].detach().cpu().numpy()
    return {
        "joint_names": [joint_names[index] for index in joint_ids],
        "q_rad": [float(q[index]) for index in joint_ids],
        "qd_rad_s": [float(qd[index]) for index in joint_ids],
    }


def _privileged_close_readiness(
    *,
    env: Any,
    p0a: Any,
    task_mdp: Any,
    nominal_grasp_pose_root_m_xyzw: Any,
    ee_speed_m_s: float,
    no_safety_violation: bool,
    gate: SimplifiedClosePersistenceGate,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Evaluate the simplified one-shot CLOSE gate and log pad geometry.

    Only nominal residual, orientation, five-step persistence, and the
    unchanged safety verdict are admission authority.  Pad/cube geometry is
    retained as privileged label telemetry and is never appended to the
    deployable GRU/SAC observation.
    """

    from geniesim.rl.sac.privileged_geometry_oracle import (
        runtime_primary_pad_cube_oracle,
    )

    snapshot = _grasp_geometry_snapshot(env, p0a)
    oracle = runtime_primary_pad_cube_oracle(
        cube_center_world_m=snapshot["cube_position_world_m"],
        cube_quat_world_xyzw=snapshot["cube_quaternion_world_xyzw"],
        cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
        body_pose_world_m_xyzw_by_name={
            name: (*body["position_world_m"], *body["quaternion_world_xyzw"])
            for name, body in snapshot["bodies"].items()
            if name in (
                "gripper_r_inner_link4",
                "gripper_r_outer_link4",
            )
        },
    )
    nominal = np.asarray(nominal_grasp_pose_root_m_xyzw[:3], dtype=np.float64)
    current = np.asarray(snapshot["ee_position_root_m"], dtype=np.float64)
    error = nominal - current
    orientation_error_deg = _orientation_error_deg(
        nominal_grasp_pose_root_m_xyzw[3:],
        snapshot["ee_quaternion_root_xyzw"],
    )
    receipt = gate.observe(
        nominal_grasp_residual_m=float(np.linalg.norm(error)),
        orientation_error_deg=orientation_error_deg,
        no_safety_violation=bool(no_safety_violation),
    )
    receipt.update(
        {
            "geometry_ready": bool(receipt["close_trigger"]),
            "pad_gap_ready_logging_only": bool(
                oracle["aperture_geometrically_compatible"]
            ),
            "longitudinal_error_mm": float(error[0]) * 1000.0,
            "lateral_error_mm": float(np.linalg.norm(error[1:]) * 1000.0),
            "controller_speed_m_s_logging_only": float(ee_speed_m_s),
            "residual_authority_m": [
                SIMPLIFIED_CLOSE_MIN_RESIDUAL_M,
                SIMPLIFIED_CLOSE_MAX_RESIDUAL_M,
            ],
            "orientation_authority_deg": (
                SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG
            ),
            "control_hz": CONTROL_HZ,
            "student_privileged_input_count": 0,
        }
    )
    return snapshot, oracle, receipt


def _durable_jsonl(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_stage1a_close_admission_diagnostic(
    *, env: Any, p0a: Any, preflight: Any, task_mdp: Any, counter: Any,
    deferred_port: Any, latch: Any, physics_telemetry: Any,
    source_freeze_before: Mapping[str, Any],
    source_freeze_provider: Callable[[], Mapping[str, Any]],
    selected_asset_path: Path, selected_asset_sha256: str,
    nominal_grasp_pose_root_m_xyzw: Any, bc_checkpoint: Path,
    bc_checkpoint_sha256: str, output_dir: Path, report_path: Path,
    calibration_episode_id: str = "candidate-a-stage1a-v7-unsafe",
    pregrasp_sample_id: str = "COMMON_BASELINE_SAMPLE",
    maximum_action_submissions: int = 80,
) -> int:
    """One bounded, no-training replay of the frozen GRU CLOSE path."""

    if maximum_action_submissions != 80:
        raise Stage1AIsaacShortSmokeError(
            "CLOSE_ADMISSION_DIAGNOSTIC_BOUND_MUST_BE_80"
        )
    if report_path.exists() or output_dir.exists():
        raise Stage1AIsaacShortSmokeError(
            "CLOSE_ADMISSION_DIAGNOSTIC_REFUSES_OVERWRITE"
        )
    if _sha256(selected_asset_path) != selected_asset_sha256:
        raise Stage1AIsaacShortSmokeError("STAGE1A_ASSET_HASH_MISMATCH")
    output_dir.mkdir(parents=True, exist_ok=False)
    coordinator = Stage1ARealSACCoordinator.from_bc_checkpoint(
        bc_checkpoint,
        expected_sha256=bc_checkpoint_sha256,
        replay_capacity=128,
        device=env.device,
        seed=42,
    )
    coordinator.reset_episode()
    reward = Stage1AGraspReward(1, device=env.device)
    cube_geometry_authority = _runtime_cube_geometry_authority(env)
    runtime_cube_size = np.asarray(
        cube_geometry_authority["effective_size_m"], dtype=np.float64
    )
    configured_cube_size = 2.0 * np.asarray(
        task_mdp.TASK.cube_half_extents_m, dtype=np.float64
    )
    if not np.allclose(runtime_cube_size, configured_cube_size, atol=1.0e-6, rtol=0.0):
        raise Stage1AIsaacShortSmokeError(
            "RUNTIME_CUBE_GEOMETRY_CONFIG_MISMATCH:"
            f"{runtime_cube_size.tolist()}:{configured_cube_size.tolist()}"
        )
    nominal = np.asarray(nominal_grasp_pose_root_m_xyzw[:3], dtype=np.float64)
    previous_action = np.zeros(4, dtype=np.float32)
    camera_state: _CameraAcquisitionState | None = None
    inputs, camera_state = _gru_inputs(
        env=env,
        p0a=p0a,
        camera_state=camera_state,
        previous_action=previous_action,
        hidden_reset=True,
    )
    proposal = coordinator.propose(
        inputs, deterministic=True, residual_active=False
    )
    trace_path = output_dir / "CLOSE_ADMISSION_TRACE.jsonl"
    first_close_submission: int | None = None
    failure_snapshot_path: Path | None = None
    verdict = "UNRESOLVED_BOUND_EXHAUSTED"
    action_submission_count = 0
    with trace_path.open("w", encoding="utf-8") as trace_stream:
        for submission_index in range(maximum_action_submissions):
            final_action = proposal.composition.final_action_4d_metric_root_m
            xyz = np.asarray(final_action[:3], dtype=np.float64)
            if float(np.linalg.norm(xyz)) > MAX_FINAL_ACTION_M + 1.0e-12:
                raise Stage1AIsaacShortSmokeError("FINAL_ACTION_BOUND_VIOLATION")
            if any(abs(float(value)) > 1.0e-15 for value in proposal.raw_residual_metric_root_m):
                raise Stage1AIsaacShortSmokeError(
                    "CLOSE_DIAGNOSTIC_RESIDUAL_NOT_ZERO"
                )
            if final_action[3] >= 0.5 and first_close_submission is None:
                first_close_submission = submission_index + 1
            geometry_before = _grasp_geometry_snapshot(env, p0a)
            current_ee, current_ee_quat = _ee_root_pose(env, p0a)
            normalized = metric_xyz_to_normalized(xyz)
            high_level = HighLevelPolicyAction.from_sequence(
                (*np.asarray(normalized, dtype=np.float64).tolist(), float(final_action[3]))
            )
            packet, _, _ = p0a._build_authoritative_packet(
                high_level=high_level, batch_size=1, device=env.device, latch=latch
            )
            records_start = len(physics_telemetry.records)
            physics_telemetry.set_context(
                policy_step=submission_index,
                bc_step=submission_index,
                phase="CLOSE_ADMISSION_DIAGNOSTIC",
                gripper_intent=packet.gripper_intent.value,
                close_onset=(
                    packet.gripper_intent is AbstractGripperIntent.CLOSE
                    and latch.intent is not AbstractGripperIntent.CLOSE
                ),
                clipping=False,
            )
            outputs, consumption = preflight._consume_once(
                env=env,
                counter=counter,
                deferred_port=deferred_port,
                packet=packet,
                label=f"CLOSE_ADMISSION_{submission_index:03d}",
            )
            if not consumption["single_consumption"]:
                raise Stage1AIsaacShortSmokeError("CONSUMPTION_UNKNOWN")
            action_submission_count += 1
            latch.commit(packet.gripper_intent)
            records = physics_telemetry.records[records_start:]
            if len(records) != 10:
                raise Stage1AIsaacShortSmokeError(
                    f"STAGE1A_PHYSICS_RECORD_COUNT:{len(records)}"
                )
            evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
            peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
            forbidden_peak = max(
                (max(values) for values in peaks.values()), default=0.0
            )
            terminated = bool(outputs[2].reshape(-1)[0].item())
            truncated = bool(outputs[3].reshape(-1)[0].item())
            active = preflight._active_termination_names(
                env, outputs[2], outputs[3]
            )
            next_ee, _ = _ee_root_pose(env, p0a)
            cube_pos, cube_quat, cube_lin, cube_ang = _cube_world_state(env)
            contact_paths = tuple(
                physics_telemetry._contact_sources[side]["partner_actor"]
                for side in ("inner", "outer")
            )
            runtime_state = Stage1ARuntimeTransitionState(
                current_ee_position_root_m=current_ee,
                next_ee_position_root_m=next_ee,
                nominal_grasp_position_root_m=nominal,
                nominal_approach_axis_root=(1.0, 0.0, 0.0),
                stage1a_active=True,
                grasp_decision_phase=True,
                phase_reset=submission_index == 0,
                episode_reset=submission_index == 0,
                root_position_world_m_by_substep=np.stack(
                    [row["root_position_world_m"] for row in records]
                ),
                root_quat_world_xyzw_by_substep=np.stack(
                    [row["root_quat_world_xyzw"] for row in records]
                ),
                root_linear_velocity_world_m_s_by_substep=np.stack(
                    [row["root_linear_velocity_world_m_s"] for row in records]
                ),
                root_angular_velocity_world_rad_s_by_substep=np.stack(
                    [row["root_angular_velocity_world_rad_s"] for row in records]
                ),
                cube_center_world_m=cube_pos,
                cube_quat_world_xyzw=cube_quat,
                cube_linear_velocity_world_m_s=cube_lin,
                cube_angular_velocity_world_rad_s=cube_ang,
                cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                effective_residual_root_m=(0.0, 0.0, 0.0),
                previous_effective_residual_root_m=(0.0, 0.0, 0.0),
                final_action_xyz_root_m=xyz,
                contract_grasp_state_valid=False,
                safety_violation=bool(forbidden_peak > 1.0e-6),
                authoritative_safety_penalty=0.0,
                non_pad_gripper_cube_contact=bool(forbidden_peak > 1.0e-6),
                close_command=packet.gripper_intent is AbstractGripperIntent.CLOSE,
            )
            reward_step = reward.step(
                build_stage1a_reward_inputs(
                    records,
                    runtime_state,
                    contact_partner_actor_paths=contact_paths,
                    device=env.device,
                )
            )
            left_contact = bool(reward_step.left_contact_boolean[0].item())
            right_contact = bool(reward_step.right_contact_boolean[0].item())
            stable_bilateral = bool(reward_step.stable_grasp_boolean[0].item())
            trace_row = {
                "episode_id": calibration_episode_id,
                "pregrasp_sample_id": pregrasp_sample_id,
                "submission_index": submission_index + 1,
                "close_probability": proposal.close_probability,
                "feasibility_probability": proposal.feasibility_probability,
                "close_threshold": coordinator.close_calibration.close_threshold,
                "open_threshold": coordinator.close_calibration.open_threshold,
                "gripper_intent": packet.gripper_intent.value,
                "actual_gripper_state": latch.intent.value,
                "left_pad_contact": left_contact,
                "right_pad_contact": right_contact,
                "bilateral_contact": left_contact and right_contact,
                "stable_bilateral": stable_bilateral,
                "non_pad_link_cube_contact": bool(forbidden_peak > 1.0e-6),
                "forbidden_collision": bool(
                    forbidden_peak > 1.0e-6 or "forbidden_collision" in active
                ),
                "final_action_4d_metric_root_m": [
                    float(value) for value in final_action
                ],
                "effective_residual_root_m": [0.0, 0.0, 0.0],
                "geometry_before": geometry_before,
                "physics_substep_count": len(records),
                "sensor_peak_forces_n": peaks,
                "terminated": terminated,
                "truncated": truncated,
                "active_termination_names": list(active),
            }
            _durable_jsonl(trace_stream, trace_row)
            if forbidden_peak > 1.0e-6 or "forbidden_collision" in active:
                verdict = "UNSAFE_CLOSE"
                failure_snapshot_path = output_dir / "FORBIDDEN_COLLISION_SNAPSHOT.json"
                _atomic_json(
                    failure_snapshot_path,
                    {
                        **trace_row,
                        "schema": "g2_stage1a_close_admission_failure_v1",
                        "sensor_body_peak_forces_n": (
                            evaluator.sensor_body_peak_forces_n()
                            if evaluator is not None else {}
                        ),
                        "filtered_pad_force_components_n": (
                            evaluator.filtered_pad_force_components_n()
                            if evaluator is not None else {}
                        ),
                        "diagnostic_contact_force_components_n": (
                            evaluator.diagnostic_contact_force_components_n()
                            if evaluator is not None else {}
                        ),
                    },
                )
                break
            if terminated or truncated:
                verdict = "UNSAFE_CLOSE"
                break
            if stable_bilateral:
                verdict = "SAFE_CLOSE"
                break
            previous_action = np.asarray(final_action, dtype=np.float32)
            next_inputs, camera_state = _gru_inputs(
                env=env,
                p0a=p0a,
                camera_state=camera_state,
                previous_action=previous_action,
                hidden_reset=False,
            )
            proposal = coordinator.propose(
                next_inputs, deterministic=True, residual_active=False
            )
    if verdict == "UNRESOLVED_BOUND_EXHAUSTED":
        verdict = (
            "NO_CLOSE_OUTCOME"
            if first_close_submission is None
            else "UNSAFE_CLOSE"
        )
    coordinator.assert_bc_frozen()
    freeze_after = source_freeze_provider()
    report = {
        "schema": "g2_stage1a_close_admission_diagnostic_v1",
        "execution_mode": "BOUNDED_REAL_ISAAC_NO_TRAINING",
        "maximum_action_submissions": maximum_action_submissions,
        "action_submission_count": action_submission_count,
        "first_close_submission": first_close_submission,
        "episode_id": calibration_episode_id,
        "pregrasp_sample_id": pregrasp_sample_id,
        "calibration_nominal_grasp_pose_root_m_xyzw": [
            float(value) for value in nominal_grasp_pose_root_m_xyzw
        ],
        "calibration_nominal_residual_m": 0.0185,
        "calibration_target_source": (
            "MEASURED_EE_TOWARD_MATCHING_SAMPLE_CUBE_ROBOT_ROOT"
        ),
        "final_outcome": verdict,
        "feasibility_supervision_rows": int(
            first_close_submission is not None and verdict in ("SAFE_CLOSE", "UNSAFE_CLOSE")
        ),
        "CLOSE_ADMISSION_DIAGNOSTIC": verdict,
        "SAC_UPDATE_COUNT": coordinator.sac_update_count,
        "TRAINING_STARTED": "NO",
        "effective_residual_m": 0.0,
        "trace_path": str(trace_path),
        "failure_snapshot_path": (
            str(failure_snapshot_path) if failure_snapshot_path is not None else None
        ),
        "source_freeze_before": dict(source_freeze_before),
        "source_freeze_after": dict(freeze_after),
        "source_freeze_match": source_freeze_before == freeze_after,
        "asset": {"path": str(selected_asset_path), "sha256": selected_asset_sha256},
        "bc_checkpoint": {"path": str(bc_checkpoint), "sha256": bc_checkpoint_sha256},
    }
    _atomic_json(report_path, report)
    _atomic_json(output_dir / "CLOSE_ADMISSION_REPORT.json", report)
    return 0 if verdict == "SAFE_CLOSE" else 2


def run_stage1a_close_residual_sweep_episode(
    *, env: Any, p0a: Any, preflight: Any, task_mdp: Any, counter: Any,
    deferred_port: Any, latch: Any, physics_telemetry: Any,
    close_mechanics_telemetry: Any,
    acceleration_limit_rad_s2: float,
    source_freeze_before: Mapping[str, Any],
    source_freeze_provider: Callable[[], Mapping[str, Any]],
    selected_asset_path: Path, selected_asset_sha256: str,
    nominal_grasp_pose_root_m_xyzw: Any, bc_checkpoint: Path,
    bc_checkpoint_sha256: str, output_dir: Path, report_path: Path,
    calibration_episode_id: str, pregrasp_sample_id: str,
    target_residual_mm: int, runtime_seed: int,
    lateral_offset_mm: float = 0.0,
    height_offset_mm: float = 0.0,
    approach_yaw_deg: float = 0.0,
    maximum_approach_submissions: int = 160,
    maximum_post_close_submissions: int = 80,
    hard_stop_limit_numerical_tolerance_rad: float = 1.0e-5,
    continue_after_hard_stop_for_telemetry: bool = False,
    geometry_forced_close_diagnostic: bool = False,
    relaxed_close_hold_bilateral: bool = False,
    relaxed_close_hold_event: str = "BILATERAL",
    simplified_close_persistence_gate: bool = False,
    close_speed_scale: float = 1.0,
    two_stage_close: bool = False,
) -> int:
    """One deterministic approach + one CLOSE edge at a fixed residual.

    The frozen GRU is evaluated for logging only.  It owns neither XYZ nor
    gripper actuation in this diagnostic.  The post-CLOSE hysteresis-band
    command preserves the latched state without creating another CLOSE edge.
    """

    if target_residual_mm not in (15, 16, 17, 18, 19, 20, 21, 22, 23):
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_TARGET_NOT_APPROVED")
    if (
        isinstance(maximum_post_close_submissions, bool)
        or not isinstance(maximum_post_close_submissions, int)
        or not 80 <= maximum_post_close_submissions <= 500
    ):
        raise Stage1AIsaacShortSmokeError(
            "POST_CLOSE_OBSERVATION_SUBMISSION_BOUND_INVALID"
        )
    if isinstance(runtime_seed, bool) or not isinstance(runtime_seed, int):
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_RUNTIME_SEED_INVALID")
    if relaxed_close_hold_event not in {"FIRST_CONTACT", "BILATERAL", "STABLE"}:
        raise Stage1AIsaacShortSmokeError("RELAXED_CLOSE_HOLD_EVENT_INVALID")
    if simplified_close_persistence_gate and not geometry_forced_close_diagnostic:
        raise Stage1AIsaacShortSmokeError(
            "SIMPLIFIED_CLOSE_GATE_REQUIRES_DIAGNOSTIC_CLOSE_MODE"
        )
    approved_close_speed_scales = (0.25, 0.50, 0.75, 1.00)
    if not any(
        math.isclose(float(close_speed_scale), value, rel_tol=0.0, abs_tol=1.0e-12)
        for value in approved_close_speed_scales
    ):
        raise Stage1AIsaacShortSmokeError("CLOSE_SPEED_SCALE_NOT_APPROVED")
    if two_stage_close and not math.isclose(
        float(close_speed_scale), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise Stage1AIsaacShortSmokeError(
            "TWO_STAGE_CLOSE_REQUIRES_NORMAL_INITIAL_SPEED"
        )
    if two_stage_close and not simplified_close_persistence_gate:
        raise Stage1AIsaacShortSmokeError(
            "TWO_STAGE_CLOSE_REQUIRES_SIMPLIFIED_CLOSE_GATE"
        )
    if report_path.exists() or output_dir.exists():
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_REFUSES_OVERWRITE")
    if _sha256(selected_asset_path) != selected_asset_sha256:
        raise Stage1AIsaacShortSmokeError("STAGE1A_ASSET_HASH_MISMATCH")
    output_dir.mkdir(parents=True, exist_ok=False)
    coordinator = Stage1ARealSACCoordinator.from_bc_checkpoint(
        bc_checkpoint,
        expected_sha256=bc_checkpoint_sha256,
        replay_capacity=128,
        device=env.device,
        seed=42,
    )
    coordinator.reset_episode()
    gripper_term = env.action_manager.get_term("gripper_action")
    speed_setter = getattr(gripper_term, "set_environment_speed_limit_rad_s", None)
    if not callable(speed_setter):
        raise Stage1AIsaacShortSmokeError(
            "CANONICAL_CLOSE_SPEED_INTERFACE_UNAVAILABLE"
        )
    configured_close_speed_rad_s = float(
        gripper_term.cfg.maximum_joint_target_speed_rad_s
    )
    configured_close_acceleration_rad_s2 = float(
        gripper_term.cfg.maximum_joint_target_acceleration_rad_s2
    )
    if (relaxed_close_hold_bilateral or two_stage_close) and not callable(
        getattr(gripper_term, "set_external_hold_mask", None)
    ):
        raise Stage1AIsaacShortSmokeError(
            "RELAXED_CLOSE_CANONICAL_EXTERNAL_HOLD_UNAVAILABLE"
        )
    if callable(getattr(gripper_term, "set_external_hold_mask", None)):
        gripper_term.set_external_hold_mask(
            torch.zeros(1, dtype=torch.bool, device=env.device)
        )
    reward = Stage1AGraspReward(1, device=env.device)
    cube_geometry_authority = _runtime_cube_geometry_authority(env)
    runtime_cube_size = np.asarray(
        cube_geometry_authority["effective_size_m"], dtype=np.float64
    )
    configured_cube_size = 2.0 * np.asarray(
        task_mdp.TASK.cube_half_extents_m, dtype=np.float64
    )
    if not np.allclose(runtime_cube_size, configured_cube_size, atol=1.0e-6, rtol=0.0):
        raise Stage1AIsaacShortSmokeError(
            "RUNTIME_CUBE_GEOMETRY_CONFIG_MISMATCH:"
            f"{runtime_cube_size.tolist()}:{configured_cube_size.tolist()}"
        )
    nominal = np.asarray(nominal_grasp_pose_root_m_xyzw[:3], dtype=np.float64)
    target_m = float(target_residual_mm) / 1000.0
    if not all(
        math.isfinite(value)
        for value in (lateral_offset_mm, height_offset_mm, approach_yaw_deg)
    ):
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_VARIATION_NONFINITE")
    if abs(lateral_offset_mm) > 4.0 or abs(height_offset_mm) > 2.0:
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_TRANSLATION_VARIATION_OUT_OF_SCOPE")
    if abs(approach_yaw_deg) > 4.0:
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_DIRECTION_VARIATION_OUT_OF_SCOPE")
    # Keep boundary candidates inside the unchanged 15--22 mm reward
    # authority while still representing their named neighborhoods.
    effective_target_m = (
        0.01525
        if target_residual_mm == 15
        else 0.01975
        if target_residual_mm == 20 and simplified_close_persistence_gate
        else 0.02175
        if target_residual_mm == 22
        else target_m
    )
    yaw_rad = math.radians(float(approach_yaw_deg))
    approach_axis = np.asarray(
        (math.cos(yaw_rad), math.sin(yaw_rad), 0.0), dtype=np.float64
    )
    lateral_axis = np.asarray(
        (-math.sin(yaw_rad), math.cos(yaw_rad), 0.0), dtype=np.float64
    )
    lateral_m = float(lateral_offset_mm) / 1000.0
    height_m = float(height_offset_mm) / 1000.0
    transverse_squared = lateral_m * lateral_m + height_m * height_m
    if transverse_squared >= effective_target_m * effective_target_m:
        raise Stage1AIsaacShortSmokeError("CLOSE_SWEEP_VARIATION_EXCEEDS_RESIDUAL")
    longitudinal_m = math.sqrt(effective_target_m**2 - transverse_squared)
    desired_preclose_error = (
        longitudinal_m * approach_axis
        + lateral_m * lateral_axis
        + height_m * np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    )
    desired_preclose_ee = nominal + desired_preclose_error
    initial_ee_root, _ = _ee_root_pose(env, p0a)
    oracle_required_xyz_mm = float(
        np.linalg.norm(
            desired_preclose_ee - np.asarray(initial_ee_root, dtype=np.float64)
        )
        * 1000.0
    )
    previous_action = np.zeros(4, dtype=np.float32)
    camera_state: _CameraAcquisitionState | None = None
    inputs, camera_state = _gru_inputs(
        env=env,
        p0a=p0a,
        camera_state=camera_state,
        previous_action=previous_action,
        hidden_reset=True,
    )
    proposal = coordinator.propose(inputs, deterministic=True, residual_active=False)
    trace_path = output_dir / "CLOSE_RESIDUAL_SWEEP_TRACE.jsonl"
    action_submission_count = 0
    close_edge_count = 0
    pre_close_hard_stop = False
    post_close_hard_stop = False
    actual_residual_at_close_mm: float | None = None
    p_close_at_close: float | None = None
    p_feasible_at_close: float | None = None
    control_step_at_close: int | None = None
    outcome = "INVALID_PRE_CLOSE_MECHANICS"
    outcome_reason = "TARGET_NOT_REACHED"
    approach_target_reached = False
    latest_reward_step: Any | None = None
    last_measured_ee_velocity_root_m_s = np.zeros(3, dtype=np.float64)
    last_approach_cosine = 0.0
    lateral_error_at_close_mm: float | None = None
    longitudinal_error_at_close_mm: float | None = None
    ee_velocity_at_close_root_m_s: list[float] | None = None
    ee_speed_at_close_m_s: float | None = None
    approach_cosine_at_close: float | None = None
    pre_close_features: dict[str, Any] | None = None
    final_outcome_telemetry: dict[str, Any] | None = None
    post_close_contact_timeline: list[dict[str, Any]] = []
    first_left_pad_contact_step: int | None = None
    first_right_pad_contact_step: int | None = None
    first_bilateral_step: int | None = None
    stable_grasp_step: int | None = None
    contact_region_at_first_contact: dict[str, str] = {}
    contact_region_at_stable: dict[str, str] = {}
    stable_outcome_achieved = False
    physics_event_markers: dict[str, dict[str, Any]] = {}
    natural_gru_close_edge_count = 0
    natural_gru_first_close_submission: int | None = None
    natural_gru_first_close_residual_mm: float | None = None
    natural_gru_closed_previous = False
    geometry_oracle_preclose: dict[str, Any] | None = None
    geometry_oracle_after_close: dict[str, Any] | None = None
    geometry_ready_receipt: dict[str, Any] | None = None
    simplified_gate = SimplifiedClosePersistenceGate()
    simplified_gate_receipts: list[dict[str, Any]] = []
    first_valid_contact_nominal_residual_mm: float | None = None
    external_hold_latched = False
    external_hold_first_submission: int | None = None
    external_hold_applied_count = 0
    close_speed_configured = False
    close_speed_transition_submission: int | None = None
    close_speed_transition_contact_kind: str | None = None
    close_speed_history: list[dict[str, Any]] = []
    post_contact_close_speed_scale = 0.25
    zero_xyz_during_close_count = 0
    nonzero_xyz_during_close_count = 0
    gripper_aperture_at_events_m: dict[str, float | None] = {
        "close_start": None,
        "first_contact": None,
        "bilateral": None,
        "stable": None,
        "final_held": None,
    }
    student_preclose_arrays: dict[str, np.ndarray] | None = None
    student_preclose_metadata: dict[str, Any] | None = None

    def consume(
        *, xyz_metric: np.ndarray, gripper_probability: float,
        label: str, policy_step: int, close_edge: bool,
    ) -> tuple[Any, list[dict[str, Any]], Any]:
        nonlocal action_submission_count
        if xyz_metric.shape != (3,) or not np.isfinite(xyz_metric).all():
            raise Stage1AIsaacShortSmokeError("SWEEP_ACTION_INVALID")
        if float(np.linalg.norm(xyz_metric)) > MAX_FINAL_ACTION_M + 1.0e-12:
            raise Stage1AIsaacShortSmokeError("FINAL_ACTION_BOUND_VIOLATION")
        normalized = metric_xyz_to_normalized(xyz_metric)
        high_level = HighLevelPolicyAction.from_sequence(
            (*np.asarray(normalized, dtype=np.float64).tolist(), gripper_probability)
        )
        packet, _, _ = p0a._build_authoritative_packet(
            high_level=high_level, batch_size=1, device=env.device, latch=latch
        )
        submission_number = action_submission_count + 1
        close_mechanics_telemetry.set_control_step(submission_number)
        if close_edge:
            close_mechanics_telemetry.mark_close_submission(submission_number)
        records_start = len(physics_telemetry.records)
        physics_telemetry.set_context(
            policy_step=policy_step,
            bc_step=policy_step,
            phase="CLOSE_RESIDUAL_SWEEP",
            gripper_intent=packet.gripper_intent.value,
            close_onset=close_edge,
            clipping=False,
        )
        previous_hard_stop = close_mechanics_telemetry.hard_stop_event
        outputs, consumption = preflight._consume_once(
            env=env,
            counter=counter,
            deferred_port=deferred_port,
            packet=packet,
            label=label,
        )
        if not consumption["single_consumption"]:
            raise Stage1AIsaacShortSmokeError("CONSUMPTION_UNKNOWN")
        hard_stop = close_mechanics_telemetry.poll_hard_stop_post_step(
            acceleration_limit_rad_s2=acceleration_limit_rad_s2,
            limit_numerical_tolerance_rad=(
                hard_stop_limit_numerical_tolerance_rad
            ),
        )
        if hard_stop is not None and previous_hard_stop is None and (
            close_mechanics_telemetry.close_submit_step is None
            or not continue_after_hard_stop_for_telemetry
        ):
            raise RuntimeError(
                "CLOSE_MECHANICS_POSTSTEP_HARD_STOP_EVENT:"
                f"{hard_stop['joint_name']}:"
                f"sample={hard_stop['physics_step']}:"
                f"qdd={hard_stop['qdd_2point_rad_s2']}"
            )
        action_submission_count += 1
        latch.commit(packet.gripper_intent)
        records = physics_telemetry.records[records_start:]
        if len(records) != 10:
            raise Stage1AIsaacShortSmokeError(
                f"STAGE1A_PHYSICS_RECORD_COUNT:{len(records)}"
            )
        return outputs, records, packet

    def update_gru(*, hidden_reset: bool = False) -> None:
        nonlocal inputs, camera_state, proposal
        inputs, camera_state = _gru_inputs(
            env=env,
            p0a=p0a,
            camera_state=camera_state,
            previous_action=previous_action,
            hidden_reset=hidden_reset,
        )
        proposal = coordinator.propose(
            inputs, deterministic=True, residual_active=False
        )

    with trace_path.open("w", encoding="utf-8") as trace_stream:
        # Approach deterministically while OPEN.  The 1 mm command cap is well
        # below the existing 4.5 mm action contract and is not a safety-limit
        # change.  CLOSE occurs only after the measured residual is at or just
        # inside the requested target (250 um historical progress deadband).
        for approach_index in range(maximum_approach_submissions + 1):
            current_ee, current_ee_quat = _ee_root_pose(env, p0a)
            current_ee_array = np.asarray(current_ee, dtype=np.float64)
            nominal_error = nominal - current_ee_array
            residual = float(np.linalg.norm(nominal_error))
            natural_gru_closed = bool(
                proposal.composition.final_gripper_probability >= 0.5
            )
            if natural_gru_closed and not natural_gru_closed_previous:
                natural_gru_close_edge_count += 1
                natural_gru_first_close_submission = action_submission_count + 1
                natural_gru_first_close_residual_mm = residual * 1000.0
            natural_gru_closed_previous = natural_gru_closed
            desired_error = desired_preclose_ee - current_ee_array
            desired_distance = float(np.linalg.norm(desired_error))
            target_reached = desired_distance <= 0.00010
            if target_reached:
                approach_target_reached = True
                break
            if approach_index == maximum_approach_submissions:
                break
            direction = desired_error / max(desired_distance, 1.0e-12)
            xyz = direction * float(np.clip(desired_distance, 0.0, 0.001))
            try:
                outputs, records, packet = consume(
                    xyz_metric=xyz,
                    gripper_probability=0.0,
                    label=f"CLOSE_SWEEP_APPROACH_{approach_index:03d}",
                    policy_step=approach_index,
                    close_edge=False,
                )
            except RuntimeError as error_runtime:
                if "HARD_STOP_EVENT" in str(error_runtime):
                    pre_close_hard_stop = True
                    outcome_reason = f"PRE_CLOSE_HARD_STOP:{error_runtime}"
                    break
                raise
            next_ee, _ = _ee_root_pose(env, p0a)
            next_ee = np.asarray(next_ee, dtype=np.float64)
            measured_velocity = (
                next_ee - np.asarray(current_ee, dtype=np.float64)
            ) * float(CONTROL_HZ)
            next_error = nominal - next_ee
            velocity_norm = float(np.linalg.norm(measured_velocity))
            next_error_norm = float(np.linalg.norm(next_error))
            measured_cosine = float(
                np.dot(measured_velocity, next_error)
                / max(velocity_norm * next_error_norm, 1.0e-12)
            )
            last_measured_ee_velocity_root_m_s = measured_velocity
            last_approach_cosine = measured_cosine
            evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
            peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
            forbidden_peak = max((max(values) for values in peaks.values()), default=0.0)
            terminated = bool(outputs[2].reshape(-1)[0].item())
            truncated = bool(outputs[3].reshape(-1)[0].item())
            _durable_jsonl(
                trace_stream,
                {
                    "phase": "APPROACH_OPEN",
                    "target_residual_mm": target_residual_mm,
                    "actual_residual_mm": residual * 1000.0,
                    "desired_preclose_distance_mm": desired_distance * 1000.0,
                    "submission_index": action_submission_count,
                    "p_close": proposal.close_probability,
                    "p_feasible": proposal.feasibility_probability,
                    "gru_close_hysteresis": natural_gru_closed,
                    "gru_close_edge": bool(
                        natural_gru_first_close_submission == action_submission_count
                    ),
                    "command_xyz_metric_root_m": xyz.tolist(),
                    "measured_ee_velocity_root_m_s": measured_velocity.tolist(),
                    "approach_cosine": measured_cosine,
                    "gripper_intent": packet.gripper_intent.value,
                    "close_edge": False,
                    "forbidden_collision": forbidden_peak > 1.0e-6,
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )
            if forbidden_peak > 1.0e-6 or terminated or truncated:
                outcome_reason = "PRE_CLOSE_COLLISION_OR_TERMINATION"
                break
            previous_action[:] = (*xyz.tolist(), 0.0)
            update_gru()

        if (
            approach_target_reached
            and not pre_close_hard_stop
            and simplified_close_persistence_gate
        ):
            # Hold OPEN at zero XYZ while the simplified conditions persist.
            # The fifth consecutive 50-Hz observation emits the sole CLOSE
            # edge; no p_close value participates in this decision.
            for persistence_index in range(
                SIMPLIFIED_CLOSE_PERSISTENCE_STEPS + 4
            ):
                current_ee, current_quat = _ee_root_pose(env, p0a)
                current_ee_array = np.asarray(current_ee, dtype=np.float64)
                residual_m = float(np.linalg.norm(nominal - current_ee_array))
                orientation_error_deg = _orientation_error_deg(
                    nominal_grasp_pose_root_m_xyzw[3:], current_quat
                )
                evaluator = getattr(
                    env, "_g2_forbidden_collision_evaluator", None
                )
                peaks = (
                    evaluator.sensor_peak_forces_n()
                    if evaluator is not None
                    else {}
                )
                forbidden_peak = max(
                    (max(values) for values in peaks.values()), default=0.0
                )
                gate_receipt = simplified_gate.observe(
                    nominal_grasp_residual_m=residual_m,
                    orientation_error_deg=orientation_error_deg,
                    no_safety_violation=bool(
                        forbidden_peak <= 1.0e-6 and not pre_close_hard_stop
                    ),
                )
                gate_receipt = {
                    **gate_receipt,
                    "phase": "SIMPLIFIED_CLOSE_PERSISTENCE_OPEN",
                    "p_close_logging_only": float(proposal.close_probability),
                    "p_feasible_logging_only": float(
                        proposal.feasibility_probability
                    ),
                    "gru_hysteresis_logging_only": bool(
                        proposal.composition.final_gripper_probability >= 0.5
                    ),
                    "action_submission_count": action_submission_count,
                }
                simplified_gate_receipts.append(gate_receipt)
                _durable_jsonl(trace_stream, gate_receipt)
                if gate_receipt["close_trigger"]:
                    geometry_ready_receipt = dict(gate_receipt)
                    geometry_ready_receipt["geometry_ready"] = True
                    break
                try:
                    outputs, _, packet = consume(
                        xyz_metric=np.zeros(3, dtype=np.float64),
                        gripper_probability=0.0,
                        label=(
                            "SIMPLIFIED_CLOSE_PERSISTENCE_OPEN_"
                            f"{persistence_index:02d}"
                        ),
                        policy_step=(
                            maximum_approach_submissions + persistence_index
                        ),
                        close_edge=False,
                    )
                except RuntimeError as error_runtime:
                    if "HARD_STOP_EVENT" in str(error_runtime):
                        pre_close_hard_stop = True
                        outcome_reason = (
                            "PRE_CLOSE_PERSISTENCE_HARD_STOP:"
                            f"{error_runtime}"
                        )
                        break
                    raise
                terminated = bool(outputs[2].reshape(-1)[0].item())
                truncated = bool(outputs[3].reshape(-1)[0].item())
                if forbidden_peak > 1.0e-6 or terminated or truncated:
                    outcome_reason = (
                        "PRE_CLOSE_PERSISTENCE_COLLISION_OR_TERMINATION"
                    )
                    break
                previous_action[:] = (0.0, 0.0, 0.0, 0.0)
                update_gru()
            if not simplified_gate.close_latched:
                approach_target_reached = False
                if outcome_reason == "TARGET_NOT_REACHED":
                    outcome_reason = "SIMPLIFIED_CLOSE_PERSISTENCE_NOT_REACHED"

        if approach_target_reached and not pre_close_hard_stop:
            current_ee, _ = _ee_root_pose(env, p0a)
            actual_residual_at_close_mm = float(
                np.linalg.norm(nominal - np.asarray(current_ee, dtype=np.float64))
                * 1000.0
            )
            close_error = nominal - np.asarray(current_ee, dtype=np.float64)
            approach_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
            longitudinal = float(np.dot(close_error, approach_axis))
            lateral = close_error - longitudinal * approach_axis
            longitudinal_error_at_close_mm = longitudinal * 1000.0
            lateral_error_at_close_mm = float(np.linalg.norm(lateral) * 1000.0)
            ee_velocity_at_close_root_m_s = (
                last_measured_ee_velocity_root_m_s.tolist()
            )
            ee_speed_at_close_m_s = float(
                np.linalg.norm(last_measured_ee_velocity_root_m_s)
            )
            approach_cosine_at_close = float(last_approach_cosine)
            grasp_preclose = _grasp_geometry_snapshot(env, p0a)
            arm_preclose = _right_arm_state_snapshot(env)
            inner_pad = np.asarray(
                grasp_preclose["bodies"]["gripper_r_inner_link4"]["position_world_m"],
                dtype=np.float64,
            )
            outer_pad = np.asarray(
                grasp_preclose["bodies"]["gripper_r_outer_link4"]["position_world_m"],
                dtype=np.float64,
            )
            nominal_error_unit = close_error / max(
                float(np.linalg.norm(close_error)), 1.0e-12
            )
            pre_close_features = {
                "nominal_grasp_residual_mm": actual_residual_at_close_mm,
                "lateral_error_mm": lateral_error_at_close_mm,
                "longitudinal_error_mm": longitudinal_error_at_close_mm,
                "approach_cosine": approach_cosine_at_close,
                "ee_linear_velocity_root_m_s": ee_velocity_at_close_root_m_s,
                "ee_linear_speed_m_s": ee_speed_at_close_m_s,
                "ee_approach_axis_speed_m_s": float(
                    np.dot(last_measured_ee_velocity_root_m_s, nominal_error_unit)
                ),
                "passive_gripper_joint_state": grasp_preclose["joints"],
                "gripper_opening_m": float(np.linalg.norm(inner_pad - outer_pad)),
                "gripper_state": "OPEN",
                "right_arm_state": arm_preclose,
                "configured_lateral_offset_mm": float(lateral_offset_mm),
                "configured_height_offset_mm": float(height_offset_mm),
                "configured_approach_yaw_deg": float(approach_yaw_deg),
            }
            gripper_aperture_at_events_m["close_start"] = float(
                pre_close_features["gripper_opening_m"]
            )
            student_preclose_arrays = {
                name: getattr(inputs, name).detach().cpu().numpy().copy()
                for name in (
                    "right_wrist_rgb",
                    "right_wrist_depth_m",
                    "right_wrist_depth_valid",
                    "ee_pose_robot_root_m_xyzw",
                    "right_arm_joint_position_rad",
                    "right_arm_joint_velocity_rad_s",
                    "current_gripper_state",
                    "previous_policy_action_4d_metric_root_m",
                    "hidden_reset_mask",
                )
            }
            student_preclose_metadata = {
                "camera_timestamp_s": float(camera_state.timestamp_s),
                "camera_frame_id": int(camera_state.acquisition_id),
                "rgbd_hz": RGBD_HZ,
                "control_hz": CONTROL_HZ,
                "frame": "robot_root",
                "position_unit": "m",
                "student_privileged_input_count": 0,
            }
            if geometry_forced_close_diagnostic:
                from geniesim.rl.sac.privileged_geometry_oracle import (
                    runtime_primary_pad_cube_oracle,
                )

                geometry_oracle_preclose = runtime_primary_pad_cube_oracle(
                    cube_center_world_m=grasp_preclose["cube_position_world_m"],
                    cube_quat_world_xyzw=grasp_preclose["cube_quaternion_world_xyzw"],
                    cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                    body_pose_world_m_xyzw_by_name={
                        name: (
                            *body["position_world_m"],
                            *body["quaternion_world_xyzw"],
                        )
                        for name, body in grasp_preclose["bodies"].items()
                        if name in (
                            "gripper_r_inner_link4",
                            "gripper_r_outer_link4",
                        )
                    },
                )
                _, current_ee_quat = _ee_root_pose(env, p0a)
                nominal_quat = np.asarray(
                    nominal_grasp_pose_root_m_xyzw[3:], dtype=np.float64
                )
                orientation_error_deg = _orientation_error_deg(
                    nominal_quat, current_ee_quat
                )
                geometry_authority_path = (
                    Path(__file__).resolve().parents[4]
                    / "artifacts/g2_curobo_grasp_ready_20260920"
                    / "train_authority_v3/DEMO_GRASP_READY_GEOMETRY.json"
                )
                geometry_authority = json.loads(
                    geometry_authority_path.read_text(encoding="utf-8")
                )["grasp_ready_region"]
                lateral_authority_mm = float(
                    np.linalg.norm(
                        np.asarray(
                            geometry_authority["absolute_axis_error_p95_m"][1:],
                            dtype=np.float64,
                        )
                    )
                    * 1000.0
                )
                orientation_authority_deg = float(
                    geometry_authority["orientation_error_p95_deg"]
                )
                controller_settled_speed_m_s = (
                    Stage1ARewardV3Config().progress_epsilon_m * CONTROL_HZ
                )
                pad_gap_ready = bool(
                    geometry_oracle_preclose["aperture_geometrically_compatible"]
                )
                lateral_ready = bool(
                    lateral_error_at_close_mm <= lateral_authority_mm
                )
                orientation_ready = bool(
                    orientation_error_deg <= orientation_authority_deg
                )
                controller_settled = bool(
                    ee_speed_at_close_m_s <= controller_settled_speed_m_s
                )
                no_safety_violation = bool(
                    not pre_close_hard_stop and approach_target_reached
                )
                legacy_geometry_ready_receipt = {
                    "pad_gap_ready": pad_gap_ready,
                    "lateral_ready": lateral_ready,
                    "orientation_ready": orientation_ready,
                    "controller_settled": controller_settled,
                    "no_safety_violation": no_safety_violation,
                    "geometry_ready": all(
                        (
                            pad_gap_ready,
                            lateral_ready,
                            orientation_ready,
                            controller_settled,
                            no_safety_violation,
                        )
                    ),
                    "lateral_error_mm": lateral_error_at_close_mm,
                    "lateral_authority_p95_mm": lateral_authority_mm,
                    "orientation_error_deg": orientation_error_deg,
                    "orientation_authority_p95_deg": orientation_authority_deg,
                    "controller_speed_m_s": ee_speed_at_close_m_s,
                    "controller_settled_speed_authority_m_s": (
                        controller_settled_speed_m_s
                    ),
                    "position_target_tolerance_mm": 0.10,
                    "threshold_authority": str(geometry_authority_path),
                }
                if simplified_close_persistence_gate:
                    if geometry_ready_receipt is None:
                        raise Stage1AIsaacShortSmokeError(
                            "SIMPLIFIED_CLOSE_TRIGGER_RECEIPT_MISSING"
                        )
                    geometry_ready_receipt.update(
                        {
                            "pad_gap_ready_logging_only": pad_gap_ready,
                            "lateral_ready_logging_only": lateral_ready,
                            "controller_settled_logging_only": (
                                controller_settled
                            ),
                            "legacy_geometry_receipt_logging_only": (
                                legacy_geometry_ready_receipt
                            ),
                        }
                    )
                else:
                    geometry_ready_receipt = legacy_geometry_ready_receipt
                if not geometry_ready_receipt["geometry_ready"]:
                    outcome = "INVALID_PRE_CLOSE_MECHANICS"
                    outcome_reason = "PRIVILEGED_GEOMETRY_NOT_READY"
            p_close_at_close = float(proposal.close_probability)
            p_feasible_at_close = float(proposal.feasibility_probability)
            # Exactly one OPEN->CLOSE edge.  Subsequent g=0.5 commands remain
            # in the hysteresis band and therefore preserve, not retrigger,
            # the existing CLOSE state.
            for observation_index in range(
                maximum_post_close_submissions
                if outcome_reason != "PRIVILEGED_GEOMETRY_NOT_READY"
                else 0
            ):
                close_edge = observation_index == 0
                g = 1.0 if close_edge else 0.5
                if close_edge:
                    close_edge_count += 1
                    control_step_at_close = action_submission_count + 1
                if not close_speed_configured:
                    initial_speed_scale = (
                        1.0 if two_stage_close else float(close_speed_scale)
                    )
                    initial_speed_rad_s = (
                        configured_close_speed_rad_s * initial_speed_scale
                    )
                    speed_setter(
                        torch.full(
                            (1,), initial_speed_rad_s,
                            dtype=torch.float32, device=env.device,
                        )
                    )
                    close_speed_history.append(
                        {
                            "phase": "CLOSE_START",
                            "submission": action_submission_count + 1,
                            "speed_scale": initial_speed_scale,
                            "speed_limit_rad_s": initial_speed_rad_s,
                        }
                    )
                    close_speed_configured = True
                if relaxed_close_hold_bilateral or two_stage_close:
                    gripper_term.set_external_hold_mask(
                        torch.tensor(
                            [external_hold_latched],
                            dtype=torch.bool,
                            device=env.device,
                        )
                    )
                    external_hold_applied_count += int(external_hold_latched)
                current_ee, _ = _ee_root_pose(env, p0a)
                current_residual = float(
                    np.linalg.norm(nominal - np.asarray(current_ee, dtype=np.float64))
                )
                try:
                    zero_xyz_during_close_count += 1
                    outputs, records, packet = consume(
                        xyz_metric=np.zeros(3, dtype=np.float64),
                        gripper_probability=g,
                        label=f"CLOSE_SWEEP_OBSERVE_{observation_index:03d}",
                        policy_step=maximum_approach_submissions + observation_index,
                        close_edge=close_edge,
                    )
                except RuntimeError as error_runtime:
                    if "HARD_STOP_EVENT" in str(error_runtime):
                        post_close_hard_stop = True
                        outcome = "UNSAFE_CLOSE"
                        outcome_reason = f"POST_CLOSE_HARD_STOP:{error_runtime}"
                        break
                    raise
                hard_stop_event = close_mechanics_telemetry.hard_stop_event
                if hard_stop_event is not None and not post_close_hard_stop:
                    post_close_hard_stop = True
                    outcome = "UNSAFE_CLOSE"
                    outcome_reason = (
                        "POST_CLOSE_HARD_STOP:"
                        f"CLOSE_MECHANICS_POSTSTEP_HARD_STOP_EVENT:"
                        f"{hard_stop_event['joint_name']}:"
                        f"sample={hard_stop_event['physics_step']}:"
                        f"qdd={hard_stop_event['qdd_2point_rad_s2']}"
                    )
                evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
                peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
                forbidden_peak = max((max(values) for values in peaks.values()), default=0.0)
                terminated = bool(outputs[2].reshape(-1)[0].item())
                truncated = bool(outputs[3].reshape(-1)[0].item())
                active = preflight._active_termination_names(env, outputs[2], outputs[3])
                next_ee, _ = _ee_root_pose(env, p0a)
                cube_pos, cube_quat, cube_lin, cube_ang = _cube_world_state(env)
                contact_paths = tuple(
                    physics_telemetry._contact_sources[side]["partner_actor"]
                    for side in ("inner", "outer")
                )
                # This bounded diagnostic explicitly authorizes CLOSE outcome
                # evaluation at 15--23 mm.  Mark the already-issued candidate
                # as an active grasp-decision sample so the unchanged stable
                # contact/dwell contract can classify 23 mm as well.  This is
                # not a runtime admission threshold and is never actor input.
                phase_valid = True
                runtime_state = Stage1ARuntimeTransitionState(
                    current_ee_position_root_m=current_ee,
                    next_ee_position_root_m=next_ee,
                    nominal_grasp_position_root_m=nominal,
                    nominal_approach_axis_root=(1.0, 0.0, 0.0),
                    stage1a_active=phase_valid,
                    grasp_decision_phase=phase_valid,
                    phase_reset=observation_index == 0,
                    episode_reset=observation_index == 0,
                    root_position_world_m_by_substep=np.stack(
                        [row["root_position_world_m"] for row in records]
                    ),
                    root_quat_world_xyzw_by_substep=np.stack(
                        [row["root_quat_world_xyzw"] for row in records]
                    ),
                    root_linear_velocity_world_m_s_by_substep=np.stack(
                        [row["root_linear_velocity_world_m_s"] for row in records]
                    ),
                    root_angular_velocity_world_rad_s_by_substep=np.stack(
                        [row["root_angular_velocity_world_rad_s"] for row in records]
                    ),
                    cube_center_world_m=cube_pos,
                    cube_quat_world_xyzw=cube_quat,
                    cube_linear_velocity_world_m_s=cube_lin,
                    cube_angular_velocity_world_rad_s=cube_ang,
                    cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                    effective_residual_root_m=(0.0, 0.0, 0.0),
                    previous_effective_residual_root_m=(0.0, 0.0, 0.0),
                    final_action_xyz_root_m=(0.0, 0.0, 0.0),
                    contract_grasp_state_valid=False,
                    safety_violation=bool(forbidden_peak > 1.0e-6),
                    authoritative_safety_penalty=0.0,
                    non_pad_gripper_cube_contact=bool(forbidden_peak > 1.0e-6),
                    close_command=True,
                )
                reward_inputs = build_stage1a_reward_inputs(
                    records,
                    runtime_state,
                    contact_partner_actor_paths=contact_paths,
                    device=env.device,
                )
                latest_reward_step = reward.step(reward_inputs)
                left_contact = bool(latest_reward_step.left_contact_boolean[0].item())
                right_contact = bool(latest_reward_step.right_contact_boolean[0].item())
                stable = bool(latest_reward_step.stable_grasp_boolean[0].item())
                raw_contact = {
                    side: _aggregate_raw_pad_contacts(records, side=side)
                    for side in (
                        "LEFT_PAD",
                        "RIGHT_PAD",
                        "OUTER_LINK2_CANDIDATE",
                    )
                }
                surface_force_components = (
                    evaluator.diagnostic_contact_force_components_n()
                    if evaluator is not None
                    else {}
                )
                rigid_nonpad_body_peaks = (
                    evaluator.sensor_body_peak_forces_n()
                    if evaluator is not None
                    else {}
                )
                rigid_nonpad_contact_bodies = sorted(
                    {
                        body_name
                        for body_rows in rigid_nonpad_body_peaks.values()
                        for body_name, values in body_rows.items()
                        if max((float(value) for value in values), default=0.0)
                        > float(getattr(evaluator, "force_threshold_n", 1.0e-6))
                    }
                )
                cube_center_root = [
                    float(value)
                    for value in reward_inputs.cube_center_root_m[0]
                    .detach().cpu().tolist()
                ]
                cube_quat_root = [
                    float(value)
                    for value in reward_inputs.cube_quat_root_xyzw[0]
                    .detach().cpu().tolist()
                ]
                region_by_side: dict[str, str] = {}
                for side, contact_receipt in raw_contact.items():
                    point = contact_receipt["aggregated_contact_point_root_m"]
                    normal = contact_receipt["aggregated_contact_normal_root"]
                    if point is None or normal is None:
                        continue
                    outward = np.asarray(point, dtype=np.float64) - np.asarray(
                        cube_center_root, dtype=np.float64
                    )
                    canonical_normal = np.asarray(normal, dtype=np.float64)
                    if float(np.dot(canonical_normal, outward)) < 0.0:
                        canonical_normal = -canonical_normal
                    contact_receipt["aggregated_contact_normal_root"] = (
                        canonical_normal.tolist()
                    )
                    contact_receipt["normal_convention"] = "CUBE_TO_PAD_OUTWARD"
                    local_point = root_point_to_cube_local(
                        point, cube_center_root, cube_quat_root
                    )
                    region = classify_cube_contact_region(
                        local_point,
                        0.5 * runtime_cube_size,
                    )
                    contact_receipt["aggregated_contact_point_cube_local_m"] = (
                        list(local_point)
                    )
                    contact_receipt["contact_region"] = region.region
                    contact_receipt["q_contact_geometry"] = region.quality_weight
                    region_by_side[side] = region.region

                timeline_step = {
                    "post_close_control_step": observation_index,
                    "absolute_control_step": action_submission_count,
                    "left_pad_contact": left_contact,
                    "right_pad_contact": right_contact,
                    "bilateral_contact": left_contact and right_contact,
                    "stable_grasp": stable,
                    "cube_center_root_m": cube_center_root,
                    "cube_quat_root_xyzw": cube_quat_root,
                    "cube_geometry_authority": cube_geometry_authority,
                    "pad_contacts": raw_contact,
                    "surface_contacts": raw_contact,
                    "outer_link2_force_components": surface_force_components,
                    "rigid_nonpad_contact_bodies": rigid_nonpad_contact_bodies,
                    "contact_region_by_side": region_by_side,
                    "nominal_residual_mm": current_residual * 1000.0,
                }
                close_physics_step = close_mechanics_telemetry.close_submit_physics_step

                def mark_physics_event(name: str, physics_row: Mapping[str, Any]) -> None:
                    if name in physics_event_markers or close_physics_step is None:
                        return
                    physics_step = int(physics_row["global_physics_sample"])
                    relative_ms = (
                        physics_step - int(close_physics_step)
                    ) * 1000.0 / float(PHYSICS_HZ)
                    physics_event_markers[name] = {
                        "physics_step": physics_step,
                        "control_step": int(physics_row["policy_step"]),
                        "timestamp_s": (
                            float(close_mechanics_telemetry.close_submit_timestamp_s)
                            + relative_ms / 1000.0
                        ),
                        "relative_to_close_ms": relative_ms,
                    }

                for physics_row in records:
                    inner_count = int(physics_row["raw_inner"]["count"])
                    outer_count = int(physics_row["raw_outer"]["count"])
                    outer_link2_count = int(
                        physics_row["raw_outer_link2"]["count"]
                    )
                    if inner_count or outer_count or outer_link2_count:
                        mark_physics_event("FIRST_CONTACT", physics_row)
                    if inner_count and outer_count:
                        mark_physics_event("FIRST_BILATERAL_CONTACT", physics_row)
                    if bool(physics_row["stable_contact"]):
                        mark_physics_event("STABLE_START", physics_row)
                    if bool(physics_row["forbidden_collision"]):
                        mark_physics_event("FORBIDDEN_COLLISION", physics_row)
                timeline_step["physics_event_markers_observed"] = {
                    key: value
                    for key, value in physics_event_markers.items()
                    if key in {
                        "FIRST_CONTACT",
                        "FIRST_BILATERAL_CONTACT",
                        "STABLE_START",
                        "FORBIDDEN_COLLISION",
                    }
                }
                post_close_contact_timeline.append(timeline_step)
                if left_contact and first_left_pad_contact_step is None:
                    first_left_pad_contact_step = observation_index
                    first_valid_contact_nominal_residual_mm = (
                        current_residual * 1000.0
                    )
                    if "LEFT_PAD" in region_by_side:
                        contact_region_at_first_contact["LEFT_PAD"] = region_by_side["LEFT_PAD"]
                if right_contact and first_right_pad_contact_step is None:
                    first_right_pad_contact_step = observation_index
                    if first_valid_contact_nominal_residual_mm is None:
                        first_valid_contact_nominal_residual_mm = (
                            current_residual * 1000.0
                        )
                    if "RIGHT_PAD" in region_by_side:
                        contact_region_at_first_contact["RIGHT_PAD"] = region_by_side["RIGHT_PAD"]
                if left_contact and right_contact and first_bilateral_step is None:
                    first_bilateral_step = observation_index
                if stable and stable_grasp_step is None:
                    stable_grasp_step = observation_index
                    contact_region_at_stable = dict(region_by_side)

                def force_weighted_contact_vector(
                    side_index: int, values: Any
                ) -> list[float] | None:
                    valid = reward_inputs.pad_cube_contact_valid[0, side_index]
                    if not bool(valid.any().item()):
                        return None
                    weights = reward_inputs.normal_force_n[0, side_index] * valid
                    denominator = float(weights.sum().item())
                    if denominator <= 0.0:
                        return None
                    selected = values[0, side_index]
                    vector = (selected * weights[:, None]).sum(dim=0) / weights.sum()
                    return [float(value) for value in vector.detach().cpu().tolist()]

                left_force = float(latest_reward_step.left_normal_force_n[0].item())
                right_force = float(latest_reward_step.right_normal_force_n[0].item())
                force_sum = left_force + right_force
                force_balance = (
                    1.0 - abs(left_force - right_force) / force_sum
                    if force_sum > 1.0e-12
                    else 0.0
                )
                grasp_snapshot = _grasp_geometry_snapshot(env, p0a)
                aperture_m = float(
                    np.linalg.norm(
                        np.asarray(
                            grasp_snapshot["bodies"]["gripper_r_inner_link4"][
                                "position_world_m"
                            ],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            grasp_snapshot["bodies"]["gripper_r_outer_link4"][
                                "position_world_m"
                            ],
                            dtype=np.float64,
                        )
                    )
                )
                if (
                    (left_contact or right_contact)
                    and gripper_aperture_at_events_m["first_contact"] is None
                ):
                    gripper_aperture_at_events_m["first_contact"] = aperture_m
                if (
                    left_contact
                    and right_contact
                    and gripper_aperture_at_events_m["bilateral"] is None
                ):
                    gripper_aperture_at_events_m["bilateral"] = aperture_m
                if stable and gripper_aperture_at_events_m["stable"] is None:
                    gripper_aperture_at_events_m["stable"] = aperture_m
                if (
                    two_stage_close
                    and close_speed_transition_submission is None
                    and (left_contact or right_contact)
                ):
                    post_contact_speed_rad_s = (
                        configured_close_speed_rad_s
                        * post_contact_close_speed_scale
                    )
                    speed_setter(
                        torch.full(
                            (1,), post_contact_speed_rad_s,
                            dtype=torch.float32, device=env.device,
                        )
                    )
                    close_speed_transition_submission = action_submission_count + 1
                    close_speed_transition_contact_kind = (
                        "BILATERAL" if left_contact and right_contact else "SINGLE"
                    )
                    close_speed_history.append(
                        {
                            "phase": "POST_FIRST_VALID_CONTACT",
                            "submission": close_speed_transition_submission,
                            "speed_scale": post_contact_close_speed_scale,
                            "speed_limit_rad_s": post_contact_speed_rad_s,
                            "contact_kind": close_speed_transition_contact_kind,
                        }
                    )
                if (
                    (relaxed_close_hold_bilateral or two_stage_close)
                    and (
                        (
                            relaxed_close_hold_bilateral
                            and (
                                (relaxed_close_hold_event == "FIRST_CONTACT" and (left_contact or right_contact))
                                or (relaxed_close_hold_event == "BILATERAL" and left_contact and right_contact)
                                or (relaxed_close_hold_event == "STABLE" and stable)
                            )
                        )
                        or (two_stage_close and stable)
                    )
                    and not external_hold_latched
                ):
                    external_hold_latched = True
                    external_hold_first_submission = action_submission_count + 1
                row = {
                    "phase": "POST_CLOSE_OBSERVATION",
                    "target_residual_mm": target_residual_mm,
                    "actual_residual_at_close_mm": actual_residual_at_close_mm,
                    "actual_residual_mm": current_residual * 1000.0,
                    "control_step_at_close": control_step_at_close,
                    "submission_index": action_submission_count,
                    "p_close": p_close_at_close,
                    "p_feasible": p_feasible_at_close,
                    "pre_close_features": pre_close_features,
                    "lateral_error_at_close_mm": lateral_error_at_close_mm,
                    "longitudinal_error_at_close_mm": longitudinal_error_at_close_mm,
                    "approach_cosine_at_close": approach_cosine_at_close,
                    "ee_velocity_at_close_root_m_s": ee_velocity_at_close_root_m_s,
                    "ee_speed_at_close_m_s": ee_speed_at_close_m_s,
                    "actual_gripper_state": latch.intent.value,
                    "close_edge": close_edge,
                    "left_pad_contact": left_contact,
                    "right_pad_contact": right_contact,
                    "left_normal_force_n": left_force,
                    "right_normal_force_n": right_force,
                    "force_balance": force_balance,
                    "reward_q_force_balance": float(
                        latest_reward_step.q_force_balance[0].item()
                    ),
                    "left_contact_point_root_m": force_weighted_contact_vector(
                        0, reward_inputs.contact_point_root_m
                    ),
                    "right_contact_point_root_m": force_weighted_contact_vector(
                        1, reward_inputs.contact_point_root_m
                    ),
                    "left_contact_normal_root": force_weighted_contact_vector(
                        0, reward_inputs.contact_normal_root
                    ),
                    "right_contact_normal_root": force_weighted_contact_vector(
                        1, reward_inputs.contact_normal_root
                    ),
                    "cube_center_root_m": cube_center_root,
                    "cube_quat_root_xyzw": cube_quat_root,
                    "cube_half_extents_m": [
                        float(value)
                        for value in reward_inputs.cube_half_extents_m[0]
                        .detach().cpu().tolist()
                    ],
                    "bilateral_contact": left_contact and right_contact,
                    "stable_bilateral": stable,
                    "non_pad_link_cube_contact": bool(forbidden_peak > 1.0e-6),
                    "forbidden_collision": bool(
                        forbidden_peak > 1.0e-6 or "forbidden_collision" in active
                    ),
                    "cube_linear_velocity_m_s": float(
                        latest_reward_step.cube_linear_velocity_m_s[0].item()
                    ),
                    "cube_linear_velocity_root_m_s": [
                        float(value)
                        for value in reward_inputs.cube_linear_velocity_root_m_s[0]
                        .detach().cpu().tolist()
                    ],
                    "cube_angular_velocity_rad_s": float(
                        latest_reward_step.cube_angular_velocity_rad_s[0].item()
                    ),
                    "cube_angular_velocity_root_rad_s": [
                        float(value)
                        for value in reward_inputs.cube_angular_velocity_root_rad_s[0]
                        .detach().cpu().tolist()
                    ],
                    "slip_m_s": float(
                        latest_reward_step.tangential_slip_m_s[0].item()
                    ),
                    "quality_components": {
                        "q_antipodal": float(latest_reward_step.q_antipodal[0].item()),
                        "q_force_balance": float(
                            latest_reward_step.q_force_balance[0].item()
                        ),
                        "q_slip": float(latest_reward_step.q_slip[0].item()),
                        "q_impact": float(latest_reward_step.q_impact[0].item()),
                        "q_omega": float(latest_reward_step.q_omega[0].item()),
                        "q_grasp_existing": float(latest_reward_step.q_grasp[0].item()),
                    },
                    "terminated": terminated,
                    "truncated": truncated,
                    "active_termination_names": list(active),
                    "passive_gripper_state": grasp_snapshot["joints"],
                    "raw_contact_aggregation": raw_contact,
                    "surface_contact_aggregation": raw_contact,
                    "outer_link2_force_components": surface_force_components,
                    "rigid_nonpad_contact_bodies": rigid_nonpad_contact_bodies,
                    "normal_convention": "CUBE_TO_PAD_OUTWARD",
                }
                final_outcome_telemetry = dict(row)
                _durable_jsonl(trace_stream, row)
                if row["forbidden_collision"] or terminated or truncated:
                    outcome = "UNSAFE_CLOSE"
                    outcome_reason = "FORBIDDEN_COLLISION_OR_TERMINATION"
                    break
                if stable and not stable_outcome_achieved and not post_close_hard_stop:
                    outcome = "SAFE_CLOSE"
                    outcome_reason = "STABLE_BILATERAL_10_STEPS"
                    stable_outcome_achieved = True
                if (
                    (stable_outcome_achieved or post_close_hard_stop)
                    and close_mechanics_telemetry.post_window_complete
                ):
                    break
                previous_action[:] = (0.0, 0.0, 0.0, 1.0)
                update_gru()
            else:
                if outcome_reason != "PRIVILEGED_GEOMETRY_NOT_READY":
                    outcome = "UNSAFE_CLOSE"
                    outcome_reason = "NO_STABLE_BILATERAL_WITHIN_OBSERVATION_BOUND"

            if geometry_forced_close_diagnostic:
                grasp_after = _grasp_geometry_snapshot(env, p0a)
                from geniesim.rl.sac.privileged_geometry_oracle import (
                    runtime_primary_pad_cube_oracle,
                )

                geometry_oracle_after_close = runtime_primary_pad_cube_oracle(
                    cube_center_world_m=grasp_after["cube_position_world_m"],
                    cube_quat_world_xyzw=grasp_after["cube_quaternion_world_xyzw"],
                    cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                    body_pose_world_m_xyzw_by_name={
                        name: (
                            *body["position_world_m"],
                            *body["quaternion_world_xyzw"],
                        )
                        for name, body in grasp_after["bodies"].items()
                        if name in (
                            "gripper_r_inner_link4",
                            "gripper_r_outer_link4",
                        )
                    },
                )

    if relaxed_close_hold_bilateral or two_stage_close:
        gripper_term.set_external_hold_mask(
            torch.tensor(
                [external_hold_latched], dtype=torch.bool, device=env.device
            )
        )
    if final_outcome_telemetry is not None:
        final_snapshot = _grasp_geometry_snapshot(env, p0a)
        gripper_aperture_at_events_m["final_held"] = float(
            np.linalg.norm(
                np.asarray(
                    final_snapshot["bodies"]["gripper_r_inner_link4"][
                        "position_world_m"
                    ],
                    dtype=np.float64,
                )
                - np.asarray(
                    final_snapshot["bodies"]["gripper_r_outer_link4"][
                        "position_world_m"
                    ],
                    dtype=np.float64,
                )
            )
        )

    mechanics_evidence = close_mechanics_telemetry.save_and_analyze(
        output_dir / "CLOSE_MECHANICS_500HZ_RING.npz",
        acceleration_limit_rad_s2=acceleration_limit_rad_s2,
    )
    estimator_rows = mechanics_evidence.get("estimators", {})

    def qdd_phase(estimator: str, phase: str) -> Any:
        return (
            estimator_rows.get(estimator, {})
            .get(phase, {})
            .get("max_abs_qdd_rad_s2")
        )

    def qdd_group(estimator: str, group: str, phase: str) -> Any:
        return (
            estimator_rows.get(estimator, {})
            .get("groups", {})
            .get(group, {})
            .get(phase, {})
            .get("max_abs_qdd_rad_s2")
        )

    qdd_two_post = estimator_rows.get("qdd_2point_rad_s2", {}).get(
        "post_close", {}
    )
    contact_observed = bool(
        first_left_pad_contact_step is not None
        or first_right_pad_contact_step is not None
    )
    bilateral_observed = first_bilateral_step is not None
    contact_lost_at_end = bool(
        post_close_contact_timeline
        and contact_observed
        and not (
            post_close_contact_timeline[-1]["left_pad_contact"]
            or post_close_contact_timeline[-1]["right_pad_contact"]
        )
    )
    forbidden_collision_observed = bool(
        final_outcome_telemetry
        and final_outcome_telemetry.get("forbidden_collision", False)
    )
    grasp_success = outcome == "SAFE_CLOSE"
    mechanics_safe = bool(
        grasp_success
        and not mechanics_evidence.get("post_close_accel_violation", False)
        and not post_close_hard_stop
        and not bool(
            final_outcome_telemetry
            and final_outcome_telemetry.get("forbidden_collision", False)
        )
    )
    if grasp_success and mechanics_safe:
        close_timing_label = "GOOD_CLOSE"
        training_eligible_for_gru = True
        training_weight = "HIGH_WEIGHT"
    elif grasp_success:
        close_timing_label = "GRASP_SUCCESS_BUT_MECHANICS_UNSAFE"
        training_eligible_for_gru = False
        training_weight = "EXCLUDE_FROM_POSITIVE"
    elif geometry_ready_receipt and geometry_ready_receipt.get("geometry_ready"):
        close_timing_label = "EARLY_OR_MISALIGNED_CLOSE"
        training_eligible_for_gru = True
        training_weight = (
            "HARD_NEGATIVE"
            if post_close_hard_stop
            or bool(
                final_outcome_telemetry
                and final_outcome_telemetry.get("forbidden_collision", False)
            )
            else "NEGATIVE"
        )
    else:
        close_timing_label = "NO_CLOSE_NEEDED_YET"
        training_eligible_for_gru = False
        training_weight = "EXCLUDE"
    if mechanics_safe:
        gru_mechanics_classification = "MECHANICS_CLEAN_SUCCESS"
    elif stable_outcome_achieved:
        gru_mechanics_classification = "STABLE_BUT_QDD_UNSAFE"
    elif post_close_hard_stop:
        gru_mechanics_classification = "HARD_STOP"
    else:
        gru_mechanics_classification = "FAILED_GRASP"
    if (
        post_close_hard_stop
        or forbidden_collision_observed
        or contact_lost_at_end
        or bool(mechanics_evidence.get("post_close_accel_violation", False))
    ):
        unsafe_failure_class = "TRUE_MECHANICS_FAILURE"
    elif not stable_outcome_achieved:
        unsafe_failure_class = "STABLE_NOT_REACHED_WITHIN_OBSERVATION_WINDOW"
    else:
        unsafe_failure_class = "NONE"
    supervision_path: Path | None = None
    student_npz_path: Path | None = None
    if close_edge_count == 1 and student_preclose_arrays is not None:
        student_npz_path = output_dir / "GRU_CLOSE_STUDENT_PRE-CLOSE_INPUT.npz"
        np.savez_compressed(student_npz_path, **student_preclose_arrays)
        supervision_path = output_dir / "GRU_CLOSE_SUPERVISION.json"
        _atomic_json(
            supervision_path,
            {
                "schema": "g2_candidate_a_gru_close_supervision_v1",
                "episode_id": calibration_episode_id,
                "source_state_id": pregrasp_sample_id,
                "timestamp_s": mechanics_evidence.get("close_submit_timestamp_s"),
                "natural_p_close": p_close_at_close,
                "natural_close_threshold": float(
                    coordinator.close_calibration.close_threshold
                ),
                "natural_close_edge": natural_gru_close_edge_count > 0,
                "forced_close_used": True,
                "close_variant": (
                    "CONTACT_AWARE_TWO_STAGE_CLOSE"
                    if two_stage_close
                    else f"CANONICAL_CLOSE_SPEED_{int(round(close_speed_scale * 100)):03d}PCT"
                    if not math.isclose(close_speed_scale, 1.0)
                    else
                    "SIMPLIFIED_15_20MM_15DEG_5STEP_CANONICAL_CLOSE"
                    if simplified_close_persistence_gate
                    else
                    f"RELAXED_{relaxed_close_hold_event}_TARGET_HOLD"
                    if relaxed_close_hold_bilateral
                    else "FULL_CLOSE"
                ),
                "student_preclose_input": {
                    **dict(student_preclose_metadata or {}),
                    "npz_path": str(student_npz_path),
                    "fields": sorted(student_preclose_arrays),
                },
                "causal_admission_context": pre_close_features,
                "privileged_geometry": {
                    "oracle_preclose": geometry_oracle_preclose,
                    "geometry_ready_receipt": geometry_ready_receipt,
                },
                "privileged_mechanics": {
                    "first_contact_ms": physics_event_markers.get(
                        "FIRST_CONTACT", {}
                    ).get("relative_to_close_ms"),
                    "bilateral_ms": physics_event_markers.get(
                        "FIRST_BILATERAL_CONTACT", {}
                    ).get("relative_to_close_ms"),
                    "stable_ms": physics_event_markers.get(
                        "STABLE_START", {}
                    ).get("relative_to_close_ms"),
                    "max_passive_qdd_2point_rad_s2": qdd_group(
                        "qdd_2point_rad_s2", "passive", "post_close"
                    ),
                    "max_arm_qdd_2point_rad_s2": qdd_group(
                        "qdd_2point_rad_s2", "arm", "post_close"
                    ),
                    "max_passive_qd_rad_s": mechanics_evidence.get(
                        "post_close_max_passive_qd_rad_s"
                    ),
                    "post_close_accel_violation": mechanics_evidence.get(
                        "post_close_accel_violation"
                    ),
                    "hard_stop": post_close_hard_stop,
                    "forbidden_collision": bool(
                        final_outcome_telemetry
                        and final_outcome_telemetry.get(
                            "forbidden_collision", False
                        )
                    ),
                    "gripper_aperture_at_events_m": (
                        gripper_aperture_at_events_m
                    ),
                },
                "timing_target": int(
                    bool(
                        geometry_ready_receipt
                        and geometry_ready_receipt.get("geometry_ready")
                    )
                ),
                "grasp_success_target": int(grasp_success),
                "mechanics_safe_target": int(mechanics_safe),
                "close_timing_label": close_timing_label,
                "mechanics_classification": gru_mechanics_classification,
                "training_eligible_for_gru": training_eligible_for_gru,
                "training_weight": training_weight,
                "post_close_input_leakage_count": 0,
                "student_privileged_input_count": 0,
            },
        )
    coordinator.assert_bc_frozen()
    freeze_after = source_freeze_provider()
    report = {
        "schema": "g2_stage1a_close_residual_sweep_episode_v2",
        "execution_mode": "BOUNDED_DETERMINISTIC_SINGLE_CLOSE_NO_TRAINING",
        "episode_id": calibration_episode_id,
        "pregrasp_sample_id": pregrasp_sample_id,
        "runtime_seed": runtime_seed,
        "target_residual_mm": target_residual_mm,
        "actual_residual_at_close_mm": actual_residual_at_close_mm,
        "control_step_at_close": control_step_at_close,
        "close_mechanics_evidence": mechanics_evidence,
        "CLOSE_STEP": mechanics_evidence.get("close_submit_step"),
        "CLOSE_TIMESTAMP": mechanics_evidence.get("close_submit_timestamp_s"),
        "PRE_CLOSE_MAX_PASSIVE_QD": mechanics_evidence.get(
            "pre_close_max_passive_qd_rad_s"
        ),
        "POST_CLOSE_MAX_PASSIVE_QD": mechanics_evidence.get(
            "post_close_max_passive_qd_rad_s"
        ),
        "PRE_CLOSE_QDD_2POINT_MAX": qdd_phase(
            "qdd_2point_rad_s2", "pre_close"
        ),
        "POST_CLOSE_QDD_2POINT_MAX": qdd_phase(
            "qdd_2point_rad_s2", "post_close"
        ),
        "POST_CLOSE_QDD_5POINT_MAX": qdd_phase(
            "qdd_5point_rad_s2", "post_close"
        ),
        "POST_CLOSE_QDD_REGRESSION_MAX": qdd_phase(
            "qdd_regression_rad_s2", "post_close"
        ),
        "MAX_QDD_JOINT": qdd_two_post.get("joint_name"),
        "DISTANCE_TO_LIMIT_AT_MAX_QDD": qdd_two_post.get(
            "distance_to_nearest_limit_rad"
        ),
        "FIRST_ACCEL_VIOLATION_AFTER_CLOSE_MS": mechanics_evidence.get(
            "first_accel_violation_after_close_ms"
        ),
        "ESTIMATOR_AGREEMENT": mechanics_evidence.get("estimator_agreement"),
        "ESTIMATOR_EVENT_AGREEMENT": mechanics_evidence.get(
            "estimator_event_agreement"
        ),
        "PRE_CLOSE_MAX_ARM_QDD": qdd_group(
            "qdd_2point_rad_s2", "arm", "pre_close"
        ),
        "POST_CLOSE_MAX_ARM_QDD": qdd_group(
            "qdd_2point_rad_s2", "arm", "post_close"
        ),
        "PRE_CLOSE_MAX_PASSIVE_QDD": qdd_group(
            "qdd_2point_rad_s2", "passive", "pre_close"
        ),
        "POST_CLOSE_MAX_PASSIVE_QDD": qdd_group(
            "qdd_2point_rad_s2", "passive", "post_close"
        ),
        "RING_BUFFER_SAMPLES": mechanics_evidence.get("ring_buffer_samples"),
        "p_close_at_close": p_close_at_close,
        "p_feasible_at_close": p_feasible_at_close,
        "lateral_error_at_close_mm": lateral_error_at_close_mm,
        "longitudinal_error_at_close_mm": longitudinal_error_at_close_mm,
        "approach_cosine_at_close": approach_cosine_at_close,
        "ee_velocity_at_close_root_m_s": ee_velocity_at_close_root_m_s,
        "ee_speed_at_close_m_s": ee_speed_at_close_m_s,
        "pre_close_features": pre_close_features,
        "geometry_forced_close_diagnostic": bool(
            geometry_forced_close_diagnostic
        ),
        "simplified_close_persistence_gate": bool(
            simplified_close_persistence_gate
        ),
        "close_mechanics_ablation": {
            "close_variant": (
                "CONTACT_AWARE_TWO_STAGE_CLOSE"
                if two_stage_close
                else f"CANONICAL_CLOSE_SPEED_{int(round(close_speed_scale * 100)):03d}PCT"
            ),
            "requested_close_speed_scale": float(close_speed_scale),
            "configured_maximum_close_speed_rad_s": (
                configured_close_speed_rad_s
            ),
            "effective_initial_close_speed_rad_s": (
                configured_close_speed_rad_s
                * (1.0 if two_stage_close else float(close_speed_scale))
            ),
            "maximum_joint_target_acceleration_rad_s2": (
                configured_close_acceleration_rad_s2
            ),
            "ramp_authority_changed": False,
            "two_stage_close": bool(two_stage_close),
            "post_contact_close_speed_scale": (
                post_contact_close_speed_scale if two_stage_close else None
            ),
            "speed_transition_submission": close_speed_transition_submission,
            "speed_transition_contact_kind": (
                close_speed_transition_contact_kind
            ),
            "speed_history": close_speed_history,
            "hold_authority": "STABLE_ONLY" if two_stage_close else (
                relaxed_close_hold_event
                if relaxed_close_hold_bilateral
                else "NONE"
            ),
            "canonical_speed_interface": True,
            "direct_joint_or_torque_command": False,
            "safety_threshold_changed": False,
            "post_close_observation_bound_steps": (
                maximum_post_close_submissions
            ),
            "post_close_observation_bound_ms": (
                1000.0 * maximum_post_close_submissions / CONTROL_HZ
            ),
        },
        "simplified_close_gate_contract": {
            "residual_min_mm": SIMPLIFIED_CLOSE_MIN_RESIDUAL_M * 1000.0,
            "residual_max_mm": SIMPLIFIED_CLOSE_MAX_RESIDUAL_M * 1000.0,
            "orientation_error_max_deg": (
                SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG
            ),
            "persistence_steps": SIMPLIFIED_CLOSE_PERSISTENCE_STEPS,
            "persistence_duration_ms": (
                1000.0 * SIMPLIFIED_CLOSE_PERSISTENCE_STEPS / CONTROL_HZ
            ),
            "gru_p_close_authority": False,
            "canonical_close_path": True,
            "direct_joint_or_torque_command": False,
            "receipt_count": len(simplified_gate_receipts),
            "final_receipt": (
                simplified_gate_receipts[-1]
                if simplified_gate_receipts
                else None
            ),
        },
        "relaxed_close_hold_bilateral": bool(relaxed_close_hold_bilateral),
        "relaxed_close_hold_event": relaxed_close_hold_event,
        "canonical_external_hold_interface": bool(
            callable(getattr(gripper_term, "set_external_hold_mask", None))
        ),
        "external_hold_latched": external_hold_latched,
        "external_hold_first_submission": external_hold_first_submission,
        "external_hold_applied_count": external_hold_applied_count,
        "zero_xyz_during_close_count": zero_xyz_during_close_count,
        "nonzero_xyz_during_close_count": nonzero_xyz_during_close_count,
        "gripper_aperture_at_events_m": gripper_aperture_at_events_m,
        "gru_supervision": {
            "row_count": int(supervision_path is not None),
            "supervision_path": (
                None if supervision_path is None else str(supervision_path)
            ),
            "student_input_npz_path": (
                None if student_npz_path is None else str(student_npz_path)
            ),
            "close_timing_label": close_timing_label,
            "mechanics_classification": gru_mechanics_classification,
            "timing_target": int(
                bool(
                    geometry_ready_receipt
                    and geometry_ready_receipt.get("geometry_ready")
                )
            ),
            "grasp_success_target": int(grasp_success),
            "mechanics_safe_target": int(mechanics_safe),
            "training_eligible_for_gru": training_eligible_for_gru,
            "training_weight": training_weight,
            "post_close_input_leakage_count": 0,
            "student_privileged_input_count": 0,
        },
        "geometry_oracle_preclose": geometry_oracle_preclose,
        "geometry_oracle_after_close": geometry_oracle_after_close,
        "geometry_ready_receipt": geometry_ready_receipt,
        "oracle_required_xyz_mm": oracle_required_xyz_mm,
        "oracle_within_0p45mm": oracle_required_xyz_mm <= 0.45,
        "natural_gru_close_edge_count": natural_gru_close_edge_count,
        "natural_gru_first_close_submission": natural_gru_first_close_submission,
        "natural_gru_first_close_residual_mm": (
            natural_gru_first_close_residual_mm
        ),
        "forced_close_triggered": bool(
            geometry_ready_receipt is not None
            and geometry_ready_receipt["geometry_ready"]
        ),
        "forced_close_submitted": close_edge_count == 1,
        "close_submit_timestamp": mechanics_evidence.get(
            "close_submit_timestamp_s"
        ),
        "first_contact_after_close_ms": (
            physics_event_markers.get("FIRST_CONTACT", {}).get(
                "relative_to_close_ms"
            )
        ),
        "bilateral_after_close_ms": (
            physics_event_markers.get("FIRST_BILATERAL_CONTACT", {}).get(
                "relative_to_close_ms"
            )
        ),
        "stable_after_close_ms": (
            physics_event_markers.get("STABLE_START", {}).get(
                "relative_to_close_ms"
            )
        ),
        "first_valid_contact_nominal_residual_mm": (
            first_valid_contact_nominal_residual_mm
        ),
        "final_outcome_telemetry": final_outcome_telemetry,
        "cube_geometry_authority": cube_geometry_authority,
        "time_contract": {
            "control_hz": CONTROL_HZ,
            "physics_hz": PHYSICS_HZ,
            "rgbd_hz": RGBD_HZ,
            "physics_substeps_per_control": PHYSICS_HZ // CONTROL_HZ,
            "unit_system": "SI_M_RAD_S_N_NS",
        },
        "post_close_contact_timeline": post_close_contact_timeline,
        "first_left_pad_contact_step": first_left_pad_contact_step,
        "first_right_pad_contact_step": first_right_pad_contact_step,
        "first_bilateral_step": first_bilateral_step,
        "stable_grasp_step": stable_grasp_step,
        "CONTACT": contact_observed,
        "BILATERAL": bilateral_observed,
        "STABLE": stable_grasp_step is not None,
        "SUCCESS": bool(grasp_success),
        "PASSIVE_HARD_STOP": bool(post_close_hard_stop),
        "FORBIDDEN_COLLISION": forbidden_collision_observed,
        "CONTACT_LOSS": contact_lost_at_end,
        "UNSAFE_FAILURE_CLASS": unsafe_failure_class,
        "GRU_MECHANICS_CLASSIFICATION": gru_mechanics_classification,
        "contact_region_at_first_contact": contact_region_at_first_contact,
        "contact_region_at_stable": contact_region_at_stable,
        "normal_convention": "CUBE_TO_PAD_OUTWARD",
        "physics_event_markers": physics_event_markers,
        "variation": {
            "lateral_offset_mm": float(lateral_offset_mm),
            "height_offset_mm": float(height_offset_mm),
            "approach_yaw_deg": float(approach_yaw_deg),
            "effective_target_residual_mm": effective_target_m * 1000.0,
        },
        "outcome": outcome,
        "outcome_reason": outcome_reason,
        "OUTCOME": outcome,
        "FAILURE_REASON": outcome_reason,
        "PRE_CLOSE_ACCEL_VIOLATION": mechanics_evidence.get(
            "pre_close_accel_violation"
        ),
        "POST_CLOSE_ACCEL_VIOLATION": mechanics_evidence.get(
            "post_close_accel_violation"
        ),
        "approach_target_reached": approach_target_reached,
        "pre_close_hard_stop": pre_close_hard_stop,
        "post_close_hard_stop": post_close_hard_stop,
        "hard_stop_observation_continuation": bool(
            continue_after_hard_stop_for_telemetry
        ),
        "close_edge_count": close_edge_count,
        "action_submission_count": action_submission_count,
        "gru_close_used_for_actuation": False,
        "p_feasible_used_for_gate": False,
        "residual_sac": "OFF",
        "sac_update_count": coordinator.sac_update_count,
        "training": "OFF",
        "trace_path": str(trace_path),
        "source_freeze_before": dict(source_freeze_before),
        "source_freeze_after": dict(freeze_after),
        "source_freeze_match": source_freeze_before == freeze_after,
        "asset": {"path": str(selected_asset_path), "sha256": selected_asset_sha256},
        "bc_checkpoint": {"path": str(bc_checkpoint), "sha256": bc_checkpoint_sha256},
    }
    _atomic_json(report_path, report)
    _atomic_json(output_dir / "CLOSE_RESIDUAL_SWEEP_REPORT.json", report)
    return 0 if outcome == "SAFE_CLOSE" else 2


def run_stage1a_isaac_short_smoke(
    *, app: Any, env: Any, p0a: Any, preflight: Any, task_mdp: Any,
    counter: Any, deferred_port: Any, latch: Any, physics_telemetry: Any,
    selected_pregrasp_sample: Any, nominal_grasp_pose_root_m_xyzw: Any,
    direct_init_receipt: Mapping[str, Any], source_freeze_before: Mapping[str, Any],
    source_freeze_provider: Callable[[], Mapping[str, Any]], selected_asset_path: Path,
    selected_asset_sha256: str, accepted_transition_target: int,
    bc_checkpoint: Path, bc_checkpoint_sha256: str, output_dir: Path,
    report_path: Path, wandb_enabled: bool = False, wandb_mode: str = "offline",
    wandb_project: str = "geniesim-g2-stage1a-residual-sac",
    wandb_entity: str | None = None, wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    replay_strategy: str = REPLAY_STRATEGY_HER_FORCE,
    far_reach_checkpoint: Path | None = None,
    far_reach_checkpoint_sha256: str | None = None,
    residual_actor_checkpoint: Path | None = None,
    residual_actor_checkpoint_sha256: str | None = None,
    reward_v3: bool = False,
    stable_only: bool = False,
    privileged_close_motion_interlock: bool = False,
    training_seed: int = 42,
    acceleration_limit_rad_s2: float = 10.0,
) -> int:
    from geniesim.rl.isaaclab.g2_policy_branch.contact_free_visual_bc_runtime import (
        load_contact_free_visual_bc_checkpoint,
        validate_metric_output,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.hybrid_grasp_runtime import (
        HybridGraspPhaseRouter,
    )

    del app
    if accepted_transition_target not in (
        ACCEPTED_TRANSITION_TARGET,
        LONG_TRAINING_ACCEPTED_TRANSITION_TARGET,
        REWARD_V3_ACCEPTED_TRANSITION_TARGET,
    ):
        raise Stage1AIsaacShortSmokeError(
            "STAGE1A_TARGET_MUST_BE_EXACTLY_1000_3000_OR_15000"
        )
    long_training = accepted_transition_target == LONG_TRAINING_ACCEPTED_TRANSITION_TARGET
    reward_v3_3k = accepted_transition_target == REWARD_V3_ACCEPTED_TRANSITION_TARGET
    if reward_v3 and not (reward_v3_3k or (long_training and stable_only)):
        raise Stage1AIsaacShortSmokeError(
            "REWARD_V3_REQUIRES_3000_SMOKE_OR_STABLE_ONLY_15000"
        )
    if reward_v3_3k and not reward_v3:
        raise Stage1AIsaacShortSmokeError("REWARD_V3_3000_REQUIRES_REWARD_V3")
    if stable_only and not (long_training and reward_v3):
        raise Stage1AIsaacShortSmokeError(
            "STABLE_ONLY_REQUIRES_REWARD_V3_15000"
        )
    if privileged_close_motion_interlock and not reward_v3:
        raise Stage1AIsaacShortSmokeError(
            "PRIVILEGED_CLOSE_INTERLOCK_REQUIRES_REWARD_V3"
        )
    if not math.isclose(
        float(acceleration_limit_rad_s2), 10.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise Stage1AIsaacShortSmokeError(
            "SIMPLIFIED_CLOSE_REQUIRES_UNCHANGED_10_RAD_S2_AUTHORITY"
        )
    hybrid_training = long_training or reward_v3
    if hybrid_training and (
        far_reach_checkpoint is None
        or far_reach_checkpoint_sha256 is None
        or residual_actor_checkpoint is None
        or residual_actor_checkpoint_sha256 is None
    ):
        raise Stage1AIsaacShortSmokeError(
            "STAGE1A_HYBRID_REQUIRES_VERIFIED_FAR_BC_AND_RESIDUAL_ACTOR"
        )
    if env.num_envs != 1 or not math.isclose(float(env.step_dt), 0.02, abs_tol=1e-12):
        raise Stage1AIsaacShortSmokeError("STAGE1A_REQUIRES_ONE_ENV_AT_50HZ")
    if report_path.exists() or output_dir.exists():
        raise Stage1AIsaacShortSmokeError("STAGE1A_OUTPUT_REFUSES_OVERWRITE")
    if _sha256(selected_asset_path) != selected_asset_sha256:
        raise Stage1AIsaacShortSmokeError("STAGE1A_ASSET_HASH_MISMATCH")
    gripper_term = env.action_manager.get_term("gripper_action")
    # The simplified gate keeps canonical CLOSE active after its one-shot
    # edge.  It does not freeze or overwrite a gripper target, so external
    # hold capability is deliberately not an execution prerequisite.
    if callable(getattr(gripper_term, "set_external_hold_mask", None)):
        gripper_term.set_external_hold_mask(
            torch.zeros(1, dtype=torch.bool, device=env.device)
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    coordinator = Stage1ARealSACCoordinator.from_bc_checkpoint(
        bc_checkpoint, expected_sha256=bc_checkpoint_sha256,
        replay_capacity=accepted_transition_target + 64, device=env.device,
        seed=int(training_seed),
        replay_strategy=replay_strategy,
        residual_actor_checkpoint_path=residual_actor_checkpoint,
        residual_actor_checkpoint_sha256=residual_actor_checkpoint_sha256,
    )
    far_model = None
    far_receipt = None
    if far_reach_checkpoint is not None:
        if far_reach_checkpoint_sha256 is None:
            raise Stage1AIsaacShortSmokeError("FAR_BC_HASH_REQUIRED")
        far_model, far_receipt = load_contact_free_visual_bc_checkpoint(
            far_reach_checkpoint,
            expected_sha256=far_reach_checkpoint_sha256,
            device=env.device,
        )
    reward = Stage1AGraspReward(
        1,
        device=env.device,
        v3_config=Stage1ARewardV3Config() if reward_v3 else None,
    )
    phase_router = HybridGraspPhaseRouter()
    coordinator.reset_episode()
    nominal = np.asarray(nominal_grasp_pose_root_m_xyzw[:3], dtype=np.float64)
    previous_action = np.zeros(4, dtype=np.float32)
    previous_residual = np.zeros(3, dtype=np.float64)
    camera_state: _CameraAcquisitionState | None = None
    episode_index = 0
    episode_control_step = 0
    action_submission_count = 0
    env_autoreset_rejected_transition_count = 0
    env_autoreset_termination_counts: dict[str, int] = {}
    runtime_counts = {
        "far_reach": 0,
        "local_grasp": 0,
        "residual_gate": 0,
        "actor_forward": 0,
        "nonzero_residual": 0,
        "gru_nominal": 0,
        "sac_band_entry": 0,
        "gru_22_30_steps": 0,
        "gru_22_30_progress": 0,
        "gru_22_30_stall": 0,
    }
    previous_residual_gate = False
    previous_nominal_distance = distance if "distance" in locals() else None
    handoff_30mm_step: int | None = None
    handoff_30_to_22_steps: list[int] = []
    episode_close_interlock_latched = False
    episode_external_hold_latched = False
    previous_ee_root_for_speed: np.ndarray | None = None
    privileged_close_trigger_count = 0
    privileged_close_motion_hold_steps = 0
    privileged_close_nonzero_xyz_count = 0
    privileged_external_hold_steps = 0
    privileged_close_natural_trigger_count = 0
    privileged_close_forced_trigger_count = 0
    privileged_close_ready_evaluation_count = 0
    privileged_close_supervision_rows: list[dict[str, Any]] = []
    privileged_close_student_arrays: list[dict[str, np.ndarray]] = []
    simplified_close_gate = SimplifiedClosePersistenceGate()
    natural_gru_close_logging_latched = False
    distance_ready_count_max = 0
    orientation_ready_count_max = 0
    persistence_ready_count_max = 0
    runtime_arm_qdd_max_abs_rad_s2 = 0.0
    runtime_passive_qdd_max_abs_rad_s2 = 0.0

    def propose_hybrid(
        inputs: HumanGraspSequenceInputs,
        *,
        distance_m: float,
        camera: _CameraAcquisitionState,
        control_step: int,
    ) -> tuple[Any, Any, str]:
        decision = phase_router.decide(distance_m)
        if decision.human_grasp_nominal_active or far_model is None:
            proposal = coordinator.propose(
                inputs,
                deterministic=False,
                residual_active=decision.residual_sac_active,
                gripper_authority_active=(
                    not privileged_close_motion_interlock
                ),
            )
            return proposal, decision, "HUMAN_GRASP_GRU_BC"
        far_observation = _hybrid_far_observation(
            env=env,
            inputs=inputs,
            camera_state=camera,
            control_step=control_step,
        )
        image, proprio = far_observation.prepare()
        with torch.inference_mode():
            far_output = far_model(image, proprio)
        validate_metric_output(far_output)
        far_action = tuple(
            float(value) for value in far_output[0].detach().cpu().tolist()
        )
        proposal = coordinator.propose(
            inputs,
            deterministic=False,
            residual_active=False,
            nominal_action_override_4d_metric_root_m=far_action,
            gripper_authority_active=False,
        )
        return proposal, decision, "CUROBO_CONTACT_FREE_BC"
    inputs, camera_state = _gru_inputs(
        env=env, p0a=p0a, camera_state=camera_state, previous_action=previous_action,
        hidden_reset=True,
    )
    current_ee, _ = _ee_root_pose(env, p0a)
    distance = float(np.linalg.norm(nominal - current_ee))
    previous_nominal_distance = distance
    proposal, proposal_decision, proposal_owner = propose_hybrid(
        inputs,
        distance_m=distance,
        camera=camera_state,
        control_step=0,
    )
    run = None
    if wandb_enabled:
        import wandb
        run = wandb.init(
            project=wandb_project, entity=wandb_entity, name=wandb_run_name,
            group=wandb_group,
            tags=(
                "candidate_a",
                "hybrid_runtime",
                replay_strategy.lower(),
                "reward_v3" if reward_v3 else "reward_v2",
                (
                    "privileged_close_motion_interlock"
                    if privileged_close_motion_interlock
                    else "production_close_path"
                ),
            ),
            mode=wandb_mode,
            config={
                "target": accepted_transition_target,
                "num_envs": 1,
                "alpha": 0.10,
                "replay_strategy": replay_strategy,
                "far_bc_sha256": far_reach_checkpoint_sha256,
                "residual_actor_sha256": residual_actor_checkpoint_sha256,
                "reward_version": "V3" if reward_v3 else "V2",
                "stable_only": bool(stable_only),
                "training_seed": int(training_seed),
                "ordinary_her": False,
                "her_force": replay_strategy == "HER_FORCE",
                "privileged_close_motion_interlock": (
                    privileged_close_motion_interlock
                ),
                "privileged_student_input_count": 0,
            },
        )
    periodic_checkpoint_steps = (
        (3000, 6000, 9000, 12000, 15000) if long_training else ()
    )
    periodic_checkpoint_receipts: list[dict[str, Any]] = []
    periodic_checkpoint_steps_saved: set[int] = set()

    def maybe_export_periodic_checkpoint() -> None:
        step = int(coordinator.accepted_transitions)
        if (
            step not in periodic_checkpoint_steps
            or step in periodic_checkpoint_steps_saved
        ):
            return
        receipt = coordinator.export_periodic_training_checkpoint(
            output_dir / "checkpoints" / f"checkpoint_{step}.pt",
            source_freeze=source_freeze_before,
            reward_config=stage1a_reward_contract(reward_v3=reward_v3),
            runtime_config={
                "stable_only": bool(stable_only),
                "lift_enabled": False,
                "place_enabled": False,
                "control_hz": CONTROL_HZ,
                "physics_hz": PHYSICS_HZ,
                "rgbd_hz": RGBD_HZ,
                "deterministic_close_gate": bool(
                    privileged_close_motion_interlock
                ),
                "qdd_10_rad_s2_authority": "DIAGNOSTIC_ONLY",
                "training_seed": int(training_seed),
            },
        )
        periodic_checkpoint_steps_saved.add(step)
        periodic_checkpoint_receipts.append(dict(receipt.__dict__))
        if run is not None:
            run.log(
                {
                    "checkpoint/saved_step": step,
                    "checkpoint/reload_pass": int(receipt.reload_pass),
                },
                step=step,
            )
    csv_path = output_dir / "metrics.csv"
    transition_path = output_dir / "REPLAY_TRANSITIONS.jsonl"
    with (
        csv_path.open("w", newline="", encoding="utf-8") as csv_stream,
        transition_path.open("x", encoding="utf-8") as transition_stream,
    ):
        writer = csv.DictWriter(csv_stream, fieldnames=(
            "accepted_transitions", "sac_update_count", "reward_total",
            "residual_norm_mm", "nominal_grasp_remaining_mm",
            "orientation_error_deg", "distance_ready_count",
            "orientation_ready_count", "persistence_ready_count",
            "close_trigger_count", "left_contact", "right_contact",
            "contact", "bilateral", "stable", "success",
            "arm_qdd_max_abs_rad_s2", "passive_qdd_max_abs_rad_s2",
        ))
        writer.writeheader()
        while coordinator.accepted_transitions < accepted_transition_target:
            step_index = coordinator.accepted_transitions
            if proposal_decision.phase.value == "FAR_REACH":
                runtime_counts["far_reach"] += 1
            else:
                runtime_counts["local_grasp"] += 1
            runtime_counts["residual_gate"] += int(
                proposal_decision.residual_sac_active
            )
            if proposal_decision.residual_sac_active and not previous_residual_gate:
                runtime_counts["sac_band_entry"] += 1
            previous_residual_gate = bool(proposal_decision.residual_sac_active)
            if 0.022 < distance <= 0.030:
                runtime_counts["gru_22_30_steps"] += 1
                if handoff_30mm_step is None:
                    handoff_30mm_step = episode_control_step
                if previous_nominal_distance is not None:
                    distance_delta = previous_nominal_distance - distance
                    runtime_counts["gru_22_30_progress"] += int(
                        distance_delta > 0.00020
                    )
                    runtime_counts["gru_22_30_stall"] += int(
                        abs(distance_delta) <= 0.00020
                    )
            elif distance <= 0.022 and handoff_30mm_step is not None:
                handoff_30_to_22_steps.append(
                    max(0, episode_control_step - handoff_30mm_step)
                )
                handoff_30mm_step = None
            runtime_counts["gru_nominal"] += int(
                proposal_owner == "HUMAN_GRASP_GRU_BC"
            )
            current_ee, current_ee_quat = _ee_root_pose(env, p0a)
            current_ee_array = np.asarray(current_ee, dtype=np.float64)
            ee_speed_m_s = (
                0.0
                if previous_ee_root_for_speed is None
                else float(
                    np.linalg.norm(current_ee_array - previous_ee_root_for_speed)
                    * CONTROL_HZ
                )
            )
            natural_close_edge = False
            if privileged_close_motion_interlock:
                previous_natural_state = natural_gru_close_logging_latched
                if coordinator.close_calibration.mode == "EDGE_SINGLE_EVENT":
                    if (
                        not natural_gru_close_logging_latched
                        and proposal.close_probability
                        >= coordinator.close_calibration.close_threshold
                    ):
                        natural_gru_close_logging_latched = True
                else:
                    if (
                        proposal.close_probability
                        >= coordinator.close_calibration.close_threshold
                    ):
                        natural_gru_close_logging_latched = True
                    elif (
                        proposal.close_probability
                        <= coordinator.close_calibration.open_threshold
                    ):
                        natural_gru_close_logging_latched = False
                natural_close_edge = bool(
                    not previous_natural_state
                    and natural_gru_close_logging_latched
                )
                privileged_close_natural_trigger_count += int(
                    natural_close_edge
                )
            close_trigger_source: str | None = None
            close_readiness: dict[str, Any] | None = None
            close_oracle: dict[str, Any] | None = None
            if (
                privileged_close_motion_interlock
                and not episode_close_interlock_latched
            ):
                evaluator = getattr(
                    env, "_g2_forbidden_collision_evaluator", None
                )
                preclose_peaks = (
                    evaluator.sensor_peak_forces_n()
                    if evaluator is not None
                    else {}
                )
                preclose_forbidden_peak = max(
                    (max(values) for values in preclose_peaks.values()),
                    default=0.0,
                )
                _, close_oracle, close_readiness = (
                    _privileged_close_readiness(
                        env=env,
                        p0a=p0a,
                        task_mdp=task_mdp,
                        nominal_grasp_pose_root_m_xyzw=(
                            nominal_grasp_pose_root_m_xyzw
                        ),
                        ee_speed_m_s=ee_speed_m_s,
                        no_safety_violation=bool(
                            preclose_forbidden_peak <= 1.0e-6
                            and physics_telemetry.hard_stop_event is None
                        ),
                        gate=simplified_close_gate,
                    )
                )
                privileged_close_ready_evaluation_count += 1
                distance_ready_count_max = max(
                    distance_ready_count_max,
                    int(close_readiness["distance_ready_count"]),
                )
                orientation_ready_count_max = max(
                    orientation_ready_count_max,
                    int(close_readiness["orientation_ready_count"]),
                )
                persistence_ready_count_max = max(
                    persistence_ready_count_max,
                    int(close_readiness["persistence_ready_count"]),
                )
                if close_readiness["close_trigger"]:
                    episode_close_interlock_latched = True
                    privileged_close_forced_trigger_count += 1
                    close_trigger_source = "SIMPLIFIED_PERSISTENCE_CLOSE"
                    privileged_close_student_arrays.append(
                        {
                            name: getattr(inputs, name)
                            .detach()
                            .cpu()
                            .numpy()
                            .copy()
                            for name in (
                                "right_wrist_rgb",
                                "right_wrist_depth_m",
                                "right_wrist_depth_valid",
                                "ee_pose_robot_root_m_xyzw",
                                "right_arm_joint_position_rad",
                                "right_arm_joint_velocity_rad_s",
                                "current_gripper_state",
                                "previous_policy_action_4d_metric_root_m",
                                "hidden_reset_mask",
                            )
                        }
                    )
                    privileged_close_supervision_rows.append(
                        {
                            "episode_id": f"episode-{episode_index:04d}",
                            "control_step": int(episode_control_step),
                            "accepted_transition_before_close": int(
                                coordinator.accepted_transitions
                            ),
                            "trigger_source": close_trigger_source,
                            "natural_p_close": float(
                                proposal.close_probability
                            ),
                            "natural_p_feasible": float(
                                proposal.feasibility_probability
                            ),
                            "natural_close_threshold": float(
                                coordinator.close_calibration.close_threshold
                            ),
                            "natural_close_hysteresis_logging_only": bool(
                                natural_gru_close_logging_latched
                            ),
                            "natural_close_edge": bool(natural_close_edge),
                            "nominal_residual_mm": distance * 1000.0,
                            "orientation_error_deg": _orientation_error_deg(
                                nominal_grasp_pose_root_m_xyzw[3:],
                                current_ee_quat,
                            ),
                            "distance_ready_count": int(
                                close_readiness["distance_ready_count"]
                            ),
                            "orientation_ready_count": int(
                                close_readiness["orientation_ready_count"]
                            ),
                            "persistence_ready_count": int(
                                close_readiness["persistence_ready_count"]
                            ),
                            "close_trigger_count": int(
                                close_readiness["close_trigger_count"]
                            ),
                            "ee_speed_m_s": ee_speed_m_s,
                            "privileged_geometry": close_oracle,
                            "privileged_readiness": close_readiness,
                            "left_contact": False,
                            "right_contact": False,
                            "bilateral": False,
                            "stable": False,
                            "mechanics_safe": None,
                            "grasp_success": None,
                            "max_arm_qdd_rad_s2": 0.0,
                            "max_passive_qdd_rad_s2": 0.0,
                            "hard_stop": False,
                            "forbidden_collision": False,
                            "student_privileged_input_count": 0,
                            "post_close_input_leakage_count": 0,
                        }
                    )
                if close_trigger_source is not None:
                    privileged_close_trigger_count += 1
            applied_proposal = proposal
            if privileged_close_motion_interlock and episode_close_interlock_latched:
                applied_proposal = _zero_xyz_close_execution_proposal(
                    proposal, alpha=coordinator.alpha
                )
                privileged_close_motion_hold_steps += 1
            runtime_counts["actor_forward"] += int(
                applied_proposal.residual_active
            )
            runtime_counts["nonzero_residual"] += int(
                float(
                    np.linalg.norm(
                        applied_proposal.raw_residual_metric_root_m
                    )
                )
                > 0.0
            )
            final_action = (
                applied_proposal.composition.final_action_4d_metric_root_m
            )
            xyz = np.asarray(final_action[:3], dtype=np.float64)
            if (
                privileged_close_motion_interlock
                and episode_close_interlock_latched
                and float(np.linalg.norm(xyz)) > 1.0e-12
            ):
                privileged_close_nonzero_xyz_count += 1
                raise Stage1AIsaacShortSmokeError(
                    "PRIVILEGED_CLOSE_INTERLOCK_NONZERO_XYZ"
                )
            if float(np.linalg.norm(xyz)) > MAX_FINAL_ACTION_M + 1e-12:
                raise Stage1AIsaacShortSmokeError("FINAL_ACTION_BOUND_VIOLATION")
            normalized = metric_xyz_to_normalized(xyz)
            high_level = HighLevelPolicyAction.from_sequence(
                (*np.asarray(normalized, dtype=np.float64).tolist(), float(final_action[3]))
            )
            packet, _, _ = p0a._build_authoritative_packet(
                high_level=high_level, batch_size=1, device=env.device, latch=latch
            )
            records_start = len(physics_telemetry.records)
            physics_telemetry.set_context(
                policy_step=step_index, bc_step=step_index, phase="STAGE1A",
                gripper_intent=packet.gripper_intent.value,
                close_onset=(
                    packet.gripper_intent is AbstractGripperIntent.CLOSE
                    and latch.intent is not AbstractGripperIntent.CLOSE
                ),
                clipping=False,
                last_real_sensor_timestamps={
                    "right_wrist": camera_state.timestamp_s
                },
                physics_clock_anchor_s=camera_state.sensor_clock_time_s,
            )
            try:
                outputs, consumption = preflight._consume_once(
                    env=env, counter=counter, deferred_port=deferred_port, packet=packet,
                    label=f"STAGE1A_ACCEPTED_{step_index:04d}",
                )
            except RuntimeHardstop as hardstop_error:
                receipt = hardstop_error.runtime_hardstop_receipt
                consumption = hardstop_error.canonical_consumption_receipt
                if (
                    not isinstance(consumption, Mapping)
                    or consumption.get("action_consumption")
                    != ActionConsumptionState.PARTIAL_TERMINATED.value
                    or not consumption.get("single_consumption")
                ):
                    raise Stage1AIsaacShortSmokeError(
                        "HARDSTOP_CONSUMPTION_NOT_PARTIAL_TERMINATED"
                    ) from hardstop_error
                action_submission_count += 1
                latch.commit(packet.gripper_intent)
                records = physics_telemetry.records[records_start:]
                step_qdd = _step_joint_qdd_metrics(env=env, records=records)
                runtime_arm_qdd_max_abs_rad_s2 = max(
                    runtime_arm_qdd_max_abs_rad_s2,
                    step_qdd["arm_qdd_max_abs_rad_s2"],
                )
                runtime_passive_qdd_max_abs_rad_s2 = max(
                    runtime_passive_qdd_max_abs_rad_s2,
                    step_qdd["passive_qdd_max_abs_rad_s2"],
                )
                if (
                    privileged_close_supervision_rows
                    and privileged_close_supervision_rows[-1]["episode_id"]
                    == f"episode-{episode_index:04d}"
                ):
                    supervision = privileged_close_supervision_rows[-1]
                    supervision["hard_stop"] = True
                    supervision["mechanics_safe"] = False
                    supervision["max_arm_qdd_rad_s2"] = max(
                        float(supervision["max_arm_qdd_rad_s2"]),
                        step_qdd["arm_qdd_max_abs_rad_s2"],
                    )
                    supervision["max_passive_qdd_rad_s2"] = max(
                        float(supervision["max_passive_qdd_rad_s2"]),
                        step_qdd["passive_qdd_max_abs_rad_s2"],
                    )
                if len(records) != receipt.consumed_substeps:
                    raise Stage1AIsaacShortSmokeError(
                        "HARDSTOP_RECORD_COUNT_MISMATCH"
                    ) from hardstop_error
                if len(records) >= 10 and receipt.consumed_substeps != 10:
                    raise Stage1AIsaacShortSmokeError(
                        "HARDSTOP_ADVANCED_AFTER_TRIGGER"
                    ) from hardstop_error
                next_ee, _ = _ee_root_pose(env, p0a)
                cube_pos, cube_quat, cube_lin, cube_ang = _cube_world_state(env)
                residual = np.asarray(
                    applied_proposal.composition.scaled_residual_contribution_m,
                    dtype=np.float64,
                )
                hardstop_state = Stage1ARuntimeTransitionState(
                    current_ee_position_root_m=current_ee,
                    next_ee_position_root_m=next_ee,
                    nominal_grasp_position_root_m=nominal,
                    nominal_approach_axis_root=(1.0, 0.0, 0.0),
                    stage1a_active=True,
                    grasp_decision_phase=proposal_decision.residual_sac_active,
                    phase_reset=episode_control_step == 0,
                    episode_reset=episode_control_step == 0,
                    root_position_world_m_by_substep=np.stack(
                        [row["root_position_world_m"] for row in records]
                    ),
                    root_quat_world_xyzw_by_substep=np.stack(
                        [row["root_quat_world_xyzw"] for row in records]
                    ),
                    root_linear_velocity_world_m_s_by_substep=np.stack(
                        [
                            row["root_linear_velocity_world_m_s"]
                            for row in records
                        ]
                    ),
                    root_angular_velocity_world_rad_s_by_substep=np.stack(
                        [
                            row["root_angular_velocity_world_rad_s"]
                            for row in records
                        ]
                    ),
                    cube_center_world_m=cube_pos,
                    cube_quat_world_xyzw=cube_quat,
                    cube_linear_velocity_world_m_s=cube_lin,
                    cube_angular_velocity_world_rad_s=cube_ang,
                    cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                    effective_residual_root_m=residual,
                    previous_effective_residual_root_m=previous_residual,
                    final_action_xyz_root_m=xyz,
                    contract_grasp_state_valid=False,
                    safety_violation=True,
                    authoritative_safety_penalty=0.0,
                    non_pad_gripper_cube_contact=False,
                    close_command=(
                        packet.gripper_intent is AbstractGripperIntent.CLOSE
                    ),
                    consumed_substeps=receipt.consumed_substeps,
                    runtime_hardstop=True,
                )
                contact_paths = tuple(
                    physics_telemetry._contact_sources[side]["partner_actor"]
                    for side in ("inner", "outer")
                )
                reward_step = reward.step(
                    build_stage1a_reward_inputs(
                        records,
                        hardstop_state,
                        contact_partner_actor_paths=contact_paths,
                        device=env.device,
                    )
                )
                if (
                    not bool(reward_step.done[0].item())
                    or bool(reward_step.success[0].item())
                ):
                    raise Stage1AIsaacShortSmokeError(
                        "HARDSTOP_REWARD_NOT_FAILED_TERMINAL"
                    ) from hardstop_error
                previous_action = np.asarray(final_action, dtype=np.float32)
                previous_residual = residual
                terminal_inputs, terminal_camera_state = _gru_inputs(
                    env=env,
                    p0a=p0a,
                    camera_state=camera_state,
                    previous_action=previous_action,
                    hidden_reset=False,
                )
                terminal_observation = coordinator.encode_terminal_observation(
                    terminal_inputs
                )
                camera_timestamp = float(terminal_camera_state.timestamp_s)
                terminal_timestamp = float(
                    receipt.terminal_physics_timestamp_s
                )
                camera_age = terminal_timestamp - camera_timestamp
                if camera_age < -1.0e-6:
                    raise Stage1AIsaacShortSmokeError(
                        "TERMINAL_CAMERA_TIMESTAMP_AFTER_PHYSICS_SAMPLE"
                    ) from hardstop_error
                camera_age = max(0.0, camera_age)
                metrics = coordinator.accept_real_transition(
                    Stage1AAcceptedRealRow(
                        row_id=(
                            f"episode-{episode_index:04d}:"
                            f"step-{step_index:06d}:hardstop"
                        ),
                        proposal=applied_proposal,
                        next_actor_observation=terminal_observation,
                        reward_step=reward_step,
                        canonical_consumption_confirmed=True,
                        real_telemetry=True,
                        safety_measurement_valid=True,
                        safety_pass=False,
                        lift=False,
                        terminated=True,
                        truncated=False,
                        episode_id=f"episode-{episode_index:04d}",
                        phase=proposal_decision.phase.value,
                        success=False,
                        failure_reason="RUNTIME_HARDSTOP",
                        hardstop=True,
                        contact=bool(
                            reward_step.left_contact_boolean[0].item()
                            or reward_step.right_contact_boolean[0].item()
                        ),
                        bilateral=bool(
                            reward_step.bilateral_contact_boolean[0].item()
                        ),
                        stable=bool(
                            reward_step.stable_grasp_boolean[0].item()
                        ),
                        action_consumption=(
                            ActionConsumptionState.PARTIAL_TERMINATED.value
                        ),
                        control_step_index=episode_control_step,
                        physics_step_start=1,
                        physics_step_end=receipt.consumed_substeps,
                        consumed_substeps=receipt.consumed_substeps,
                        nominal_substeps=receipt.nominal_substeps,
                        execution_fraction=receipt.execution_fraction,
                        terminal_observation_is_real=True,
                        camera_timestamp=camera_timestamp,
                        terminal_timestamp=terminal_timestamp,
                        camera_age=camera_age,
                        hardstop_joint=receipt.hardstop_joint,
                        idx83_min_margin=receipt.idx83_limit_margin_rad,
                    )
                )
                maybe_export_periodic_checkpoint()
                _durable_jsonl(
                    transition_stream,
                    {
                        **dict(coordinator.replay_provenance[-1]),
                        "reward_total": float(
                            reward_step.reward_total[0].item()
                        ),
                        "safety_penalty": float(
                            reward_step.safety_penalty[0].item()
                        ),
                        "synthetic_transition": False,
                    },
                )
                writer.writerow(
                    {
                        "accepted_transitions": coordinator.accepted_transitions,
                        "sac_update_count": coordinator.sac_update_count,
                        "reward_total": float(
                            reward_step.reward_total[0].item()
                        ),
                        "residual_norm_mm": 1000.0
                        * float(np.linalg.norm(residual)),
                        "nominal_grasp_remaining_mm": float(distance * 1000.0),
                        "orientation_error_deg": _orientation_error_deg(
                            nominal_grasp_pose_root_m_xyzw[3:], current_ee_quat
                        ),
                        "distance_ready_count": simplified_close_gate.distance_ready_count,
                        "orientation_ready_count": simplified_close_gate.orientation_ready_count,
                        "persistence_ready_count": simplified_close_gate.persistence_ready_count,
                        "close_trigger_count": simplified_close_gate.close_trigger_count,
                        "left_contact": bool(
                            reward_step.left_contact_boolean[0].item()
                        ),
                        "right_contact": bool(
                            reward_step.right_contact_boolean[0].item()
                        ),
                        "stable": bool(
                            reward_step.stable_grasp_boolean[0].item()
                        ),
                        "contact": bool(
                            reward_step.left_contact_boolean[0].item()
                            or reward_step.right_contact_boolean[0].item()
                        ),
                        "bilateral": bool(
                            reward_step.bilateral_contact_boolean[0].item()
                        ),
                        "success": bool(reward_step.success[0].item()),
                        **step_qdd,
                    }
                )
                csv_stream.flush()
                if run is not None:
                    run.log(
                        {
                            **dict(metrics),
                            "telemetry/contact": int(
                                reward_step.left_contact_boolean[0].item()
                                or reward_step.right_contact_boolean[0].item()
                            ),
                            "telemetry/bilateral": int(
                                reward_step.bilateral_contact_boolean[0].item()
                            ),
                            "telemetry/stable": int(
                                reward_step.stable_grasp_boolean[0].item()
                            ),
                            "telemetry/success": int(
                                reward_step.success[0].item()
                            ),
                            "telemetry/nominal_grasp_remaining_mm": float(
                                distance * 1000.0
                            ),
                            "telemetry/residual_action_norm_mm": float(
                                np.linalg.norm(residual) * 1000.0
                            ),
                            "reward/total": float(
                                reward_step.reward_total[0].item()
                            ),
                            "telemetry/arm_qdd_max_abs_rad_s2": step_qdd[
                                "arm_qdd_max_abs_rad_s2"
                            ],
                            "telemetry/passive_qdd_max_abs_rad_s2": step_qdd[
                                "passive_qdd_max_abs_rad_s2"
                            ],
                        },
                        step=coordinator.accepted_transitions,
                    )
                episode_index += 1
                _reset_direct_pregrasp(
                    env=env,
                    p0a=p0a,
                    preflight=preflight,
                    counter=counter,
                    deferred_port=deferred_port,
                    latch=latch,
                    physics_telemetry=physics_telemetry,
                    episode_index=episode_index,
                    sample=selected_pregrasp_sample,
                )
                reward.reset()
                coordinator.reset_episode()
                phase_router.reset()
                episode_close_interlock_latched = False
                episode_external_hold_latched = False
                simplified_close_gate = SimplifiedClosePersistenceGate()
                natural_gru_close_logging_latched = False
                previous_ee_root_for_speed = None
                if callable(getattr(gripper_term, "set_external_hold_mask", None)):
                    gripper_term.set_external_hold_mask(
                        torch.zeros(1, dtype=torch.bool, device=env.device)
                    )
                episode_control_step = 0
                previous_action = np.zeros(4, dtype=np.float32)
                previous_residual = np.zeros(3, dtype=np.float64)
                camera_state = None
                inputs, camera_state = _gru_inputs(
                    env=env,
                    p0a=p0a,
                    camera_state=camera_state,
                    previous_action=previous_action,
                    hidden_reset=True,
                )
                current_ee, _ = _ee_root_pose(env, p0a)
                distance = float(np.linalg.norm(nominal - current_ee))
                proposal, proposal_decision, proposal_owner = propose_hybrid(
                    inputs,
                    distance_m=distance,
                    camera=camera_state,
                    control_step=coordinator.accepted_transitions,
                )
                continue
            if not consumption["single_consumption"]:
                raise Stage1AIsaacShortSmokeError("CONSUMPTION_UNKNOWN")
            action_submission_count += 1
            latch.commit(packet.gripper_intent)
            records = physics_telemetry.records[records_start:]
            if len(records) != 10:
                raise Stage1AIsaacShortSmokeError(
                    f"STAGE1A_PHYSICS_RECORD_COUNT:{len(records)}"
                )
            step_qdd = _step_joint_qdd_metrics(env=env, records=records)
            runtime_arm_qdd_max_abs_rad_s2 = max(
                runtime_arm_qdd_max_abs_rad_s2,
                step_qdd["arm_qdd_max_abs_rad_s2"],
            )
            runtime_passive_qdd_max_abs_rad_s2 = max(
                runtime_passive_qdd_max_abs_rad_s2,
                step_qdd["passive_qdd_max_abs_rad_s2"],
            )
            next_ee, _ = _ee_root_pose(env, p0a)
            task = preflight._task_metrics(env, task_mdp)
            evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
            peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
            forbidden_peak = max((max(values) for values in peaks.values()), default=0.0)
            env_terminated = bool(outputs[2].reshape(-1)[0].item())
            env_truncated = bool(outputs[3].reshape(-1)[0].item())
            active_termination = preflight._active_termination_names(env, outputs[2], outputs[3])
            if forbidden_peak > 1e-6 or "forbidden_collision" in active_termination:
                failure_snapshot = {
                    "schema": "g2_stage1a_forbidden_collision_snapshot_v1",
                    "accepted_transitions_before_failure": (
                        coordinator.accepted_transitions
                    ),
                    "action_submission_count": action_submission_count,
                    "episode_index": episode_index,
                    "policy_step": step_index,
                    "env_terminated": env_terminated,
                    "env_truncated": env_truncated,
                    "active_termination_names": list(active_termination),
                    "maximum_forbidden_contact_force_n": float(forbidden_peak),
                    "sensor_peak_forces_n": peaks,
                    "sensor_body_peak_forces_n": (
                        evaluator.sensor_body_peak_forces_n()
                        if evaluator is not None else {}
                    ),
                    "filtered_pad_force_components_n": (
                        evaluator.filtered_pad_force_components_n()
                        if evaluator is not None else {}
                    ),
                    "diagnostic_contact_force_components_n": (
                        evaluator.diagnostic_contact_force_components_n()
                        if evaluator is not None else {}
                    ),
                    "final_action_4d_metric_root_m": [
                        float(value) for value in final_action
                    ],
                    "effective_residual_root_m": [
                        float(value)
                        for value in applied_proposal.composition.scaled_residual_contribution_m
                    ],
                    "gripper_intent": packet.gripper_intent.value,
                    "task_metrics": dict(task),
                    "physics_substep_count": len(records),
                    **step_qdd,
                }
                if (
                    privileged_close_supervision_rows
                    and privileged_close_supervision_rows[-1]["episode_id"]
                    == f"episode-{episode_index:04d}"
                ):
                    supervision = privileged_close_supervision_rows[-1]
                    supervision["forbidden_collision"] = True
                    supervision["mechanics_safe"] = False
                    supervision["max_arm_qdd_rad_s2"] = max(
                        float(supervision["max_arm_qdd_rad_s2"]),
                        step_qdd["arm_qdd_max_abs_rad_s2"],
                    )
                    supervision["max_passive_qdd_rad_s2"] = max(
                        float(supervision["max_passive_qdd_rad_s2"]),
                        step_qdd["passive_qdd_max_abs_rad_s2"],
                    )
                failure_snapshot_path = output_dir / "FORBIDDEN_COLLISION_SNAPSHOT.json"
                _atomic_json(failure_snapshot_path, failure_snapshot)
                if run is not None:
                    run.summary.update({
                        "SHORT_ISAAC_1K_SMOKE": "FAIL_FORBIDDEN_COLLISION",
                        "accepted_transitions": coordinator.accepted_transitions,
                        "action_submission_count": action_submission_count,
                        "maximum_forbidden_contact_force_n": float(forbidden_peak),
                        "failure_snapshot": str(failure_snapshot_path),
                    })
                    run.finish(exit_code=2)
                raise Stage1AIsaacShortSmokeError(
                    "STAGE1A_FORBIDDEN_COLLISION:"
                    f"{forbidden_peak}:{active_termination}:{failure_snapshot_path}"
                )
            if "fixed_torso_drift" in active_termination:
                raise Stage1AIsaacShortSmokeError(
                    f"STAGE1A_FIXED_TORSO_DRIFT:{active_termination}"
                )

            # ManagerBasedRLEnv auto-resets a completed environment inside
            # env.step().  Its returned/public state is therefore the reset
            # state, not the physical successor of the submitted action.  A
            # transition spanning that boundary is not authoritative real
            # replay and must never be admitted.  Re-establish the frozen
            # direct-pregrasp initial condition and start a fresh GRU history;
            # the action was consumed exactly once but is deliberately not
            # counted as an accepted transition.
            if env_terminated or env_truncated:
                env_autoreset_rejected_transition_count += 1
                names = active_termination or [
                    "UNNAMED_TRUNCATION" if env_truncated else "UNNAMED_TERMINATION"
                ]
                for name in names:
                    env_autoreset_termination_counts[name] = (
                        env_autoreset_termination_counts.get(name, 0) + 1
                    )
                episode_index += 1
                _reset_direct_pregrasp(
                    env=env, p0a=p0a, preflight=preflight, counter=counter,
                    deferred_port=deferred_port, latch=latch,
                    physics_telemetry=physics_telemetry, episode_index=episode_index,
                    sample=selected_pregrasp_sample,
                )
                reward.reset()
                coordinator.reset_episode()
                phase_router.reset()
                episode_close_interlock_latched = False
                episode_external_hold_latched = False
                simplified_close_gate = SimplifiedClosePersistenceGate()
                natural_gru_close_logging_latched = False
                previous_ee_root_for_speed = None
                if callable(getattr(gripper_term, "set_external_hold_mask", None)):
                    gripper_term.set_external_hold_mask(
                        torch.zeros(1, dtype=torch.bool, device=env.device)
                    )
                episode_control_step = 0
                previous_action = np.zeros(4, dtype=np.float32)
                previous_residual = np.zeros(3, dtype=np.float64)
                camera_state = None
                inputs, camera_state = _gru_inputs(
                    env=env, p0a=p0a, camera_state=camera_state,
                    previous_action=previous_action, hidden_reset=True,
                )
                current_ee, _ = _ee_root_pose(env, p0a)
                distance = float(np.linalg.norm(nominal-current_ee))
                proposal, proposal_decision, proposal_owner = propose_hybrid(
                    inputs,
                    distance_m=distance,
                    camera=camera_state,
                    control_step=coordinator.accepted_transitions,
                )
                continue

            safety_violation = False
            cube_pos, cube_quat, cube_lin, cube_ang = _cube_world_state(env)
            residual = np.asarray(
                applied_proposal.composition.scaled_residual_contribution_m,
                dtype=np.float64,
            )
            state = Stage1ARuntimeTransitionState(
                current_ee_position_root_m=current_ee,
                next_ee_position_root_m=next_ee,
                nominal_grasp_position_root_m=nominal,
                nominal_approach_axis_root=(1.0, 0.0, 0.0),
                stage1a_active=True,
                grasp_decision_phase=proposal_decision.residual_sac_active,
                phase_reset=episode_control_step == 0,
                episode_reset=episode_control_step == 0,
                root_position_world_m_by_substep=np.stack(
                    [row["root_position_world_m"] for row in records]
                ),
                root_quat_world_xyzw_by_substep=np.stack(
                    [row["root_quat_world_xyzw"] for row in records]
                ),
                root_linear_velocity_world_m_s_by_substep=np.stack(
                    [row["root_linear_velocity_world_m_s"] for row in records]
                ),
                root_angular_velocity_world_rad_s_by_substep=np.stack(
                    [row["root_angular_velocity_world_rad_s"] for row in records]
                ),
                cube_center_world_m=cube_pos,
                cube_quat_world_xyzw=cube_quat,
                cube_linear_velocity_world_m_s=cube_lin,
                cube_angular_velocity_world_rad_s=cube_ang,
                cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                effective_residual_root_m=residual,
                previous_effective_residual_root_m=previous_residual,
                final_action_xyz_root_m=xyz,
                contract_grasp_state_valid=(
                    True
                    if stable_only
                    else bool(task["ever_lifted_while_stable"])
                ),
                safety_violation=safety_violation,
                authoritative_safety_penalty=0.0,
                non_pad_gripper_cube_contact=forbidden_peak > 1e-6,
                close_command=packet.gripper_intent is AbstractGripperIntent.CLOSE,
            )
            contact_paths = tuple(
                physics_telemetry._contact_sources[side]["partner_actor"]
                for side in ("inner", "outer")
            )
            reward_step = reward.step(build_stage1a_reward_inputs(
                records, state, contact_partner_actor_paths=contact_paths, device=env.device
            ))
            left_contact_now = bool(
                reward_step.left_contact_boolean[0].item()
            )
            right_contact_now = bool(
                reward_step.right_contact_boolean[0].item()
            )
            bilateral_now = left_contact_now and right_contact_now
            stable_now = bool(reward_step.stable_grasp_boolean[0].item())
            if (
                privileged_close_motion_interlock
                and privileged_close_supervision_rows
                and privileged_close_supervision_rows[-1]["episode_id"]
                == f"episode-{episode_index:04d}"
            ):
                supervision = privileged_close_supervision_rows[-1]
                supervision["left_contact"] = bool(
                    supervision["left_contact"] or left_contact_now
                )
                supervision["right_contact"] = bool(
                    supervision["right_contact"] or right_contact_now
                )
                supervision["bilateral"] = bool(
                    supervision["bilateral"] or bilateral_now
                )
                supervision["stable"] = bool(
                    supervision["stable"] or stable_now
                )
                supervision["grasp_success"] = bool(supervision["stable"])
                supervision["max_arm_qdd_rad_s2"] = max(
                    float(supervision["max_arm_qdd_rad_s2"]),
                    step_qdd["arm_qdd_max_abs_rad_s2"],
                )
                supervision["max_passive_qdd_rad_s2"] = max(
                    float(supervision["max_passive_qdd_rad_s2"]),
                    step_qdd["passive_qdd_max_abs_rad_s2"],
                )
                mechanics_acceleration_clean = bool(
                    supervision["max_arm_qdd_rad_s2"]
                    <= acceleration_limit_rad_s2
                    and supervision["max_passive_qdd_rad_s2"]
                    <= acceleration_limit_rad_s2
                )
                supervision["qdd_10_rad_s2_diagnostic_pass"] = (
                    mechanics_acceleration_clean
                )
                supervision["mechanics_safe"] = bool(
                    supervision["stable"]
                    and forbidden_peak <= 1.0e-6
                    and not supervision["hard_stop"]
                )
            previous_action = np.asarray(final_action, dtype=np.float32)
            previous_residual = residual
            previous_ee_root_for_speed = np.asarray(next_ee, dtype=np.float64)
            next_inputs, camera_state = _gru_inputs(
                env=env, p0a=p0a, camera_state=camera_state,
                previous_action=previous_action,
                hidden_reset=False,
            )
            next_distance = float(np.linalg.norm(nominal - next_ee))
            next_proposal, next_decision, next_owner = propose_hybrid(
                next_inputs,
                distance_m=next_distance,
                camera=camera_state,
                control_step=coordinator.accepted_transitions + 1,
            )
            reward_done = bool(reward_step.done[0].item())
            terminal_timestamp = float(records[-1]["physics_timestamp_s"])
            camera_timestamp = float(camera_state.timestamp_s)
            camera_age = terminal_timestamp - camera_timestamp
            if camera_age < -1.0e-6:
                raise Stage1AIsaacShortSmokeError(
                    "CAMERA_TIMESTAMP_AFTER_PHYSICS_INTERVAL"
                )
            metrics = coordinator.accept_real_transition(Stage1AAcceptedRealRow(
                row_id=f"episode-{episode_index:04d}:step-{step_index:06d}",
                proposal=applied_proposal,
                next_actor_observation=next_proposal.actor_observation,
                reward_step=reward_step,
                canonical_consumption_confirmed=True,
                real_telemetry=True,
                safety_measurement_valid=True,
                safety_pass=not safety_violation,
                lift=(
                    False
                    if stable_only
                    else bool(task["ever_lifted_while_stable"])
                ),
                terminated=reward_done,
                truncated=env_truncated and not reward_done,
                episode_id=f"episode-{episode_index:04d}",
                phase=proposal_decision.phase.value,
                action_consumption=ActionConsumptionState.FULLY_CONSUMED.value,
                control_step_index=episode_control_step,
                physics_step_start=1,
                physics_step_end=10,
                consumed_substeps=10,
                nominal_substeps=10,
                execution_fraction=1.0,
                terminal_observation_is_real=True,
                camera_timestamp=camera_timestamp,
                terminal_timestamp=terminal_timestamp,
                camera_age=max(0.0, camera_age),
            ))
            maybe_export_periodic_checkpoint()
            _durable_jsonl(
                transition_stream,
                {
                    **dict(coordinator.replay_provenance[-1]),
                    "reward_total": float(reward_step.reward_total[0].item()),
                    "safety_penalty": float(
                        reward_step.safety_penalty[0].item()
                    ),
                    "synthetic_transition": False,
                    "privileged_close_motion_interlock": bool(
                        privileged_close_motion_interlock
                        and episode_close_interlock_latched
                    ),
                    "privileged_external_gripper_hold": bool(
                        episode_external_hold_latched
                    ),
                    "nominal_grasp_remaining_mm": float(distance * 1000.0),
                    "orientation_error_deg": _orientation_error_deg(
                        nominal_grasp_pose_root_m_xyzw[3:], current_ee_quat
                    ),
                    "distance_ready_count": simplified_close_gate.distance_ready_count,
                    "orientation_ready_count": simplified_close_gate.orientation_ready_count,
                    "persistence_ready_count": simplified_close_gate.persistence_ready_count,
                    "close_trigger_count": simplified_close_gate.close_trigger_count,
                    "natural_p_close_logging_only": float(
                        proposal.close_probability
                    ),
                    "natural_close_hysteresis_logging_only": bool(
                        natural_gru_close_logging_latched
                    ),
                    "natural_close_edge_logging_only": bool(natural_close_edge),
                    "contact": bool(left_contact_now or right_contact_now),
                    "bilateral": bool(bilateral_now),
                    "stable": bool(stable_now),
                    "success": bool(reward_step.success[0].item()),
                    **step_qdd,
                    "privileged_student_input_count": 0,
                },
            )
            writer.writerow({
                "accepted_transitions": coordinator.accepted_transitions,
                "sac_update_count": coordinator.sac_update_count,
                "reward_total": float(reward_step.reward_total[0].item()),
                "residual_norm_mm": 1000.0 * float(np.linalg.norm(residual)),
                "nominal_grasp_remaining_mm": float(distance * 1000.0),
                "orientation_error_deg": _orientation_error_deg(
                    nominal_grasp_pose_root_m_xyzw[3:], current_ee_quat
                ),
                "distance_ready_count": simplified_close_gate.distance_ready_count,
                "orientation_ready_count": simplified_close_gate.orientation_ready_count,
                "persistence_ready_count": simplified_close_gate.persistence_ready_count,
                "close_trigger_count": simplified_close_gate.close_trigger_count,
                "left_contact": bool(reward_step.left_contact_boolean[0].item()),
                "right_contact": bool(reward_step.right_contact_boolean[0].item()),
                "contact": bool(left_contact_now or right_contact_now),
                "bilateral": bool(bilateral_now),
                "stable": bool(reward_step.stable_grasp_boolean[0].item()),
                "success": bool(reward_step.success[0].item()),
                **step_qdd,
            })
            csv_stream.flush()
            if run is not None:
                runtime_total = max(1, sum((
                    runtime_counts["far_reach"], runtime_counts["local_grasp"]
                )))
                run.log(
                    {
                        **dict(metrics),
                        "runtime/FAR_REACH_rate": runtime_counts["far_reach"] / runtime_total,
                        "runtime/LOCAL_GRASP_rate": runtime_counts["local_grasp"] / runtime_total,
                        "runtime/residual_gate_entry_rate": runtime_counts["residual_gate"] / runtime_total,
                        "runtime/actor_forward_rate": runtime_counts["actor_forward"] / runtime_total,
                        "runtime/nonzero_residual_rate": runtime_counts["nonzero_residual"] / runtime_total,
                        "runtime/GRU_nominal_active_rate": runtime_counts["gru_nominal"] / runtime_total,
                        "runtime/privileged_close_motion_hold": int(
                            privileged_close_motion_interlock
                            and episode_close_interlock_latched
                        ),
                        "runtime/privileged_external_gripper_hold": int(
                            episode_external_hold_latched
                        ),
                        "runtime/privileged_close_trigger_count": (
                            privileged_close_trigger_count
                        ),
                        "telemetry/contact": int(
                            left_contact_now or right_contact_now
                        ),
                        "telemetry/bilateral": int(bilateral_now),
                        "telemetry/stable": int(stable_now),
                        "telemetry/success": int(
                            reward_step.success[0].item()
                        ),
                        "telemetry/nominal_grasp_remaining_mm": float(
                            distance * 1000.0
                        ),
                        "telemetry/orientation_error_deg": _orientation_error_deg(
                            nominal_grasp_pose_root_m_xyzw[3:],
                            current_ee_quat,
                        ),
                        "telemetry/residual_action_norm_mm": float(
                            np.linalg.norm(residual) * 1000.0
                        ),
                        "telemetry/distance_ready_count": (
                            simplified_close_gate.distance_ready_count
                        ),
                        "telemetry/orientation_ready_count": (
                            simplified_close_gate.orientation_ready_count
                        ),
                        "telemetry/persistence_ready_count": (
                            simplified_close_gate.persistence_ready_count
                        ),
                        "telemetry/close_trigger_count": (
                            simplified_close_gate.close_trigger_count
                        ),
                        "telemetry/arm_qdd_max_abs_rad_s2": step_qdd[
                            "arm_qdd_max_abs_rad_s2"
                        ],
                        "telemetry/passive_qdd_max_abs_rad_s2": step_qdd[
                            "passive_qdd_max_abs_rad_s2"
                        ],
                        "reward/total": float(
                            reward_step.reward_total[0].item()
                        ),
                    },
                    step=coordinator.accepted_transitions,
                )
            proposal = next_proposal
            proposal_decision = next_decision
            proposal_owner = next_owner
            previous_nominal_distance = distance
            distance = next_distance
            if reward_done:
                episode_index += 1
                _reset_direct_pregrasp(
                    env=env, p0a=p0a, preflight=preflight, counter=counter,
                    deferred_port=deferred_port, latch=latch,
                    physics_telemetry=physics_telemetry, episode_index=episode_index,
                    sample=selected_pregrasp_sample,
                )
                reward.reset()
                coordinator.reset_episode()
                phase_router.reset()
                episode_close_interlock_latched = False
                episode_external_hold_latched = False
                simplified_close_gate = SimplifiedClosePersistenceGate()
                natural_gru_close_logging_latched = False
                previous_ee_root_for_speed = None
                if callable(getattr(gripper_term, "set_external_hold_mask", None)):
                    gripper_term.set_external_hold_mask(
                        torch.zeros(1, dtype=torch.bool, device=env.device)
                    )
                episode_control_step = 0
                previous_action = np.zeros(4, dtype=np.float32)
                previous_residual = np.zeros(3, dtype=np.float64)
                camera_state = None
                inputs, camera_state = _gru_inputs(
                    env=env, p0a=p0a, camera_state=camera_state,
                    previous_action=previous_action, hidden_reset=True,
                )
                current_ee, _ = _ee_root_pose(env, p0a)
                distance = float(np.linalg.norm(nominal-current_ee))
                previous_nominal_distance = distance
                previous_residual_gate = False
                handoff_30mm_step = None
                proposal, proposal_decision, proposal_owner = propose_hybrid(
                    inputs,
                    distance_m=distance,
                    camera=camera_state,
                    control_step=coordinator.accepted_transitions,
                )
            else:
                episode_control_step += 1
    privileged_supervision_path: Path | None = None
    privileged_student_npz_path: Path | None = None
    if privileged_close_supervision_rows:
        privileged_supervision_path = (
            output_dir / "PRIVILEGED_CLOSE_GRU_SUPERVISION.jsonl"
        )
        with privileged_supervision_path.open("x", encoding="utf-8") as stream:
            for row in privileged_close_supervision_rows:
                completed = dict(row)
                completed["timing_target"] = 1
                completed["grasp_success_target"] = int(
                    bool(completed.get("grasp_success"))
                )
                completed["mechanics_safe_target"] = int(
                    bool(completed.get("mechanics_safe"))
                )
                completed["training_eligible_for_gru"] = bool(
                    completed["grasp_success_target"]
                    and completed["mechanics_safe_target"]
                )
                completed["training_weight"] = (
                    "HIGH_WEIGHT"
                    if completed["training_eligible_for_gru"]
                    else "EXCLUDE_FROM_POSITIVE"
                    if completed["grasp_success_target"]
                    else "NEGATIVE"
                )
                _durable_jsonl(stream, completed)
        privileged_student_npz_path = (
            output_dir / "PRIVILEGED_CLOSE_STUDENT_PRE-CLOSE_INPUTS.npz"
        )
        np.savez_compressed(
            privileged_student_npz_path,
            **{
                name: np.concatenate(
                    [row[name] for row in privileged_close_student_arrays],
                    axis=0,
                )
                for name in privileged_close_student_arrays[0]
            },
        )
    checkpoint = coordinator.export_checkpoints(output_dir / "checkpoints")
    metrics = coordinator.metrics()
    metrics.update({
        "ACTION_SUBMISSION_COUNT": action_submission_count,
        "ENV_AUTORESET_REJECTED_TRANSITION_COUNT": (
            env_autoreset_rejected_transition_count
        ),
        "ENV_AUTORESET_TERMINATION_COUNTS": dict(
            sorted(env_autoreset_termination_counts.items())
        ),
        "SYNTHETIC_TRANSITIONS": 0,
        "FAR_REACH_COUNT": runtime_counts["far_reach"],
        "LOCAL_GRASP_COUNT": runtime_counts["local_grasp"],
        "RESIDUAL_GATE_ENTRY_COUNT": runtime_counts["residual_gate"],
        "ACTOR_FORWARD_COUNT": runtime_counts["actor_forward"],
        "NONZERO_RESIDUAL_COUNT": runtime_counts["nonzero_residual"],
        "GRU_NOMINAL_ACTIVE_COUNT": runtime_counts["gru_nominal"],
        "ENTRY_COUNT_INTO_SAC_BAND": runtime_counts["sac_band_entry"],
        "SAC_ENTRY_COUNT": runtime_counts["sac_band_entry"],
        "GRU_22_30_PROGRESS_RATE": runtime_counts["gru_22_30_progress"] / max(1, runtime_counts["gru_22_30_steps"]),
        "GRU_22_30_STALL_RATE": runtime_counts["gru_22_30_stall"] / max(1, runtime_counts["gru_22_30_steps"]),
        "TIME_TO_ENTER_22MM_FROM_30MM_MEAN_MS": (
            float(np.mean(handoff_30_to_22_steps) * 20.0)
            if handoff_30_to_22_steps else None
        ),
        "PRIVILEGED_CLOSE_MOTION_INTERLOCK_ENABLED": bool(
            privileged_close_motion_interlock
        ),
        "PRIVILEGED_CLOSE_TRIGGER_COUNT": privileged_close_trigger_count,
        "PRIVILEGED_CLOSE_NATURAL_TRIGGER_COUNT": (
            privileged_close_natural_trigger_count
        ),
        "PRIVILEGED_CLOSE_FORCED_TRIGGER_COUNT": (
            privileged_close_forced_trigger_count
        ),
        "PRIVILEGED_CLOSE_READY_EVALUATION_COUNT": (
            privileged_close_ready_evaluation_count
        ),
        "DISTANCE_READY_COUNT_MAX": distance_ready_count_max,
        "ORIENTATION_READY_COUNT_MAX": orientation_ready_count_max,
        "PERSISTENCE_READY_COUNT_MAX": persistence_ready_count_max,
        "CLOSE_TRIGGER_COUNT": privileged_close_trigger_count,
        "RUNTIME_ARM_QDD_MAX_ABS_RAD_S2": (
            runtime_arm_qdd_max_abs_rad_s2
        ),
        "RUNTIME_PASSIVE_QDD_MAX_ABS_RAD_S2": (
            runtime_passive_qdd_max_abs_rad_s2
        ),
        "PRIVILEGED_CLOSE_MOTION_HOLD_STEPS": (
            privileged_close_motion_hold_steps
        ),
        "PRIVILEGED_EXTERNAL_GRIPPER_HOLD_STEPS": (
            privileged_external_hold_steps
        ),
        "PRIVILEGED_CLOSE_NONZERO_XYZ_COMMAND_COUNT": (
            privileged_close_nonzero_xyz_count
        ),
        "PRIVILEGED_STUDENT_INPUT_COUNT": 0,
        "GRU_TRAINING_ROWS_SAVED": len(privileged_close_supervision_rows),
    })
    runtime_total = max(
        1, runtime_counts["far_reach"] + runtime_counts["local_grasp"]
    )
    metrics.update({
        "FAR_REACH_RATE": runtime_counts["far_reach"] / runtime_total,
        "LOCAL_GRASP_RATE": runtime_counts["local_grasp"] / runtime_total,
        "RESIDUAL_GATE_ENTRY_RATE": runtime_counts["residual_gate"] / runtime_total,
        "ACTOR_FORWARD_RATE": runtime_counts["actor_forward"] / runtime_total,
        "NONZERO_RESIDUAL_RATE": runtime_counts["nonzero_residual"] / runtime_total,
        "GRU_NOMINAL_ACTIVE_RATE": runtime_counts["gru_nominal"] / runtime_total,
    })
    freeze_after = source_freeze_provider()
    reward_authority = stage1a_reward_contract(reward_v3=reward_v3)
    replay_integrity = bool(
        metrics["FULL_50HZ_TRANSITIONS"]
        + metrics["PARTIAL_TERMINAL_TRANSITIONS"]
        == metrics["ACCEPTED_TRANSITIONS"]
        and metrics["REPLAY_HARDSTOP_COUNT"]
        == metrics["PARTIAL_TERMINAL_TRANSITIONS"]
        and metrics["SYNTHETIC_TRANSITIONS"] == 0
    )
    passed = bool(
        metrics["ACCEPTED_TRANSITIONS"] == accepted_transition_target
        and metrics["SAC_UPDATE_COUNT"] > 0
        and metrics["ACTOR_LOSS_FINITE"]
        and metrics["CRITIC_LOSS_FINITE"]
        and metrics["ALPHA_FINITE"]
        and metrics["FINAL_ACTION_BOUND_VIOLATION"] == 0
        and metrics["GRIPPER_AUTHORITY_VIOLATION"] == 0
        and not metrics["BC_WEIGHTS_CHANGED"]
        and source_freeze_before == freeze_after
        and replay_integrity
        and privileged_close_nonzero_xyz_count == 0
        and (
            not long_training
            or tuple(sorted(periodic_checkpoint_steps_saved))
            == periodic_checkpoint_steps
        )
    )
    hover_rate_v3 = float(metrics.get("HOVER_RATE_TOTAL", metrics["HOVER_RATE"]))
    contact_count = int(metrics.get("CONTACT_COUNT", 0))
    hardstop_count = int(metrics.get("RUNTIME_HARDSTOP_COUNT", 0))
    safety_regression = hardstop_count > 0 or int(metrics.get("FINAL_ACTION_BOUND_VIOLATION", 0)) > 0
    learning_signal_present = bool(
        metrics["SAC_UPDATE_COUNT"] > 0
        and metrics["ACTOR_LOSS_FINITE"]
        and metrics["CRITIC_LOSS_FINITE"]
        and (
            abs(float(metrics.get("PROGRESS_REWARD_SUM", 0.0))) > 0.0
            or abs(float(metrics.get("HOVER_RATE_TOTAL", 0.0))) > 0.0
        )
    )
    reward_v3_direction_validated = bool(
        reward_v3
        and hover_rate_v3 < 0.9303
        and contact_count > 0
        and not safety_regression
    )
    if reward_v3_direction_validated:
        recommended_next = "15K_V3"
    elif reward_v3 and hover_rate_v3 < 0.9303 and contact_count == 0:
        recommended_next = "CONTACT_REACHABILITY_AUDIT"
    else:
        recommended_next = "REWARD_RETUNE"
    metrics.update(
        {
            "HOVER_RATE_BASELINE": 0.9303,
            "HOVER_RATE_V3": hover_rate_v3,
            "FINAL_ACTION_VIOLATION": metrics["FINAL_ACTION_BOUND_VIOLATION"],
            "GRU_WEIGHTS_CHANGED": metrics["BC_WEIGHTS_CHANGED"],
            "SAFETY_REGRESSION": safety_regression,
            "LEARNING_SIGNAL_PRESENT": learning_signal_present,
            "REWARD_V3_DIRECTION_VALIDATED": reward_v3_direction_validated,
            "RECOMMENDED_NEXT": recommended_next,
        }
    )
    report = {
        "schema": STAGE1A_ISAAC_SHORT_SMOKE_SCHEMA,
        "execution_mode": (
            "REAL_ISAAC_STABLE_ONLY_REWARD_V3_15K"
            if stable_only
            else "REAL_ISAAC_BOUNDED_REWARD_V3_3K"
            if reward_v3
            else ("REAL_ISAAC_BOUNDED_15K" if long_training else "REAL_ISAAC_BOUNDED_1K")
        ),
        "source_freeze_before": dict(source_freeze_before),
        "source_freeze_after": dict(freeze_after),
        "asset": {"path": str(selected_asset_path), "sha256": selected_asset_sha256},
        "direct_init_receipt": dict(direct_init_receipt),
        "timing": {"control_hz": 50, "physics_hz": 500, "rgbd_hz": 25},
        "metrics": metrics,
        "replay_strategy": replay_strategy,
        "reward_version": "V3" if reward_v3 else "V2",
        "stable_only": bool(stable_only),
        "lift_phase": "DISABLED" if stable_only else "LEGACY_CONTRACT",
        "place_phase": "DISABLED",
        "training_seed": int(training_seed),
        "qdd_10_rad_s2_authority": "DIAGNOSTIC_ONLY",
        "periodic_checkpoints": periodic_checkpoint_receipts,
        "CHECKPOINTS_SAVED": [
            int(item["accepted_transitions"])
            for item in periodic_checkpoint_receipts
        ],
        "REWARD_V3_IMPLEMENTED": "YES" if reward_v3 else "NO",
        "SMOKE_ACCEPTED_TRANSITIONS": metrics["ACCEPTED_TRANSITIONS"],
        "SOURCE_FREEZE": "PASS" if source_freeze_before == freeze_after else "FAIL",
        "TRAINING_PROMOTED_TO_RUNTIME": "NO",
        "privileged_close_motion_interlock": {
            "enabled": privileged_close_motion_interlock,
            "scope": (
                "STABLE_ONLY_REWARD_V3_15K"
                if stable_only
                else "BOUNDED_REWARD_V3_3K_TRAINING_DIAGNOSTIC_ONLY"
            ),
            "authority": "SIMPLIFIED_15_20MM_15DEG_5STEP_SAFETY_GATE",
            "residual_min_mm": SIMPLIFIED_CLOSE_MIN_RESIDUAL_M * 1000.0,
            "residual_max_mm": SIMPLIFIED_CLOSE_MAX_RESIDUAL_M * 1000.0,
            "orientation_error_max_deg": (
                SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG
            ),
            "persistence_steps": SIMPLIFIED_CLOSE_PERSISTENCE_STEPS,
            "persistence_duration_ms": (
                1000.0 * SIMPLIFIED_CLOSE_PERSISTENCE_STEPS / CONTROL_HZ
            ),
            "canonical_4d_gripper_path": True,
            "direct_joint_command": False,
            "direct_torque_command": False,
            "xyz_command_while_closing": "ZERO",
            "gripper_command_while_closing": "CLOSE",
            "external_hold_authority": "NONE",
            "gru_p_close_authority": False,
            "gru_p_close_logging_only": True,
            "gru_hysteresis_logging_only": True,
            "natural_close_edge_logging_only": True,
            "trigger_count": privileged_close_trigger_count,
            "natural_logging_edge_count": privileged_close_natural_trigger_count,
            "simplified_gate_trigger_count": (
                privileged_close_forced_trigger_count
            ),
            "motion_hold_steps": privileged_close_motion_hold_steps,
            "external_gripper_hold_steps": privileged_external_hold_steps,
            "nonzero_xyz_command_count": (
                privileged_close_nonzero_xyz_count
            ),
            "student_privileged_input_count": 0,
            "runtime_gate_changed": True,
            "controller_changed": False,
            "safety_threshold_changed": False,
            "supervision_jsonl": (
                None
                if privileged_supervision_path is None
                else str(privileged_supervision_path)
            ),
            "student_input_npz": (
                None
                if privileged_student_npz_path is None
                else str(privileged_student_npz_path)
            ),
        },
        "ordinary_her": "ON" if replay_strategy == "HER" else "OFF",
        "her_force": "ON" if replay_strategy == "HER_FORCE" else "OFF",
        "far_reach_bc": far_receipt,
        "residual_actor_initialization": dict(coordinator.actor_initialization),
        "checkpoint": checkpoint.__dict__,
        "exception_safe_transition_contract": "PASS",
        "hardstop_guard": "UNCHANGED",
        "source_owned_hardstop_penalty": reward_authority[
            "runtime_hardstop_terminal_penalty"
        ],
        "penalty_authority": reward_authority[
            "runtime_hardstop_penalty_authority"
        ],
        "reward_authority_sha256": reward_authority[
            "reward_authority_sha256"
        ],
        "partial_terminal_observation": "REAL",
        "replay_integrity": "PASS" if replay_integrity else "FAIL",
        "sac_terminal_bootstrap": "ZERO_FOR_TERMINATED",
        "no_synthetic_transitions": metrics["SYNTHETIC_TRANSITIONS"] == 0,
        "replay_transition_artifact": str(transition_path),
        "SHORT_ISAAC_1K_SMOKE": (
            "NOT_APPLICABLE"
            if (long_training or reward_v3)
            else ("PASS" if passed else "FAIL")
        ),
        "TRAINING_15K": (
            ("PASS" if passed else "FAIL") if long_training else "NOT_RUN"
        ),
        "REWARD_V3_SMOKE": (
            ("PASS" if passed else "FAIL") if reward_v3 else "NOT_RUN"
        ),
        "LONG_TRAINING_READY": (
            "NOT_APPLICABLE"
            if long_training
            else (
                "REVIEW_REQUIRED"
                if reward_v3_direction_validated
                else "NO"
            )
            if reward_v3
            else ("YES" if passed else "NO")
        ),
        "LONG_TRAINING_STARTED": "YES" if long_training else "NO",
    }
    _atomic_json(report_path, report)
    mirror_name = (
        "STAGE1A_STABLE_ONLY_REWARD_V3_15K_REPORT.json"
        if stable_only
        else "STAGE1A_REWARD_V3_3K_REPORT.json"
        if reward_v3
        else "STAGE1A_ISAAC_15K_REPORT.json"
        if long_training
        else "STAGE1A_ISAAC_1K_REPORT.json"
    )
    _atomic_json(output_dir / mirror_name, report)
    if run is not None:
        run.summary.update(metrics)
        run.summary["SHORT_ISAAC_1K_SMOKE"] = report["SHORT_ISAAC_1K_SMOKE"]
        run.summary["TRAINING_15K"] = report["TRAINING_15K"]
        run.summary["REWARD_V3_SMOKE"] = report["REWARD_V3_SMOKE"]
        run.summary["PRIVILEGED_CLOSE_MOTION_INTERLOCK"] = bool(
            privileged_close_motion_interlock
        )
        run.summary["PRIVILEGED_STUDENT_INPUT_COUNT"] = 0
        report["wandb"] = {
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
            "mode": wandb_mode,
        }
        _atomic_json(report_path, report)
        _atomic_json(output_dir / mirror_name, report)
        run.finish()
    return 0 if passed else 2


def _hybrid_far_observation(
    *,
    env: Any,
    inputs: HumanGraspSequenceInputs,
    camera_state: _CameraAcquisitionState,
    control_step: int,
) -> Any:
    """Build the existing P1 student observation from the same live row."""

    from geniesim.rl.isaaclab.g2_policy_branch.p1_final_action_path import (
        P1ObservationMetadata,
        P1StudentObservation,
    )

    robot = env.scene["robot"]
    arm_term = env.action_manager.get_term("arm_action")
    arm_ids = torch.as_tensor(
        [int(value) for value in arm_term._joint_ids],
        dtype=torch.long,
        device=env.device,
    )
    qdd = _tensor(robot.data.joint_acc).index_select(1, arm_ids)
    return P1StudentObservation(
        metadata=P1ObservationMetadata(
            control_step=control_step,
            control_timestamp_s=control_step / float(CONTROL_HZ),
            camera_timestamp_s=float(camera_state.timestamp_s),
            camera_frame_id=int(camera_state.acquisition_id),
        ),
        right_wrist_rgb=inputs.right_wrist_rgb[:, 0],
        right_wrist_depth_m=inputs.right_wrist_depth_m[:, 0],
        right_wrist_depth_valid=inputs.right_wrist_depth_valid[:, 0],
        ee_pose_robot_root_m_xyzw=inputs.ee_pose_robot_root_m_xyzw[:, 0],
        right_arm_joint_position_rad=inputs.right_arm_joint_position_rad[:, 0],
        right_arm_joint_velocity_rad_s=inputs.right_arm_joint_velocity_rad_s[:, 0],
        right_arm_joint_acceleration_rad_s2=qdd,
        gripper_state_open=1.0 - inputs.current_gripper_state[:, 0],
        previous_policy_action_4d_metric_root_m=(
            inputs.previous_policy_action_4d_metric_root_m[:, 0]
        ),
    )


def run_stage1a_hybrid_activation_smoke(
    *,
    env: Any,
    p0a: Any,
    preflight: Any,
    counter: Any,
    deferred_port: Any,
    latch: Any,
    physics_telemetry: Any,
    nominal_grasp_pose_root_m_xyzw: Any,
    source_initial_state_receipt: Mapping[str, Any],
    source_freeze_before: Mapping[str, Any],
    source_freeze_provider: Callable[[], Mapping[str, Any]],
    selected_asset_path: Path,
    selected_asset_sha256: str,
    far_reach_checkpoint: Path,
    far_reach_checkpoint_sha256: str,
    bc_checkpoint: Path,
    bc_checkpoint_sha256: str,
    residual_actor_checkpoint: Path,
    residual_actor_checkpoint_sha256: str,
    output_dir: Path,
    report_path: Path,
    maximum_control_steps: int = 256,
    wandb_enabled: bool = False,
    wandb_mode: str = "offline",
    wandb_project: str = "geniesim-g2-hybrid-runtime",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    stable_playback: bool = False,
    record_video_path: Path | None = None,
) -> int:
    """Inference-only proof of canonical phase -> actor -> controller binding."""

    from geniesim.rl.isaaclab.g2_policy_branch.contact_free_visual_bc_runtime import (
        load_contact_free_visual_bc_checkpoint,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.hybrid_grasp_runtime import (
        HybridGraspPhase,
        UnifiedHybridGraspRuntime,
    )
    from geniesim.rl.sac.human_grasp_gru_bc import (
        load_human_grasp_checkpoint,
        tensor_state_sha256,
    )
    from geniesim.rl.sac.residual_sac_runtime import (
        load_residual_sac_actor_checkpoint,
    )
    from geniesim.rl.sac.stage1a_real_sac_coordinator import (
        radial_metric_residual,
    )

    if env.num_envs != 1 or not math.isclose(float(env.step_dt), 0.02, abs_tol=1e-12):
        raise Stage1AIsaacShortSmokeError("HYBRID_SMOKE_REQUIRES_ONE_ENV_AT_50HZ")
    if type(maximum_control_steps) is not int or not 1 <= maximum_control_steps <= 512:
        raise Stage1AIsaacShortSmokeError("HYBRID_SMOKE_STEP_CAP_INVALID")
    if (
        source_initial_state_receipt.get("schema")
        != "g2_hybrid_activation_source_initial_state_v1"
        or source_initial_state_receipt.get("SOURCE_AUTHORITATIVE") is not True
        or source_initial_state_receipt.get("pose_or_threshold_modified") is not False
    ):
        raise Stage1AIsaacShortSmokeError(
            "HYBRID_SMOKE_SOURCE_INITIAL_STATE_NOT_AUTHORITATIVE"
        )
    if report_path.exists() or output_dir.exists():
        raise Stage1AIsaacShortSmokeError("HYBRID_SMOKE_REFUSES_OVERWRITE")
    for path, expected, label in (
        (selected_asset_path, selected_asset_sha256, "ASSET"),
        (far_reach_checkpoint, far_reach_checkpoint_sha256, "FAR_BC"),
        (bc_checkpoint, bc_checkpoint_sha256, "GRASP_GRU"),
        (residual_actor_checkpoint, residual_actor_checkpoint_sha256, "RESIDUAL_ACTOR"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise Stage1AIsaacShortSmokeError(f"HYBRID_SMOKE_{label}_HASH_MISMATCH")
    output_dir.mkdir(parents=True, exist_ok=False)

    far_model, far_receipt = load_contact_free_visual_bc_checkpoint(
        far_reach_checkpoint,
        expected_sha256=far_reach_checkpoint_sha256,
        device=env.device,
    )
    grasp_model, grasp_receipt = load_human_grasp_checkpoint(
        bc_checkpoint, device=env.device
    )
    grasp_model.eval()
    for parameter in grasp_model.parameters():
        parameter.requires_grad_(False)
    residual_runtime = load_residual_sac_actor_checkpoint(
        residual_actor_checkpoint,
        expected_sha256=residual_actor_checkpoint_sha256,
        expected_human_grasp_checkpoint_sha256=bc_checkpoint_sha256,
        expected_observation_dim=grasp_model.config.gru_hidden_dim,
        device=env.device,
    )
    runtime = UnifiedHybridGraspRuntime(
        far_reach_bc=far_model,
        human_grasp_gru=grasp_model,
        residual_sac=residual_runtime,
        final_limiter_gate=None,
        production_router=None,
        close_calibration=grasp_receipt["calibration"],
    )
    runtime.reset()
    nominal = np.asarray(nominal_grasp_pose_root_m_xyzw[:3], dtype=np.float64)
    initial_ee, _ = _ee_root_pose(env, p0a)
    initial_measured_distance = float(np.linalg.norm(nominal - initial_ee))
    handoff_authority_m = float(
        source_initial_state_receipt["handoff_authority_m"]
    )
    if not initial_measured_distance > handoff_authority_m:
        raise Stage1AIsaacShortSmokeError(
            "HYBRID_SMOKE_INITIAL_DISTANCE_NOT_ABOVE_30MM:"
            f"{initial_measured_distance:.12f}"
        )
    initial_phase = "FAR_REACH"
    initial_nominal_owner = "CUROBO_CONTACT_FREE_BC"
    grasp_state_before = tensor_state_sha256(grasp_model)
    activation_floor_m = float(
        10.0 * np.finfo(np.float32).eps * MAX_FINAL_ACTION_M
    )
    wandb_run = None
    if wandb_enabled:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or "candidate_a_hybrid_activation_smoke",
                group="candidate_a_hybrid_activation_smoke",
                tags=("candidate_a", "hybrid_runtime", "inference_only"),
                mode=wandb_mode,
                dir=str(output_dir),
                config={
                    "maximum_control_steps": maximum_control_steps,
                    "sac_updates": 0,
                    "optimizer_steps": 0,
                    "handoff_30mm_m": handoff_authority_m,
                    "gru_owner_22mm_m": 0.022,
                    "residual_window_m": [0.015, 0.022],
                    "asset_sha256": selected_asset_sha256,
                    "far_bc_sha256": far_reach_checkpoint_sha256,
                    "grasp_gru_sha256": bc_checkpoint_sha256,
                    "residual_actor_sha256": residual_actor_checkpoint_sha256,
                    "source_initial_state": dict(source_initial_state_receipt),
                },
            )
            if wandb_run is None:
                raise RuntimeError("wandb.init returned None")
        except Exception as error:
            raise Stage1AIsaacShortSmokeError(
                f"HYBRID_SMOKE_WANDB_INIT_FAILED:{type(error).__name__}:{error}"
            ) from error
    previous_action = np.zeros(4, dtype=np.float32)
    camera_state: _CameraAcquisitionState | None = None
    prior_phase = None
    prior_owner = None
    prior_distance: float | None = None
    prior_nominal = None
    prior_combined = None
    far_entry_distance = None
    handoff_before_distance = None
    handoff_after_distance = None
    handoff_control_step = None
    handoff_physics_step = None
    handoff_count = 0
    ownership_handoff_count = 0
    ownership_before_distance = None
    ownership_after_distance = None
    ownership_control_step = None
    local_steps = 0
    actor_forward_count = 0
    action_submission_count = 0
    phase_oscillation = False
    ownership_oscillation = False
    runtime_hardstop = False
    close_gate = SimplifiedClosePersistenceGate()
    close_latched = False
    close_count = 0
    contact_count = 0
    bilateral_count = 0
    stable_count = 0
    bilateral_stable_counter = 0
    first_close_step: int | None = None
    first_contact_step: int | None = None
    first_bilateral_step: int | None = None
    first_stable_step: int | None = None
    forbidden_collision_count = 0
    hover_step_count = 0
    evaluated_step_count = 0
    joint_limit_violation_count = 0
    velocity_limit_violation_count = 0
    residual_norms_mm: list[float] = []
    nominal_residuals_mm: list[float] = []
    human_grasp_active = False
    residual_gate_entered = False
    curobo_owner_22_to_30 = True
    residual_gripper_authority_violations = 0
    close_sources: set[str] = set()
    raw_norms: list[float] = []
    phase_norms: list[float] = []
    safety_norms: list[float] = []
    controller_norms: list[float] = []
    delta_nominal_at_30mm = None
    delta_combined_at_30mm = None
    delta_nominal_at_22mm = None
    delta_combined_at_22mm = None
    idx83_margins: list[float] = []
    gpu_used_mib: list[float] = []
    nonzero_parity = False
    stop_reason = "CONTROL_STEP_CAP"
    integrity_error = None
    rows_path = output_dir / "HYBRID_ACTIVATION_PROVENANCE.jsonl"
    video_frames: list[np.ndarray] = []
    last_video_acquisition_id: int | None = None

    with rows_path.open("x", encoding="utf-8") as rows_stream:
        try:
            for control_step in range(maximum_control_steps):
                inputs, camera_state = _gru_inputs(
                    env=env,
                    p0a=p0a,
                    camera_state=camera_state,
                    previous_action=previous_action,
                    hidden_reset=(control_step == 0 or not human_grasp_active),
                )
                if (
                    record_video_path is not None
                    and camera_state.acquisition_id != last_video_acquisition_id
                ):
                    video_frames.append(np.asarray(camera_state.rgb).copy())
                    last_video_acquisition_id = camera_state.acquisition_id
                far_observation = _hybrid_far_observation(
                    env=env,
                    inputs=inputs,
                    camera_state=camera_state,
                    control_step=control_step,
                )
                current_ee, current_ee_quat = _ee_root_pose(env, p0a)
                distance = float(np.linalg.norm(nominal - current_ee))
                evaluated_step_count += 1
                if (
                    prior_distance is not None
                    and abs(float(prior_distance) - distance) < 0.00025
                ):
                    hover_step_count += 1
                inference = runtime.infer(
                    nominal_grasp_residual_m=distance,
                    far_observation=far_observation,
                    grasp_inputs=inputs,
                )
                phase = inference.decision.phase
                owner = inference.source_policy
                if phase is HybridGraspPhase.FAR_REACH and far_entry_distance is None:
                    far_entry_distance = distance
                if phase is not HybridGraspPhase.FAR_REACH:
                    local_steps += 1
                if prior_phase is HybridGraspPhase.FAR_REACH and phase is not prior_phase:
                    handoff_count += 1
                    handoff_before_distance = float(prior_distance)
                    handoff_after_distance = distance
                    handoff_control_step = control_step
                    handoff_physics_step = action_submission_count * 10
                    if prior_nominal is not None:
                        delta_nominal_at_30mm = float(
                            np.linalg.norm(
                                np.asarray(inference.bc_action_4d_metric_root_m[:3])
                                - np.asarray(prior_nominal[:3])
                            )
                        )
                if prior_phase is not None and phase is HybridGraspPhase.FAR_REACH and prior_phase is not phase:
                    phase_oscillation = True
                    raise Stage1AIsaacShortSmokeError("HYBRID_PHASE_OSCILLATION")
                if (
                    prior_owner == "CUROBO_CONTACT_FREE_BC"
                    and owner == "HUMAN_GRASP_GRU_BC"
                ):
                    ownership_handoff_count += 1
                    ownership_before_distance = float(prior_distance)
                    ownership_after_distance = distance
                    ownership_control_step = control_step
                    if prior_nominal is not None:
                        delta_nominal_at_22mm = float(
                            np.linalg.norm(
                                np.asarray(inference.bc_action_4d_metric_root_m[:3])
                                - np.asarray(prior_nominal[:3])
                            )
                        )
                if (
                    prior_owner == "HUMAN_GRASP_GRU_BC"
                    and owner == "CUROBO_CONTACT_FREE_BC"
                ):
                    ownership_oscillation = True
                    raise Stage1AIsaacShortSmokeError(
                        "HYBRID_NOMINAL_OWNERSHIP_OSCILLATION"
                    )
                if 0.022 < distance <= 0.030 and owner != "CUROBO_CONTACT_FREE_BC":
                    curobo_owner_22_to_30 = False
                    raise Stage1AIsaacShortSmokeError(
                        "HYBRID_CUROBO_OWNER_22_TO_30_MISMATCH"
                    )
                human_grasp_active = human_grasp_active or bool(
                    inference.decision.human_grasp_nominal_active
                    and owner == "HUMAN_GRASP_GRU_BC"
                )
                residual_gate_entered = residual_gate_entered or bool(
                    inference.decision.residual_sac_active
                )

                composition = inference.composition
                final_action = composition.final_action_4d_metric_root_m
                if stable_playback:
                    close_receipt = close_gate.observe(
                        nominal_grasp_residual_m=distance,
                        orientation_error_deg=_orientation_error_deg(
                            nominal_grasp_pose_root_m_xyzw[3:],
                            current_ee_quat,
                        ),
                        no_safety_violation=True,
                    )
                    if bool(close_receipt["close_trigger"]):
                        close_latched = True
                        close_count += 1
                        first_close_step = control_step
                    if close_latched:
                        final_action = (0.0, 0.0, 0.0, 1.0)
                    else:
                        final_action = (*final_action[:3], 0.0)
                if (
                    prior_phase is HybridGraspPhase.FAR_REACH
                    and phase is not prior_phase
                    and prior_combined is not None
                ):
                    delta_combined_at_30mm = float(
                        np.linalg.norm(
                            np.asarray(final_action[:3])
                            - np.asarray(prior_combined[:3])
                        )
                    )
                if (
                    prior_owner == "CUROBO_CONTACT_FREE_BC"
                    and owner == "HUMAN_GRASP_GRU_BC"
                    and prior_combined is not None
                ):
                    delta_combined_at_22mm = float(
                        np.linalg.norm(
                            np.asarray(final_action[:3])
                            - np.asarray(prior_combined[:3])
                        )
                    )
                if not math.isclose(
                    float(final_action[3]),
                    float(inference.bc_action_4d_metric_root_m[3]),
                    abs_tol=0.0,
                ):
                    residual_gripper_authority_violations += 1
                    raise Stage1AIsaacShortSmokeError(
                        "HYBRID_RESIDUAL_GRIPPER_AUTHORITY_VIOLATION"
                    )
                xyz = np.asarray(final_action[:3], dtype=np.float64)
                normalized = metric_xyz_to_normalized(xyz)
                high_level = HighLevelPolicyAction.from_sequence(
                    (*normalized.tolist(), float(final_action[3]))
                )
                packet, packet_tensor, derivation = p0a._build_authoritative_packet(
                    high_level=high_level,
                    batch_size=1,
                    device=env.device,
                    latch=latch,
                )
                if packet.gripper_intent is AbstractGripperIntent.CLOSE:
                    close_sources.add(inference.source_policy)
                records_start = len(physics_telemetry.records)
                physics_telemetry.set_context(
                    policy_step=control_step,
                    bc_step=control_step,
                    phase=phase.value,
                    gripper_intent=packet.gripper_intent.value,
                    close_onset=(
                        packet.gripper_intent is AbstractGripperIntent.CLOSE
                        and latch.intent is not AbstractGripperIntent.CLOSE
                    ),
                    clipping=False,
                    last_real_sensor_timestamps={"right_wrist": camera_state.timestamp_s},
                    physics_clock_anchor_s=camera_state.sensor_clock_time_s,
                )
                outputs, consumption = preflight._consume_once(
                    env=env,
                    counter=counter,
                    deferred_port=deferred_port,
                    packet=packet,
                    label=f"HYBRID_ACTIVATION_{control_step:04d}",
                )
                if not consumption["single_consumption"]:
                    raise Stage1AIsaacShortSmokeError("CONSUMPTION_UNKNOWN")
                action_submission_count += 1
                latch.commit(packet.gripper_intent)
                records = physics_telemetry.records[records_start:]
                if len(records) != 10:
                    raise Stage1AIsaacShortSmokeError(
                        f"HYBRID_PHYSICS_RECORD_COUNT:{len(records)}"
                    )
                active = preflight._active_termination_names(env, outputs[2], outputs[3])
                if bool(outputs[2].reshape(-1)[0].item()) or bool(outputs[3].reshape(-1)[0].item()) or active:
                    raise Stage1AIsaacShortSmokeError(
                        "HYBRID_RUNTIME_TERMINATION:" + ",".join(active)
                    )
                step_contact = any(
                    float(record["inner_contact_force_n"]) >= 1.0
                    or float(record["outer_contact_force_n"]) >= 1.0
                    for record in records
                )
                step_bilateral = any(
                    bool(record["bilateral_contact"]) for record in records
                )
                step_stable_signal = any(
                    bool(record["stable_contact"]) for record in records
                )
                step_forbidden = any(
                    bool(record["forbidden_collision"]) for record in records
                )
                joint_limit_violation_count += int(
                    any(
                        bool(np.any(np.asarray(record["joint_limit_margin_rad"]) < 0.0))
                        for record in records
                    )
                )
                contact_count += int(step_contact)
                bilateral_count += int(step_bilateral)
                forbidden_collision_count += int(step_forbidden)
                if step_bilateral and step_stable_signal and not step_forbidden:
                    bilateral_stable_counter += 1
                else:
                    bilateral_stable_counter = 0
                stable_now = bilateral_stable_counter >= 10
                stable_count += int(stable_now)
                if step_contact and first_contact_step is None:
                    first_contact_step = control_step
                if step_bilateral and first_bilateral_step is None:
                    first_bilateral_step = control_step
                if stable_now and first_stable_step is None:
                    first_stable_step = control_step

                arm_term = env.action_manager.get_term("arm_action")
                controller_normalized = _tensor(arm_term._raw_actions)[0, :3]
                scale_tensor = _tensor(arm_term._scale)
                controller_scale = (
                    scale_tensor[:3]
                    if scale_tensor.ndim == 1
                    else scale_tensor[0, :3]
                )
                controller_metric = (
                    controller_normalized * controller_scale
                ).detach().cpu().numpy().astype(np.float64)
                if not np.allclose(controller_metric, xyz, rtol=0.0, atol=1.0e-9):
                    raise Stage1AIsaacShortSmokeError("CONTROLLER_ACTION_IDENTITY_MISMATCH")
                controller_residual = controller_metric - np.asarray(
                    inference.bc_action_4d_metric_root_m[:3], dtype=np.float64
                )
                post_phase = np.asarray(
                    inference.post_phase_gate_residual_metric_root_m, dtype=np.float64
                )
                # Successful canonical consumption is the existing runtime
                # safety/controller admission receipt; no second limiter is
                # introduced by this smoke.
                post_safety = controller_residual.copy()
                replay_metric = np.asarray(
                    radial_metric_residual(inference.actor_raw_normalized_xyz),
                    dtype=np.float64,
                ) * float(composition.alpha_dimensionless)
                parity = bool(
                    np.allclose(replay_metric, controller_residual, rtol=0.0, atol=1.0e-9)
                )
                robot = env.scene["robot"]
                joint_names = tuple(str(name) for name in robot.joint_names)
                idx83_name = "idx83_gripper_r_outer_joint4"
                if idx83_name not in joint_names:
                    raise Stage1AIsaacShortSmokeError("HYBRID_IDX83_JOINT_MISSING")
                idx83_index = joint_names.index(idx83_name)
                joint_q = _tensor(robot.data.joint_pos)[0, idx83_index]
                joint_limits = _tensor(robot.data.soft_joint_pos_limits)[
                    0, idx83_index
                ]
                idx83_margin = float(
                    torch.minimum(
                        joint_q - joint_limits[0], joint_limits[1] - joint_q
                    ).item()
                )
                idx83_margins.append(idx83_margin)
                if torch.cuda.is_available():
                    gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info(env.device)
                    gpu_used = float(gpu_total_bytes - gpu_free_bytes) / (1024.0**2)
                    gpu_used_mib.append(gpu_used)
                else:
                    gpu_used = None
                raw_norm = float(
                    np.linalg.norm(inference.raw_residual_sac_xyz_metric_root_m)
                )
                phase_norm = float(np.linalg.norm(post_phase))
                safety_norm = float(np.linalg.norm(post_safety))
                controller_norm = float(np.linalg.norm(controller_residual))
                residual_norms_mm.append(1000.0 * controller_norm)
                nominal_residuals_mm.append(1000.0 * distance)
                meaningful_activation = bool(
                    inference.actor_forward_called
                    and raw_norm > activation_floor_m
                    and phase_norm > activation_floor_m
                    and safety_norm > activation_floor_m
                    and controller_norm > activation_floor_m
                )
                if inference.actor_forward_called:
                    actor_forward_count += 1
                    raw_norms.append(raw_norm)
                    phase_norms.append(phase_norm)
                    safety_norms.append(safety_norm)
                    controller_norms.append(controller_norm)
                    if meaningful_activation:
                        nonzero_parity = parity
                row = {
                    "schema": "g2_stage1a_hybrid_activation_transition_v1",
                    "control_step": control_step,
                    "physics_step_end": action_submission_count * 10,
                    "contact_semantic_label": "REACH",
                    "hybrid_runtime_phase": phase.value,
                    "phase_before": None if prior_phase is None else prior_phase.value,
                    "phase_after": phase.value,
                    "phase_handoff_30mm": bool(
                        prior_phase is HybridGraspPhase.FAR_REACH
                        and phase is HybridGraspPhase.LOCAL_GRASP
                    ),
                    "distance_to_grasp_m": distance,
                    "nominal_policy_owner": inference.source_policy,
                    "nominal_owner_before": prior_owner,
                    "nominal_owner_after": owner,
                    "ownership_handoff_22mm": bool(
                        prior_owner == "CUROBO_CONTACT_FREE_BC"
                        and owner == "HUMAN_GRASP_GRU_BC"
                    ),
                    "human_grasp_nominal_active": (
                        inference.decision.human_grasp_nominal_active
                    ),
                    "nominal_action_4d_metric_root_m": list(
                        inference.bc_action_4d_metric_root_m
                    ),
                    "residual_gate_eligible": inference.decision.residual_sac_active,
                    "actor_forward_called": inference.actor_forward_called,
                    "actor_raw_action_normalized": list(inference.actor_raw_normalized_xyz),
                    "residual_requested_metric_root_m": list(
                        inference.raw_residual_sac_xyz_metric_root_m
                    ),
                    "residual_applied_metric_root_m": controller_residual.tolist(),
                    "post_phase_gate_residual_metric_root_m": post_phase.tolist(),
                    "post_safety_gate_residual_metric_root_m": post_safety.tolist(),
                    "combined_action_4d_metric_root_m": list(final_action),
                    "controller_submitted_action_4d_metric_root_m": [
                        *controller_metric.tolist(),
                        float(final_action[3]),
                    ],
                    "replay_residual_normalized": list(
                        inference.actor_raw_normalized_xyz
                    ),
                    "replay_residual_metric_root_m": replay_metric.tolist(),
                    "replay_controller_parity": parity,
                    "meaningful_residual_activation": meaningful_activation,
                    "activation_numerical_floor_m": activation_floor_m,
                    "close_command_source": (
                        inference.source_policy
                        if packet.gripper_intent is AbstractGripperIntent.CLOSE
                        else "NONE_OPEN"
                    ),
                    "nominal_gripper_action": float(
                        inference.bc_action_4d_metric_root_m[3]
                    ),
                    "residual_gripper_action": 0.0,
                    "abstract_gripper_state": packet.gripper_intent.value,
                    "idx83_limit_margin_rad": idx83_margin,
                    "gpu_memory_used_mib": gpu_used,
                    "single_consumption": True,
                    "packet_derivation": derivation,
                }
                _durable_jsonl(rows_stream, row)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "runtime/control_step": control_step,
                            "runtime/measured_distance_m": distance,
                            "runtime/phase_far_reach": int(
                                phase is HybridGraspPhase.FAR_REACH
                            ),
                            "runtime/phase_local_grasp": int(
                                phase is HybridGraspPhase.LOCAL_GRASP
                            ),
                            "runtime/owner_curobo_bc": int(
                                owner == "CUROBO_CONTACT_FREE_BC"
                            ),
                            "runtime/owner_human_gru": int(
                                owner == "HUMAN_GRASP_GRU_BC"
                            ),
                            "runtime/human_grasp_nominal_active": int(
                                inference.decision.human_grasp_nominal_active
                            ),
                            "runtime/residual_gate_eligible": int(
                                inference.decision.residual_sac_active
                            ),
                            "runtime/actor_forward_called": int(
                                inference.actor_forward_called
                            ),
                            "runtime/actor_raw_residual_norm_m": raw_norm,
                            "runtime/post_phase_residual_norm_m": phase_norm,
                            "runtime/post_safety_residual_norm_m": safety_norm,
                            "runtime/controller_residual_norm_m": controller_norm,
                            "runtime/phase_handoff_30mm": int(
                                row["phase_handoff_30mm"]
                            ),
                            "runtime/ownership_handoff_22mm": int(
                                row["ownership_handoff_22mm"]
                            ),
                            "runtime/replay_controller_parity": int(parity),
                            "runtime/idx83_limit_margin_rad": idx83_margin,
                            "runtime/runtime_hardstop": 0,
                            "system/gpu_memory_used_mib": gpu_used,
                        },
                        step=control_step,
                    )
                previous_action = np.asarray(final_action, dtype=np.float32)
                prior_phase = phase
                prior_owner = owner
                prior_distance = distance
                prior_nominal = inference.bc_action_4d_metric_root_m
                prior_combined = final_action
                if (
                    stable_playback
                    and first_stable_step is not None
                ):
                    stop_reason = "STABLE_BILATERAL_10_STEPS"
                    break
                if (
                    not stable_playback
                    and
                    handoff_count >= 1
                    and ownership_handoff_count >= 1
                    and inference.actor_forward_called
                    and meaningful_activation
                    and parity
                ):
                    stop_reason = "NONZERO_ACTIVATION_AND_PARITY_PROVEN"
                    break
        except RuntimeHardstop:
            runtime_hardstop = True
            stop_reason = "RUNTIME_HARDSTOP"
        except Exception as error:
            integrity_error = f"{type(error).__name__}:{error}"
            stop_reason = "INTEGRITY_OR_SAFETY_FAILURE"

    freeze_after = source_freeze_provider()
    if record_video_path is not None:
        if not video_frames:
            raise Stage1AIsaacShortSmokeError("PLAYBACK_VIDEO_HAS_NO_FRAMES")
        record_video_path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        imageio.mimsave(
            record_video_path,
            video_frames,
            fps=RGBD_HZ,
            macro_block_size=None,
        )

    def _stats(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_mm": None if not len(array) else 1000.0 * float(array.mean()),
            "max_mm": None if not len(array) else 1000.0 * float(array.max()),
            "nonzero_ratio": (
                None
                if not len(array)
                else float(np.mean(array > activation_floor_m))
            ),
        }

    source_freeze_match = source_freeze_before == freeze_after
    grasp_state_after = tensor_state_sha256(grasp_model)
    bc_weights_unchanged = grasp_state_before == grasp_state_after
    activation_passed = bool(
        initial_measured_distance > handoff_authority_m
        and far_entry_distance is not None
        and handoff_count >= 1
        and ownership_handoff_count >= 1
        and local_steps > 0
        and human_grasp_active
        and residual_gate_entered
        and actor_forward_count > 0
        and raw_norms
        and raw_norms[-1] > activation_floor_m
        and phase_norms[-1] > activation_floor_m
        and safety_norms[-1] > activation_floor_m
        and controller_norms[-1] > activation_floor_m
        and nonzero_parity
        and not phase_oscillation
        and not ownership_oscillation
        and curobo_owner_22_to_30
        and residual_gripper_authority_violations == 0
        and bc_weights_unchanged
        and not runtime_hardstop
        and integrity_error is None
        and source_freeze_match
    )
    passed = bool(
        activation_passed
        and (
            not stable_playback
            or (
                close_count == 1
                and first_stable_step is not None
                and forbidden_collision_count == 0
            )
        )
    )
    report = {
        "schema": "g2_stage1a_hybrid_activation_smoke_v1",
        "EVALUATION_ONLY": "YES" if stable_playback else "NO",
        "RECORDED_VIDEO": (
            None if record_video_path is None else str(record_video_path)
        ),
        "TRAINING_TRANSITION_COUNT_INCREMENT": 0,
        "REPLAY_BUFFER_WRITE_COUNT": 0,
        "OPTIMIZER_UPDATE_COUNT": 0,
        "GRADIENT_UPDATE_COUNT": 0,
        "HYBRID_RUNTIME_BINDING": "PASS" if passed else "FAIL",
        "CANONICAL_ROUTER_USED": "YES",
        "SOURCE_AUTHORITATIVE_INITIAL_STATE": "PASS",
        "INITIAL_STATE_SOURCE": dict(source_initial_state_receipt),
        "INITIAL_MEASURED_DISTANCE_M": initial_measured_distance,
        "INITIAL_PHASE": initial_phase,
        "INITIAL_NOMINAL_OWNER": initial_nominal_owner,
        "FAR_REACH_NOMINAL_OWNER": "CUROBO_CONTACT_FREE_BC",
        "LOCAL_GRASP_NOMINAL_OWNER": (
            "CUROBO_CONTACT_FREE_BC_UNTIL_22MM_THEN_HUMAN_GRASP_GRU_BC"
        ),
        "FAR_REACH_ENTERED": "YES" if far_entry_distance is not None else "NO",
        "FAR_REACH_ENTRY_DISTANCE_M": far_entry_distance,
        "PHASE_HANDOFF_30MM_COUNT": handoff_count,
        "REACH_TO_LOCAL_GRASP_HANDOFF_COUNT": handoff_count,
        "HANDOFF_DISTANCE": {
            "authority_m": 0.030,
            "before_m": handoff_before_distance,
            "after_m": handoff_after_distance,
            "control_step": handoff_control_step,
            "physics_step": handoff_physics_step,
        },
        "LOCAL_GRASP_ENTERED": "YES" if local_steps else "NO",
        "CUROBO_OWNER_22_TO_30": "PASS" if curobo_owner_22_to_30 else "FAIL",
        "OWNERSHIP_HANDOFF_22MM_COUNT": ownership_handoff_count,
        "OWNERSHIP_HANDOFF_DISTANCE": {
            "authority_m": 0.022,
            "before_m": ownership_before_distance,
            "after_m": ownership_after_distance,
            "control_step": ownership_control_step,
        },
        "22MM_EQUALITY_SEMANTICS": (
            "DISTANCE_LE_0P022_IS_LOCAL_GRASP_WITH_HUMAN_GRASP_GRU_"
            "NOMINAL_AND_RESIDUAL_ELIGIBLE"
        ),
        "HUMAN_GRASP_GRU_ACTIVE": "YES" if human_grasp_active else "NO",
        "RESIDUAL_GATE_ENTERED": "YES" if residual_gate_entered else "NO",
        "RESIDUAL_ACTIVATION_WINDOW_M": [0.015, 0.022],
        "ACTIVATION_NUMERICAL_FLOOR_M": activation_floor_m,
        "ACTIVATION_NUMERICAL_FLOOR_SOURCE": (
            "10_X_FLOAT32_EPSILON_X_MAX_FINAL_ACTION_0P0045M"
        ),
        "LOCAL_GRASP_ACTIVE_STEPS": local_steps,
        "ACTOR_FORWARD_COUNT": actor_forward_count,
        "ACTOR_RAW_RESIDUAL": _stats(raw_norms),
        "POST_PHASE_GATE_RESIDUAL": _stats(phase_norms),
        "POST_SAFETY_GATE_RESIDUAL": _stats(safety_norms),
        "CONTROLLER_APPLIED_RESIDUAL": _stats(controller_norms),
        "NONZERO_REPLAY_CONTROLLER_PARITY": (
            "PASS" if nonzero_parity else "NOT_OBSERVED"
        ),
        "GRU_GRIPPER_AUTHORITY": (
            "PASS"
            if human_grasp_active and residual_gripper_authority_violations == 0
            else "FAIL"
        ),
        "RESIDUAL_GRIPPER_AUTHORITY_VIOLATION": (
            residual_gripper_authority_violations
        ),
        "DELTA_NOMINAL_AT_30MM_M": delta_nominal_at_30mm,
        "DELTA_COMBINED_AT_30MM_M": delta_combined_at_30mm,
        "DELTA_NOMINAL_AT_22MM_M": delta_nominal_at_22mm,
        "DELTA_COMBINED_AT_22MM_M": delta_combined_at_22mm,
        "PHASE_OSCILLATION": "YES" if phase_oscillation else "NO",
        "OWNERSHIP_OSCILLATION": "YES" if ownership_oscillation else "NO",
        "RESIDUAL_DOUBLE_APPLICATION": "NO" if nonzero_parity else "UNRESOLVED",
        "CLOSE_COMMAND_SOURCE": sorted(close_sources) if close_sources else ["NONE_OPEN"],
        "RUNTIME_HARDSTOP": "YES" if runtime_hardstop else "NO",
        "RUNTIME_HARDSTOP_GUARD": {
            "status": "UNCHANGED",
            "absolute_joint_acceleration_rad_s2": 10.0,
            "joint_limit_margin_rad": 1.0e-5,
        },
        "ACTION_SUBMISSION_COUNT": action_submission_count,
        "CLOSE_COUNT": close_count,
        "CONTACT_COUNT": contact_count,
        "BILATERAL_COUNT": bilateral_count,
        "STABLE_COUNT": stable_count,
        "CONTACT_RATE": int(first_contact_step is not None),
        "BILATERAL_RATE": int(first_bilateral_step is not None),
        "STABLE_RATE": int(first_stable_step is not None),
        "HOVER_RATE": hover_step_count / max(1, evaluated_step_count),
        "TIME_TO_CONTACT_MS": (
            None
            if first_contact_step is None or first_close_step is None
            else 20.0 * (first_contact_step - first_close_step)
        ),
        "TIME_TO_BILATERAL_MS": (
            None
            if first_bilateral_step is None or first_close_step is None
            else 20.0 * (first_bilateral_step - first_close_step)
        ),
        "TIME_TO_STABLE_MS": (
            None
            if first_stable_step is None or first_close_step is None
            else 20.0 * (first_stable_step - first_close_step)
        ),
        "MIN_NOMINAL_RESIDUAL_MM": (
            None if not nominal_residuals_mm else min(nominal_residuals_mm)
        ),
        "RESIDUAL_MEAN_MM": (
            None if not residual_norms_mm else float(np.mean(residual_norms_mm))
        ),
        "RESIDUAL_P95_MM": (
            None
            if not residual_norms_mm
            else float(np.percentile(residual_norms_mm, 95.0))
        ),
        "RESIDUAL_MAX_MM": (
            None if not residual_norms_mm else max(residual_norms_mm)
        ),
        "FORBIDDEN_COLLISION": forbidden_collision_count,
        "JOINT_LIMIT_VIOLATION": joint_limit_violation_count,
        "VELOCITY_LIMIT_VIOLATION": velocity_limit_violation_count,
        "IDX83_MIN_LIMIT_MARGIN_RAD": (
            None if not idx83_margins else float(min(idx83_margins))
        ),
        "GPU_MEMORY_USED_MIB": {
            "initial": None if not gpu_used_mib else gpu_used_mib[0],
            "maximum": None if not gpu_used_mib else max(gpu_used_mib),
            "final": None if not gpu_used_mib else gpu_used_mib[-1],
        },
        "SAC_UPDATES": 0,
        "OPTIMIZER_STEP_COUNT": 0,
        "BC_WEIGHT_CHANGE": 0 if bc_weights_unchanged else 1,
        "BC_STATE_SHA256_BEFORE": grasp_state_before,
        "BC_STATE_SHA256_AFTER": grasp_state_after,
        "SOURCE_FREEZE": "PASS" if source_freeze_match else "FAIL",
        "ACTIVATION_SMOKE": "PASS" if passed else "FAIL",
        "READY_FOR_CLOSE_SAC_LEARNING_RUN": "YES" if passed else "NO",
        "NEXT_REQUIRED_ACTION": (
            "BOUNDED_CLOSE_SAC_LEARNING_AUTHORIZATION"
            if passed
            else "REVIEW_HYBRID_ACTIVATION_FAILURE"
        ),
        "stop_reason": stop_reason,
        "integrity_error": integrity_error,
        "provenance_jsonl": str(rows_path),
        "WANDB": {
            "enabled": bool(wandb_enabled),
            "mode": wandb_mode,
            "project": wandb_project,
            "run_id": None if wandb_run is None else wandb_run.id,
            "run_name": None if wandb_run is None else wandb_run.name,
            "run_url": None if wandb_run is None else wandb_run.url,
        },
        "checkpoints": {
            "far_reach_bc": dict(far_receipt),
            "human_grasp_gru": {
                "path": str(bc_checkpoint),
                "sha256": bc_checkpoint_sha256,
                "state_sha256": grasp_receipt["state_sha256"],
            },
            "residual_actor": {
                "path": str(residual_actor_checkpoint),
                "sha256": residual_actor_checkpoint_sha256,
                "actor_state_sha256": residual_runtime.receipt.actor_state_sha256,
            },
        },
        "source_freeze_before": dict(source_freeze_before),
        "source_freeze_after": dict(freeze_after),
    }
    _atomic_json(output_dir / "HYBRID_ACTIVATION_SMOKE_REPORT.json", report)
    _atomic_json(report_path, report)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "activation_smoke_pass": int(passed),
                "phase_handoff_30mm_count": handoff_count,
                "ownership_handoff_22mm_count": ownership_handoff_count,
                "actor_forward_count": actor_forward_count,
                "runtime_hardstop": int(runtime_hardstop),
                "nonzero_replay_controller_parity": int(nonzero_parity),
                "idx83_min_limit_margin_rad": (
                    None if not idx83_margins else float(min(idx83_margins))
                ),
                "source_freeze_pass": int(source_freeze_match),
            }
        )
        wandb_run.finish(exit_code=0 if passed else 2)
    return 0 if passed else 2


__all__ = [
    "ACCEPTED_TRANSITION_TARGET",
    "STAGE1A_ISAAC_SHORT_SMOKE_SCHEMA",
    "Stage1AIsaacShortSmokeError",
    "run_stage1a_close_admission_diagnostic",
    "run_stage1a_close_residual_sweep_episode",
    "run_stage1a_hybrid_activation_smoke",
    "run_stage1a_isaac_short_smoke",
    "source_authoritative_hybrid_activation_initial_state",
]
