# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical batched ingress for the Stage-1A vector runtime.

The scalar deferred-packet implementation intentionally holds one semantic
packet and expands it across a batch.  This module is deliberately separate:
it accepts only :class:`ImmutablePerEnvActionPacket`, calls ``env.step`` once,
and verifies that the exact independently-materialized ``[N,8]`` tensor is
the tensor seen by the existing ActionManager.  It has no direct
``ActionManager.process_action`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from dataclasses import fields
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .stage1a_vector_contract import (
    ImmutablePerEnvActionPacket,
    PerEnvPacketPort,
    Stage1AVectorContractError,
)


def _runtime_tensor(value: Any) -> torch.Tensor:
    """Return a public Isaac/torch buffer as a torch tensor without copying it."""

    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    return torch.from_dlpack(value)


VECTOR_STAGE1A_CONSUMPTION_SCHEMA = "g2_stage1a_vector_canonical_consumption_v1"


class Stage1AVectorRuntimeError(RuntimeError):
    """Unknown, repeated, broadcast, or non-canonical vector consumption."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class VectorActionConsumptionReceipt:
    schema: str
    label: str
    packet_id: str
    content_fingerprint: str
    batch_shape: tuple[int, int]
    packet_sha256: str
    env_step_count: int
    action_manager_process_action_count: int
    router_direct_process_action_count: int
    single_batched_consumption: bool
    action_broadcast_detected: bool
    unique_4d_action_rows: int
    unique_8d_packet_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "label": self.label,
            "packet_id": self.packet_id,
            "content_fingerprint": self.content_fingerprint,
            "batch_shape": list(self.batch_shape),
            "packet_sha256": self.packet_sha256,
            "env_step_count": self.env_step_count,
            "action_manager_process_action_count": self.action_manager_process_action_count,
            "router_direct_process_action_count": self.router_direct_process_action_count,
            "single_batched_consumption": self.single_batched_consumption,
            "action_broadcast_detected": self.action_broadcast_detected,
            "unique_4d_action_rows": self.unique_4d_action_rows,
            "unique_8d_packet_rows": self.unique_8d_packet_rows,
        }


def consume_per_env_packet_once(
    *,
    env: Any,
    counter: Any,
    port: PerEnvPacketPort,
    packet: ImmutablePerEnvActionPacket,
    label: str,
) -> tuple[Any, VectorActionConsumptionReceipt]:
    """Submit exactly one independently-rowed 8-D packet through ``env.step``.

    ``counter`` is the existing passive lifecycle instrumentation installed by
    the P0 harness.  The caller must not provide an ActionManager; that keeps
    the vector path under the same ``env.step -> process_action`` ownership as
    the frozen scalar contract.
    """

    if not isinstance(port, PerEnvPacketPort):
        raise Stage1AVectorRuntimeError("VECTOR_PACKET_PORT_TYPE_INVALID")
    if not isinstance(packet, ImmutablePerEnvActionPacket):
        raise Stage1AVectorRuntimeError("VECTOR_PACKET_TYPE_INVALID")
    if packet.batch_size != int(env.num_envs) or port.batch_size != packet.batch_size:
        raise Stage1AVectorRuntimeError("VECTOR_PACKET_BATCH_SIZE_MISMATCH")
    if packet.broadcast_detected:
        raise Stage1AVectorRuntimeError("BATCH_ACTION_BROADCAST_DETECTED")
    claimed = port.claim(packet.packet_id)
    tensor = claimed.as_env_step_tensor()
    if tuple(tensor.shape) != (int(env.num_envs), 8):
        raise Stage1AVectorRuntimeError("VECTOR_ENV_STEP_TENSOR_SHAPE_INVALID")
    expected_sha256 = _tensor_sha256(tensor)
    before = counter.checkpoint()
    counter.context = str(label)
    try:
        output = env.step(tensor)
    except BaseException as error:
        # The port intentionally has no reusable-packet state after an
        # unknown exception.  A caller must terminate that vector episode
        # rather than replaying a potentially consumed packet.
        raise Stage1AVectorRuntimeError(
            f"VECTOR_ENV_STEP_CONSUMPTION_UNKNOWN:{label}"
        ) from error
    interval: Mapping[str, Any] = counter.interval(before)
    if (
        int(interval.get("env_step_calls", -1)) != 1
        or int(interval.get("action_manager_process_action_calls", -1)) != 1
    ):
        raise Stage1AVectorRuntimeError(
            f"VECTOR_NOT_SINGLE_BATCHED_CONSUMPTION:{label}:{dict(interval)}"
        )
    env_packets = interval.get("env_step_packets")
    process_packets = interval.get("process_action_packets")
    if not isinstance(env_packets, list) or not isinstance(process_packets, list):
        raise Stage1AVectorRuntimeError("VECTOR_COUNTER_PACKET_EVIDENCE_MISSING")
    if len(env_packets) != 1 or len(process_packets) != 1:
        raise Stage1AVectorRuntimeError("VECTOR_COUNTER_PACKET_CARDINALITY_INVALID")
    env_capture, process_capture = env_packets[0], process_packets[0]
    expected_shape = [int(env.num_envs), 8]
    if (
        env_capture.get("shape") != expected_shape
        or process_capture.get("shape") != expected_shape
        or env_capture.get("sha256") != expected_sha256
        or process_capture.get("sha256") != expected_sha256
        or env_capture.get("sha256") != process_capture.get("sha256")
    ):
        raise Stage1AVectorRuntimeError("VECTOR_PACKET_IDENTITY_MISMATCH")
    port.acknowledge(packet.packet_id)
    return output, VectorActionConsumptionReceipt(
        schema=VECTOR_STAGE1A_CONSUMPTION_SCHEMA,
        label=str(label),
        packet_id=packet.packet_id,
        content_fingerprint=packet.content_fingerprint,
        batch_shape=(int(env.num_envs), 8),
        packet_sha256=expected_sha256,
        env_step_count=1,
        action_manager_process_action_count=1,
        router_direct_process_action_count=0,
        single_batched_consumption=True,
        action_broadcast_detected=False,
        unique_4d_action_rows=packet.unique_4d_rows,
        unique_8d_packet_rows=packet.unique_8d_rows,
    )


def capture_vector_wrist_gru_inputs(
    *,
    env: Any,
    p0a: Any,
    previous_actions_4d_metric_root_m: np.ndarray,
    hidden_reset_mask: Sequence[bool],
    camera_cache: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Capture independent 25-Hz wrist frames and make one causal GRU row/env.

    This deliberately returns *one* ``HumanGraspSequenceInputs`` object per
    environment.  The frozen coordinator has a contractual ``[1, 1]``
    streaming interface; concatenating these rows into a faux batch would
    silently share its hidden state.  ``camera_cache`` is the vector contract
    cache, so a 50-Hz control row may reuse only its *own* 25-Hz frame.

    The helper owns no physics/action code and therefore remains usable by the
    bounded 1-env parity smoke as well as the 10-env runtime.
    """

    from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
    from geniesim.rl.sac.human_grasp_gru_bc import HumanGraspSequenceInputs
    from geniesim.rl.sac.keyboard_v3_dataset import preprocess_depth_m
    from .stage1a_vector_contract import PerEnvCameraFrame

    num_envs = int(env.num_envs)
    previous = np.asarray(previous_actions_4d_metric_root_m, dtype=np.float32)
    if previous.shape != (num_envs, 4) or not np.isfinite(previous).all():
        raise Stage1AVectorRuntimeError("VECTOR_PREVIOUS_ACTION_SHAPE_INVALID")
    reset = tuple(bool(value) for value in hidden_reset_mask)
    if len(reset) != num_envs:
        raise Stage1AVectorRuntimeError("VECTOR_HIDDEN_RESET_MASK_SHAPE_INVALID")
    if getattr(camera_cache, "num_envs", None) != num_envs:
        raise Stage1AVectorRuntimeError("VECTOR_CAMERA_CACHE_CARDINALITY_INVALID")

    camera = env.scene["right_wrist_camera"]
    output = camera.data.output
    if "rgb" not in output or "distance_to_image_plane" not in output:
        raise Stage1AVectorRuntimeError("VECTOR_WRIST_RGBD_STREAM_MISSING")
    rgb = _runtime_tensor(output["rgb"])[..., :3]
    raw_depth = _runtime_tensor(output["distance_to_image_plane"])
    if raw_depth.ndim == 3:
        raw_depth = raw_depth.unsqueeze(-1)
    expected_rgb = (num_envs, 192, 256, 3)
    expected_depth = (num_envs, 192, 256, 1)
    if tuple(rgb.shape) != expected_rgb or rgb.dtype != torch.uint8:
        raise Stage1AVectorRuntimeError("VECTOR_WRIST_RGB_CONTRACT_MISMATCH")
    if tuple(raw_depth.shape) != expected_depth:
        raise Stage1AVectorRuntimeError("VECTOR_WRIST_DEPTH_CONTRACT_MISMATCH")
    raw_depth = raw_depth.to(dtype=torch.float32)
    corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (
        torch.isfinite(raw_depth) & (raw_depth < 0.0)
    )
    if bool(corrupt.any().item()):
        raise Stage1AVectorRuntimeError("VECTOR_WRIST_DEPTH_CORRUPT")
    source_valid = torch.isfinite(raw_depth) & (raw_depth >= 0.0)
    frame_ids = _runtime_tensor(camera.frame).reshape(-1)
    captured, age = camera_capture_time_and_age(camera)
    capture_times = _runtime_tensor(captured).reshape(-1)
    ages = _runtime_tensor(age).reshape(-1)
    if (
        frame_ids.numel() != num_envs
        or capture_times.numel() != num_envs
        or ages.numel() != num_envs
    ):
        raise Stage1AVectorRuntimeError("VECTOR_WRIST_CAMERA_IDENTITY_CARDINALITY")

    position, quaternion = p0a._world_ee_pose_to_root(
        env.scene["robot"], env.scene["ee_frame"]
    )[:2]
    position_t = _runtime_tensor(position).to(dtype=torch.float32, device=env.device)
    quaternion_t = _runtime_tensor(quaternion).to(dtype=torch.float32, device=env.device)
    if tuple(position_t.shape) != (num_envs, 3) or tuple(quaternion_t.shape) != (num_envs, 4):
        raise Stage1AVectorRuntimeError("VECTOR_EE_ROOT_POSE_SHAPE_INVALID")
    robot = env.scene["robot"]
    arm_term = env.action_manager.get_term("arm_action")
    arm_ids = torch.as_tensor(
        [int(value) for value in arm_term._joint_ids],
        dtype=torch.long,
        device=env.device,
    )
    q = _runtime_tensor(robot.data.joint_pos).index_select(1, arm_ids).to(torch.float32)
    qd = _runtime_tensor(robot.data.joint_vel).index_select(1, arm_ids).to(torch.float32)
    if tuple(q.shape) != (num_envs, 7) or tuple(qd.shape) != (num_envs, 7):
        raise Stage1AVectorRuntimeError("VECTOR_ARM_STATE_SHAPE_INVALID")

    inputs: list[Any] = []
    frames: list[Any] = []
    for env_id in range(num_envs):
        timestamp = float(capture_times[env_id].item())
        frame_id = int(frame_ids[env_id].item())
        sensor_time = timestamp + float(ages[env_id].item())
        if (
            not np.isfinite(timestamp)
            or not np.isfinite(sensor_time)
            or timestamp < 0.0
            or sensor_time < timestamp
            or frame_id < 0
        ):
            raise Stage1AVectorRuntimeError("VECTOR_WRIST_CAMERA_TIME_INVALID")
        prior = camera_cache.get(env_id)
        same_timestamp = bool(
            prior is not None and timestamp == float(prior.timestamp_s)
        )
        same_frame_id = bool(
            prior is not None and frame_id == int(prior.frame_id)
        )
        if same_timestamp and same_frame_id:
            # Do not assign a 50-Hz control timestamp to a reused 25-Hz frame.
            frame = prior
        else:
            if prior is not None:
                raw_identity_regressed = bool(
                    frame_id < int(prior.frame_id)
                    or timestamp < float(prior.timestamp_s)
                    or (same_timestamp and not same_frame_id)
                )
                if (
                    raw_identity_regressed
                    and not camera_cache.reset_epoch_transition_allowed(env_id)
                ):
                    raise Stage1AVectorRuntimeError(
                        "VECTOR_WRIST_TIMESTAMP_REGRESSED"
                    )
            depth, valid = preprocess_depth_m(
                raw_depth[env_id].detach().to("cpu").numpy(),
                source_valid[env_id].detach().to("cpu").numpy(),
                maximum_depth_m=2.0,
            )
            frame = PerEnvCameraFrame(
                env_id=env_id,
                frame_id=frame_id,
                timestamp_s=timestamp,
                rgb=rgb[env_id].detach().to("cpu").numpy(),
                depth_m=depth,
                depth_valid=valid,
            )
            try:
                camera_cache.put(frame)
            except Stage1AVectorContractError as error:
                raise Stage1AVectorRuntimeError(
                    "VECTOR_WRIST_CAMERA_CACHE_REJECTED"
                ) from error
        current_gripper = 1.0 if float(previous[env_id, 3]) >= 0.5 else 0.0
        inputs.append(
            HumanGraspSequenceInputs(
                right_wrist_rgb=torch.as_tensor(frame.rgb, dtype=torch.uint8, device=env.device)[None, None],
                right_wrist_depth_m=torch.as_tensor(frame.depth_m, dtype=torch.float32, device=env.device)[None, None],
                right_wrist_depth_valid=torch.as_tensor(frame.depth_valid, dtype=torch.bool, device=env.device)[None, None],
                ee_pose_robot_root_m_xyzw=torch.cat(
                    (position_t[env_id], quaternion_t[env_id]), dim=0
                )[None, None],
                right_arm_joint_position_rad=q[env_id][None, None],
                right_arm_joint_velocity_rad_s=qd[env_id][None, None],
                current_gripper_state=torch.tensor(
                    [[[current_gripper]]], dtype=torch.float32, device=env.device
                ),
                previous_policy_action_4d_metric_root_m=torch.as_tensor(
                    previous[env_id], dtype=torch.float32, device=env.device
                )[None, None],
                hidden_reset_mask=torch.tensor(
                    [[reset[env_id]]], dtype=torch.bool, device=env.device
                ),
            )
        )
        frames.append(frame)
    return tuple(inputs), tuple(frames)


def capture_vector_rgbd_sensor_frames(
    *,
    env: Any,
    camera_scene_key: str,
    camera_cache: Any,
) -> tuple[Any, ...]:
    """Capture one named RGB-D sensor with its real 25-Hz identity.

    This sidecar helper is intentionally independent from the frozen wrist
    policy input.  It is used by the deterministic-FSM sequence collector to
    persist the existing head and right-wrist sensor bindings without adding
    either camera frame, timestamp, or geometry to the SAC/BC actor input.
    """

    from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
    from geniesim.rl.sac.keyboard_v3_dataset import preprocess_depth_m
    from .stage1a_vector_contract import PerEnvCameraFrame

    if camera_scene_key not in ("head_camera", "right_wrist_camera"):
        raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_CAMERA_BINDING_UNKNOWN")
    num_envs = int(env.num_envs)
    if getattr(camera_cache, "num_envs", None) != num_envs:
        raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_CAMERA_CACHE_CARDINALITY")
    camera = env.scene[camera_scene_key]
    output = camera.data.output
    if "rgb" not in output or "distance_to_image_plane" not in output:
        raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_RGBD_STREAM_MISSING")
    rgb = _runtime_tensor(output["rgb"])[..., :3]
    raw_depth = _runtime_tensor(output["distance_to_image_plane"])
    if raw_depth.ndim == 3:
        raw_depth = raw_depth.unsqueeze(-1)
    if (
        tuple(rgb.shape) != (num_envs, 192, 256, 3)
        or rgb.dtype != torch.uint8
        or tuple(raw_depth.shape) != (num_envs, 192, 256, 1)
    ):
        raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_RGBD_CONTRACT_MISMATCH")
    raw_depth = raw_depth.to(dtype=torch.float32)
    corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (
        torch.isfinite(raw_depth) & (raw_depth < 0.0)
    )
    if bool(corrupt.any().item()):
        raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_DEPTH_CORRUPT")
    source_valid = torch.isfinite(raw_depth) & (raw_depth >= 0.0)
    frame_ids = _runtime_tensor(camera.frame).reshape(-1)
    captured, age = camera_capture_time_and_age(camera)
    capture_times = _runtime_tensor(captured).reshape(-1)
    ages = _runtime_tensor(age).reshape(-1)
    if (
        frame_ids.numel() != num_envs
        or capture_times.numel() != num_envs
        or ages.numel() != num_envs
    ):
        raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_CAMERA_IDENTITY_CARDINALITY")

    frames: list[Any] = []
    for env_id in range(num_envs):
        timestamp = float(capture_times[env_id].item())
        frame_id = int(frame_ids[env_id].item())
        sensor_time = timestamp + float(ages[env_id].item())
        if (
            not np.isfinite(timestamp)
            or not np.isfinite(sensor_time)
            or timestamp < 0.0
            or sensor_time < timestamp
            or frame_id < 0
        ):
            raise Stage1AVectorRuntimeError("VECTOR_SEQUENCE_CAMERA_TIME_INVALID")
        prior = camera_cache.get(env_id)
        same_timestamp = bool(
            prior is not None and timestamp == float(prior.timestamp_s)
        )
        same_frame_id = bool(
            prior is not None and frame_id == int(prior.frame_id)
        )
        if same_timestamp and same_frame_id:
            frame = prior
        else:
            if prior is not None:
                raw_identity_regressed = bool(
                    frame_id < int(prior.frame_id)
                    or timestamp < float(prior.timestamp_s)
                    or (same_timestamp and not same_frame_id)
                )
                if (
                    raw_identity_regressed
                    and not camera_cache.reset_epoch_transition_allowed(env_id)
                ):
                    raise Stage1AVectorRuntimeError(
                        "VECTOR_SEQUENCE_CAMERA_TIMESTAMP_REGRESSED"
                    )
            depth, valid = preprocess_depth_m(
                raw_depth[env_id].detach().to("cpu").numpy(),
                source_valid[env_id].detach().to("cpu").numpy(),
                maximum_depth_m=2.0,
            )
            frame = PerEnvCameraFrame(
                env_id=env_id,
                frame_id=frame_id,
                timestamp_s=timestamp,
                rgb=rgb[env_id].detach().to("cpu").numpy(),
                depth_m=depth,
                depth_valid=valid,
            )
            try:
                camera_cache.put(frame)
            except Stage1AVectorContractError as error:
                raise Stage1AVectorRuntimeError(
                    "VECTOR_SEQUENCE_CAMERA_CACHE_REJECTED"
                ) from error
        frames.append(frame)
    return tuple(frames)


def vector_ee_and_cube_state(*, env: Any, p0a: Any) -> Mapping[str, torch.Tensor]:
    """Read every clone's EE/root and cube/world state with explicit XYZW.

    This is the only vector runtime helper that translates world geometry to
    robot-root geometry.  It prevents an env-0 convenience helper from being
    reused in a batched rollout and therefore preserves the P0 full-SE(3)
    frame contract for each clone.
    """

    from geniesim.rl.isaaclab.g2_quaternion import (
        isaaclab_native_quaternion_order,
        quaternion_native_to_xyzw,
    )
    from geniesim.rl.isaaclab.g2_rebuild.sensor_packet import (
        world_points_to_root,
        world_quaternions_to_root,
    )

    num_envs = int(env.num_envs)
    robot = env.scene["robot"]
    cube = env.scene["object"].data
    ee_position, ee_quaternion = p0a._world_ee_pose_to_root(
        robot, env.scene["ee_frame"]
    )[:2]
    root_position = _runtime_tensor(robot.data.root_pos_w)
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        _runtime_tensor(robot.data.root_quat_w), isaaclab_native_quaternion_order()
    )
    cube_position_root = world_points_to_root(
        _runtime_tensor(cube.root_pos_w), root_position, root_quaternion_xyzw
    )
    cube_quaternion_world_xyzw = quaternion_native_to_xyzw(
        _runtime_tensor(cube.root_quat_w), isaaclab_native_quaternion_order()
    )
    cube_quaternion_root_xyzw = world_quaternions_to_root(
        cube_quaternion_world_xyzw, root_quaternion_xyzw
    )
    values = {
        "ee_position_root_m": _runtime_tensor(ee_position),
        "ee_quaternion_root_xyzw": _runtime_tensor(ee_quaternion),
        "root_position_world_m": root_position,
        "root_quaternion_world_xyzw": root_quaternion_xyzw,
        "root_linear_velocity_world_m_s": _runtime_tensor(robot.data.root_lin_vel_w),
        "root_angular_velocity_world_rad_s": _runtime_tensor(robot.data.root_ang_vel_w),
        "cube_position_root_m": cube_position_root,
        "cube_quaternion_root_xyzw": cube_quaternion_root_xyzw,
        "cube_position_world_m": _runtime_tensor(cube.root_pos_w),
        "cube_quaternion_world_xyzw": cube_quaternion_world_xyzw,
        "cube_linear_velocity_world_m_s": _runtime_tensor(cube.root_lin_vel_w),
        "cube_angular_velocity_world_rad_s": _runtime_tensor(cube.root_ang_vel_w),
    }
    for name, value in values.items():
        if value.ndim < 1 or int(value.shape[0]) != num_envs or not bool(torch.isfinite(value).all()):
            raise Stage1AVectorRuntimeError(f"VECTOR_STATE_INVALID:{name}")
    return values


def stack_single_env_reward_inputs(inputs_by_env: Sequence[Any]) -> Any:
    """Stack scalar adapter outputs into one existing ``Stage1ARewardInputs``.

    The established telemetry adapter intentionally produces a one-row input
    after proving a raw 500-Hz contact interval.  This adapter preserves that
    authority: it does not recompute any contact or reward field, it merely
    concatenates validated row-zero tensors in environment order.
    """

    from .stage1a_grasp_reward import Stage1ARewardInputs

    if not inputs_by_env or not all(isinstance(item, Stage1ARewardInputs) for item in inputs_by_env):
        raise Stage1AVectorRuntimeError("VECTOR_REWARD_INPUT_TYPE_INVALID")
    payload: dict[str, Any] = {}
    for item in fields(Stage1ARewardInputs):
        values = [getattr(value, item.name) for value in inputs_by_env]
        if values[0] is None:
            if any(value is not None for value in values):
                raise Stage1AVectorRuntimeError(f"VECTOR_REWARD_OPTIONAL_FIELD_MISMATCH:{item.name}")
            payload[item.name] = None
            continue
        if any(not isinstance(value, torch.Tensor) or value.shape[0] != 1 for value in values):
            raise Stage1AVectorRuntimeError(f"VECTOR_REWARD_FIELD_SHAPE_INVALID:{item.name}")
        payload[item.name] = torch.cat(values, dim=0)
    return Stage1ARewardInputs(**payload)


__all__ = [
    "Stage1AVectorRuntimeError",
    "VECTOR_STAGE1A_CONSUMPTION_SCHEMA",
    "VectorActionConsumptionReceipt",
    "consume_per_env_packet_once",
    "capture_vector_wrist_gru_inputs",
    "vector_ee_and_cube_state",
    "stack_single_env_reward_inputs",
]
