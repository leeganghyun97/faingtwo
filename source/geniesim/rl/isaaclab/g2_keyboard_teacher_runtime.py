# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Live RGB-D/privileged transition extraction for G2 keyboard collection."""

from __future__ import annotations

from typing import Callable, Mapping

import torch

from .g2_camera_timing import dual_camera_capture_timing
from .g2_keyboard_teacher_dataset import G2ExpertSource, G2KeyboardTeacherTransitionContract
from .g2_lift_methodology import G2LiftSandboxContract, TORSO_JOINTS
from .g2_lift_task_mdp import (
    contact_grasp_telemetry,
    object_stably_grasped_and_lifted,
)
from .g2_redundancy_teleop import tensor_value
from .g2_quaternion import (
    canonicalize_quaternion_xyzw,
    quaternion_native_to_xyzw,
)
from .g2_rebuild.isaac_runtime_adapter import (
    PadSurfaceCalibration,
    calibrated_pad_surface_midpoint_root_m,
)
from .g2_rebuild.sensor_packet import world_point_velocity_to_root, world_points_to_root
from .g2_teacher_sac import G2TeacherObservationContract
from .g2_teacher_runtime import G2TeacherRuntimeObservationBuilder
from .g2_training_readiness import audit_g2_long_training_readiness


class G2KeyboardTeacherRuntimeRecorder:
    """Capture synchronized teacher state, camera input, action and outcome."""

    def __init__(
        self,
        env,
        *,
        pad_surface_calibration: PadSurfaceCalibration,
        forbidden_collision_evaluator: Callable[[object], torch.Tensor] | None = None,
        camera_visibility_evaluator: object | None = None,
    ) -> None:
        if not isinstance(pad_surface_calibration, PadSurfaceCalibration):
            raise TypeError("explicit PadSurfaceCalibration is required")
        self.env = env
        self.builder = G2TeacherRuntimeObservationBuilder(env)
        self.contract = G2KeyboardTeacherTransitionContract()
        self.head = env.scene["head_camera"]
        self.wrist = env.scene["right_wrist_camera"]
        self.inner = env.scene["right_inner_finger_contact"]
        self.outer = env.scene["right_outer_finger_contact"]
        self.previous_teacher_action = torch.zeros((env.num_envs, 7), device=env.device)
        self.previous_teacher_action[:, 6] = 1.0
        self.sandbox = G2LiftSandboxContract().validated()
        self.teacher_contract = G2TeacherObservationContract()
        self.robot = env.scene["robot"]
        self.pad_surface_calibration = pad_surface_calibration
        self.collision_readiness = audit_g2_long_training_readiness()
        self._forbidden_collision_evaluator = forbidden_collision_evaluator
        self._camera_visibility_evaluator = camera_visibility_evaluator
        self._camera_capture_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.runtime_collision_attestation = None
        if forbidden_collision_evaluator is not None:
            attestation = getattr(forbidden_collision_evaluator, "attestation", None)
            if (
                attestation is None
                or not attestation.live_full_body_forbidden_collision_authority
            ):
                raise RuntimeError(
                    "G2_FORBIDDEN_COLLISION_EVALUATOR_WITHOUT_LIVE_RUNTIME_ATTESTATION"
                )
            self.runtime_collision_attestation = attestation
        # Bind telemetry to the active reward configuration instead of copying
        # a second, potentially drifting push threshold into the recorder.
        persistent_push_cfg = env.reward_manager.get_term_cfg("persistent_push")
        self.push_free_motion_m = float(
            persistent_push_cfg.params["free_motion_m"]
        )
        self.torso_indices = torch.tensor(
            [self.robot.joint_names.index(name) for name in TORSO_JOINTS],
            dtype=torch.long,
            device=env.device,
        )
        self.inner_distal_pad_body_index = self.robot.body_names.index(
            pad_surface_calibration.inner_link_name
        )
        self.outer_distal_pad_body_index = self.robot.body_names.index(
            pad_surface_calibration.outer_link_name
        )
        self.episode_id = 0
        self.sequence_step = 0
        self._episode_start_env_step = tensor_value(env.episode_length_buf).clone()
        self._before = None

    @staticmethod
    def _quaternion_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_xyz, left_w = left[..., :3], left[..., 3:4]
        right_xyz, right_w = right[..., :3], right[..., 3:4]
        return canonicalize_quaternion_xyzw(
            torch.cat(
                (
                    left_w * right_xyz
                    + right_w * left_xyz
                    + torch.linalg.cross(left_xyz, right_xyz, dim=-1),
                    left_w * right_w - (left_xyz * right_xyz).sum(
                        dim=-1, keepdim=True
                    ),
                ),
                dim=-1,
            )
        )

    def _camera_pose_robot_root_xyzw(
        self,
        camera,
        *,
        root_position_world_m: torch.Tensor,
        root_quaternion_world_xyzw: torch.Tensor,
    ) -> torch.Tensor:
        """Read the optical pose captured with the current RGB-D frame.

        Isaac Lab updates ``CameraData.pos_w``/``quat_w_ros`` in the same
        camera-buffer transaction as RGB-D and the frame counter.  A live USD
        transform is deliberately not used here because it can be newer than
        a repeated image.
        """

        position_world = tensor_value(camera.data.pos_w).to(torch.float32)
        quaternion_world = canonicalize_quaternion_xyzw(
            tensor_value(camera.data.quat_w_ros).to(torch.float32)
        )
        position_root = world_points_to_root(
            position_world,
            root_position_world_m,
            root_quaternion_world_xyzw,
        )
        inverse_root = root_quaternion_world_xyzw.clone()
        inverse_root[..., :3] *= -1.0
        quaternion_root = self._quaternion_multiply_xyzw(
            inverse_root, quaternion_world
        )
        return torch.cat((position_root, quaternion_root), dim=-1).to(torch.float32)

    def _capture_camera_authority(
        self,
        *,
        camera_timestamp_s: torch.Tensor,
        local_episode_time_s: torch.Tensor,
        cube_pose_robot_root_xyzw: torch.Tensor,
        root_position_world_m: torch.Tensor,
        root_quaternion_world_xyzw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Freeze RGB-D, optical pose, cube snapshot and visibility per frame id.

        A new camera sequence is accepted only when its capture timestamp is
        the current pre-action physics state.  When a renderer repeats a frame,
        every capture-authored value is reused from the cache; current cube/FK
        state is never substituted for stale RGB-D.
        """

        if self._camera_visibility_evaluator is None:
            raise RuntimeError(
                "G2_M1_EXPLICIT_CUBE_VISIBILITY_AUTHORITY_REQUIRED"
            )
        visibility = self._camera_visibility_evaluator.evaluate(
            require_rendered_depth=True
        )
        result: dict[str, torch.Tensor] = {}
        for camera_index, (camera_name, camera) in enumerate(
            (("head", self.head), ("right_wrist", self.wrist))
        ):
            sequence = tensor_value(camera.frame).to(
                device=local_episode_time_s.device, dtype=torch.int64
            )
            if sequence.ndim == 0:
                sequence = sequence.expand(local_episode_time_s.shape[0])
            timestamp = camera_timestamp_s[:, camera_index]
            cache = self._camera_capture_cache.get(camera_name)
            if cache is not None and bool(torch.any(sequence < cache["sequence_id"]).item()):
                raise RuntimeError(
                    f"G2_M1_CAMERA_SEQUENCE_MOVED_BACKWARD:{camera_name}"
                )
            repeated = (
                torch.zeros_like(sequence, dtype=torch.bool)
                if cache is None
                else sequence == cache["sequence_id"]
            )
            new = ~repeated
            if bool(new.any().item()):
                # No interpolation or nearest control-state substitution is
                # permitted.  If a new renderer frame is already stale at the
                # only available state boundary, exact cube supervision cannot
                # be authored and M1 is blocked explicitly.
                mismatch = torch.abs(timestamp - local_episode_time_s) > 1.0e-6
                if bool((new & mismatch).any().item()):
                    raise RuntimeError(
                        "G2_M1_CAPTURE_TIME_CUBE_SNAPSHOT_UNAVAILABLE:"
                        f"{camera_name}"
                    )
            if cache is not None and bool(
                (repeated & (torch.abs(timestamp - cache["timestamp_s"]) > 1.0e-7)).any().item()
            ):
                raise RuntimeError(
                    f"G2_M1_REPEATED_CAMERA_TIMESTAMP_CHANGED:{camera_name}"
                )

            rgb = tensor_value(camera.data.output["rgb"])[..., :3].clone()
            depth = self._depth_image(
                tensor_value(camera.data.output["distance_to_image_plane"])
            ).to(torch.float32).clone()
            depth_valid = torch.isfinite(depth) & (depth > 0.0)
            depth = torch.where(depth_valid, depth, torch.zeros_like(depth))
            pose = self._camera_pose_robot_root_xyzw(
                camera,
                root_position_world_m=root_position_world_m,
                root_quaternion_world_xyzw=root_quaternion_world_xyzw,
            )
            camera_visibility = visibility.cameras[camera_name]
            visible = camera_visibility.rendered_cube_visible
            if visible is None:
                raise RuntimeError(
                    f"G2_M1_CUBE_VISIBILITY_UNMEASURED:{camera_name}"
                )
            visible = visible.to(device=sequence.device, dtype=torch.bool)
            visibility_valid = torch.ones_like(visible, dtype=torch.bool)
            candidate = {
                "sequence_id": sequence.clone(),
                "timestamp_s": timestamp.clone(),
                "rgb": rgb,
                "depth_m": depth,
                "depth_valid": depth_valid,
                "camera_pose_robot_root_xyzw": pose,
                "cube_gt_pose_at_capture_robot_root_xyzw": (
                    cube_pose_robot_root_xyzw.clone()
                ),
                "cube_visible": visible.unsqueeze(-1),
                "cube_visibility_valid": visibility_valid.unsqueeze(-1),
            }
            if cache is not None and bool(repeated.any().item()):
                for field, current in tuple(candidate.items()):
                    previous = cache[field]
                    mask = repeated.reshape(
                        (repeated.shape[0],) + (1,) * (current.ndim - 1)
                    )
                    candidate[field] = torch.where(mask, previous, current)
            self._camera_capture_cache[camera_name] = {
                field: value.detach().clone() for field, value in candidate.items()
            }
            prefix = "head" if camera_name == "head" else "right_wrist"
            for field, value in candidate.items():
                result[f"{prefix}_{field}"] = value
        result["camera_sequence_id"] = torch.stack(
            (
                result["head_sequence_id"],
                result["right_wrist_sequence_id"],
            ),
            dim=-1,
        )
        return result

    def forbidden_collision_telemetry(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return collision value and validity without inventing a safe zero."""

        shape = (self.env.num_envs, 1)
        if self._forbidden_collision_evaluator is None:
            return (
                torch.zeros(shape, dtype=torch.bool, device=self.env.device),
                torch.zeros(shape, dtype=torch.bool, device=self.env.device),
            )
        value = self._forbidden_collision_evaluator(self.env)
        if value.shape == (self.env.num_envs,):
            value = value.unsqueeze(-1)
        if value.shape != shape or value.dtype != torch.bool:
            raise ValueError(
                "forbidden collision evaluator must return bool tensor [N,1]"
            )
        return value, torch.ones_like(value)

    @staticmethod
    def _depth_image(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(-1) if value.ndim == 3 else value

    def reset(self, env_ids=None, *, episode_id: int | None = None) -> None:
        if env_ids is None:
            self.previous_teacher_action.zero_()
            self.previous_teacher_action[:, 6] = 1.0
        else:
            self.previous_teacher_action[env_ids] = 0.0
            self.previous_teacher_action[env_ids, 6] = 1.0
        if episode_id is not None:
            self.episode_id = int(episode_id)
        self.sequence_step = 0
        # Collection may deliberately run unrecorded reset settling (for
        # example, opening the gripper for 60 policy steps).  Preserve the
        # environment counter but make the dataset clock relative to the end
        # of that settling period.
        self._episode_start_env_step = tensor_value(
            self.env.episode_length_buf
        ).clone()
        self._before = None
        self._camera_capture_cache.clear()

    def begin_step(
        self,
        *,
        keyboard_physical_command: torch.Tensor,
        normalized_teleop_action: torch.Tensor,
        timestamp_s: torch.Tensor,
        expert_source: G2ExpertSource = G2ExpertSource.KEYBOARD,
        remote_metadata: Mapping[str, float | int | str] | None = None,
    ) -> None:
        """Capture actor inputs immediately before applying the action."""

        batch = self.env.num_envs
        if keyboard_physical_command.shape != (batch, 8):
            raise ValueError("keyboard_physical_command must have shape (N, 8)")
        if normalized_teleop_action.shape != (batch, 8):
            raise ValueError("normalized_teleop_action must have shape (N, 8)")
        if timestamp_s.shape != (batch, 1):
            raise ValueError("timestamp_s must have shape (N, 1)")
        if expert_source not in (
            G2ExpertSource.KEYBOARD,
            G2ExpertSource.REMOTE_KEYBOARD,
        ):
            raise ValueError("runtime teleop recorder requires a keyboard source")
        if expert_source is G2ExpertSource.REMOTE_KEYBOARD:
            if remote_metadata is None:
                raise ValueError("remote keyboard recording requires packet provenance")
            remote_sequence_id = int(remote_metadata["sequence_id"])
            remote_source_timestamp = float(remote_metadata["source_timestamp_s"])
            remote_receive_age = float(remote_metadata["receive_age_s"])
            if remote_sequence_id < 0 or remote_receive_age < 0.0:
                raise ValueError("invalid remote keyboard packet provenance")
            remote_command_valid = 1
        else:
            if remote_metadata is not None:
                raise ValueError("local keyboard rows cannot carry remote provenance")
            remote_sequence_id = -1
            remote_source_timestamp = 0.0
            remote_receive_age = 0.0
            remote_command_valid = 0
        local_episode_time = (
            (
                tensor_value(self.env.episode_length_buf)
                - self._episode_start_env_step
            ).to(torch.float32)
            * float(self.env.step_dt)
        )
        if bool(
            (
                torch.abs(timestamp_s.reshape(-1) - local_episode_time)
                > max(float(self.env.step_dt), 1.0e-6)
            ).any()
        ):
            raise ValueError("timestamp_s differs from the episode-local simulation clock")
        camera_timestamp, camera_frame_age = dual_camera_capture_timing(
            self.head,
            self.wrist,
            fallback_episode_time_s=local_episode_time,
        )
        joint_position = tensor_value(self.robot.data.joint_pos)
        default_position = tensor_value(self.robot.data.default_joint_pos)
        joint_velocity = tensor_value(self.robot.data.joint_vel)
        inner_force, outer_force, bilateral, slip_speed, stable_grasp = contact_grasp_telemetry(
            self.env
        )
        cube_world = tensor_value(self.env.scene["object"].data.root_pos_w)
        cube_planar_displacement = torch.linalg.vector_norm(
            cube_world[:, :2] - self.env._g2_task_initial_cube_w[:, :2], dim=-1
        )
        push_before_grasp_trial = (
            (cube_planar_displacement > self.push_free_motion_m) & (~bilateral)
        )
        body_position_world_m = tensor_value(self.robot.data.body_pos_w)
        body_quaternion_world_xyzw = quaternion_native_to_xyzw(
            tensor_value(self.robot.data.body_quat_w),
            self.builder.native_quaternion_order,
        )
        root_position_world_m = tensor_value(self.robot.data.root_pos_w)
        root_quaternion_world_xyzw = quaternion_native_to_xyzw(
            tensor_value(self.robot.data.root_quat_w),
            self.builder.native_quaternion_order,
        )
        teacher_observation = self.builder.build(self.previous_teacher_action).clone()
        cube_pose_robot_root_xyzw = teacher_observation[
            :, self.teacher_contract.slices["cube_pose_root_xyzw"]
        ]
        camera_capture = self._capture_camera_authority(
            camera_timestamp_s=camera_timestamp,
            local_episode_time_s=local_episode_time,
            cube_pose_robot_root_xyzw=cube_pose_robot_root_xyzw,
            root_position_world_m=root_position_world_m,
            root_quaternion_world_xyzw=root_quaternion_world_xyzw,
        )
        ee_position_world_m = tensor_value(
            self.env.scene["ee_frame"].data.target_pos_w
        )[:, 0]
        ee_body_origin_world_m = body_position_world_m[:, self.builder.ee_index]
        ee_linear_velocity_world_m_s = tensor_value(
            self.robot.data.body_lin_vel_w
        )[:, self.builder.ee_index] + torch.linalg.cross(
            tensor_value(self.robot.data.body_ang_vel_w)[:, self.builder.ee_index],
            ee_position_world_m - ee_body_origin_world_m,
            dim=-1,
        )
        ee_linear_velocity_robot_root_m_s = world_point_velocity_to_root(
            ee_position_world_m,
            ee_linear_velocity_world_m_s,
            root_position_world_m,
            root_quaternion_world_xyzw,
            tensor_value(self.robot.data.root_lin_vel_w),
            tensor_value(self.robot.data.root_ang_vel_w),
        )
        if tuple(ee_linear_velocity_robot_root_m_s.shape) != (batch, 3) or not bool(
            torch.isfinite(ee_linear_velocity_robot_root_m_s).all()
        ):
            raise RuntimeError("G2_EE_ROOT_RELATIVE_VELOCITY_INVALID")
        grasp_center_position_root_m = calibrated_pad_surface_midpoint_root_m(
            body_position_world_m=body_position_world_m,
            body_quaternion_world_xyzw=body_quaternion_world_xyzw,
            root_position_world_m=root_position_world_m,
            root_quaternion_world_xyzw=root_quaternion_world_xyzw,
            inner_body_index=self.inner_distal_pad_body_index,
            outer_body_index=self.outer_distal_pad_body_index,
            calibration=self.pad_surface_calibration,
        )
        if tuple(grasp_center_position_root_m.shape) != (batch, 3) or not bool(
            torch.isfinite(grasp_center_position_root_m).all()
        ):
            raise RuntimeError("G2_DISTAL_PAD_MIDPOINT_FK_INVALID")
        self._before = {
            "teacher_observation": teacher_observation,
            "grasp_center_position_root_m": grasp_center_position_root_m.clone(),
            # Complete pre-action articulation state for the controlled-
            # rebuild recorder.  This is kept separate from the legacy 59-D
            # teacher observation so no eight-joint control vector can be
            # mislabeled as the physical robot state.
            "robot_joint_position_rad": joint_position.clone(),
            "robot_joint_velocity_rad_s": joint_velocity.clone(),
            "ee_linear_velocity_robot_root_m_s": (
                ee_linear_velocity_robot_root_m_s.clone()
            ),
            "cube_center_height_world_m": cube_world[:, 2:3].clone(),
            "keyboard_physical_command": keyboard_physical_command.clone(),
            "teleop_action": normalized_teleop_action.clone(),
            "head_rgb": camera_capture["head_rgb"],
            "head_depth": camera_capture["head_depth_m"],
            "head_depth_valid": camera_capture["head_depth_valid"],
            "right_wrist_rgb": camera_capture["right_wrist_rgb"],
            "right_wrist_depth": camera_capture["right_wrist_depth_m"],
            "right_wrist_depth_valid": camera_capture[
                "right_wrist_depth_valid"
            ],
            "head_camera_pose_robot_root_xyzw": camera_capture[
                "head_camera_pose_robot_root_xyzw"
            ],
            "right_wrist_camera_pose_robot_root_xyzw": camera_capture[
                "right_wrist_camera_pose_robot_root_xyzw"
            ],
            "camera_sequence_id": camera_capture["camera_sequence_id"],
            "head_cube_gt_pose_at_capture_robot_root_xyzw": camera_capture[
                "head_cube_gt_pose_at_capture_robot_root_xyzw"
            ],
            "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw": camera_capture[
                "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw"
            ],
            "head_cube_visible": camera_capture["head_cube_visible"],
            "right_wrist_cube_visible": camera_capture[
                "right_wrist_cube_visible"
            ],
            "head_cube_visibility_valid": camera_capture[
                "head_cube_visibility_valid"
            ],
            "right_wrist_cube_visibility_valid": camera_capture[
                "right_wrist_cube_visibility_valid"
            ],
            "timestamp": local_episode_time.unsqueeze(-1).clone(),
            "camera_timestamp": camera_timestamp,
            "camera_frame_age": camera_frame_age,
            "torso_joint_position_relative_rad": (
                joint_position.index_select(1, self.torso_indices)
                - default_position.index_select(1, self.torso_indices)
            ).clone(),
            "torso_joint_velocity_rad_s": joint_velocity.index_select(
                1, self.torso_indices
            ).clone(),
            "stable_grasp": stable_grasp.unsqueeze(-1).clone(),
            "slip_speed_m_s": slip_speed.unsqueeze(-1).clone(),
            "contact_force_n": torch.stack(
                (inner_force, outer_force), dim=-1
            ).clone(),
            "bilateral_contact": bilateral.unsqueeze(-1).clone(),
            "cube_planar_displacement_m": cube_planar_displacement.unsqueeze(-1).clone(),
            "push_before_grasp_trial": push_before_grasp_trial.unsqueeze(-1).clone(),
            "expert_source_code": int(expert_source),
            "remote_command_valid": remote_command_valid,
            "remote_sequence_id": remote_sequence_id,
            "remote_source_timestamp": remote_source_timestamp,
            "remote_receive_age": remote_receive_age,
        }

    def controlled_rebuild_pre_action_snapshot(self) -> dict[str, torch.Tensor]:
        """Return an immutable clone of the aligned pre-action evidence.

        The native v2 writer calls this after :meth:`begin_step` and before
        ``env.step``.  Exposing a clone avoids reaching into ``_before`` and,
        crucially, prevents a post-step articulation read from being paired
        with the pre-action RGB-D packet.
        """

        if self._before is None:
            raise RuntimeError("begin_step must precede controlled-rebuild snapshot")
        result: dict[str, torch.Tensor] = {}
        for name, value in self._before.items():
            if isinstance(value, torch.Tensor):
                result[name] = value.detach().clone()
        required = {
            "teacher_observation",
            "grasp_center_position_root_m",
            "robot_joint_position_rad",
            "robot_joint_velocity_rad_s",
            "ee_linear_velocity_robot_root_m_s",
            "cube_center_height_world_m",
            "keyboard_physical_command",
            "teleop_action",
            "head_rgb",
            "head_depth",
            "head_depth_valid",
            "right_wrist_rgb",
            "right_wrist_depth",
            "right_wrist_depth_valid",
            "head_camera_pose_robot_root_xyzw",
            "right_wrist_camera_pose_robot_root_xyzw",
            "camera_sequence_id",
            "head_cube_gt_pose_at_capture_robot_root_xyzw",
            "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw",
            "head_cube_visible",
            "right_wrist_cube_visible",
            "head_cube_visibility_valid",
            "right_wrist_cube_visibility_valid",
            "timestamp",
            "camera_timestamp",
            "camera_frame_age",
            "stable_grasp",
            "contact_force_n",
        }
        missing = sorted(required.difference(result))
        if missing:
            raise RuntimeError(f"controlled-rebuild pre-action snapshot incomplete: {missing}")
        return result

    def end_step(
        self,
        *,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        outcome_code: torch.Tensor,
        forbidden_collision: torch.Tensor | None = None,
        forbidden_collision_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Capture outputs after ``env.step`` and return one complete row batch."""

        if self._before is None:
            raise RuntimeError("begin_step must be called before end_step")
        arm_term = self.env.action_manager._terms["arm_action"]
        gripper_term = self.env.action_manager._terms["gripper_action"]
        # ``raw_actions`` is the normalized controller request.  It is not an
        # actuator-level applied action: IK, joint-limit clipping and the
        # per-physics-step rate limiter still follow.  Record both surfaces.
        controller_request = torch.cat(
            (arm_term.raw_actions, gripper_term.raw_actions), dim=-1
        )
        emitted_joint_target = torch.cat(
            (
                arm_term.last_emitted_joint_position_target,
                gripper_term.last_emitted_joint_position_target,
            ),
            dim=-1,
        )
        teacher_action = torch.cat(
            (controller_request[:, :6], controller_request[:, 7:8]), dim=-1
        )
        next_teacher = self.builder.build(teacher_action)
        slices = self.teacher_contract.slices
        success = object_stably_grasped_and_lifted(self.env).unsqueeze(-1)
        before = self._before
        if forbidden_collision is None or forbidden_collision_valid is None:
            if forbidden_collision is not None or forbidden_collision_valid is not None:
                raise ValueError(
                    "collision value and validity must be supplied together"
                )
            forbidden_collision, forbidden_collision_valid = (
                self.forbidden_collision_telemetry()
            )
        result = self.contract.build(
            teacher_observation=before["teacher_observation"],
            next_teacher_observation=next_teacher,
            grasp_center_position_root_m=before[
                "grasp_center_position_root_m"
            ],
            keyboard_physical_command=before["keyboard_physical_command"],
            teleop_action=before["teleop_action"],
            # Retained v2 field: this is now explicitly a controller request,
            # while ``applied_actuator_joint_target_rad`` is the authority.
            applied_teleop_action=controller_request,
            applied_actuator_joint_target_rad=emitted_joint_target,
            expert_source_code=torch.full(
                (self.env.num_envs, 1),
                before["expert_source_code"],
                dtype=torch.uint8,
                device=self.env.device,
            ),
            remote_command_valid=torch.full(
                (self.env.num_envs, 1),
                before["remote_command_valid"],
                dtype=torch.uint8,
                device=self.env.device,
            ),
            remote_sequence_id=torch.full(
                (self.env.num_envs, 1),
                before["remote_sequence_id"],
                dtype=torch.int64,
                device=self.env.device,
            ),
            remote_source_timestamp=torch.full(
                (self.env.num_envs, 1),
                before["remote_source_timestamp"],
                dtype=torch.float64,
                device=self.env.device,
            ),
            remote_receive_age=torch.full(
                (self.env.num_envs, 1),
                before["remote_receive_age"],
                dtype=torch.float32,
                device=self.env.device,
            ),
            expert_confidence=torch.ones(
                (self.env.num_envs, 1), device=self.env.device
            ),
            episode_id=torch.full(
                (self.env.num_envs, 1), self.episode_id,
                dtype=torch.int64, device=self.env.device,
            ),
            sequence_step=torch.full(
                (self.env.num_envs, 1), self.sequence_step,
                dtype=torch.int64, device=self.env.device,
            ),
            processed_arm_command=arm_term.processed_actions,
            nullspace_joint_delta=arm_term._last_nullspace_joint_delta,
            nullspace_task_leakage=arm_term._last_nullspace_task_leakage.unsqueeze(-1),
            nullspace_limit_clipped=arm_term._last_nullspace_clipped.unsqueeze(-1),
            head_rgb=before["head_rgb"],
            head_depth=before["head_depth"],
            right_wrist_rgb=before["right_wrist_rgb"],
            right_wrist_depth=before["right_wrist_depth"],
            # Contact/slip/stable labels share the pre-action observation and
            # camera timestamp.  Post-action force belongs to the next row and
            # would otherwise create a one-transition target leak.
            contact_force_n=before["contact_force_n"],
            bilateral_contact=before["bilateral_contact"],
            stable_grasp=before["stable_grasp"],
            slip_speed_m_s=before["slip_speed_m_s"],
            cube_planar_displacement_m=before["cube_planar_displacement_m"],
            push_before_grasp_trial=before["push_before_grasp_trial"],
            # RewardManager._step_reward stores weighted reward *rates*
            # (value / dt).  Dataset components are transition rewards and
            # must sum to the returned scalar reward in the same unit.
            reward_components=(
                self.env.reward_manager._step_reward * float(self.env.step_dt)
            ),
            reward=reward.unsqueeze(-1) if reward.ndim == 1 else reward,
            terminated=terminated.unsqueeze(-1) if terminated.ndim == 1 else terminated,
            truncated=truncated.unsqueeze(-1) if truncated.ndim == 1 else truncated,
            success=success,
            outcome_code=outcome_code,
            forbidden_collision=forbidden_collision,
            forbidden_collision_valid=forbidden_collision_valid,
            timestamp=before["timestamp"],
            camera_timestamp=before["camera_timestamp"],
            camera_frame_age=before["camera_frame_age"],
            torso_joint_position_relative_rad=before[
                "torso_joint_position_relative_rad"
            ],
            torso_joint_velocity_rad_s=before["torso_joint_velocity_rad_s"],
        )
        self.previous_teacher_action = teacher_action.detach().clone()
        self.sequence_step += 1
        self._before = None
        return result


__all__ = ["G2KeyboardTeacherRuntimeRecorder"]
