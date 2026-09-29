# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-environment 500-Hz contact evidence for the Stage-1A vector runner.

The historical ``_PassiveContactPhysicsTelemetry`` diagnostic is intentionally
an env-0, deep-mechanics instrument.  Reusing it for a batched learner would
silently read clone zero ten times.  This smaller recorder has one job: retain
the raw pad/Object evidence required by the existing Stage-1A reward adapter,
with every contact-pair index resolved from ``env_id``.

It is read-only.  It wraps ``scene.update`` once, calls the original exactly
once, and only then copies public PhysX/Isaac buffers.  It has no controller,
ActionManager, reward, reset, or safety authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch

from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
from geniesim.rl.isaaclab.g2_quaternion import (
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
)


VECTOR_PHYSICS_TELEMETRY_SCHEMA = "g2_stage1a_vector_500hz_pad_telemetry_v1"
_CONTACT_CAPACITY = 32
_SIDE_SENSORS = (
    ("inner", "right_inner_finger_contact", "gripper_r_inner_link4"),
    ("outer", "right_outer_finger_contact", "gripper_r_outer_link4"),
)


class Stage1AVectorTelemetryError(RuntimeError):
    pass


def _tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    return torch.from_dlpack(value)


def _cpu(value: Any, *, dtype: Any) -> np.ndarray:
    return _tensor(value).detach().to("cpu").numpy().astype(dtype, copy=False)


@dataclass(frozen=True)
class VectorTelemetryContext:
    policy_step: int
    phase_by_env: tuple[str, ...]
    gripper_intent_by_env: tuple[str, ...]
    close_onset_by_env: tuple[bool, ...]


class VectorPassiveContactPhysicsTelemetry:
    """Capture ten source-owned substeps per 50-Hz vector action."""

    def __init__(self, *, env: Any, task_mdp: Any) -> None:
        self.env = env
        self.task_mdp = task_mdp
        self.num_envs = int(env.num_envs)
        if self.num_envs <= 0:
            raise Stage1AVectorTelemetryError("VECTOR_TELEMETRY_NUM_ENVS_INVALID")
        self.robot = env.scene["robot"]
        self._original_update = env.scene.update
        self._installed = False
        self._context: VectorTelemetryContext | None = None
        self._substep = 0
        self._global_sample = 0
        self._previous_qd: np.ndarray | None = None
        self._records: list[list[dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self._sources = self._bind_sources()

    @staticmethod
    def _view(sensor: Any) -> Any:
        result = getattr(sensor, "contact_view", None)
        if result is None:
            result = getattr(sensor, "contact_physx_view", None)
        if result is None:
            raise Stage1AVectorTelemetryError("VECTOR_CONTACT_VIEW_UNAVAILABLE")
        return result

    def _bind_sources(self) -> dict[str, Mapping[str, Any]]:
        body_names = tuple(str(name) for name in self.robot.body_names)
        sources: dict[str, Mapping[str, Any]] = {}
        for side, sensor_key, body_name in _SIDE_SENSORS:
            if body_name not in body_names:
                raise Stage1AVectorTelemetryError(
                    f"VECTOR_PAD_BODY_MISSING:{body_name}"
                )
            sensor = self.env.scene[sensor_key]
            view = self._view(sensor)
            if not callable(getattr(view, "get_contact_data", None)):
                raise Stage1AVectorTelemetryError(
                    f"VECTOR_CONTACT_DATA_API_MISSING:{sensor_key}"
                )
            if not callable(getattr(view, "get_friction_data", None)):
                raise Stage1AVectorTelemetryError(
                    f"VECTOR_FRICTION_DATA_API_MISSING:{sensor_key}"
                )
            sources[side] = {
                "view": view,
                "body_index": body_names.index(body_name),
                "partner_actor": "/World/envs/env_{env_id}/Object",
            }
        return sources

    def install(self) -> None:
        if self._installed:
            raise Stage1AVectorTelemetryError("VECTOR_TELEMETRY_ALREADY_INSTALLED")

        def wrapped_update(dt: float) -> Any:
            result = self._original_update(dt)
            if self._context is not None:
                self._capture(float(dt))
            return result

        self.env.scene.update = wrapped_update
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self.env.scene.update = self._original_update
            self._installed = False

    def begin_packet(
        self,
        *,
        policy_step: int,
        phase_by_env: tuple[str, ...],
        gripper_intent_by_env: tuple[str, ...],
        close_onset_by_env: tuple[bool, ...],
    ) -> tuple[int, ...]:
        if self._context is not None:
            raise Stage1AVectorTelemetryError("VECTOR_TELEMETRY_PACKET_ALREADY_ACTIVE")
        if not (
            len(phase_by_env) == len(gripper_intent_by_env) == len(close_onset_by_env) == self.num_envs
        ):
            raise Stage1AVectorTelemetryError("VECTOR_TELEMETRY_CONTEXT_CARDINALITY")
        self._context = VectorTelemetryContext(
            policy_step=int(policy_step),
            phase_by_env=tuple(str(value) for value in phase_by_env),
            gripper_intent_by_env=tuple(str(value) for value in gripper_intent_by_env),
            close_onset_by_env=tuple(bool(value) for value in close_onset_by_env),
        )
        self._substep = 0
        return tuple(len(rows) for rows in self._records)

    def end_packet(self, starts: tuple[int, ...]) -> tuple[tuple[dict[str, Any], ...], ...]:
        if self._context is None:
            raise Stage1AVectorTelemetryError("VECTOR_TELEMETRY_PACKET_NOT_ACTIVE")
        if len(starts) != self.num_envs:
            raise Stage1AVectorTelemetryError("VECTOR_TELEMETRY_START_CARDINALITY")
        result = tuple(tuple(self._records[index][starts[index]:]) for index in range(self.num_envs))
        self._context = None
        if any(len(rows) != 10 for rows in result):
            raise Stage1AVectorTelemetryError("VECTOR_PHYSICS_SUBSTEP_COUNT_NOT_10")
        return result

    def reset_envs(self, env_ids: tuple[int, ...]) -> None:
        """Drop only the terminated clones' differencing history."""

        if self._previous_qd is not None:
            self._previous_qd[list(env_ids)] = np.nan

    @staticmethod
    def _padded(values: np.ndarray, *, width: int | None = None) -> np.ndarray:
        shape = (_CONTACT_CAPACITY,) if width is None else (_CONTACT_CAPACITY, width)
        output = np.full(shape, np.nan, dtype=np.float64)
        output[: values.shape[0]] = values
        return output

    def _pair_index(self, counts: np.ndarray, starts: np.ndarray, env_id: int, side: str) -> int:
        if counts.size != starts.size or counts.size == 0 or counts.size % self.num_envs != 0:
            raise Stage1AVectorTelemetryError(
                f"VECTOR_CONTACT_PAIR_LAYOUT_INVALID:{side}:{counts.size}:{self.num_envs}"
            )
        pairs_per_env = counts.size // self.num_envs
        # Filter index zero is explicitly Object for every existing pad sensor.
        return env_id * pairs_per_env

    def _raw_side(
        self,
        *,
        side: str,
        env_id: int,
        dt_s: float,
        body_link_pose_world_xyzw: np.ndarray,
        body_link_velocity_world: np.ndarray,
        cube_position_world_m: np.ndarray,
        cube_linear_velocity_world_m_s: np.ndarray,
        cube_angular_velocity_world_rad_s: np.ndarray,
    ) -> dict[str, Any]:
        source = self._sources[side]
        view = source["view"]
        force, point, normal, _separation, counts, starts = tuple(view.get_contact_data(dt=dt_s))
        native_force, _native_point, _native_normal, _native_sep, native_counts, native_starts = tuple(
            view.get_contact_data(dt=1.0)
        )
        force_a = _cpu(force, dtype=np.float64).reshape(-1)
        point_a = _cpu(point, dtype=np.float64).reshape(-1, 3)
        normal_a = _cpu(normal, dtype=np.float64).reshape(-1, 3)
        counts_a = _cpu(counts, dtype=np.int64).reshape(-1)
        starts_a = _cpu(starts, dtype=np.int64).reshape(-1)
        native_force_a = _cpu(native_force, dtype=np.float64).reshape(-1)
        native_counts_a = _cpu(native_counts, dtype=np.int64).reshape(-1)
        native_starts_a = _cpu(native_starts, dtype=np.int64).reshape(-1)
        pair = self._pair_index(counts_a, starts_a, env_id, side)
        native_pair = self._pair_index(native_counts_a, native_starts_a, env_id, side)
        count, start = int(counts_a[pair]), int(starts_a[pair])
        if count < 0 or count > _CONTACT_CAPACITY or start < 0 or start + count > force_a.size:
            raise Stage1AVectorTelemetryError(f"VECTOR_CONTACT_SLICE_INVALID:{side}:{env_id}")
        if int(native_counts_a[native_pair]) != count or int(native_starts_a[native_pair]) != start:
            raise Stage1AVectorTelemetryError(f"VECTOR_NATIVE_CONTACT_PAIR_MISMATCH:{side}:{env_id}")
        selected = slice(start, start + count)
        normal_active = normal_a[selected]
        normal_norm = np.linalg.norm(normal_active, axis=1, keepdims=True)
        normal_unit = np.divide(normal_active, normal_norm, out=np.zeros_like(normal_active), where=normal_norm > 1.0e-12)
        body_index = int(source["body_index"])
        body_position = np.asarray(
            body_link_pose_world_xyzw[body_index, :3], dtype=np.float64
        )
        body_linear = np.asarray(
            body_link_velocity_world[body_index, :3], dtype=np.float64
        )
        body_angular = np.asarray(
            body_link_velocity_world[body_index, 3:6], dtype=np.float64
        )
        if count:
            pad_point_velocity = body_linear + np.cross(
                np.broadcast_to(body_angular, (count, 3)),
                point_a[selected] - body_position,
            )
            cube_point_velocity = cube_linear_velocity_world_m_s + np.cross(
                np.broadcast_to(cube_angular_velocity_world_rad_s, (count, 3)),
                point_a[selected] - cube_position_world_m,
            )
            relative_velocity = cube_point_velocity - pad_point_velocity
            relative_normal_velocity = np.sum(
                relative_velocity * normal_unit, axis=1
            )
            tangential_velocity = (
                relative_velocity
                - relative_normal_velocity[:, None] * normal_unit
            )
            slip_speed = np.linalg.norm(tangential_velocity, axis=1)
        else:
            relative_normal_velocity = np.empty((0,), dtype=np.float64)
            slip_speed = np.empty((0,), dtype=np.float64)
        return {
            "count": count,
            "normal_force_n": self._padded(force_a[selected]),
            "native_normal_impulse_ns": self._padded(native_force_a[selected]),
            "point_world_m": self._padded(point_a[selected], width=3),
            "normal_world": self._padded(normal_unit, width=3),
            "slip_speed_m_s": self._padded(slip_speed),
            "relative_normal_velocity_m_s": self._padded(relative_normal_velocity),
        }

    def _capture(self, dt_s: float) -> None:
        context = self._context
        assert context is not None
        if not math.isclose(dt_s, 0.002, rel_tol=0.0, abs_tol=1.0e-9):
            raise Stage1AVectorTelemetryError("VECTOR_PHYSICS_DT_NOT_500HZ")
        q = _cpu(self.robot.data.joint_pos, dtype=np.float64)
        qd = _cpu(self.robot.data.joint_vel, dtype=np.float64)
        if q.shape[0] != self.num_envs or qd.shape != q.shape:
            raise Stage1AVectorTelemetryError("VECTOR_Q_QD_BATCH_SHAPE_INVALID")
        qdd = np.full_like(qd, np.nan)
        if self._previous_qd is not None:
            valid = np.isfinite(self._previous_qd).all(axis=1)
            qdd[valid] = (qd[valid] - self._previous_qd[valid]) / dt_s
        self._previous_qd = qd.copy()
        root_quat_xyzw = _cpu(
            quaternion_native_to_xyzw(
                _tensor(self.robot.data.root_quat_w), isaaclab_native_quaternion_order()
            ),
            dtype=np.float64,
        )
        root_pos = _cpu(self.robot.data.root_pos_w, dtype=np.float64)
        root_lin = _cpu(self.robot.data.root_lin_vel_w, dtype=np.float64)
        root_ang = _cpu(self.robot.data.root_ang_vel_w, dtype=np.float64)
        body_link_pose = _cpu(self.robot.data.body_link_pose_w, dtype=np.float64)
        body_link_velocity = _cpu(self.robot.data.body_link_vel_w, dtype=np.float64)
        cube = self.env.scene["object"]
        cube_position = _cpu(cube.data.root_pos_w, dtype=np.float64)
        cube_linear_velocity = _cpu(cube.data.root_lin_vel_w, dtype=np.float64)
        cube_angular_velocity = _cpu(cube.data.root_ang_vel_w, dtype=np.float64)
        if (
            body_link_pose.shape[:2] != (self.num_envs, len(self.robot.body_names))
            or body_link_pose.shape[-1] != 7
            or body_link_velocity.shape != (self.num_envs, len(self.robot.body_names), 6)
            or cube_position.shape != (self.num_envs, 3)
            or cube_linear_velocity.shape != (self.num_envs, 3)
            or cube_angular_velocity.shape != (self.num_envs, 3)
        ):
            raise Stage1AVectorTelemetryError("VECTOR_BODY_OR_CUBE_STATE_SHAPE_INVALID")
        camera = self.env.scene["right_wrist_camera"]
        captured, age = camera_capture_time_and_age(camera)
        source_clock = _cpu(captured, dtype=np.float64).reshape(-1) + _cpu(age, dtype=np.float64).reshape(-1)
        if source_clock.shape != (self.num_envs,) or not np.isfinite(source_clock).all():
            raise Stage1AVectorTelemetryError("VECTOR_PHYSICS_CLOCK_INVALID")
        for env_id in range(self.num_envs):
            raw_kwargs = {
                "env_id": env_id,
                "dt_s": dt_s,
                "body_link_pose_world_xyzw": body_link_pose[env_id],
                "body_link_velocity_world": body_link_velocity[env_id],
                "cube_position_world_m": cube_position[env_id],
                "cube_linear_velocity_world_m_s": cube_linear_velocity[env_id],
                "cube_angular_velocity_world_rad_s": cube_angular_velocity[env_id],
            }
            raw_inner = self._raw_side(side="inner", **raw_kwargs)
            raw_outer = self._raw_side(side="outer", **raw_kwargs)
            self._records[env_id].append(
                {
                    "schema": VECTOR_PHYSICS_TELEMETRY_SCHEMA,
                    "env_id": env_id,
                    "policy_step": context.policy_step,
                    "physics_substep": self._substep,
                    "global_physics_sample": self._global_sample,
                    "dt_s": dt_s,
                    "physics_timestamp_s": float(source_clock[env_id]),
                    "phase": context.phase_by_env[env_id],
                    "gripper_intent": context.gripper_intent_by_env[env_id],
                    "close_onset": context.close_onset_by_env[env_id],
                    "q_rad": q[env_id].copy(),
                    "raw_qdot_rad_s": qd[env_id].copy(),
                    "fd_qdd_rad_s2": qdd[env_id].copy(),
                    "root_position_world_m": root_pos[env_id].copy(),
                    "root_quat_world_xyzw": root_quat_xyzw[env_id].copy(),
                    "root_linear_velocity_world_m_s": root_lin[env_id].copy(),
                    "root_angular_velocity_world_rad_s": root_ang[env_id].copy(),
                    "raw_inner": raw_inner,
                    "raw_outer": raw_outer,
                }
            )
        self._substep += 1
        self._global_sample += 1

    def contact_partner_paths(self, env_id: int) -> tuple[str, str]:
        if not 0 <= env_id < self.num_envs:
            raise Stage1AVectorTelemetryError("VECTOR_ENV_ID_INVALID")
        return tuple(
            str(self._sources[side]["partner_actor"]).format(env_id=env_id)
            for side in ("inner", "outer")
        )


__all__ = [
    "Stage1AVectorTelemetryError",
    "VECTOR_PHYSICS_TELEMETRY_SCHEMA",
    "VectorPassiveContactPhysicsTelemetry",
]
