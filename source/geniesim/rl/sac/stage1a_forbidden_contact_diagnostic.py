# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Diagnostic-only exact PhysX contact-pair receipts for Stage-1A.

The recorder subscribes to the existing PhysX contact-report stream.  It does
not author collision filters, change a threshold, or participate in the task
termination decision.  Encoded paths are copied in the callback and decoded
only when a terminal receipt is requested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


_ENV = re.compile(r"/env_(\d+)(?:/|$)")
_ALLOWED_OBJECT_LINKS = {
    "gripper_r_inner_link4",
    "gripper_r_outer_link4",
    "gripper_r_outer_link2",
}


@dataclass(frozen=True)
class _Context:
    vector_step: int
    control_timestamp_s: float
    episode_ids: tuple[int, ...]
    source_sample_ids: tuple[str, ...]
    control_steps: tuple[int, ...]


class Stage1AForbiddenContactEventDiagnostic:
    """Bounded read-only recorder for exact actor/collider contact pairs."""

    def __init__(self, *, num_envs: int, physics_dt_s: float = 0.002) -> None:
        if num_envs <= 0 or not math.isclose(physics_dt_s, 0.002, abs_tol=1e-12):
            raise ValueError("FORBIDDEN_CONTACT_DIAGNOSTIC_CONTRACT_INVALID")
        import omni.physx
        from omni.physx.scripts import physicsUtils

        self.num_envs = int(num_envs)
        self.physics_dt_s = float(physics_dt_s)
        self._decode = physicsUtils.PhysicsSchemaTools.intToSdfPath
        self._context: _Context | None = None
        self._callback_sequence = 0
        self._records: deque[dict[str, Any]] = deque(maxlen=200_000)
        interface = omni.physx.get_physx_simulation_interface()
        self._subscription = interface.subscribe_contact_report_events(
            self._on_contact_report
        )
        if self._subscription is None:
            raise RuntimeError("FORBIDDEN_CONTACT_DIAGNOSTIC_SUBSCRIBE_FAILED")

    def set_control_context(
        self,
        *,
        vector_step: int,
        control_timestamp_s: float,
        episode_ids: Sequence[int],
        source_sample_ids: Sequence[str],
        control_steps: Sequence[int],
    ) -> None:
        if not (
            len(episode_ids)
            == len(source_sample_ids)
            == len(control_steps)
            == self.num_envs
        ):
            raise ValueError("FORBIDDEN_CONTACT_DIAGNOSTIC_CONTEXT_CARDINALITY")
        self._context = _Context(
            vector_step=int(vector_step),
            control_timestamp_s=float(control_timestamp_s),
            episode_ids=tuple(int(value) for value in episode_ids),
            source_sample_ids=tuple(str(value) for value in source_sample_ids),
            control_steps=tuple(int(value) for value in control_steps),
        )

    @staticmethod
    def _vector(value: Any) -> list[float] | None:
        if value is None:
            return None
        try:
            result = [float(component) for component in value]
        except (TypeError, ValueError):
            return None
        return result if len(result) == 3 and all(math.isfinite(x) for x in result) else None

    def _on_contact_report(self, headers: Any, contact_data: Any) -> None:
        context = self._context
        if context is None:
            return
        self._callback_sequence += 1
        for header in headers:
            offset = int(header.contact_data_offset)
            count = int(header.num_contact_data)
            points: list[dict[str, Any]] = []
            for index in range(offset, offset + count):
                datum = contact_data[index]
                impulse = self._vector(getattr(datum, "impulse", None))
                impulse_norm = (
                    None
                    if impulse is None
                    else math.sqrt(math.fsum(value * value for value in impulse))
                )
                points.append(
                    {
                        "position_world_m": self._vector(
                            getattr(datum, "position", getattr(datum, "point", None))
                        ),
                        "normal_world": self._vector(getattr(datum, "normal", None)),
                        "impulse_ns": impulse,
                        "impulse_norm_ns": impulse_norm,
                        "equivalent_force_n": (
                            None if impulse_norm is None else impulse_norm / self.physics_dt_s
                        ),
                        "separation_m": (
                            float(datum.separation)
                            if math.isfinite(float(getattr(datum, "separation", math.nan)))
                            else None
                        ),
                    }
                )
            self._records.append(
                {
                    "event_type": int(header.type),
                    "actor0": int(header.actor0),
                    "actor1": int(header.actor1),
                    "collider0": int(header.collider0),
                    "collider1": int(header.collider1),
                    "callback_sequence": self._callback_sequence,
                    "context": context,
                    "points": points,
                }
            )

    @staticmethod
    def _env_id(paths: Sequence[str]) -> int | None:
        values = {
            int(match.group(1))
            for path in paths
            for match in [_ENV.search(path)]
            if match is not None
        }
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _classify(actor_a: str, actor_b: str) -> tuple[str, bool, str | None, str | None]:
        paths = (actor_a, actor_b)
        robot = next((path for path in paths if "/Robot/" in path), None)
        other = next((path for path in paths if path != robot), None)
        link = None if robot is None else robot.rsplit("/", 1)[-1]
        if all("/Robot/" in path for path in paths):
            return "SELF", True, link, other
        if any(path.endswith("/Table") for path in paths):
            return "TABLE", robot is not None, link, other
        if any(path.endswith("/Object") for path in paths):
            allowed = link in _ALLOWED_OBJECT_LINKS
            return "CUBE", not allowed, link, other
        if robot is not None:
            return "ENVIRONMENT", True, link, other
        return "OTHER", False, link, other

    def terminal_window(
        self,
        *,
        env_id: int,
        episode_id: int,
        source_sample_id: str,
        termination_vector_step: int,
        lookback_control_steps: int = 20,
    ) -> dict[str, Any]:
        decoded: list[dict[str, Any]] = []
        decode_errors: list[str] = []
        for raw in self._records:
            context: _Context = raw["context"]
            if not (
                context.vector_step >= termination_vector_step - lookback_control_steps
                and context.vector_step <= termination_vector_step
                and context.episode_ids[env_id] == episode_id
                and context.source_sample_ids[env_id] == source_sample_id
            ):
                continue
            try:
                actor_a = str(self._decode(raw["actor0"]))
                actor_b = str(self._decode(raw["actor1"]))
                collider_a = str(self._decode(raw["collider0"]))
                collider_b = str(self._decode(raw["collider1"]))
            except Exception as exc:  # diagnostic remains fail-explicit
                decode_errors.append(f"{type(exc).__name__}:{exc}")
                continue
            paths = (actor_a, actor_b, collider_a, collider_b)
            if self._env_id(paths) != env_id:
                continue
            collision_type, forbidden, robot_link, other_body = self._classify(
                actor_a, actor_b
            )
            decoded.append(
                {
                    "vector_step": context.vector_step,
                    "control_step": context.control_steps[env_id],
                    "control_timestamp_s": context.control_timestamp_s,
                    "event_type": raw["event_type"],
                    "actor_paths": [actor_a, actor_b],
                    "collider_paths": [collider_a, collider_b],
                    "collision_pair": f"{actor_a} <-> {actor_b}",
                    "collision_type": collision_type,
                    "forbidden_by_existing_policy": forbidden,
                    "robot_link": robot_link,
                    "other_body": other_body,
                    "points": raw["points"],
                }
            )
        forbidden = [row for row in decoded if row["forbidden_by_existing_policy"]]
        return {
            "schema": "g2_stage1a_forbidden_contact_event_window_v1",
            "authority": "PHYSX_POST_STEP_CONTACT_REPORT_READ_ONLY",
            "lookback_control_steps": int(lookback_control_steps),
            "post_termination_steps_available": 0,
            "post_termination_reason": "TASK_RESET_BOUNDARY",
            "event_count": len(decoded),
            "forbidden_event_count": len(forbidden),
            "first_forbidden_contact_vector_step": min(
                (int(row["vector_step"]) for row in forbidden), default=None
            ),
            "last_forbidden_contact_vector_step": max(
                (int(row["vector_step"]) for row in forbidden), default=None
            ),
            "decode_errors": decode_errors,
            "forbidden_events": forbidden,
            "all_events": decoded,
            "behavior_changed": False,
        }

    def close(self) -> None:
        subscription = self._subscription
        self._subscription = None
        if subscription is not None:
            unsubscribe = getattr(subscription, "unsubscribe", None)
            if callable(unsubscribe):
                unsubscribe()


__all__ = ["Stage1AForbiddenContactEventDiagnostic"]
