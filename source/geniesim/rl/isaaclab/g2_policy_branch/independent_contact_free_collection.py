# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure contracts for independent contact-free runtime-replan collection.

The runtime planner is allowed to inspect the post-reset cube pose in order to
produce a *nominal* OPEN-only trajectory.  That pose is privileged and must
never become an actor feature or an HDF5 replay field.  This module keeps the
two concerns separate:

* an immutable per-episode sidecar binds a source-owned reset seed to the
  planner-only receipt; and
* the canonical HDF5 manifest binds only the sidecar digest, preserving the
  existing deployable actor inventory and 4-D action contract.

It intentionally has no Isaac, cuRobo, controller, or HDF5 dependency.  The
live runner remains the sole owner of ``env.step`` and packet consumption.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from ..g2_keyboard_pose import G2_KEYBOARD_CUBE_X_RANGE_M, G2_KEYBOARD_CUBE_Y_RANGE_M
from .canonical_contact_free_collection_v2 import collection_manifest_v2


INDEPENDENT_CONTACT_FREE_COLLECTION_SCHEMA = (
    "g2_independent_contact_free_runtime_replan_collection_v1"
)
RUNTIME_PLAN_SIDECAR_SCHEMA = "g2_contact_free_runtime_plan_sidecar_v1"
CANONICAL_FRAME = "robot_root"
POLICY_DT_S = 0.020
PHYSICS_DT_S = 0.002
ACTION_CONTRACT = "[dx,dy,dz,HOLD_OPEN] robot_root meters"
_EPISODE_ID = re.compile(r"^episode-[0-9]{5,}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")


class IndependentContactFreeCollectionError(ValueError):
    """An independent collection configuration lacks immutable provenance."""


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise IndependentContactFreeCollectionError(f"{label}_MUST_BE_SHA256")
    return value


def _finite_vector(value: Any, *, width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise IndependentContactFreeCollectionError(f"{label}_MUST_BE_NUMERIC")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise IndependentContactFreeCollectionError(f"{label}_MUST_BE_NUMERIC") from error
    if len(result) != width or not all(math.isfinite(component) for component in result):
        raise IndependentContactFreeCollectionError(
            f"{label}_MUST_BE_{width}_FINITE_VALUES"
        )
    return result


@dataclass(frozen=True)
class IndependentCollectionEpisode:
    """One independently seeded source-owned reset episode."""

    episode_id: str
    seed: int

    def validated(self) -> "IndependentCollectionEpisode":
        if not isinstance(self.episode_id, str) or _EPISODE_ID.fullmatch(self.episode_id) is None:
            raise IndependentContactFreeCollectionError(
                "COLLECTION_EPISODE_ID_MUST_BE_ZERO_PADDED_AND_UNIQUE"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise IndependentContactFreeCollectionError(
                "COLLECTION_SEED_MUST_BE_NONNEGATIVE_INT"
            )
        return self


def independent_episode_schedule(
    *, count: int, seed_base: int
) -> tuple[IndependentCollectionEpisode, ...]:
    """Create an explicit, non-repeating reset schedule without sampling poses.

    Pose sampling stays exclusively inside the existing source-owned reset
    event.  A different seed is therefore a request for an independent reset,
    not an unreviewed workspace or arm-state randomizer.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise IndependentContactFreeCollectionError("COLLECTION_COUNT_MUST_BE_POSITIVE_INT")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise IndependentContactFreeCollectionError("COLLECTION_SEED_BASE_MUST_BE_NONNEGATIVE_INT")
    schedule = tuple(
        IndependentCollectionEpisode(
            episode_id=f"episode-{index:05d}", seed=seed_base + index
        ).validated()
        for index in range(count)
    )
    if len({item.seed for item in schedule}) != len(schedule):
        raise IndependentContactFreeCollectionError("COLLECTION_SEEDS_NOT_UNIQUE")
    return schedule


def source_reset_distribution_contract() -> dict[str, Any]:
    """Describe, but never expand, the task-authored reset distribution."""

    return {
        "cube": {
            "owner": "G2RedundancyTeleopEnvCfg.events.reset_object_position",
            "sampling": "source_owned_uniform_pose_range_at_env_reset",
            "frame": CANONICAL_FRAME,
            "unit": "m",
            "x_range_m": list(G2_KEYBOARD_CUBE_X_RANGE_M),
            "y_range_m": list(G2_KEYBOARD_CUBE_Y_RANGE_M),
            "z": "source_authored_nominal_table_supported_center",
        },
        "robot_initial_state": {
            "owner": "G2RedundancyTeleopEnvCfg keyboard initial pose plus existing OPEN settle",
            "sampling": "NO_NEW_ARM_JOINT_RANDOMIZATION",
            "runtime_readback_required": True,
        },
        "action": {
            "components": ["dx", "dy", "dz", "HOLD_OPEN"],
            "frame": CANONICAL_FRAME,
            "unit": "m,m,m,HOLD_OPEN=0",
            "policy_dt_s": POLICY_DT_S,
            "physics_dt_s": PHYSICS_DT_S,
        },
    }


def runtime_plan_sidecar(
    *,
    episode: IndependentCollectionEpisode,
    runtime_replan_receipt: Mapping[str, Any],
    post_reset_input: Mapping[str, Any],
    runner_source_sha256: str,
    candidate_asset_sha256: str,
) -> dict[str, Any]:
    """Build privileged planner provenance retained *outside* policy data."""

    episode = episode.validated()
    runner_source_sha256 = _sha256(runner_source_sha256, label="RUNNER_SOURCE_SHA256")
    candidate_asset_sha256 = _sha256(
        candidate_asset_sha256, label="CANDIDATE_ASSET_SHA256"
    )
    receipt = dict(runtime_replan_receipt)
    reset = dict(post_reset_input)
    expected_receipt = {
        "schema": "g2_contact_free_runtime_replan_v1",
        "actor_input_contains_cube_gt": False,
        "replay_row_contains_planner_cube_gt": False,
        "gripper_command_generated": False,
        "action_packet_generated": False,
    }
    expected_reset = {
        "schema": "g2_post_reset_runtime_replan_input_v2",
        "planner_frame": CANONICAL_FRAME,
        "actor_input_contains_cube_gt": False,
        "replay_row_contains_planner_cube_gt": False,
        "gripper_command_generated": False,
        "action_packet_generated": False,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise IndependentContactFreeCollectionError(
                f"RUNTIME_PLAN_RECEIPT_INVALID:{key}"
            )
    for key, expected in expected_reset.items():
        if reset.get(key) != expected:
            raise IndependentContactFreeCollectionError(
                f"POST_RESET_INPUT_INVALID:{key}"
            )
    if receipt.get("planner_seed") != episode.seed or reset.get("seed") != episode.seed:
        raise IndependentContactFreeCollectionError("RUNTIME_PLAN_SEED_BINDING_MISMATCH")
    cube = _finite_vector(
        receipt.get("planner_input_cube_center_root_m"),
        width=3,
        label="RUNTIME_PLAN_CUBE_ROOT_M",
    )
    reset_cube = _finite_vector(
        reset.get("cube_center_root_m"),
        width=3,
        label="POST_RESET_CUBE_ROOT_M",
    )
    if cube != reset_cube:
        raise IndependentContactFreeCollectionError("RUNTIME_PLAN_RESET_CUBE_BINDING_MISMATCH")
    _finite_vector(
        receipt.get("planner_input_right_arm_q_rad"),
        width=7,
        label="RUNTIME_PLAN_RIGHT_ARM_Q_RAD",
    )
    _finite_vector(
        reset.get("right_arm_q_rad"), width=7, label="POST_RESET_RIGHT_ARM_Q_RAD"
    )
    return {
        "schema": RUNTIME_PLAN_SIDECAR_SCHEMA,
        "episode": {"episode_id": episode.episode_id, "seed": episode.seed},
        "planner_provenance": "POST_RESET_RUNTIME_REPLAN",
        "privileged_data_scope": "SIDE_CAR_ONLY_NOT_ACTOR_NOT_HDF5_ROW",
        "runtime_replan_receipt": receipt,
        "post_reset_input": reset,
        "source_hashes": {
            "runner": runner_source_sha256,
            "candidate_asset": candidate_asset_sha256,
        },
        "actor_input_contains_cube_gt": False,
        "replay_row_contains_planner_cube_gt": False,
        "action_packet_generated_by_planner": False,
        "gripper_command_generated_by_planner": False,
    }


def runtime_replan_collection_manifest_v2(
    *,
    source_sha256: str,
    asset_sha256: str,
    runtime_plan_sidecar_sha256: str,
    episode: IndependentCollectionEpisode,
) -> dict[str, Any]:
    """Make a v2 actor HDF5 manifest without embedding cube ground truth."""

    episode = episode.validated()
    manifest = collection_manifest_v2(
        source_sha256=source_sha256,
        asset_sha256=asset_sha256,
        trajectory_sha256=_sha256(
            runtime_plan_sidecar_sha256, label="RUNTIME_PLAN_SIDECAR_SHA256"
        ),
        snapshot_hook_enabled=True,
    )
    manifest.update(
        {
            "independent_collection": {
                "schema": INDEPENDENT_CONTACT_FREE_COLLECTION_SCHEMA,
                "episode_id": episode.episode_id,
                "reset_seed": episode.seed,
                "source_reset_distribution": source_reset_distribution_contract(),
                "runtime_plan_sidecar_sha256": runtime_plan_sidecar_sha256,
                "runtime_plan_sidecar_embedded": False,
                "actor_input_contains_cube_gt": False,
                "replay_row_contains_planner_cube_gt": False,
            },
            "trajectory_provenance": "POST_RESET_RUNTIME_PLAN_SIDECAR_SHA256",
        }
    )
    return manifest


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Return the immutable digest to bind an emitted sidecar."""

    return _sha256_json(value)


__all__ = [
    "ACTION_CONTRACT",
    "CANONICAL_FRAME",
    "INDEPENDENT_CONTACT_FREE_COLLECTION_SCHEMA",
    "IndependentCollectionEpisode",
    "IndependentContactFreeCollectionError",
    "POLICY_DT_S",
    "PHYSICS_DT_S",
    "RUNTIME_PLAN_SIDECAR_SCHEMA",
    "canonical_json_sha256",
    "independent_episode_schedule",
    "runtime_plan_sidecar",
    "runtime_replan_collection_manifest_v2",
    "source_reset_distribution_contract",
]
