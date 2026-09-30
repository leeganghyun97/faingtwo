#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Strict offline audit for canonical G2 keyboard/remote/Teacher HDF5 data."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from geniesim.rl.isaaclab.g2_keyboard_teacher_dataset import (  # noqa: E402
    G2EpisodeOutcome,
    G2ExpertSource,
    G2KeyboardTeacherTransitionContract,
    G2StudentSequenceContract,
    G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
    G2_V35_KEYBOARD_COLLECTION_CONTRACT_SCHEMA,
    keyboard_teacher_dataset_metadata,
)
from geniesim.rl.isaaclab.g2_student_training import (  # noqa: E402
    list_canonical_episodes,
    load_canonical_episode,
)
from geniesim.rl.isaaclab.g2_teacher_sac import (  # noqa: E402
    G2TeacherObservationContract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _stored_environment(path: Path) -> dict[str, object] | None:
    with h5py.File(path, "r") as file:
        data = file.get("data")
        if data is None or "env_args" not in data.attrs:
            return None
        return json.loads(_json_text(data.attrs["env_args"]))


def audit(
    dataset: Path,
    *,
    sequence_length: int,
    burn_in_steps: int,
    stride: int,
) -> dict[str, object]:
    sequence_contract = G2StudentSequenceContract(
        sequence_length=sequence_length,
        burn_in_steps=burn_in_steps,
        stride=stride,
    )
    keys = list_canonical_episodes(dataset)
    environment = _stored_environment(dataset)
    canonical_metadata = keyboard_teacher_dataset_metadata()
    stored_contract = (
        environment.get("tensor_contract") if environment is not None else None
    )
    stored_model_contract = (
        stored_contract.get("recurrent_visual_policy_contract")
        if isinstance(stored_contract, dict)
        else None
    )
    current_model_contract = canonical_metadata.get(
        "recurrent_visual_policy_contract"
    )
    # Encoder/GRU architecture is a consumer choice, not an immutable HDF5
    # tensor contract. Historical v25 files remain compatible when their
    # dimensions, units, frames, alignment and geometry match even if the
    # currently selected visual backbone differs.
    stored_data_contract = copy.deepcopy(stored_contract)
    current_data_contract = copy.deepcopy(canonical_metadata)
    if isinstance(stored_data_contract, dict):
        stored_data_contract.pop("recurrent_visual_policy_contract", None)
    current_data_contract.pop("recurrent_visual_policy_contract", None)
    metadata_matches = bool(
        environment is not None
        and environment.get("dataset_schema") == G2_KEYBOARD_TEACHER_DATASET_SCHEMA
        and stored_data_contract == current_data_contract
    )
    model_contract_matches = stored_model_contract == current_model_contract
    stored_v35_contract = (
        stored_contract.get("v35_keyboard_collection_contract")
        if isinstance(stored_contract, dict)
        else None
    )
    current_v35_contract = canonical_metadata["v35_keyboard_collection_contract"]
    v35_contract_matches = bool(
        stored_v35_contract == current_v35_contract
        and stored_v35_contract.get("schema")
        == G2_V35_KEYBOARD_COLLECTION_CONTRACT_SCHEMA
        and stored_v35_contract.get("ground_truth_cube_actor_input") is False
        and stored_v35_contract.get("demo_live_relative_feature_semantics_equal")
        is True
    ) if isinstance(stored_v35_contract, dict) else False

    transition_count = 0
    sequence_count = 0
    teacher_bc_rows = 0
    collision_valid_rows = 0
    forbidden_collision_rows = 0
    her_force_safe_rows = 0
    depth_valid_pixels = {"head": 0, "right_wrist": 0}
    depth_total_pixels = {"head": 0, "right_wrist": 0}
    source_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    episode_reports: list[dict[str, object]] = []
    canonical_action_min = math.inf
    canonical_action_max = -math.inf
    camera_resolutions: set[tuple[int, int]] = set()
    successful_episodes = 0
    success_with_stable_grasp_evidence = 0
    success_with_lift_phase_evidence = 0
    success_with_stable_and_lift_evidence = 0
    success_without_stable_or_lift_evidence: list[str] = []
    episodes_without_eligible_sequences: list[str] = []
    short_episode_transition_count = 0
    task_geometries: list[dict[str, object]] = []
    first_close_episode_count = 0
    successful_first_close_episode_count = 0
    first_close_distance_mm: list[float] = []
    episodes_starting_with_close: list[str] = []
    pad_midpoint_to_ee_distance_m: list[torch.Tensor] = []

    if environment is not None:
        geometry = environment.get("keyboard_task_geometry")
        if isinstance(geometry, dict):
            task_geometries.append(copy.deepcopy(geometry))

    for key in keys:
        episode = load_canonical_episode(dataset, key)
        count = int(episode["teacher_observation"].shape[0])
        height, width = map(int, episode["head_rgb"].shape[1:3])
        camera_resolutions.add((width, height))
        contract = G2KeyboardTeacherTransitionContract(height=height, width=width)
        strict = contract.validate_hdf_episode(episode)
        if not strict["pass"]:
            raise ValueError(f"strict HDF validation failed for {key}: {strict}")
        views = contract.extract_training_views(episode)
        sequence = sequence_contract.build(episode, transition_contract=contract)
        lengths = sequence["sequence_lengths"]
        if not isinstance(lengths, torch.Tensor):
            raise RuntimeError("sequence_lengths is not a tensor")
        eligible_sequences = int((lengths > burn_in_steps).sum().item())
        transition_count += count
        sequence_count += eligible_sequences
        if eligible_sequences == 0:
            episodes_without_eligible_sequences.append(key)
            short_episode_transition_count += count
        teacher_bc_rows += int(views["teacher_bc"]["observation"].shape[0])

        collision_valid = episode["forbidden_collision_valid"].to(torch.bool)
        forbidden = episode["forbidden_collision"].to(torch.bool)
        safe = collision_valid & ~forbidden
        collision_valid_rows += int(collision_valid.sum().item())
        forbidden_collision_rows += int(forbidden.sum().item())
        her_force_safe_rows += int(safe.sum().item())

        for prefix in ("head", "right_wrist"):
            mask = episode[f"{prefix}_depth_valid"].to(torch.bool)
            depth_valid_pixels[prefix] += int(mask.sum().item())
            depth_total_pixels[prefix] += mask.numel()
        for code in episode["expert_source_code"].reshape(-1).tolist():
            source_counts[G2ExpertSource(int(code)).name] += 1
        for code in episode["outcome_code"].reshape(-1).tolist():
            outcome_counts[G2EpisodeOutcome(int(code)).name] += 1

        action = episode["canonical_policy_action"]
        canonical_action_min = min(canonical_action_min, float(action.min().item()))
        canonical_action_max = max(canonical_action_max, float(action.max().item()))
        timestamps = episode["timestamp"].reshape(-1).to(torch.float64)
        timestamp_step = (
            float(torch.median(timestamps[1:] - timestamps[:-1]).item())
            if count > 1
            else None
        )
        terminal_outcome = G2EpisodeOutcome(
            int(episode["outcome_code"][-1, 0].item())
        )
        teacher_slices = G2TeacherObservationContract().slices
        ee_position = episode["teacher_observation"][:, teacher_slices[
            "end_effector_pose_root_xyzw"
        ].start:teacher_slices["end_effector_pose_root_xyzw"].start + 3]
        cube_position = episode["teacher_observation"][:, teacher_slices[
            "cube_pose_root_xyzw"
        ].start:teacher_slices["cube_pose_root_xyzw"].start + 3]
        grasp_center = episode["grasp_center_position_root_m"]
        pad_midpoint_to_ee_distance_m.append(
            torch.linalg.vector_norm(grasp_center - ee_position, dim=-1)
        )
        closed = action[:, -1] < 0.0
        previous_closed = torch.cat(
            (torch.zeros(1, dtype=torch.bool), closed[:-1]), dim=0
        )
        close_onset = closed & ~previous_closed
        if bool(close_onset.any()):
            close_index = int(torch.nonzero(close_onset, as_tuple=False)[0, 0].item())
            first_close_episode_count += 1
            episodes_starting_with_close.extend([key] if close_index == 0 else [])
            first_close_distance_mm.append(
                1000.0 * float(torch.linalg.vector_norm(
                    cube_position[close_index] - grasp_center[close_index]
                ).item())
            )
            successful_first_close_episode_count += int(
                terminal_outcome is G2EpisodeOutcome.SUCCESS
            )
        stable_evidence = bool(
            episode["stable_grasp"].to(torch.bool).any().item()
        )
        phase_slice = G2TeacherObservationContract().slices[
            "curriculum_phase_features"
        ]
        lift_evidence = bool(
            (
                episode["next_teacher_observation"][:, phase_slice.start + 2]
                > 0.5
            ).any().item()
        )
        success_episode = terminal_outcome is G2EpisodeOutcome.SUCCESS
        if success_episode:
            successful_episodes += 1
            success_with_stable_grasp_evidence += int(stable_evidence)
            success_with_lift_phase_evidence += int(lift_evidence)
            success_with_stable_and_lift_evidence += int(
                stable_evidence and lift_evidence
            )
            if not stable_evidence or not lift_evidence:
                success_without_stable_or_lift_evidence.append(key)
        episode_reports.append(
            {
                "episode": key,
                "transitions": count,
                "terminal_final_only": bool(
                    not (
                        episode["terminated"][:-1].to(torch.bool)
                        | episode["truncated"][:-1].to(torch.bool)
                    ).any()
                    and bool(
                        (
                            episode["terminated"][-1].to(torch.bool)
                            | episode["truncated"][-1].to(torch.bool)
                        ).item()
                    )
                ),
                "median_timestamp_step_s": timestamp_step,
                "eligible_student_sequences": eligible_sequences,
                "collision_authority_coverage": float(collision_valid.float().mean()),
                "terminal_outcome": terminal_outcome.name,
                "stable_grasp_evidence": stable_evidence,
                "lift_phase_evidence": lift_evidence,
            }
        )

    collision_authority_complete = collision_valid_rows == transition_count
    camera_contract_matches = camera_resolutions == {(256, 192)}
    success_evidence_complete = not success_without_stable_or_lift_evidence
    midpoint_offsets = torch.cat(pad_midpoint_to_ee_distance_m)
    midpoint_linkage_pass = bool(
        torch.isfinite(midpoint_offsets).all()
        and (midpoint_offsets > 0.001).all()
        and (midpoint_offsets < 0.25).all()
    )
    v35_keyboard_compatibility_pass = bool(
        metadata_matches and v35_contract_matches and midpoint_linkage_pass
    )
    functional_pass = bool(
        transition_count > 0
        and sequence_count > 0
        and metadata_matches
        and camera_contract_matches
        and success_evidence_complete
        and v35_keyboard_compatibility_pass
    )
    report = {
        "schema": "geniesim_g2_keyboard_teacher_dataset_audit_v1",
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "dataset_schema": G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
        "strict_hdf_contract_pass": True,
        "stored_metadata_matches_current_contract": metadata_matches,
        "stored_model_contract_matches_selected_architecture": (
            model_contract_matches
        ),
        "v35_keyboard_compatibility_pass": v35_keyboard_compatibility_pass,
        "v35_keyboard_collection_contract": {
            "schema": G2_V35_KEYBOARD_COLLECTION_CONTRACT_SCHEMA,
            "stored_contract_matches": v35_contract_matches,
            "actor_cube_source": "HEAD_PREDICTION_NOT_GROUND_TRUTH",
            "grasp_center_source": "LIVE_DISTAL_PAD_MIDPOINT_FK",
            "relative_feature_materialized_by_consumer": True,
            "ground_truth_cube_actor_input": False,
            "pad_midpoint_to_ee_offset_min_mm": 1000.0 * float(midpoint_offsets.min()),
            "pad_midpoint_to_ee_offset_max_mm": 1000.0 * float(midpoint_offsets.max()),
            "pad_midpoint_linkage_pass": midpoint_linkage_pass,
        },
        "model_contract_mismatch_is_data_blocker": False,
        "teacher_student_compatibility_pass": functional_pass,
        "episode_count": len(keys),
        "transition_count": transition_count,
        "eligible_student_sequence_count": sequence_count,
        "teacher_bc_compatible_rows": teacher_bc_rows,
        "dimensions": {
            "teacher_observation": 59,
            "keyboard_action": 8,
            "canonical_teacher_student_action": 7,
            "deployable_student_proprioception": 45,
            "camera_resolution_wh": [256, 192],
            "observed_camera_resolutions_wh": [
                list(value) for value in sorted(camera_resolutions)
            ],
            "camera_resolution_contract_pass": camera_contract_matches,
        },
        "canonical_action_range": [canonical_action_min, canonical_action_max],
        "expert_source_rows": dict(sorted(source_counts.items())),
        "outcome_rows": dict(sorted(outcome_counts.items())),
        "episode_quality": {
            "successful_episodes": successful_episodes,
            "success_with_stable_grasp_evidence": (
                success_with_stable_grasp_evidence
            ),
            "success_with_lift_phase_evidence": success_with_lift_phase_evidence,
            "success_with_stable_and_lift_evidence": (
                success_with_stable_and_lift_evidence
            ),
            "success_without_stable_or_lift_evidence": (
                success_without_stable_or_lift_evidence
            ),
            "success_evidence_complete": success_evidence_complete,
            "episode_first_close_count": first_close_episode_count,
            "successful_episode_first_close_count": (
                successful_first_close_episode_count
            ),
            "episodes_starting_with_close": episodes_starting_with_close,
            "first_close_cube_to_pad_midpoint_distance_mm": {
                "mean": (
                    sum(first_close_distance_mm) / len(first_close_distance_mm)
                    if first_close_distance_mm else None
                ),
                "minimum": min(first_close_distance_mm) if first_close_distance_mm else None,
                "maximum": max(first_close_distance_mm) if first_close_distance_mm else None,
                "metric_uses_cube_gt_for_offline_audit_only": True,
                "actor_input_uses_cube_gt": False,
            },
            "episodes_without_eligible_sequences": (
                episodes_without_eligible_sequences
            ),
            "short_episode_transition_count": short_episode_transition_count,
            "short_episodes_are_excluded_from_recurrent_training": True,
        },
        "task_geometry": task_geometries[0] if task_geometries else None,
        "depth_valid_pixel_ratio": {
            key: depth_valid_pixels[key] / max(depth_total_pixels[key], 1)
            for key in depth_valid_pixels
        },
        "collision": {
            "authority_valid_rows": collision_valid_rows,
            "authority_coverage": collision_valid_rows / transition_count,
            "forbidden_collision_rows": forbidden_collision_rows,
            "her_force_safe_rows": her_force_safe_rows,
            "full_coverage": collision_authority_complete,
            "unmeasured_is_safe": False,
        },
        "dataset_collision_safety_approved": collision_authority_complete,
        "large_scale_training_approved": False,
        "large_scale_blockers": [
            *(
                []
                if collision_authority_complete
                else ["NO_LIVE_FULL_BODY_COLLISION_AUTHORITY"]
            ),
            "DATASET_AUDIT_DOES_NOT_ATTEST_LIVE_RUNTIME_OR_CLEAN_SHUTDOWN",
        ],
        "functional_verdict": (
            "KEYBOARD_REMOTE_TEACHER_STUDENT_CONTRACT_PASS"
            if functional_pass
            else "KEYBOARD_REMOTE_TEACHER_STUDENT_CONTRACT_FAIL"
        ),
        "episode_reports": episode_reports,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--burn-in-steps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=36)
    args = parser.parse_args()
    try:
        report = audit(
            args.dataset,
            sequence_length=args.sequence_length,
            burn_in_steps=args.burn_in_steps,
            stride=args.stride,
        )
        code = 0 if report["functional_verdict"].endswith("_PASS") else 2
    except BaseException as error:
        report = {
            "schema": "geniesim_g2_keyboard_teacher_dataset_audit_v1",
            "dataset": str(args.dataset.resolve()),
            "functional_verdict": "KEYBOARD_REMOTE_TEACHER_STUDENT_CONTRACT_FAIL",
            "error": repr(error),
        }
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
