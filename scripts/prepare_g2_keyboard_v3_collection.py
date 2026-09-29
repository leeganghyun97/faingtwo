#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Create an immutable, simulator-free keyboard-v3 collection plan.

This command verifies the Candidate-A provenance chain and emits the 21 pilot
conditions plus canonical contracts.  It does not launch Isaac, move a robot,
or claim CLOSE/contact authorization while Candidate A remains M2-failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgbd-hz", type=int, choices=(25,), default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("KEYBOARD_V3_PREPARATION_OUTPUT_ALREADY_EXISTS")

    from geniesim.rl.isaaclab.g2_policy_branch.contact_free_candidate_a_binding import (
        resolve_contact_free_candidate_a,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_collection_contract import (
        COLLECTION_TARGETS_MM,
        TELEOP_COLLECTION_AUTHORITY,
        collection_target_conditions,
        collection_contract_payload,
        pilot_conditions,
    )
    from geniesim.rl.sac.bc_residual_sac_contract import residual_interface_provenance
    from geniesim.rl.sac.human_grasp_gru_bc import HumanGraspGRUConfig
    from geniesim.rl.isaaclab.g2_policy_branch.omnipicker_product_contract import (
        OMNIPICKER_MANUAL_URL,
        OMNIPICKER_PRODUCT_CONTRACT,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.rgbd_logging_contract import (
        DEPTH_RAW_UNIT,
        DEPTH_TO_METER_SCALE,
        RGBD_EVIDENCE_SCHEMA,
    )
    from geniesim.rl.sac.keyboard_v3_dataset import (
        KEYBOARD_V3_CAMERA_NAMES,
        KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES,
        KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES,
        KEYBOARD_V3_OPERATOR_VIEW_NAMES,
        KEYBOARD_V3_FILE_SCHEMA,
        KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES,
    )

    from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
        BASELINE_NAME,
        LEFT_ARM_POSTURE_ID,
        load_baseline_manifest,
        selected_pregrasp_initial_state,
    )

    candidate = resolve_contact_free_candidate_a(repo_root=ROOT)
    baseline_path, baseline, baseline_sha256 = load_baseline_manifest()
    selected_sample = selected_pregrasp_initial_state()
    config_path = ROOT / "configs/g2_policy_branch/keyboard_collection_v3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["asset_sha256"] != candidate.candidate_asset_sha256:
        raise SystemExit("KEYBOARD_V3_CONFIG_CANDIDATE_A_HASH_MISMATCH")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "episodes").mkdir()
    (root / "rejected_episodes").mkdir()
    (root / "result_receipts").mkdir()
    conditions = [
        {
            "condition_id": item.condition_id,
            "curobo_backoff_m": item.backoff_m,
            "start_offset_xyz_m": list(item.start_offset_xyz_m),
            "target_successful_demonstrations": 5,
        }
        for item in pilot_conditions()
    ]
    collection_targets = [
        {
            "condition_id": item.condition_id,
            "target_residual_mm": int(round(item.backoff_m * 1000.0)),
            "curobo_backoff_m": item.backoff_m,
            "start_offset_xyz_m": list(item.start_offset_xyz_m),
            "target_episode_count": "BALANCED_LEAST_SAMPLED_SCHEDULER",
        }
        for item in collection_target_conditions()
    ]
    manifest = {
        "schema": "g2_keyboard_v3_collection_manifest_v6_boolean_contact_primary",
        "execution_mode": "PREPARE_ONLY_NO_ISAAC_NO_ACTION",
        "seed": args.seed,
        "episode_directory": str(root / "episodes"),
        "rgbd_hz": args.rgbd_hz,
        "control_hz": 50,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "candidate_a_binding": candidate.as_dict(),
        "baseline_variant": BASELINE_NAME,
        "baseline_manifest_path": str(baseline_path),
        "baseline_manifest_sha256": baseline_sha256,
        "left_arm_posture_id": LEFT_ARM_POSTURE_ID,
        "left_arm_joint_position_rad": baseline["left_arm"]["joint_position_rad"],
        "pregrasp_init_source": "CUROBO_DATASET",
        "pregrasp_sample_id": selected_sample.sample_id,
        "cube_sample_id": selected_sample.sample_id,
        "head_camera": "ENABLED",
        "wrist_camera": "ENABLED",
        "omnipicker_product_model": "OmniPicker",
        "omnipicker_manual_authority_url": OMNIPICKER_MANUAL_URL,
        "omnipicker_manual_hardware_version": "1.2",
        "omnipicker_pcba_version": "UNRESOLVED_20_OR_30",
        "omnipicker_firmware_version": "UNRESOLVED_RUNTIME_DEVICE",
        "omnipicker_maximum_gripping_force_n": (
            OMNIPICKER_PRODUCT_CONTRACT.maximum_gripping_force_n
        ),
        "omnipicker_physical_pad_touch_sensor_available": False,
        "omnipicker_protocol_force_feedback_semantics": (
            OMNIPICKER_PRODUCT_CONTRACT.protocol_force_feedback_semantics
        ),
        "sim_contact_telemetry_role": (
            OMNIPICKER_PRODUCT_CONTRACT.simulator_contact_role
        ),
        "student_contact_input_count": 0,
        "m2_oem_gripping_force_metric": (
            OMNIPICKER_PRODUCT_CONTRACT.m2_gripping_force_metric
        ),
        "m2_sim_contact_to_oem_force_mapping": (
            OMNIPICKER_PRODUCT_CONTRACT.sim_contact_to_oem_gripping_force_mapping
        ),
        "m2_oem_force_gate_status": (
            OMNIPICKER_PRODUCT_CONTRACT.as_dict()["m2_force_gate_status"]
        ),
        "hdf5_file_schema": KEYBOARD_V3_FILE_SCHEMA,
        "gru_actor_camera_names": list(KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES),
        "recorded_non_actor_camera_names": list(KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES),
        "operator_display_only_view_names": list(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
        "camera_evidence_contract": {
            "schema": RGBD_EVIDENCE_SCHEMA,
            "camera_names": list(KEYBOARD_V3_CAMERA_NAMES),
            "depth_raw_unit": DEPTH_RAW_UNIT,
            "depth_to_meter_scale": DEPTH_TO_METER_SCALE,
            "raw_depth_preserved": True,
            "validity_mask_separate": True,
            "intrinsics_required": True,
            "extrinsics_robot_root_required": True,
            "acquisition_timestamp_required": True,
            "frame_id_required": True,
            "control_to_frame_mapping_required": True,
            "current_gru_input_changed": False,
        },
        "required_outcome_fields": sorted(KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES),
        "future_outcome_horizon_control_steps": 64,
        "outcome_alignment": "POST_ACTION_TRANSITION_T_PLUS_1",
        "outcome_label_authority": "RAW_TELEMETRY_OFFLINE_DERIVATION_ONLY",
        "human_close_label_authority": "KEYBOARD_K_CLOSE_EDGE",
        "privileged_feasible_label_authority": "VERIFIED_SIM_GEOMETRY_ONLY",
        "human_and_privileged_labels_merged": False,
        "primary_privileged_learning_signal": "BOOLEAN_CONTACT",
        "force_learning_role": "RAW_DIAGNOSTIC_SAFETY_AUXILIARY_ONLY",
        "force_required_for_contact": False,
        "force_required_for_stable": False,
        "contact_channel_names": ["right_inner", "right_outer"],
        "contact_boolean_extraction": "FILTERED_FORCE_MATRIX_W_GREATER_THAN_1N_TO_BOOLEAN",
        "contact_boolean_threshold_n": 1.0,
        "pad_pose_authority": "DISTAL_LINK_FRAME_RAW_NOT_CALIBRATED_PAD_SURFACE",
        "control_source": "TERMINAL_KEYBOARD",
        "terminal_window_mode": "SEPARATE_GNOME_TERMINAL",
        "terminal_translation_step_m": 0.001,
        "terminal_motion_min_interval_s": 0.050,
        "terminal_maximum_requested_speed_m_s": 0.020,
        "startup_curobo_planning": "ONCE_PER_CONTINUOUS_SESSION",
        "between_episode_reset_source": "IN_PROCESS_QUALIFIED_CUROBO_HANDOFF_STATE",
        "between_episode_curobo_replan": False,
        "terminal_smoothing": {
            "profile": "RAISED_COSINE_PER_ACCEPTED_PULSE",
            "default_control_steps": 5,
            "control_hz": 50,
            "silent_clipping": False,
        },
        "collection_targets_mm": list(COLLECTION_TARGETS_MM),
        "collection_target_reference": (
            "CUROBO_NOMINAL_GRASP_POSE_TRANSLATION_RESIDUAL_NOT_PAD_SURFACE"
        ),
        "collection_target_conditions": collection_targets,
        "target_scheduler": "LEAST_SAMPLED_THEN_ASCENDING_NO_OUTCOME_BALANCING",
        "collection_contract": collection_contract_payload(),
        "gru_contract": HumanGraspGRUConfig().contract_payload(),
        "residual_sac_contract": residual_interface_provenance(),
        "pilot_conditions": conditions,
        "pilot_condition_count": len(conditions),
        "pilot_episode_target": len(conditions) * 5,
        "pad_surface_schema_ready": True,
        "pad_calibration_required": True,
        "pad_surface_valid_default": False,
        "student_privileged_input_count": 0,
        "teleop_collection_authority": TELEOP_COLLECTION_AUTHORITY,
        "bounded_operator_open_authorized": True,
        "bounded_operator_close_authorized": True,
        "autonomous_close_authorized": False,
        "residual_sac_authorized": False,
        "lift_authorized": False,
        "place_authorized": False,
        "candidate_a_underlying_contact_training_authorized": candidate.contact_training_authorized,
        "pad_calibration_required_for_collection": False,
        "pad_calibration_required_for_human_only_bc": False,
        "pad_calibration_required_for_privileged_feasibility": True,
        "operator_result_ui": "ENTER_THEN_EXPLICIT_0_TO_6_SELECTION",
        "canonical_episode_directory": str(root / "episodes"),
        "rejected_episode_directory": str(root / "rejected_episodes"),
        "result_receipt_directory": str(root / "result_receipts"),
        "v1_layout_aggregate_path": str(root / "g2_keyboard_v3_rgbd.hdf5"),
        "v1_layout_rejected_aggregate_path": str(
            root / "g2_keyboard_v3_rejected_rgbd.hdf5"
        ),
        "storage_layout": "V1_DATA_DEMO_N_WITH_V3_EXACT_4D_SEMANTICS",
        "legacy_v1_8d_action_compatible": False,
        "m2": candidate.m2_contact_verdict,
        "training_authorized": False,
    }
    _atomic_json(root / "COLLECTION_MANIFEST.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
