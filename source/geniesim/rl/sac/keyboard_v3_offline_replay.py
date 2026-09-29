# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Offline-only keyboard-v3 and grasp-GRU replay/debugging support.

This module deliberately imports no Isaac/Omniverse package.  It reads the
canonical Head/Right-Wrist keyboard-v3 artifact, applies the strict Wrist-only
GRU checkpoint loader, and reconstructs every seek from the episode start so
the displayed recurrent prediction is causal.  Head remains replay-visible but
is never forwarded to the grasp actor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .human_grasp_gru_bc import (
    HumanGraspGRUOutput,
    HumanGraspSequenceInputs,
    HysteresisCalibration,
    load_human_grasp_checkpoint,
)
from .keyboard_grasp_contract import canonical_keyboard_grasp_contract
from .keyboard_v3_dataset import (
    KEYBOARD_V3_CAMERA_NAMES,
    KEYBOARD_V3_OPERATOR_VIEW_NAMES,
    KeyboardV3Episode,
    episode_to_gru,
    read_keyboard_v3_episode,
    read_recovered_keyboard_v3_episode,
    validate_episode,
)
from .keyboard_v3_recovery import load_recovery_manifest


OFFLINE_REPLAY_SCHEMA = "g2_keyboard_v3_offline_gru_replay_v1"


class KeyboardV3OfflineReplayError(RuntimeError):
    """Fail-closed replay error; never fall back to another data/model path."""


@dataclass(frozen=True)
class ReplayPrediction:
    step: int
    expert_xyz_m: tuple[float, float, float]
    predicted_xyz_m: tuple[float, float, float]
    residual_xyz_m: tuple[float, float, float]
    final_xyz_m: tuple[float, float, float]
    close_probability: float
    feasibility_probability: float
    expert_closed: bool
    predicted_closed: bool
    action_bound_violation: bool
    teacher_feasible: bool | None


def _slice_inputs(
    value: HumanGraspSequenceInputs, stop: int, device: torch.device
) -> HumanGraspSequenceInputs:
    if stop <= 0:
        raise KeyboardV3OfflineReplayError("causal replay prefix must be non-empty")
    payload: dict[str, torch.Tensor] = {}
    for item in fields(HumanGraspSequenceInputs):
        tensor = getattr(value, item.name)
        payload[item.name] = tensor[:, :stop].to(device)
    return HumanGraspSequenceInputs(**payload)


def _classification_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    truth = target.astype(bool)
    guess = predicted.astype(bool)
    tp = int(np.count_nonzero(guess & truth))
    fp = int(np.count_nonzero(guess & ~truth))
    fn = int(np.count_nonzero(~guess & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _hysteresis(
    probability: np.ndarray, calibration: HysteresisCalibration
) -> np.ndarray:
    result = np.zeros(probability.shape, dtype=np.bool_)
    closed = False
    for index, value in enumerate(probability):
        if not closed and float(value) >= calibration.close_threshold:
            closed = True
        elif closed and float(value) <= calibration.open_threshold:
            closed = False
        result[index] = closed
    return result


def _metrics(
    output: HumanGraspGRUOutput,
    episode: KeyboardV3Episode,
    calibration: HysteresisCalibration,
) -> dict[str, float]:
    predicted_xyz = output.xyz_action_robot_root_m[0].detach().cpu().numpy()
    expert_xyz = np.asarray(episode.actions["action_4d"], dtype=np.float64)[:, :3]
    error = predicted_xyz - expert_xyz
    rmse_mm = float(np.sqrt(np.mean(np.square(error))) * 1000.0)
    directional = np.linalg.norm(expert_xyz, axis=1) > 1.0e-8
    if np.any(directional):
        predicted_norm = np.linalg.norm(predicted_xyz[directional], axis=1)
        expert_norm = np.linalg.norm(expert_xyz[directional], axis=1)
        denominator = np.maximum(predicted_norm * expert_norm, 1.0e-12)
        cosine = np.sum(
            predicted_xyz[directional] * expert_xyz[directional], axis=1
        ) / denominator
        direction_error = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())
    else:
        direction_error = 0.0
    probability = output.close_probability[0, :, 0].detach().cpu().numpy()
    predicted_close = _hysteresis(probability, calibration)
    expert_close = np.asarray(episode.events["close_state"], dtype=np.bool_)
    precision, recall, f1 = _classification_metrics(predicted_close, expert_close)
    predicted_edges = np.flatnonzero(predicted_close & ~np.r_[False, predicted_close[:-1]])
    expert_edges = np.flatnonzero(
        np.asarray(episode.events["close_edge"], dtype=np.bool_)
    )
    first_predicted = int(predicted_edges[0]) if predicted_edges.size else len(probability)
    first_expert = int(expert_edges[0]) if expert_edges.size else len(probability)
    bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
    violations = int(np.count_nonzero(np.linalg.norm(predicted_xyz, axis=1) > bound + 1.0e-8))
    return {
        "xyz_rmse_mm": rmse_mm,
        "xyz_direction_error_deg": direction_error,
        "close_precision": precision,
        "close_recall": recall,
        "close_f1": f1,
        "early_close_steps": float(max(0, first_expert - first_predicted)),
        "late_close_steps": float(max(0, first_predicted - first_expert)),
        "action_bound_violation_count": float(violations),
    }


class KeyboardV3OfflineReplay:
    """One validated episode and one strict GRU checkpoint, replayed offline."""

    def __init__(
        self,
        dataset_path: str | Path,
        checkpoint_path: str | Path,
        *,
        episode_id: str | None = None,
        device: str | torch.device = "cpu",
        residual_checkpoint_path: str | Path | None = None,
        recovery_manifest_path: str | Path | None = None,
    ) -> None:
        if residual_checkpoint_path is not None:
            raise KeyboardV3OfflineReplayError(
                "RESIDUAL_SAC_CHECKPOINT_LOADER_NOT_IMPLEMENTED; refusing fake residual output"
            )
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device)
        self.recovery_manifest_path = (
            None
            if recovery_manifest_path is None
            else Path(recovery_manifest_path).expanduser().resolve()
        )
        self.available_episode_ids: tuple[str, ...]
        try:
            if self.recovery_manifest_path is not None:
                manifest = load_recovery_manifest(self.recovery_manifest_path)
                records = {
                    str(record["training_key"]): record
                    for record in manifest["eligible_episodes"]
                }
                self.available_episode_ids = tuple(sorted(records))
                selected = episode_id or self.available_episode_ids[0]
                if selected not in records:
                    raise KeyboardV3OfflineReplayError(
                        f"episode is not recovery-authorized: {selected}"
                    )
                record = records[selected]
                self.dataset_path = Path(record["source_path"]).resolve()
                self.episode = read_recovered_keyboard_v3_episode(
                    self.dataset_path,
                    expected_sha256=str(record["source_sha256"]),
                    episode_id=str(record["episode_id"]),
                )
                self.selected_episode_id = selected
                self.recovery_manifest_sha256 = str(manifest["manifest_sha256"])
            else:
                self.episode = read_keyboard_v3_episode(
                    self.dataset_path, episode_id=episode_id
                )
                self.available_episode_ids = (self.episode.episode_id,)
                self.selected_episode_id = self.episode.episode_id
                self.recovery_manifest_sha256 = None
            self.validation = validate_episode(self.episode)
            if not self.validation.passed:
                raise KeyboardV3OfflineReplayError(
                    "dataset validation failed: "
                    + ",".join(self.validation.errors)
                )
            self.inputs, self.targets = episode_to_gru(self.episode)
            self.model, self.checkpoint_receipt = load_human_grasp_checkpoint(
                self.checkpoint_path, device=self.device
            )
        except KeyboardV3OfflineReplayError:
            raise
        except Exception as error:
            raise KeyboardV3OfflineReplayError(str(error)) from error
        self.model.eval()
        contract = self.model.config.contract_payload()
        if contract.get("action_output") != "[dx,dy,dz,p_close]":
            raise KeyboardV3OfflineReplayError("checkpoint action output mismatch")
        if contract.get("xyz_frame") != "robot_root" or contract.get("xyz_unit") != "m":
            raise KeyboardV3OfflineReplayError("checkpoint frame/unit mismatch")
        if contract.get("wrist_rgbd_only") is not True or contract.get("head_camera") is not False:
            raise KeyboardV3OfflineReplayError(
                "checkpoint camera contract is not the supported Wrist-only model"
            )
        self.calibration: HysteresisCalibration = self.checkpoint_receipt[
            "calibration"
        ]
        with torch.inference_mode():
            self.full_output = self.model(
                _slice_inputs(
                    self.inputs,
                    int(self.inputs.right_wrist_rgb.shape[1]),
                    self.device,
                )
            )
        self.metrics = _metrics(self.full_output, self.episode, self.calibration)
        probability = self.full_output.close_probability[0, :, 0].detach().cpu().numpy()
        self.full_hysteresis = _hysteresis(probability, self.calibration)

    @property
    def row_count(self) -> int:
        return int(np.asarray(self.episode.time["control_step"]).shape[0])

    def frame(self, step: int) -> ReplayPrediction:
        """Re-run [0..step] from zero hidden state; never infer an isolated frame."""

        if not 0 <= int(step) < self.row_count:
            raise KeyboardV3OfflineReplayError("replay step out of range")
        step = int(step)
        with torch.inference_mode():
            output = self.model(_slice_inputs(self.inputs, step + 1, self.device))
        predicted_xyz = output.xyz_action_robot_root_m[0, -1].detach().cpu().numpy()
        expert = np.asarray(self.episode.actions["action_4d"], dtype=np.float64)[step]
        close_probability = float(output.close_probability[0, -1, 0].detach().cpu())
        feasibility_probability = float(
            output.feasibility_probability[0, -1, 0].detach().cpu()
        )
        prefix_probability = output.close_probability[0, :, 0].detach().cpu().numpy()
        predicted_closed = bool(_hysteresis(prefix_probability, self.calibration)[-1])
        teacher: bool | None = None
        if "feasible_target" in self.episode.privileged:
            teacher = bool(np.asarray(self.episode.privileged["feasible_target"])[step])
        residual = np.zeros(3, dtype=np.float64)
        final = predicted_xyz.astype(np.float64)
        bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
        return ReplayPrediction(
            step=step,
            expert_xyz_m=tuple(float(value) for value in expert[:3]),
            predicted_xyz_m=tuple(float(value) for value in predicted_xyz),
            residual_xyz_m=tuple(float(value) for value in residual),
            final_xyz_m=tuple(float(value) for value in final),
            close_probability=close_probability,
            feasibility_probability=feasibility_probability,
            expert_closed=bool(expert[3]),
            predicted_closed=predicted_closed,
            action_bound_violation=bool(np.linalg.norm(final) > bound + 1.0e-8),
            teacher_feasible=teacher,
        )

    def camera_frame(self, camera_name: str, step: int) -> tuple[np.ndarray, np.ndarray]:
        if camera_name not in KEYBOARD_V3_CAMERA_NAMES:
            raise KeyboardV3OfflineReplayError(f"unknown camera: {camera_name}")
        if not 0 <= int(step) < self.row_count:
            raise KeyboardV3OfflineReplayError("camera step out of range")
        rgb = np.asarray(self.episode.observations[f"{camera_name}_rgb"])[step]
        depth = np.asarray(self.episode.observations[f"{camera_name}_depth_m"])[step]
        return rgb, depth

    def report(self) -> dict[str, Any]:
        return {
            "schema": OFFLINE_REPLAY_SCHEMA,
            "dataset_path": str(self.dataset_path),
            "episode_id": self.selected_episode_id,
            "source_episode_id": self.episode.episode_id,
            "recovery_manifest": (
                str(self.recovery_manifest_path)
                if self.recovery_manifest_path is not None else None
            ),
            "recovery_manifest_sha256": self.recovery_manifest_sha256,
            "row_count": self.row_count,
            "recorded_camera_count": len(KEYBOARD_V3_CAMERA_NAMES),
            "recorded_camera_names": list(KEYBOARD_V3_CAMERA_NAMES),
            "grasp_actor_camera_names": ["right_wrist"],
            "head_camera_policy_input": False,
            "operator_view_count": len(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
            "operator_view_names": list(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
            "success": bool(self.episode.metadata.get("success", False)),
            "failure_type": str(self.episode.metadata.get("failure_type")),
            "curobo_backoff_m": float(
                self.episode.metadata.get("curobo_backoff_m")
            ),
            "start_offset_xyz_m": np.asarray(
                self.episode.metadata.get("start_offset_xyz_m"), dtype=np.float64
            ).tolist(),
            "schema_version": str(self.episode.metadata.get("schema_version")),
            "validator_result": "PASS" if self.validation.passed else "FAIL",
            "close_event_count": self.validation.close_event_count,
            "nonzero_xyz_row_count": self.validation.nonzero_micro_approach_rows,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_receipt["file_sha256"],
            "checkpoint_state_sha256": self.checkpoint_receipt["state_sha256"],
            "checkpoint_input_contract": "RIGHT_WRIST_RGBD_PLUS_STATE_CAUSAL_GRU",
            "dataset_extra_views_are_checkpoint_inputs": False,
            "residual_sac": "NOT_AVAILABLE",
            "open_threshold": self.calibration.open_threshold,
            "close_threshold": self.calibration.close_threshold,
            "metrics": dict(self.metrics),
            "isaac_started": False,
            "physics_steps": 0,
            "robot_asset_loaded": False,
            "action_submissions": 0,
            "training_started": False,
        }


__all__ = [
    "KeyboardV3OfflineReplay",
    "KeyboardV3OfflineReplayError",
    "OFFLINE_REPLAY_SCHEMA",
    "ReplayPrediction",
]
