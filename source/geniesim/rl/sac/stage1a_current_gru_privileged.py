# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Causal deployable GRU auxiliary supervised by privileged CLOSE labels.

This module is deliberately independent from SAC replay and runtime gripper
authority.  Its inputs are the current frozen 128-D policy feature and the
current frozen Conditional-CORAL Wrist score.  Privileged geometry appears
only on the loss side as an exact binary target and normalized signed margin.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA = "g2_stage1a_current_gru_privileged_auxiliary_v1"
INPUT_DIM = 129
HIDDEN_SIZE = 64
NUM_LAYERS = 1
SEQUENCE_LENGTH = 8


class CurrentGruPrivilegedError(ValueError):
    pass


class _CausalCloseReadinessGru(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalizer = torch.nn.LayerNorm(INPUT_DIM)
        self.gru = torch.nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.binary_head = torch.nn.Linear(HIDDEN_SIZE, 1)
        self.margin_head = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _hidden = self.gru(self.normalizer(sequence))
        last = encoded[:, -1, :]
        return self.binary_head(last).squeeze(-1), self.margin_head(last).squeeze(-1)


@dataclass(frozen=True)
class CurrentGruPrivilegedReceipt:
    score: float
    predicted_margin: float
    trained: bool
    binary_loss: float
    signed_margin_loss: float
    gru_loss: float
    teacher_target: bool | None
    teacher_margin: float | None
    sequence_steps: int


class CurrentGruPrivilegedAuxiliary:
    """Per-environment causal GRU with replay-isolated teacher supervision."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        learning_rate: float = 1.0e-4,
        margin_weight: float = 0.25,
    ) -> None:
        if int(num_envs) <= 0:
            raise CurrentGruPrivilegedError("CURRENT_GRU_NUM_ENVS_INVALID")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.model = _CausalCloseReadinessGru().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=1.0e-5
        )
        self.margin_weight = float(margin_weight)
        self.histories = [
            deque(maxlen=SEQUENCE_LENGTH) for _ in range(self.num_envs)
        ]
        self.positive_count = 0
        self.negative_count = 0
        self.teacher_rows = 0
        self.update_count = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.binary_loss_sum = 0.0
        self.margin_loss_sum = 0.0
        self.gru_loss_sum = 0.0
        self.last_receipt: CurrentGruPrivilegedReceipt | None = None

    def reset(self, env_id: int) -> None:
        self._check_env(env_id)
        self.histories[int(env_id)].clear()

    def _check_env(self, env_id: int) -> None:
        if not 0 <= int(env_id) < self.num_envs:
            raise CurrentGruPrivilegedError("CURRENT_GRU_ENV_ID_INVALID")

    @staticmethod
    def _input(actor_observation: Any, frozen_student_score: float) -> np.ndarray:
        feature = np.asarray(actor_observation, dtype=np.float32)
        score = float(frozen_student_score)
        if feature.shape != (128,) or not np.isfinite(feature).all():
            raise CurrentGruPrivilegedError("CURRENT_GRU_FEATURE_INVALID")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise CurrentGruPrivilegedError("CURRENT_GRU_FROZEN_SCORE_INVALID")
        return np.concatenate((feature, np.asarray([score], dtype=np.float32)))

    def observe(
        self,
        *,
        env_id: int,
        actor_observation: Any,
        frozen_student_score: float,
        teacher_target: bool | None,
        signed_margin: float | None,
        supervision_eligible: bool,
    ) -> CurrentGruPrivilegedReceipt:
        """Append one causal input and optionally perform one auxiliary update."""

        self._check_env(env_id)
        vector = self._input(actor_observation, frozen_student_score)
        history = self.histories[int(env_id)]
        history.append(vector.copy())
        sequence = torch.as_tensor(
            np.stack(tuple(history)), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.model.train(bool(supervision_eligible))
        logit, predicted_margin = self.model(sequence)
        score = torch.sigmoid(logit)
        trained = False
        binary_loss_value = margin_loss_value = gru_loss_value = 0.0

        if supervision_eligible:
            if teacher_target is None or signed_margin is None:
                raise CurrentGruPrivilegedError(
                    "CURRENT_GRU_ELIGIBLE_TEACHER_TARGET_MISSING"
                )
            margin_value = float(signed_margin)
            if not math.isfinite(margin_value):
                raise CurrentGruPrivilegedError("CURRENT_GRU_SIGNED_MARGIN_INVALID")
            target_value = 1.0 if bool(teacher_target) else 0.0
            # Running inverse-frequency balancing uses only past/current
            # teacher labels; it does not inspect a future row or family id.
            positives = self.positive_count + int(target_value == 1.0)
            negatives = self.negative_count + int(target_value == 0.0)
            total = positives + negatives
            class_weight = (
                total / (2.0 * positives)
                if target_value == 1.0
                else total / (2.0 * negatives)
            )
            target = torch.tensor([target_value], dtype=torch.float32, device=self.device)
            margin_target = torch.tensor(
                [margin_value], dtype=torch.float32, device=self.device
            )
            binary_loss = F.binary_cross_entropy_with_logits(
                logit.reshape(1), target, reduction="none"
            ).mean() * float(class_weight)
            margin_loss = F.smooth_l1_loss(
                predicted_margin.reshape(1), margin_target
            )
            loss = binary_loss + self.margin_weight * margin_loss
            if not torch.isfinite(loss):
                raise CurrentGruPrivilegedError("CURRENT_GRU_LOSS_NONFINITE")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise CurrentGruPrivilegedError("CURRENT_GRU_GRADIENT_NONFINITE")
            self.optimizer.step()
            trained = True
            self.positive_count = positives
            self.negative_count = negatives
            self.teacher_rows += 1
            self.update_count += 1
            binary_loss_value = float(binary_loss.detach())
            margin_loss_value = float(margin_loss.detach())
            gru_loss_value = float(loss.detach())
            self.binary_loss_sum += binary_loss_value
            self.margin_loss_sum += margin_loss_value
            self.gru_loss_sum += gru_loss_value

            predicted_ready = bool(float(score.detach()) >= 0.5)
            target_ready = bool(teacher_target)
            self.true_positive += int(predicted_ready and target_ready)
            self.true_negative += int(not predicted_ready and not target_ready)
            self.false_positive += int(predicted_ready and not target_ready)
            self.false_negative += int(not predicted_ready and target_ready)

        receipt = CurrentGruPrivilegedReceipt(
            score=float(score.detach()),
            predicted_margin=float(predicted_margin.detach()),
            trained=trained,
            binary_loss=binary_loss_value,
            signed_margin_loss=margin_loss_value,
            gru_loss=gru_loss_value,
            teacher_target=None if teacher_target is None else bool(teacher_target),
            teacher_margin=None if signed_margin is None else float(signed_margin),
            sequence_steps=len(history),
        )
        self.last_receipt = receipt
        return receipt

    def metrics(self) -> dict[str, float | int | bool]:
        denominator = self.teacher_rows
        agreement = self.true_positive + self.true_negative
        false_accept_denominator = self.false_positive + self.true_negative
        false_reject_denominator = self.false_negative + self.true_positive
        mean = lambda value: float(value / denominator) if denominator else 0.0
        rate = lambda value, total: float(value / total) if total else 0.0
        return {
            "enabled": True,
            "schema": SCHEMA,
            "input_dim": INPUT_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "sequence_length": SEQUENCE_LENGTH,
            "student_privileged_input_count": 0,
            "teacher_rows": denominator,
            "positive_rows": self.positive_count,
            "negative_rows": self.negative_count,
            "update_count": self.update_count,
            "teacher_student_agreement": rate(agreement, denominator),
            "false_accept_rate": rate(self.false_positive, false_accept_denominator),
            "false_reject_rate": rate(self.false_negative, false_reject_denominator),
            "binary_loss": mean(self.binary_loss_sum),
            "signed_margin_loss": mean(self.margin_loss_sum),
            "gru_loss": mean(self.gru_loss_sum),
        }

    def save(self, path: Path, *, runtime_variant: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "schema": SCHEMA,
                "runtime_variant": str(runtime_variant),
                "student_privileged_input_count": 0,
                "input_fields": [
                    "causal_frozen_gru_feature_128d",
                    "frozen_family_invariant_wrist_score",
                ],
                "teacher_fields_excluded_from_input": [
                    "privileged_close_ready_target",
                    "signed_readiness_margin",
                ],
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": self.metrics(),
            },
            temporary,
        )
        temporary.replace(path)


__all__ = (
    "CurrentGruPrivilegedAuxiliary",
    "CurrentGruPrivilegedError",
    "CurrentGruPrivilegedReceipt",
    "HIDDEN_SIZE",
    "INPUT_DIM",
    "NUM_LAYERS",
    "SCHEMA",
    "SEQUENCE_LENGTH",
)
