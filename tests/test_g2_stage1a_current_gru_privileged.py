# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from pathlib import Path

import numpy as np
import torch

from geniesim.rl.sac.stage1a_current_gru_privileged import (
    CurrentGruPrivilegedAuxiliary,
    HIDDEN_SIZE,
    INPUT_DIM,
    NUM_LAYERS,
)


def test_current_gru_privileged_causal_teacher_only_contract(tmp_path: Path) -> None:
    torch.manual_seed(7)
    auxiliary = CurrentGruPrivilegedAuxiliary(num_envs=2, device="cpu")
    first = auxiliary.observe(
        env_id=0,
        actor_observation=np.zeros(128, dtype=np.float32),
        frozen_student_score=0.25,
        teacher_target=False,
        signed_margin=-0.4,
        supervision_eligible=True,
    )
    assert first.trained is True
    assert first.sequence_steps == 1
    assert 0.0 <= first.score <= 1.0
    auxiliary.observe(
        env_id=0,
        actor_observation=np.ones(128, dtype=np.float32),
        frozen_student_score=0.75,
        teacher_target=True,
        signed_margin=0.2,
        supervision_eligible=True,
    )
    metrics = auxiliary.metrics()
    assert metrics["input_dim"] == INPUT_DIM == 129
    assert metrics["hidden_size"] == HIDDEN_SIZE == 64
    assert metrics["num_layers"] == NUM_LAYERS == 1
    assert metrics["student_privileged_input_count"] == 0
    assert metrics["teacher_rows"] == 2
    assert metrics["positive_rows"] == metrics["negative_rows"] == 1

    auxiliary.reset(0)
    reset_receipt = auxiliary.observe(
        env_id=0,
        actor_observation=np.zeros(128, dtype=np.float32),
        frozen_student_score=0.5,
        teacher_target=None,
        signed_margin=None,
        supervision_eligible=False,
    )
    assert reset_receipt.sequence_steps == 1
    assert reset_receipt.trained is False

    checkpoint = tmp_path / "aux.pt"
    auxiliary.save(checkpoint, runtime_variant="TEST")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["student_privileged_input_count"] == 0
    assert "privileged_close_ready_target" in payload[
        "teacher_fields_excluded_from_input"
    ]
