# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.rl.sac.stage1a_grasp_evaluation_contract import (
    GRASP_SUCCESS,
    GRASP_SUCCESS_AUTHORITY,
    STABLE_BILATERAL_CONTROL_STEPS,
    canonical_grasp_evaluation_summary,
    canonical_grasp_evaluation_wandb_metrics,
    episode_contact_timeline_flags,
)


def test_canonical_grasp_evaluation_keeps_stable_as_only_success_authority() -> None:
    raw = {
        "episode_count": 21,
        "contact_rate": 1.0,
        "bilateral_rate": 18.0 / 21.0,
        "stable_rate": 11.0 / 21.0,
        "contact_to_bilateral": 18.0 / 21.0,
        "bilateral_to_stable": 11.0 / 18.0,
        "contact_loss_after_bilateral_rate": 7.0 / 18.0,
    }

    report = canonical_grasp_evaluation_summary(raw)
    wandb = canonical_grasp_evaluation_wandb_metrics(raw)

    assert report["GRASP_SUCCESS_AUTHORITY"] == GRASP_SUCCESS_AUTHORITY
    assert report["GRASP_SUCCESS"] == GRASP_SUCCESS
    assert report["STABLE_BILATERAL_CONTROL_STEPS"] == STABLE_BILATERAL_CONTROL_STEPS
    assert report["GRASP_SUCCESS_STABLE_RATE"] == raw["stable_rate"]
    assert report["BILATERAL_CONTACT_CANDIDATE_RATE"] == raw["bilateral_rate"]
    assert "eval/grasp_success_stable_rate" in wandb
    assert "eval/success_rate" not in wandb
    assert "eval/any_contact_rate" in wandb
    assert "eval/premature_pre_close_contact_rate" in wandb
    assert "eval/post_close_contact_rate" in wandb
    assert "eval/valid_close_ready_contact_rate" in wandb


def test_contact_origin_separates_premature_and_valid_fsm_contact() -> None:
    premature_then_closed = episode_contact_timeline_flags(
        [
            {
                "vector_step": 1,
                "accepted_transitions": 1,
                "contact": True,
                "bilateral": False,
                "stable": False,
                "close_trigger": False,
                "contact_loss_fail": False,
            },
            {
                "vector_step": 2,
                "accepted_transitions": 2,
                "contact": False,
                "bilateral": False,
                "stable": False,
                "close_trigger": True,
                "contact_loss_fail": False,
            },
            {
                "vector_step": 3,
                "accepted_transitions": 3,
                "contact": True,
                "bilateral": True,
                "stable": False,
                "close_trigger": False,
                "contact_loss_fail": False,
            },
            {
                "vector_step": 4,
                "accepted_transitions": 4,
                "contact": False,
                "bilateral": False,
                "stable": False,
                "close_trigger": False,
                "contact_loss_fail": True,
            },
        ]
    )
    assert premature_then_closed["any_contact"] is True
    assert premature_then_closed["premature_pre_close_contact"] is True
    assert premature_then_closed["post_close_contact"] is True
    assert premature_then_closed["valid_close_ready_contact"] is False
    assert premature_then_closed["contact_loss_after_bilateral"] is True

    clean_close = episode_contact_timeline_flags(
        [
            {
                "vector_step": 1,
                "accepted_transitions": 1,
                "contact": False,
                "bilateral": False,
                "stable": False,
                "close_trigger": False,
                "contact_loss_fail": False,
            },
            {
                "vector_step": 2,
                "accepted_transitions": 2,
                "contact": True,
                "bilateral": True,
                "stable": True,
                "close_trigger": True,
                "contact_loss_fail": False,
            },
        ]
    )
    assert clean_close["premature_pre_close_contact"] is False
    assert clean_close["post_close_contact"] is True
    assert clean_close["valid_close_ready_contact"] is True
