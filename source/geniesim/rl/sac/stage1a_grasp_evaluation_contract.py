# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical, episode-level Stage-1A grasp evaluation semantics.

This module deliberately distinguishes geometric/contact milestones from the
only terminal grasp-success authority.  It contains no policy, reward, or
controller logic, so it can be shared by future reports and W&B publishers
without changing action authority.
"""

from __future__ import annotations

from typing import Any, Mapping


CONTROL_HZ = 50
STABLE_BILATERAL_CONTROL_STEPS = 10
STABLE_BILATERAL_DURATION_MS = 200
GRASP_SUCCESS = "STABLE_GRASP"
GRASP_SUCCESS_AUTHORITY = "STABLE_ONLY"

CONTACT_DEFINITION = (
    "At least one primary pad is in contact with the cube; this is contact "
    "only and is not grasp success."
)
PREMATURE_PRE_CLOSE_CONTACT_DEFINITION = (
    "At least one primary-pad contact is observed before the first canonical "
    "FSM CLOSE onset in an episode.  It is a reset/initial-state diagnostic, "
    "not a valid CLOSE result."
)
POST_CLOSE_CONTACT_DEFINITION = (
    "At least one primary-pad contact is observed at or after the first "
    "canonical FSM CLOSE onset in an episode."
)
VALID_CLOSE_READY_CONTACT_DEFINITION = (
    "Post-CLOSE primary-pad contact from an episode with no earlier "
    "pre-CLOSE contact.  The canonical FSM CLOSE onset is the ready-state "
    "authority; this remains contact only, not grasp success."
)
BILATERAL_DEFINITION = (
    "Inner and outer primary pads simultaneously contact the cube; this is "
    "a momentary bilateral-contact grasp candidate, not final grasp success."
)
STABLE_GRASP_DEFINITION = (
    "Bilateral primary-pad contact is maintained for 10 consecutive 50-Hz "
    "control steps (approximately 200 ms)."
)

# These are the sole canonical names for future report/W&B emission.  The
# values are episode rates, never transition rates.
WANDB_GRASP_EVALUATION_KEYS = {
    "contact_rate": "eval/contact_rate",
    "any_contact_rate": "eval/any_contact_rate",
    "premature_pre_close_contact_rate": "eval/premature_pre_close_contact_rate",
    "post_close_contact_rate": "eval/post_close_contact_rate",
    "valid_close_ready_contact_rate": "eval/valid_close_ready_contact_rate",
    "bilateral_contact_candidate_rate": "eval/bilateral_contact_candidate_rate",
    "grasp_success_stable_rate": "eval/grasp_success_stable_rate",
    "contact_to_bilateral": "eval/contact_to_bilateral",
    "bilateral_to_stable": "eval/bilateral_to_stable",
    "post_bilateral_contact_loss_rate": "eval/post_bilateral_contact_loss_rate",
}


def episode_contact_timeline_flags(
    rows: list[Mapping[str, Any]],
) -> dict[str, bool]:
    """Classify contact relative to the first canonical CLOSE onset.

    Contact is sampled after the canonical action packet for a control row.
    A contact observed on the same row as ``close_trigger`` therefore belongs
    to the post-CLOSE side of the event timeline.  A contact strictly before
    it is a reset/initial-state diagnostic and cannot count as a valid
    CLOSE-to-contact conversion.

    This function is intentionally pure report bookkeeping.  It has no
    reward, replay, action, controller, or CLOSE-admission authority.
    """

    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("vector_step", -1)),
            int(row.get("accepted_transitions", -1)),
        ),
    )
    close_index = next(
        (
            index
            for index, row in enumerate(ordered)
            if bool(row.get("close_trigger", False))
        ),
        None,
    )
    any_contact = any(bool(row.get("contact", False)) for row in ordered)
    any_bilateral = any(bool(row.get("bilateral", False)) for row in ordered)
    any_stable = any(bool(row.get("stable", False)) for row in ordered)

    # A carried-over CLOSE latch is not an in-episode CLOSE onset.  If no
    # onset exists, any contact remains pre-CLOSE so a reset fault cannot be
    # hidden as a valid close conversion.
    premature_pre_close_contact = any(
        bool(row.get("contact", False))
        for row in (
            ordered if close_index is None else ordered[:close_index]
        )
    )
    post_close_rows = () if close_index is None else ordered[close_index:]
    post_close_contact = any(
        bool(row.get("contact", False)) for row in post_close_rows
    )
    post_close_bilateral = any(
        bool(row.get("bilateral", False)) for row in post_close_rows
    )
    post_close_stable = any(
        bool(row.get("stable", False)) for row in post_close_rows
    )

    bilateral_index = next(
        (
            index
            for index, row in enumerate(ordered)
            if bool(row.get("bilateral", False))
        ),
        None,
    )
    contact_loss_after_bilateral = bool(
        bilateral_index is not None
        and any(
            bool(row.get("contact_loss_fail", False))
            for row in ordered[bilateral_index:]
        )
    )
    close = close_index is not None
    return {
        "any_contact": any_contact,
        "any_bilateral": any_bilateral,
        "any_stable": any_stable,
        "close": close,
        "premature_pre_close_contact": premature_pre_close_contact,
        "post_close_contact": post_close_contact,
        "post_close_bilateral": post_close_bilateral,
        "post_close_stable": post_close_stable,
        # The canonical FSM CLOSE onset is the ready-state authority.  A
        # valid contact additionally has no earlier pre-CLOSE contact.
        "valid_close_ready_contact": bool(
            close and post_close_contact and not premature_pre_close_contact
        ),
        "contact_loss_after_bilateral": contact_loss_after_bilateral,
    }


def canonical_grasp_evaluation_summary(
    summary: Mapping[str, Any],
) -> dict[str, float | int | str]:
    """Translate a completed-episode summary into the canonical schema.

    ``summary`` must already have used completed episodes as its denominator.
    The function intentionally does not derive rewards or redefine an event.
    """

    return {
        "EVALUATION_DENOMINATOR": "COMPLETED_EPISODES_ONLY",
        "CONTACT_DEFINITION": CONTACT_DEFINITION,
        "PREMATURE_PRE_CLOSE_CONTACT_DEFINITION": (
            PREMATURE_PRE_CLOSE_CONTACT_DEFINITION
        ),
        "POST_CLOSE_CONTACT_DEFINITION": POST_CLOSE_CONTACT_DEFINITION,
        "VALID_CLOSE_READY_CONTACT_DEFINITION": (
            VALID_CLOSE_READY_CONTACT_DEFINITION
        ),
        "BILATERAL_DEFINITION": BILATERAL_DEFINITION,
        "STABLE_GRASP_DEFINITION": STABLE_GRASP_DEFINITION,
        "GRASP_SUCCESS": GRASP_SUCCESS,
        "GRASP_SUCCESS_AUTHORITY": GRASP_SUCCESS_AUTHORITY,
        "CONTROL_HZ": CONTROL_HZ,
        "STABLE_BILATERAL_CONTROL_STEPS": STABLE_BILATERAL_CONTROL_STEPS,
        "STABLE_BILATERAL_DURATION_MS": STABLE_BILATERAL_DURATION_MS,
        "COMPLETED_EPISODE_COUNT": int(summary["episode_count"]),
        # ``CONTACT_RATE`` is preserved as a backwards-compatible alias.  New
        # reports must use the explicit names below when they need to reason
        # about reset/pre-CLOSE confounding.
        "ANY_CONTACT_RATE": float(
            summary.get("any_contact_rate", summary["contact_rate"])
        ),
        "CONTACT_RATE": float(summary["contact_rate"]),
        "PREMATURE_PRE_CLOSE_CONTACT_RATE": float(
            summary.get("premature_pre_close_contact_rate", 0.0)
        ),
        "POST_CLOSE_CONTACT_RATE": float(
            summary.get("post_close_contact_rate", 0.0)
        ),
        "VALID_CLOSE_READY_CONTACT_RATE": float(
            summary.get("valid_close_ready_contact_rate", 0.0)
        ),
        "BILATERAL_CONTACT_CANDIDATE_RATE": float(summary["bilateral_rate"]),
        "GRASP_SUCCESS_STABLE_RATE": float(summary["stable_rate"]),
        "CONTACT_TO_BILATERAL": float(summary["contact_to_bilateral"]),
        "BILATERAL_TO_STABLE": float(summary["bilateral_to_stable"]),
        "POST_BILATERAL_CONTACT_LOSS_RATE": float(
            summary["contact_loss_after_bilateral_rate"]
        ),
    }


def canonical_grasp_evaluation_wandb_metrics(
    summary: Mapping[str, Any],
) -> dict[str, float]:
    """Return only unambiguous episode-level W&B metric names."""

    return {
        WANDB_GRASP_EVALUATION_KEYS["contact_rate"]: float(summary["contact_rate"]),
        WANDB_GRASP_EVALUATION_KEYS["any_contact_rate"]: float(
            summary.get("any_contact_rate", summary["contact_rate"])
        ),
        WANDB_GRASP_EVALUATION_KEYS["premature_pre_close_contact_rate"]: float(
            summary.get("premature_pre_close_contact_rate", 0.0)
        ),
        WANDB_GRASP_EVALUATION_KEYS["post_close_contact_rate"]: float(
            summary.get("post_close_contact_rate", 0.0)
        ),
        WANDB_GRASP_EVALUATION_KEYS["valid_close_ready_contact_rate"]: float(
            summary.get("valid_close_ready_contact_rate", 0.0)
        ),
        WANDB_GRASP_EVALUATION_KEYS["bilateral_contact_candidate_rate"]: float(
            summary["bilateral_rate"]
        ),
        WANDB_GRASP_EVALUATION_KEYS["grasp_success_stable_rate"]: float(
            summary["stable_rate"]
        ),
        WANDB_GRASP_EVALUATION_KEYS["contact_to_bilateral"]: float(
            summary["contact_to_bilateral"]
        ),
        WANDB_GRASP_EVALUATION_KEYS["bilateral_to_stable"]: float(
            summary["bilateral_to_stable"]
        ),
        WANDB_GRASP_EVALUATION_KEYS["post_bilateral_contact_loss_rate"]: float(
            summary["contact_loss_after_bilateral_rate"]
        ),
    }
