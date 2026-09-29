# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.rl.sac.stage1a_close_readiness_advisory import (
    CloseReadinessAdvice,
    FrozenCloseReadinessAdvisory,
)


def test_advisory_never_changes_fsm_close_authority() -> None:
    rule = FrozenCloseReadinessAdvisory(.79, .31)
    assert rule.advise(.90) is CloseReadinessAdvice.READY_ADVISORY
    assert rule.advise(.50) is CloseReadinessAdvice.DEFER_TO_FSM
    assert rule.advise(.20) is CloseReadinessAdvice.NOT_READY_ADVISORY
    assert rule.preserve_fsm_close(fsm_close_triggered=True) is True
    assert rule.preserve_fsm_close(fsm_close_triggered=False) is False
