# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.rl.sac.stage1a_deferred_wandb import DeferredWandbRun


class _FakeRun:
    id = "fake-id"
    url = "https://example.invalid/fake"

    def __init__(self) -> None:
        self.logged = []
        self.summary = {}
        self.finished = []

    def log(self, payload, *, step):
        self.logged.append((payload, step))

    def finish(self, *args, **kwargs):
        self.finished.append((args, kwargs))


def test_deferred_run_does_not_materialize_on_pre_gate_finish() -> None:
    created = []
    run = DeferredWandbRun(lambda: created.append(_FakeRun()) or created[-1])

    run.finish(exit_code=2)

    assert created == []
    assert run.initialized is False
    assert run.initialization_count == 0


def test_deferred_run_materializes_once_on_first_post_gate_log() -> None:
    created = []
    run = DeferredWandbRun(lambda: created.append(_FakeRun()) or created[-1])

    run.log({"accepted_transitions": 1}, step=1)
    run.log({"accepted_transitions": 2}, step=2)

    assert len(created) == 1
    assert run.initialized is True
    assert run.initialization_count == 1
    assert created[0].logged == [
        ({"accepted_transitions": 1}, 1),
        ({"accepted_transitions": 2}, 2),
    ]
