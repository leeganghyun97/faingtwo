# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from geniesim.rl.sac.stage1a_termination_receipt import (
    capture_task_termination_receipt,
)


class _Manager:
    active_terms = (
        "precontact_camera_visibility",
        "cube_left_table_persistently",
        "fixed_torso_drift",
        "time_out",
    )

    def __init__(self) -> None:
        self._values = {
            "precontact_camera_visibility": torch.tensor([False, True]),
            "cube_left_table_persistently": torch.tensor([False, False]),
            "fixed_torso_drift": torch.tensor([False, False]),
            "time_out": torch.tensor([False, False]),
        }

    def get_term(self, name: str) -> torch.Tensor:
        return self._values[name]


class _IsaacTensorWrapper:
    def __init__(self, value: torch.Tensor) -> None:
        self.torch = value


def _camera(frame: tuple[int, int]) -> SimpleNamespace:
    rgb = torch.zeros((2, 2, 2, 4), dtype=torch.uint8)
    depth = torch.ones((2, 2, 2, 1), dtype=torch.float32)
    return SimpleNamespace(
        frame=_IsaacTensorWrapper(torch.tensor(frame)),
        _timestamp=torch.tensor([1.0, 1.0]),
        _timestamp_last_update=torch.tensor([0.98, 0.96]),
        data=SimpleNamespace(
            output={"rgb": rgb, "distance_to_image_plane": depth}
        ),
    )


def _frustum(pixel_counts: tuple[int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        rendered_cube_visible_pixel_count=torch.tensor(pixel_counts),
        rendered_cube_visible=torch.tensor([True, False]),
        any_cube_part_in_frustum=torch.tensor([True, True]),
        rendered_depth_consistent=torch.tensor([True, False]),
    )


def test_captures_cached_predicates_and_visibility_without_re_evaluation() -> None:
    manager = _Manager()
    env = SimpleNamespace(
        termination_manager=manager,
        step_dt=0.02,
        scene={
            "head_camera": _camera((8, 9)),
            "right_wrist_camera": _camera((18, 19)),
        },
        _g2_precontact_camera_visibility_last_result=SimpleNamespace(
            cameras={
                "head": _frustum((11, 0)),
                "right_wrist": _frustum((13, 0)),
            },
            visible_camera_count=torch.tensor([2, 0]),
            precontact_pass=torch.tensor([True, False]),
        ),
        _g2_precontact_camera_loss_age_s=torch.tensor([0.0, 0.32]),
        _g2_precontact_camera_visibility_last_loss_now=torch.tensor([False, True]),
        _g2_task_outside_table_time_s=torch.tensor([0.0, 0.0]),
    )

    receipt = capture_task_termination_receipt(
        env=env,
        env_id=1,
        episode_id=4,
        source_sample_id="episode-00005/row-000133",
        control_step=110,
        vector_step=380,
        runtime_terminated=True,
        runtime_truncated=False,
        reward_done=False,
        gate_receipt={"geometry_valid": True, "owner_valid": True},
        cube_position_world_m=torch.tensor([0.5, 0.0, 0.8]),
        robot_root_position_world_m=torch.tensor([0.0, 0.0, 0.0]),
        maximum_episode_control_steps=640,
    )

    assert receipt["triggered_predicate_names"] == [
        "precontact_camera_visibility"
    ]
    assert receipt["termination_flag_sources"] == ["RUNTIME_TERMINATED"]
    assert receipt["reset_scheduler_request"] is True
    assert receipt["precontact_visibility_triggered"] is True
    assert receipt["visibility"]["visibility_loss_age_ms"] == pytest.approx(320.0)
    assert receipt["visibility"]["visibility_consecutive_lost_control_steps"] == 16
    assert receipt["visibility"]["cameras"]["head"]["cube_visible_pixel_count"] == 0
    assert receipt["visibility"]["cameras"]["right_wrist"]["frame_id"] == 19
    assert receipt["geometry"]["geometry_valid"] is True
    assert receipt["behavior_changed"] is False
