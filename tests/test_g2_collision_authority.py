# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from geniesim.rl.isaaclab.g2_collision_authority import (
    G2_DIAGNOSTIC_FILTERED_CONTACT_SENSORS,
    G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS,
    G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS,
    G2_RIGHT_PAD_FILTER_TARGETS,
    G2ForbiddenCollisionEvaluator,
    install_g2_forbidden_collision_sensors,
)


class _Scene(dict):
    pass


def _env(*, collision_sensor: str | None = None):
    scene = _Scene()
    for name in (
        *G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS,
        *G2_DIAGNOSTIC_FILTERED_CONTACT_SENSORS,
    ):
        filtered = name in (
            *G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS,
            *G2_DIAGNOSTIC_FILTERED_CONTACT_SENSORS,
        )
        data = SimpleNamespace(
            force_matrix_w=(
                torch.zeros(2, 1, len(G2_RIGHT_PAD_FILTER_TARGETS), 3)
                if filtered
                else None
            ),
            net_forces_w=(
                torch.zeros(2, 1, 3)
                if filtered
                else torch.zeros(2, 3, 3)
            ),
        )
        body_names = (
            (
                "gripper_r_base_link",
                "gripper_r_inner_link2",
                "gripper_r_outer_link3",
            )
            if name == "forbidden_right_gripper_nonpad_contact"
            else tuple(f"{name}_body_{index}" for index in range(3))
        )
        scene[name] = SimpleNamespace(data=data, body_names=body_names)
    if collision_sensor is not None:
        sensor = scene[collision_sensor]
        force = (
            sensor.data.force_matrix_w
            if collision_sensor in (
                *G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS,
                *G2_DIAGNOSTIC_FILTERED_CONTACT_SENSORS,
            )
            else sensor.data.net_forces_w
        )
        force[1, 0, 0] = 2.0
    return SimpleNamespace(scene=scene, num_envs=2, device=torch.device("cpu"))


def test_collision_authority_requires_every_finite_sensor_and_detects_per_env():
    env = _env(collision_sensor="forbidden_arm_contact")
    evaluator = G2ForbiddenCollisionEvaluator(
        env,
        runtime_schema_probe={
            "self_collision_enabled": True,
            "collision_shape_count": 12,
            "enabled_collision_shape_count": 12,
            "collision_candidate_body_count": 12,
            "collision_covered_body_count": 12,
            "collision_missing_body_names": (),
            "right_arm_collision_links_missing": (),
            "self_collision_filtered_pairs_valid": True,
            "self_collision_filtered_pair_count": 62,
        },
    )
    assert evaluator.attestation.live_full_body_forbidden_collision_authority
    torch.testing.assert_close(evaluator(), torch.tensor([False, True]))
    assert evaluator.sensor_peak_forces_n()["forbidden_arm_contact"] == [0.0, 2.0]


def test_filtered_pad_allows_object_but_rejects_table_and_residual():
    env = _env()
    sensor = env.scene["forbidden_right_inner_pad_contact"]
    # Filter zero is Object and is allowed; keep net consistent with it.
    sensor.data.force_matrix_w[0, 0, 0, 0] = 2.0
    sensor.data.net_forces_w[0, 0, 0] = 2.0
    # Filter one is Table and remains a forbidden named pair.
    sensor.data.force_matrix_w[1, 0, 1, 1] = 3.0
    sensor.data.net_forces_w[1, 0, 1] = 3.0
    evaluator = G2ForbiddenCollisionEvaluator(
        env,
        runtime_schema_probe={
            "self_collision_enabled": True,
            "collision_shape_count": 12,
            "enabled_collision_shape_count": 12,
            "collision_candidate_body_count": 12,
            "collision_covered_body_count": 12,
            "collision_missing_body_names": (),
            "right_arm_collision_links_missing": (),
            "self_collision_filtered_pairs_valid": True,
            "self_collision_filtered_pair_count": 62,
        },
    )

    torch.testing.assert_close(evaluator(), torch.tensor([False, True]))
    components = evaluator.filtered_pad_force_components_n()[
        "forbidden_right_inner_pad_contact"
    ]
    assert components["filter_target_order"] == ["Object", "Table"]
    assert components["object_magnitude_n"] == [2.0, 0.0]
    assert components["table_magnitude_n"] == [0.0, 3.0]
    assert components["other_residual_magnitude_n"] == [0.0, 0.0]


def test_filtered_pad_rejects_and_attributes_unfiltered_other_residual():
    env = _env()
    sensor = env.scene["forbidden_right_outer_pad_contact"]
    sensor.data.net_forces_w[1, 0, 2] = 4.0
    evaluator = G2ForbiddenCollisionEvaluator(
        env,
        runtime_schema_probe={
            "self_collision_enabled": True,
            "collision_shape_count": 12,
            "enabled_collision_shape_count": 12,
            "collision_candidate_body_count": 12,
            "collision_covered_body_count": 12,
            "collision_missing_body_names": (),
            "right_arm_collision_links_missing": (),
            "self_collision_filtered_pairs_valid": True,
            "self_collision_filtered_pair_count": 62,
        },
    )

    torch.testing.assert_close(evaluator(), torch.tensor([False, True]))
    components = evaluator.filtered_pad_force_components_n()[
        "forbidden_right_outer_pad_contact"
    ]
    assert components["table_magnitude_n"] == [0.0, 0.0]
    assert components["other_residual_magnitude_n"] == [0.0, 4.0]


def test_outer_link2_allows_object_but_rejects_table_contact():
    env = _env()
    diagnostic = env.scene["diagnostic_right_outer_link2_contact"]
    # The rubber-coated link's Object contact is intentional and allowed.
    diagnostic.data.force_matrix_w[0, 0, 0] = torch.tensor([3.0, 4.0, 0.0])
    diagnostic.data.net_forces_w[0, 0] = torch.tensor([3.0, 4.0, 0.0])
    # The same rigid body touching Table remains forbidden.
    diagnostic.data.force_matrix_w[1, 0, 1] = torch.tensor([0.0, 0.0, 2.0])
    diagnostic.data.net_forces_w[1, 0] = torch.tensor([0.0, 0.0, 2.0])
    evaluator = G2ForbiddenCollisionEvaluator(
        env,
        runtime_schema_probe={
            "self_collision_enabled": True,
            "collision_shape_count": 12,
            "enabled_collision_shape_count": 12,
            "collision_candidate_body_count": 12,
            "collision_covered_body_count": 12,
            "collision_missing_body_names": (),
            "right_arm_collision_links_missing": (),
            "self_collision_filtered_pairs_valid": True,
            "self_collision_filtered_pair_count": 62,
        },
    )

    torch.testing.assert_close(evaluator(), torch.tensor([False, True]))
    assert "diagnostic_right_outer_link2_contact" in (
        evaluator.attestation.required_sensor_names
    )
    components = evaluator.diagnostic_contact_force_components_n()[
        "diagnostic_right_outer_link2_contact"
    ]
    assert components["subject_body"] == "gripper_r_outer_link2"
    assert components["filter_target_order"] == ["Object", "Table"]
    assert components["safety_authority"] is True
    assert components["object_contact_allowed"] is True
    assert components["object_magnitude_n"] == [5.0, 0.0]
    assert components["table_magnitude_n"] == [0.0, 2.0]


def test_remaining_nonpad_links_stay_forbidden():
    env = _env()
    nonpad = env.scene["forbidden_right_gripper_nonpad_contact"]
    assert "gripper_r_outer_link2" not in nonpad.body_names
    outer_link3_index = nonpad.body_names.index("gripper_r_outer_link3")
    nonpad.data.net_forces_w[1, outer_link3_index, 0] = 0.25
    evaluator = G2ForbiddenCollisionEvaluator(
        env,
        runtime_schema_probe={
            "self_collision_enabled": True,
            "collision_shape_count": 12,
            "enabled_collision_shape_count": 12,
            "collision_candidate_body_count": 12,
            "collision_covered_body_count": 12,
            "collision_missing_body_names": (),
            "right_arm_collision_links_missing": (),
            "self_collision_filtered_pairs_valid": True,
            "self_collision_filtered_pair_count": 62,
        },
    )

    torch.testing.assert_close(evaluator(), torch.tensor([False, True]))
    peaks = evaluator.sensor_body_peak_forces_n()[
        "forbidden_right_gripper_nonpad_contact"
    ]
    assert peaks["gripper_r_outer_link3"] == [0.0, 0.25]


def test_sensor_partition_keeps_outer_link2_out_of_broad_nonpad_regex(monkeypatch):
    isaaclab_module = ModuleType("isaaclab")
    sensors_module = ModuleType("isaaclab.sensors")

    def _contact_sensor_cfg(**kwargs):
        return SimpleNamespace(**kwargs)

    sensors_module.ContactSensorCfg = _contact_sensor_cfg
    isaaclab_module.sensors = sensors_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors_module)
    scene = SimpleNamespace()

    install_g2_forbidden_collision_sensors(scene)

    broad = scene.forbidden_right_gripper_nonpad_contact.prim_path
    assert "outer_link[13]" in broad
    assert "outer_link[123]" not in broad
    outer_link2 = scene.diagnostic_right_outer_link2_contact
    assert outer_link2.prim_path.endswith("/Robot/gripper_r_outer_link2")
    assert outer_link2.filter_prim_paths_expr == [
        "{ENV_REGEX_NS}/Object",
        "{ENV_REGEX_NS}/Table",
    ]


def test_outer_link2_forbidden_residual_caches_attribution_before_reset_mutation():
    env = _env()
    diagnostic = env.scene["diagnostic_right_outer_link2_contact"]
    diagnostic.data.force_matrix_w[1, 0, 0] = torch.tensor([3.0, 4.0, 0.0])
    diagnostic.data.force_matrix_w[1, 0, 1] = torch.tensor([0.0, 0.0, 2.0])
    # Include an unfiltered residual of [1,0,0].
    diagnostic.data.net_forces_w[1, 0] = torch.tensor([4.0, 4.0, 2.0])
    evaluator = G2ForbiddenCollisionEvaluator(
        env,
        runtime_schema_probe={
            "self_collision_enabled": True,
            "collision_shape_count": 12,
            "enabled_collision_shape_count": 12,
            "collision_candidate_body_count": 12,
            "collision_covered_body_count": 12,
            "collision_missing_body_names": (),
            "right_arm_collision_links_missing": (),
            "self_collision_filtered_pairs_valid": True,
            "self_collision_filtered_pair_count": 62,
        },
    )

    torch.testing.assert_close(evaluator(), torch.tensor([False, True]))
    # Simulate ManagerBased auto-reset reusing and zeroing the live buffers.
    diagnostic.data.force_matrix_w.zero_()
    diagnostic.data.net_forces_w.zero_()
    components = evaluator.diagnostic_contact_force_components_n()[
        "diagnostic_right_outer_link2_contact"
    ]
    assert components["object_vector_n"][1] == [3.0, 4.0, 0.0]
    assert components["object_magnitude_n"] == [0.0, 5.0]
    assert components["table_vector_n"][1] == [0.0, 0.0, 2.0]
    assert components["table_magnitude_n"] == [0.0, 2.0]
    assert components["other_residual_vector_n"][1] == [1.0, 0.0, 0.0]
    assert components["other_residual_magnitude_n"] == [0.0, 1.0]
    assert components["safety_authority"] is True
    assert components["object_contact_allowed"] is True


def test_collision_authority_fails_closed_on_missing_or_nonfinite_sensor():
    env = _env()
    del env.scene[G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS[0]]
    with pytest.raises(RuntimeError, match="SENSOR_MISSING"):
        G2ForbiddenCollisionEvaluator(
            env,
            runtime_schema_probe={
                "self_collision_enabled": True,
                "collision_shape_count": 12,
                "enabled_collision_shape_count": 12,
                "collision_candidate_body_count": 12,
                "collision_covered_body_count": 12,
                "collision_missing_body_names": (),
                "right_arm_collision_links_missing": (),
                "self_collision_filtered_pairs_valid": True,
                "self_collision_filtered_pair_count": 62,
            },
        )
    env = _env()
    env.scene["forbidden_head_contact"].data.net_forces_w[0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="NOT_LIVE"):
        G2ForbiddenCollisionEvaluator(
            env,
            runtime_schema_probe={
                "self_collision_enabled": True,
                "collision_shape_count": 12,
                "enabled_collision_shape_count": 12,
                "collision_candidate_body_count": 12,
                "collision_covered_body_count": 12,
                "collision_missing_body_names": (),
                "right_arm_collision_links_missing": (),
                "self_collision_filtered_pairs_valid": True,
                "self_collision_filtered_pair_count": 62,
            },
        )


def test_collision_authority_rejects_unapplied_runtime_schema_override():
    with pytest.raises(RuntimeError, match="NOT_LIVE"):
        G2ForbiddenCollisionEvaluator(
            _env(),
            runtime_schema_probe={
                "self_collision_enabled": False,
                "collision_shape_count": 12,
                "enabled_collision_shape_count": 5,
                "collision_candidate_body_count": 12,
                "collision_covered_body_count": 5,
                "collision_missing_body_names": ("arm_r_link2",),
                "right_arm_collision_links_missing": ("arm_r_link2",),
                "self_collision_filtered_pairs_valid": False,
                "self_collision_filtered_pair_count": 62,
            },
        )
