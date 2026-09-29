# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Source-owned OmniPicker product and force-observation contract.

This module deliberately separates three quantities that were previously
conflated:

* the OEM whole-gripper maximum gripping-force specification;
* motor torque/force-command feedback exposed by the OmniPicker protocol; and
* Isaac contact-sensor forces used as privileged simulation evidence.

Only the first quantity is a product limit.  The latter two do not become
pad-touch measurements merely because their unit or field name contains
``force``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


OMNIPICKER_MANUAL_URL = "https://www.agibot.com/filepage/265.html"


class OmniPickerProductContractError(ValueError):
    """Raised when a force value is presented with the wrong authority."""


@dataclass(frozen=True)
class OmniPickerProductContract:
    model: str = "OmniPicker"
    manual_hardware_version: str = "1.2"
    maximum_gripping_force_n: float = 30.0
    maximum_recommended_payload_kg: float = 1.5
    maximum_stroke_m: float = 0.120
    typical_open_close_time_s: float = 0.7
    active_dof: int = 1
    physical_pad_touch_sensor_available: bool = False
    physical_touch_sensor_authority: str = (
        "USER_CONFIRMED_DEVICE;OEM_MANUAL_EXPOSES_MOTOR_TORQUE_NOT_PAD_TOUCH"
    )
    protocol_force_feedback_semantics: str = "CURRENT_MOTOR_TORQUE_NORMALIZED_0_TO_FF"
    simulator_contact_role: str = "PRIVILEGED_TEACHER_LABEL_AND_M2_DIAGNOSTIC_ONLY"
    student_contact_input_count: int = 0
    legacy_telemetry_ceiling_n: float = 500.0
    legacy_telemetry_ceiling_role: str = (
        "NUMERIC_TELEMETRY_SANITY_ONLY_NOT_OMNIPICKER_PRODUCT_LIMIT"
    )
    m2_gripping_force_metric: str = "whole_gripper_gripping_force_estimate_n"
    sim_contact_to_oem_gripping_force_mapping: str = "UNRESOLVED"

    def __post_init__(self) -> None:
        positive = (
            self.maximum_gripping_force_n,
            self.maximum_recommended_payload_kg,
            self.maximum_stroke_m,
            self.typical_open_close_time_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise OmniPickerProductContractError("OmniPicker product values must be positive")
        if self.active_dof != 1:
            raise OmniPickerProductContractError("OmniPicker must remain one active DOF")
        if self.physical_pad_touch_sensor_available:
            raise OmniPickerProductContractError(
                "current physical OmniPicker contract has no pad touch sensor"
            )
        if self.student_contact_input_count != 0:
            raise OmniPickerProductContractError(
                "privileged simulator contact must not enter the student"
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manual_url"] = OMNIPICKER_MANUAL_URL
        payload["m2_force_gate_status"] = (
            "BLOCKED_UNTIL_SIM_CONTACT_TO_WHOLE_GRIPPER_FORCE_MAPPING_IS_VALIDATED"
            if self.sim_contact_to_oem_gripping_force_mapping == "UNRESOLVED"
            else "READY"
        )
        return payload

    def validate_m2_gripping_force(
        self,
        value_n: float,
        *,
        metric: str,
        mapping_validated: bool,
    ) -> float:
        """Validate an OEM-comparable whole-gripper force measurement.

        Raw contact-point, per-pad, summed-manifold, projected-effort, and
        motor-torque values are rejected.  Converting any of those into the
        OEM metric requires a separately validated mapping.
        """

        if metric != self.m2_gripping_force_metric:
            raise OmniPickerProductContractError("M2_FORCE_METRIC_NOT_OEM_COMPARABLE")
        if not mapping_validated:
            raise OmniPickerProductContractError("M2_FORCE_MAPPING_UNVALIDATED")
        force = float(value_n)
        if not math.isfinite(force) or force < 0.0:
            raise OmniPickerProductContractError("M2_GRIPPING_FORCE_INVALID")
        if force > self.maximum_gripping_force_n:
            raise OmniPickerProductContractError("M2_GRIPPING_FORCE_EXCEEDS_30N")
        return force


OMNIPICKER_PRODUCT_CONTRACT = OmniPickerProductContract()


__all__ = [
    "OMNIPICKER_MANUAL_URL",
    "OMNIPICKER_PRODUCT_CONTRACT",
    "OmniPickerProductContract",
    "OmniPickerProductContractError",
]
