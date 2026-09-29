# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""P0-only runtime binding metadata for the high-level G2 policy branch.

This module deliberately contains no Isaac, USD, ROS, articulation, or
low-level actuator import.  It describes the *meaning* of data transferred
between a bounded live P0 adapter and the offline policy contract.  The live
adapter remains responsible for obtaining controller-level acknowledgements
and sensor data from an already-approved high-level controller surface.

In particular, it does not make a claim about an OmniPicker joint, passive
link, motor, torque, current, CAN interface, or H4.7 mechanical authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

from .action_interface import HighLevelPolicyAction, PolicyActionMode
from .observation import PolicyDataSemantics, PolicyObservation


P0_RUNTIME_OBSERVATION_BINDING_SCHEMA = "g2_policy_branch_p0_runtime_binding_v2"
P0_PREVIOUS_ACTION_SEMANTICS = "last_accepted_branch_policy_action_lag1_v1"
P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS = (
    "existing_high_level_controller_command_state_open0_closed1_v1"
)
# Compatibility import only.  New bindings must use the explicitly named
# command-state field below; this alias must never be read as physical
# actuation, contact, or finger-position feedback.
P0_ABSTRACT_GRIPPER_FEEDBACK_SEMANTICS = P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS
P0_CAMERA_TIMESTAMP_SEQUENCE_SEMANTICS = (
    "episode_local_capture_time_strict_sequence_monotonic_v1"
)
P1_COLLECTION_RUNTIME_SEMANTIC_CONTRACT_SCHEMA = (
    "g2_policy_branch_p1_collection_runtime_semantic_contract_v1"
)


class P0RuntimeContractError(ValueError):
    """Raised when a P0 runtime binding would change policy-data meaning."""


def _require_sha256_fingerprint(name: str, value: str) -> None:
    """Reject an absent, malformed, or non-canonical external fingerprint."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise P0RuntimeContractError(
            f"{name} must be a lowercase SHA-256 hexadecimal fingerprint"
        )


@dataclass(frozen=True)
class WristCameraRuntimeContract:
    """Runtime-only wrist camera facts that must bind collection to P1/P2.

    Resolution and cadence are not actor tensors themselves, but changing them
    changes the meaning/distribution of RGB-D observations.  They are therefore
    hashed together with :class:`PolicyDataSemantics` for P0 attestation.
    """

    width_px: int
    height_px: int
    capture_interval_physics_steps: int
    maximum_frame_age_s: float
    timestamp_sequence_semantics: str = P0_CAMERA_TIMESTAMP_SEQUENCE_SEMANTICS
    rgb_dtype: str = "uint8"
    depth_dtype: str = "float32"
    depth_valid_dtype: str = "bool"
    depth_unit: str = "m"

    def __post_init__(self) -> None:
        if type(self.width_px) is not int or self.width_px <= 0:
            raise P0RuntimeContractError("wrist camera width must be positive int")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise P0RuntimeContractError("wrist camera height must be positive int")
        if (
            type(self.capture_interval_physics_steps) is not int
            or self.capture_interval_physics_steps <= 0
        ):
            raise P0RuntimeContractError(
                "wrist camera capture interval must be a positive integer"
            )
        if (
            not math.isfinite(self.maximum_frame_age_s)
            or self.maximum_frame_age_s < 0.0
        ):
            raise P0RuntimeContractError("maximum wrist frame age must be finite >= 0")
        if self.timestamp_sequence_semantics != P0_CAMERA_TIMESTAMP_SEQUENCE_SEMANTICS:
            raise P0RuntimeContractError("unsupported wrist timestamp/sequence contract")
        if self.rgb_dtype != "uint8" or self.depth_dtype != "float32":
            raise P0RuntimeContractError("P0 requires uint8 RGB and float32 depth")
        if self.depth_valid_dtype != "bool" or self.depth_unit != "m":
            raise P0RuntimeContractError("P0 requires boolean metric-depth validity")

    def payload(self) -> dict[str, object]:
        return {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "capture_interval_physics_steps": self.capture_interval_physics_steps,
            "maximum_frame_age_s": self.maximum_frame_age_s,
            "timestamp_sequence_semantics": self.timestamp_sequence_semantics,
            "rgb_dtype": self.rgb_dtype,
            "depth_dtype": self.depth_dtype,
            "depth_valid_dtype": self.depth_valid_dtype,
            "depth_unit": self.depth_unit,
        }


@dataclass(frozen=True)
class P0RuntimeObservationBinding:
    """Hashable bridge between offline policy semantics and a live P0 source.

    This is metadata only; no timestamp, camera pose, ground-truth object
    state, contact, joint target, or actuator quantity is added to the actor
    input.  ``current_gripper_state`` is specifically the existing
    high-level controller *command state* encoded ``OPEN=0`` / ``CLOSED=1``.
    It is not a claim that a physical finger, passive linkage, contact sensor,
    or actuator has reached that state.
    """

    data_semantics: PolicyDataSemantics
    wrist_camera: WristCameraRuntimeContract
    previous_action_semantics: str = P0_PREVIOUS_ACTION_SEMANTICS
    gripper_command_state_semantics: str = P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS
    schema: str = P0_RUNTIME_OBSERVATION_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.data_semantics, PolicyDataSemantics):
            raise P0RuntimeContractError("P0 binding needs PolicyDataSemantics")
        if not isinstance(self.wrist_camera, WristCameraRuntimeContract):
            raise P0RuntimeContractError("P0 binding needs a wrist camera contract")
        if self.previous_action_semantics != P0_PREVIOUS_ACTION_SEMANTICS:
            raise P0RuntimeContractError("unsupported previous-action authority")
        if (
            self.gripper_command_state_semantics
            != P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS
        ):
            raise P0RuntimeContractError(
                "unsupported abstract gripper command-state authority"
            )
        if self.schema != P0_RUNTIME_OBSERVATION_BINDING_SCHEMA:
            raise P0RuntimeContractError("unsupported P0 runtime binding schema")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_data_semantic_fingerprint": self.data_semantics.fingerprint(),
            "wrist_camera": self.wrist_camera.payload(),
            "previous_action_semantics": self.previous_action_semantics,
            "gripper_command_state_semantics": self.gripper_command_state_semantics,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_observation(self, observation: PolicyObservation) -> None:
        """Validate a B=1,T=1 live P0 observation without hidden conversion."""

        observation.assert_compatible_data_semantics(self.data_semantics)
        observation.validate(self.data_semantics.observation_config)
        rgb = observation.right_wrist_rgb
        expected = (1, 1, self.wrist_camera.height_px, self.wrist_camera.width_px, 3)
        if tuple(rgb.shape) != expected:
            raise P0RuntimeContractError(
                f"P0 wrist RGB shape mismatch: got {tuple(rgb.shape)}, expected {expected}"
            )
        if str(rgb.dtype).removeprefix("torch.") != self.wrist_camera.rgb_dtype:
            raise P0RuntimeContractError("P0 wrist RGB dtype mismatch")
        depth = observation.right_wrist_depth_m
        if tuple(depth.shape) != (*expected[:-1], 1):
            raise P0RuntimeContractError("P0 wrist depth shape mismatch")
        if str(depth.dtype).removeprefix("torch.") != self.wrist_camera.depth_dtype:
            raise P0RuntimeContractError("P0 wrist depth dtype mismatch")
        valid = observation.right_wrist_depth_valid
        if str(valid.dtype).removeprefix("torch.") != self.wrist_camera.depth_valid_dtype:
            raise P0RuntimeContractError("P0 wrist depth-valid dtype mismatch")
        if observation.right_wrist_frame_age_s is None:
            raise P0RuntimeContractError("P0 wrist frame age is required")
        if bool((observation.right_wrist_frame_age_s > self.wrist_camera.maximum_frame_age_s).any()):
            raise P0RuntimeContractError("P0 wrist frame age exceeds bound")
        if observation.hidden_reset_mask is None:
            raise P0RuntimeContractError("P0 hidden reset mask is required")


@dataclass(frozen=True)
class P1CollectionRuntimeSemanticContract:
    """Immutable external semantic receipt required before P1 collection.

    A P1 harness must load this receipt from its external P0 evidence, then
    compare it against both its local :class:`PolicyDataSemantics` and the
    complete P0 runtime observation binding.  Storing only the base data
    fingerprint would allow a collection process to silently change camera
    cadence/resolution, frame-age semantics, previous-action provenance, or
    the abstract gripper command-state meaning.

    This object deliberately does *not* create a dataset or encode an action,
    controller target, physical gripper measurement, joint target, or any
    actuator quantity.
    """

    data_semantic_fingerprint: str
    p0_runtime_observation_binding_schema: str
    p0_runtime_observation_binding_fingerprint: str
    schema: str = P1_COLLECTION_RUNTIME_SEMANTIC_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != P1_COLLECTION_RUNTIME_SEMANTIC_CONTRACT_SCHEMA:
            raise P0RuntimeContractError("unsupported P1 collection runtime semantic schema")
        _require_sha256_fingerprint(
            "P1 policy-data semantic fingerprint", self.data_semantic_fingerprint
        )
        if self.p0_runtime_observation_binding_schema != P0_RUNTIME_OBSERVATION_BINDING_SCHEMA:
            raise P0RuntimeContractError("unsupported P0 runtime binding schema in P1 receipt")
        _require_sha256_fingerprint(
            "P1 P0 runtime binding fingerprint",
            self.p0_runtime_observation_binding_fingerprint,
        )

    @classmethod
    def from_p0_binding(
        cls, binding: P0RuntimeObservationBinding
    ) -> "P1CollectionRuntimeSemanticContract":
        """Create the only valid P1 receipt from a validated P0 binding."""

        if not isinstance(binding, P0RuntimeObservationBinding):
            raise P0RuntimeContractError("P1 receipt requires P0RuntimeObservationBinding")
        return cls(
            data_semantic_fingerprint=binding.data_semantics.fingerprint(),
            p0_runtime_observation_binding_schema=binding.schema,
            p0_runtime_observation_binding_fingerprint=binding.fingerprint(),
        )

    def payload(self) -> dict[str, str]:
        """Return the exact serializable receipt payload; no optional fields."""

        return {
            "schema": self.schema,
            "data_semantic_fingerprint": self.data_semantic_fingerprint,
            "p0_runtime_observation_binding_schema": (
                self.p0_runtime_observation_binding_schema
            ),
            "p0_runtime_observation_binding_fingerprint": (
                self.p0_runtime_observation_binding_fingerprint
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "P1CollectionRuntimeSemanticContract":
        """Strictly load an external receipt without defaulting missing data."""

        if not isinstance(payload, Mapping):
            raise P0RuntimeContractError("P1 external semantic receipt must be a mapping")
        required = {
            "schema",
            "data_semantic_fingerprint",
            "p0_runtime_observation_binding_schema",
            "p0_runtime_observation_binding_fingerprint",
        }
        if set(payload) != required:
            raise P0RuntimeContractError(
                "P1 external semantic receipt must have exactly the required fields"
            )
        values = {key: payload[key] for key in required}
        if not all(isinstance(value, str) for value in values.values()):
            raise P0RuntimeContractError("P1 external semantic receipt values must be strings")
        return cls(
            schema=values["schema"],
            data_semantic_fingerprint=values["data_semantic_fingerprint"],
            p0_runtime_observation_binding_schema=(
                values["p0_runtime_observation_binding_schema"]
            ),
            p0_runtime_observation_binding_fingerprint=(
                values["p0_runtime_observation_binding_fingerprint"]
            ),
        )

    @classmethod
    def from_json(cls, serialized: str | bytes | bytearray) -> "P1CollectionRuntimeSemanticContract":
        """Load JSON evidence; malformed or non-object data is fail-closed."""

        if isinstance(serialized, bytearray):
            serialized = bytes(serialized)
        if isinstance(serialized, bytes):
            try:
                serialized = serialized.decode("utf-8")
            except UnicodeDecodeError as error:
                raise P0RuntimeContractError("P1 external semantic receipt is not UTF-8") from error
        if not isinstance(serialized, str):
            raise P0RuntimeContractError("P1 external semantic receipt must be JSON text")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise P0RuntimeContractError("P1 external semantic receipt is invalid JSON") from error
        return cls.from_payload(payload)

    def assert_matches(
        self,
        *,
        data_semantics: PolicyDataSemantics,
        p0_runtime_observation_binding: P0RuntimeObservationBinding,
    ) -> None:
        """Fail closed unless both independent semantic fingerprints match."""

        if not isinstance(data_semantics, PolicyDataSemantics):
            raise P0RuntimeContractError("P1 requires local PolicyDataSemantics")
        if not isinstance(p0_runtime_observation_binding, P0RuntimeObservationBinding):
            raise P0RuntimeContractError("P1 requires local P0RuntimeObservationBinding")
        if (
            p0_runtime_observation_binding.data_semantics.fingerprint()
            != data_semantics.fingerprint()
        ):
            raise P0RuntimeContractError(
                "local P0 binding and local P1 data semantics do not agree"
            )
        if self.data_semantic_fingerprint != data_semantics.fingerprint():
            raise P0RuntimeContractError(
                "P1 external receipt data semantic fingerprint does not match local semantics"
            )
        if (
            self.p0_runtime_observation_binding_schema
            != p0_runtime_observation_binding.schema
            or self.p0_runtime_observation_binding_fingerprint
            != p0_runtime_observation_binding.fingerprint()
        ):
            raise P0RuntimeContractError(
                "P1 external receipt P0 runtime binding does not match local binding"
            )


def load_p1_collection_runtime_semantic_contract(
    external_contract: Mapping[str, object] | str | bytes | bytearray,
    *,
    data_semantics: PolicyDataSemantics,
    p0_runtime_observation_binding: P0RuntimeObservationBinding,
) -> P1CollectionRuntimeSemanticContract:
    """Strictly load and validate the required external P0-to-P1 receipt.

    ``None``, an in-memory shortcut object, missing fields, an invalid schema,
    or either fingerprint mismatch raises :class:`P0RuntimeContractError`.
    A future P1 harness can therefore invoke this before opening any collector
    or dataset writer and remain fail-closed.
    """

    if isinstance(external_contract, Mapping):
        receipt = P1CollectionRuntimeSemanticContract.from_payload(external_contract)
    else:
        receipt = P1CollectionRuntimeSemanticContract.from_json(external_contract)
    receipt.assert_matches(
        data_semantics=data_semantics,
        p0_runtime_observation_binding=p0_runtime_observation_binding,
    )
    return receipt


class PreviousAcceptedPolicyAction:
    """P0-local lag-one branch action ring.

    The actor sees the last fully accepted **branch action**, not a projected
    controller target or an actuator command.  Rejected actions are separately
    logged by the P0 harness and never overwrite this state.
    """

    def __init__(
        self,
        *,
        action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    ) -> None:
        if not isinstance(action_mode, PolicyActionMode):
            raise P0RuntimeContractError("previous-action ring needs PolicyActionMode")
        self._action_mode = action_mode
        self._action = self._zero_action()

    def _zero_action(self) -> HighLevelPolicyAction:
        return HighLevelPolicyAction.from_sequence(
            [0.0] * (self._action_mode.dimension - 1) + [0.0],
            mode=self._action_mode,
        )

    @property
    def action_mode(self) -> PolicyActionMode:
        return self._action_mode

    @property
    def value(self) -> HighLevelPolicyAction:
        return self._action

    def reset(self) -> HighLevelPolicyAction:
        self._action = self._zero_action()
        return self._action

    def record_accepted(self, action: HighLevelPolicyAction) -> HighLevelPolicyAction:
        if not isinstance(action, HighLevelPolicyAction) or action.mode is not self._action_mode:
            raise P0RuntimeContractError("accepted previous action has incompatible schema")
        self._action = action
        return self._action


__all__ = [
    "P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS",
    "P0_ABSTRACT_GRIPPER_FEEDBACK_SEMANTICS",
    "P0_CAMERA_TIMESTAMP_SEQUENCE_SEMANTICS",
    "P0_PREVIOUS_ACTION_SEMANTICS",
    "P0_RUNTIME_OBSERVATION_BINDING_SCHEMA",
    "P1_COLLECTION_RUNTIME_SEMANTIC_CONTRACT_SCHEMA",
    "P1CollectionRuntimeSemanticContract",
    "P0RuntimeContractError",
    "P0RuntimeObservationBinding",
    "PreviousAcceptedPolicyAction",
    "WristCameraRuntimeContract",
    "load_p1_collection_runtime_semantic_contract",
]
