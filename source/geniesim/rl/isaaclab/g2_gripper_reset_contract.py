# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared, fail-closed G2 OmniPicker reset authority.

The articulation is always spawned/reset from one complete 46-DoF *seed*.
That seed is not claimed to be dynamically settled on a particular PhysX
build.  Only the independent right outer master is then driven to OPEN through
the ordinary bounded action term.  Once the full articulation has supplied at
least two real position intervals and five consecutive safe, settled samples,
all eight *measured* right-hand coordinates may be cached as articulation reset
state.  Passive/mimic coordinates are never independent drive targets.

This module contains no Isaac/Kit imports.  Keyboard collection, privileged
Teacher and visual Student runtimes consume the same names, cache semantics
and full-articulation finite-difference safety audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .g2_lift_methodology import stable_original_home
from .g2_rebuild.data_contract import G2_RUNTIME_JOINT_ORDER


G2_GRIPPER_RESET_CONTRACT_SCHEMA = "g2_constraint_stable_gripper_reset_v2"

G2_RIGHT_HAND_JOINT_NAMES = tuple(
    name for name in G2_RUNTIME_JOINT_ORDER if "gripper_r_" in name
)
G2_LEFT_HAND_JOINT_NAMES = tuple(
    name for name in G2_RUNTIME_JOINT_ORDER if "gripper_l_" in name
)
G2_RIGHT_GRIPPER_MASTER = "idx81_gripper_r_outer_joint1"
G2_LEFT_GRIPPER_MASTER = "idx41_gripper_l_outer_joint1"
G2_RIGHT_HAND_INDEPENDENT_TARGET_JOINT_NAMES = (G2_RIGHT_GRIPPER_MASTER,)
G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES = tuple(
    name
    for name in G2_RIGHT_HAND_JOINT_NAMES
    if name not in G2_RIGHT_HAND_INDEPENDENT_TARGET_JOINT_NAMES
)
G2_LEFT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES = tuple(
    name for name in G2_LEFT_HAND_JOINT_NAMES if name != G2_LEFT_GRIPPER_MASTER
)
G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES = (
    *G2_LEFT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
    *G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
)
G2_LEFT_HAND_RESET_POLICY = "CONFIGURED_SEED_HELD_NO_OPEN_COMMAND"
G2_RIGHT_HAND_RESET_POLICY = (
    "CONFIGURED_SEED_THEN_BOUNDED_MASTER_OPEN_THEN_SETTLED_MEASURED_FULL8_CACHE"
)
G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S = 0.8
G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2 = 10.0
# This is a cache-settle criterion, not a relaxation or replacement of the
# 0.8-rad/s hard motion limit.  It reuses the project's audited
# pre-continuation stabilization threshold and prevents a merely safe but
# still-moving passive chain from becoming reset authority.
G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S = 0.05
G2_RIGHT_MASTER_OPEN_TARGET_RAD = math.pi / 4.0
G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS = 60
# This is a fail-closed reset-lifecycle observation budget, not a controller
# target, tolerance, or speed-cap change.  A clean startup OPEN reached its
# endpoint by step 178, but the reset contract must also cover a legitimate
# post-CLOSE measured four-bar state.  In r4, that state began near 0.20 rad;
# the unchanged passive-limit governor safely emitted only about 0.02--0.04
# rad/s and therefore reached merely 0.327 rad after 240 50-Hz packets.  The
# target remained 0.785398 rad and the path continued monotonically, so this
# is a bounded-travel observation issue rather than an endpoint, mechanics, or
# controller-tracking defect.  1600 packets (32 s at 50 Hz) retains the same
# 60-step minimum, 0.01-rad endpoint tolerance, 0.05-rad/s measured-settle
# criterion, and all governor mechanics, while covering a full measured OPEN
# recovery from a post-CLOSE state.  It still fails closed if that recovery
# does not complete.
G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS = 1600
G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD = 0.01
G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES = 5
# The passive four-bar is read after one physics interval and responds to a
# newly emitted master target over the following interval.  Reserve those two
# intervals when computing the passive-joint stopping distance.  This is a
# conservative retiming parameter, not a relaxation of either hard limit.
G2_PASSIVE_LIMIT_GUARD_REACTION_STEPS = 2
G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S = 1.0e-6
# The initial live diagnostic observed at most about 0.003276 rad/s of
# passive-chain noise before commanded motion.  Round that evidence upward to
# 0.005 rad/s for two-sample causal settling.  At the 2 ms physics interval,
# the unchanged 10 rad/s^2 hard acceleration limit permits at most 0.020
# rad/s to disappear at a hard stop.  Creep may therefore contribute only the
# remaining 0.015 rad/s; residual settle motion plus commanded creep stays at
# or below the existing hard limit.
G2_PASSIVE_LIMIT_GOVERNOR_SETTLE_SPEED_RAD_S = 0.005
G2_PASSIVE_LIMIT_GOVERNOR_STATIONARY_SAMPLES = 2
# The immutable landing6 trace established that a coordinate already latched as
# landed can be projected back 0.000143401 rad inside its authored limit, while
# the next active coordinate was 0.000154492 rad inside the opposite limit.
# Round the largest observed residual upward to a 0.0002-rad *hysteresis* band.
# This band only keeps an active coordinate under the CREEP envelope instead of
# releasing it into a full-cap POST_STOP cycle; it is never by itself a landing
# proof.
G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD = 2.0e-4
# Candidate-A reset-parity r5 found a distinct numeric deadlock: idx73 was
# physically stationary (|qd| <= 4.41e-4 rad/s) and only 0.66--3.90e-6 rad
# from its authored stop, yet the previous float32-epsilon (1.19e-7 rad)
# comparison could never mark it landed.  Keep the numeric landing allowance
# 50x tighter than the existing stop band.  It is not a hard-limit or motion
# threshold change: the existing two causal <=0.005-rad/s samples and all
# passive envelopes still gate landing; this merely admits the measured PhysX
# constraint-solver residual once the joint is demonstrably stationary.
G2_PASSIVE_LIMIT_GOVERNOR_NUMERIC_LANDING_TOLERANCE_RAD = 4.0e-6
# The governor reduces its command-speed cap at 2 rad/s^2.  This is a
# conservative retiming authority below the unchanged 10 rad/s^2 measured
# hard limit; it is not a new actuator gain or a relaxed safety threshold.
G2_PASSIVE_LIMIT_GOVERNOR_BRAKE_ACCELERATION_RAD_S2 = 2.0
# The second live clamp entered the terminal interval at 0.0420268625 rad/s,
# while the induced CREEP budget was 0.015 rad/s: 2.80179 times larger.  Round
# that observed mismatch upward once.  It applies only to the predicted CREEP
# branch; measured speed and the TRACK full-authority prediction remain raw.
# This changes only command retiming; it does not relax the 0.8/10 hard motion
# limits.
G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER = 3.0
G2_PASSIVE_LIMIT_GOVERNOR_TRACK = 0
G2_PASSIVE_LIMIT_GOVERNOR_BRAKE = 1
G2_PASSIVE_LIMIT_GOVERNOR_SETTLE = 2
G2_PASSIVE_LIMIT_GOVERNOR_CREEP = 3
G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP = 4
G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE = 5
# Backward-readable name for reports/tests written while this state was
# called APPROACH.  It is exactly the TRACK state, not a second state.
G2_PASSIVE_LIMIT_GOVERNOR_APPROACH = G2_PASSIVE_LIMIT_GOVERNOR_TRACK

# Offline, source-linked four-bar audit.  The composed USD authors every
# passive coordinate at zero, but its closed-loop spherical-joint anchors are
# not exactly coincident in the converted rest geometry.  The historical
# non-zero right-hand seed was read from PhysX after composition (before the
# old runtime wrote its home state); the same numeric pattern was later copied
# to the left hand in ``stable_original_home`` without an equivalent left-hand
# readback artifact.  The right seed reduces the URDF loop-anchor residual
# substantially but neither side is thereby proven to be in equilibrium on
# the current backend.
G2_FOUR_BAR_INITIAL_STATE_CONTRACT_SCHEMA = "g2_four_bar_initial_state_audit_v1"
G2_FOUR_BAR_ASSET_SHA256 = (
    "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
)
G2_FOUR_BAR_URDF_SHA256 = (
    "a5efa8eba603ee259ef502b2b371875764ecbf161776a46559fce654ada9af09"
)
G2_FOUR_BAR_HISTORICAL_READBACK_MANIFEST_SHA256 = (
    "084647f9bb28cfe18a2df85d818f89490f81d7972b47f1974cc87c89f9043f47"
)
G2_FOUR_BAR_USD_AUTHORED_PASSIVE_POSITION_RAD = 0.0
G2_FOUR_BAR_ASSET_SOLVER_POSITION_ITERATIONS = 32
G2_FOUR_BAR_ASSET_SOLVER_VELOCITY_ITERATIONS = 1
G2_FOUR_BAR_PAD_DIAGNOSTIC_POSITION_ITERATIONS = 8
G2_FOUR_BAR_PAD_DIAGNOSTIC_VELOCITY_ITERATIONS = 0
G2_FOUR_BAR_ZERO_SEED_LOOP_CLOSURE_MEAN_MM = 0.5717872437872544
G2_FOUR_BAR_CONFIGURED_SEED_LOOP_CLOSURE_MEAN_MM = 0.10683166071140864
G2_FOUR_BAR_CONFIGURED_SEED_RESIDUAL_REDUCTION_FRACTION = 0.813161867682


def four_bar_initial_state_contract_metadata() -> dict[str, object]:
    """Return source-linked facts; never promote them as live evidence."""

    return {
        "schema": G2_FOUR_BAR_INITIAL_STATE_CONTRACT_SCHEMA,
        "asset_sha256": G2_FOUR_BAR_ASSET_SHA256,
        "urdf_sha256": G2_FOUR_BAR_URDF_SHA256,
        "usd_authored_passive_position_rad": (
            G2_FOUR_BAR_USD_AUTHORED_PASSIVE_POSITION_RAD
        ),
        "configured_seed_provenance": (
            "RIGHT_HISTORICAL_READBACK_LEFT_CODE_MIRROR_UNATTESTED"
        ),
        "configured_seed_provenance_by_side": {
            "right": (
                "HISTORICAL_COMPOSED_PHYSX_INITIAL_READBACK_BEFORE_"
                "HOME_STATE_WRITE"
            ),
            "left": (
                "CODE_MIRRORED_FROM_RIGHT_PATTERN_WITHOUT_LEFT_LIVE_"
                "READBACK_EVIDENCE"
            ),
        },
        "historical_manifest_contains_right_hand_readback": True,
        "historical_manifest_contains_left_hand_readback": False,
        "historical_readback_manifest_sha256": (
            G2_FOUR_BAR_HISTORICAL_READBACK_MANIFEST_SHA256
        ),
        "historical_evidence_acceptance_eligible": False,
        "zero_seed_loop_closure_mean_mm": (
            G2_FOUR_BAR_ZERO_SEED_LOOP_CLOSURE_MEAN_MM
        ),
        "configured_seed_loop_closure_mean_mm": (
            G2_FOUR_BAR_CONFIGURED_SEED_LOOP_CLOSURE_MEAN_MM
        ),
        "configured_seed_residual_reduction_fraction": (
            G2_FOUR_BAR_CONFIGURED_SEED_RESIDUAL_REDUCTION_FRACTION
        ),
        "configured_seed_is_exact_closed_loop_solution": False,
        "configured_seed_is_live_equilibrium_evidence": False,
        "left_configured_seed_is_live_equilibrium_evidence": False,
        "asset_solver_position_iterations": (
            G2_FOUR_BAR_ASSET_SOLVER_POSITION_ITERATIONS
        ),
        "asset_solver_velocity_iterations": (
            G2_FOUR_BAR_ASSET_SOLVER_VELOCITY_ITERATIONS
        ),
        "pad_diagnostic_solver_position_iterations": (
            G2_FOUR_BAR_PAD_DIAGNOSTIC_POSITION_ITERATIONS
        ),
        "pad_diagnostic_solver_velocity_iterations": (
            G2_FOUR_BAR_PAD_DIAGNOSTIC_VELOCITY_ITERATIONS
        ),
        "pad_diagnostic_matches_asset_solver_contract": False,
        "solver_change_authorized_from_offline_audit": False,
        "required_solver_probe": [
            "PINNED_8_0_BASELINE",
            "PINNED_8_1_ISOLATE_VELOCITY_ITERATIONS",
            "ASSET_AUTHORED_32_1_FIDELITY",
        ],
        "live_q0_q1_q2_revalidation_required": True,
    }


def validate_complete_stable_hand_configuration(
    pose: dict[str, float],
) -> None:
    """Require the complete historical 46-DoF seed for both closed chains.

    The function name is retained for call-site compatibility.  Equality with
    this configured seed is not a live PhysX stability attestation.
    """

    missing = set(G2_RUNTIME_JOINT_ORDER).difference(pose)
    extra = set(pose).difference(G2_RUNTIME_JOINT_ORDER)
    if missing or extra:
        raise ValueError(
            "G2_RESET_POSE_NOT_COMPLETE_46DOF:"
            f"missing={sorted(missing)}:extra={sorted(extra)}"
        )
    stable = stable_original_home()
    for name in (*G2_LEFT_HAND_JOINT_NAMES, *G2_RIGHT_HAND_JOINT_NAMES):
        if not math.isclose(float(pose[name]), float(stable[name]), abs_tol=1.0e-12):
            raise ValueError(
                "G2_RESET_HAND_NOT_COMPLETE_CONFIGURED_SEED:"
                f"{name}:{pose[name]}:{stable[name]}"
            )


def resolve_full_right_hand_indices(joint_names: Sequence[str]) -> tuple[int, ...]:
    """Resolve the complete measured right hand once, by exact name."""

    if len(set(joint_names)) != len(joint_names):
        raise ValueError("G2_RUNTIME_JOINT_NAMES_NOT_BIJECTIVE")
    missing = set(G2_RIGHT_HAND_JOINT_NAMES).difference(joint_names)
    if missing:
        raise ValueError(f"G2_RIGHT_HAND_CACHE_JOINT_MISSING:{sorted(missing)}")
    indices = tuple(joint_names.index(name) for name in G2_RIGHT_HAND_JOINT_NAMES)
    if len(indices) != 8 or len(set(indices)) != 8:
        raise ValueError("G2_RIGHT_HAND_CACHE_NOT_FULL8")
    return indices


def acceleration_limited_position_target_step(
    previous_target,
    desired_target,
    previous_target_velocity,
    *,
    maximum_speed_rad_s,
    maximum_acceleration_rad_s2: float,
    dt_s: float,
):
    """Advance one continuous position-target sample on its existing device.

    The function intentionally performs tensor-only arithmetic.  It is shared
    by the normal Isaac action term and bounded diagnostics so a diagnostic
    cannot silently use a more permissive target generator than teleop.
    ``maximum_speed_rad_s`` may be a scalar or a tensor broadcastable to the
    target shape; the passive-limit guard supplies a per-environment tensor.
    """

    import torch

    if previous_target.shape != desired_target.shape:
        raise ValueError("G2_TARGET_STEP_POSITION_SHAPE_MISMATCH")
    if previous_target_velocity.shape != previous_target.shape:
        raise ValueError("G2_TARGET_STEP_VELOCITY_SHAPE_MISMATCH")
    if not math.isfinite(maximum_acceleration_rad_s2) or (
        maximum_acceleration_rad_s2 <= 0.0
    ):
        raise ValueError("G2_TARGET_STEP_ACCELERATION_INVALID")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("G2_TARGET_STEP_DT_INVALID")
    speed_limit = torch.as_tensor(
        maximum_speed_rad_s,
        dtype=previous_target.dtype,
        device=previous_target.device,
    ).clamp_min(0.0)
    dt = float(dt_s)
    acceleration = float(maximum_acceleration_rad_s2)
    maximum_velocity_delta = acceleration * dt

    # Select a speed that can take this sample *and* still stop before the
    # requested endpoint.  The extra ``v * dt`` term reserves the current
    # forward-Euler sample; ``v**2 / (2a)`` reserves subsequent braking.  Its
    # positive root is deliberately more conservative than a raw
    # ``sqrt(2*a*distance)`` cap, which can reach an endpoint with too much
    # velocity to emit the required zero-velocity hold on the next sample.
    error = desired_target - previous_target
    distance = torch.abs(error)
    direction = torch.sign(error)
    stopping_speed = torch.sqrt(
        maximum_velocity_delta * maximum_velocity_delta
        + 2.0 * acceleration * distance
    ) - maximum_velocity_delta
    desired_speed = torch.minimum(speed_limit, stopping_speed.clamp_min(0.0))

    # Express the prior velocity in the current endpoint direction, then
    # apply the unchanged acceleration bound.  Normal operation starts from
    # rest and therefore remains non-negative in this coordinate.  Clamping
    # the displacement to the remaining distance makes the emitted position
    # monotonic and prohibits endpoint overshoot.
    previous_speed_toward_target = previous_target_velocity * direction
    target_speed_toward_target = previous_speed_toward_target + torch.clamp(
        desired_speed - previous_speed_toward_target,
        -maximum_velocity_delta,
        maximum_velocity_delta,
    )
    target_speed_toward_target = target_speed_toward_target.clamp_min(0.0)
    target_delta = direction * torch.minimum(
        target_speed_toward_target * dt,
        distance,
    )
    target = previous_target + target_delta
    target_velocity = target_delta / dt
    return target, target_velocity


def _passive_limit_aware_master_speed_cap_per_passive(
    passive_position,
    passive_fd_velocity,
    passive_position_limits,
    current_master_target_velocity,
    desired_master_delta,
    *,
    maximum_master_speed_rad_s: float,
    maximum_master_target_acceleration_rad_s2: float,
    dt_s: float,
    reaction_steps: int,
):
    """Return the tensor cap for every passive coordinate.

    Inputs are already validated by the public scalar helper.  Keeping this
    implementation private preserves the existing public ``[N, 1]`` API while
    allowing the stateful governor to latch every coordinate whose *same cap
    equation* is restrictive.  This avoids a second, non-equivalent TTC
    authority and is permutation-equivariant across passive joint order.
    """

    import torch

    if passive_position_limits.ndim == 2:
        limits = passive_position_limits.unsqueeze(0)
    else:
        limits = passive_position_limits
    lower = limits[..., 0]
    upper = limits[..., 1]
    speed = torch.abs(passive_fd_velocity)
    margin = torch.where(
        passive_fd_velocity > 0.0,
        upper - passive_position,
        passive_position - lower,
    ).clamp_min(0.0)
    master_acceleration = float(maximum_master_target_acceleration_rad_s2)
    reaction_time = float(reaction_steps) * float(dt_s)
    master_speed = torch.abs(current_master_target_velocity)
    reaction_distance = speed * reaction_time
    master_braking_distance = 0.5 * speed * master_speed / master_acceleration
    required_margin = reaction_distance + master_braking_distance
    moving_toward_limit = (
        speed > G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S
    )
    braking_required = moving_toward_limit & (margin <= required_margin)
    remaining_braking_margin = (margin - reaction_distance).clamp_min(0.0)
    allowed_master_speed = (
        2.0
        * master_acceleration
        * remaining_braking_margin
        / torch.clamp_min(
            speed, G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S
        )
    )
    configured_cap = torch.full_like(
        allowed_master_speed, float(maximum_master_speed_rad_s)
    )
    per_passive_cap = torch.where(
        braking_required,
        torch.minimum(allowed_master_speed, configured_cap),
        configured_cap,
    )
    continuing_direction = (
        current_master_target_velocity * desired_master_delta >= 0.0
    )
    return torch.where(
        continuing_direction,
        per_passive_cap,
        configured_cap,
    ).clamp(0.0, float(maximum_master_speed_rad_s))


def passive_limit_aware_master_speed_cap(
    passive_position,
    passive_fd_velocity,
    passive_position_limits,
    current_master_target_velocity,
    desired_master_delta,
    *,
    maximum_master_speed_rad_s: float,
    maximum_master_target_acceleration_rad_s2: float,
    dt_s: float,
    reaction_steps: int = G2_PASSIVE_LIMIT_GUARD_REACTION_STEPS,
):
    """Return a per-environment master speed cap before a passive hard stop.

    A passive coordinate can reach its joint limit even when only the master
    is targeted.  The actuator can decelerate the *master target*, not the
    passive coordinate directly.  If a measured passive velocity is directed
    toward a limit, reserve the distance travelled during the response delay
    and while the master target decelerates to rest:

    ``margin >= v_passive * reaction_time``
    ``          + 0.5 * v_passive * abs(v_master) / a_master``

    Solving this inequality for master speed gives a per-passive-coordinate
    braking cap.  The minimum cap is handed to the existing master target
    acceleration limiter, which performs the continuous deceleration.
    Reversing the master direction is never blocked.  This helper changes
    neither the joint limit nor the 0.8/10 hard safety limits and does not
    target passive joints.

    All inputs and outputs remain tensors on the caller's device.  No per-step
    CPU materialization is performed.
    """

    import torch

    if passive_position.shape != passive_fd_velocity.shape:
        raise ValueError("G2_PASSIVE_GUARD_STATE_SHAPE_MISMATCH")
    if passive_position.ndim != 2:
        raise ValueError("G2_PASSIVE_GUARD_STATE_MUST_BE_BATCHED")
    if passive_position_limits.ndim == 2:
        if passive_position_limits.shape != (*passive_position.shape[1:], 2):
            raise ValueError("G2_PASSIVE_GUARD_LIMIT_SHAPE_MISMATCH")
        passive_position_limits = passive_position_limits.unsqueeze(0)
    if passive_position_limits.shape not in (
        (1, passive_position.shape[1], 2),
        (passive_position.shape[0], passive_position.shape[1], 2),
    ):
        raise ValueError("G2_PASSIVE_GUARD_LIMIT_SHAPE_MISMATCH")
    expected_master_shape = (passive_position.shape[0], 1)
    if current_master_target_velocity.shape != expected_master_shape:
        raise ValueError("G2_PASSIVE_GUARD_MASTER_VELOCITY_SHAPE_MISMATCH")
    if desired_master_delta.shape != expected_master_shape:
        raise ValueError("G2_PASSIVE_GUARD_MASTER_DELTA_SHAPE_MISMATCH")
    if not math.isfinite(maximum_master_speed_rad_s) or (
        maximum_master_speed_rad_s <= 0.0
    ):
        raise ValueError("G2_PASSIVE_GUARD_MASTER_SPEED_INVALID")
    if not math.isfinite(maximum_master_target_acceleration_rad_s2) or (
        maximum_master_target_acceleration_rad_s2 <= 0.0
    ):
        raise ValueError("G2_PASSIVE_GUARD_ACCELERATION_INVALID")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("G2_PASSIVE_GUARD_DT_INVALID")
    if int(reaction_steps) != reaction_steps or reaction_steps < 1:
        raise ValueError("G2_PASSIVE_GUARD_REACTION_STEPS_INVALID")

    per_passive_cap = _passive_limit_aware_master_speed_cap_per_passive(
        passive_position,
        passive_fd_velocity,
        passive_position_limits,
        current_master_target_velocity,
        desired_master_delta,
        maximum_master_speed_rad_s=maximum_master_speed_rad_s,
        maximum_master_target_acceleration_rad_s2=(
            maximum_master_target_acceleration_rad_s2
        ),
        dt_s=dt_s,
        reaction_steps=reaction_steps,
    )
    guarded_cap = torch.amin(per_passive_cap, dim=1, keepdim=True)
    configured_cap = torch.full_like(
        guarded_cap, float(maximum_master_speed_rad_s)
    )
    return torch.minimum(guarded_cap, configured_cap).clamp(
        0.0, float(maximum_master_speed_rad_s)
    )


class PassiveLimitMasterGovernor:
    """Causal TRACK/BRAKE/SETTLE/CREEP governor for one hand master.

    The stateless envelope detects when a passive four-bar coordinate needs
    braking, but handing a zero speed cap directly to the generic 10-rad/s^2
    limiter still produced a nonlinear passive response in live PhysX.  This
    state machine therefore ramps the *command speed cap* down at 2 rad/s^2,
    waits for two measured stationary samples, and crosses the final distance
    using only the remaining passive landing-speed budget.

    After the passive coordinate lands, POST_STOP remains latched for the same
    master direction.  This prevents measured stop-band noise from repeatedly
    entering BRAKE.  A direction reversal (including CLOSE after OPEN) or an
    explicit reset clears the latch.  No passive coordinate is ever targeted;
    all transitions use causal GPU-resident measurements.

    A coordinate that is merely moving away from its previous limit is not
    automatically safe for a full 0.8-rad/s authority.  Such a release enters
    POST_RELEASE_CONSERVATIVE: its active-mask ownership is cleared, but its
    last safe full-authority envelope cap remains authoritative until the
    full-authority proof itself clears.  This prevents a CREEP release from
    immediately re-entering TRACK -> BRAKE on the same coordinate.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        passive_joint_count: int,
        device,
        dtype,
    ) -> None:
        import torch

        if num_envs < 1 or passive_joint_count < 1:
            raise ValueError("G2_PASSIVE_GOVERNOR_SHAPE_INVALID")
        self._phase = torch.full(
            (num_envs,),
            G2_PASSIVE_LIMIT_GOVERNOR_TRACK,
            dtype=torch.long,
            device=device,
        )
        self._stationary_count = torch.zeros(
            (num_envs,), dtype=torch.long, device=device
        )
        self._release_candidate_count = torch.zeros(
            (num_envs, passive_joint_count),
            dtype=torch.long,
            device=device,
        )
        self._active_passive = torch.zeros(
            (num_envs, passive_joint_count),
            dtype=torch.bool,
            device=device,
        )
        self._active_toward_upper = torch.zeros_like(self._active_passive)
        self._post_release_passive = torch.zeros_like(self._active_passive)
        self._post_release_toward_upper = torch.zeros_like(
            self._active_passive
        )
        self._landed_passive = torch.zeros_like(self._active_passive)
        self._landed_toward_upper = torch.zeros_like(self._active_passive)
        self._active_master_direction = torch.zeros(
            (num_envs, 1), dtype=dtype, device=device
        )
        self._maximum_transmission_ratio = torch.zeros(
            (num_envs, passive_joint_count), dtype=dtype, device=device
        )
        self._governor_speed_cap = torch.zeros(
            (num_envs, 1), dtype=dtype, device=device
        )
        # Diagnostic receipt only.  ``compute_speed_cap`` replaces this with
        # tensors already calculated for its normal decision; no receipt value
        # is ever fed back into the governor.  Keeping the causal ingredients
        # together is essential for distinguishing a true passive-envelope
        # re-entry from a stale cache or a later coupled coordinate.
        self._last_decision_receipt: dict[str, object] | None = None

    @property
    def phase(self):
        return self._phase

    @property
    def governor_speed_cap(self):
        return self._governor_speed_cap

    @property
    def active_passive_mask(self):
        return self._active_passive

    @property
    def landed_passive_mask(self):
        return self._landed_passive

    @property
    def active_toward_upper_mask(self):
        return self._active_toward_upper

    @property
    def post_release_passive_mask(self):
        return self._post_release_passive

    @property
    def landed_toward_upper_mask(self):
        return self._landed_toward_upper

    @property
    def release_candidate_count(self):
        return self._release_candidate_count

    @property
    def last_decision_receipt(self):
        """Return the telemetry from the most recent governor decision.

        The returned tensors are observability state, not control authority.
        Callers must read them after the normal ``compute_speed_cap`` call and
        must never replay that call merely to obtain a receipt.
        """

        if self._last_decision_receipt is None:
            raise RuntimeError("G2_PASSIVE_GOVERNOR_RECEIPT_UNAVAILABLE")
        return self._last_decision_receipt

    def reset(self, env_ids=None) -> None:
        selection = slice(None) if env_ids is None else env_ids
        self._phase[selection] = G2_PASSIVE_LIMIT_GOVERNOR_TRACK
        self._stationary_count[selection] = 0
        self._release_candidate_count[selection] = 0
        self._active_passive[selection] = False
        self._active_toward_upper[selection] = False
        self._post_release_passive[selection] = False
        self._post_release_toward_upper[selection] = False
        self._landed_passive[selection] = False
        self._landed_toward_upper[selection] = False
        self._active_master_direction[selection] = 0.0
        self._maximum_transmission_ratio[selection] = 0.0
        self._governor_speed_cap[selection] = 0.0

    def compute_speed_cap(
        self,
        passive_position,
        passive_fd_velocity,
        passive_position_limits,
        current_master_target_velocity,
        desired_master_delta,
        *,
        maximum_master_speed_rad_s: float,
        maximum_master_target_acceleration_rad_s2: float,
        dt_s: float,
        reaction_steps: int = G2_PASSIVE_LIMIT_GUARD_REACTION_STEPS,
    ):
        import torch

        phase_at_entry = self._phase.clone()
        active_mask_at_entry = self._active_passive.clone()
        post_release_mask_at_entry = self._post_release_passive.clone()
        speed_cap_at_entry = self._governor_speed_cap.clone()

        # Reuse the stateless contract for complete shape/value validation.
        passive_limit_aware_master_speed_cap(
            passive_position,
            passive_fd_velocity,
            passive_position_limits,
            current_master_target_velocity,
            desired_master_delta,
            maximum_master_speed_rad_s=maximum_master_speed_rad_s,
            maximum_master_target_acceleration_rad_s2=(
                maximum_master_target_acceleration_rad_s2
            ),
            dt_s=dt_s,
            reaction_steps=reaction_steps,
        )
        if passive_position.shape != self._active_passive.shape:
            raise ValueError("G2_PASSIVE_GOVERNOR_STATE_SHAPE_MISMATCH")

        if passive_position_limits.ndim == 2:
            limits = passive_position_limits.unsqueeze(0)
        else:
            limits = passive_position_limits
        lower = limits[..., 0]
        upper = limits[..., 1]
        passive_speed = torch.abs(passive_fd_velocity)
        master_speed = torch.abs(current_master_target_velocity)
        configured_cap = torch.full_like(
            current_master_target_velocity,
            float(maximum_master_speed_rad_s),
        )
        brake_acceleration = float(
            G2_PASSIVE_LIMIT_GOVERNOR_BRAKE_ACCELERATION_RAD_S2
        )
        if brake_acceleration > float(
            maximum_master_target_acceleration_rad_s2
        ):
            raise ValueError("G2_PASSIVE_GOVERNOR_BRAKE_AUTHORITY_INVALID")
        governor_speed_step = brake_acceleration * float(dt_s)
        settle_speed = float(G2_PASSIVE_LIMIT_GOVERNOR_SETTLE_SPEED_RAD_S)
        landing_speed = float(maximum_master_target_acceleration_rad_s2) * float(
            dt_s
        )
        creep_passive_speed_budget = max(landing_speed - settle_speed, 0.0)
        if creep_passive_speed_budget <= 0.0:
            raise ValueError("G2_PASSIVE_GOVERNOR_CREEP_BUDGET_INVALID")

        # A requested direction reversal moves away from every stop latched in
        # the previous master direction and is never blocked.  The direction
        # field remains non-zero after a non-limiting SETTLE returns to TRACK,
        # so it also serves as the existing-state "first trigger happened"
        # bit: transmission learning stays frozen until reversal/reset.
        direction_latched = (
            torch.abs(self._active_master_direction)
            > G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S
        ).squeeze(1)
        reversing = direction_latched & (
            desired_master_delta * self._active_master_direction < 0.0
        ).squeeze(1)
        self._phase = torch.where(
            reversing,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_TRACK),
            self._phase,
        )
        self._active_passive = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._active_passive),
            self._active_passive,
        )
        self._active_toward_upper = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._active_toward_upper),
            self._active_toward_upper,
        )
        self._post_release_passive = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._post_release_passive),
            self._post_release_passive,
        )
        self._post_release_toward_upper = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._post_release_toward_upper),
            self._post_release_toward_upper,
        )
        self._landed_passive = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._landed_passive),
            self._landed_passive,
        )
        self._landed_toward_upper = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._landed_toward_upper),
            self._landed_toward_upper,
        )
        self._stationary_count = torch.where(
            reversing,
            torch.zeros_like(self._stationary_count),
            self._stationary_count,
        )
        self._release_candidate_count = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._release_candidate_count),
            self._release_candidate_count,
        )
        self._active_master_direction = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._active_master_direction),
            self._active_master_direction,
        )
        self._maximum_transmission_ratio = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._maximum_transmission_ratio),
            self._maximum_transmission_ratio,
        )
        self._governor_speed_cap = torch.where(
            reversing.unsqueeze(1),
            torch.zeros_like(self._governor_speed_cap),
            self._governor_speed_cap,
        )
        direction_latched = direction_latched & ~reversing

        # A later four-bar coordinate can remain stationary until an earlier
        # coordinate has landed.  Keep a monotone maximum for every coordinate
        # throughout clean TRACK, including TRACK resumed after an earlier
        # constraint.  BRAKE/SETTLE/CREEP response is still excluded, so the
        # estimate never learns from stop projection or terminal chatter.
        ratio_learning_phase = (
            self._phase == G2_PASSIVE_LIMIT_GOVERNOR_TRACK
        ).unsqueeze(1)
        ratio_valid = (
            ratio_learning_phase
            & (master_speed > settle_speed)
        )
        instantaneous_ratio = passive_speed / torch.clamp_min(
            master_speed, settle_speed
        )
        self._maximum_transmission_ratio = torch.where(
            ratio_valid,
            torch.maximum(
                self._maximum_transmission_ratio, instantaneous_ratio
            ),
            self._maximum_transmission_ratio,
        )
        # TRACK/SETTLE/POST_STOP use the learned, non-inflated prediction at
        # full configured authority.  CREEP and POST_RELEASE_CONSERVATIVE
        # predict at their *current* bounded cap and apply the evidence-based
        # 3x factor to that predicted branch only.  Measured speed is never
        # multiplied: max(measured, 3*predicted).
        conservative_phase = (
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
            | (
                self._phase
                == G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
            )
        )
        transmission_reference_master_speed = torch.where(
            conservative_phase.unsqueeze(1),
            self._governor_speed_cap,
            torch.full_like(master_speed, float(maximum_master_speed_rad_s)),
        )
        predicted_passive_speed = (
            self._maximum_transmission_ratio
            * transmission_reference_master_speed
        )
        predicted_uncertainty = torch.where(
            conservative_phase.unsqueeze(1),
            torch.full_like(
                predicted_passive_speed,
                float(
                    G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER
                ),
            ),
            torch.ones_like(predicted_passive_speed),
        )
        conservative_passive_speed = torch.maximum(
            passive_speed,
            predicted_uncertainty * predicted_passive_speed,
        )
        measured_direction = torch.sign(passive_fd_velocity)
        active_direction = torch.where(
            self._active_toward_upper,
            torch.ones_like(passive_fd_velocity),
            -torch.ones_like(passive_fd_velocity),
        )
        post_release_direction = torch.where(
            self._post_release_toward_upper,
            torch.ones_like(passive_fd_velocity),
            -torch.ones_like(passive_fd_velocity),
        )
        conservative_passive_mask = (
            self._active_passive | self._post_release_passive
        )
        conservative_direction = torch.where(
            self._active_passive,
            active_direction,
            post_release_direction,
        )
        envelope_direction = torch.where(
            conservative_passive_mask,
            conservative_direction,
            torch.where(
                passive_speed
                > G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S,
                measured_direction,
                torch.zeros_like(passive_fd_velocity),
            ),
        )
        envelope_toward_upper = envelope_direction > 0.0
        stop_band = torch.full_like(
            passive_position,
            float(G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD),
        )
        landed_limit_margin = torch.where(
            self._landed_toward_upper,
            upper - passive_position,
            passive_position - lower,
        ).clamp_min(0.0)
        landed_same_direction = (
            self._landed_passive
            & (envelope_direction != 0.0)
            & (self._landed_toward_upper == envelope_toward_upper)
            & (landed_limit_margin <= stop_band)
        )
        numeric_limit_tolerance = float(
            G2_PASSIVE_LIMIT_GOVERNOR_NUMERIC_LANDING_TOLERANCE_RAD
        )
        current_active_limit_margin = torch.where(
            self._active_toward_upper,
            upper - passive_position,
            passive_position - lower,
        ).clamp_min(0.0)
        landing_candidate_mask = (
            self._active_passive
            & (current_active_limit_margin <= numeric_limit_tolerance)
            & (passive_speed <= settle_speed)
        )
        active_in_stop_band = self._active_passive & (
            current_active_limit_margin <= stop_band
        )
        envelope_velocity = envelope_direction * conservative_passive_speed
        envelope_velocity = torch.where(
            landed_same_direction,
            torch.zeros_like(envelope_velocity),
            envelope_velocity,
        )
        per_passive_envelope_cap = (
            _passive_limit_aware_master_speed_cap_per_passive(
            passive_position,
            envelope_velocity,
            passive_position_limits,
            current_master_target_velocity,
            desired_master_delta,
            maximum_master_speed_rad_s=maximum_master_speed_rad_s,
            maximum_master_target_acceleration_rad_s2=brake_acceleration,
            dt_s=dt_s,
            reaction_steps=reaction_steps,
            )
        )
        # Release decisions restore the full TRACK authority, so validate that
        # next state with a separate full-authority prediction.  Reusing the
        # current low CREEP prediction here would make a low cap certify its
        # own release and immediately recreate the risk at 0.8 rad/s.
        full_authority_predicted_passive_speed = torch.maximum(
            passive_speed,
            float(
                G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER
            )
            * self._maximum_transmission_ratio
            * configured_cap,
        )
        # Release proof follows the latched direction for an active coordinate
        # and for a moving-away coordinate retained in the post-release
        # conservative state.  Tiny away-directed solver noise must not make
        # the proof inspect the opposite, distant joint limit.
        full_authority_direction = torch.where(
            self._active_passive,
            active_direction,
            torch.where(
                self._post_release_passive,
                post_release_direction,
                envelope_direction,
            ),
        )
        full_authority_envelope_velocity = (
            full_authority_direction * full_authority_predicted_passive_speed
        )
        full_authority_envelope_velocity = torch.where(
            landed_same_direction,
            torch.zeros_like(full_authority_envelope_velocity),
            full_authority_envelope_velocity,
        )
        # The release proof evaluates the authority that would be restored on
        # the next TRACK sample, not the currently stopped/slow CREEP target.
        # In particular, the passive-limit helper's braking-distance term uses
        # ``abs(current_master_target_velocity)``.  Passing the current CREEP
        # velocity (often zero) here would prove only the reaction distance and
        # omit braking from the proposed full 0.8-rad/s authority.
        full_authority_master_direction = torch.where(
            torch.abs(self._active_master_direction)
            > G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S,
            torch.sign(self._active_master_direction),
            torch.sign(desired_master_delta),
        )
        full_authority_master_velocity = (
            full_authority_master_direction * configured_cap
        )
        full_authority_per_passive_cap = (
            _passive_limit_aware_master_speed_cap_per_passive(
                passive_position,
                full_authority_envelope_velocity,
                passive_position_limits,
                full_authority_master_velocity,
                desired_master_delta,
                maximum_master_speed_rad_s=maximum_master_speed_rad_s,
                maximum_master_target_acceleration_rad_s2=brake_acceleration,
                dt_s=dt_s,
                reaction_steps=reaction_steps,
            )
        )
        # A CREEP coordinate may be released because it is clearly travelling
        # away from its previous stop even though restoring 0.8 rad/s would
        # immediately be unsafe.  Retain that coordinate's direction and cap
        # authority until this exact full-authority proof succeeds.  This is a
        # state transition only: it neither changes the configured cap nor a
        # limit, transmission, controller, or mechanics parameter.
        post_release_full_authority_safe = (
            self._post_release_passive
            & (full_authority_per_passive_cap >= configured_cap)
        )
        self._post_release_passive = (
            self._post_release_passive & ~post_release_full_authority_safe
        )
        self._post_release_toward_upper = (
            self._post_release_toward_upper
            & ~post_release_full_authority_safe
        )
        post_release_complete = (
            (
                self._phase
                == G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
            )
            & ~torch.any(self._post_release_passive, dim=1)
            & ~torch.any(self._active_passive, dim=1)
        )
        landed_any_before_post_release_complete = torch.any(
            self._landed_passive, dim=1
        )
        self._phase = torch.where(
            post_release_complete & landed_any_before_post_release_complete,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP),
            self._phase,
        )
        self._phase = torch.where(
            post_release_complete & ~landed_any_before_post_release_complete,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_TRACK),
            self._phase,
        )
        # Use the exact per-coordinate cap comparison, not a TTC argmin.  All
        # equally restrictive coordinates remain visible and joint ordering
        # cannot change the latch identity.
        trigger_reference_cap = torch.where(
            conservative_phase.unsqueeze(1),
            self._governor_speed_cap,
            configured_cap,
        )
        risk_mask = (
            per_passive_envelope_cap < trigger_reference_cap
        ) & ~landed_same_direction & ~landing_candidate_mask

        active_speed = torch.amax(
            torch.where(
                self._active_passive,
                passive_speed,
                torch.zeros_like(passive_speed),
            ),
            dim=1,
        )

        # BRAKE ramps the speed cap itself, so the generic target limiter sees
        # a continuous 2-rad/s^2 command rather than an instantaneous zero cap.
        braking = self._phase == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE
        reduced_brake_cap = torch.clamp_min(
            self._governor_speed_cap - governor_speed_step, 0.0
        )
        self._governor_speed_cap = torch.where(
            braking.unsqueeze(1),
            reduced_brake_cap,
            self._governor_speed_cap,
        )
        brake_complete = (
            braking
            & (self._governor_speed_cap.squeeze(1) <= settle_speed)
            & (master_speed.squeeze(1) <= settle_speed)
        )
        self._phase = torch.where(
            brake_complete,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_SETTLE),
            self._phase,
        )
        self._stationary_count = torch.where(
            brake_complete,
            torch.zeros_like(self._stationary_count),
            self._stationary_count,
        )

        # SETTLE requires two consecutive causal samples with
        # both master target and the latched passive coordinate stationary.
        settling = self._phase == G2_PASSIVE_LIMIT_GOVERNOR_SETTLE
        stationary_now = (active_speed <= settle_speed) & (
            master_speed.squeeze(1) <= settle_speed
        )
        self._stationary_count = torch.where(
            settling & stationary_now,
            self._stationary_count + 1,
            torch.where(
                settling,
                torch.zeros_like(self._stationary_count),
                self._stationary_count,
            ),
        )
        settle_complete = settling & (
            self._stationary_count
            >= G2_PASSIVE_LIMIT_GOVERNOR_STATIONARY_SAMPLES
        )
        # BRAKE/SETTLE is a temporary stop for the coordinates that caused the
        # stop.  A zero-velocity SETTLE sample must not erase that identity and
        # release full TRACK authority; CREEP owns the bounded release test.
        settle_active_mask = (
            self._active_passive
            | risk_mask
            | landing_candidate_mask
            | active_in_stop_band
        )
        still_limited_after_settle = torch.any(settle_active_mask, dim=1)
        settle_active_direction = torch.where(
            self._active_passive,
            self._active_toward_upper,
            envelope_toward_upper,
        )
        begin_creep = settle_complete & still_limited_after_settle
        resume_without_creep = settle_complete & ~still_limited_after_settle
        landed_any = torch.any(self._landed_passive, dim=1)
        self._phase = torch.where(
            begin_creep,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_CREEP),
            self._phase,
        )
        self._phase = torch.where(
            resume_without_creep & landed_any,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP),
            self._phase,
        )
        self._phase = torch.where(
            resume_without_creep & ~landed_any,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_TRACK),
            self._phase,
        )
        self._active_passive = torch.where(
            begin_creep.unsqueeze(1),
            settle_active_mask,
            torch.where(
                resume_without_creep.unsqueeze(1),
                torch.zeros_like(self._active_passive),
                self._active_passive,
            ),
        )
        self._active_toward_upper = torch.where(
            begin_creep.unsqueeze(1),
            torch.where(
                settle_active_mask,
                settle_active_direction,
                torch.zeros_like(self._active_toward_upper),
            ),
            torch.where(
                resume_without_creep.unsqueeze(1),
                torch.zeros_like(self._active_toward_upper),
                self._active_toward_upper,
            ),
        )
        self._stationary_count = torch.where(
            settle_complete,
            torch.zeros_like(self._stationary_count),
            self._stationary_count,
        )
        self._release_candidate_count = torch.where(
            settle_complete.unsqueeze(1),
            torch.zeros_like(self._release_candidate_count),
            self._release_candidate_count,
        )
        self._governor_speed_cap = torch.where(
            begin_creep.unsqueeze(1),
            torch.zeros_like(self._governor_speed_cap),
            self._governor_speed_cap,
        )

        # The first envelope crossing latches every restrictive coordinate and
        # starts continuous BRAKE.  During CREEP, later coordinates are unioned
        # into the active mask before braking, so simultaneous and sequential
        # risks are both preserved.  A SETTLE completion is excluded from
        # immediate re-trigger: it either enters CREEP with the audited mask or
        # returns to TRACK/POST_STOP above.
        trigger_eligible = (
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_TRACK)
            | (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
            | (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP)
            | (
                self._phase
                == G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
            )
        ) & ~settle_complete & ~torch.any(landing_candidate_mask, dim=1)
        # In CREEP, risk from an already-active coordinate is expected: CREEP
        # is the deliberately bounded approach to that coordinate's stop.  It
        # remains protected by the active-coordinate envelope cap and measured
        # landing-overspeed check below.  Only a newly restrictive coordinate
        # requires another BRAKE/SETTLE cycle and union into the active set.
        trigger_risk_mask = torch.where(
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP).unsqueeze(1),
            risk_mask & ~self._active_passive,
            torch.where(
                (
                    self._phase
                    == G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
                ).unsqueeze(1),
                # The retained post-release coordinate is already governed by
                # its full-authority envelope cap.  It must not recreate an
                # active-mask BRAKE loop merely because full TRACK is not yet
                # proven; distinct newly restrictive coordinates still enter
                # the ordinary BRAKE/SETTLE path.
                risk_mask & ~self._post_release_passive,
                risk_mask,
            ),
        )
        trigger = trigger_eligible & torch.any(trigger_risk_mask, dim=1)
        triggered_active = self._active_passive | trigger_risk_mask
        self._active_passive = torch.where(
            trigger.unsqueeze(1), triggered_active, self._active_passive
        )
        self._active_toward_upper = torch.where(
            trigger.unsqueeze(1),
            torch.where(
                trigger_risk_mask,
                envelope_toward_upper,
                self._active_toward_upper,
            ),
            self._active_toward_upper,
        )
        current_direction = torch.where(
            master_speed
            > G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S,
            torch.sign(current_master_target_velocity),
            torch.sign(desired_master_delta),
        )
        first_trigger = trigger.unsqueeze(1) & ~direction_latched.unsqueeze(1)
        self._active_master_direction = torch.where(
            first_trigger,
            current_direction,
            self._active_master_direction,
        )
        self._phase = torch.where(
            trigger,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_BRAKE),
            self._phase,
        )
        self._stationary_count = torch.where(
            trigger,
            torch.zeros_like(self._stationary_count),
            self._stationary_count,
        )
        self._release_candidate_count = torch.where(
            trigger.unsqueeze(1),
            torch.zeros_like(self._release_candidate_count),
            self._release_candidate_count,
        )
        # The trigger sample retains the current cap.  The following samples
        # decrease it by exactly governor_speed_step, avoiding a discontinuity.
        self._governor_speed_cap = torch.where(
            trigger.unsqueeze(1),
            torch.minimum(master_speed, configured_cap),
            self._governor_speed_cap,
        )

        # CREEP authority is derived only from the observed transmission and
        # the residual acceleration budget: 0.005 + 0.015 <= 0.020 rad/s.
        active_ratio = torch.amax(
            torch.where(
                self._active_passive,
                self._maximum_transmission_ratio,
                torch.zeros_like(self._maximum_transmission_ratio),
            ),
            dim=1,
            keepdim=True,
        )
        conservative_active_ratio = active_ratio * float(
            G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER
        )
        requested_creep_cap = torch.where(
            conservative_active_ratio
            > G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S,
            torch.full_like(active_ratio, creep_passive_speed_budget)
            / torch.clamp_min(
                conservative_active_ratio,
                G2_PASSIVE_LIMIT_GUARD_VELOCITY_EPSILON_RAD_S,
            ),
            torch.zeros_like(active_ratio),
        ).clamp(0.0, float(maximum_master_speed_rad_s))
        creeping = self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
        previous_creep_cap = self._governor_speed_cap
        ramped_creep_cap = torch.minimum(
            requested_creep_cap,
            previous_creep_cap + governor_speed_step,
        )
        active_envelope_cap = torch.amin(
            torch.where(
                self._active_passive,
                per_passive_envelope_cap,
                configured_cap.expand_as(per_passive_envelope_cap),
            ),
            dim=1,
            keepdim=True,
        )
        creep_envelope_target = torch.minimum(
            ramped_creep_cap, active_envelope_cap
        )
        # The per-passive envelope reserves the configured reaction interval
        # and assumes the same 2-rad/s^2 braking authority.  Respect it through
        # a causal downward slew rather than an instantaneous cap discontinuity.
        creep_envelope_slew = torch.where(
            creep_envelope_target < previous_creep_cap,
            torch.maximum(
                creep_envelope_target,
                torch.clamp_min(
                    previous_creep_cap - governor_speed_step, 0.0
                ),
            ),
            creep_envelope_target,
        )
        self._governor_speed_cap = torch.where(
            creeping.unsqueeze(1),
            creep_envelope_slew,
            self._governor_speed_cap,
        )
        landing_overspeed = creeping & (active_speed > landing_speed)
        self._phase = torch.where(
            landing_overspeed,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_BRAKE),
            self._phase,
        )
        self._stationary_count = torch.where(
            landing_overspeed,
            torch.zeros_like(self._stationary_count),
            self._stationary_count,
        )
        self._governor_speed_cap = torch.where(
            landing_overspeed.unsqueeze(1),
            torch.maximum(master_speed, self._governor_speed_cap),
            self._governor_speed_cap,
        )

        active_limit_margin = torch.where(
            self._active_toward_upper,
            upper - passive_position,
            passive_position - lower,
        ).clamp_min(0.0)
        active_limit_margin = torch.where(
            self._active_passive,
            active_limit_margin,
            torch.full_like(active_limit_margin, float("inf")),
        )
        active_at_limit = self._active_passive & (
            active_limit_margin <= numeric_limit_tolerance
        )
        active_in_stop_band = self._active_passive & (
            active_limit_margin <= stop_band
        )
        landed_stationary = (
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
            & torch.any(active_at_limit, dim=1)
            & (active_speed <= settle_speed)
        )
        self._stationary_count = torch.where(
            landed_stationary,
            self._stationary_count + 1,
            torch.where(
                self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP,
                torch.zeros_like(self._stationary_count),
                self._stationary_count,
            ),
        )
        commit_landing = (
            self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
        ) & (
            self._stationary_count
            >= G2_PASSIVE_LIMIT_GOVERNOR_STATIONARY_SAMPLES
        )
        clearly_moving_away = self._active_passive & (
            (
                self._active_toward_upper
                & (
                    passive_fd_velocity
                    < -settle_speed
                )
            )
            | (
                ~self._active_toward_upper
                & (
                    passive_fd_velocity
                    > settle_speed
                )
            )
        )
        nonrestrictive_and_settled = (
            self._active_passive
            # Releasing an active coordinate returns this environment to the
            # full TRACK cap.  Prove non-restriction at that *next* authority,
            # not merely at the current low CREEP cap, otherwise the next
            # sample can immediately trigger another BRAKE cycle.
            & (full_authority_per_passive_cap >= configured_cap)
            & (passive_speed <= settle_speed)
        )
        release_evidence = (
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP).unsqueeze(1)
            & ~active_at_limit
            & ~active_in_stop_band
            & (clearly_moving_away | nonrestrictive_and_settled)
        )
        self._release_candidate_count = torch.where(
            release_evidence,
            self._release_candidate_count + 1,
            torch.zeros_like(self._release_candidate_count),
        )
        release_nonrestrictive = (
            self._release_candidate_count
            >= G2_PASSIVE_LIMIT_GOVERNOR_STATIONARY_SAMPLES
        ) & self._active_passive
        newly_landed = commit_landing.unsqueeze(1) & active_at_limit
        # ``clearly_moving_away`` is permitted to release the active mask so
        # the joint no longer blocks a different terminal coordinate.  It is
        # not sufficient evidence for immediate full TRACK authority.  Keep
        # exactly those releases under their previously latched direction
        # until the separately computed full-authority cap reaches the
        # unchanged configured cap.
        moving_away_only_release = (
            release_nonrestrictive
            & clearly_moving_away
            # The post-release state is needed only when the full-authority
            # envelope itself is still restrictive.  A moving coordinate can
            # safely resume TRACK when its per-passive full cap is already
            # proven to reach the unchanged configured cap; requiring it to
            # be stationary here would create unnecessary conservative holds.
            & (full_authority_per_passive_cap < configured_cap)
            & ~newly_landed
        )
        self._post_release_toward_upper = torch.where(
            moving_away_only_release,
            self._active_toward_upper,
            self._post_release_toward_upper,
        )
        self._post_release_passive = (
            self._post_release_passive | moving_away_only_release
        )
        self._landed_toward_upper = torch.where(
            newly_landed,
            self._active_toward_upper,
            self._landed_toward_upper,
        )
        self._landed_passive = self._landed_passive | newly_landed
        released_coordinates = newly_landed | release_nonrestrictive
        self._active_passive = self._active_passive & ~released_coordinates
        self._active_toward_upper = (
            self._active_toward_upper & ~released_coordinates
        )
        self._release_candidate_count = torch.where(
            released_coordinates,
            torch.zeros_like(self._release_candidate_count),
            self._release_candidate_count,
        )
        active_remaining = torch.any(self._active_passive, dim=1)
        release = (
            (commit_landing | torch.any(release_nonrestrictive, dim=1))
            & ~active_remaining
        )
        landed_any_after_release = torch.any(self._landed_passive, dim=1)
        post_release_remaining = torch.any(self._post_release_passive, dim=1)
        self._phase = torch.where(
            release & post_release_remaining,
            torch.full_like(
                self._phase,
                G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE,
            ),
            self._phase,
        )
        self._phase = torch.where(
            release & ~post_release_remaining & landed_any_after_release,
            torch.full_like(
                self._phase, G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP
            ),
            self._phase,
        )
        self._phase = torch.where(
            release & ~post_release_remaining & ~landed_any_after_release,
            torch.full_like(self._phase, G2_PASSIVE_LIMIT_GOVERNOR_TRACK),
            self._phase,
        )
        self._stationary_count = torch.where(
            commit_landing | torch.any(release_nonrestrictive, dim=1),
            torch.zeros_like(self._stationary_count),
            self._stationary_count,
        )

        # POST_RELEASE_CONSERVATIVE owns a released moving-away coordinate
        # until full TRACK authority has been proved.  It uses the same causal
        # envelope computation as the existing guard, but does not re-add the
        # coordinate to the active mask.  Its only authority is a bounded
        # master-target speed cap; no joint, torque, limit, or mechanics value
        # is written or changed here.
        post_releasing = (
            self._phase
            == G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
        )
        post_release_envelope_target = torch.amin(
            torch.where(
                self._post_release_passive,
                full_authority_per_passive_cap,
                configured_cap.expand_as(full_authority_per_passive_cap),
            ),
            dim=1,
            keepdim=True,
        )
        previous_post_release_cap = self._governor_speed_cap
        post_release_cap_slew = torch.where(
            post_release_envelope_target < previous_post_release_cap,
            torch.maximum(
                post_release_envelope_target,
                torch.clamp_min(
                    previous_post_release_cap - governor_speed_step, 0.0
                ),
            ),
            torch.minimum(
                post_release_envelope_target,
                previous_post_release_cap + governor_speed_step,
            ),
        )
        self._governor_speed_cap = torch.where(
            post_releasing.unsqueeze(1),
            post_release_cap_slew,
            self._governor_speed_cap,
        )

        result = configured_cap
        result = torch.where(
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE).unsqueeze(1),
            self._governor_speed_cap,
            result,
        )
        result = torch.where(
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_SETTLE).unsqueeze(1),
            torch.zeros_like(result),
            result,
        )
        result = torch.where(
            (self._phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP).unsqueeze(1),
            self._governor_speed_cap,
            result,
        )
        result = torch.where(
            (
                self._phase
                == G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
            ).unsqueeze(1),
            self._governor_speed_cap,
            result,
        )

        # Persist the exact internal decision operands.  The code below only
        # classifies changes that have already happened above; it deliberately
        # does not alter the masks, phase, cap, or return value.
        active_mask_after = self._active_passive
        post_release_mask_after = self._post_release_passive
        actual_added = ~active_mask_at_entry & active_mask_after
        actual_removed = active_mask_at_entry & ~active_mask_after
        add_reason_code = torch.zeros_like(self._release_candidate_count)
        add_from_trigger = actual_added & trigger_risk_mask
        add_from_creep = add_from_trigger & (
            phase_at_entry.unsqueeze(1)
            == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
        )
        add_from_full_authority = add_from_trigger & ~add_from_creep
        add_from_settle = actual_added & settle_active_mask & ~add_from_trigger
        add_reason_code = torch.where(
            add_from_full_authority,
            torch.ones_like(add_reason_code),
            add_reason_code,
        )
        add_reason_code = torch.where(
            add_from_creep,
            torch.full_like(add_reason_code, 2),
            add_reason_code,
        )
        add_reason_code = torch.where(
            add_from_settle,
            torch.full_like(add_reason_code, 3),
            add_reason_code,
        )
        remove_reason_code = torch.zeros_like(self._release_candidate_count)
        remove_by_landing = actual_removed & newly_landed
        remove_by_nonrestrictive = actual_removed & release_nonrestrictive
        remove_by_reversal = actual_removed & reversing.unsqueeze(1)
        remove_reason_code = torch.where(
            remove_by_landing,
            torch.ones_like(remove_reason_code),
            remove_reason_code,
        )
        remove_reason_code = torch.where(
            remove_by_nonrestrictive,
            torch.full_like(remove_reason_code, 2),
            remove_reason_code,
        )
        remove_reason_code = torch.where(
            remove_by_reversal,
            torch.full_like(remove_reason_code, 3),
            remove_reason_code,
        )
        envelope_limit_margin = torch.where(
            envelope_toward_upper,
            upper - passive_position,
            passive_position - lower,
        ).clamp_min(0.0)
        self._last_decision_receipt = {
            "phase_at_entry": phase_at_entry,
            "phase_after": self._phase.clone(),
            "speed_cap_at_entry_rad_s": speed_cap_at_entry,
            "speed_cap_after_rad_s": self._governor_speed_cap.clone(),
            "active_mask_at_entry": active_mask_at_entry,
            "active_mask_after": active_mask_after.clone(),
            "post_release_mask_at_entry": post_release_mask_at_entry,
            "post_release_mask_after": post_release_mask_after.clone(),
            "per_passive_envelope_cap_rad_s": per_passive_envelope_cap,
            "full_authority_per_passive_cap_rad_s": (
                full_authority_per_passive_cap
            ),
            "trigger_reference_cap_rad_s": trigger_reference_cap,
            "passive_risk_score_rad_s": (
                trigger_reference_cap - per_passive_envelope_cap
            ),
            "passive_risk_mask": risk_mask,
            "trigger_risk_mask": trigger_risk_mask,
            "predicted_passive_speed_rad_s": predicted_passive_speed,
            "conservative_passive_speed_rad_s": conservative_passive_speed,
            "projected_master_space_velocity_rad_s": (
                per_passive_envelope_cap
            ),
            "transmission_ratio_used": self._maximum_transmission_ratio.clone(),
            "envelope_limit_margin_rad": envelope_limit_margin,
            "active_mask_add_reason_code": add_reason_code,
            "active_mask_remove_reason_code": remove_reason_code,
            "release_evidence_clearly_moving_away": clearly_moving_away,
            "release_evidence_full_authority_safe": (
                nonrestrictive_and_settled
            ),
            "post_release_full_authority_safe": (
                post_release_full_authority_safe
            ),
        }
        return result


def capture_measured_right_hand_cache(
    measured_joint_position,
    joint_names: Sequence[str],
    *,
    safety_attestation: dict[str, object],
):
    """Clone all eight measured right-hand coordinates after safe settling.

    A position snapshot alone is not evidence that the passive four-bar is
    settled.  Requiring the full-articulation attestation here prevents a
    caller from turning an intermediate constraint-projection state into the
    reset authority for all future episodes.
    """

    if safety_attestation.get("schema") != G2_GRIPPER_RESET_CONTRACT_SCHEMA:
        raise ValueError("G2_RIGHT_HAND_CACHE_ATTESTATION_SCHEMA_MISMATCH")
    if not bool(safety_attestation.get("pass", False)):
        raise ValueError("G2_RIGHT_HAND_CACHE_REQUIRES_SAFE_SETTLE_PASS")
    # Do not let a downstream caller mask a failed raw audit by changing only
    # the aggregate ``pass`` bit.  Cache authority requires the v2 evidence
    # fields themselves to be complete and internally consistent.
    if int(safety_attestation.get("joint_count", 0)) != len(
        G2_RUNTIME_JOINT_ORDER
    ):
        raise ValueError("G2_RIGHT_HAND_CACHE_ATTESTATION_JOINT_COUNT_MISMATCH")
    if int(safety_attestation.get("sample_count", 0)) < (
        G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
    ):
        raise ValueError("G2_RIGHT_HAND_CACHE_REQUIRES_MINIMUM_OPEN_SETTLE")
    if not bool(
        safety_attestation.get("first_post_reset_position_interval_included", False)
    ):
        raise ValueError("G2_RIGHT_HAND_CACHE_REQUIRES_FIRST_INTERVAL")
    maximum_velocity = float(
        safety_attestation.get("maximum_fd_velocity_rad_s", float("inf"))
    )
    maximum_acceleration = float(
        safety_attestation.get("maximum_fd_acceleration_rad_s2", float("inf"))
    )
    if (
        not math.isfinite(maximum_velocity)
        or maximum_velocity > G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S
    ):
        raise ValueError("G2_RIGHT_HAND_CACHE_ATTESTATION_VELOCITY_FAILED")
    if (
        not math.isfinite(maximum_acceleration)
        or maximum_acceleration > G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
    ):
        raise ValueError("G2_RIGHT_HAND_CACHE_ATTESTATION_ACCELERATION_FAILED")
    if int(
        safety_attestation.get(
            "full46_validated_fd_acceleration_sample_count", 0
        )
    ) < 1:
        raise ValueError("G2_RIGHT_HAND_CACHE_REQUIRES_VALID_ACCELERATION")
    if int(
        safety_attestation.get("consecutive_settled_full46_samples", 0)
    ) < G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES:
        raise ValueError("G2_RIGHT_HAND_CACHE_REQUIRES_CONSECUTIVE_SETTLE")

    indices = resolve_full_right_hand_indices(joint_names)
    if measured_joint_position.ndim != 2:
        raise ValueError("G2_MEASURED_JOINT_POSITION_MUST_BE_BATCHED")
    if measured_joint_position.shape[1] != len(joint_names):
        raise ValueError("G2_MEASURED_JOINT_POSITION_WIDTH_MISMATCH")
    return measured_joint_position[:, list(indices)].detach().clone()


def install_measured_right_hand_cache(
    default_joint_position,
    cache,
    joint_names: Sequence[str],
    env_ids=None,
) -> None:
    """Install measured full-hand state for future resets, without stepping.

    This mutates only the articulation's default/reset state.  It neither
    writes a drive target nor advances physics, so it is safe for vector
    asynchronous resets.  The ordinary outer simulation loop remains the
    sole owner of physics stepping.
    """

    indices = resolve_full_right_hand_indices(joint_names)
    if default_joint_position.ndim != 2 or cache.ndim != 2:
        raise ValueError("G2_RIGHT_HAND_CACHE_MUST_BE_BATCHED")
    if cache.shape[1] != len(indices):
        raise ValueError("G2_RIGHT_HAND_CACHE_WIDTH_MISMATCH")
    if env_ids is None:
        if cache.shape[0] != default_joint_position.shape[0]:
            raise ValueError("G2_RIGHT_HAND_CACHE_BATCH_MISMATCH")
        default_joint_position[:, list(indices)] = cache
        return
    if cache.shape[0] != len(env_ids):
        raise ValueError("G2_RIGHT_HAND_CACHE_ENV_SELECTION_MISMATCH")
    default_joint_position[env_ids[:, None], list(indices)] = cache


@dataclass
class FullArticulationFDSafetyAudit:
    """GPU-compatible bounded reset audit over all 46 named DoFs.

    The first ``q0 -> q1`` interval always contributes displacement and FD
    velocity to the hard gate.  Acceleration is *not* fabricated by assuming a
    zero velocity before ``q0``.  Unless a caller supplies an independently
    valid initial velocity, the first full-46 acceleration is the second
    difference of three real positions ``q0, q1, q2`` at observation two.

    Thus the first sample is retained, while its unknowable acceleration is
    explicitly reported as unresolved.  A PASS requires at least one valid
    full-46 acceleration sample and five consecutive settled full-46 samples.
    """

    joint_names: tuple[str, ...]
    dt_s: float
    previous_position: object
    previous_velocity: object | None
    initial_acceleration_valid_mask: object
    maximum_velocity_rad_s: float = 0.0
    maximum_acceleration_rad_s2: float | None = None
    maximum_velocity_event: dict[str, int | float | str] | None = None
    maximum_acceleration_event: dict[str, int | float | str] | None = None
    first_interval_maximum_position_delta_rad: float | None = None
    first_interval_maximum_fd_velocity_rad_s: float | None = None
    first_interval_maximum_velocity_event: dict[str, int | float | str] | None = None
    validated_fd_acceleration_sample_count: int = 0
    full46_validated_fd_acceleration_sample_count: int = 0
    consecutive_settled_full46_samples: int = 0
    sample_count: int = 0

    @classmethod
    def start(
        cls,
        initial_position,
        joint_names: Sequence[str],
        *,
        dt_s: float,
        initial_velocity=None,
        initial_velocity_valid_mask=None,
    ) -> "FullArticulationFDSafetyAudit":
        import torch

        names = tuple(joint_names)
        if names != G2_RUNTIME_JOINT_ORDER:
            raise ValueError("G2_FULL46_FD_JOINT_ORDER_MISMATCH")
        if initial_position.ndim != 2 or initial_position.shape[1] != 46:
            raise ValueError("G2_FULL46_FD_INITIAL_SHAPE_MISMATCH")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("G2_FULL46_FD_DT_INVALID")
        if initial_velocity is None:
            previous_velocity = None
            valid_mask = torch.zeros_like(initial_position, dtype=torch.bool)
        else:
            if initial_velocity.shape != initial_position.shape:
                raise ValueError("G2_FULL46_FD_INITIAL_VELOCITY_SHAPE_MISMATCH")
            if not bool(torch.isfinite(initial_velocity).all()):
                raise ValueError("G2_FULL46_FD_INITIAL_VELOCITY_NONFINITE")
            previous_velocity = initial_velocity.detach().clone()
            if initial_velocity_valid_mask is None:
                raise ValueError("G2_FULL46_FD_INITIAL_VELOCITY_VALIDITY_REQUIRED")
            if initial_velocity_valid_mask.shape != initial_position.shape:
                raise ValueError("G2_FULL46_FD_INITIAL_VALIDITY_SHAPE_MISMATCH")
            valid_mask = initial_velocity_valid_mask.detach().clone().bool()
        return cls(
            joint_names=names,
            dt_s=float(dt_s),
            previous_position=initial_position.detach().clone(),
            previous_velocity=previous_velocity,
            initial_acceleration_valid_mask=valid_mask,
        )

    def observe(self, position, *, step: int) -> None:
        import torch

        if position.shape != self.previous_position.shape:
            raise ValueError("G2_FULL46_FD_POSITION_SHAPE_CHANGED")
        delta = torch.atan2(
            torch.sin(position - self.previous_position),
            torch.cos(position - self.previous_position),
        )
        velocity = delta / self.dt_s
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("G2_FULL46_FD_NONFINITE")

        def maximum_event(values, metric: str):
            flat_index = int(torch.argmax(torch.abs(values)).item())
            width = values.shape[1]
            env_index, joint_index = divmod(flat_index, width)
            magnitude = float(torch.abs(values[env_index, joint_index]).item())
            event = {
                "sample": int(self.sample_count + 1),
                "step": int(step),
                "env": int(env_index),
                "joint_index": int(joint_index),
                "joint_name": self.joint_names[joint_index],
                "signed_value": float(values[env_index, joint_index].item()),
            }
            return magnitude, event

        velocity_magnitude, velocity_event = maximum_event(velocity, "velocity")
        if velocity_magnitude > self.maximum_velocity_rad_s:
            self.maximum_velocity_rad_s = velocity_magnitude
            self.maximum_velocity_event = velocity_event

        if self.sample_count == 0:
            delta_magnitude, _ = maximum_event(delta, "position_delta")
            self.first_interval_maximum_position_delta_rad = delta_magnitude
            self.first_interval_maximum_fd_velocity_rad_s = velocity_magnitude
            self.first_interval_maximum_velocity_event = velocity_event

        acceleration = None
        acceleration_valid_mask = None
        if self.previous_velocity is not None:
            acceleration = (velocity - self.previous_velocity) / self.dt_s
            if not bool(torch.isfinite(acceleration).all()):
                raise RuntimeError("G2_FULL46_FD_NONFINITE")
            acceleration_valid_mask = (
                self.initial_acceleration_valid_mask
                if self.sample_count == 0
                else torch.ones_like(position, dtype=torch.bool)
            )
            if bool(acceleration_valid_mask.any()):
                valid_abs = torch.abs(acceleration).masked_fill(
                    ~acceleration_valid_mask, float("-inf")
                )
                flat_index = int(torch.argmax(valid_abs).item())
                width = acceleration.shape[1]
                env_index, joint_index = divmod(flat_index, width)
                acceleration_magnitude = float(
                    valid_abs[env_index, joint_index].item()
                )
                acceleration_event = {
                    "sample": int(self.sample_count + 1),
                    "step": int(step),
                    "env": int(env_index),
                    "joint_index": int(joint_index),
                    "joint_name": self.joint_names[joint_index],
                    "signed_value": float(
                        acceleration[env_index, joint_index].item()
                    ),
                }
                self.validated_fd_acceleration_sample_count += 1
                if bool(acceleration_valid_mask.all()):
                    self.full46_validated_fd_acceleration_sample_count += 1
                if (
                    self.maximum_acceleration_rad_s2 is None
                    or acceleration_magnitude > self.maximum_acceleration_rad_s2
                ):
                    self.maximum_acceleration_rad_s2 = acceleration_magnitude
                    self.maximum_acceleration_event = acceleration_event

        full46_acceleration_valid = bool(
            acceleration_valid_mask is not None and acceleration_valid_mask.all()
        )
        step_velocity_safe = (
            velocity_magnitude <= G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S
        )
        step_acceleration_safe = False
        if full46_acceleration_valid and acceleration is not None:
            step_acceleration_safe = bool(
                torch.max(torch.abs(acceleration)).item()
                <= G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
            )
        step_settled = (
            velocity_magnitude <= G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S
        )
        if (
            full46_acceleration_valid
            and step_velocity_safe
            and step_acceleration_safe
            and step_settled
        ):
            self.consecutive_settled_full46_samples += 1
        else:
            self.consecutive_settled_full46_samples = 0

        self.previous_position.copy_(position)
        if self.previous_velocity is None:
            self.previous_velocity = velocity.detach().clone()
        else:
            self.previous_velocity.copy_(velocity)
        self.sample_count += 1

    @property
    def settled_for_cache(self) -> bool:
        """Whether the latest consecutive window can authorize cache capture."""

        return (
            self.full46_validated_fd_acceleration_sample_count > 0
            and self.consecutive_settled_full46_samples
            >= G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES
        )

    def result(self) -> dict[str, object]:
        velocity_pass = (
            self.maximum_velocity_rad_s
            <= G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S
        )
        acceleration_pass = bool(
            self.maximum_acceleration_rad_s2 is not None
            and self.maximum_acceleration_rad_s2
            <= G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
            and self.full46_validated_fd_acceleration_sample_count > 0
        )
        settled_pass = (
            self.consecutive_settled_full46_samples
            >= G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES
        )
        first_acceleration_resolved = bool(
            self.initial_acceleration_valid_mask.all()
        )
        unresolved_names = (
            []
            if first_acceleration_resolved
            else [
                self.joint_names[index]
                for index in range(len(self.joint_names))
                if not bool(
                    self.initial_acceleration_valid_mask[:, index].all().item()
                )
            ]
        )
        return {
            "schema": G2_GRIPPER_RESET_CONTRACT_SCHEMA,
            "joint_count": len(self.joint_names),
            "sample_count": self.sample_count,
            "first_post_reset_46dof_fd_sample_included": True,
            "first_post_reset_position_interval_included": True,
            "first_post_reset_acceleration_resolved": first_acceleration_resolved,
            "first_post_reset_acceleration_status": (
                "VALIDATED_INITIAL_VELOCITY"
                if first_acceleration_resolved
                else "UNRESOLVED_NO_PRIOR_POSITION_INTERVAL"
            ),
            "first_post_reset_unresolved_acceleration_joint_names": unresolved_names,
            "first_interval_maximum_position_delta_rad": (
                self.first_interval_maximum_position_delta_rad
            ),
            "first_interval_maximum_fd_velocity_rad_s": (
                self.first_interval_maximum_fd_velocity_rad_s
            ),
            "first_interval_maximum_velocity_event": (
                self.first_interval_maximum_velocity_event
            ),
            "validated_fd_acceleration_sample_count": (
                self.validated_fd_acceleration_sample_count
            ),
            "full46_validated_fd_acceleration_sample_count": (
                self.full46_validated_fd_acceleration_sample_count
            ),
            "consecutive_settled_full46_samples": (
                self.consecutive_settled_full46_samples
            ),
            "minimum_consecutive_settled_full46_samples": (
                G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES
            ),
            "settled_maximum_fd_velocity_rad_s": (
                G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S
            ),
            "maximum_fd_velocity_rad_s": self.maximum_velocity_rad_s,
            "maximum_fd_acceleration_rad_s2": self.maximum_acceleration_rad_s2,
            "maximum_velocity_event": self.maximum_velocity_event,
            "maximum_acceleration_event": self.maximum_acceleration_event,
            "maximum_allowed_fd_velocity_rad_s": (
                G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S
            ),
            "maximum_allowed_fd_acceleration_rad_s2": (
                G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
            ),
            "pass": bool(
                velocity_pass
                and acceleration_pass
                and settled_pass
                and self.sample_count >= 2
            ),
        }


def shared_gripper_reset_contract_metadata() -> dict[str, object]:
    return {
        "schema": G2_GRIPPER_RESET_CONTRACT_SCHEMA,
        "spawn_reset_authority": (
            "COMPLETE_CONFIGURED_46DOF_SEED_REQUIRING_LIVE_SETTLE_ATTESTATION"
        ),
        "configured_seed_is_live_stability_evidence": False,
        "four_bar_initial_state_audit": four_bar_initial_state_contract_metadata(),
        "right_hand_policy": G2_RIGHT_HAND_RESET_POLICY,
        "right_hand_cached_joint_names": list(G2_RIGHT_HAND_JOINT_NAMES),
        "independently_targeted_right_hand_joint_names": list(
            G2_RIGHT_HAND_INDEPENDENT_TARGET_JOINT_NAMES
        ),
        "passive_or_mimic_target_write_forbidden": list(
            G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES
        ),
        "right_hand_passive_or_mimic_joint_names": list(
            G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES
        ),
        "left_hand_passive_or_mimic_joint_names": list(
            G2_LEFT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES
        ),
        "usd_override_authored_passive_position_rad": 0.0,
        "configured_seed_passive_positions_match_usd_authored_zero": False,
        "measured_full8_cache_is_articulation_state_not_drive_target": True,
        "left_hand_policy": G2_LEFT_HAND_RESET_POLICY,
        "left_hand_joint_names": list(G2_LEFT_HAND_JOINT_NAMES),
        "left_hand_open_commanded": False,
        "hidden_per_environment_physics_step": False,
        "maximum_fd_velocity_rad_s": G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S,
        "maximum_fd_acceleration_rad_s2": (
            G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
        ),
        "settled_maximum_fd_velocity_rad_s": (
            G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S
        ),
        "first_interval_fd_velocity_is_gated": True,
        "first_interval_acceleration_without_prior_velocity": "UNRESOLVED",
        "first_valid_full46_acceleration_source": "Q0_Q1_Q2_SECOND_DIFFERENCE",
        "minimum_consecutive_settled_full46_samples": (
            G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES
        ),
        "right_master_open_target_rad": G2_RIGHT_MASTER_OPEN_TARGET_RAD,
        "minimum_open_settle_policy_steps": (
            G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
        ),
        "maximum_open_settle_policy_steps": (
            G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS
        ),
        "open_settle_tolerance_rad": (
            G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD
        ),
        "passive_limit_aware_open_retiming": {
            "enabled": True,
            "passive_coordinates_are_targeted": False,
            "velocity_source": "PER_PHYSICS_STEP_POSITION_FINITE_DIFFERENCE",
            "reaction_steps": G2_PASSIVE_LIMIT_GUARD_REACTION_STEPS,
            "state_machine": (
                "TRACK_BRAKE_SETTLE_CREEP_POST_STOP_POST_RELEASE_CONSERVATIVE"
            ),
            "braking_authority": "MASTER_TARGET_ACCELERATION",
            "braking_model": (
                "PASSIVE_DISTANCE_DURING_MASTER_TARGET_STOP"
            ),
            "governor_command_braking_acceleration_rad_s2": (
                G2_PASSIVE_LIMIT_GOVERNOR_BRAKE_ACCELERATION_RAD_S2
            ),
            "transmission_uncertainty_multiplier": (
                G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER
            ),
            "transmission_uncertainty_evidence": (
                "LIVE_INCOMING_0.0420268625_OVER_CREEP_BUDGET_0.015_"
                "EQUALS_2.80179_ROUNDED_UP"
            ),
            "transmission_uncertainty_application": (
                "CREEP_PREDICTED_BRANCH_ONLY_MAX_MEASURED_3X_PREDICTED"
            ),
            "track_prediction_uncertainty_multiplier": 1.0,
            "transmission_ratio_learning": (
                "CLEAN_TRACK_MONOTONIC_PER_COORDINATE_MAX_INCLUDING_LATE_"
                "SEQUENTIAL_MOTION"
            ),
            "risk_latch": "PER_CAP_MULTI_HOT_ORDER_EQUIVARIANT",
            "settle_release_authority": (
                "TWO_SAMPLE_NONRESTRICTIVE_PROOF_AT_NEXT_FULL_TRACK_CAP"
            ),
            "creep_existing_risk_authority": (
                "CAUSAL_ACTIVE_ENVELOPE_CAP_AND_MEASURED_LANDING_OVERSPEED_"
                "BRAKE_ONLY_NEW_SEQUENTIAL_RISK"
            ),
            "landed_latch": (
                "CUMULATIVE_DIRECTION_SPECIFIC_CLEAR_ON_REVERSAL_RESET_"
                "ENVELOPE_SUPPRESSION_ONLY_WITHIN_STOP_BAND"
            ),
            "position_target_generator": (
                "STOPPING_DISTANCE_AWARE_MONOTONIC_NO_OVERSHOOT"
            ),
            "master_target_braking_acceleration_rad_s2": (
                G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
            ),
            "settle_speed_rad_s": (
                G2_PASSIVE_LIMIT_GOVERNOR_SETTLE_SPEED_RAD_S
            ),
            "settle_consecutive_samples": (
                G2_PASSIVE_LIMIT_GOVERNOR_STATIONARY_SAMPLES
            ),
            "stop_band_hysteresis_rad": (
                G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD
            ),
            "stop_band_hysteresis_evidence": (
                "LANDING6_LIVE_STOP_BAND_MAX_0.000154492_RAD_ROUNDED_UP_"
                "PRESERVE_CREEP_ENVELOPE_NOT_LANDED_AUTHORITY"
            ),
            "creep_passive_speed_budget_rad_s_at_2ms": 0.015,
            "maximum_landing_speed_rad_s_at_2ms": 0.020,
            "post_stop_latch_clear_events": ["DIRECTION_REVERSAL", "RESET"],
            "hard_velocity_limit_rad_s": G2_RESET_MAXIMUM_FD_VELOCITY_RAD_S,
            "hard_acceleration_limit_rad_s2": (
                G2_RESET_MAXIMUM_FD_ACCELERATION_RAD_S2
            ),
            "joint_limits_changed": False,
            "drive_gain_or_effort_changed": False,
        },
    }


__all__ = [name for name in globals() if name.startswith("G2_")]
__all__.extend(
    (
        "FullArticulationFDSafetyAudit",
        "PassiveLimitMasterGovernor",
        "acceleration_limited_position_target_step",
        "capture_measured_right_hand_cache",
        "install_measured_right_hand_cache",
        "passive_limit_aware_master_speed_cap",
        "resolve_full_right_hand_indices",
        "shared_gripper_reset_contract_metadata",
        "validate_complete_stable_hand_configuration",
    )
)
