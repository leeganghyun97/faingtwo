# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""One-shot Isaac adapter for a bounded keyboard-v3 demonstration.

The caller must finish the cuRobo handoff before entering this function.  The
adapter samples the operator once per 50-Hz control epoch and delegates the
sole action consumption to the existing ``preflight._consume_once`` route.
There is no retry, autonomous close, residual policy, or direct joint command.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np

from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from ..g2_redundancy_action import G2RedundancySe3Keyboard
from ..g2_camera_timing import camera_capture_time_and_age
from ..g2_quaternion import isaaclab_native_quaternion_order
from ..g2_quaternion import quaternion_native_to_xyzw
from ..g2_rebuild.sensor_packet import (
    world_quaternions_to_root,
    world_points_to_root_from_native_quaternion,
)
from .action_interface import (
    AbstractGripperIntent,
    HighLevelPolicyAction,
)
from .keyboard_v3_curobo_planner import KeyboardV3CuroboPlan
from .keyboard_v3_operator_ui import (
    KeyboardV3OperatorUI,
    create_omni_result_panel,
)
from .keyboard_v3_runtime import (
    KeyboardV3EpisodeRecorder,
    KeyboardV3Row,
    PlannerStartReceipt,
)
from .keyboard_v3_collection_contract import (
    COLLECTION_TARGETS_MM,
    format_terminal_status,
    preclose_geometry_metrics,
)
from geniesim.rl.sac.keyboard_v3_dataset import (
    KEYBOARD_V3_CAMERA_NAMES,
    KEYBOARD_V3_FILE_SCHEMA,
    KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES,
    KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES,
    KEYBOARD_V3_OPERATOR_VIEW_NAMES,
    KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES,
)
from geniesim.rl.sac.keyboard_grasp_contract import (
    canonical_keyboard_grasp_contract,
    classify_local_grasp_phase,
)
from .keyboard_v3_operator_views import KeyboardV3OperatorViews
from .keyboard_v3_terminal_input import (
    DEFAULT_MOTION_MIN_INTERVAL_S,
    DEFAULT_TRANSLATION_STEP_M,
    KeyboardV3TerminalInput,
)
from .omnipicker_product_contract import OMNIPICKER_PRODUCT_CONTRACT
from .rgbd_logging_contract import (
    CAMERA_TIMESTAMP_SOURCE,
    DEPTH_RAW_UNIT,
    DEPTH_TO_METER_SCALE,
    RGBD_EVIDENCE_SCHEMA,
)
from .keyboard_v3_action_adapter import (
    KeyboardV3ActionAdapterError,
    adapt_legacy_keyboard_physical_8d,
)


KEYBOARD_V3_LIVE_DRIVER_SCHEMA = "g2_keyboard_v3_one_shot_live_qualification_v1"
MAXIMUM_OPERATOR_TRANSLATION_M = 0.0045
MEASURED_DIRECTION_TOLERANCE_M = 5.0e-5


class KeyboardV3LiveDriverError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise KeyboardV3LiveDriverError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measured_translation_direction_receipt(
    command_xyz_root_m: np.ndarray,
    measured_delta_xyz_root_m: np.ndarray,
    *,
    tolerance_m: float = MEASURED_DIRECTION_TOLERANCE_M,
) -> dict[str, Any]:
    """Classify one post-step EE response without hiding controller latency.

    A response inside the tolerance is neutral because a position controller
    may need more than one 20-ms policy epoch to produce measurable motion.
    A response beyond the tolerance in the opposite half-space is a real sign
    inversion and fails the episode.  In particular, terminal ``E`` is a pure
    robot-root ``-Z`` request and must never produce significant ``+Z`` EE
    motion.
    """

    command = np.asarray(command_xyz_root_m, dtype=np.float64)
    measured = np.asarray(measured_delta_xyz_root_m, dtype=np.float64)
    if command.shape != (3,) or measured.shape != (3,):
        raise KeyboardV3LiveDriverError("TRANSLATION_DIRECTION_VECTOR_SHAPE_INVALID")
    if not np.isfinite(command).all() or not np.isfinite(measured).all():
        raise KeyboardV3LiveDriverError("TRANSLATION_DIRECTION_VECTOR_NONFINITE")
    if not math.isfinite(float(tolerance_m)) or float(tolerance_m) <= 0.0:
        raise KeyboardV3LiveDriverError("TRANSLATION_DIRECTION_TOLERANCE_INVALID")
    command_norm = float(np.linalg.norm(command))
    measured_norm = float(np.linalg.norm(measured))
    dot = float(np.dot(command, measured))
    response_observable = measured_norm > float(tolerance_m)
    inverted = bool(command_norm > 1.0e-12 and response_observable and dot < 0.0)
    e_key_command = bool(
        command[2] < -1.0e-12
        and abs(float(command[0])) <= 1.0e-12
        and abs(float(command[1])) <= 1.0e-12
    )
    return {
        "command_xyz_root_m": command.tolist(),
        "measured_delta_xyz_root_m": measured.tolist(),
        "dot_m2": dot,
        "response_observable": response_observable,
        "direction_inverted": inverted,
        "e_key_negative_root_z_command": e_key_command,
        "e_key_measured_down": bool(
            e_key_command and measured[2] < -float(tolerance_m)
        ),
        "e_key_measured_up_inversion": bool(
            e_key_command and measured[2] > float(tolerance_m)
        ),
    }


def _close_operator_panel(panel: Any) -> str:
    """Release V3 UI objects before environment/Kit shutdown.

    This is deterministic V3 cleanup, not a waiver for the independently
    reproduced Isaac/Kit native-finalization SIGSEGV.
    """

    try:
        if hasattr(panel, "visible"):
            panel.visible = False
        destroy = getattr(panel, "destroy", None)
        if callable(destroy):
            destroy()
        return "PASS"
    except Exception as error:
        return f"FAIL:{type(error).__name__}:{error}"


def _tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    return torch.from_dlpack(value)


def _ee_root_pose(env: Any, p0a: Any) -> tuple[np.ndarray, np.ndarray]:
    position, quaternion = p0a._world_ee_pose_to_root(
        env.scene["robot"], env.scene["ee_frame"]
    )[:2]
    return (
        _tensor(position)[0].detach().to("cpu").numpy().astype(np.float64),
        _tensor(quaternion)[0].detach().to("cpu").numpy().astype(np.float64),
    )


def _cube_root_position(env: Any) -> np.ndarray:
    robot_data = env.scene["robot"].data
    root_position_world_m = _tensor(robot_data.root_pos_w)
    root_quaternion_native = _tensor(robot_data.root_quat_w)
    cube_position_world_m = _tensor(env.scene["object"].data.root_pos_w)
    cube_position_root_m = world_points_to_root_from_native_quaternion(
        cube_position_world_m,
        root_position_world_m,
        root_quaternion_native,
        native_order=isaaclab_native_quaternion_order(),
    )
    return (
        cube_position_root_m[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )


def _cube_root_pose_xyzw(env: Any) -> np.ndarray:
    """Return the raw cube rigid-body pose in robot-root metres/XYZW."""

    robot_data = env.scene["robot"].data
    cube_data = env.scene["object"].data
    root_position_world_m = _tensor(robot_data.root_pos_w)
    root_quaternion_native = _tensor(robot_data.root_quat_w)
    cube_position_root_m = world_points_to_root_from_native_quaternion(
        _tensor(cube_data.root_pos_w),
        root_position_world_m,
        root_quaternion_native,
        native_order=isaaclab_native_quaternion_order(),
    )
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        root_quaternion_native, isaaclab_native_quaternion_order()
    )
    cube_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(cube_data.root_quat_w), isaaclab_native_quaternion_order()
    )
    cube_quaternion_root_xyzw = world_quaternions_to_root(
        cube_quaternion_xyzw, root_quaternion_xyzw
    )
    return (
        __import__("torch")
        .cat((cube_position_root_m, cube_quaternion_root_xyzw), dim=-1)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )


def _body_pose_root_xyzw(robot: Any, body_name: str) -> np.ndarray:
    """Persist raw link-frame pose; this is not a calibrated pad surface."""

    names = tuple(robot.body_names)
    if body_name not in names:
        raise KeyboardV3LiveDriverError(f"BODY_NOT_FOUND:{body_name}")
    index = names.index(body_name)
    root_position = _tensor(robot.data.root_pos_w)
    root_quaternion_native = _tensor(robot.data.root_quat_w)
    position = world_points_to_root_from_native_quaternion(
        _tensor(robot.data.body_pos_w)[:, index],
        root_position,
        root_quaternion_native,
        native_order=isaaclab_native_quaternion_order(),
    )
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        root_quaternion_native, isaaclab_native_quaternion_order()
    )
    body_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(robot.data.body_quat_w)[:, index],
        isaaclab_native_quaternion_order(),
    )
    quaternion = world_quaternions_to_root(
        body_quaternion_xyzw, root_quaternion_xyzw
    )
    return __import__("torch").cat((position, quaternion), dim=-1)[0].detach().to("cpu").numpy().astype(np.float64)


def _capture_camera_rgbd(
    env: Any,
    *,
    camera_name: str,
    prior_frame: int | None,
    prior_sensor_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    scene_name = f"{camera_name}_camera"
    camera = env.scene[scene_name]
    output = camera.data.output
    if "rgb" not in output or "distance_to_image_plane" not in output:
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_RGBD_STREAM_MISSING"
        )
    rgb = _tensor(output["rgb"])[..., :3]
    depth = _tensor(output["distance_to_image_plane"])
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    if tuple(rgb.shape) != (1, 192, 256, 3):
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_RGB_SHAPE_MISMATCH"
        )
    if tuple(depth.shape) != (1, 192, 256, 1):
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_DEPTH_SHAPE_MISMATCH"
        )
    raw_depth = depth.to(dtype=__import__("torch").float32)
    if bool((__import__("torch").isneginf(raw_depth) | (raw_depth < 0.0)).any().item()):
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_DEPTH_CORRUPT"
        )
    source_valid = __import__("torch").isfinite(raw_depth) & (raw_depth >= 0.0)
    frame_tensor = _tensor(camera.frame).reshape(-1)
    if frame_tensor.numel() != 1:
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_CAMERA_FRAME_CARDINALITY_MISMATCH"
        )
    frame = int(frame_tensor[0].item())
    try:
        captured, _age = camera_capture_time_and_age(camera)
    except (RuntimeError, TypeError, ValueError) as error:
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_ACQUISITION_TIMESTAMP_UNAVAILABLE"
        ) from error
    captured_tensor = _tensor(captured).reshape(-1)
    if captured_tensor.numel() != 1:
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_CAMERA_TIMESTAMP_CARDINALITY_MISMATCH"
        )
    sensor_time_s = float(captured_tensor[0].item())
    if not math.isfinite(sensor_time_s) or sensor_time_s < 0.0:
        raise KeyboardV3LiveDriverError(
            f"{camera_name.upper()}_ACQUISITION_TIMESTAMP_INVALID"
        )
    if prior_frame is not None:
        if frame == prior_frame and sensor_time_s != float(prior_sensor_time_s):
            raise KeyboardV3LiveDriverError(
                f"{camera_name.upper()}_REUSED_FRAME_TIMESTAMP_CHANGED"
            )
        if frame != prior_frame and sensor_time_s <= float(prior_sensor_time_s):
            raise KeyboardV3LiveDriverError(
                f"{camera_name.upper()}_NEW_FRAME_TIMESTAMP_NOT_ADVANCED"
            )
    return (
        rgb[0].detach().to("cpu").numpy().astype(np.uint8, copy=True),
        raw_depth[0].detach().to("cpu").numpy().astype(np.float32, copy=True),
        source_valid[0].detach().to("cpu").numpy().astype(np.bool_, copy=True),
        frame,
        sensor_time_s,
    )


def _camera_pose_root_m_xyzw(env: Any, *, camera_name: str) -> np.ndarray:
    """Read the source camera pose and express it in robot_root, XYZW."""

    camera = env.scene[f"{camera_name}_camera"]
    robot = env.scene["robot"]
    position_root = world_points_to_root_from_native_quaternion(
        _tensor(camera.data.pos_w),
        _tensor(robot.data.root_pos_w),
        _tensor(robot.data.root_quat_w),
        native_order=isaaclab_native_quaternion_order(),
    )
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(robot.data.root_quat_w), isaaclab_native_quaternion_order()
    )
    # Isaac Lab CameraData documents quat_w_world as XYZW already.
    camera_quaternion_root_xyzw = world_quaternions_to_root(
        _tensor(camera.data.quat_w_world), root_quaternion_xyzw
    )
    pose = __import__("torch").cat(
        (position_root, camera_quaternion_root_xyzw), dim=-1
    )
    return pose[0].detach().to("cpu").numpy().astype(np.float32, copy=True)


def _camera_calibration_metadata(env: Any) -> dict[str, dict[str, Any]]:
    """Snapshot source-owned intrinsics and authored frame bindings."""

    result: dict[str, dict[str, Any]] = {}
    for camera_name in KEYBOARD_V3_CAMERA_NAMES:
        camera = env.scene[f"{camera_name}_camera"]
        intrinsics = (
            _tensor(camera.data.intrinsic_matrices)[0]
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        )
        result[camera_name] = {
            "camera_name": camera_name,
            "source_scene_key": f"{camera_name}_camera",
            "source_frame_id": str(camera.cfg.prim_path),
            "intrinsics_3x3": intrinsics.tolist(),
            "image_width": int(camera.cfg.width),
            "image_height": int(camera.cfg.height),
            "extrinsic_reference_frame": "robot_root",
            "pose_convention": "position_m+quaternion_xyzw",
            "timestamp_source": CAMERA_TIMESTAMP_SOURCE,
        }
    return result


def run_keyboard_v3_one_shot(
    *,
    app: Any,
    env: Any,
    p0a: Any,
    preflight: Any,
    task_mdp: Any,
    counter: Any,
    deferred_port: Any,
    latch: Any,
    plan: Any,
    collection_root: Path,
    episode_id: str,
    report_path: Path,
    maximum_operator_steps: int,
    source_freeze_before: Mapping[str, Any],
    source_freeze_provider: Callable[[], Mapping[str, Any]],
    selected_asset_path: Path,
    selected_asset_sha256: str,
    planner_path_records: list[Mapping[str, Any]],
    planner_maximum_forbidden_contact_force_n: float,
    planner_final_error_m: float,
    planner_settled: bool,
    terminal_keyboard: bool = False,
    terminal_translation_step_m: float = DEFAULT_TRANSLATION_STEP_M,
    terminal_motion_min_interval_s: float = DEFAULT_MOTION_MIN_INTERVAL_S,
    terminal_smoothing_steps: int = 1,
    baseline_metadata: Mapping[str, Any] | None = None,
    direct_init_receipt: Mapping[str, Any] | None = None,
) -> int:
    """Run the exactly-once human part after a completed cuRobo handoff."""

    import h5py
    import torch
    from isaaclab.devices import Se3KeyboardCfg
    from geniesim.rl.sac.keyboard_v3_dataset import read_keyboard_v3_episode
    contract = canonical_keyboard_grasp_contract()

    plan = plan.validated()
    root = collection_root.resolve()
    manifest = root / "COLLECTION_MANIFEST.json"
    canonical_dir = root / "episodes"
    rejected_dir = root / "rejected_episodes"
    receipt_dir = root / "result_receipts"
    for required in (manifest, canonical_dir, rejected_dir, receipt_dir):
        if not required.exists():
            raise KeyboardV3LiveDriverError(f"COLLECTION_ROOT_INCOMPLETE:{required}")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        manifest_payload.get("schema")
        != "g2_keyboard_v3_collection_manifest_v6_boolean_contact_primary"
        or manifest_payload.get("hdf5_file_schema") != KEYBOARD_V3_FILE_SCHEMA
        or manifest_payload.get("gru_actor_camera_names")
        != list(KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES)
        or manifest_payload.get("recorded_non_actor_camera_names")
        != list(KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES)
        or manifest_payload.get("outcome_label_authority")
        != "RAW_TELEMETRY_OFFLINE_DERIVATION_ONLY"
        or manifest_payload.get("primary_privileged_learning_signal")
        != "BOOLEAN_CONTACT"
        or manifest_payload.get("force_learning_role")
        != "RAW_DIAGNOSTIC_SAFETY_AUXILIARY_ONLY"
        or manifest_payload.get("force_required_for_contact") is not False
        or manifest_payload.get("force_required_for_stable") is not False
        or manifest_payload.get("required_outcome_fields")
        != sorted(KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES)
        or manifest_payload.get("future_outcome_horizon_control_steps") != 64
        or manifest_payload.get("omnipicker_maximum_gripping_force_n")
        != OMNIPICKER_PRODUCT_CONTRACT.maximum_gripping_force_n
        or manifest_payload.get("omnipicker_physical_pad_touch_sensor_available")
        is not False
        or manifest_payload.get("sim_contact_telemetry_role")
        != OMNIPICKER_PRODUCT_CONTRACT.simulator_contact_role
        or manifest_payload.get("student_contact_input_count") != 0
        or manifest_payload.get("camera_evidence_contract", {}).get("schema")
        != RGBD_EVIDENCE_SCHEMA
        or manifest_payload.get("camera_evidence_contract", {}).get(
            "camera_names"
        )
        != list(KEYBOARD_V3_CAMERA_NAMES)
        or manifest_payload.get("camera_evidence_contract", {}).get(
            "depth_raw_unit"
        )
        != DEPTH_RAW_UNIT
        or manifest_payload.get("camera_evidence_contract", {}).get(
            "depth_to_meter_scale"
        )
        != DEPTH_TO_METER_SCALE
        or manifest_payload.get("camera_evidence_contract", {}).get(
            "current_gru_input_changed"
        )
        is not False
    ):
        raise KeyboardV3LiveDriverError("COLLECTION_MANIFEST_SCHEMA_OR_OUTCOME_CONTRACT_MISMATCH")
    if report_path.exists():
        raise KeyboardV3LiveDriverError("LIVE_REPORT_ALREADY_EXISTS")
    if maximum_operator_steps <= 0:
        raise KeyboardV3LiveDriverError("MAXIMUM_OPERATOR_STEPS_INVALID")
    if not 0.0 < float(terminal_translation_step_m) <= MAXIMUM_OPERATOR_TRANSLATION_M:
        raise KeyboardV3LiveDriverError("TERMINAL_TRANSLATION_STEP_INVALID")
    if not 0.0 <= float(terminal_motion_min_interval_s) <= 1.0:
        raise KeyboardV3LiveDriverError("TERMINAL_MOTION_INTERVAL_INVALID")
    if isinstance(terminal_smoothing_steps, bool) or not 1 <= int(
        terminal_smoothing_steps
    ) <= 10:
        raise KeyboardV3LiveDriverError("TERMINAL_SMOOTHING_STEPS_INVALID")

    robot = env.scene["robot"]
    try:
        gripper_master_index = tuple(robot.joint_names).index(
            "idx81_gripper_r_outer_joint1"
        )
    except ValueError as error:
        raise KeyboardV3LiveDriverError(
            "CANDIDATE_A_RIGHT_GRIPPER_MASTER_NOT_FOUND"
        ) from error
    arm_term = env.action_manager.get_term("arm_action")
    arm_indices = torch.as_tensor(
        [int(value) for value in arm_term._joint_ids],
        device=env.device,
        dtype=torch.long,
    )
    initial_ee, initial_quaternion = _ee_root_pose(env, p0a)
    near_target = np.asarray(plan.near_grasp_pose_root_m_xyzw[:3], dtype=np.float64)
    measured_arm_q = (
        _tensor(robot.data.joint_pos)[0]
        .index_select(0, arm_indices)
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    emitted_target = (
        _tensor(arm_term.last_emitted_joint_position_target)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    controller_joint_target_error_rad = float(
        np.max(np.abs(measured_arm_q - emitted_target))
    )
    measured_handoff_error_m = float(np.linalg.norm(initial_ee - near_target))
    near_grasp_reached = bool(
        float(planner_final_error_m) <= 0.003
        and measured_handoff_error_m <= 0.003
        and planner_settled
    )
    no_forbidden_collision = bool(
        planner_maximum_forbidden_contact_force_n <= 1.0e-6
        and all(
            not bool(record.get("terminated", False))
            and not bool(record.get("truncated", False))
            and not record.get("active_termination_names", [])
            for record in planner_path_records
        )
    )
    handoff_task_metrics = preflight._task_metrics(env, task_mdp)
    no_object_contact_before_handoff = bool(
        float(handoff_task_metrics["inner_contact_force_n"]) <= 0.0
        and float(handoff_task_metrics["outer_contact_force_n"]) <= 0.0
        and not bool(handoff_task_metrics["bilateral_contact"])
        and not bool(handoff_task_metrics["ever_bilateral_contact"])
    )
    if not near_grasp_reached:
        raise KeyboardV3LiveDriverError("NEAR_GRASP_POSE_NOT_REACHED")
    if not no_forbidden_collision:
        raise KeyboardV3LiveDriverError("FORBIDDEN_COLLISION_BEFORE_HANDOFF")
    if not no_object_contact_before_handoff:
        raise KeyboardV3LiveDriverError("OBJECT_CONTACT_BEFORE_HANDOFF")

    planner_receipt = PlannerStartReceipt(
        nominal_grasp_pose_root_m_xyzw=plan.nominal_grasp_pose_root_m_xyzw,
        near_grasp_pose_root_m_xyzw=plan.near_grasp_pose_root_m_xyzw,
        approach_axis_root=(1.0, 0.0, 0.0),
        condition=plan.condition,
        nominal_grasp_q_rad=None,
        near_grasp_q_rad=tuple(float(value) for value in plan.q_rad[-1]),
    ).validated()
    recorder = KeyboardV3EpisodeRecorder(
        episode_id=episode_id,
        planner=planner_receipt,
        rgbd_hz=25,
        pad_surface_valid=False,
        pad_calibration_verified=False,
        baseline_metadata=baseline_metadata,
        camera_calibration_metadata=_camera_calibration_metadata(env),
    )
    ui = KeyboardV3OperatorUI(
        recorder,
        canonical_episode_directory=canonical_dir,
        rejected_episode_directory=rejected_dir,
        receipt_directory=receipt_dir,
    )
    terminal = None
    keyboard = None
    if not terminal_keyboard:
        keyboard = G2RedundancySe3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=MAXIMUM_OPERATOR_TRANSLATION_M,
                rot_sensitivity=0.0,
                sim_device="cpu",
                gripper_term=True,
            )
        )
        keyboard.reset()
        if keyboard.input_device_name.strip().lower() == "offscreen":
            raise KeyboardV3LiveDriverError("LIVE_GUI_KEYBOARD_UNAVAILABLE")
        ui.bind_keyboard_callbacks(keyboard)
    result_panel = create_omni_result_panel(ui)
    operator_views = KeyboardV3OperatorViews()
    operator_views.create()
    operator_view_receipt = operator_views.receipt()
    # Enter cbreak mode only after fallible viewport setup.  Thereafter the
    # single finally block owns restoration on normal exit, error and signal.
    if terminal_keyboard:
        terminal = KeyboardV3TerminalInput(
            translation_step_m=terminal_translation_step_m,
            motion_min_interval_s=terminal_motion_min_interval_s,
            smoothing_steps=terminal_smoothing_steps,
        )

    # No keyboard object exists before this point; therefore the pre-handoff
    # keyboard submission count is structurally zero.
    keyboard_action_submission_before_handoff = 0
    recorder.start_recording()
    print("KEYBOARD_V3_RECORDING_STARTED", flush=True)
    print(
        "Move: W/S=root X, A/D=root Y, Q=+Z wrist up, E=-Z wrist down "
        "| K=CLOSE once (repeat is idempotent)",
        flush=True,
    )
    print(
        f"Motion: {terminal_translation_step_m * 1000.0:.3f} mm/key, "
        f"minimum interval {terminal_motion_min_interval_s * 1000.0:.1f} ms",
        flush=True,
    )
    print("Terminal: R=end/reset request, X/Esc=end/quit request", flush=True)
    print("ENTER ends the episode and opens mandatory result selection", flush=True)

    action_submission_count = 0
    process_action_count = 0
    open_command_count = 0
    close_command_count = 0
    close_event_count = 0
    nonzero_xyz_row_count = 0
    action_bound_violation_count = 0
    maximum_action_norm_m = 0.0
    previous_operator_closed = False
    camera_names = KEYBOARD_V3_CAMERA_NAMES
    last_camera_frames: dict[str, int | None] = {
        name: None for name in camera_names
    }
    last_camera_times_s = {name: 0.0 for name in camera_names}
    latest_camera_capture: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]
    ] = {}
    camera_capture_period_steps = 50 // recorder.rgbd_hz
    runtime_error: str | None = None
    session_stop_requested = False
    # A terminal collection session is operator-owned.  Do not end it merely
    # because the operator pauses before assigning the result.  Viewport-only
    # one-shot qualification retains a bounded timeout.
    selection_timeout_s: float | None = None if terminal_keyboard else 300.0
    previous_cube_position_root_m = _cube_root_position(env)
    initial_inner_pose = _body_pose_root_xyzw(robot, "gripper_r_inner_link4")
    initial_outer_pose = _body_pose_root_xyzw(robot, "gripper_r_outer_link4")
    previous_pad_center_root_m = 0.5 * (
        initial_inner_pose[:3] + initial_outer_pose[:3]
    )
    previous_display_ee_position_root_m: np.ndarray | None = None
    e_key_command_count = 0
    e_key_measured_down_count = 0
    e_key_measured_up_inversion_count = 0
    measured_direction_inversion_count = 0
    e_key_measured_delta_z_m: list[float] = []
    target_residual_mm = int(round(plan.condition.backoff_m * 1000.0))
    # Dataset/control cadence remains 50 Hz.  Terminal rendering is diagnostic
    # only; flushing a full status line at 50 Hz needlessly stalls interactive
    # collection, especially over remote terminals.
    terminal_status_print_interval_steps = 5
    if target_residual_mm not in COLLECTION_TARGETS_MM:
        raise KeyboardV3LiveDriverError("COLLECTION_TARGET_OUTSIDE_16_22_MM")

    try:
        for control_step in range(maximum_operator_steps):
            if not ui.recording_enabled:
                break
            terminal_key: str | None = None
            if terminal is not None:
                polled = terminal.poll()
                terminal_key = polled.key
                if polled.event == "ENTER":
                    ui.handle_key("ENTER")
                    break
                if polled.event and polled.event.startswith("RESULT_"):
                    # A numeric result key cannot label a live recording.
                    continue
                if polled.event in ("RESET_REQUEST", "QUIT_REQUEST"):
                    session_stop_requested = polled.event == "QUIT_REQUEST"
                    ui.handle_key("ENTER")
                    break
                operator_action_4d = np.asarray(
                    polled.action_4d_metric_root_m, dtype=np.float64
                )
            else:
                assert keyboard is not None
                physical, _legacy_normalized_unused = keyboard.sample()
                if not ui.recording_enabled:
                    break
                physical_np = physical.detach().to("cpu").numpy().astype(np.float64)
                try:
                    adapted = adapt_legacy_keyboard_physical_8d(physical_np)
                except KeyboardV3ActionAdapterError as error:
                    raise KeyboardV3LiveDriverError(
                        f"KEYBOARD_LEGACY_ACTION_REJECTED:{error}"
                    ) from error
                operator_action_4d = adapted.action_4d
            xyz = operator_action_4d[0:3]
            if terminal is not None and int(np.count_nonzero(np.abs(xyz) > 1.0e-12)) > 1:
                raise KeyboardV3LiveDriverError(
                    "TERMINAL_CROSS_AXIS_COMMAND_FORBIDDEN"
                )
            norm = float(np.linalg.norm(xyz))
            maximum_action_norm_m = max(maximum_action_norm_m, norm)
            if norm > MAXIMUM_OPERATOR_TRANSLATION_M + 1.0e-12:
                action_bound_violation_count += 1
                raise KeyboardV3LiveDriverError(
                    f"OPERATOR_ACTION_BOUND_VIOLATION:{norm}"
                )
            g = float(operator_action_4d[3])
            operator_closed = bool(g == 1.0)
            close_edge_now = bool(operator_closed and not previous_operator_closed)
            if close_edge_now:
                close_event_count += 1
            if not operator_closed and previous_operator_closed:
                raise KeyboardV3LiveDriverError("REOPEN_AFTER_CLOSE_FORBIDDEN")
            previous_operator_closed = operator_closed
            if norm > 1.0e-9:
                nonzero_xyz_row_count += 1
            if operator_closed:
                close_command_count += 1
            else:
                open_command_count += 1

            control_time_s = float(control_step) * 0.020
            camera_capture: dict[
                str, tuple[np.ndarray, np.ndarray, np.ndarray, float]
            ] = {}
            if control_step % camera_capture_period_steps == 0:
                for camera_name in camera_names:
                    rgb, depth, depth_valid, frame, sensor_time_s = (
                        _capture_camera_rgbd(
                            env,
                            camera_name=camera_name,
                            prior_frame=last_camera_frames[camera_name],
                            prior_sensor_time_s=last_camera_times_s[camera_name],
                        )
                    )
                    last_camera_frames[camera_name] = frame
                    last_camera_times_s[camera_name] = sensor_time_s
                    latest_camera_capture[camera_name] = (
                        rgb,
                        depth,
                        depth_valid,
                        sensor_time_s,
                        _camera_pose_root_m_xyzw(
                            env, camera_name=camera_name
                        ),
                    )
            if set(latest_camera_capture) != set(camera_names):
                raise KeyboardV3LiveDriverError(
                    "RECORDED_CAMERA_SYNCHRONIZED_CAPTURE_INCOMPLETE"
                )
            camera_capture.update(latest_camera_capture)
            ee_position, ee_quaternion = _ee_root_pose(env, p0a)
            cube_position_before_action = _cube_root_position(env)
            display_metrics = preclose_geometry_metrics(
                ee_position_root_m=ee_position,
                previous_ee_position_root_m=previous_display_ee_position_root_m,
                cube_center_root_m=cube_position_before_action,
                nominal_grasp_pose_root_m_xyzw=plan.nominal_grasp_pose_root_m_xyzw,
                approach_axis_root=planner_receipt.approach_axis_root,
                action_xyz_root_m=xyz,
                control_dt_s=contract.policy_dt_s,
            )
            phase = classify_local_grasp_phase(
                display_metrics["current_nominal_grasp_residual_m"],
                gripper_closed=operator_closed,
            )
            if (
                control_step % terminal_status_print_interval_steps == 0
                or terminal_key is not None
                or close_edge_now
            ):
                print(
                    format_terminal_status(
                        episode_id=episode_id,
                        control_step=control_step,
                        target_residual_mm=target_residual_mm,
                        metrics=display_metrics,
                        action_xyz_root_m=xyz,
                        gripper_state="CLOSE" if operator_closed else "OPEN",
                        close_edge=close_edge_now,
                        close_state=operator_closed,
                        phase=phase,
                    ),
                    flush=True,
                )
            previous_display_ee_position_root_m = ee_position.copy()
            arm_q = (
                _tensor(robot.data.joint_pos)
                .index_select(1, arm_indices)[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            arm_qd = (
                _tensor(robot.data.joint_vel)
                .index_select(1, arm_indices)[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            normalized_xyz = tuple(
                float(value) / float(G2_TRANSLATION_ACTION_SCALE_M) for value in xyz
            )
            action = HighLevelPolicyAction.from_sequence((*normalized_xyz, g))
            packet, _, _derivation = p0a._build_authoritative_packet(
                high_level=action,
                batch_size=env.num_envs,
                device=env.device,
                latch=latch,
            )
            if tuple(float(value) for value in packet.values[3:7]) != (0.0, 0.0, 0.0, 0.0):
                raise KeyboardV3LiveDriverError("ORIENTATION_OR_ELBOW_ACTION_NONZERO")
            packet_xyz = np.asarray(
                [float(value) for value in packet.values[:3]], dtype=np.float64
            )
            if not np.allclose(
                packet_xyz,
                np.asarray(normalized_xyz, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise KeyboardV3LiveDriverError(
                    "KEYBOARD_TO_CONTROLLER_TRANSLATION_SIGN_OR_SCALE_MISMATCH"
                )
            outputs, receipt = preflight._consume_once(
                env=env,
                counter=counter,
                deferred_port=deferred_port,
                packet=packet,
                label=f"KEYBOARD_V3_OPERATOR_{control_step:04d}",
            )
            action_submission_count += 1
            process_action_count += int(receipt["action_manager_process_action_count"])
            if not bool(receipt["single_consumption"]):
                raise KeyboardV3LiveDriverError("ACTION_NOT_SINGLE_CONSUMPTION")
            latch.commit(packet.gripper_intent)
            _, _reward, terminated, truncated, _ = outputs
            measured_ee_after, _measured_quaternion_after = _ee_root_pose(env, p0a)
            direction_receipt = _measured_translation_direction_receipt(
                xyz,
                measured_ee_after - ee_position,
            )
            measured_direction_inversion_count += int(
                direction_receipt["direction_inverted"]
            )
            if bool(direction_receipt["e_key_negative_root_z_command"]):
                e_key_command_count += 1
                e_key_measured_down_count += int(
                    direction_receipt["e_key_measured_down"]
                )
                e_key_measured_up_inversion_count += int(
                    direction_receipt["e_key_measured_up_inversion"]
                )
                e_key_measured_delta_z_m.append(
                    float(direction_receipt["measured_delta_xyz_root_m"][2])
                )
            if bool(direction_receipt["direction_inverted"]):
                raise KeyboardV3LiveDriverError(
                    "MEASURED_EE_TRANSLATION_DIRECTION_INVERTED"
                )
            active = preflight._active_termination_names(env, terminated, truncated)
            evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
            peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
            forbidden_peak = float(
                max((max(values) for values in peaks.values()), default=0.0)
            )
            task_metrics = preflight._task_metrics(env, task_mdp)
            cube_position = _cube_root_position(env)
            cube_pose = _cube_root_pose_xyzw(env)
            cube_velocity = (
                cube_position - previous_cube_position_root_m
            ) / float(contract.policy_dt_s)
            previous_cube_position_root_m = cube_position.copy()
            inner_link_pose = _body_pose_root_xyzw(robot, "gripper_r_inner_link4")
            outer_link_pose = _body_pose_root_xyzw(robot, "gripper_r_outer_link4")
            pad_center_root_m = 0.5 * (
                inner_link_pose[:3] + outer_link_pose[:3]
            )
            pad_center_velocity = (
                pad_center_root_m - previous_pad_center_root_m
            ) / float(contract.policy_dt_s)
            previous_pad_center_root_m = pad_center_root_m.copy()
            pad_center_relative_to_cube_velocity = (
                pad_center_velocity - cube_velocity
            )
            terminated_now = bool(terminated.reshape(-1)[0].item()) or bool(
                truncated.reshape(-1)[0].item()
            ) or bool(active)
            # Consume source-owned boolean contact telemetry.  The collector
            # must never recreate learning labels from raw force.
            exact_inner = bool(task_metrics["inner_contact"])
            exact_outer = bool(task_metrics["outer_contact"])
            contact = bool(task_metrics["contact"])
            if contact != (exact_inner or exact_outer):
                raise KeyboardV3LiveDriverError("CONTACT_BOOLEAN_TELEMETRY_MISMATCH")
            if bool(task_metrics["bilateral_contact"]) != (exact_inner and exact_outer):
                raise KeyboardV3LiveDriverError("CONTACT_TELEMETRY_DERIVATION_MISMATCH")
            row = KeyboardV3Row(
                head_rgb=camera_capture["head"][0],
                head_depth_source_m=camera_capture["head"][1],
                head_depth_source_valid=camera_capture["head"][2],
                head_rgb_timestamp_s=camera_capture["head"][3],
                head_depth_timestamp_s=camera_capture["head"][3],
                head_frame_id=int(last_camera_frames["head"]),
                head_camera_pose_root_m_xyzw=camera_capture["head"][4],
                right_wrist_rgb=camera_capture["right_wrist"][0],
                right_wrist_depth_source_m=camera_capture["right_wrist"][1],
                right_wrist_depth_source_valid=camera_capture["right_wrist"][2],
                right_wrist_rgb_timestamp_s=camera_capture["right_wrist"][3],
                right_wrist_depth_timestamp_s=camera_capture["right_wrist"][3],
                right_wrist_frame_id=int(last_camera_frames["right_wrist"]),
                right_wrist_camera_pose_root_m_xyzw=camera_capture[
                    "right_wrist"
                ][4],
                ee_position_root_m=ee_position,
                ee_quat_root_xyzw=ee_quaternion,
                arm_q_rad=arm_q,
                arm_qd_rad_s=arm_qd,
                gripper_state=1.0 if latch.intent is AbstractGripperIntent.CLOSE else 0.0,
                action_4d=(*xyz.tolist(), g),
                cube_center_root_m=cube_position,
                timestamp_s=control_time_s,
                control_step=control_step,
                privileged_optional={
                    "contact_force_left_n": float(task_metrics["inner_contact_force_n"]),
                    "contact_force_right_n": float(task_metrics["outer_contact_force_n"]),
                    "total_normal_force_n": float(task_metrics["inner_contact_force_n"])
                    + float(task_metrics["outer_contact_force_n"]),
                    "contact_force_valid": (True, True),
                    "contact_measurement_valid": True,
                    "contact_timestamp_s": control_time_s + float(contract.policy_dt_s),
                    "contact_control_step": control_step + 1,
                    "left_contact": exact_inner,
                    "right_contact": exact_outer,
                    "contact": contact,
                    "exact_inner_contact": exact_inner,
                    "exact_outer_contact": exact_outer,
                    "bilateral_contact": exact_inner and exact_outer,
                    "stable_grasp": bool(task_metrics["stable_now"]),
                    "physical_lift": bool(task_metrics["ever_lifted_while_stable"]),
                    "slip_speed_m_s": float(task_metrics["slip_speed_m_s"]),
                    "forbidden_collision": forbidden_peak > 1.0e-6,
                    "forbidden_collision_valid": evaluator is not None,
                    "safety_violation": terminated_now,
                    "safety_measurement_valid": True,
                    "cube_linear_velocity_root_m_s": cube_velocity,
                    "cube_pose_root_m_xyzw": cube_pose,
                    "pad_center_relative_to_cube_velocity_root_m_s": pad_center_relative_to_cube_velocity,
                    "gripper_command": g,
                    "gripper_master_position_rad": float(
                        _tensor(robot.data.joint_pos)[0, gripper_master_index].item()
                    ),
                    "gripper_master_velocity_rad_s": float(
                        _tensor(robot.data.joint_vel)[0, gripper_master_index].item()
                    ),
                    "right_inner_distal_link_pose_root_m_xyzw": inner_link_pose,
                    "right_outer_distal_link_pose_root_m_xyzw": outer_link_pose,
                },
            )
            # The row pairs pre-action student observation/action with the
            # resulting t+1 privileged outcome. Preserve even an unsafe
            # transition as rejected evidence before failing closed.
            recorder.append(row)
            if terminated_now:
                raise KeyboardV3LiveDriverError(
                    "UNEXPECTED_TERMINATION:" + ",".join(active)
                )
            if forbidden_peak > 1.0e-6:
                raise KeyboardV3LiveDriverError(
                    f"FORBIDDEN_COLLISION:{forbidden_peak}"
                )
        if ui.recording_enabled:
            raise KeyboardV3LiveDriverError("OPERATOR_EPISODE_STEP_BUDGET_EXHAUSTED")

        selection_started = time.monotonic()
        while ui.result_selection_active and not ui.next_episode_allowed:
            if (
                selection_timeout_s is not None
                and time.monotonic() - selection_started > selection_timeout_s
            ):
                raise KeyboardV3LiveDriverError("RESULT_SELECTION_TIMEOUT")
            if terminal is not None:
                polled = terminal.poll()
                if polled.event and polled.event.startswith("RESULT_"):
                    ui.handle_key(polled.event.removeprefix("RESULT_"))
            app.update()
            time.sleep(0.01)
        if ui.publication is None:
            raise KeyboardV3LiveDriverError("RESULT_SELECTION_NOT_COMMITTED")
    except BaseException as error:
        runtime_error = f"{type(error).__name__}:{error}"
    finally:
        if terminal is not None:
            terminal.close()
    publication = ui.publication
    saved_path = Path(publication.save_path) if publication and publication.save_path else None
    reload_result = "FAIL"
    save_class = "NONE"
    if publication is not None:
        if publication.result == "DISCARD":
            save_class = "DISCARD"
            reload_result = "PASS_NO_HDF5_AS_REQUIRED"
        elif publication.canonical_training_registered:
            save_class = "CANONICAL"
            assert saved_path is not None
            reloaded = read_keyboard_v3_episode(saved_path, episode_id=episode_id)
            reload_result = "PASS" if reloaded.episode_id == episode_id else "FAIL"
        else:
            save_class = "REJECTED"
            assert saved_path is not None
            with h5py.File(saved_path, "r") as handle:
                reload_result = (
                    "PASS"
                    if not bool(handle.attrs["training_eligible"])
                    and handle.attrs["artifact_class"] == "REJECTED_EPISODE_EVIDENCE"
                    and episode_id in handle["episodes"]
                    else "FAIL"
                )

    source_freeze_after = source_freeze_provider()
    source_freeze_stable = bool(
        source_freeze_before == source_freeze_after
        and source_freeze_after.get("SOURCE_FREEZE") == "PASS"
    )
    asset_stable = bool(
        selected_asset_path.is_file()
        and _sha256(selected_asset_path) == selected_asset_sha256
    )
    result_selection = publication.result if publication else "NOT_COMPLETED"
    validator_result = publication.validator_result if publication else "NOT_RUN"
    row_count = publication.row_count if publication else len(recorder._rows)
    close_contract_ok = bool(
        publication is not None
        and (
            close_event_count == 1
            if publication.success
            else close_event_count in (0, 1)
        )
    )
    functional = bool(
        runtime_error is None
        and publication is not None
        and near_grasp_reached
        and no_forbidden_collision
        and action_submission_count > 0
        and process_action_count == action_submission_count
        and action_bound_violation_count == 0
        and close_contract_ok
        and nonzero_xyz_row_count > 0
        and reload_result.startswith("PASS")
        and source_freeze_stable
        and asset_stable
    )
    report = {
        "schema": KEYBOARD_V3_LIVE_DRIVER_SCHEMA,
        "authority": "BOUNDED_DEMONSTRATION_ONLY_ONE_SHOT_NO_RETRY",
        "source_freeze_before": dict(source_freeze_before),
        "source_freeze_after": dict(source_freeze_after),
        "selected_asset_path": str(selected_asset_path),
        "selected_asset_sha256": selected_asset_sha256,
        "selected_asset_hash_stable": asset_stable,
        "planner_receipt": plan.planner_only_receipt(),
        "direct_init_receipt": dict(direct_init_receipt or {}),
        "baseline_metadata": dict(baseline_metadata or {}),
        "planner_path_action_count": len(planner_path_records),
        "curobo_plan": "PASS",
        "near_grasp_backoff_m": float(plan.condition.backoff_m),
        "near_grasp_target_root_m": near_target.tolist(),
        "near_grasp_initial_measured_root_m": initial_ee.tolist(),
        "near_grasp_initial_orientation_xyzw": initial_quaternion.tolist(),
        "near_grasp_final_error_m": float(planner_final_error_m),
        "near_grasp_measured_handoff_error_m": measured_handoff_error_m,
        "near_grasp_reached": near_grasp_reached,
        "joint_target_reached": bool(near_grasp_reached and planner_settled),
        "controller_joint_target_error_max_rad": controller_joint_target_error_rad,
        "no_forbidden_collision_before_handoff": no_forbidden_collision,
        "no_object_contact_before_handoff": no_object_contact_before_handoff,
        "keyboard_action_submission_before_handoff": keyboard_action_submission_before_handoff,
        "recording_started": True,
        "terminal_keyboard_input": bool(terminal_keyboard),
        "terminal_translation_step_m": float(terminal_translation_step_m),
        "terminal_motion_min_interval_s": float(terminal_motion_min_interval_s),
        "terminal_smoothing_steps": int(terminal_smoothing_steps),
        "terminal_status_print_interval_steps": int(
            terminal_status_print_interval_steps
        ),
        "terminal_status_rate_hz": float(
            contract.control_hz / terminal_status_print_interval_steps
        ),
        "terminal_maximum_concurrent_motion_pulses": int(
            terminal.maximum_concurrent_motion_pulses if terminal is not None else 0
        ),
        "terminal_maximum_requested_speed_m_s": float(
            terminal_translation_step_m
            / max(terminal_motion_min_interval_s, 0.020)
        ),
        "terminal_suppressed_motion_key_count": int(
            terminal.suppressed_motion_key_count if terminal is not None else 0
        ),
        "terminal_cross_axis_tail_cancel_count": int(
            terminal.cross_axis_tail_cancel_count if terminal is not None else 0
        ),
        "terminal_multi_axis_command_count": 0,
        "e_key_contract": "E=ROBOT_ROOT_NEGATIVE_Z_WRIST_DOWN",
        "e_key_command_count": int(e_key_command_count),
        "e_key_measured_down_count": int(e_key_measured_down_count),
        "e_key_measured_up_inversion_count": int(
            e_key_measured_up_inversion_count
        ),
        "e_key_measured_delta_z_m": list(e_key_measured_delta_z_m),
        "e_key_runtime_direction_verdict": (
            "NOT_EXERCISED"
            if e_key_command_count == 0
            else (
                "PASS_WRIST_DOWN"
                if e_key_measured_up_inversion_count == 0
                and e_key_measured_down_count > 0
                else "FAIL_OR_NOT_OBSERVED"
            )
        ),
        "measured_direction_inversion_count": int(
            measured_direction_inversion_count
        ),
        "isaac_viewport_focus_required": not bool(terminal_keyboard),
        "terminal_state_restored_on_exit": bool(
            terminal is None or terminal.closed
        ),
        "operator_view_count": len(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
        "operator_view_names": list(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
        "operator_view_receipt": operator_view_receipt,
        "head_camera_binding": operator_view_receipt["head_camera_binding"],
        "wrist_camera_binding": operator_view_receipt["wrist_camera_binding"],
        "head_rgb_visible": operator_view_receipt["head_rgb_visible"],
        "wrist_rgb_visible": operator_view_receipt["wrist_rgb_visible"],
        "policy_observation_changed": False,
        "recorded_camera_count": len(camera_names),
        "recorded_camera_names": list(camera_names),
        "rgbd_capture_rate_hz": recorder.rgbd_hz,
        "camera_capture_period_control_steps": camera_capture_period_steps,
        "last_camera_frame_ids": {
            name: int(last_camera_frames[name]) for name in camera_names
        },
        "camera_capture_contract": (
            "ACTUAL_ACQUISITION_TIMESTAMP_HELD_ACROSS_TWO_CONTROL_ROWS"
        ),
        "action_dim": 4,
        "action_frame": "robot_root",
        "action_unit": "meter",
        "orientation_action": 0,
        "elbow_action": 0,
        "direct_finger_action": 0,
        "torque_action": 0,
        "open_command_source": "OPERATOR_ONLY",
        "close_command_source": "OPERATOR_ONLY",
        "autonomous_close": False,
        "residual_sac": False,
        "training": False,
        "action_submission_count": action_submission_count,
        "action_manager_process_action_count": process_action_count,
        "row_count": int(row_count),
        "nonzero_xyz_row_count": int(nonzero_xyz_row_count),
        "close_event_count": int(close_event_count),
        "maximum_action_norm_m": float(maximum_action_norm_m),
        "action_bound_violation_count": int(action_bound_violation_count),
        "open_command_count": int(open_command_count),
        "close_command_count": int(close_command_count),
        "result_selection": result_selection,
        "success": bool(publication.success) if publication else False,
        "failure_type": publication.failure_type if publication else "RUNTIME_FAILURE",
        "session_stop_requested": bool(session_stop_requested),
        "validator_result": validator_result,
        "save_class": save_class,
        "saved_path": str(saved_path) if saved_path is not None else None,
        "reload_result": reload_result,
        "pad_surface_valid": False,
        "midpoint_proxy_used": False,
        "student_privileged_input_count": 0,
        "hidden_legacy_8d_dataset_path": False,
        "explicit_controller_8d_adapter_only": True,
        "runtime_error": runtime_error,
        "keyboard_v3_live_driver_qualified": functional,
        "gru_bc_training_ready_after_data": bool(
            functional
            and save_class == "CANONICAL"
            and publication is not None
            and publication.canonical_training_registered
        ),
        "training_started": False,
        "residual_sac_started": False,
    }
    operator_views.close()
    report["operator_views_cleanup"] = "PASS"
    report["operator_result_panel_cleanup"] = _close_operator_panel(result_panel)
    report["v3_ui_cleanup_complete"] = bool(
        report["operator_views_cleanup"] == "PASS"
        and report["operator_result_panel_cleanup"] == "PASS"
    )
    _atomic_json(report_path, report)
    print("KEYBOARD_V3_LIVE_REPORT_SAVED", flush=True)
    return 0 if functional else 2


__all__ = [
    "KEYBOARD_V3_LIVE_DRIVER_SCHEMA",
    "KeyboardV3LiveDriverError",
    "run_keyboard_v3_one_shot",
]
