#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the shared scripted pick-and-place policy on the physical SO-101.

The hardware adapter supplies rectified overhead and wrist RGB plus raw
hardware-frame joint readback at 30 Hz. ``ScriptedPolicy`` owns localization,
search, planning, wrist visual servoing, trajectory playback, and checkpoint
replanning, exactly as it does in ``eval_policy_sim.py``.
"""

from __future__ import annotations

import argparse
import datetime
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import (
    LOCAL_CAMERA_INTRINSICS_DIR,
    load_camera_intrinsics,
)
from pick_and_place.dataset_metadata import cube_pose_metadata, driver_metadata
from pick_and_place.episode_loop import episode_loop
from pick_and_place.episodes import _build_model, sample_recovery_cube, set_cube_pose, set_joint
from pick_and_place.executor import CONTROL_HZ, follower_clamp_limits
from pick_and_place.follower import (
    GRIPPER_INDEX,
    GRIPPER_READBACK_CLOSED,
    action_to_joints,
    joints_to_action,
    make_so101_follower,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.image_rectify import (
    build_undistort_map,
    rectified_camera_matrix,
    transform_frame,
)
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.overhead_localization import OverheadLocalizer
from pick_and_place.overhead_detection import DEFAULT_ALERT_SOUND, OperatorNotifier
from pick_and_place.paper_detection import set_paper_target_marker
from pick_and_place.policy import DEFAULT_IMAGE_HW
from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.physical_rig import PhysicalRig, require_joint_zero_offsets
from pick_and_place.physical_collection import (
    CameraDriftError,
    recover_cube,
    reject_camera_drift,
    wait_for_target_movement,
)
from pick_and_place.policy_real import (
    PhysicalEpisodeOutcome,
    PhysicalPolicyTick,
    calibrated_state,
    prepare_physical_policy_episode,
    run_physical_policy_episode,
)
from pick_and_place.policy_recording import PolicyRecordingSession
from pick_and_place.scripted_policy import (
    AsyncWristLocalization,
    ScriptedPolicy,
    WristCameraLocalizer,
)
from pick_and_place.trajectory import (
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)
from pick_and_place.workspace_overlays import (
    PAN_AXIS,
    is_cube_recovery_target_allowed,
    workspace_interior_corners_world,
)

OVERHEAD_CAPTURE_SIZE = (1920, 1080)
WRIST_CAPTURE_SIZE = (1280, 720)
REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_POLICY_SLEW_PER_SECOND = np.array([60.0, 60.0, 75.0, 90.0, 120.0, 150.0])
PICKUP_GRIPPER_MARGIN = 5.0


class LatestCamera:
    """Continuously capture a camera while exposing its freshest frame."""

    def __init__(
        self,
        source: str,
        label: str,
        capture_size: tuple[int, int],
        cv2_module: Any,
    ) -> None:
        from pick_and_place.cam_align_solve import parse_index_or_path

        backend = (
            cv2_module.CAP_AVFOUNDATION
            if hasattr(cv2_module, "CAP_AVFOUNDATION")
            else cv2_module.CAP_ANY
        )
        self._capture = cv2_module.VideoCapture(parse_index_or_path(source), backend)
        self._capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, capture_size[0])
        self._capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, capture_size[1])
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"could not open {label} camera {source!r}")
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._last_read_sequence = 0
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()

    def latest(self) -> np.ndarray:
        with self._condition:
            if self._frame is None:
                self._condition.wait_for(lambda: self._frame is not None, timeout=2.0)
            if self._frame is None:
                raise RuntimeError("timed out waiting for camera frames")
            return self._frame.copy()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """OpenCV-compatible fresh-frame read used by calibration."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > self._last_read_sequence or not self._running,
                timeout=1.0,
            )
            if self._frame is None or self._sequence <= self._last_read_sequence:
                return False, None
            self._last_read_sequence = self._sequence
            return True, self._frame.copy()

    def close(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        self._capture.release()


def _rectified_rgb_reader(
    camera: LatestCamera,
    intrinsics: dict[str, Any],
    frame_size: tuple[int, int],
    cv2: Any,
    *,
    output_size: tuple[int, int] | None = None,
):
    maps = build_undistort_map(intrinsics, *frame_size, cv2)

    def read() -> np.ndarray:
        bgr = camera.latest()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if output_size is None:
            return cv2.remap(rgb, maps[0], maps[1], cv2.INTER_LINEAR)
        return transform_frame(
            rgb,
            maps,
            *output_size,
            cv2,
        )

    return read


def _ramp_follower(
    follower: Any,
    target: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    max_joint_speed: float,
) -> None:
    """Move smoothly to a non-policy parking pose in hardware-frame units."""
    current = action_to_joints(follower.get_observation(), target)
    arm_travel = float(np.max(np.abs(target[:-1] - current[:-1])))
    duration = max(1.0, arm_travel / max_joint_speed)
    steps = max(1, round(duration * CONTROL_HZ))
    for index in range(1, steps + 1):
        started = time.monotonic()
        t = index / steps
        smooth = t * t * (3.0 - 2.0 * t)
        command = np.clip(current + smooth * (target - current), low, high)
        follower.send_action(joints_to_action(command))
        remaining = 1.0 / CONTROL_HZ - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", required=True, help="serial port of the SO-101 follower")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id")
    parser.add_argument(
        "--joint-zeros",
        type=Path,
        default=REPO_ROOT / "config" / "joint_zeros.json",
    )
    parser.add_argument(
        "--allow-uncalibrated-debug",
        action="store_true",
        help="run without joint-zero correction for safe bench diagnostics only",
    )
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera")
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera")
    parser.add_argument("--camera-name", default="overhead_camera", help="overhead MuJoCo camera name")
    parser.add_argument("--overhead-intrinsics", type=Path, default=None)
    parser.add_argument("--wrist-intrinsics", type=Path, default=None)
    parser.add_argument(
        "--workspace-camera",
        default=None,
        help="optional OpenCV index/path of a synchronized workspace camera",
    )
    parser.add_argument("--workspace-intrinsics", type=Path, default=None)
    parser.add_argument(
        "--drop-zone-color", choices=("black", "white"), default="black"
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes to run; 0 means continuous")
    parser.add_argument(
        "--rest-every",
        type=int,
        default=10,
        help="completed episodes between cooldowns; 0 disables cooldowns",
    )
    parser.add_argument("--rest-duration", type=float, default=30.0)
    parser.add_argument(
        "--operator-alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--alert-sound", default=DEFAULT_ALERT_SOUND)
    parser.add_argument("--target-change-min-distance", type=float, default=0.03)
    parser.add_argument("--target-change-alert-min-seconds", type=float, default=10.0)
    parser.add_argument("--target-change-alert-max-seconds", type=float, default=120.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--recording-format", choices=("video", "dataset", "none"), default="video"
    )
    parser.add_argument("--recording-root", type=Path, default=REPO_ROOT / "episodes")
    parser.add_argument("--dataset-repo-id", default="physical-scripted-v2")
    parser.add_argument(
        "--max-steps", type=int, default=450, help="30 Hz ticks per episode (default: 450)"
    )
    parser.add_argument("--max-localization-steps", type=int, default=60)
    parser.add_argument("--localization-steps-per-search", type=int, default=15)
    parser.add_argument("--planning-attempts", type=int, default=40)
    parser.add_argument(
        "--preflight-debug",
        action="store_true",
        help="print collision details for rejected trajectory candidates",
    )
    parser.add_argument("--preflight-debug-limit", type=int, default=12)
    parser.add_argument("--save-failed-trajectories", type=Path, default=None)
    parser.add_argument("--failed-trajectory-limit", type=int, default=8)
    parser.add_argument(
        "--show-camera-feeds",
        action="store_true",
        help="show the rectified overhead and wrist observations",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="show the measured arm and localized objects in MuJoCo",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve overhead extrinsics from workspace AprilTags at startup",
    )
    parser.add_argument("--recalibrate-samples", type=int, default=10)
    parser.add_argument("--recalibrate-max-seconds", type=float, default=15.0)
    parser.add_argument("--recalibrate-check-min-cooldown", type=float, default=15.0)
    parser.add_argument("--recalibrate-drift-mm", type=float, default=10.0)
    parser.add_argument("--recalibrate-drift-deg", type=float, default=2.0)
    parser.add_argument("--cube-recovery-attempts", type=int, default=3)
    parser.add_argument(
        "--park-speed",
        type=float,
        default=30.0,
        help="maximum arm-joint speed while parking, in degrees/s",
    )
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error("--episodes must be non-negative")
    if args.rest_every < 0:
        parser.error("--rest-every must be non-negative")
    if args.rest_duration < 0.0:
        parser.error("--rest-duration must be non-negative")
    if args.target_change_min_distance < 0.0:
        parser.error("--target-change-min-distance must be non-negative")
    if args.target_change_alert_min_seconds <= 0.0:
        parser.error("--target-change-alert-min-seconds must be positive")
    if args.target_change_alert_max_seconds < args.target_change_alert_min_seconds:
        parser.error("--target-change-alert-max-seconds must be at least the minimum")
    if args.recalibrate_check_min_cooldown < 0.0:
        parser.error("--recalibrate-check-min-cooldown must be non-negative")
    if args.recalibrate_drift_mm < 0.0 or args.recalibrate_drift_deg < 0.0:
        parser.error("camera drift limits must be non-negative")
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be in (0, 1]")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.planning_attempts < 1:
        parser.error("--planning-attempts must be at least 1")
    if args.preflight_debug_limit < 1:
        parser.error("--preflight-debug-limit must be at least 1")
    if args.failed_trajectory_limit < 0:
        parser.error("--failed-trajectory-limit must be non-negative")
    if args.park_speed <= 0.0:
        parser.error("--park-speed must be positive")
    if args.cube_recovery_attempts < 1:
        parser.error("--cube-recovery-attempts must be at least 1")
    return args


def main() -> None:
    args = _parse_args()
    import cv2

    from pick_and_place.cam_align_solve import (
        ExtrinsicsSolveError,
        apply_solve_result,
        check_solve_plausible,
        solve_overhead_extrinsics,
    )
    notifier = OperatorNotifier(enabled=args.operator_alerts, sound_path=args.alert_sound)
    try:
        joint_zero_offsets = require_joint_zero_offsets(
            args.joint_zeros,
            allow_uncalibrated=args.allow_uncalibrated_debug,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not joint_zero_offsets:
        print("WARNING: running without joint-zero correction (debug override).")

    overhead_path = args.overhead_intrinsics or (
        LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
    )
    wrist_path = args.wrist_intrinsics or (
        LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
    )
    workspace_path = (
        args.workspace_intrinsics or LOCAL_CAMERA_INTRINSICS_DIR / "workspace_camera.json"
        if args.workspace_camera is not None
        else None
    )
    required_intrinsics = [("overhead", overhead_path), ("wrist", wrist_path)]
    if workspace_path is not None:
        required_intrinsics.append(("workspace", workspace_path))
    for label, path in required_intrinsics:
        if not path.exists():
            raise SystemExit(f"missing {label} camera intrinsics at {path}")
    overhead_intrinsics = load_camera_intrinsics(overhead_path)
    wrist_intrinsics = load_camera_intrinsics(wrist_path)
    workspace_intrinsics = (
        load_camera_intrinsics(workspace_path) if workspace_path is not None else None
    )

    print("Opening cameras...")
    overhead = LatestCamera(args.camera, "overhead", OVERHEAD_CAPTURE_SIZE, cv2)
    wrist: LatestCamera | None = None
    workspace: LatestCamera | None = None
    follower = None
    rig = None
    controller = None
    recovery_controller = None
    recording = None
    debug_viewer = None
    torque_released = False
    try:
        wrist = LatestCamera(args.wrist_camera, "wrist", WRIST_CAPTURE_SIZE, cv2)
        if args.workspace_camera is not None:
            workspace = LatestCamera(
                args.workspace_camera,
                "workspace",
                OVERHEAD_CAPTURE_SIZE,
                cv2,
            )
        overhead_frame = overhead.latest()
        wrist_frame = wrist.latest()
        overhead_size = (overhead_frame.shape[1], overhead_frame.shape[0])
        wrist_size = (wrist_frame.shape[1], wrist_frame.shape[0])
        workspace_size = None
        if workspace is not None:
            workspace_frame = workspace.latest()
            workspace_size = (workspace_frame.shape[1], workspace_frame.shape[0])
        print(
            f"Camera resolutions: overhead {overhead_size[0]}x{overhead_size[1]}, "
            f"wrist {wrist_size[0]}x{wrist_size[1]}"
        )

        dummy_source = CubePose(PAN_AXIS[0] + 0.1, PAN_AXIS[1], CUBE_HALF_SIZE)
        model, data = _build_model(
            dummy_source,
            include_environment=True,
            paper_target_marker=True,
        )
        apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
        mujoco.mj_forward(model, data)

        startup_extrinsics: tuple[np.ndarray, np.ndarray] | None = None
        if args.recalibrate:
            print("Solving overhead camera extrinsics from workspace AprilTags...")
            result = solve_overhead_extrinsics(
                model,
                data,
                overhead,
                camera_name=args.camera_name,
                intrinsics_path=overhead_path,
                samples=args.recalibrate_samples,
                max_seconds=args.recalibrate_max_seconds,
                width=overhead_size[0],
                height=overhead_size[1],
                cv2_module=cv2,
            )
            if result is None:
                raise SystemExit("overhead calibration failed: all four tags were not visible")
            try:
                check_solve_plausible(result)
            except ExtrinsicsSolveError as exc:
                raise SystemExit(f"overhead calibration rejected: {exc}") from exc
            apply_solve_result(model, data, args.camera_name, result)
            startup_extrinsics = (
                np.asarray(result.pos, dtype=float),
                np.asarray(result.quat, dtype=float),
            )
            print(f"Overhead calibration reprojection error: {result.reprojection_error_px:.2f}px")

        overhead_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name
        )
        overhead_matrix = np.asarray(
            rectified_camera_matrix(
                overhead_intrinsics,
                DEFAULT_IMAGE_HW[1],
                DEFAULT_IMAGE_HW[0],
            )
        )
        wrist_matrix = np.asarray(
            rectified_camera_matrix(
                wrist_intrinsics,
                DEFAULT_IMAGE_HW[1],
                DEFAULT_IMAGE_HW[0],
            )
        )
        def make_controller(*, recovery: bool) -> ScriptedPolicy:
            return ScriptedPolicy(
                OverheadLocalizer(
                    overhead_matrix,
                    data.cam_xpos[overhead_id],
                    data.cam_xmat[overhead_id].reshape(3, 3),
                ),
                workspace_interior_corners_world(),
                target_color=args.drop_zone_color,
                max_localization_steps=args.max_localization_steps,
                localization_steps_per_search=args.localization_steps_per_search,
                planning_max_attempts=args.planning_attempts,
                planning_verbose=True,
                preflight_debug=args.preflight_debug,
                preflight_debug_limit=args.preflight_debug_limit,
                failed_trajectory_dir=args.save_failed_trajectories,
                failed_trajectory_limit=args.failed_trajectory_limit,
                rng_seed=args.rng_seed,
                control_hz=CONTROL_HZ / args.speed,
                wrist_localizer=AsyncWristLocalization(
                    WristCameraLocalizer(model, wrist_matrix, free_grasp=recovery)
                ),
                target_sampler=sample_recovery_cube if recovery else None,
                free_grasp=recovery,
            )

        controller = make_controller(recovery=False)
        recovery_controller = make_controller(recovery=True)
        target_localizer = OverheadLocalizer(
            overhead_matrix,
            data.cam_xpos[overhead_id],
            data.cam_xmat[overhead_id].reshape(3, 3),
        )
        overhead_rgb = _rectified_rgb_reader(
            overhead,
            overhead_intrinsics,
            overhead_size,
            cv2,
            output_size=(DEFAULT_IMAGE_HW[1], DEFAULT_IMAGE_HW[0]),
        )
        wrist_rgb = _rectified_rgb_reader(
            wrist,
            wrist_intrinsics,
            wrist_size,
            cv2,
            output_size=(DEFAULT_IMAGE_HW[1], DEFAULT_IMAGE_HW[0]),
        )
        workspace_rgb = None
        if workspace is not None:
            assert workspace_intrinsics is not None and workspace_size is not None
            workspace_rgb = _rectified_rgb_reader(
                workspace,
                workspace_intrinsics,
                workspace_size,
                cv2,
                output_size=(DEFAULT_IMAGE_HW[1], DEFAULT_IMAGE_HW[0]),
            )

        recording = None
        if args.recording_format != "none":
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.recording_format == "video":
                from pick_and_place.episode_video import EpisodeVideoSession

                camera_intrinsics = {"overhead": overhead_path, "wrist": wrist_path}
                if workspace_path is not None:
                    camera_intrinsics["workspace"] = workspace_path
                session = EpisodeVideoSession(
                    root=args.recording_root / stamp,
                    fps=CONTROL_HZ,
                    task="scripted physical pick and place",
                    camera_intrinsics=camera_intrinsics,
                    input_rectified=True,
                )
            else:
                from pick_and_place.recording import RecordingSession

                session = RecordingSession(
                    repo_id=args.dataset_repo_id,
                    root=args.recording_root / stamp,
                    task="scripted physical pick and place",
                    fps=CONTROL_HZ,
                )
            recording = PolicyRecordingSession(
                session,
                "scripted physical pick and place",
                workspace_rgb=workspace_rgb,
                episode_metadata=lambda: (
                    {
                        **cube_pose_metadata(
                            controller.cube_pose,
                            CubePose(
                                float(controller.drop_target.xy[0]),
                                float(controller.drop_target.xy[1]),
                                CUBE_HALF_SIZE,
                            ),
                        ),
                        **driver_metadata("scripted"),
                    }
                    if controller.cube_pose is not None
                    and controller.drop_target is not None
                    else None
                ),
            )

        print("Connecting to follower...")
        follower = make_so101_follower(
            args.follower_port,
            args.follower_id,
            disable_torque_on_disconnect=False,
        )
        follower.connect()
        limits = follower_clamp_limits(derive_kinematics(model))
        clamp_low, clamp_high = limits

        def park_action() -> None:
            print("Parking at NEUTRAL, then REST...")
            follower.bus.enable_torque()
            _ramp_follower(
                follower,
                sim_frame_to_real(
                    NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, joint_zero_offsets
                ),
                clamp_low,
                clamp_high,
                max_joint_speed=args.park_speed,
            )
            _ramp_follower(
                follower,
                sim_frame_to_real(
                    REST_ARM_JOINTS, REST_GRIPPER, joint_zero_offsets
                ),
                clamp_low,
                clamp_high,
                max_joint_speed=args.park_speed,
            )

        rig = PhysicalRig(
            follower=follower,
            overhead=overhead,
            wrist=wrist,
            workspace=workspace,
            clamp_low=clamp_low,
            clamp_high=clamp_high,
            joint_zero_offsets=joint_zero_offsets,
            park_action=park_action,
        )

        print("Moving to NEUTRAL before localization and planning...")
        _ramp_follower(
            follower,
            sim_frame_to_real(
                NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, joint_zero_offsets
            ),
            clamp_low,
            clamp_high,
            max_joint_speed=args.park_speed,
        )

        if args.viewer:
            from mujoco import viewer as mujoco_viewer

            debug_viewer = mujoco_viewer.launch_passive(model, data)

        def show_observation(observation: dict[str, np.ndarray]) -> None:
            overhead_bgr = cv2.cvtColor(
                observation[OVERHEAD_FEATURE], cv2.COLOR_RGB2BGR
            )
            wrist_bgr = cv2.cvtColor(
                observation[WRIST_FEATURE], cv2.COLOR_RGB2BGR
            )
            cv2.imshow(
                "ScriptedPolicy observations: overhead | wrist",
                np.concatenate((overhead_bgr, wrist_bgr), axis=1),
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise KeyboardInterrupt

        def sync_viewer(tick: PhysicalPolicyTick) -> None:
            if debug_viewer is None:
                return
            if not debug_viewer.is_running():
                raise KeyboardInterrupt
            reported_joints, reported_gripper = real_frame_to_sim(
                tick.observation[STATE_FEATURE]
            )
            for name, value in reported_joints.items():
                set_joint(model, data, name, value)
            set_joint(model, data, "gripper", reported_gripper)
            if controller.cube_pose is not None:
                set_cube_pose(model, data, controller.cube_pose)
            if controller.drop_target is not None:
                set_paper_target_marker(
                    model,
                    data,
                    controller.drop_target,
                    usable=True,
                )
            mujoco.mj_forward(model, data)
            debug_viewer.sync()

        cooldown_reference_target: CubePose | None = None

        def check_overhead_drift() -> None:
            if startup_extrinsics is None:
                return
            print("Cooldown drift check: re-solving overhead extrinsics...")
            check = solve_overhead_extrinsics(
                model,
                data,
                overhead,
                camera_name=args.camera_name,
                intrinsics_path=overhead_path,
                samples=args.recalibrate_samples,
                max_seconds=args.recalibrate_max_seconds,
                width=overhead_size[0],
                height=overhead_size[1],
                cv2_module=cv2,
            )
            if check is None:
                print("Drift check skipped: all four tags were not visible.")
                return
            try:
                drift_mm, drift_deg = reject_camera_drift(
                    startup_extrinsics[0],
                    startup_extrinsics[1],
                    np.asarray(check.pos, dtype=float),
                    np.asarray(check.quat, dtype=float),
                    max_translation_mm=args.recalibrate_drift_mm,
                    max_rotation_deg=args.recalibrate_drift_deg,
                )
            except CameraDriftError as exc:
                notifier.alert(f"{exc}. The collection run is stopping.", repeat_sound=2)
                raise SystemExit(str(exc)) from exc
            print(f"Overhead drift vs startup: {drift_mm:.1f}mm / {drift_deg:.2f}deg.")

        def detect_target() -> CubePose | None:
            target = target_localizer.localize_drop_target(
                overhead_rgb(),
                target_color=args.drop_zone_color,
                workspace_corners_world=workspace_interior_corners_world(),
            )
            if target is None:
                return None
            return CubePose(float(target.xy[0]), float(target.xy[1]), CUBE_HALF_SIZE)

        def cooldown() -> None:
            print("Cooldown: moving to REST and releasing torque...")
            notifier.alert("Cooldown started. Move the target plate before the next episode.")
            _ramp_follower(
                follower,
                sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER, joint_zero_offsets),
                clamp_low,
                clamp_high,
                max_joint_speed=args.park_speed,
            )
            follower.bus.disable_torque()
            target_localizer.reset()
            try:
                wait_for_target_movement(
                    cooldown_reference_target,
                    minimum_distance=args.target_change_min_distance,
                    minimum_rest_until=time.monotonic() + args.rest_duration,
                    detect_target=detect_target,
                    alert=notifier.alert,
                    alert_min_seconds=args.target_change_alert_min_seconds,
                    alert_max_seconds=args.target_change_alert_max_seconds,
                )
            finally:
                follower.bus.enable_torque()
            _ramp_follower(
                follower,
                sim_frame_to_real(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, joint_zero_offsets),
                clamp_low,
                clamp_high,
                max_joint_speed=args.park_speed,
            )
            if (
                args.recalibrate_check_min_cooldown > 0.0
                and args.rest_duration >= args.recalibrate_check_min_cooldown
            ):
                check_overhead_drift()

        def pickup_verifier(policy: ScriptedPolicy):
            previous_phase = [policy.phase_name]

            def verify(tick: PhysicalPolicyTick) -> bool | None:
                phase = policy.phase_name
                completed_lift = previous_phase[0] in ("lift", "recovery_lift") and phase not in (
                    "lift",
                    "recovery_lift",
                )
                previous_phase[0] = phase
                if not completed_lift:
                    return None
                raw = action_to_joints(follower.get_observation(), tick.command)
                position = float(calibrated_state(raw, joint_zero_offsets)[GRIPPER_INDEX])
                confidence = (position - GRIPPER_READBACK_CLOSED) / PICKUP_GRIPPER_MARGIN
                print(f"Pickup confidence: {confidence:.2f} (gripper {position:.1f})")
                return confidence >= 1.0

            return verify

        def rehome() -> None:
            print("Opening and re-homing before retry...")
            _ramp_follower(
                follower,
                sim_frame_to_real(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, joint_zero_offsets),
                clamp_low,
                clamp_high,
                max_joint_speed=args.park_speed,
            )

        recovery_sequence = [0]
        def relocate_cube(recovery_attempt: int) -> bool:
            assert recovery_controller is not None
            recovery_sequence[0] += 1
            recovery_controller.rng_seed = args.rng_seed + recovery_sequence[0]
            print(
                f"Cube recovery {recovery_attempt}/{args.cube_recovery_attempts} "
                "(unrecorded): preflight"
            )
            preflight = prepare_physical_policy_episode(
                recovery_controller,
                follower=follower,
                overhead_rgb=overhead_rgb,
                wrist_rgb=wrist_rgb,
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                joint_zero_offsets=joint_zero_offsets,
                max_steps=args.max_localization_steps + 1,
            )
            if preflight is not None:
                print(f"Cube recovery preflight failed: {preflight.outcome.value}")
                rehome()
                return False
            result = run_physical_policy_episode(
                recovery_controller,
                follower=follower,
                overhead_rgb=overhead_rgb,
                wrist_rgb=wrist_rgb,
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                control_hz=CONTROL_HZ,
                max_steps=args.max_steps,
                joint_zero_offsets=joint_zero_offsets,
                max_slew_per_second=MAX_POLICY_SLEW_PER_SECOND,
                pickup_verifier=pickup_verifier(recovery_controller),
                reset_controller=False,
            )
            if result.outcome is PhysicalEpisodeOutcome.OPERATOR_ABORT:
                raise KeyboardInterrupt
            if not result.succeeded:
                rehome()
            return result.succeeded

        def locate_recovered_cube() -> CubePose | None:
            assert recovery_controller is not None
            recovery_controller.localizer.reset()
            for _ in range(args.localization_steps_per_search):
                cube = recovery_controller.localizer.localize_cube(
                    overhead_rgb(),
                    free_grasp=True,
                )
                if cube is not None:
                    return cube
                time.sleep(0.05)
            return None

        def run_cube_recovery() -> bool:
            recovered = recover_cube(
                max_attempts=args.cube_recovery_attempts,
                relocate=relocate_cube,
                locate=locate_recovered_cube,
                is_allowed=is_cube_recovery_target_allowed,
            )
            if recovered is None:
                notifier.alert("Cube recovery failed after retries. The run is stopping.")
                return False
            print(f"Cube recovery verified at ({recovered.x:.3f}, {recovered.y:.3f}).")
            return True

        for ep in episode_loop(
            target=args.episodes,
            rest_every=args.rest_every,
            cooldown=cooldown,
        ):
            label = "continuous" if args.episodes == 0 else str(args.episodes)
            print(f"Attempt {ep.attempt}, completed {ep.index - 1}/{label}: preflight")
            preflight_result = prepare_physical_policy_episode(
                controller,
                follower=follower,
                overhead_rgb=overhead_rgb,
                wrist_rgb=wrist_rgb,
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                joint_zero_offsets=joint_zero_offsets,
                max_steps=args.max_localization_steps + 1,
            )
            if preflight_result is not None:
                print(f"Preflight failed: {preflight_result.outcome.value}")
                _ramp_follower(
                    follower,
                    sim_frame_to_real(
                        NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, joint_zero_offsets
                    ),
                    clamp_low,
                    clamp_high,
                    max_joint_speed=args.park_speed,
                )
                continue

            def verify_placement() -> bool:
                cube = controller.localizer.localize_cube(overhead_rgb())
                if cube is None or controller.drop_target is None:
                    return False
                error = np.linalg.norm(np.asarray((cube.x, cube.y)) - controller.drop_target.xy)
                print(f"Final overhead placement error: {error * 100.0:.1f} cm")
                return bool(error <= 0.04)

            print("Execution starting at 30 Hz")
            result = run_physical_policy_episode(
                controller,
                follower=follower,
                overhead_rgb=overhead_rgb,
                wrist_rgb=wrist_rgb,
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                control_hz=CONTROL_HZ,
                max_steps=args.max_steps,
                joint_zero_offsets=joint_zero_offsets,
                max_slew_per_second=MAX_POLICY_SLEW_PER_SECOND,
                pickup_verifier=pickup_verifier(controller),
                placement_verifier=verify_placement,
                recording=recording,
                observation_callback=show_observation if args.show_camera_feeds else None,
                tick_callback=sync_viewer if debug_viewer is not None else None,
                reset_controller=False,
            )
            if result.succeeded:
                print(f"Episode completed in {result.control_steps} control ticks.")
                if controller.drop_target is not None:
                    cooldown_reference_target = CubePose(
                        float(controller.drop_target.xy[0]),
                        float(controller.drop_target.xy[1]),
                        CUBE_HALF_SIZE,
                    )
                ep.complete()
                is_last = args.episodes != 0 and ep.index >= args.episodes
                if not is_last and not run_cube_recovery():
                    break
                continue
            if result.outcome is PhysicalEpisodeOutcome.OPERATOR_ABORT:
                print("Operator aborted the run.")
                break
            if result.controller_failure is not None:
                failure = result.controller_failure
                print(f"Controller failed safely [{failure.code}]: {failure.message}")
            else:
                print(f"Attempt discarded: {result.outcome.value}")
            notifier.alert(f"Episode discarded: {result.outcome.value}. Re-homing and retrying.")
            rehome()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if debug_viewer is not None:
            debug_viewer.close()
        if args.show_camera_feeds:
            cv2.destroyAllWindows()
        if recording is not None:
            recording.finalize()
        if controller is not None:
            controller.close()
        if recovery_controller is not None:
            recovery_controller.close()
        if rig is not None:
            try:
                rig.park_and_release()
                torque_released = True
            except Exception as exc:  # noqa: BLE001 - best-effort emergency parking
                print(f"Warning: could not finish parking: {exc}")
            if torque_released:
                print("At REST with torque released.")
        else:
            if follower is not None:
                follower.disconnect()
            overhead.close()
            if wrist is not None:
                wrist.close()
            if workspace is not None:
                workspace.close()


if __name__ == "__main__":
    main()
