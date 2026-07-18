#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the analytic pick-and-place on the physical SO-101 as a continuous loop.

Refuses to start unless the full rig is present: both the overhead and wrist
cameras open and both have calibrated intrinsics. It then solves the overhead
camera extrinsics from the workspace-frame AprilTags and refuses to start if the
tags are missing or implausible (swapped/rotated). During a long-enough cooldown
it re-solves and stops the run if the camera has drifted from that startup pose.

Homes the arm to a near-neutral pose, then repeats: look for the cube from the
current pose (re-homing to fresh near-neutral poses if it can't be seen), plan a
collision-free ``pick_and_carry`` episode from wherever the arm currently is, and
run it on the real arm via ``pick_and_place.executor``. A failed plan or a failed
checkpoint replan aborts the episode and re-homes. Every ``--rest-every`` episodes
the arm takes a torque-off cooldown at REST; the operator is expected to move the
target plate during that window, and the run stays halted with repeated audible
alerts until the plate has moved far enough. Press Ctrl-C to stop: the arm parks
to NEUTRAL, then REST, and releases torque.

The runner owns the follower, the viewer and the cameras and keeps them alive
across the whole loop; ``execute_episode`` runs a single pass against them. For
sim-only playback (no arm) use ``sim.py``.

By default every run writes synchronized episode videos into
``episodes/<timestamp>/``. Each successful episode has one frame per control
tick from the wrist, overhead, and optional workspace cameras, plus a matching
state/action/simulation timeline for replay. ``--recording-format dataset``
retains the LeRobotDataset output for collection work. Aborted/restarted
episodes are discarded. See ``execute_episode``'s docstring.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place.dataset_metadata import cube_pose_metadata, driver_metadata
from pick_and_place.episode_loop import episode_loop
from pick_and_place.episodes import (
    EpisodeSamplingError,
    _build_model,
    prepare_episode,
    sample_hunt_pose,
    sample_near_neutral,
    sample_recovery_cube,
)
from pick_and_place.executor import (
    CONTROL_HZ,
    HARDWARE_SIMULATION_HZ,
    clamp_and_warn,
    execute_episode,
    follower_clamp_limits,
    ramp_to_resting,
)
from pick_and_place.recording import RecordingSession
from pick_and_place.episode_video import EpisodeVideoSession
from pick_and_place.follower import (
    action_to_joints,
    load_joint_zero_offsets,
    make_so101_follower,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.overhead_detection import (
    CUBE_LOOK_TIMEOUT,
    DEFAULT_ALERT_SOUND,
    MockViewer,
    OperatorNotifier,
    OverheadDetectionDebug,
    empty_overhead_debug,
    final_placement_metadata,
    track_cube,
    track_drop_zone_square,
    write_overhead_debug_image,
)
from pick_and_place.paper_detection import PaperTracker
from pick_and_place.safety import EpisodeAborted, recover_on
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Plan-search budget per episode: how many source/target/end resamples to try
# before declaring the cube unreachable from the current pose and aborting.
EPISODE_MAX_ATTEMPTS = 40
# Cube-recovery relocation retries. The move is unrecorded but required so
# unattended collection can continue from a cube location that is usable for the
# next recorded pickup.
CUBE_RECOVERY_MAX_ATTEMPTS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="number of episodes to run; 0 means loop until Ctrl-C (default: 0)",
    )
    parser.add_argument(
        "--max-hunt-tries",
        type=int,
        default=5,
        help="near-neutral poses to try while looking for the cube before giving up (default: 5)",
    )
    parser.add_argument(
        "--rest-every",
        type=int,
        default=10,
        help="episodes between cooldown rests; 0 to disable (default: 10)",
    )
    parser.add_argument(
        "--rest-duration",
        type=float,
        default=30.0,
        help="cooldown rest duration in seconds, torque off at REST (default: 30.0)",
    )
    parser.add_argument(
        "--operator-alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play a signal sound and speak operator alerts on macOS (default: on)",
    )
    parser.add_argument(
        "--alert-sound",
        default=DEFAULT_ALERT_SOUND,
        help=f"sound file to play before spoken alerts (default: {DEFAULT_ALERT_SOUND})",
    )
    parser.add_argument(
        "--target-change-min-distance",
        type=float,
        default=0.03,
        help="minimum cooldown target-plate center movement before the run resumes, in metres "
        "(default: 0.03)",
    )
    parser.add_argument(
        "--target-change-alert-min-seconds",
        type=float,
        default=10.0,
        help="initial backoff between repeated target-plate alerts (default: 10)",
    )
    parser.add_argument(
        "--target-change-alert-max-seconds",
        type=float,
        default=120.0,
        help="maximum backoff between repeated target-plate alerts (default: 120)",
    )
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the source cube (x, y) on the floor; omit to track it with the camera",
    )
    parser.add_argument(
        "--drop-zone-color",
        dest="drop_zone_color",
        choices=("black", "white"),
        default="black",
        help="color of the drop-zone square to detect (default: black)",
    )
    parser.add_argument("--follower-port", required=True, help="serial port of the SO-101 follower")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id (default: folly)")
    parser.add_argument(
        "--joint-zeros",
        type=Path,
        default=REPO_ROOT / "config" / "joint_zeros.json",
        help="session joint-zero calibration to apply feed-forward "
        "(default: config/joint_zeros.json)",
    )
    parser.add_argument(
        "--no-joint-zero-correction",
        action="store_true",
        help="ignore the joint-zero calibration and command raw servo angles",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier of nominal pace (1.0 = nominal; default: 1.0)",
    )
    parser.add_argument(
        "--pickup-empty-gripper-position",
        type=float,
        default=2.3,
        help="physical gripper readback expected after an empty close (default: 2.3)",
    )
    parser.add_argument(
        "--pickup-gripper-margin",
        type=float,
        default=5.0,
        help="denominator for pickup_confidence = gripper_delta / margin (default: 5.0)",
    )
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera (default: 0)")
    parser.add_argument("--camera-name", default="overhead_camera", help="camera name in the model")
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve the overhead camera extrinsics live from the workspace-frame AprilTags at "
        "startup and refuse to start if they are missing or implausible (default: on; "
        "--no-recalibrate uses the saved sidecar extrinsics instead)",
    )
    parser.add_argument(
        "--recalibrate-samples",
        type=int,
        default=10,
        help="overhead frames to average per extrinsics solve (default: 10)",
    )
    parser.add_argument(
        "--recalibrate-max-seconds",
        type=float,
        default=15.0,
        help="time budget to gather the solve frames before giving up (default: 15)",
    )
    parser.add_argument(
        "--overhead-intrinsics",
        type=Path,
        default=None,
        help="overhead camera intrinsics JSON for the solve (default: local sidecar)",
    )
    parser.add_argument(
        "--recalibrate-check-min-cooldown",
        type=float,
        default=15.0,
        help="during a cooldown at least this long (s), re-solve the overhead extrinsics at "
        "near-neutral and stop the run if the camera has drifted past the threshold; 0 disables "
        "the cooldown drift check (default: 15)",
    )
    parser.add_argument(
        "--recalibrate-drift-mm",
        type=float,
        default=10.0,
        help="translation drift from the startup solve that stops the run (default: 10 mm)",
    )
    parser.add_argument(
        "--recalibrate-drift-deg",
        type=float,
        default=2.0,
        help="rotation drift from the startup solve that stops the run (default: 2 deg)",
    )
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera (default: 1)")
    parser.add_argument("--wrist-intrinsics", default=None, help="path to wrist camera intrinsics JSON")
    parser.add_argument(
        "--workspace-camera",
        default=None,
        help="optional OpenCV index/path of a workspace camera to record",
    )
    parser.add_argument(
        "--workspace-intrinsics",
        type=Path,
        default=None,
        help="workspace camera intrinsics JSON (default: local workspace_camera sidecar)",
    )
    parser.add_argument(
        "--workspace-audio",
        action="store_true",
        help="capture the workspace audio input and mux it into workspace.mp4",
    )
    parser.add_argument(
        "--workspace-audio-device",
        default=None,
        help="sounddevice input name or index (default: system input device)",
    )
    parser.add_argument(
        "--live-videos",
        action="store_true",
        help="also record native-rate wrist, overhead, and workspace MP4s on a shared "
        "monotonic clock; the regular videos remain control-tick aligned",
    )
    parser.add_argument("--show-wrist-cam", action="store_true", help="show the live wrist camera feed")
    parser.add_argument("--show-wrist-mixed", action="store_true", help="overlay the sim render on the wrist feed")
    parser.add_argument(
        "--save-overhead-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save preflight/final overhead verification images into the run directory "
        "(default: on)",
    )
    parser.add_argument(
        "--overhead-debug-dir",
        type=Path,
        default=None,
        help="directory for overhead verification images "
        "(default: <dataset-root>/overhead_debug)",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the 3D MuJoCo viewer (default: off, headless)",
    )
    parser.add_argument(
        "--preflight-debug",
        action="store_true",
        help="print detailed collision diagnostics for rejected trajectory candidates",
    )
    parser.add_argument(
        "--preflight-debug-limit",
        type=int,
        default=12,
        help="maximum detailed contact rows to print per rejected candidate",
    )
    parser.add_argument(
        "--save-failed-trajectories",
        type=Path,
        default=None,
        help="directory for replayable .npz rollouts of rejected preflight candidates",
    )
    parser.add_argument(
        "--failed-trajectory-limit",
        type=int,
        default=8,
        help="maximum rejected candidates to save",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="output directory (default: episodes/<timestamp> for videos or datasets/<timestamp>)",
    )
    parser.add_argument(
        "--recording-format",
        choices=("videos", "dataset"),
        default="videos",
        help="write synced annotated episode MP4s or a LeRobotDataset (default: videos)",
    )
    parser.add_argument(
        "--video-rest-to-rest",
        action="store_true",
        help="for website videos, finish preflight detection before recording, then record "
        "each episode from REST back to REST without a post-run placement scan",
    )
    parser.add_argument(
        "--repo-id",
        default="local/pick-and-place-so101",
        help="dataset repo id stored in metadata",
    )
    parser.add_argument(
        "--task",
        default="Pick up the cube and place it at the target.",
        help="natural-language task instruction saved with every frame",
    )
    parser.add_argument(
        "--vcodec",
        default="auto",
        help="LeRobot video codec (default: auto = best available HW encoder, "
        "e.g. h264_videotoolbox on macOS)",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video in real time during capture (near-instant save_episode, "
        "no PNG scratch files); --no-streaming-encoding falls back to PNG-then-encode",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads LeRobot uses for PNG-then-encode mode",
    )
    args = parser.parse_args()
    if args.workspace_audio and args.workspace_camera is None:
        parser.error("--workspace-audio requires --workspace-camera")
    if args.workspace_audio and args.recording_format != "videos":
        parser.error("--workspace-audio is available only with --recording-format videos")
    if args.live_videos and args.recording_format != "videos":
        parser.error("--live-videos requires --recording-format videos")
    if args.live_videos and args.workspace_camera is None:
        parser.error("--live-videos requires --workspace-camera")
    if args.video_rest_to_rest and args.recording_format != "videos":
        parser.error("--video-rest-to-rest requires --recording-format videos")
    if args.target_change_min_distance < 0:
        parser.error("--target-change-min-distance must be non-negative")
    if args.target_change_alert_min_seconds <= 0:
        parser.error("--target-change-alert-min-seconds must be positive")
    if args.target_change_alert_max_seconds < args.target_change_alert_min_seconds:
        parser.error("--target-change-alert-max-seconds must be at least the minimum")
    notifier = OperatorNotifier(
        enabled=args.operator_alerts,
        sound_path=args.alert_sound,
    )

    import cv2

    from pick_and_place.cam_align_solve import (
        ExtrinsicsSolveError,
        apply_solve_result,
        check_solve_plausible,
        parse_index_or_path,
        pose_delta_mm_deg,
        solve_overhead_extrinsics,
    )
    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )
    from pick_and_place.workspace_overlays import PAN_AXIS

    # One persistent scene for the whole loop: the cube is a freejoint that
    # prepare_episode repositions per episode, so a single viewer can stay bound.
    print("Building scene...")
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(
        dummy_source,
        include_environment=True,
        paper_target_marker=True,
    )
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)

    kinematics = derive_kinematics(model)
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    rng = np.random.default_rng()
    # Set by the startup overhead solve; the cooldown drift check compares against it.
    startup_extrinsics: tuple[np.ndarray, np.ndarray] | None = None
    # Cooldowns require the operator to move away from the most recent
    # successfully completed episode's target before the next episode is planned.
    cooldown_reference_target: CubePose | None = None

    # Refuse to start unless the full rig is present: both cameras open and both
    # have calibrated intrinsics. The overhead extrinsics are solved from the
    # workspace-frame tags once the viewer is up; the wrist camera is opened per
    # episode by the executor, so it is only probed here.
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR

    def intrinsics_path(camera_name: str, override) -> Path:
        path = Path(override) if override is not None else LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
        if not path.exists():
            raise SystemExit(f"Missing {camera_name} intrinsics at {path}. Calibrate the camera first.")
        return path

    overhead_intrinsics = intrinsics_path(args.camera_name, args.overhead_intrinsics)
    wrist_intrinsics = intrinsics_path("wrist_camera", args.wrist_intrinsics)
    workspace_intrinsics = (
        intrinsics_path("workspace_camera", args.workspace_intrinsics)
        if args.workspace_camera is not None
        else None
    )

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    print("Opening overhead camera...")
    overhead_cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    overhead_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    overhead_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not overhead_cap.isOpened():
        overhead_cap.release()
        raise SystemExit(f"Could not open the overhead camera {args.camera!r}.")

    workspace_cap = None
    if args.workspace_camera is not None:
        print("Opening workspace camera...")
        workspace_cap = cv2.VideoCapture(parse_index_or_path(args.workspace_camera), backend)
        workspace_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        workspace_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not workspace_cap.isOpened():
            workspace_cap.release()
            overhead_cap.release()
            raise SystemExit(f"Could not open the workspace camera {args.workspace_camera!r}.")

    print("Checking wrist camera...")
    wrist_probe = cv2.VideoCapture(parse_index_or_path(args.wrist_camera), backend)
    wrist_open = wrist_probe.isOpened()
    wrist_probe.release()
    if not wrist_open:
        overhead_cap.release()
        if workspace_cap is not None:
            workspace_cap.release()
        raise SystemExit(f"Could not open the wrist camera {args.wrist_camera!r}.")

    print("Connecting to follower...")
    # Keep torque on a plain disconnect (crash / mid-loop exit) so the arm holds
    # rather than going limp; torque is only released deliberately at REST.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    if args.no_joint_zero_correction:
        joint_offsets: dict[str, float] | None = None
        print("Joint-zero correction disabled; commanding raw servo angles.")
    elif args.joint_zeros.exists():
        joint_offsets = load_joint_zero_offsets(args.joint_zeros)
        pretty = ", ".join(f"{k}={v:+.2f}" for k, v in joint_offsets.items())
        print(f"Applying session joint-zero correction from {args.joint_zeros}: {pretty} deg")
    else:
        joint_offsets = None
        print(
            f"No joint-zero calibration at {args.joint_zeros}; commanding raw servo angles. "
            "Run scripts/calibrate_joint_zeros.py at session start."
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(__file__).resolve().parents[2]
        / ("episodes" if args.recording_format == "videos" else "datasets")
        / timestamp
    )
    overhead_debug_dir = (
        args.overhead_debug_dir
        if args.overhead_debug_dir is not None
        else dataset_root / "overhead_debug"
    )
    if args.recording_format == "videos":
        recording = EpisodeVideoSession(
            root=dataset_root,
            task=args.task,
            fps=CONTROL_HZ,
            camera_intrinsics={
                "wrist": wrist_intrinsics,
                "overhead": overhead_intrinsics,
                **({"workspace": workspace_intrinsics} if workspace_intrinsics is not None else {}),
            },
            workspace_audio=args.workspace_audio,
            workspace_audio_device=(
                int(args.workspace_audio_device)
                if args.workspace_audio_device is not None
                and args.workspace_audio_device.isdecimal()
                else args.workspace_audio_device
            ),
            live_videos=args.live_videos,
        )
        print(f"Recording synchronized episode videos at: {dataset_root}")
    else:
        recording = RecordingSession(
            repo_id=args.repo_id,
            root=dataset_root,
            task=args.task,
            fps=CONTROL_HZ,
            vcodec=args.vcodec,
            streaming_encoding=args.streaming_encoding,
            image_writer_threads=args.image_writer_threads,
        )
        print(f"Recording into LeRobotDataset at: {dataset_root}")

    drop_zone_tracker = PaperTracker()

    def read_current_sim_pose() -> tuple[dict[str, float], float]:
        """Read the real arm and convert to the sim joint frame."""
        actual = action_to_joints(follower.get_observation(), clamp_low)
        return real_frame_to_sim(actual, joint_offsets)

    def move_to(arm_joints: dict[str, float], gripper: float, viewer) -> None:
        """Smoothly ramp the real arm and the sim onto ``arm_joints``/``gripper``."""
        target_real = clamp_and_warn(
            sim_frame_to_real(arm_joints, gripper, joint_offsets),
            clamp_low,
            clamp_high,
            clip_warned,
        )
        ramp_to_resting(
            follower, target_real, arm_joints, gripper, actuator_id, model, data, viewer
        )

    def open_gripper_in_place(viewer) -> None:
        """Open the gripper while holding the current arm pose (drop any cube)."""
        arm, _ = read_current_sim_pose()
        move_to(arm, GRIPPER_OPEN, viewer)

    def abort_to_near_neutral(viewer) -> None:
        """Recover from a failed episode: drop the cube and re-home near neutral."""
        print("Aborting episode: opening gripper and re-homing to near-neutral...")
        open_gripper_in_place(viewer)
        arm, grip = sample_near_neutral(rng)
        move_to(arm, grip, viewer)

    def check_overhead_drift() -> None:
        """Re-solve the overhead extrinsics from the current (near-neutral) pose and
        stop the run if the camera has drifted from the startup calibration. Skips
        quietly if the tags are occluded."""
        if not args.recalibrate or startup_extrinsics is None:
            return
        print("Cooldown drift check: re-solving overhead extrinsics...")
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
        saved_pos = model.cam_pos[cam_id].copy()
        saved_quat = model.cam_quat[cam_id].copy()
        check = solve_overhead_extrinsics(
            model,
            data,
            overhead_cap,
            camera_name=args.camera_name,
            intrinsics_path=args.overhead_intrinsics,
            samples=args.recalibrate_samples,
            max_seconds=args.recalibrate_max_seconds,
            cv2_module=cv2,
        )
        # The re-solve only decides whether to stop; the startup calibration stays
        # live and is never re-applied mid-run.
        model.cam_pos[cam_id] = saved_pos
        model.cam_quat[cam_id] = saved_quat
        mujoco.mj_forward(model, data)
        if check is None:
            print("Drift check skipped: could not see all four tags (occluded). Continuing.")
            return
        drift_mm, drift_deg = pose_delta_mm_deg(
            startup_extrinsics[0],
            startup_extrinsics[1],
            np.array(check.pos, dtype=float),
            np.array(check.quat, dtype=float),
        )
        print(f"Overhead drift vs startup: {drift_mm:.1f}mm / {drift_deg:.2f}deg.")
        if drift_mm > args.recalibrate_drift_mm or drift_deg > args.recalibrate_drift_deg:
            raise SystemExit(
                f"Overhead camera drifted {drift_mm:.1f}mm / {drift_deg:.2f}deg since startup "
                f"(limits {args.recalibrate_drift_mm:.0f}mm / {args.recalibrate_drift_deg:.1f}deg). "
                "Stopping so the operator can check the mount and recalibrate."
            )

    def target_distance(a: CubePose, b: CubePose) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def wait_for_rest_and_target_plate_change(viewer, min_rest_until: float) -> None:
        """Stay at REST with torque off until cooldown and target movement are done."""
        reference = cooldown_reference_target
        if reference is None or args.target_change_min_distance == 0:
            remaining = min_rest_until - time.time()
            if remaining > 0.0:
                time.sleep(remaining)
            return

        def look_from_rest_pose() -> CubePose | None:
            if not viewer.is_running():
                return None
            print("Checking target plate movement while resting with torque off...")
            return track_drop_zone_square(
                overhead_cap,
                args.camera_name,
                model,
                data,
                drop_zone_tracker,
                args.drop_zone_color,
            )

        threshold = args.target_change_min_distance
        poll_interval = 1.0
        alert_interval = args.target_change_alert_min_seconds
        next_alert_time = time.time()
        notifier.alert(
            "Please move the target plate to a substantially different position.",
            repeat_sound=2,
        )
        while viewer.is_running():
            target = look_from_rest_pose()
            if not viewer.is_running():
                return
            if target is None:
                if time.time() >= next_alert_time:
                    notifier.alert(
                        "Target plate is not visible. Move it into view before the run can continue."
                    )
                    next_alert_time = time.time() + alert_interval
                    alert_interval = min(
                        alert_interval * 2.0,
                        args.target_change_alert_max_seconds,
                    )
            else:
                moved = target_distance(reference, target)
                if moved >= threshold:
                    remaining = min_rest_until - time.time()
                    if remaining > 0.0:
                        print(
                            f"Target plate moved {moved * 100.0:.1f}cm "
                            f"(required {threshold * 100.0:.1f}cm). "
                            f"Finishing {remaining:.0f}s rest."
                        )
                        time.sleep(remaining)
                    else:
                        print(
                            f"Target plate moved {moved * 100.0:.1f}cm "
                            f"(required {threshold * 100.0:.1f}cm). Resuming."
                        )
                    return
                if time.time() >= next_alert_time:
                    notifier.alert(
                        "Target plate has not moved enough. Move it to a new position before "
                        "the run can continue."
                    )
                    print(
                        f"Plate movement: {moved * 100.0:.1f}cm "
                        f"(required {threshold * 100.0:.1f}cm); "
                        f"from ({reference.x:.3f}, {reference.y:.3f}) "
                        f"to ({target.x:.3f}, {target.y:.3f})."
                    )
                    next_alert_time = time.time() + alert_interval
                    alert_interval = min(
                        alert_interval * 2.0,
                        args.target_change_alert_max_seconds,
                    )
            time.sleep(poll_interval)

    def cooldown(viewer) -> None:
        """Park at REST with torque off for the cooldown, then re-home near neutral.
        A long-enough cooldown doubles as an overhead-camera drift check. The
        cooldown also gives the operator time to move the target plate; the run
        remains paused until that movement is confirmed by the overhead camera."""
        print(f"Cooldown: resting with torque off for {args.rest_duration:.0f}s...")
        if cooldown_reference_target is not None and args.target_change_min_distance > 0:
            notifier.alert(
                "Cooldown started. Move the target plate before the next episode.",
                repeat_sound=2,
            )
        move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
        follower.bus.disable_torque()
        wait_for_rest_and_target_plate_change(viewer, time.time() + args.rest_duration)
        if not viewer.is_running():
            return
        follower.bus.enable_torque()
        arm, grip = sample_near_neutral(rng)
        move_to(arm, grip, viewer)
        if (
            args.recalibrate
            and args.recalibrate_check_min_cooldown > 0
            and args.rest_duration >= args.recalibrate_check_min_cooldown
        ):
            check_overhead_drift()

    def park_from_interrupt() -> None:
        """User ended the loop: park to NEUTRAL, then REST. The real viewer has
        already been torn down by the `with`, so park against a mock one. Make
        sure torque is on first — a Ctrl-C during the cooldown sleep leaves it
        off, and parking commands would be ignored by a limp arm."""
        nonlocal ended_at_rest
        print("\nCtrl-C: parking to NEUTRAL then REST...")
        try:
            follower.bus.enable_torque()
        except Exception as exc:  # noqa: BLE001 - best-effort re-enable before parking
            print(f"Warning: could not enable torque before parking: {exc}")
        park = MockViewer()
        move_to(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, park)
        move_to(REST_ARM_JOINTS, REST_GRIPPER, park)
        ended_at_rest = True

    from pick_and_place.workspace_overlays import (
        is_cube_pickup_allowed,
        is_cube_recovery_target_allowed,
    )

    def pan_and_track_cube(
        viewer,
        *,
        tries: int,
        free_grasp: bool = False,
        return_out_of_zone: bool = False,
        debug: OverheadDetectionDebug | None = None,
    ) -> CubePose | None:
        """Look for the cube from the current pose, panning to fresh near-neutral
        search poses up to ``tries`` times. Returns the cube pose or ``None``."""
        for attempt in range(tries):
            if not viewer.is_running():
                return None
            if attempt > 0:
                arm, grip = sample_hunt_pose(rng)
                print(f"Look {attempt + 1}/{tries}: panning to a new search pose...")
                move_to(arm, grip, viewer)
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"Look {attempt + 1}/{tries}: searching from current pose...")
            source = track_cube(
                overhead_cap,
                args.camera_name,
                model,
                data,
                CUBE_LOOK_TIMEOUT,
                free_grasp=free_grasp,
                return_out_of_zone=return_out_of_zone,
                debug=debug,
            )
            if source is not None:
                return source
        return None

    def hunt_for_cube(
        viewer,
        *,
        free_grasp: bool = False,
        return_out_of_zone: bool = False,
        debug: OverheadDetectionDebug | None = None,
    ) -> CubePose | None:
        """Look for the cube from the current pose, re-homing near neutral up to
        ``--max-hunt-tries`` times. Returns the cube pose or ``None`` if not found."""
        if args.source is not None and not free_grasp:
            source = CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
            if debug is not None:
                debug.cube = source
            return source
        return pan_and_track_cube(
            viewer,
            tries=args.max_hunt_tries,
            free_grasp=free_grasp,
            return_out_of_zone=return_out_of_zone,
            debug=debug,
        )

    def hunt_for_drop_zone(
        viewer,
        *,
        debug: OverheadDetectionDebug | None = None,
    ) -> CubePose | None:
        """Look for the drop-zone square on the overhead camera.

        The arm can sit between the overhead camera and the square, so we re-home
        to fresh near-neutral poses up to ``--max-hunt-tries`` times to clear the
        view, exactly like ``hunt_for_cube``. Returns the target or ``None``."""
        tries = args.max_hunt_tries
        for attempt in range(tries):
            if not viewer.is_running():
                return None
            if attempt > 0:
                arm, grip = sample_hunt_pose(rng)
                print(
                    f"Drop-zone look {attempt + 1}/{tries}: panning to a new search pose..."
                )
                move_to(arm, grip, viewer)
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"Drop-zone look {attempt + 1}/{tries}: searching from current pose...")
            target = track_drop_zone_square(
                overhead_cap,
                args.camera_name,
                model,
                data,
                drop_zone_tracker,
                args.drop_zone_color,
                debug=debug,
            )
            if target is not None:
                return target
        return None

    def recover_cube(viewer) -> CubePose | None:
        """Move the cube to a fresh random source pose before the next episode."""
        for recovery_attempt in range(1, CUBE_RECOVERY_MAX_ATTEMPTS + 1):
            print(
                f"\n--- Cube recovery {recovery_attempt}/{CUBE_RECOVERY_MAX_ATTEMPTS} "
                "(not recorded) ---"
            )
            recovery_source = hunt_for_cube(
                viewer,
                free_grasp=True,
                return_out_of_zone=True,
            )
            if not viewer.is_running():
                return None
            if recovery_source is None:
                print("Cube recovery could not locate the cube.")
                continue

            current_joints, current_gripper = read_current_sim_pose()
            with recover_on(
                EpisodeSamplingError,
                EpisodeAborted,
                recover=lambda: abort_to_near_neutral(viewer),
            ):
                try:
                    recovery = prepare_episode(
                        rng,
                        recovery_source,
                        start_joints=current_joints,
                        start_gripper=current_gripper,
                        model=model,
                        data=data,
                        max_attempts=EPISODE_MAX_ATTEMPTS,
                        verbose=True,
                        include_environment=True,
                        preflight_debug=args.preflight_debug,
                        preflight_debug_limit=args.preflight_debug_limit,
                        failed_trajectory_dir=args.save_failed_trajectories,
                        failed_trajectory_limit=args.failed_trajectory_limit,
                        free_grasp=True,
                        target_sampler=sample_recovery_cube,
                    )
                except EpisodeSamplingError:
                    print("Cube recovery: no feasible relocation plan.")
                    raise

                print(
                    f"Recovering cube to pickup zone "
                    f"({recovery.target.x:.3f}, {recovery.target.y:.3f})."
                )
                status = execute_episode(
                    recovery,
                    follower=follower,
                    viewer=viewer,
                    speed=args.speed,
                    wrist_camera=args.wrist_camera,
                    wrist_intrinsics=args.wrist_intrinsics,
                    show_wrist_cam=args.show_wrist_cam,
                    show_wrist_mixed=args.show_wrist_mixed,
                    failed_trajectory_dir=args.save_failed_trajectories,
                    free_grasp=True,
                    pickup_empty_gripper_position=args.pickup_empty_gripper_position,
                    pickup_gripper_margin=args.pickup_gripper_margin,
                    joint_offsets_deg=joint_offsets,
                )
                if status == "restart":
                    raise EpisodeAborted

                recovered = hunt_for_cube(viewer, return_out_of_zone=True)
                if recovered is None:
                    print("Cube recovery completed, but the cube could not be located afterward.")
                    continue
                if is_cube_recovery_target_allowed(recovered.x, recovered.y):
                    print(
                        "Cube recovery verified pickup start "
                        f"({recovered.x:.3f}, {recovered.y:.3f})."
                    )
                    return recovered
                print(
                    "Cube recovery landed too close to/outside the recovery-safe pickup zone "
                    f"({recovered.x:.3f}, {recovered.y:.3f}); retrying relocation."
                )

        return None

    disable_viewer = (not args.viewer) or (
        (args.show_wrist_cam or args.show_wrist_mixed) and sys.platform == "darwin"
    )
    viewer_ctx = MockViewer() if disable_viewer else mujoco.viewer.launch_passive(model, data)

    # Run-level session: home near-neutral on entry; whatever ends the run
    # (episode budget met, cube lost, viewer closed, or Ctrl-C) flows to REST
    # before the hardware is released in the `finally` below.
    ended_at_rest = False
    try:
        with recover_on(KeyboardInterrupt, recover=park_from_interrupt):
            with viewer_ctx as viewer:
                if args.recalibrate:
                    print("Solving overhead camera extrinsics from the workspace-frame tags...")
                    result = solve_overhead_extrinsics(
                        model,
                        data,
                        overhead_cap,
                        camera_name=args.camera_name,
                        intrinsics_path=args.overhead_intrinsics,
                        samples=args.recalibrate_samples,
                        max_seconds=args.recalibrate_max_seconds,
                        cv2_module=cv2,
                    )
                    if result is None:
                        raise SystemExit(
                            "Overhead calibration failed: never saw all four workspace-frame "
                            "tags in one frame. Clear the camera view and check the tags."
                        )
                    try:
                        check_solve_plausible(result)
                    except ExtrinsicsSolveError as exc:
                        raise SystemExit(f"Overhead calibration rejected: {exc}") from exc
                    apply_solve_result(model, data, args.camera_name, result)
                    startup_extrinsics = (
                        np.array(result.pos, dtype=float),
                        np.array(result.quat, dtype=float),
                    )
                    print(
                        f"Overhead extrinsics solved: {result.reprojection_error_px:.2f}px, "
                        f"{result.nominal_delta.translation_m * 1000.0:.1f}mm / "
                        f"{result.nominal_delta.rotation_deg:.2f}deg from nominal."
                    )

                print("Homing to near-neutral...")
                arm, grip = sample_near_neutral(rng)
                move_to(arm, grip, viewer)

                for ep in episode_loop(
                    target=args.episodes,
                    rest_every=args.rest_every,
                    cooldown=lambda: cooldown(viewer),
                    should_continue=viewer.is_running,
                ):
                    overhead_debug = empty_overhead_debug()
                    episode_target = hunt_for_drop_zone(viewer, debug=overhead_debug)
                    if not viewer.is_running():
                        break
                    if episode_target is None:
                        print(
                            "Drop zone square not found; checking whether the cube is covering it..."
                        )
                        covering_cube = hunt_for_cube(
                            viewer,
                            return_out_of_zone=True,
                            debug=overhead_debug,
                        )
                        if not viewer.is_running():
                            break
                        if covering_cube is not None:
                            notifier.alert(
                                "Drop zone is hidden but the cube is visible. "
                                "Running an unrecorded recovery."
                            )
                            recovered_source = recover_cube(viewer)
                            if recovered_source is None:
                                notifier.alert(
                                    "Cube recovery failed after retries. The run is stopping."
                                )
                                break
                            continue
                        notifier.alert(
                            f"Drop zone square not found after {args.max_hunt_tries} looks. "
                            "The run is stopping."
                        )
                        break

                    source = hunt_for_cube(
                        viewer,
                        return_out_of_zone=True,
                        debug=overhead_debug,
                    )
                    if not viewer.is_running():
                        break
                    if source is None:
                        notifier.alert(
                            f"Cube not found after {args.max_hunt_tries} looks. "
                            "The run is stopping."
                        )
                        break
                    if not is_cube_pickup_allowed(source.x, source.y):
                        notifier.alert(
                            "Cube is outside the pickup zone. Running an unrecorded recovery."
                        )
                        recovered_source = recover_cube(viewer)
                        if recovered_source is None:
                            notifier.alert("Cube recovery failed after retries. The run is stopping.")
                            break
                        source = recovered_source
                        if overhead_debug is not None:
                            overhead_debug.cube = source
                        if not viewer.is_running():
                            break

                    current_joints, current_gripper = read_current_sim_pose()
                    # Per-episode guard: an infeasible plan or a failed checkpoint
                    # replan opens the gripper and re-homes, then this attempt is
                    # abandoned without calling `ep.complete()`.
                    with recover_on(
                        EpisodeSamplingError, EpisodeAborted,
                        recover=lambda: abort_to_near_neutral(viewer),
                    ):
                        try:
                            episode = prepare_episode(
                                rng,
                                source,
                                episode_target,
                                start_joints=current_joints,
                                start_gripper=current_gripper,
                                model=model,
                                data=data,
                                max_attempts=EPISODE_MAX_ATTEMPTS,
                                verbose=True,
                                include_environment=True,
                                preflight_debug=args.preflight_debug,
                                preflight_debug_limit=args.preflight_debug_limit,
                                failed_trajectory_dir=args.save_failed_trajectories,
                                failed_trajectory_limit=args.failed_trajectory_limit,
                            )
                        except EpisodeSamplingError:
                            notifier.alert(
                                "No feasible plan from the current pose. Re-homing the arm."
                            )
                            print("No feasible plan from the current pose.")
                            raise

                        print(f"\n--- Episode {ep.index}"
                              f"{f'/{args.episodes}' if args.episodes else ''} ---")
                        initial_overhead_debug = overhead_debug
                        preflight_debug_written = False

                        if args.video_rest_to_rest:
                            # Detection and collision-checked planning are complete. Move
                            # to REST before the recorder begins; the executor records the
                            # REST-to-start and final-pose-to-REST transitions separately.
                            print("Preflight complete. Moving to REST before recording...")
                            move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)

                        if args.video_rest_to_rest and (
                            args.save_overhead_debug and initial_overhead_debug.bgr.size
                        ):
                            path = overhead_debug_dir / f"episode_{ep.index:05d}_preflight.jpg"
                            write_overhead_debug_image(path, initial_overhead_debug)
                            print(f"Saved overhead preflight debug image: {path}")
                            preflight_debug_written = True

                        def check_final_placement() -> dict[str, object]:
                            nonlocal preflight_debug_written
                            metadata = cube_pose_metadata(episode.source, episode.target)
                            metadata.update(driver_metadata("analytic"))

                            if (
                                args.save_overhead_debug
                                and initial_overhead_debug.bgr.size
                                and not preflight_debug_written
                            ):
                                path = (
                                    overhead_debug_dir
                                    / f"episode_{ep.index:05d}_preflight.jpg"
                                )
                                write_overhead_debug_image(path, initial_overhead_debug)
                                print(f"Saved overhead preflight debug image: {path}")
                                preflight_debug_written = True

                            print("Checking final cube placement from the overhead camera...")
                            final_debug = OverheadDetectionDebug(
                                bgr=initial_overhead_debug.bgr,
                                camera_matrix=initial_overhead_debug.camera_matrix,
                                camera_position=initial_overhead_debug.camera_position,
                                camera_rotation=initial_overhead_debug.camera_rotation,
                                target=initial_overhead_debug.target,
                            )
                            try:
                                # The arm can sit between the overhead camera and
                                # the placed cube; pan through fresh search poses
                                # (unrecorded) until it comes into view, exactly
                                # like the pre-episode cube hunt.
                                final_cube = pan_and_track_cube(
                                    viewer,
                                    tries=args.max_hunt_tries,
                                    return_out_of_zone=True,
                                    debug=final_debug,
                                )
                            except Exception as exc:
                                print(f"placement error: final cube check failed: {exc}")
                                if args.save_overhead_debug and final_debug.bgr.size:
                                    path = (
                                        overhead_debug_dir
                                        / f"episode_{ep.index:05d}_final_failed.jpg"
                                    )
                                    write_overhead_debug_image(
                                        path,
                                        final_debug,
                                        show_distance=True,
                                    )
                                    print(f"Saved overhead final debug image: {path}")
                                metadata.update(final_placement_metadata(None, episode.target))
                                return metadata
                            if args.save_overhead_debug and final_debug.bgr.size:
                                path = overhead_debug_dir / f"episode_{ep.index:05d}_final.jpg"
                                write_overhead_debug_image(
                                    path,
                                    final_debug,
                                    show_distance=final_cube is not None,
                                )
                                print(f"Saved overhead final debug image: {path}")
                            metadata.update(final_placement_metadata(final_cube, episode.target))
                            return metadata

                        status = execute_episode(
                            episode,
                            follower=follower,
                            viewer=viewer,
                            recording=recording,
                            overhead_camera_cap=overhead_cap,
                            workspace_camera_cap=workspace_cap,
                            speed=args.speed,
                            wrist_camera=args.wrist_camera,
                            wrist_intrinsics=args.wrist_intrinsics,
                            show_wrist_cam=args.show_wrist_cam,
                            show_wrist_mixed=args.show_wrist_mixed,
                            failed_trajectory_dir=args.save_failed_trajectories,
                            pickup_empty_gripper_position=args.pickup_empty_gripper_position,
                            pickup_gripper_margin=args.pickup_gripper_margin,
                            success_metadata=(
                                (lambda: {
                                    **cube_pose_metadata(episode.source, episode.target),
                                    **driver_metadata("analytic"),
                                    **final_placement_metadata(None, episode.target),
                                })
                                if args.video_rest_to_rest
                                else check_final_placement
                            ),
                            record_rest_to_rest=args.video_rest_to_rest,
                            joint_offsets_deg=joint_offsets,
                        )

                        if status == "restart":
                            notifier.alert("Episode restarted or aborted. Re-homing the arm.")
                            raise EpisodeAborted

                        ep.complete()
                        cooldown_reference_target = episode_target
                        is_last = args.episodes != 0 and ep.index >= args.episodes
                        if not is_last:
                            if recover_cube(viewer) is None:
                                notifier.alert("Cube recovery failed after retries. The run is stopping.")
                                break

                # Normal end (episode budget met, cube lost, or viewer closed): the
                # arm is at the last near-neutral pose — flow it straight to REST.
                if viewer.is_running():
                    notifier.alert("Collection loop is done. Moving the arm to rest.")
                    print("Loop done. Moving to REST...")
                    move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
                    ended_at_rest = True
    finally:
        if recording.initialized:
            print("Finalizing dataset...")
            recording.finalize()
            print(f"Dataset written to {dataset_root}")
        if ended_at_rest:
            print("At REST — releasing torque.")
            try:
                follower.bus.disable_torque()
            except Exception as exc:  # noqa: BLE001 - best-effort torque release
                print(f"Warning: could not release torque: {exc}")
        print("Disconnecting hardware...")
        follower.disconnect()
        if overhead_cap is not None:
            overhead_cap.release()
        if workspace_cap is not None:
            workspace_cap.release()


if __name__ == "__main__":
    main()
