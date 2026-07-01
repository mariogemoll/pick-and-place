#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the analytic pick-and-place on the physical SO-101 as a continuous loop.

Homes the arm to a near-neutral pose, then repeats: look for the cube from the
current pose (re-homing to fresh near-neutral poses if it can't be seen), plan a
collision-free ``pick_and_carry`` episode from wherever the arm currently is, and
run it on the real arm via ``pick_and_place.executor``. A failed plan or a failed
checkpoint replan aborts the episode and re-homes. Every ``--rest-every`` episodes
the arm takes a torque-off cooldown at REST. Press Ctrl-C to stop: the arm parks
to NEUTRAL, then REST, and releases torque.

The runner owns the follower, the viewer and the cameras and keeps them alive
across the whole loop; ``execute_episode`` runs a single pass against them. For
sim-only playback (no arm) use ``sim.py``.

Every run records straight into a LeRobotDataset (``datasets/<timestamp>/`` by
default): each successful episode is committed with one frame per control tick
(measured joints as state, commanded joints as action, plus the wrist and
overhead camera frames); aborted/restarted episodes are discarded. See
``execute_episode``'s docstring.
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

from pick_and_place.episode_loop import episode_loop
from pick_and_place.episodes import (
    EpisodeSamplingError,
    _build_model,
    prepare_episode,
    sample_cube,
    sample_hunt_pose,
    sample_near_neutral,
)
from pick_and_place.executor import (
    CONTROL_HZ,
    HARDWARE_SIMULATION_HZ,
    REAL_ARM_DEFAULT_SPEED,
    RecordingSession,
    clamp_and_warn,
    execute_episode,
    follower_clamp_limits,
    ramp_to_resting,
)
from pick_and_place.follower import (
    action_to_joints,
    load_follower_joint_offsets,
    make_so101_follower,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.paper_detection import (
    PaperTracker,
    detect_paper_target,
    set_paper_target_marker,
)
from pick_and_place.safety import EpisodeAborted, recover_on
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)

# How long a single look attempt stares at the camera feed before giving up and
# moving to a fresh near-neutral pose to try again.
CUBE_LOOK_TIMEOUT = 2.0
# Plan-search budget per episode: how many source/target/end resamples to try
# before declaring the cube unreachable from the current pose and aborting.
EPISODE_MAX_ATTEMPTS = 40
# Cube-recovery relocation retries. The move is unrecorded but required so
# unattended collection can continue from a cube location that is usable for the
# next recorded pickup.
CUBE_RECOVERY_MAX_ATTEMPTS = 3


class MockViewer:
    """Stand-in viewer for headless runs and for parking after the real viewer
    has closed. Exposes the ``is_running``/``sync`` surface the ramps rely on."""

    def __init__(self) -> None:
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def sync(self) -> None:
        pass

    def close(self) -> None:
        self._running = False

    def __enter__(self) -> "MockViewer":
        return self

    def __exit__(self, *args) -> None:
        pass


def track_cube(
    cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    timeout: float,
    *,
    free_grasp: bool = False,
    return_out_of_zone: bool = False,
) -> CubePose | None:
    """Look for the cube on ``cap`` for up to ``timeout`` seconds.

    Returns the cube pose if it is detected inside the allowed workspace, or
    when ``return_out_of_zone`` is set. Otherwise returns ``None`` if nothing
    usable is seen before the timeout (not visible, or outside the workspace)."""
    import cv2
    from scipy.spatial.transform import Rotation

    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.cube_detection import (
        cube_pose_to_world,
        estimate_cube_pose,
        make_cube_detector,
    )
    from pick_and_place.workspace_overlays import is_cube_pickup_allowed

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()

    intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
    if intrinsics.exists():
        camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)
    else:
        focal = (1080 / 2.0) / np.tan(np.radians(model.cam_fovy[camera_id]) / 2.0)
        camera_matrix = np.array(
            [[focal, 0, 1920 / 2.0], [0, focal, 1080 / 2.0], [0, 0, 1]], dtype=float
        )
        undistort_map = None

    detector = make_cube_detector()

    # Flush a few stale frames so we read what the arm sees now, not buffered.
    for _ in range(5):
        cap.read()

    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)

        estimate = estimate_cube_pose(rgb, detector, camera_matrix)
        if estimate is None:
            continue

        rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
        allowed = is_cube_pickup_allowed
        if not allowed(float(position[0]), float(position[1])):
            print(
                f"Cube seen at ({position[0]:.3f}, {position[1]:.3f}) but outside the "
                "allowed pick-up zone."
            )
            if not return_out_of_zone:
                continue

        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
        print(f"Tracked cube: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        return CubePose(
            x=float(position[0]),
            y=float(position[1]),
            z=CUBE_HALF_SIZE,
            roll=float(roll) if free_grasp else 0.0,
            pitch=float(pitch) if free_grasp else 0.0,
            yaw=float(yaw),
        )

    return None


def track_drop_zone_square(
    cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tracker: PaperTracker,
    target_color: str,
    timeout: float = CUBE_LOOK_TIMEOUT,
) -> CubePose | None:
    """Look for a black/white drop-zone square and return its center as a target."""
    import cv2

    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.workspace_overlays import (
        is_cube_drop_allowed,
        workspace_interior_corners_world,
    )

    workspace_corners = workspace_interior_corners_world()

    # This function is called once per episode. Do not let the tracker's cached
    # estimate satisfy a new episode when the square is no longer visible.
    tracker.reset()

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()

    intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
    if intrinsics.exists():
        camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)
    else:
        focal = (1080 / 2.0) / np.tan(np.radians(model.cam_fovy[camera_id]) / 2.0)
        camera_matrix = np.array(
            [[focal, 0, 1920 / 2.0], [0, focal, 1080 / 2.0], [0, 0, 1]], dtype=float
        )
        undistort_map = None

    for _ in range(5):
        cap.read()

    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)

        raw_target = detect_paper_target(
            rgb,
            camera_matrix,
            cam_pos,
            cam_rot,
            target_color=target_color,
            workspace_corners_world=workspace_corners,
        )
        target = tracker.update(raw_target)
        if target is None:
            continue

        usable = is_cube_drop_allowed(*target.xy)
        set_paper_target_marker(model, data, target, usable=usable)
        if not usable:
            print(
                f"Drop zone seen at ({target.xy[0]:.3f}, {target.xy[1]:.3f}) "
                "but outside the allowed drop zone."
            )
            continue

        print(f"Tracked drop zone: pos=({target.xy[0]:.3f}, {target.xy[1]:.3f})")
        return CubePose(x=target.xy[0], y=target.xy[1], z=CUBE_HALF_SIZE)

    return None


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
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the source cube (x, y) on the floor; omit to track it with the camera",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the target (x, y); omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--target-drop-zone",
        dest="target_drop_zone",
        action="store_true",
        help="use the overhead camera to set the target from a black/white drop-zone square",
    )
    parser.add_argument(
        "--show-drop-zone",
        dest="show_drop_zone",
        action="store_true",
        help="show the tracked drop-zone square marker in the MuJoCo viewer",
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
        "--offsets-path",
        default=None,
        help="JSON of per-joint sim→real degree offsets (default: zero offsets)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="playback speed multiplier of nominal pace "
        f"(1.0 = nominal; default {REAL_ARM_DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--environment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the calibration workspace_frame and overhead camera mount",
    )
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera (default: 0)")
    parser.add_argument("--camera-name", default="overhead_camera", help="camera name in the model")
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera (default: 1)")
    parser.add_argument("--wrist-intrinsics", default=None, help="path to wrist camera intrinsics JSON")
    parser.add_argument("--show-wrist-cam", action="store_true", help="show the live wrist camera feed")
    parser.add_argument("--show-wrist-mixed", action="store_true", help="overlay the sim render on the wrist feed")
    parser.add_argument("--no-viewer", action="store_true", help="run headless (no 3D MuJoCo viewer)")
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
        help="output dir for the LeRobotDataset (default: datasets/<timestamp>)",
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

    if args.target is not None and args.target_drop_zone:
        parser.error("--target and --target-drop-zone are mutually exclusive")

    import cv2

    from pick_and_place.cam_align_solve import parse_index_or_path
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
        include_environment=args.environment,
        paper_target_marker=args.target_drop_zone or args.show_drop_zone,
    )
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)

    kinematics = derive_kinematics(model)
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    offsets = load_follower_joint_offsets(args.offsets_path)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    rng = np.random.default_rng()

    print("Connecting to follower...")
    # Keep torque on a plain disconnect (crash / mid-loop exit) so the arm holds
    # rather than going limp; torque is only released deliberately at REST.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(__file__).resolve().parents[2] / "datasets" / timestamp
    )
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

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    print("Opening overhead camera...")
    overhead_cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    overhead_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    overhead_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    fixed_target = (
        CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
        if args.target is not None
        else None
    )
    drop_zone_tracker = PaperTracker() if (args.target_drop_zone or args.show_drop_zone) else None

    def read_current_sim_pose() -> tuple[dict[str, float], float]:
        """Read the real arm and convert to the sim joint frame."""
        actual = action_to_joints(follower.get_observation(), clamp_low)
        return real_frame_to_sim(actual, offsets)

    def move_to(arm_joints: dict[str, float], gripper: float, viewer) -> None:
        """Smoothly ramp the real arm and the sim onto ``arm_joints``/``gripper``."""
        target_real = clamp_and_warn(
            sim_frame_to_real(arm_joints, gripper, offsets), clamp_low, clamp_high, clip_warned
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

    def cooldown(viewer) -> None:
        """Park at REST with torque off for the cooldown, then re-home near neutral."""
        print(f"Cooldown: resting with torque off for {args.rest_duration:.0f}s...")
        move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
        follower.bus.disable_torque()
        time.sleep(args.rest_duration)
        follower.bus.enable_torque()
        arm, grip = sample_near_neutral(rng)
        move_to(arm, grip, viewer)

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

    from pick_and_place.workspace_overlays import is_cube_pickup_allowed

    def hunt_for_cube(
        viewer,
        *,
        free_grasp: bool = False,
        return_out_of_zone: bool = False,
    ) -> CubePose | None:
        """Look for the cube from the current pose, re-homing near neutral up to
        ``--max-hunt-tries`` times. Returns the cube pose or ``None`` if not found."""
        if args.source is not None and not free_grasp:
            return CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
        for attempt in range(args.max_hunt_tries):
            if not viewer.is_running():
                return None
            if attempt > 0:
                arm, grip = sample_hunt_pose(rng)
                print(f"Look {attempt + 1}/{args.max_hunt_tries}: panning to a new search pose...")
                move_to(arm, grip, viewer)
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"Look {attempt + 1}/{args.max_hunt_tries}: searching from current pose...")
            source = track_cube(
                overhead_cap,
                args.camera_name,
                model,
                data,
                CUBE_LOOK_TIMEOUT,
                free_grasp=free_grasp,
                return_out_of_zone=return_out_of_zone,
            )
            if source is not None:
                return source
        return None

    def hunt_for_drop_zone(viewer, *, hunt: bool) -> CubePose | None:
        """Look for the drop-zone square on the overhead camera.

        The arm can sit between the overhead camera and the square, so when
        ``hunt`` is set we re-home to fresh near-neutral poses up to
        ``--max-hunt-tries`` times to clear the view, exactly like
        ``hunt_for_cube``. With ``hunt`` off (marker display only) we take a
        single look from the current pose. Returns the target or ``None``."""
        assert drop_zone_tracker is not None
        tries = args.max_hunt_tries if hunt else 1
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
            elif hunt:
                print(f"Drop-zone look {attempt + 1}/{tries}: searching from current pose...")
            target = track_drop_zone_square(
                overhead_cap,
                args.camera_name,
                model,
                data,
                drop_zone_tracker,
                args.drop_zone_color,
            )
            if target is not None:
                return target
        return None

    def recover_cube(viewer) -> bool:
        """Move the cube to a fresh random source pose before the next episode."""
        for recovery_attempt in range(1, CUBE_RECOVERY_MAX_ATTEMPTS + 1):
            print(
                f"\n--- Cube recovery {recovery_attempt}/{CUBE_RECOVERY_MAX_ATTEMPTS} "
                "(not recorded) ---"
            )
            recovery_source = hunt_for_cube(viewer, free_grasp=True)
            if not viewer.is_running():
                return False
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
                        include_environment=args.environment,
                        preflight_debug=args.preflight_debug,
                        preflight_debug_limit=args.preflight_debug_limit,
                        failed_trajectory_dir=args.save_failed_trajectories,
                        failed_trajectory_limit=args.failed_trajectory_limit,
                        free_grasp=True,
                        target_sampler=sample_cube,
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
                    offsets_path=args.offsets_path,
                    speed=args.speed,
                    wrist_camera=args.wrist_camera,
                    wrist_intrinsics=args.wrist_intrinsics,
                    show_wrist_cam=args.show_wrist_cam,
                    show_wrist_mixed=args.show_wrist_mixed,
                    failed_trajectory_dir=args.save_failed_trajectories,
                    free_grasp=True,
                )
                if status == "restart":
                    raise EpisodeAborted
                return True

        return False

    disable_viewer = args.no_viewer or (
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
                print("Homing to near-neutral...")
                arm, grip = sample_near_neutral(rng)
                move_to(arm, grip, viewer)

                for ep in episode_loop(
                    target=args.episodes,
                    rest_every=args.rest_every,
                    cooldown=lambda: cooldown(viewer),
                    should_continue=viewer.is_running,
                ):
                    episode_target = fixed_target
                    if drop_zone_tracker is not None:
                        tracked_target = hunt_for_drop_zone(
                            viewer, hunt=args.target_drop_zone
                        )
                        if not viewer.is_running():
                            break
                        if args.target_drop_zone:
                            if tracked_target is None:
                                print(
                                    f"Drop-zone square not found after "
                                    f"{args.max_hunt_tries} looks. Ending loop."
                                )
                                break
                            episode_target = tracked_target

                    source = hunt_for_cube(viewer, return_out_of_zone=True)
                    if not viewer.is_running():
                        break
                    if source is None:
                        print(f"Cube not found after {args.max_hunt_tries} looks. Ending loop.")
                        break
                    if not is_cube_pickup_allowed(source.x, source.y):
                        print("Cube needs recovery before the next recorded pickup.")
                        if not recover_cube(viewer):
                            print("Cube recovery failed after retries. Ending loop.")
                            break
                        source = hunt_for_cube(viewer)
                        if not viewer.is_running():
                            break
                        if source is None:
                            print(
                                f"Cube not found after recovery and "
                                f"{args.max_hunt_tries} looks. Ending loop."
                            )
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
                                include_environment=args.environment,
                                preflight_debug=args.preflight_debug,
                                preflight_debug_limit=args.preflight_debug_limit,
                                failed_trajectory_dir=args.save_failed_trajectories,
                                failed_trajectory_limit=args.failed_trajectory_limit,
                            )
                        except EpisodeSamplingError:
                            print("No feasible plan from the current pose.")
                            raise

                        print(f"\n--- Episode {ep.index}"
                              f"{f'/{args.episodes}' if args.episodes else ''} ---")
                        status = execute_episode(
                            episode,
                            follower=follower,
                            viewer=viewer,
                            offsets_path=args.offsets_path,
                            recording=recording,
                            overhead_camera_cap=overhead_cap,
                            speed=args.speed,
                            wrist_camera=args.wrist_camera,
                            wrist_intrinsics=args.wrist_intrinsics,
                            show_wrist_cam=args.show_wrist_cam,
                            show_wrist_mixed=args.show_wrist_mixed,
                            failed_trajectory_dir=args.save_failed_trajectories,
                        )

                        if status == "restart":
                            raise EpisodeAborted

                        ep.complete()
                        is_last = args.episodes != 0 and ep.index >= args.episodes
                        if not is_last:
                            if not recover_cube(viewer):
                                print("Cube recovery failed after retries. Ending loop.")
                                break

                # Normal end (episode budget met, cube lost, or viewer closed): the
                # arm is at the last near-neutral pose — flow it straight to REST.
                if viewer.is_running():
                    print("Loop done. Moving to REST...")
                    move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
                    ended_at_rest = True
    finally:
        if recording.dataset is not None:
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


if __name__ == "__main__":
    main()
