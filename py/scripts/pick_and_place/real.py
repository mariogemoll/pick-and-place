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

Every run records to ``records/<timestamp>/`` unconditionally: per episode, the
full wrist/overhead mp4s, the full-rate motor npz, and the decimated
frame-index npz (see ``execute_episode``'s docstring).
"""

from __future__ import annotations

import argparse
import datetime
import math
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
    sample_near_neutral,
)
from pick_and_place.executor import (
    REAL_ARM_DEFAULT_SPEED,
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
) -> CubePose | None:
    """Look for the cube on ``cap`` for up to ``timeout`` seconds.

    Returns the cube pose if it is detected inside the allowed clearance annulus,
    or ``None`` if nothing usable is seen before the timeout (not visible, or
    outside the workspace)."""
    import cv2
    from scipy.spatial.transform import Rotation

    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.cube_detection import (
        cube_pose_to_world,
        estimate_cube_pose,
        make_cube_detector,
    )
    from pick_and_place.workspace_overlays import PAN_AXIS, WORKSPACE_OVERLAYS

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
    clearance = next(o for o in WORKSPACE_OVERLAYS if o.name == "workspace_clearance_pregrasp")
    r_inner, r_outer = clearance.inner_radius, clearance.outer_radius

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
        r = math.hypot(position[0] - PAN_AXIS[0], position[1] - PAN_AXIS[1])
        if r < r_inner or r > r_outer:
            print(
                f"Cube seen at ({position[0]:.3f}, {position[1]:.3f}) but outside the "
                f"clearance annulus (r={r:.3f}m, allowed {r_inner:.3f}-{r_outer:.3f}m)."
            )
            continue

        _, _, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
        print(f"Tracked cube: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        return CubePose(
            x=float(position[0]),
            y=float(position[1]),
            z=CUBE_HALF_SIZE,
            roll=0.0,
            pitch=0.0,
            yaw=float(yaw),
        )

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
    args = parser.parse_args()

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
    model, data = _build_model(dummy_source, include_environment=args.environment)
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
    record_dir_path = Path(__file__).resolve().parents[2] / "records" / timestamp
    record_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Recording. Saving to: {record_dir_path}")

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

    def hunt_for_cube(viewer) -> CubePose | None:
        """Look for the cube from the current pose, re-homing near neutral up to
        ``--max-hunt-tries`` times. Returns the cube pose or ``None`` if not found."""
        if args.source is not None:
            return CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
        for attempt in range(args.max_hunt_tries):
            if not viewer.is_running():
                return None
            if attempt > 0:
                arm, grip = sample_near_neutral(rng)
                print(f"Look {attempt + 1}/{args.max_hunt_tries}: moving to a new near-neutral pose...")
                move_to(arm, grip, viewer)
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"Look {attempt + 1}/{args.max_hunt_tries}: searching from current pose...")
            source = track_cube(overhead_cap, args.camera_name, model, data, CUBE_LOOK_TIMEOUT)
            if source is not None:
                return source
        return None

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

                episode_attempt = 0
                for ep in episode_loop(
                    target=args.episodes,
                    rest_every=args.rest_every,
                    cooldown=lambda: cooldown(viewer),
                    should_continue=viewer.is_running,
                ):
                    source = hunt_for_cube(viewer)
                    if not viewer.is_running():
                        break
                    if source is None:
                        print(f"Cube not found after {args.max_hunt_tries} looks. Ending loop.")
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
                                fixed_target,
                                start_joints=current_joints,
                                start_gripper=current_gripper,
                                model=model,
                                data=data,
                                max_attempts=EPISODE_MAX_ATTEMPTS,
                                verbose=True,
                                include_environment=args.environment,
                            )
                        except EpisodeSamplingError:
                            print("No feasible plan from the current pose.")
                            raise

                        episode_attempt += 1
                        print(f"\n--- Episode {ep.index}"
                              f"{f'/{args.episodes}' if args.episodes else ''} ---")
                        episode_base_name = f"episode_{episode_attempt:03d}"
                        status = execute_episode(
                            episode,
                            follower=follower,
                            viewer=viewer,
                            offsets_path=args.offsets_path,
                            record_path=str(record_dir_path / f"{episode_base_name}.npz"),
                            video_dir=record_dir_path,
                            video_base_name=episode_base_name,
                            overhead_camera_cap=overhead_cap,
                            speed=args.speed,
                            wrist_camera=args.wrist_camera,
                            wrist_intrinsics=args.wrist_intrinsics,
                            show_wrist_cam=args.show_wrist_cam,
                            show_wrist_mixed=args.show_wrist_mixed,
                        )

                        if status == "restart":
                            raise EpisodeAborted

                        ep.complete()

                # Normal end (episode budget met, cube lost, or viewer closed): the
                # arm is at the last near-neutral pose — flow it straight to REST.
                if viewer.is_running():
                    print("Loop done. Moving to REST...")
                    move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
                    ended_at_rest = True
    finally:
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
