#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the analytic (closed-form planner) pick-and-place on the physical SO-101.

Prepares a collision-free episode from the closed-form ``pick_and_carry`` planner
(the same sampler ``view_trajectory`` uses) and runs it on the real arm via
``pick_and_place.executor``. Today this is open-loop feedforward — the sim is the
source of truth: it integrates physics in a live viewer while set points feed the
follower at ``CONTROL_HZ`` and motor readback is logged. The executor is where the
phase state machine and checkpoint replanning will grow (see
``docs/realworld-execution-roadmap.md``). With zero offsets the per-joint tracking
report doubles as a sim→real calibration measurement.

This is the analytic hardware path. For sim-only playback (no arm) use
``view_trajectory``; learned policies live under ``pick_and_place.il`` / ``.rl``.
"""

from __future__ import annotations

import argparse

import numpy as np

from pick_and_place.episodes import prepare_episode
from pick_and_place.executor import REAL_ARM_DEFAULT_SPEED, execute_episode
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose


def _get_tracked_cube(camera_index: str, camera_name: str) -> CubePose | None:
    import cv2
    import math
    import time
    import mujoco
    from scipy.spatial.transform import Rotation
    from pick_and_place.scene import build_scene
    from pick_and_place.camera_extrinsics import apply_camera_extrinsics_to_model, load_local_camera_extrinsics
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.cam_align_solve import parse_index_or_path
    from pick_and_place.cube_detection import estimate_cube_pose, cube_pose_to_world, make_cube_detector
    from pick_and_place.workspace_overlays import PAN_AXIS, WORKSPACE_OVERLAYS

    spec = build_scene(wrist_camera=True, include_environment=True)
    model = spec.compile()
    data = mujoco.MjData(model)
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    mujoco.mj_forward(model, data)
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

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(parse_index_or_path(camera_index), backend)
    if not cap.isOpened():
        print(f"Warning: could not open camera {camera_index!r} to track cube.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    detector = make_cube_detector()
    
    for _ in range(10):
        cap.read()
        
    clearance_overlay = next(o for o in WORKSPACE_OVERLAYS if o.name == "workspace_clearance_pregrasp")
    r_inner = clearance_overlay.inner_radius
    r_outer = clearance_overlay.outer_radius
    
    last_print = 0.0
    
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            print("Warning: failed to read frame from camera.")
            cap.release()
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
            
        estimate = estimate_cube_pose(rgb, detector, camera_matrix)
        if estimate is None:
            if time.time() - last_print > 1.0:
                print("Waiting for cube tags to be visible in camera frame...")
                last_print = time.time()
            continue
            
        rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
        
        dx = position[0] - PAN_AXIS[0]
        dy = position[1] - PAN_AXIS[1]
        r = math.hypot(dx, dy)
        
        if r < r_inner or r > r_outer:
            if time.time() - last_print > 1.0:
                print(f"Cube detected at pos=({position[0]:.3f}, {position[1]:.3f}) but is outside the allowed annulus (r={r:.3f}m, allowed: {r_inner:.3f}m - {r_outer:.3f}m). Please move the cube into the orange clearance overlay.")
                last_print = time.time()
            continue
        
        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
        
        print(f"Tracked cube pose: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        cap.release()
        return CubePose(
            x=float(position[0]), 
            y=float(position[1]), 
            z=CUBE_HALF_SIZE, 
            roll=0.0, 
            pitch=0.0, 
            yaw=float(yaw)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="source cube (x, y) on the floor; omit to use tracked cube",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="target (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--follower-port",
        required=True,
        help="serial port of the SO-101 follower",
    )
    parser.add_argument(
        "--follower-id",
        default="folly",
        help="follower calibration id used by lerobot (default: folly)",
    )
    parser.add_argument(
        "--offsets-path",
        default=None,
        help="JSON of per-joint sim→real degree offsets (default: zero offsets)",
    )
    parser.add_argument(
        "--record-path",
        default=None,
        help="CSV path for the per-tick desired-vs-actual motor log",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="playback speed multiplier of the nominal trajectory pace "
        f"(1.0 = nominal; default {REAL_ARM_DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--environment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the calibration workspace_frame and overhead camera mount in the scene",
    )
    parser.add_argument("--camera", default="0", help="OpenCV camera index or device path (default: 0)")
    parser.add_argument("--camera-name", default="overhead_camera", help="name of the camera in the model (default: overhead_camera)")
    parser.add_argument(
        "--wrist-camera", 
        default="1", 
        help="OpenCV camera index or device path for the wrist camera (default: 1)"
    )
    parser.add_argument(
        "--wrist-intrinsics", 
        default=None, 
        help="Path to wrist camera intrinsics JSON"
    )
    parser.add_argument(
        "--show-wrist-cam",
        action="store_true",
        help="Show live feed from the wrist camera in an OpenCV window"
    )
    args = parser.parse_args()

    if args.source is not None:
        source = CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
    else:
        source = _get_tracked_cube(args.camera, args.camera_name)
        if source is None:
            raise SystemExit("Error: Could not track the cube and no --source was provided.")
    target = (
        CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
        if args.target is not None
        else None
    )

    episode = prepare_episode(
        np.random.default_rng(),
        source,
        target,
        verbose=True,
        include_environment=args.environment,
    )

    from pick_and_place.camera_extrinsics import apply_camera_extrinsics_to_model, load_local_camera_extrinsics
    import mujoco
    apply_camera_extrinsics_to_model(episode.model, load_local_camera_extrinsics())
    mujoco.mj_forward(episode.model, episode.data)

    execute_episode(
        episode,
        follower_port=args.follower_port,
        follower_id=args.follower_id,
        offsets_path=args.offsets_path,
        record_path=args.record_path,
        speed=args.speed,
        wrist_camera=args.wrist_camera,
        wrist_intrinsics=args.wrist_intrinsics,
        show_wrist_cam=args.show_wrist_cam,
    )


if __name__ == "__main__":
    main()
