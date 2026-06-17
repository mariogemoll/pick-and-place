#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Overlay the 3D robot model on a live, undistorted camera feed.

This script combines the real camera image with the MuJoCo simulation,
applying undistortion based on calibrated intrinsics and aligning the
3D view using calibrated extrinsics. It can optionally synchronize with
a physical SO-101 follower to show the model at the real robot's pose.

Example:
    python py/scripts/view_mixed.py --overhead-camera 0 --wrist-camera 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pick_and_place.camera_compare import RealSource, load_intrinsics, draw_hud
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.cam_align_solve import parse_index_or_path
from pick_and_place.cube_detection import (
    CUBE_TAG_IDS,
    CubeTracker,
    detect_tags,
)
from pick_and_place.scene import build_scene
from pick_and_place.follower import (
    JOINT_NAMES,
    action_to_joints,
    make_so101_follower,
)

from pick_and_place.camera_compare import draw_tag_detections

WINDOW_TITLE = "view_mixed  (m mode  , . alpha  q quit)"


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation angle (degrees) between two rotation matrices."""
    cos_angle = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))





def _draw_cube_wireframe(bgr: np.ndarray, data: mujoco.MjData, camera_id: int, camera_matrix: np.ndarray) -> None:
    """Project the pick_cube into the given camera and draw its wireframe."""
    cube_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    if cube_id < 0 or camera_id < 0:
        return

    # Get camera pose in world frame
    cam_pos = data.cam_xpos[camera_id]
    cam_mat = data.cam_xmat[camera_id].reshape(3, 3)

    # Get cube pose in world frame
    cube_pos = data.xpos[cube_id]
    cube_mat = data.xmat[cube_id].reshape(3, 3)

    # Local corners of a 30mm cube
    h = 0.015
    corners_local = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]
    ])
    corners_world = cube_pos + corners_local @ cube_mat.T

    # Transform to camera frame. MuJoCo camera points down -Z, +Y is up, +X is right.
    # We transpose cam_mat because its columns are the world axes.
    corners_cam_mj = (corners_world - cam_pos) @ cam_mat
    
    # Convert to OpenCV camera frame: +Z forward, +Y down, +X right.
    corners_cam_cv = np.copy(corners_cam_mj)
    corners_cam_cv[:, 1] = -corners_cam_mj[:, 1]
    corners_cam_cv[:, 2] = -corners_cam_mj[:, 2]

    # Filter out points behind the camera
    valid = corners_cam_cv[:, 2] > 0.01
    if not valid.any():
        return

    # Project to pixels
    uv = corners_cam_cv[:, :2] / corners_cam_cv[:, 2:3]
    uv_px = (camera_matrix @ np.hstack((uv, np.ones((8, 1)))).T).T[:, :2]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        if valid[i] and valid[j]:
            p1 = tuple(uv_px[i].astype(int))
            p2 = tuple(uv_px[j].astype(int))
            cv2.line(bgr, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--overhead-camera", help="OpenCV camera index or device path for overhead camera")
    source.add_argument("--real-image", type=Path, help="captured real frame")
    parser.add_argument("--wrist-camera", help="OpenCV camera index or device path for wrist camera")
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--wrist-intrinsics", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280, help="overlay/render width")
    parser.add_argument("--height", type=int, default=720, help="overlay/render height")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--follower-port", help="serial port of the SO-101 follower to sync joints")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id")
    parser.add_argument(
        "--track-cube",
        action="store_true",
        help="detect the AprilTag cube each frame and mirror it into the sim (overhead only)",
    )
    parser.add_argument(
        "--cube-smooth",
        type=float,
        default=0.3,
        help="cube pose EMA factor 0..1 (0 = none, higher = steadier but laggier)",
    )
    parser.add_argument(
        "--cube-deadband-mm",
        type=float,
        default=1.0,
        help="hold the cube still until it moves more than this many mm (0 = always update)",
    )
    parser.add_argument(
        "--cube-deadband-deg",
        type=float,
        default=1.5,
        help="hold the cube still until it rotates more than this many degrees",
    )
    parser.add_argument(
        "--cube-history",
        type=int,
        default=8,
        help="frames of orientation history for rejecting single-face flips (0 = off)",
    )
    parser.add_argument(
        "--cube-single-face-weight",
        type=float,
        default=0.25,
        help="authority of a lone (depth-blind) single-face frame on the smoother, "
        "relative to a multi-face frame (1 = equal, lower = gentler nudge)",
    )
    parser.add_argument(
        "--cube-quad-decimate",
        type=float,
        default=1.0,
        help="downsample factor for AprilTag quad detection (>1 = faster/snappier, "
        "slight accuracy cost; corners still refined at full res)",
    )
    args = parser.parse_args()

    if args.overhead_camera is None and args.real_image is None and args.wrist_camera is None:
        parser.error("Must provide at least one camera source (--overhead-camera, --wrist-camera, or --real-image)")

    has_overhead = args.overhead_camera is not None or args.real_image is not None

    if args.track_cube and not has_overhead:
        parser.error("--track-cube requires --overhead-camera or --real-image")

    # 1. Build the scene with robot and environment
    spec = build_scene(wrist_camera=True, include_environment=True)
    
    # Ensure the offscreen framebuffer is large enough for the requested render size
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.height)
    
    model = spec.compile()
    data = mujoco.MjData(model)

    # 2. Apply extrinsics
    applied = apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    if "overhead_camera" not in applied and has_overhead:
        print("Warning: no local extrinsics applied for 'overhead_camera'")
    if "wrist_camera" not in applied and args.wrist_camera is not None:
        print("Warning: no local extrinsics applied for 'wrist_camera'")

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    if camera_id < 0 and has_overhead:
        raise SystemExit("unknown camera 'overhead_camera'")

    # 3. Handle framebuffer limits
    requested_size = (args.width, args.height)
    framebuffer_size = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight))
    scale = min(
        1.0,
        framebuffer_size[0] / args.width,
        framebuffer_size[1] / args.height,
    )
    args.width = max(1, int(round(args.width * scale)))
    args.height = max(1, int(round(args.height * scale)))
    if requested_size != (args.width, args.height):
        print(
            f"Clamping render to {args.width}x{args.height} "
            f"(offscreen framebuffer limit {framebuffer_size[0]}x{framebuffer_size[1]})."
        )

    # 4. Handle intrinsics and undistortion
    intrinsics = args.intrinsics
    if intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / "overhead_camera.json"
        intrinsics = candidate if candidate.exists() else None

    undistort_map = None
    if intrinsics is not None and has_overhead:
        rect_matrix, undistort_map = load_intrinsics(intrinsics, args.width, args.height, cv2)
        rect_fy = float(rect_matrix[1, 1])
        model.cam_fovy[camera_id] = float(
            np.degrees(2.0 * np.arctan((args.height / 2.0) / rect_fy))
        )
    elif has_overhead:
        print("Warning: no calibrated intrinsics supplied for overhead; showing raw frame and nominal pinhole")

    # 4. Initialize renderer and camera source
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    real = None
    if has_overhead:
        real = RealSource(
            image_path=args.real_image,
            camera=parse_index_or_path(args.overhead_camera) if args.overhead_camera is not None else None,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            cv2_module=cv2,
        )

    real_wrist = None
    wrist_undistort_map = None
    wrist_intrinsics_path = None
    w_rect_matrix = None
    w_camera_id = -1
    if args.wrist_camera is not None:
        real_wrist = RealSource(
            image_path=None,
            camera=parse_index_or_path(args.wrist_camera),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            cv2_module=cv2,
        )
        wrist_intrinsics_path = args.wrist_intrinsics
        if wrist_intrinsics_path is None:
            candidate = LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
            wrist_intrinsics_path = candidate if candidate.exists() else None
        
        if wrist_intrinsics_path is not None:
            w_rect_matrix, wrist_undistort_map = load_intrinsics(wrist_intrinsics_path, args.width, args.height, cv2)
            w_rect_fy = float(w_rect_matrix[1, 1])
            w_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
            if w_camera_id >= 0:
                model.cam_fovy[w_camera_id] = float(
                    np.degrees(2.0 * np.arctan((args.height / 2.0) / w_rect_fy))
                )
        else:
            print("Warning: no calibrated intrinsics supplied for wrist; showing raw frame and nominal pinhole")

    # 5. Set up cube tracking if requested. Detection runs on the full camera
    # resolution (not the smaller overlay frame), so the tags stay as large as
    # possible -- corner noise, and the pose jitter it causes, scale with how few
    # pixels the tag spans.
    cube_tracker = None
    cube_body_id = -1
    detection_matrix = None
    detection_map = None
    detection_size = (args.camera_width, args.camera_height)
    committed_position = None
    committed_rotation = None
    if args.track_cube:
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        if cube_body_id < 0:
            raise SystemExit("scene has no 'pick_cube' body to track")
        cube_tracker = CubeTracker(
            smooth=args.cube_smooth,
            history=args.cube_history,
            single_face_weight=args.cube_single_face_weight,
            quad_decimate=args.cube_quad_decimate,
        )
        if intrinsics is not None:
            detection_matrix, detection_map = load_intrinsics(intrinsics, *detection_size, cv2)
        else:
            focal = (args.camera_height / 2.0) / np.tan(np.radians(model.cam_fovy[camera_id]) / 2.0)
            detection_matrix = np.array(
                [[focal, 0, args.camera_width / 2.0], [0, focal, args.camera_height / 2.0], [0, 0, 1]],
                dtype=float,
            )
            print("Warning: tracking the cube on a raw frame; calibrated intrinsics recommended")

    # 6. Connect to follower if requested
    follower = None
    joint_qpos_adr = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]
    if args.follower_port:
        follower = make_so101_follower(args.follower_port, args.follower_id)
        follower.connect()
        print(f"Connected to follower on {args.follower_port}")

    mode = "blend"
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    if has_overhead:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    
    if args.wrist_camera is not None:
        cv2.namedWindow("wrist_camera (mixed)", cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow("wrist_camera (sim)", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Sync joints with real robot
            if follower is not None:
                obs = follower.get_observation()
                # follower returns joints in degrees (and gripper 0-100)
                # action_to_joints converts it to a vector.
                # But MuJoCo expects radians for joints.
                # Actually, SO101Follower with use_degrees=True returns degrees.
                # Let's check follower.py for sim_frame_to_real inversion if needed.
                # For now, let's assume we want to just read the joints.
                
                # In follower.py:
                # ARM_JOINT_NAMES are in radians in sim.
                # real_deg = sim_deg + offset
                # So sim_rad = (real_deg - offset) * (pi/180)
                
                # For simplicity, if offsets are 0: sim_rad = real_deg * (pi/180)
                joints_real = action_to_joints(obs, np.zeros(6))
                for i, name in enumerate(JOINT_NAMES[:-1]): # Arm joints
                    data.qpos[joint_qpos_adr[i]] = np.radians(joints_real[i])
                
                # Gripper is more complex, but let's just use a linear approximation for now
                # or skip it if it's too much.
                # In follower.py, gripper_angle_to_position maps angle_rad -> [2.3, 98.5]
                # We need the inverse: position -> angle_rad
                pos = joints_real[5]
                # GRIPPER_READBACK_CLOSED = 2.3, GRIPPER_READBACK_OPEN = 98.5
                # GRIPPER_RENDER_CLOSED_DEG = -10.0, GRIPPER_RENDER_OPEN_DEG = 120.0
                t = (pos - 2.3) / (98.5 - 2.3)
                angle_deg = -10.0 + t * (120.0 - -10.0)
                data.qpos[joint_qpos_adr[5]] = np.radians(angle_deg)

            mujoco.mj_forward(model, data)

            out = None
            if has_overhead:
                frame = real.read(args.width, args.height)
                if undistort_map is not None:
                    frame = cv2.remap(frame, *undistort_map, cv2.INTER_LINEAR)

            cube_status = None
            tag_detections = []
            if cube_tracker is not None and has_overhead:
                det_frame = real.read(*detection_size)
                if detection_map is not None:
                    det_frame = cv2.remap(det_frame, *detection_map, cv2.INTER_LINEAR)
                tag_detections = detect_tags(det_frame, cube_tracker.detector)
                cube_detections = [d for d in tag_detections if d.tag_id in CUBE_TAG_IDS]
                pose = cube_tracker.update(
                    cube_detections,
                    detection_matrix,
                    data.cam_xpos[camera_id],
                    data.cam_xmat[camera_id].reshape(3, 3),
                )
                if pose is None:
                    cube_status = "cube: no tags"
                else:
                    moved = False
                    if not pose.held:
                        moved = (
                            committed_position is None
                            or float(np.linalg.norm(pose.position - committed_position)) * 1000.0
                            > args.cube_deadband_mm
                            or _rotation_angle_deg(pose.rotation, committed_rotation)
                            > args.cube_deadband_deg
                        )
                        if moved:
                            quat = np.empty(4)
                            mujoco.mju_mat2Quat(quat, pose.rotation.reshape(-1))
                            model.body_pos[cube_body_id] = pose.position
                            model.body_quat[cube_body_id] = quat
                            committed_position, committed_rotation = pose.position, pose.rotation
                            mujoco.mj_forward(model, data)
                    cube_status = (
                        f"cube: {pose.num_faces} face(s) "
                        f"ids={list(pose.face_ids)} reproj {pose.reproj_px:.2f}px "
                        f"flips {pose.flip_rate:.0%}"
                        f"{'  (held)' if pose.held else '' if moved else '  (steady)'}"
                    )

            sim = None
            if has_overhead:
                renderer.update_scene(data, camera="overhead_camera")
                sim = renderer.render()
            
            wrist_bgr = None
            wrist_out = None
            renderer.update_scene(data, camera="wrist_camera")
            wrist_sim = renderer.render()
            
            if real_wrist is not None:
                wrist_frame = real_wrist.read(args.width, args.height)
                if wrist_undistort_map is not None:
                    wrist_frame = cv2.remap(wrist_frame, *wrist_undistort_map, cv2.INTER_LINEAR)
                
                if mode == "edges":
                    edges_w = cv2.Canny(cv2.cvtColor(wrist_sim, cv2.COLOR_RGB2GRAY), 60, 160)
                    wrist_out = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
                    wrist_out[edges_w > 0] = (0, 255, 0)
                else:
                    blended_w = cv2.addWeighted(wrist_frame, alpha, wrist_sim, 1.0 - alpha, 0.0)
                    wrist_out = cv2.cvtColor(blended_w, cv2.COLOR_RGB2BGR)
                
                if w_rect_matrix is not None and w_camera_id >= 0 and cube_status is not None and not cube_status.endswith("no tags"):
                    _draw_cube_wireframe(wrist_out, data, w_camera_id, w_rect_matrix)
                
                wrist_out = draw_hud(wrist_out, mode=mode, alpha=alpha, intrinsics=wrist_intrinsics_path)
            else:
                wrist_bgr = cv2.cvtColor(wrist_sim, cv2.COLOR_RGB2BGR)
                if w_rect_matrix is not None and w_camera_id >= 0 and cube_status is not None and not cube_status.endswith("no tags"):
                    _draw_cube_wireframe(wrist_bgr, data, w_camera_id, w_rect_matrix)

            if has_overhead and sim is not None and frame is not None:
                if mode == "edges":
                    edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 60, 160)
                    out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out[edges > 0] = (0, 255, 0)
                else:
                    blended = cv2.addWeighted(frame, alpha, sim, 1.0 - alpha, 0.0)
                    out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

                if tag_detections:
                    draw_tag_detections(
                        out,
                        tag_detections,
                        args.width / frame.shape[1],
                        args.height / frame.shape[0],
                    )

                out = draw_hud(out, mode=mode, alpha=alpha, intrinsics=intrinsics)
                if cube_status is not None:
                    cv2.putText(out, cube_status, (10, out.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(out, cube_status, (10, out.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW_TITLE, out)

            if wrist_out is not None:
                cv2.imshow("wrist_camera (mixed)", wrist_out)
            elif wrist_bgr is not None:
                cv2.imshow("wrist_camera (sim)", wrist_bgr)
            # 1 ms: just enough to pump the GUI event loop. A larger wait adds a
            # fixed latency floor to every frame, which reads as laggy tracking.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = "edges" if mode == "blend" else "blend"
            elif key == ord(","):
                alpha = float(np.clip(alpha - 0.05, 0.0, 1.0))
            elif key == ord("."):
                alpha = float(np.clip(alpha + 0.05, 0.0, 1.0))
            if args.real_image is not None and key == -1:
                cv2.waitKey(0)
    finally:
        renderer.close()
        if real is not None:
            real.close()
        if real_wrist is not None:
            real_wrist.close()
        if follower is not None:
            follower.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
