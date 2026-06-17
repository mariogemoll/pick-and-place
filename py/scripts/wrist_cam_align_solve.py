#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Solve the wrist camera extrinsics from workspace-frame AprilTags.

This script connects to the real robot arms to accurately track their pose while
solving. It finds the 4 workspace tags in a live camera frame, uses the physical
arm's current joints to place the MuJoCo model in the same pose, and solves the
wrist camera's local pose relative to its mount on the gripper.

If both a leader and follower port are provided, it runs a teleoperation loop so
you can jog the follower into position using the leader, and uses the follower's
accurate joint readbacks for the calibration.

The result is saved to `config/camera_extrinsics/wrist_camera.json`.

Example:
    python py/scripts/wrist_cam_align_solve.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 --camera 1 --show
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import mujoco
import numpy as np

from pick_and_place.camera_extrinsics import LOCAL_CAMERA_EXTRINSICS_DIR, save_camera_extrinsics
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.scene import build_scene
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    action_to_joints,
    joints_to_action,
    load_follower_joint_offsets,
    make_so101_leader,
    make_so101_follower,
    real_frame_to_sim,
)
from pick_and_place.cam_align_solve import (
    parse_index_or_path,
    camera_matrix_from_intrinsics,
    default_camera_matrix,
    open_camera,
    read_camera_frame,
    average_results,
    print_result,
    SolveResult,
    TAG_GEOMS,
    opencv_camera_pose_to_mujoco_parent_pose,
    quat_angle_deg,
    NominalDelta,
)
from pick_and_place.camera_compare import draw_tag_detections, draw_hud, load_intrinsics

def tag_world_corners(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, np.ndarray]:
    # The workspace tag is a 40mm printed graphic on a 60mm physical sticker.
    # pupil_apriltags detects the black border, which for tagStandard41h12 is 5/9 of the graphic edge.
    WORKSPACE_TAG_GRAPHIC_M = 0.040
    TAG_BORDER_FRACTION = 5.0 / 9.0
    half_edge = WORKSPACE_TAG_GRAPHIC_M * TAG_BORDER_FRACTION / 2.0
    
    # local corners in pupil_apriltags order: bottom-left, bottom-right, top-right, top-left
    local_corners = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float
    ) * half_edge
    
    corners: dict[int, np.ndarray] = {}
    mujoco.mj_forward(model, data)
    for tag_id, (geom_name, axis) in TAG_GEOMS.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        center = data.geom_xpos[geom_id].copy()
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if axis is not None:
            axis_index, sign = axis
            center = center + sign * rotation[:, axis_index] * model.geom_size[geom_id][axis_index]
            
        # The tag is on the face.
        tag_corners = []
        for local_pt in local_corners:
            world_pt = center + rotation @ local_pt
            tag_corners.append(world_pt)
            
        corners[tag_id] = np.array(tag_corners)
        
    return corners


def solve_wrist_camera_pose(
    *,
    frame_rgb: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    matrix: np.ndarray,
    dist: np.ndarray,
    detector: Any,
    cv2_module: Any,
    nominal_pos: np.ndarray,
    nominal_quat: np.ndarray,
) -> SolveResult | None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"unknown camera {camera_name!r}")

    gray = cv2_module.cvtColor(frame_rgb, cv2_module.COLOR_RGB2GRAY)
    detections = detector.detect(gray)
    all_corners = tag_world_corners(model, data)
    
    matched_dets = [det for det in detections if det.tag_id in all_corners]
    if not matched_dets:
        return None

    object_points = np.concatenate([all_corners[det.tag_id] for det in matched_dets]).astype(float)
    image_points = np.concatenate([det.corners for det in matched_dets]).astype(float)
    
    flags = cv2_module.SOLVEPNP_IPPE if len(matched_dets) == 1 else cv2_module.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2_module.solvePnP(object_points, image_points, matrix, dist, flags=flags)
    if not ok:
        return None
        
    rotation_camera_world, _ = cv2_module.Rodrigues(rvec)
    parent_id = int(model.cam_bodyid[camera_id])
    parent_rotation = data.xmat[parent_id].reshape(3, 3)
    parent_position = data.xpos[parent_id]
    pos, quat = opencv_camera_pose_to_mujoco_parent_pose(
        rotation_camera_world,
        tvec,
        parent_rotation,
        parent_position,
    )
    
    projected, _ = cv2_module.projectPoints(object_points, rvec, tvec, matrix, dist)
    reprojection_error = float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())
    
    delta = NominalDelta(
        translation_m=float(np.linalg.norm(pos - nominal_pos)),
        rotation_deg=quat_angle_deg(quat, nominal_quat),
    )
    
    res = SolveResult(
        used_tags=tuple(sorted(det.tag_id for det in matched_dets)),
        reprojection_error_px=reprojection_error,
        pos=tuple(float(v) for v in pos),
        quat=tuple(float(v) for v in quat),
        nominal_delta=delta,
    )
        
    model.cam_pos[camera_id] = res.pos
    model.cam_quat[camera_id] = res.quat
    mujoco.mj_forward(model, data)
    
    return res


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


@dataclass
class TeleopState:
    leader_joints: Optional[np.ndarray] = None
    follower_read_joints: Optional[np.ndarray] = None
    lock: threading.Lock = threading.Lock()
    stop_event: threading.Event = threading.Event()


def _teleop_thread_func(state: TeleopState, leader, follower, follower_start_joints, real_joints_init, fps, ramp_duration, loop_start_time):
    dt = 1.0 / fps
    real_joints = real_joints_init
    
    while not state.stop_event.is_set():
        step_start = time.perf_counter()
        elapsed_total = step_start - loop_start_time

        obs = leader.get_action()
        leader_joints = action_to_joints(obs, real_joints)
        
        follower_read_joints = None
        if follower is not None:
            if elapsed_total < ramp_duration:
                alpha = _smoothstep(elapsed_total / ramp_duration)
                follower_target = follower_start_joints + alpha * (leader_joints - follower_start_joints)
            else:
                follower_target = leader_joints
            follower.send_action(joints_to_action(follower_target))

            follower_obs = follower.get_observation()
            follower_read_joints = action_to_joints(follower_obs, follower_target)

        with state.lock:
            state.leader_joints = leader_joints
            state.follower_read_joints = follower_read_joints

        real_joints = leader_joints

        elapsed = time.perf_counter() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True, help="Serial port of the SO-101 leader")
    parser.add_argument("--leader-id", default="liddy", help="Leader ID (default: liddy)")
    parser.add_argument("--follower-port", help="Optional serial port of the SO-101 follower")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id")
    parser.add_argument("--offsets-path", default=None, help="JSON of per-joint sim->real offsets")
    parser.add_argument("--camera", required=True, help="OpenCV camera index or device path")
    parser.add_argument("--camera-name", default="wrist_camera")
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--teleop-fps", type=float, default=50.0, help="Teleop loop rate (Hz)")
    parser.add_argument("--show", action="store_true", help="show a live camera window while solving")
    parser.add_argument("--no-save", action="store_true", help="report the solve without writing JSON")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means wait forever for live camera")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="number of solved live-camera frames to average before reporting/saving (0 = endless until 's' or Ctrl+C, default: 0)",
    )
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples cannot be negative")

    try:
        import cv2
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise SystemExit(
            "camera extrinsic solving requires opencv-python and pupil-apriltags"
        ) from exc

    offsets = load_follower_joint_offsets(args.offsets_path)

    print(f"Connecting to leader on {args.leader_port}...")
    leader = make_so101_leader(args.leader_port, args.leader_id)
    leader.connect(calibrate=True)
    print("Leader connected.")

    follower = None
    follower_start_joints = None
    if args.follower_port is not None:
        print(f"Connecting to follower on {args.follower_port}...")
        follower = make_so101_follower(
            args.follower_port,
            args.follower_id,
            disable_torque_on_disconnect=False,
        )
        follower.connect(calibrate=True)
        follower_obs = follower.get_observation()
        follower_start_joints = action_to_joints(follower_obs, np.zeros(6, dtype=float))
        print("Follower connected.")

    leader_action = leader.get_action()
    real_joints = action_to_joints(leader_action, np.zeros(6, dtype=float))

    teleop_state = TeleopState()
    teleop_state.leader_joints = real_joints
    
    ramp_duration = 4.0
    loop_start_time = time.perf_counter()
    
    teleop_thread = threading.Thread(
        target=_teleop_thread_func,
        args=(teleop_state, leader, follower, follower_start_joints, real_joints, args.teleop_fps, ramp_duration, loop_start_time),
        daemon=True
    )
    teleop_thread.start()

    spec = build_scene(wrist_camera=True, include_environment=True)
    if args.show:
        spec.visual.global_.offwidth = max(args.width, 640)
        spec.visual.global_.offheight = max(args.height, 480)
        
    model = spec.compile()
    data = mujoco.MjData(model)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if camera_id < 0:
        raise SystemExit(f"unknown camera {args.camera_name!r}")
    nominal_pos = model.cam_pos[camera_id].copy()
    nominal_quat = model.cam_quat[camera_id].copy()

    intrinsics = args.intrinsics
    if intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
        intrinsics = candidate if candidate.exists() else None
        
    undistort_map = None
    if intrinsics is not None:
        matrix, dist = camera_matrix_from_intrinsics(intrinsics, args.width, args.height)
        rect_matrix, undistort_map = load_intrinsics(intrinsics, args.width, args.height, cv2)
        rect_fy = float(rect_matrix[1, 1])
        model.cam_fovy[camera_id] = float(
            np.degrees(2.0 * np.arctan((args.height / 2.0) / rect_fy))
        )
        print(f"Intrinsics  : {intrinsics}")
    else:
        matrix, dist = default_camera_matrix(
            args.width,
            args.height,
            float(model.cam_fovy[camera_id]),
        )
        print("Intrinsics  : nominal MuJoCo fovy (calibrated JSON recommended)")

    render_width = args.width
    render_height = args.height
    framebuffer_size = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight))
    scale = min(
        1.0,
        framebuffer_size[0] / render_width,
        framebuffer_size[1] / render_height,
    )
    render_width = max(1, int(round(render_width * scale)))
    render_height = max(1, int(round(render_height * scale)))
    if render_width != args.width or render_height != args.height:
        print(
            f"Clamping preview to {render_width}x{render_height} "
            f"(offscreen framebuffer limit {framebuffer_size[0]}x{framebuffer_size[1]})."
        )

    renderer = mujoco.Renderer(model, width=render_width, height=render_height)

    detector = Detector(families="tagStandard41h12", nthreads=4, refine_edges=True)

    cap = None
    result: SolveResult | None = None
    try:
        cap = open_camera(parse_index_or_path(args.camera), args.width, args.height, args.fps, cv2)
        start = time.monotonic()
        results: list[SolveResult] = []
        
        print("Waiting for wrist camera to see the 4 workspace AprilTags...")
        if args.follower_port:
            print("Use the leader arm to jog the follower into position.")
            
        mode = "blend"
        alpha = 0.5
        if args.show:
            print("Press 's' to save, or 'q'/ESC to cancel.")
            cv2.namedWindow("wrist_cam_align_solve", cv2.WINDOW_NORMAL)
        else:
            print("Press Ctrl+C to save and exit.")

        try:
            while True:
                # 1. Update MuJoCo joints to match real arm
                with teleop_state.lock:
                    l_joints = teleop_state.leader_joints
                    f_joints = teleop_state.follower_read_joints
                
                if l_joints is None:
                    continue
                
                # If we have a follower connected, the wrist camera is on it, so we
                # use its highly-accurate joint readbacks to position the MuJoCo model.
                sim_target_joints = f_joints if f_joints is not None else l_joints
                arm_rad, gripper_rad = real_frame_to_sim(sim_target_joints, offsets)

                for name in ARM_JOINT_NAMES:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0:
                        data.qpos[model.jnt_qposadr[jid]] = arm_rad[name]
                g_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
                if g_jid >= 0:
                    data.qpos[model.jnt_qposadr[g_jid]] = gripper_rad

                # Note: solve_camera_pose internally calls tag_world_points, 
                # which correctly calls mujoco.mj_forward(model, data) 
                # using our newly-updated qpos.

                # 2. Capture frame and solve
                frame_rgb = read_camera_frame(cap, cv2)
                if frame_rgb is None:
                    continue
                if frame_rgb.shape[1] != args.width or frame_rgb.shape[0] != args.height:
                    frame_rgb = cv2.resize(frame_rgb, (args.width, args.height), interpolation=cv2.INTER_AREA)

                # Detect tags on the raw distorted frame. OpenCV's solvePnP handles the distortion mathematically,
                # which avoids sub-pixel interpolation errors introduced by cv2.remap.
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                detections = detector.detect(gray)
                
                res = solve_wrist_camera_pose(
                    frame_rgb=frame_rgb,
                    model=model,
                    data=data,
                    camera_name=args.camera_name,
                    matrix=matrix,
                    dist=dist,
                    detector=detector,
                    cv2_module=cv2,
                    nominal_pos=nominal_pos,
                    nominal_quat=nominal_quat,
                )
                
                if res is not None:
                    results.append(res)
                    if args.samples > 1:
                        print(
                            f"Sample {len(results)}/{args.samples}: "
                            f"{res.reprojection_error_px:.3f} px, "
                            f"{res.nominal_delta.translation_m * 1000.0:.1f} mm, "
                            f"{res.nominal_delta.rotation_deg:.2f} deg",
                            flush=True,
                        )
                
                if args.show:
                    if undistort_map is not None:
                        display_frame = cv2.remap(frame_rgb, *undistort_map, cv2.INTER_LINEAR)
                    else:
                        display_frame = frame_rgb.copy()
                        
                    if display_frame.shape[1] != render_width or display_frame.shape[0] != render_height:
                        display_frame = cv2.resize(display_frame, (render_width, render_height), interpolation=cv2.INTER_AREA)

                    renderer.update_scene(data, camera=args.camera_name)
                    sim = renderer.render()
                    
                    if mode == "edges":
                        edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 60, 160)
                        out = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                        out[edges > 0] = (0, 255, 0)
                    else:
                        blended = cv2.addWeighted(display_frame, alpha, sim, 1.0 - alpha, 0.0)
                        out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                    
                    if detections:
                        draw_tag_detections(out, detections, render_width / args.width, render_height / args.height)
                        
                    out = draw_hud(out, mode=mode, alpha=alpha, intrinsics=intrinsics)
                    cv2.putText(out, f"Samples: {len(results)} (Press 's' to save, 'q' to cancel)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("wrist_cam_align_solve", out)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        result = None
                        break
                    if key == ord("s"):
                        if results:
                            result = average_results(
                                results,
                                nominal_pos=nominal_pos,
                                nominal_quat=nominal_quat,
                            )
                        break
                    if key == ord("m"):
                        mode = "edges" if mode == "blend" else "blend"
                    elif key == ord(","):
                        alpha = float(np.clip(alpha - 0.05, 0.0, 1.0))
                    elif key == ord("."):
                        alpha = float(np.clip(alpha + 0.05, 0.0, 1.0))
                
                if args.samples > 0 and len(results) >= args.samples:
                    result = average_results(
                        results,
                        nominal_pos=nominal_pos,
                        nominal_quat=nominal_quat,
                    )
                    break
                
                if args.max_seconds > 0.0 and time.monotonic() - start > args.max_seconds:
                    if results:
                        result = average_results(
                            results,
                            nominal_pos=nominal_pos,
                            nominal_quat=nominal_quat,
                        )
                    break
        except KeyboardInterrupt:
            if results:
                print("\nStopping and saving...")
                result = average_results(
                    results,
                    nominal_pos=nominal_pos,
                    nominal_quat=nominal_quat,
                )
            else:
                print("\nCancelled.")
                
    finally:
        teleop_state.stop_event.set()
        teleop_thread.join(timeout=1.0)
        renderer.close()
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if follower is not None:
            follower.disconnect()
        leader.disconnect()

    if result is None:
        raise SystemExit("no pose solved; need at least one workspace-frame tag visible")

    model.cam_pos[camera_id] = np.array(result.pos, dtype=float)
    model.cam_quat[camera_id] = np.array(result.quat, dtype=float)

    print_result(result)
    if args.no_save:
        return

    output = args.output or (LOCAL_CAMERA_EXTRINSICS_DIR / f"{args.camera_name}.json")
    meta = {
        "method": "workspace-frame AprilTag PnP w/ live arm joints (pick_and_place.wrist_cam_align_solve)",
        "intrinsics": str(intrinsics) if intrinsics is not None else None,
        "reference_tags": list(result.used_tags),
        "rms_reproj_px": round(result.reprojection_error_px, 3),
        "nominal_delta_mm": round(result.nominal_delta.translation_m * 1000.0, 3),
        "nominal_delta_deg": round(result.nominal_delta.rotation_deg, 3),
    }
    path = save_camera_extrinsics(model, args.camera_name, path=output, meta=meta)
    print(f"Saved       : {path}")

if __name__ == "__main__":
    main()
