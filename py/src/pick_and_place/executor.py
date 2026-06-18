# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the physical SO-101 follower through a prepared pick-and-carry episode.

This is the home of the hardware execution path, split out of the sim-only
``pick_and_place/sim.py`` viewer. Today it is **pure feedforward with the sim as the
source of truth**: the sim integrates physics, the trajectory's joint set points
stream out to the real arm at ``CONTROL_HZ``, and motor readback is logged but
never fed back. The phase state machine for checkpoint replanning (sense → plan →
execute → re-seed) will grow here — see ``docs/realworld-execution-roadmap.md``.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.episodes import Episode, _preflight, is_unexpected, scan_contacts
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    GRIPPER_INDEX,
    JOINT_NAMES,
    action_to_joints,
    clamp_joints,
    joints_to_action,
    load_follower_joint_offsets,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.trajectory import replan_remaining_phases
from pick_and_place.kinematics import So101Kinematics
from pick_and_place.recorder import EpisodeRecorder


# Rate at which set points are streamed to the physical follower and the motors
# are read back. The sim steps far faster; follower I/O is throttled to this.
CONTROL_HZ = 60.0
# Seconds spent smoothly ramping the real arm onto the trajectory's start pose
# before playback begins, so there is no jump from wherever it was parked.
RAMP_DURATION = 2.0
# Default playback pace for the physical arm: a fraction of the nominal speed so
# the first hardware passes are gentle. Scaling the trajectory clock slows every
# phase uniformly without touching the planner. Override with ``speed``.
REAL_ARM_DEFAULT_SPEED = 0.5


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def follower_clamp_limits(kinematics: So101Kinematics) -> tuple[np.ndarray, np.ndarray]:
    """Real-frame clamp bounds derived from the model: arm-joint limits in degrees
    (the same limits the trajectory was planned against) plus the gripper's 0-100
    position range. Clamping to these never alters a valid command."""
    low = np.empty(len(JOINT_NAMES))
    high = np.empty(len(JOINT_NAMES))
    for i, name in enumerate(ARM_JOINT_NAMES):
        limit = kinematics.joint_limits[name]
        low[i] = math.degrees(limit.min)
        high[i] = math.degrees(limit.max)
    low[GRIPPER_INDEX] = 0.0
    high[GRIPPER_INDEX] = 100.0
    return low, high


def clamp_and_warn(
    commanded: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    warned: set[str],
) -> np.ndarray:
    """Clamp ``commanded`` to ``[low, high]``, printing once per joint that a
    command actually exceeded the limits (so clipping never goes unnoticed)."""
    clamped = clamp_joints(commanded, low, high)
    for i, name in enumerate(JOINT_NAMES):
        if name not in warned and abs(clamped[i] - commanded[i]) > 1e-3:
            warned.add(name)
            print(
                f"warning: {name} command {commanded[i]:.1f} clipped to "
                f"[{low[i]:.1f}, {high[i]:.1f}]"
            )
    return clamped


def ramp_to_start(
    follower,
    target_real: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
) -> None:
    """Smoothstep the real arm onto the trajectory start pose.

    The sim is held at that same start pose (its ``ctrl`` is already set) and
    stepped/synced each tick, so the viewer stays live and the user can watch the
    real arm converge onto the pose the sim is showing before any playback begins.
    """
    current = action_to_joints(follower.get_observation(), target_real)
    delta = target_real - current
    steps = max(1, round(RAMP_DURATION * CONTROL_HZ))
    period = 1.0 / CONTROL_HZ
    for i in range(1, steps + 1):
        if not viewer.is_running():
            return
        step_start = time.time()
        interp = current + _smoothstep(i / steps) * delta
        follower.send_action(joints_to_action(interp))
        mujoco.mj_step(model, data)
        viewer.sync()
        remaining = period - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)


def ramp_to_resting(
    follower,
    target_real: np.ndarray,
    target_sim_joints: dict[str, float],
    target_sim_gripper: float,
    actuator_id: dict[str, int],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
) -> None:
    """Smoothstep the real arm and the sim onto the resting pose."""
    current_real = action_to_joints(follower.get_observation(), target_real)
    delta_real = target_real - current_real

    current_sim_joints = {name: data.ctrl[actuator_id[name]] for name in target_sim_joints}
    current_sim_gripper = data.ctrl[actuator_id["gripper"]]

    steps = max(1, round(RAMP_DURATION * CONTROL_HZ))
    period = 1.0 / CONTROL_HZ
    for i in range(1, steps + 1):
        if not viewer.is_running():
            return
        step_start = time.time()
        alpha = _smoothstep(i / steps)
        
        # Interpolate real arm
        interp_real = current_real + alpha * delta_real
        follower.send_action(joints_to_action(interp_real))
        
        # Interpolate sim
        for name in target_sim_joints:
            data.ctrl[actuator_id[name]] = current_sim_joints[name] + alpha * (target_sim_joints[name] - current_sim_joints[name])
        data.ctrl[actuator_id["gripper"]] = current_sim_gripper + alpha * (target_sim_gripper - current_sim_gripper)
        
        mujoco.mj_step(model, data)
        viewer.sync()
        remaining = period - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)


def _report_tracking(recorder: EpisodeRecorder) -> None:
    """Print a per-joint desired-vs-actual error summary over the recorded run."""
    if len(recorder) == 0:
        print("No follower samples recorded.")
        return
    stacked = recorder.stacked()
    t, commanded, measured = stacked["t"], stacked["commanded"], stacked["measured"]
    error = measured - commanded
    print("\nPer-joint tracking (actual − commanded):")
    print(f"  {'joint':<14}{'unit':<5}{'max|err|':>10}{'mean|err|':>11}{'mean err':>10}")
    for i, name in enumerate(JOINT_NAMES):
        unit = "pos" if i == GRIPPER_INDEX else "deg"
        col = error[:, i]
        print(
            f"  {name:<14}{unit:<5}{np.max(np.abs(col)):>10.2f}"
            f"{np.mean(np.abs(col)):>11.2f}{np.mean(col):>10.2f}"
        )
    print(f"  ({len(recorder)} samples over {t[-1]:.2f}s)")
    print("  (with zero offsets, a joint's mean err is its sim→real calibration bias)")


def _write_record(path: str, recorder: EpisodeRecorder) -> None:
    """Write the full per-tick commanded/measured log to npz (degrees; gripper position).

    Arrays: ``t`` (N,), ``commanded`` (N, J), ``measured`` (N, J), ``joint_names`` (J,)
    giving the column order of the last axis of ``commanded``/``measured``.
    """
    recorder.save(path, joint_names=np.array(JOINT_NAMES))
    print(f"Wrote {len(recorder)} samples to {path}")


def _write_frame_index(path: str, rows: list[tuple[int, int | None, int | None]]) -> None:
    """Write the decimated frame-index log to npz: no joint data, just a join key.

    One row per ``record_fps`` tick: ``motor_row`` is the row number in the
    full-rate motor npz (``_write_record``) this tick belongs to, plus the exact
    frame index each camera's continuous mp4 had just written at that instant
    (``-1`` if that camera isn't recording). The motor npz already has every
    tick's commanded/actual joints, so this file doesn't repeat them — a later
    exporter joins the two on ``motor_row`` to get (state, action, image)
    triples, pulling each frame straight out of the existing video by index.
    """
    motor_row = np.array([r[0] for r in rows], dtype=np.int64)
    wrist_frame = np.array([-1 if r[1] is None else r[1] for r in rows], dtype=np.int64)
    overhead_frame = np.array([-1 if r[2] is None else r[2] for r in rows], dtype=np.int64)
    np.savez_compressed(path, motor_row=motor_row, wrist_frame=wrist_frame, overhead_frame=overhead_frame)
    print(f"Wrote {len(rows)} synced frame-index samples to {path}")


def execute_episode(
    episode: Episode,
    *,
    follower,
    viewer,
    offsets_path: str | None = None,
    record_path: str | None = None,
    video_dir: Path | str | None = None,
    video_base_name: str | None = None,
    overhead_camera_cap=None,
    record_fps: float = 30.0,
    speed: float | None = None,
    wrist_camera: str | None = None,
    wrist_intrinsics: str | None = None,
    show_wrist_cam: bool = False,
    show_wrist_mixed: bool = False,
) -> str:
    """Run one pass of a prepared episode on an already-connected follower.

    The caller owns the ``follower`` (connected) and the ``viewer`` (a launched
    passive viewer, or a mock exposing ``is_running``/``sync``). This ramps the
    real arm onto the trajectory start pose, then steps the sim (the plant) while
    streaming set points to the arm at ``CONTROL_HZ`` and logging motor readback.

    If ``video_dir`` is given, the wrist camera (already opened for cube tracking,
    when ``wrist_camera`` is set) and ``overhead_camera_cap`` (owned by the
    caller, opened across the whole loop) are each recorded to an mp4 named
    ``{video_base_name}_wrist.mp4`` / ``{video_base_name}_overhead.mp4`` in that
    directory, continuously, at the camera's own frame rate — independent of
    ``CONTROL_HZ``. Alongside those full videos and the full-rate ``record_path``
    motor npz, a ``{video_base_name}_frames.npz`` is written: one row every
    ``record_fps`` (default 30) giving ``motor_row`` (the matching row in the
    motor npz) plus, for each camera, the index of the frame each video had most
    recently written at that instant — never a stale repeat. It carries no joint
    data of its own, only the join key, so nothing is stored twice; a later
    LeRobotDataset exporter joins the two npz files on ``motor_row`` and pulls
    each frame straight out of the existing (full-resolution, undecimated) mp4.

    Returns ``"success"`` when the trajectory ran to completion, or ``"restart"``
    when a checkpoint replan failed and the caller should abort and re-home. A
    ``KeyboardInterrupt`` propagates (after camera cleanup) so the caller can park.
    Does not connect/disconnect the follower or move to REST — the caller does.
    """
    model = episode.model
    data = episode.data
    kinematics = episode.kinematics
    actuator_id = episode.actuator_id
    robot_geom_ids = episode.robot_geom_ids
    env_geom_ids = episode.env_geom_ids
    trajectory = episode.trajectory
    start_joints = episode.start_joints
    start_gripper = episode.start_gripper

    # With zero offsets the real frame is just the sim frame in degrees, so this
    # run also measures each joint's sim→real calibration bias (Phase 2 input).
    offsets = load_follower_joint_offsets(offsets_path)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    # Playback pace: the trajectory clock runs at `speed` × wall time, so a factor
    # below 1.0 slows every phase uniformly. The sim still steps in real time (the
    # viewer shows real-time physics); only the set points evolve slower.
    speed = speed if speed is not None else REAL_ARM_DEFAULT_SPEED
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    print(f"Playback speed: {speed:g}× nominal  (run ≈ {trajectory.duration / speed:.1f}s)")

    # Per-tick log of (trajectory time, commanded real joints, motor readback).
    recorder = EpisodeRecorder()
    control_period = 1.0 / CONTROL_HZ
    last_control_t = -math.inf

    wrist_cam = None
    wrist_tracker = None
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    wrist_camera_matrix = None
    wrist_undistort_map = None

    cam_lock = None
    cam_frame = None
    cam_frame_id = 0
    last_processed_id = -1
    cam_running = False
    cam_thread = None
    wrist_renderer = None

    wrist_video_writer = None
    overhead_video_writer = None
    overhead_cam_running = False
    overhead_cam_thread = None
    overhead_frame_count = 0

    # One (motor_row, wrist_frame_index, overhead_frame_index) row per record_fps
    # tick, populated only when video_dir is set (see docstring).
    synced_log: list[tuple[int, int | None, int | None]] = []
    last_synced_t = -math.inf
    record_period = 1.0 / record_fps

    def _open_video_writer(cap, suffix: str):
        """Build a same-size, same-fps mp4 writer for ``cap`` under ``video_dir``."""
        import cv2

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 30.0
        path = Path(video_dir) / f"{video_base_name or 'episode'}_{suffix}.mp4"
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if video_dir is not None:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        if overhead_camera_cap is not None and overhead_camera_cap.isOpened():
            overhead_video_writer = _open_video_writer(overhead_camera_cap, "overhead")
            overhead_cam_running = True

            def overhead_reader():
                nonlocal overhead_frame_count
                while overhead_cam_running:
                    ok, frame = overhead_camera_cap.read()
                    if ok:
                        overhead_video_writer.write(frame)
                        overhead_frame_count += 1

            overhead_cam_thread = threading.Thread(target=overhead_reader, daemon=True)
            overhead_cam_thread.start()

    if wrist_camera is not None and wrist_cam_id >= 0:
        import cv2
        from pick_and_place.cam_align_solve import parse_index_or_path
        from pick_and_place.cube_detection import CubeTracker
        
        cam_idx = parse_index_or_path(wrist_camera)
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        wrist_cam = cv2.VideoCapture(cam_idx, backend)
        
        if wrist_cam.isOpened():
            wrist_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            wrist_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            wrist_tracker = CubeTracker(smooth=0.95)

            if video_dir is not None:
                wrist_video_writer = _open_video_writer(wrist_cam, "wrist")
            
            intrinsics_path = wrist_intrinsics
            if intrinsics_path is None:
                from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
                intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
            else:
                intrinsics_path = Path(intrinsics_path)
                
            if intrinsics_path.exists():
                from pick_and_place.camera_compare import load_intrinsics
                wrist_camera_matrix, wrist_undistort_map = load_intrinsics(intrinsics_path, 1280, 720, cv2)
                rect_fy = float(wrist_camera_matrix[1, 1])
                model.cam_fovy[wrist_cam_id] = float(np.degrees(2.0 * np.arctan((720 / 2.0) / rect_fy)))
            else:
                focal = (720 / 2.0) / np.tan(np.radians(model.cam_fovy[wrist_cam_id]) / 2.0)
                wrist_camera_matrix = np.array(
                    [[focal, 0, 1280 / 2.0], [0, focal, 720 / 2.0], [0, 0, 1]], dtype=float
                )

            if show_wrist_mixed:
                render_w, render_h = 1280, 720
                max_w = int(model.vis.global_.offwidth)
                max_h = int(model.vis.global_.offheight)
                scale = min(1.0, max_w / render_w, max_h / render_h)
                rw = max(1, int(round(render_w * scale)))
                rh = max(1, int(round(render_h * scale)))
                wrist_renderer = mujoco.Renderer(model, width=rw, height=rh)

            cam_lock = threading.Lock()
            cam_running = True
            def cam_reader():
                nonlocal cam_frame, cam_frame_id
                while cam_running:
                    ok, frame = wrist_cam.read()
                    if ok:
                        if wrist_video_writer is not None:
                            wrist_video_writer.write(frame)
                        with cam_lock:
                            cam_frame = frame
                            cam_frame_id += 1
            cam_thread = threading.Thread(target=cam_reader, daemon=True)
            cam_thread.start()
        else:
            print(f"Warning: could not open wrist camera {wrist_camera!r}")
            wrist_cam = None

    prev_contacts: set[tuple[str, str]] = set()
    try:
        # Ramp the real arm onto the trajectory start pose so the arm and the sim
        # are visibly aligned before any playback motion begins.
        start_real = clamp_and_warn(
            sim_frame_to_real(start_joints, start_gripper, offsets),
            clamp_low,
            clamp_high,
            clip_warned,
        )
        print("Ramping real arm to the trajectory start pose...")
        ramp_to_start(follower, start_real, model, data, viewer)
        # State for tracking progress
        current_traj = trajectory
        completed_phase_name = None
        dynamic_source = episode.source
        dynamic_grasp = current_traj.grasp

        while current_traj is not None and current_traj.phases and viewer.is_running():
            phase = current_traj.phases[0]
            print(f"Executing phase: {phase.name}")

            playback_start = data.time
            
            # Setup PBVS dynamically updating current source
            from pick_and_place.trajectory import DescentPhase, _shortest_delta
            from pick_and_place.geometry import CubePose, CUBE_HALF_SIZE
            from scipy.spatial.transform import Rotation
            import dataclasses
            import cv2
            
            is_descent = isinstance(phase, DescentPhase)
            
            while viewer.is_running():
                step_start = time.time()
                phase_t = (data.time - playback_start) * speed
                
                bgr = None
                if wrist_cam is not None and (show_wrist_cam or show_wrist_mixed or is_descent):
                    with cam_lock:
                        if cam_frame is not None and cam_frame_id != last_processed_id:
                            last_processed_id = cam_frame_id
                            bgr = cam_frame.copy()
                    
                    if bgr is not None and is_descent and wrist_tracker is not None:
                        if wrist_undistort_map is not None:
                            bgr = cv2.remap(bgr, *wrist_undistort_map, cv2.INTER_LINEAR)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            
                        cam_pos = data.cam_xpos[wrist_cam_id]
                        cam_rot = data.cam_xmat[wrist_cam_id].reshape(3, 3)
                        
                        from pick_and_place.cube_detection import detect_cube_faces
                        detections = detect_cube_faces(rgb, wrist_tracker.detector)
                        
                        for det in detections:
                            corners = np.array(det.corners, dtype=np.int32)
                            cv2.polylines(bgr, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(bgr, str(det.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                            
                        estimate = wrist_tracker.update(
                            detections, wrist_camera_matrix, cam_pos, cam_rot, dist=None
                        )
                        
                        if estimate is not None:
                            CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
                            pos_mj_cam = cam_rot.T @ (estimate.position - cam_pos)
                            rot_mj_cam = cam_rot.T @ estimate.rotation
                            tvec = CV_TO_MJ @ pos_mj_cam
                            rmat = CV_TO_MJ @ rot_mj_cam
                            rvec, _ = cv2.Rodrigues(rmat)
                            
                            cv2.drawFrameAxes(bgr, wrist_camera_matrix, np.zeros(5), rvec, tvec, 0.03, 2)
                            
                            s = CUBE_HALF_SIZE
                            pts_3d = np.float32([
                                [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
                                [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
                            ])
                            pts_img, _ = cv2.projectPoints(pts_3d, rvec, tvec, wrist_camera_matrix, np.zeros(5))
                            pts_img = pts_img.reshape(-1, 2).astype(int)
                            edges = [(0,1), (1,2), (2,3), (3,0),
                                     (4,5), (5,6), (6,7), (7,4),
                                     (0,4), (1,5), (2,6), (3,7)]
                            for i, j in edges:
                                cv2.line(bgr, tuple(pts_img[i]), tuple(pts_img[j]), (0, 165, 255), 2, cv2.LINE_AA)
                                
                            # Draw TCP dot
                            gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
                            if gripper_id >= 0:
                                from pick_and_place.geometry import JAW_CONTACT_POSITION
                                gripper_pos = data.xpos[gripper_id]
                                gripper_mat = data.xmat[gripper_id].reshape(3, 3)
                                tcp_world = gripper_pos + gripper_mat @ JAW_CONTACT_POSITION
                                tcp_cam_mj = cam_rot.T @ (tcp_world - cam_pos)
                                tcp_cam_cv = np.array([tcp_cam_mj[0], -tcp_cam_mj[1], -tcp_cam_mj[2]])
                                if tcp_cam_cv[2] > 0.01:
                                    uv = tcp_cam_cv[:2] / tcp_cam_cv[2]
                                    uv_px = wrist_camera_matrix @ np.array([uv[0], uv[1], 1.0])
                                    px = (int(uv_px[0]), int(uv_px[1]))
                                    cv2.circle(bgr, px, 4, (0, 0, 255), -1, cv2.LINE_AA)
                                    cv2.circle(bgr, px, 4, (255, 255, 255), 1, cv2.LINE_AA)
                                
                            roll, pitch, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
                            new_source = CubePose(
                                x=float(estimate.position[0]),
                                y=float(estimate.position[1]),
                                z=CUBE_HALF_SIZE,
                                roll=0.0,
                                pitch=0.0,
                                yaw=float(yaw)
                            )
                            
                            # Smoothly interpolate target to avoid arm jumps
                            alpha = 0.1
                            smoothed_x = dynamic_source.x * (1 - alpha) + new_source.x * alpha
                            smoothed_y = dynamic_source.y * (1 - alpha) + new_source.y * alpha
                            smoothed_yaw = dynamic_source.yaw + _shortest_delta(dynamic_source.yaw, new_source.yaw) * alpha
                            
                            smoothed_source = dataclasses.replace(
                                new_source, x=smoothed_x, y=smoothed_y, yaw=smoothed_yaw
                            )
                            phase = dataclasses.replace(phase, source=smoothed_source)
                            dynamic_source = smoothed_source
                            
                            # Update simulated cube to match camera detection
                            cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
                            if cube_body_id >= 0:
                                jnt_adr = model.body_jntadr[cube_body_id]
                                if jnt_adr >= 0 and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                                    qpos_adr = model.jnt_qposadr[jnt_adr]
                                    qvel_adr = model.jnt_dofadr[jnt_adr]
                                    data.qpos[qpos_adr:qpos_adr+3] = [new_source.x, new_source.y, new_source.z]
                                    half_yaw = new_source.yaw / 2.0
                                    data.qpos[qpos_adr+3:qpos_adr+7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
                                    data.qvel[qvel_adr:qvel_adr+6] = 0.0

                    if bgr is not None and (show_wrist_cam or show_wrist_mixed):
                        if wrist_undistort_map is not None and not is_descent:
                            bgr = cv2.remap(bgr, *wrist_undistort_map, cv2.INTER_LINEAR)
                            
                        if wrist_renderer is not None:
                            wrist_renderer.update_scene(data, camera="wrist_camera")
                            sim_rgb = wrist_renderer.render()
                            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
                            if sim_bgr.shape[:2] != bgr.shape[:2]:
                                sim_bgr = cv2.resize(sim_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                            bgr = cv2.addWeighted(bgr, 0.6, sim_bgr, 0.4, 0.0)

                        cv2.imshow("Wrist Cam", bgr)
                        cv2.waitKey(1)

                frame = phase.evaluate(phase_t)
                for name, value in frame.joints.items():
                    data.ctrl[actuator_id[name]] = value
                data.ctrl[actuator_id["gripper"]] = frame.gripper
                mujoco.mj_step(model, data)
                
                curr_contacts = {
                    (min(n1, n2), max(n1, n2))
                    for n1, n2 in scan_contacts(model, data, robot_geom_ids, env_geom_ids)
                    if is_unexpected(n1, n2)
                }
                for pair in curr_contacts - prev_contacts:
                    print(f"collision phase_t={phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
                prev_contacts = curr_contacts

                if data.time - last_control_t >= control_period:
                    last_control_t = data.time
                    commanded = clamp_and_warn(
                        sim_frame_to_real(frame.joints, frame.gripper, offsets),
                        clamp_low,
                        clamp_high,
                        clip_warned,
                    )
                    follower.send_action(joints_to_action(commanded))
                    actual = action_to_joints(follower.get_observation(), commanded)
                    # We log data.time so the overall timeline is continuous.
                    recorder.log(commanded=commanded, measured=actual, t=data.time)

                    if video_dir is not None and data.time - last_synced_t >= record_period:
                        last_synced_t = data.time
                        wrist_idx = None
                        if cam_lock is not None:
                            with cam_lock:
                                wrist_idx = cam_frame_id - 1 if cam_frame_id > 0 else None
                        overhead_idx = overhead_frame_count - 1 if overhead_frame_count > 0 else None
                        synced_log.append((len(recorder) - 1, wrist_idx, overhead_idx))

                viewer.sync()
                
                if phase_t >= phase.duration:
                    break
                
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
            
            if not viewer.is_running():
                break
                
            completed_phase_name = phase.name
            
            if completed_phase_name == "grasp":
                gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
                cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
                if gripper_id >= 0 and cube_id >= 0:
                    from pick_and_place.geometry import JAW_CONTACT_POSITION
                    g_pos = data.xpos[gripper_id]
                    g_mat = data.xmat[gripper_id].reshape(3, 3)
                    tcp_world = g_pos + g_mat @ JAW_CONTACT_POSITION
                    
                    c_pos = data.xpos[cube_id]
                    c_mat = data.xmat[cube_id].reshape(3, 3)
                    tcp_cube = c_mat.T @ (tcp_world - c_pos)
                    
                    print("\n--- GRASP DIAGNOSTICS ---")
                    print("TCP position in cube local frame:")
                    print(f"  x={tcp_cube[0]*1000:+.1f} mm")
                    print(f"  y={tcp_cube[1]*1000:+.1f} mm")
                    print(f"  z={tcp_cube[2]*1000:+.1f} mm")
                    print("-------------------------\n")
            
            if completed_phase_name == "descent" and isinstance(phase, DescentPhase):
                from pick_and_place.trajectory import grasp_candidates
                for g in grasp_candidates(kinematics, dynamic_source):
                    if g.face == phase.face and g.elbow == phase.elbow:
                        dynamic_grasp = g
                        break
            
            # Checkpoint Replanning
            if len(current_traj.phases) <= 1:
                break # All phases completed
                
            # Sense: get actual joints
            actual = action_to_joints(follower.get_observation(), commanded)
            measured_joints, measured_gripper = real_frame_to_sim(actual, offsets)
            
            print(f"Replanning remaining trajectory after {completed_phase_name}...")
            candidate_traj = replan_remaining_phases(
                kinematics,
                measured_joints,
                measured_gripper,
                completed_phase_name,
                dynamic_source,
                episode.target,
                dynamic_grasp,
                episode.end_joints,
                episode.end_gripper,
            )
            
            if candidate_traj is None:
                print("Error: No feasible plan from current state. Aborting episode.")
                return "restart"

            # Preflight the newly planned remaining trajectory.
            events = _preflight(model, candidate_traj, actuator_id, robot_geom_ids, env_geom_ids)
            unexpected = [(t, n1, n2) for t, n1, n2 in events if is_unexpected(n1, n2)]
            if unexpected:
                print("Error: Replanned segment failed preflight. Aborting episode.")
                for t, n1, n2 in unexpected:
                    print(f"  collision t={t:.3f}s {n1} ↔ {n2}")
                return "restart"

            current_traj = candidate_traj
            # Loop on to execute the replanned remaining phases.

    except KeyboardInterrupt:
        # Let the caller park the arm; clean up cameras on the way out.
        print("\nInterrupted during episode.")
        raise
    finally:
        _report_tracking(recorder)
        if record_path is not None:
            _write_record(record_path, recorder)
        if video_dir is not None:
            frames_path = Path(video_dir) / f"{video_base_name or 'episode'}_frames.npz"
            _write_frame_index(str(frames_path), synced_log)
        if wrist_cam is not None:
            cam_running = False
            if cam_thread is not None:
                cam_thread.join(timeout=1.0)
            wrist_cam.release()
        if overhead_cam_thread is not None:
            overhead_cam_running = False
            overhead_cam_thread.join(timeout=1.0)
        if wrist_video_writer is not None:
            wrist_video_writer.release()
        if overhead_video_writer is not None:
            overhead_video_writer.release()
        if show_wrist_cam or show_wrist_mixed:
            import cv2
            cv2.destroyAllWindows()
        if wrist_renderer is not None:
            wrist_renderer.close()

    return "success"

