# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the physical SO-101 follower through a prepared pick-and-carry episode.

This is the home of the hardware execution path, split out of the sim-only
``view_trajectory`` viewer. Today it is **pure feedforward with the sim as the
source of truth**: the sim integrates physics, the trajectory's joint set points
stream out to the real arm at ``CONTROL_HZ``, and motor readback is logged but
never fed back. The phase state machine for checkpoint replanning (sense → plan →
execute → re-seed) will grow here — see ``docs/realworld-execution-roadmap.md``.
"""

from __future__ import annotations

import csv
import math
import time

import mujoco
import mujoco.viewer
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
    make_so101_follower,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.trajectory import replan_remaining_phases, REST_ARM_JOINTS, REST_GRIPPER
from pick_and_place.kinematics import So101Kinematics


# Rate at which set points are streamed to the physical follower and the motors
# are read back. The sim steps far faster; follower I/O is throttled to this.
CONTROL_HZ = 50.0
# Seconds spent smoothly ramping the real arm onto the trajectory's start pose
# before playback begins, so there is no jump from wherever it was parked.
RAMP_DURATION = 4.0
# Default playback pace for the physical arm: a fraction of the nominal speed so
# the first hardware passes are gentle. Scaling the trajectory clock slows every
# phase uniformly without touching the planner. Override with ``speed``.
REAL_ARM_DEFAULT_SPEED = 0.5


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def _follower_clamp_limits(kinematics: So101Kinematics) -> tuple[np.ndarray, np.ndarray]:
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


def _clamp_and_warn(
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


def _ramp_to_start(
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


def _ramp_to_resting(
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


def _report_tracking(log_rows: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    """Print a per-joint desired-vs-actual error summary over the recorded run."""
    if not log_rows:
        print("No follower samples recorded.")
        return
    commanded = np.array([row[1] for row in log_rows])
    actual = np.array([row[2] for row in log_rows])
    error = actual - commanded
    print("\nPer-joint tracking (actual − commanded):")
    print(f"  {'joint':<14}{'unit':<5}{'max|err|':>10}{'mean|err|':>11}{'mean err':>10}")
    for i, name in enumerate(JOINT_NAMES):
        unit = "pos" if i == GRIPPER_INDEX else "deg"
        col = error[:, i]
        print(
            f"  {name:<14}{unit:<5}{np.max(np.abs(col)):>10.2f}"
            f"{np.mean(np.abs(col)):>11.2f}{np.mean(col):>10.2f}"
        )
    print(f"  ({len(log_rows)} samples over {log_rows[-1][0]:.2f}s)")
    print("  (with zero offsets, a joint's mean err is its sim→real calibration bias)")


def _write_record(path: str, log_rows: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    """Write the full per-tick commanded/actual log to CSV (degrees; gripper position)."""
    header = ["t"] + [f"cmd_{n}" for n in JOINT_NAMES] + [f"act_{n}" for n in JOINT_NAMES]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for t, commanded, actual in log_rows:
            writer.writerow(
                [f"{t:.6f}"]
                + [f"{v:.6f}" for v in commanded]
                + [f"{v:.6f}" for v in actual]
            )
    print(f"Wrote {len(log_rows)} samples to {path}")


def execute_episode(
    episode: Episode,
    *,
    follower_port: str,
    follower_id: str = "folly",
    offsets_path: str | None = None,
    record_path: str | None = None,
    speed: float | None = None,
    wrist_camera: str | None = None,
    wrist_intrinsics: str | None = None,
    show_wrist_cam: bool = False,
) -> None:
    """Stream a prepared episode's trajectory to the physical follower for one pass.

    Opens the viewer at the start pose, ramps the real arm onto it, then steps the
    sim (the plant) while streaming the trajectory's set points to the arm at
    ``CONTROL_HZ`` and logging motor readback. Reports per-joint tracking on exit;
    with zero offsets that doubles as a sim→real calibration measurement.
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
    clamp_low, clamp_high = _follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    follower = make_so101_follower(
        follower_port,
        follower_id,
        disable_torque_on_disconnect=False,
    )
    follower.connect()

    # Playback pace: the trajectory clock runs at `speed` × wall time, so a factor
    # below 1.0 slows every phase uniformly. The sim still steps in real time (the
    # viewer shows real-time physics); only the set points evolve slower.
    speed = speed if speed is not None else REAL_ARM_DEFAULT_SPEED
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    print(f"Playback speed: {speed:g}× nominal  (run ≈ {trajectory.duration / speed:.1f}s)")

    # Per-tick log of (trajectory time, commanded real joints, motor readback).
    log_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
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
            
            intrinsics_path = wrist_intrinsics
            if intrinsics_path is None:
                from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
                intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
            else:
                from pathlib import Path
                intrinsics_path = Path(intrinsics_path)
                
            if intrinsics_path.exists():
                from pick_and_place.camera_compare import load_intrinsics
                wrist_camera_matrix, wrist_undistort_map = load_intrinsics(intrinsics_path, 1280, 720, cv2)
            else:
                focal = (720 / 2.0) / np.tan(np.radians(model.cam_fovy[wrist_cam_id]) / 2.0)
                wrist_camera_matrix = np.array(
                    [[focal, 0, 1280 / 2.0], [0, focal, 720 / 2.0], [0, 0, 1]], dtype=float
                )

            import threading
            cam_lock = threading.Lock()
            cam_running = True
            def cam_reader():
                nonlocal cam_frame, cam_frame_id
                while cam_running:
                    ok, frame = wrist_cam.read()
                    if ok:
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
        import sys
        disable_viewer = show_wrist_cam and sys.platform == "darwin"
        
        class MockViewer:
            def __init__(self):
                self._running = True
            def is_running(self):
                return self._running
            def sync(self):
                pass
            def close(self):
                self._running = False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        viewer_ctx = MockViewer() if disable_viewer else mujoco.viewer.launch_passive(model, data)
        with viewer_ctx as viewer:
            # Open the viewer at the start pose first, then ramp the real arm onto
            # it so both are visibly aligned before any playback motion begins.
            start_real = _clamp_and_warn(
                sim_frame_to_real(start_joints, start_gripper, offsets),
                clamp_low,
                clamp_high,
                clip_warned,
            )
            print("Ramping real arm to the trajectory start pose...")
            _ramp_to_start(follower, start_real, model, data, viewer)
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
                from pick_and_place.trajectory import DescentPhase
                from pick_and_place.geometry import CubePose, CUBE_HALF_SIZE
                from scipy.spatial.transform import Rotation
                import dataclasses
                import cv2
                
                is_descent = isinstance(phase, DescentPhase)
                
                while viewer.is_running():
                    step_start = time.time()
                    phase_t = (data.time - playback_start) * speed
                    
                    bgr = None
                    if wrist_cam is not None and (show_wrist_cam or is_descent):
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
                                    
                                roll, pitch, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
                                new_source = CubePose(
                                    x=float(estimate.position[0]),
                                    y=float(estimate.position[1]),
                                    z=CUBE_HALF_SIZE,
                                    roll=0.0,
                                    pitch=0.0,
                                    yaw=float(yaw)
                                )
                                phase = dataclasses.replace(phase, source=new_source)
                                dynamic_source = new_source

                        if bgr is not None and show_wrist_cam:
                            if wrist_undistort_map is not None and not is_descent:
                                bgr = cv2.remap(bgr, *wrist_undistort_map, cv2.INTER_LINEAR)
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
                        commanded = _clamp_and_warn(
                            sim_frame_to_real(frame.joints, frame.gripper, offsets),
                            clamp_low,
                            clamp_high,
                            clip_warned,
                        )
                        follower.send_action(joints_to_action(commanded))
                        actual = action_to_joints(follower.get_observation(), commanded)
                        # We log data.time so the overall timeline is continuous.
                        log_rows.append((data.time, commanded, actual))

                    viewer.sync()
                    
                    if phase_t >= phase.duration:
                        break
                    
                    remaining = model.opt.timestep - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
                
                if not viewer.is_running():
                    break
                    
                completed_phase_name = phase.name
                
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
                current_traj = replan_remaining_phases(
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
                
                if current_traj is None:
                    print("Error: No feasible plan from current state. Aborting.")
                    break
                    
                # Preflight the newly planned remaining trajectory
                events = _preflight(model, current_traj, actuator_id, robot_geom_ids, env_geom_ids)
                unexpected = [(t, n1, n2) for t, n1, n2 in events if is_unexpected(n1, n2)]
                if unexpected:
                    print("Error: Replanned segment failed preflight. Aborting.")
                    for t, n1, n2 in unexpected:
                        print(f"  collision t={t:.3f}s {n1} ↔ {n2}")
                    break
                    
            if viewer.is_running():
                print("Ramping real arm back to the resting pose...")
                resting_real = _clamp_and_warn(
                    sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER, offsets),
                    clamp_low,
                    clamp_high,
                    clip_warned,
                )
                _ramp_to_resting(
                    follower,
                    resting_real,
                    REST_ARM_JOINTS,
                    REST_GRIPPER,
                    actuator_id,
                    model,
                    data,
                    viewer,
                )
                
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _report_tracking(log_rows)
        if record_path is not None:
            _write_record(record_path, log_rows)
        follower.disconnect()
        if wrist_cam is not None:
            cam_running = False
            if cam_thread is not None:
                cam_thread.join(timeout=1.0)
            wrist_cam.release()
        if show_wrist_cam:
            import cv2
            cv2.destroyAllWindows()
