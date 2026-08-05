# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the physical SO-101 follower through a prepared pick-and-carry episode.

This is the home of the hardware execution path, split out of the sim-only
``pick_and_place/sim.py`` viewer. The sim is the plant: it integrates physics and
the trajectory's joint set points stream out to the real arm at ``CONTROL_HZ``.

Feedback is applied at two points, not continuously across the whole episode:

- **Descent (wrist-camera PBVS).** During the descent onto the cube, a wrist
  camera worker detects the cube as fast as frames and AprilTag solving allow.
  The control loop consumes the latest published estimate each tick, low-pass
  filters it into the live source pose, re-derives the locked face/elbow grasp,
  and ``DescentPhase.evaluate`` re-solves IK toward the updated grasp.
- **Phase boundaries (checkpoint replanning).** After a completed phase the
  measured joints are sensed and the remaining trajectory is replanned and
  preflighted before continuing (sense → plan → execute → re-seed).

The other phases (hover, carry, release, lift) are feedforward playback. Motor
readback is logged every tick and, at checkpoints, fed back into the replan.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.runtime.episodes import Episode
from pick_and_place.runtime.preflight import (
    preflight,
    preflight_collision_is_unexpected,
    save_failed_preflight_trajectory,
    write_failed_trajectory_note,
)
from pick_and_place.sim.collisions import (
    is_unexpected,
    jaw_floor_clearance,
    jaw_geom_ids,
    scan_contacts,
)
from pick_and_place.sim.model import set_joint
from pick_and_place.spec.robot import CONTROL_HZ, GRIPPER_INDEX, HARDWARE_SIMULATION_HZ, JOINT_NAMES
from pick_and_place.core.joint_frames import (
    GRIPPER_READBACK_CLOSED,
    action_to_joints,
    clamp_and_warn,
    follower_clamp_limits,
    joints_to_action,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.core.geometry import CubePose
from pick_and_place.planning.replan import replan_remaining_candidates
from pick_and_place.spec.robot import REST_ARM_JOINTS, REST_GRIPPER
from pick_and_place.data.recorder import EpisodeRecorder
from pick_and_place.data.recording import RecordingSession
from pick_and_place.runtime.frame_reader import FrameReader, open_capture
from pick_and_place.runtime.ramp import ramp_follower
from pick_and_place.planning.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
    WristServoEstimate,
    WristServoPreview,
)


# Default playback pace for the physical arm: run at the planner's nominal speed.
# Scaling the trajectory clock slows every phase uniformly without touching the
# planner. Override with ``speed``.
REAL_ARM_DEFAULT_SPEED = 1.0
# Logging-only pickup heuristic. A held cube should keep the physical gripper
# encoder noticeably more open than an empty close.
PICKUP_GRIPPER_MARGIN = 5.0


def ramp_to_resting(
    follower,
    target_real: np.ndarray,
    target_sim_joints: dict[str, float],
    target_sim_gripper: float,
    actuator_id: dict[str, int],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    low: np.ndarray,
    high: np.ndarray,
    warned: set[str],
    on_tick: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Ramp the real arm onto a pose while walking the sim onto its match.

    Both are driven off the same eased fraction, so the viewer shows what the arm
    is doing rather than jumping to the end pose; the sim is stepped and synced
    inside the ramp's tick budget.
    """
    current_sim_joints = {name: data.ctrl[actuator_id[name]] for name in target_sim_joints}
    current_sim_gripper = data.ctrl[actuator_id["gripper"]]

    def follow_in_sim(alpha: float, commanded: np.ndarray) -> None:
        for name in target_sim_joints:
            data.ctrl[actuator_id[name]] = current_sim_joints[name] + alpha * (
                target_sim_joints[name] - current_sim_joints[name]
            )
        data.ctrl[actuator_id["gripper"]] = current_sim_gripper + alpha * (
            target_sim_gripper - current_sim_gripper
        )
        mujoco.mj_step(model, data)
        if on_tick is not None:
            on_tick(commanded)
        viewer.sync()

    ramp_follower(
        follower, target_real, low, high, warned, viewer=viewer, on_tick=follow_in_sim
    )


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
    if "wall_t" in stacked and len(stacked["wall_t"]) > 1:
        wall_dt = np.diff(stacked["wall_t"])
        missed = int(np.count_nonzero(wall_dt > 1.5 / CONTROL_HZ))
        print(
            f"  wall cadence: median {np.median(wall_dt) * 1000:.1f} ms "
            f"({1.0 / np.median(wall_dt):.1f} Hz), "
            f"p95 {np.percentile(wall_dt, 95) * 1000:.1f} ms, "
            f"{missed} missed tick(s)"
        )
    print("  (mean err is each joint's sim→real tracking bias)")


@dataclass(frozen=True)
class _TickRecord:
    """One control tick's recording payload, queued for the writer thread."""

    state: np.ndarray
    action: np.ndarray
    wrist_bgr: np.ndarray
    overhead_bgr: np.ndarray
    workspace_bgr: np.ndarray | None
    sim_qpos: np.ndarray
    wall_t: float
    servo_active: bool
    servo_source: np.ndarray | None


def _request_camera_fps(cap, label: str) -> None:
    """Ask the camera to capture at ``CONTROL_HZ`` and report what it grants.

    Pinning the capture rate near the tick rate keeps the latest-frame buffer
    fresh and avoids wasting captures. Sync no longer depends on the camera
    honoring it — the control loop logs exactly one buffered frame per tick
    either way — so a mismatch is a warning, not a failure.
    """
    import cv2

    cap.set(cv2.CAP_PROP_FPS, float(CONTROL_HZ))
    actual = cap.get(cv2.CAP_PROP_FPS)
    if not actual or actual <= 0 or math.isnan(actual):
        print(f"warning: {label} camera did not report an FPS after requesting {CONTROL_HZ:g}")
    elif not math.isclose(actual, CONTROL_HZ, rel_tol=1e-2):
        print(
            f"warning: {label} camera reports {actual:g} fps, not the requested "
            f"{CONTROL_HZ:g}; frames are still logged one per control tick"
        )


def execute_episode(
    episode: Episode,
    *,
    follower,
    viewer,
    recording: RecordingSession | None = None,
    overhead_camera_cap=None,
    workspace_camera_cap=None,
    speed: float | None = None,
    wrist_camera: str | None = None,
    wrist_intrinsics: str | None = None,
    show_wrist_cam: bool = False,
    show_wrist_mixed: bool = False,
    failed_trajectory_dir: Path | str | None = None,
    free_grasp: bool = False,
    pickup_empty_gripper_position: float = GRIPPER_READBACK_CLOSED,
    pickup_gripper_margin: float = PICKUP_GRIPPER_MARGIN,
    success_metadata: Callable[[], dict[str, Any]] | None = None,
    record_rest_to_rest: bool = False,
    joint_offsets_deg: dict[str, float] | None = None,
) -> str:
    """Run one pass of a prepared episode on an already-connected follower.

    The caller owns the ``follower`` (connected) and the ``viewer`` (a launched
    passive viewer, or a mock exposing ``is_running``/``sync``). This ramps the
    real arm onto the trajectory start pose, then steps the sim (the plant) while
    streaming set points to the arm at ``CONTROL_HZ`` and logging motor readback.
    MuJoCo advances in a batch of high-rate physics substeps per control tick.

    If ``recording`` is given, the episode is written straight into its
    ``LeRobotDataset`` and both the wrist camera (opened here when
    ``wrist_camera`` is set) and ``overhead_camera_cap`` (owned by the caller)
    are required. ``workspace_camera_cap`` is an optional third recording
    camera, also owned by the caller. Each camera runs a reader thread keeping a
    single-slot "latest frame" buffer fresh; once per control tick the loop
    snapshots the active buffers and queues one ``_TickRecord`` for a writer
    thread, which converts to RGB and calls the session's ``record_frame``.
    Capture is therefore in lockstep with the ``CONTROL_HZ`` loop — exactly one
    recorded frame per camera per tick. With the session's streaming encoder,
    LeRobot encodes frames as they arrive. A completed episode is committed
    with ``save_episode``; a ``restart``/interrupted one is discarded with
    ``discard_episode``.

    Returns ``"success"`` when the trajectory ran to completion, or ``"restart"``
    when a checkpoint replan failed and the caller should abort and re-home. A
    ``KeyboardInterrupt`` propagates (after camera cleanup) so the caller can park.
    When ``record_rest_to_rest`` is set, the recording includes the motion from
    REST to the episode's start pose and the final return to REST. The caller
    must have already parked the physical arm at REST before calling. Does not
    connect/disconnect the follower or otherwise move it to REST.

    ``joint_offsets_deg`` (from the session calibration) is applied feed-forward
    to every arm command and to the readback the checkpoint replanner starts
    from, so open-loop reaching lands near the planned model pose instead of the
    servos' miscalibrated zero. Left ``None`` the arm runs on its raw servo
    calibration. The image-space descent servo still corrects the residual.
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
    failed_trajectory_path = (
        Path(failed_trajectory_dir) if failed_trajectory_dir is not None else None
    )
    if failed_trajectory_path is not None:
        failed_trajectory_path.mkdir(parents=True, exist_ok=True)

    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    # Playback pace: the trajectory clock runs at `speed` × wall time, so a factor
    # below 1.0 slows every phase uniformly. The sim still steps in real time (the
    # viewer shows real-time physics); only the set points evolve slower.
    speed = speed if speed is not None else REAL_ARM_DEFAULT_SPEED
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if record_rest_to_rest and recording is None:
        raise ValueError("record_rest_to_rest requires recording")
    print(f"Playback speed: {speed:g}× nominal  (run ≈ {trajectory.duration / speed:.1f}s)")

    # Per-tick log of (trajectory time, commanded real joints, motor readback).
    recorder = EpisodeRecorder()
    simulation_steps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    control_period = 1.0 / CONTROL_HZ
    if not math.isclose(model.opt.timestep * simulation_steps_per_tick, control_period):
        raise ValueError(
            f"MuJoCo timestep {model.opt.timestep:g}s cannot produce {CONTROL_HZ:g} Hz exactly"
        )
    wrist_tracker = None
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    wrist_camera_matrix = None
    wrist_undistort_map = None
    wrist_renderer = None

    servo_lock = threading.Lock()
    servo_active = False
    servo_running = False
    servo_thread = None
    servo_camera_pos: np.ndarray | None = None
    servo_camera_rot: np.ndarray | None = None
    servo_estimate: WristServoEstimate | None = None
    servo_preview: WristServoPreview | None = None
    # One reader per camera, each keeping only its newest frame; the control loop
    # snapshots them once per tick and queues one ``_TickRecord`` onto
    # record_queue, so the recording gets exactly one frame per camera per
    # control tick. Their ``on_frame`` hooks feed the session's continuous
    # capture at the cameras' full native rate alongside that.
    wrist_reader: FrameReader | None = None
    overhead_reader: FrameReader | None = None
    workspace_reader: FrameReader | None = None

    record_queue: queue.Queue = queue.Queue()
    record_writer_thread = None

    def record_writer():
        """Drain queued frames into the dataset until the ``None`` sentinel.

        Color conversion and ``add_frame`` run here, off the control loop, so a
        slow frame never delays a tick. Frames are enqueued one per tick in
        order, so the dataset episode keeps that order."""
        import cv2

        while True:
            record = record_queue.get()
            if record is None:
                return
            frame = {
                "observation.state": record.state,
                "action": record.action,
                "observation.images.wrist": cv2.cvtColor(record.wrist_bgr, cv2.COLOR_BGR2RGB),
                "observation.images.overhead": cv2.cvtColor(
                    record.overhead_bgr, cv2.COLOR_BGR2RGB
                ),
                "task": recording.task,
            }
            if record.workspace_bgr is not None:
                frame["observation.images.workspace"] = cv2.cvtColor(
                    record.workspace_bgr, cv2.COLOR_BGR2RGB
                )
            recording.record_frame(
                frame,
                sim_qpos=record.sim_qpos,
                wall_t=record.wall_t,
                servo_active=record.servo_active,
                servo_source=record.servo_source,
            )

    def wrist_servo_worker():
        """Process wrist frames independently of the control/recording tick."""
        nonlocal servo_estimate, servo_preview

        import cv2
        from scipy.spatial.transform import Rotation

        from pick_and_place.perception.cube_detection import detect_cube_faces

        last_frame_id = -1
        while servo_running:
            with servo_lock:
                active = servo_active
                cam_pos = None if servo_camera_pos is None else servo_camera_pos.copy()
                cam_rot = None if servo_camera_rot is None else servo_camera_rot.copy()
            if not active or cam_pos is None or cam_rot is None or wrist_tracker is None:
                time.sleep(0.002)
                continue

            # Only ever run the detector on a frame it has not already seen;
            # this worker and the camera run at unrelated rates.
            frame = None if wrist_reader is None else wrist_reader.latest()
            if frame is None or frame.index == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame.index
            captured_at = frame.captured_at

            # The overlays below draw into ``bgr`` in place, and the published
            # frame is shared with the recorder, so this must not alias it.
            # ``remap`` already yields a fresh array; the unrectified path copies.
            bgr = (
                cv2.remap(frame.bgr, *wrist_undistort_map, cv2.INTER_LINEAR)
                if wrist_undistort_map is not None
                else frame.bgr.copy()
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detections = detect_cube_faces(rgb, wrist_tracker.detector)

            annotate_wrist = show_wrist_cam or show_wrist_mixed
            if annotate_wrist:
                for det in detections:
                    corners = np.array(det.corners, dtype=np.int32)
                    cv2.polylines(bgr, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(
                        bgr,
                        str(det.tag_id),
                        tuple(corners[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            estimate = wrist_tracker.update(
                detections, wrist_camera_matrix, cam_pos, cam_rot, dist=None
            )
            if estimate is not None:
                _, _, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
                source = CubePose(
                    x=float(estimate.position[0]),
                    y=float(estimate.position[1]),
                    z=CUBE_HALF_SIZE,
                    roll=0.0,
                    pitch=0.0,
                    yaw=float(yaw),
                )
                with servo_lock:
                    servo_estimate = WristServoEstimate(last_frame_id, source)

                if annotate_wrist:
                    cv_to_mj = np.diag([1.0, -1.0, -1.0])
                    pos_mj_cam = cam_rot.T @ (estimate.position - cam_pos)
                    rot_mj_cam = cam_rot.T @ estimate.rotation
                    tvec = cv_to_mj @ pos_mj_cam
                    rmat = cv_to_mj @ rot_mj_cam
                    rvec, _ = cv2.Rodrigues(rmat)
                    cv2.drawFrameAxes(
                        bgr, wrist_camera_matrix, np.zeros(5), rvec, tvec, 0.03, 2
                    )

                    s = CUBE_HALF_SIZE
                    pts_3d = np.float32(
                        [
                            [-s, -s, -s],
                            [s, -s, -s],
                            [s, s, -s],
                            [-s, s, -s],
                            [-s, -s, s],
                            [s, -s, s],
                            [s, s, s],
                            [-s, s, s],
                        ]
                    )
                    pts_img, _ = cv2.projectPoints(
                        pts_3d, rvec, tvec, wrist_camera_matrix, np.zeros(5)
                    )
                    pts_img = pts_img.reshape(-1, 2).astype(int)
                    edges = [
                        (0, 1),
                        (1, 2),
                        (2, 3),
                        (3, 0),
                        (4, 5),
                        (5, 6),
                        (6, 7),
                        (7, 4),
                        (0, 4),
                        (1, 5),
                        (2, 6),
                        (3, 7),
                    ]
                    for i, j in edges:
                        cv2.line(
                            bgr,
                            tuple(pts_img[i]),
                            tuple(pts_img[j]),
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if captured_at is not None and recording is not None:
                        recording.record_visual_servo_overlay(
                            captured_at,
                            {
                                "tags": [np.asarray(det.corners).tolist() for det in detections],
                                "cube_edges": [[pts_img[i].tolist(), pts_img[j].tolist()] for i, j in edges],
                            },
                        )

            if estimate is None and captured_at is not None and recording is not None:
                recording.record_visual_servo_overlay(
                    captured_at,
                    {"tags": [np.asarray(det.corners).tolist() for det in detections]},
                )

            if annotate_wrist:
                with servo_lock:
                    servo_preview = WristServoPreview(last_frame_id, bgr)

    if wrist_camera is not None and wrist_cam_id >= 0:
        import cv2
        from pick_and_place.perception.cube_detection import CubeTracker

        try:
            wrist_cap = open_capture(wrist_camera, 1280, 720, "wrist")
        except RuntimeError:
            print(f"Warning: could not open wrist camera {wrist_camera!r}")
        else:
            _request_camera_fps(wrist_cap, "wrist")
            wrist_tracker = CubeTracker(smooth=0.95)

            intrinsics_path = wrist_intrinsics
            if intrinsics_path is None:
                from pick_and_place.core.camera_calibration import LOCAL_CAMERA_INTRINSICS_DIR

                intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
            else:
                intrinsics_path = Path(intrinsics_path)

            if not intrinsics_path.exists():
                raise RuntimeError(f"Missing wrist camera intrinsics at {intrinsics_path}")

            from pick_and_place.calibration.camera_compare import load_intrinsics

            wrist_camera_matrix, wrist_undistort_map = load_intrinsics(
                intrinsics_path, 1280, 720, cv2
            )
            rect_fy = float(wrist_camera_matrix[1, 1])
            model.cam_fovy[wrist_cam_id] = float(
                np.degrees(2.0 * np.arctan((720 / 2.0) / rect_fy))
            )

            if show_wrist_mixed:
                render_w, render_h = 1280, 720
                max_w = int(model.vis.global_.offwidth)
                max_h = int(model.vis.global_.offheight)
                scale = min(1.0, max_w / render_w, max_h / render_h)
                rw = max(1, int(round(render_w * scale)))
                rh = max(1, int(round(render_h * scale)))
                wrist_renderer = mujoco.Renderer(model, width=rw, height=rh)

            wrist_reader = FrameReader(
                wrist_cap,
                "wrist",
                on_frame=(
                    None
                    if recording is None
                    else lambda bgr, t: recording.record_live_frame("wrist", bgr, t)
                ),
            )
            servo_running = True
            servo_thread = threading.Thread(target=wrist_servo_worker, daemon=True)
            servo_thread.start()

    prev_contacts: set[tuple[str, str]] = set()
    episode_status = "incomplete"
    episode_metadata: dict[str, Any] | None = None
    pickup_metadata: dict[str, Any] | None = None
    try:
        execution_wall_start = time.monotonic()

        # The wrist reader is already running from the open block above; the
        # overhead and optional workspace readers start here. Recording always
        # needs the wrist and overhead cameras.
        if recording is not None:
            if (
                wrist_reader is None
                or overhead_camera_cap is None
                or not overhead_camera_cap.isOpened()
            ):
                raise RuntimeError(
                    "Recording requires both the wrist and overhead cameras to be open"
                )
            _request_camera_fps(overhead_camera_cap, "overhead")
            overhead_reader = FrameReader(
                overhead_camera_cap,
                "overhead",
                on_frame=lambda bgr, t: recording.record_live_frame("overhead", bgr, t),
                owns_capture=False,
            )
            if workspace_camera_cap is not None:
                if not workspace_camera_cap.isOpened():
                    raise RuntimeError("Workspace camera is not open")
                _request_camera_fps(workspace_camera_cap, "workspace")
                workspace_reader = FrameReader(
                    workspace_camera_cap,
                    "workspace",
                    on_frame=lambda bgr, t: recording.record_live_frame("workspace", bgr, t),
                    owns_capture=False,
                )

            # Wait for every stream to produce a frame, so the first tick has a
            # real one to log and the dataset features get true frame shapes.
            # The readers run concurrently, so the budget is shared, not per camera.
            deadline = time.monotonic() + 2.0

            def first_frame(reader: FrameReader) -> np.ndarray:
                return reader.wait_for_frame(max(0.0, deadline - time.monotonic())).bgr

            try:
                first_wrist = first_frame(wrist_reader)
                first_overhead = first_frame(overhead_reader)
                first_workspace = (
                    None if workspace_reader is None else first_frame(workspace_reader)
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Timed out waiting for the episode camera streams to start"
                ) from exc

            if not recording.initialized:
                recording.create_dataset(
                    first_wrist.shape,
                    first_overhead.shape,
                    None if first_workspace is None else first_workspace.shape,
                )
            recording.start_live_capture(time.monotonic() - execution_wall_start)
            record_writer_thread = threading.Thread(target=record_writer, daemon=True)
            record_writer_thread.start()

        def record_tick(
            commanded: np.ndarray,
            actual: np.ndarray,
            *,
            servo_active: bool = False,
            servo_source: np.ndarray | None = None,
        ) -> None:
            if record_writer_thread is None or wrist_reader is None or overhead_reader is None:
                return
            wrist_snapshot = wrist_reader.latest()
            overhead_snapshot = overhead_reader.latest()
            workspace_snapshot = None if workspace_reader is None else workspace_reader.latest()
            if (
                wrist_snapshot is None
                or overhead_snapshot is None
                or (workspace_reader is not None and workspace_snapshot is None)
            ):
                return
            # Start optional audio capture here, alongside the first captured
            # tick, instead of in the asynchronous encoder thread so its clock
            # has the same origin as the MP4 frames.
            recording.start_audio_capture()
            record_queue.put(
                _TickRecord(
                    state=actual.astype(np.float32),
                    action=commanded.astype(np.float32),
                    wrist_bgr=wrist_snapshot.bgr,
                    overhead_bgr=overhead_snapshot.bgr,
                    workspace_bgr=None if workspace_snapshot is None else workspace_snapshot.bgr,
                    sim_qpos=data.qpos.copy(),
                    wall_t=time.monotonic() - execution_wall_start,
                    servo_active=servo_active,
                    servo_source=servo_source,
                )
            )

        # In rest-to-rest mode the physical arm is already at REST. Start the
        # cameras there, then include the transition to the planned start pose.
        start_real = clamp_and_warn(
            sim_frame_to_real(start_joints, start_gripper, joint_offsets_deg),
            clamp_low,
            clamp_high,
            clip_warned,
        )
        if record_rest_to_rest:
            rest_real = clamp_and_warn(
                sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER, joint_offsets_deg),
                clamp_low,
                clamp_high,
                clip_warned,
            )
            record_tick(rest_real, action_to_joints(follower.get_observation(), rest_real))
        print("Ramping real arm to the trajectory start pose...")
        ramp_to_resting(
            follower,
            start_real,
            start_joints,
            start_gripper,
            actuator_id,
            model,
            data,
            viewer,
            clamp_low,
            clamp_high,
            clip_warned,
            on_tick=(
                (lambda commanded: record_tick(
                    commanded,
                    action_to_joints(follower.get_observation(), commanded),
                ))
                if record_rest_to_rest
                else None
            ),
        )

        # State for tracking progress
        current_traj = trajectory
        completed_phase_name = None
        dynamic_source = episode.source
        dynamic_grasp = current_traj.grasp

        while current_traj is not None and current_traj.phases and viewer.is_running():
            phase = current_traj.phases[0]
            print(f"Executing phase: {phase.name}")

            playback_start = data.time
            next_tick = time.monotonic()

            # Setup PBVS dynamically updating current source
            from pick_and_place.planning.grasp import fold_cube_yaw, grasp_candidates
            from pick_and_place.planning.motion import shortest_delta
            from pick_and_place.planning.trajectory import DescentPhase
            import dataclasses
            import cv2

            is_descent = isinstance(phase, DescentPhase)
            show_wrist = show_wrist_cam or show_wrist_mixed
            descent_convergence = DescentServoConvergence() if is_descent else None
            descent_retry = DescentServoRetryState() if is_descent else None
            descent_saw_detection = False
            descent_max_duration = (
                max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration
            )
            with servo_lock:
                servo_active = is_descent
                servo_camera_pos = None
                servo_camera_rot = None
                last_servo_frame_id = (
                    servo_estimate.frame_id if servo_estimate is not None else -1
                )
                last_preview_frame_id = servo_preview.frame_id if servo_preview is not None else -1

            while viewer.is_running():
                raw_phase_t = (data.time - playback_start) * speed
                phase_t = raw_phase_t
                if descent_retry is not None:
                    phase_t = descent_retry.command_phase_t(raw_phase_t, phase.duration)

                bgr = None
                if wrist_reader is not None and (show_wrist or is_descent):
                    if is_descent and wrist_tracker is not None:
                        with servo_lock:
                            servo_camera_pos = data.cam_xpos[wrist_cam_id].copy()
                            servo_camera_rot = data.cam_xmat[wrist_cam_id].reshape(3, 3).copy()
                            latest_estimate = servo_estimate
                            latest_preview = servo_preview

                        if (
                            latest_estimate is not None
                            and latest_estimate.frame_id != last_servo_frame_id
                        ):
                            last_servo_frame_id = latest_estimate.frame_id
                            # A cube grasp repeats every 90 deg, so fold the
                            # detected yaw onto the quarter-turn nearest the
                            # current target. Without this the single-tag
                            # planar-pose ambiguity flips the yaw 90/180 deg and
                            # the re-solved grasp spins the wrist right around.
                            folded_yaw = fold_cube_yaw(
                                dynamic_source.yaw, latest_estimate.source.yaw
                            )
                            new_source = dataclasses.replace(
                                latest_estimate.source, yaw=folded_yaw
                            )

                            # Smoothly interpolate target to avoid arm jumps.
                            alpha = 0.1
                            smoothed_x = dynamic_source.x * (1 - alpha) + new_source.x * alpha
                            smoothed_y = dynamic_source.y * (1 - alpha) + new_source.y * alpha
                            smoothed_yaw = (
                                dynamic_source.yaw
                                + shortest_delta(dynamic_source.yaw, new_source.yaw) * alpha
                            )

                            smoothed_source = dataclasses.replace(
                                new_source, x=smoothed_x, y=smoothed_y, yaw=smoothed_yaw
                            )
                            if phase.grasp.face != "free":
                                updated_grasp = next(
                                    (
                                        g
                                        for g in grasp_candidates(kinematics, smoothed_source)
                                        if g.face == phase.grasp.face
                                        and g.elbow == phase.grasp.elbow
                                    ),
                                    None,
                                )
                                if updated_grasp is not None:
                                    phase = dataclasses.replace(phase, grasp=updated_grasp)
                            dynamic_source = smoothed_source
                            descent_saw_detection = True
                            if descent_convergence is not None:
                                descent_convergence.observe(dynamic_source)

                            # Update simulated cube to match camera detection.
                            cube_body_id = mujoco.mj_name2id(
                                model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube"
                            )
                            if cube_body_id >= 0:
                                jnt_adr = model.body_jntadr[cube_body_id]
                                if (
                                    jnt_adr >= 0
                                    and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE
                                ):
                                    qpos_adr = model.jnt_qposadr[jnt_adr]
                                    qvel_adr = model.jnt_dofadr[jnt_adr]
                                    data.qpos[qpos_adr : qpos_adr + 3] = [
                                        new_source.x,
                                        new_source.y,
                                        new_source.z,
                                    ]
                                    half_yaw = new_source.yaw / 2.0
                                    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
                                        math.cos(half_yaw),
                                        0.0,
                                        0.0,
                                        math.sin(half_yaw),
                                    ]
                                    data.qvel[qvel_adr : qvel_adr + 6] = 0.0

                        if (
                            show_wrist
                            and latest_preview is not None
                            and latest_preview.frame_id != last_preview_frame_id
                        ):
                            last_preview_frame_id = latest_preview.frame_id
                            bgr = latest_preview.bgr.copy()
                    elif show_wrist and wrist_reader is not None:
                        wrist_snapshot = wrist_reader.latest()
                        if wrist_snapshot is not None:
                            bgr = wrist_snapshot.bgr.copy()

                    if bgr is not None and show_wrist:
                        if wrist_undistort_map is not None and not is_descent:
                            bgr = cv2.remap(bgr, *wrist_undistort_map, cv2.INTER_LINEAR)

                        if wrist_renderer is not None:
                            wrist_renderer.update_scene(data, camera="wrist_camera")
                            sim_rgb = wrist_renderer.render()
                            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
                            if sim_bgr.shape[:2] != bgr.shape[:2]:
                                sim_bgr = cv2.resize(
                                    sim_bgr,
                                    (bgr.shape[1], bgr.shape[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            bgr = cv2.addWeighted(bgr, 0.6, sim_bgr, 0.4, 0.0)

                        cv2.imshow("Wrist Cam", bgr)
                        cv2.waitKey(1)

                frame = phase.evaluate(phase_t)
                for name, value in frame.joints.items():
                    data.ctrl[actuator_id[name]] = value
                data.ctrl[actuator_id["gripper"]] = frame.gripper
                mujoco.mj_step(model, data, nstep=simulation_steps_per_tick)

                curr_contacts = {
                    (min(n1, n2), max(n1, n2))
                    for n1, n2 in scan_contacts(model, data, robot_geom_ids, env_geom_ids)
                    if is_unexpected(n1, n2)
                }
                for pair in curr_contacts - prev_contacts:
                    print(f"collision phase_t={phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
                prev_contacts = curr_contacts

                commanded = clamp_and_warn(
                    sim_frame_to_real(frame.joints, frame.gripper, joint_offsets_deg),
                    clamp_low,
                    clamp_high,
                    clip_warned,
                )
                follower.send_action(joints_to_action(commanded))
                actual = action_to_joints(follower.get_observation(), commanded)
                recorder.log(
                    commanded=commanded,
                    measured=actual,
                    t=data.time,
                    wall_t=time.monotonic() - execution_wall_start,
                )

                record_tick(
                    commanded,
                    actual,
                    servo_active=is_descent,
                    servo_source=np.array(
                        [
                            dynamic_source.x,
                            dynamic_source.y,
                            dynamic_source.z,
                            dynamic_source.yaw,
                        ]
                    ) if is_descent else None,
                )

                viewer.sync()

                if is_descent:
                    if (
                        wrist_reader is not None
                        and wrist_tracker is not None
                        and not descent_saw_detection
                        and descent_retry is not None
                        and not descent_retry.is_backing_up()
                        and raw_phase_t >= phase.duration
                        and descent_retry.can_retry()
                    ):
                        descent_retry.start_backup(raw_phase_t)
                        print(
                            "warning: descent saw no cube tags; backing up to "
                            "pregrasp and retrying "
                            f"({descent_retry.retries_started}/"
                            f"{descent_retry.max_retries})"
                        )
                    if descent_retry is not None and descent_retry.is_backing_up():
                        if descent_retry.backup_complete(raw_phase_t):
                            descent_retry.finish_backup()
                            descent_convergence = DescentServoConvergence()
                            descent_saw_detection = False
                            playback_start = data.time
                            next_tick = time.monotonic()
                        continue
                    if phase_t >= descent_max_duration:
                        if descent_saw_detection and descent_convergence is not None:
                            print(
                                "warning: descent visual servo hit "
                                f"{descent_max_duration:.1f}s cap before settling "
                                f"({descent_convergence.stable_frames}/"
                                f"{DESCENT_SERVO_STABLE_FRAMES} stable frames)"
                            )
                            episode_status = "restart"
                            return "restart"
                        elif wrist_reader is not None:
                            print(
                                "warning: descent visual servo hit "
                                f"{descent_max_duration:.1f}s cap without a cube detection"
                            )
                            episode_status = "restart"
                            return "restart"
                        break
                    if wrist_reader is None or wrist_tracker is None:
                        if phase_t >= phase.duration:
                            break
                    elif (
                        phase_t >= phase.duration
                        and descent_convergence is not None
                        and descent_convergence.is_stable()
                    ):
                        break
                elif phase_t >= phase.duration:
                    break

                next_tick += control_period
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -control_period:
                    # Do not issue a burst of catch-up commands after a long
                    # vision, I/O, or scheduling stall.
                    next_tick = time.monotonic()

            if not viewer.is_running():
                break

            completed_phase_name = phase.name

            # Treat approach + descent as one section. The descent is a visual
            # servo that re-solves IK toward the cube every tick, so replanning
            # the whole remaining trajectory from the measured hover here (before
            # the servo has even run) is wasted work: the descent corrects for
            # any hover-pose error on its own. Advance straight into it.
            if completed_phase_name == "approach":
                if len(current_traj.phases) > 1 and current_traj.phases[1].name == "descent":
                    current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                    continue

            if completed_phase_name == "grasp":
                gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
                cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
                if gripper_id >= 0 and cube_id >= 0:
                    from pick_and_place.core.geometry import JAW_CONTACT_POSITION

                    g_pos = data.xpos[gripper_id]
                    g_mat = data.xmat[gripper_id].reshape(3, 3)
                    tcp_world = g_pos + g_mat @ JAW_CONTACT_POSITION

                    c_pos = data.xpos[cube_id]
                    c_mat = data.xmat[cube_id].reshape(3, 3)
                    tcp_cube = c_mat.T @ (tcp_world - c_pos)

                    print("\n--- GRASP DIAGNOSTICS ---")
                    print("TCP position in cube local frame:")
                    print(f"  x={tcp_cube[0] * 1000:+.1f} mm")
                    print(f"  y={tcp_cube[1] * 1000:+.1f} mm")
                    print(f"  z={tcp_cube[2] * 1000:+.1f} mm")
                    print("-------------------------\n")

                # Treat grasp + lift as one contact-critical section. Right
                # after closing, measured readback often maps the jaws/cube
                # slightly through the sim floor, so a checkpoint here adds a
                # pause without giving useful state. Lift immediately from the
                # locked grasp pose, then measure/replan once safely clear.
                if len(current_traj.phases) > 1 and current_traj.phases[1].name in (
                    "lift",
                    "recovery_lift",
                ):
                    current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                    continue

            if completed_phase_name == "carry":
                # The cruise waypoint is a safe, elevated, non-contact point --
                # nothing risky has happened yet, so a checkpoint here is wasted
                # work (and one more opportunity for ordinary sensor noise to
                # abort an otherwise-fine episode). Advance straight into the
                # drop descent from the locked plan.
                if len(current_traj.phases) > 1 and current_traj.phases[1].name == "drop_descent":
                    current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                    continue

            if completed_phase_name == "drop_descent":
                # Treat the drop descent + release/lift as one contact-critical
                # section. At the low drop pose, motor readback can map clear
                # hardware several millimetres through the sim floor. Release and
                # lift from the locked planned endpoint, then trust readback again.
                if len(current_traj.phases) > 1 and current_traj.phases[1].name == "release":
                    current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                    continue

            if completed_phase_name == "descent" and isinstance(phase, DescentPhase):
                if free_grasp:
                    dynamic_grasp = phase.grasp
                else:
                    from pick_and_place.planning.grasp import grasp_candidates

                    for g in grasp_candidates(kinematics, dynamic_source):
                        if g.face == phase.face and g.elbow == phase.elbow:
                            dynamic_grasp = g
                            break

                # The descent ended at the servo-corrected grasp pose. Rebuild
                # just the grasp and lift from that pose and run them from the
                # locked command (grasp + lift is the contact-critical section,
                # like grasp + lift already is below). The carry onward is
                # replanned from measured state after the lift, so there is no
                # need to enumerate it here.
                from pick_and_place.planning.trajectory import (
                    GraspPhase,
                    LiftPhase,
                    RecoveryLiftPhase,
                )
                from pick_and_place.spec.robot import GRIPPER_OPEN

                lift_cls = RecoveryLiftPhase if free_grasp else LiftPhase
                grasp_phase = GraspPhase(dynamic_grasp.grasp_joints, start_gripper=GRIPPER_OPEN)
                lift_phase = lift_cls(
                    kinematics, dynamic_grasp.grasp_joints, dynamic_grasp.lift_joints
                )
                current_traj = dataclasses.replace(
                    current_traj,
                    phases=(grasp_phase, lift_phase, *current_traj.phases[3:]),
                    grasp=dynamic_grasp,
                )
                continue

            if completed_phase_name in ("lift", "recovery_lift") and pickup_metadata is None:
                actual = action_to_joints(follower.get_observation(), commanded)
                gripper_position = float(actual[GRIPPER_INDEX])
                gripper_delta = gripper_position - pickup_empty_gripper_position
                confidence = (
                    gripper_delta / pickup_gripper_margin
                    if pickup_gripper_margin > 0.0
                    else float("inf")
                )
                pickup_metadata = {
                    "pickup_check_phase": completed_phase_name,
                    "pickup_gripper_position": gripper_position,
                    "pickup_empty_gripper_position": float(pickup_empty_gripper_position),
                    "pickup_gripper_margin": float(pickup_gripper_margin),
                    "pickup_gripper_delta": gripper_delta,
                    "pickup_confidence": confidence,
                }
                print(
                    "Pickup check after "
                    f"{completed_phase_name}: gripper={gripper_position:.1f}, "
                    f"empty={pickup_empty_gripper_position:.1f}, "
                    f"delta={gripper_delta:+.1f} "
                    f"(margin {pickup_gripper_margin:.1f})"
                )

            # Checkpoint Replanning
            if len(current_traj.phases) <= 1:
                episode_status = "success"
                break  # All phases completed

            # Sense: get actual joints
            actual = action_to_joints(follower.get_observation(), commanded)
            measured_joints, measured_gripper = real_frame_to_sim(actual, joint_offsets_deg)
            measured_shadow = mujoco.MjData(model)
            for name, value in measured_joints.items():
                set_joint(model, measured_shadow, name, value)
            set_joint(model, measured_shadow, "gripper", measured_gripper)
            mujoco.mj_forward(model, measured_shadow)
            clearance = jaw_floor_clearance(model, measured_shadow, jaw_geom_ids(model))
            if clearance < 0.005:
                print(f"Measured sim jaw clearance before replan: {clearance * 1000:+.1f} mm")

            print(f"Replanning remaining trajectory after {completed_phase_name}...")
            candidate_traj = None
            rejected_traj = None
            rejected_detail = None
            rejected_unexpected: list[tuple[float, str, str]] = []
            for replan_index, replan_traj in enumerate(
                replan_remaining_candidates(
                    kinematics,
                    measured_joints,
                    measured_gripper,
                    completed_phase_name,
                    dynamic_source,
                    episode.target,
                    dynamic_grasp,
                    episode.end_joints,
                    episode.end_gripper,
                    free_grasp=free_grasp,
                ),
                start=1,
            ):
                if failed_trajectory_path is not None:
                    detail_events = preflight(
                        model,
                        replan_traj,
                        actuator_id,
                        robot_geom_ids,
                        env_geom_ids,
                        detailed=True,
                    )
                    unexpected_detail = [
                        event
                        for event in detail_events
                        if preflight_collision_is_unexpected(event)
                    ]
                    unexpected = [
                        (event.time, event.geom1, event.geom2) for event in unexpected_detail
                    ]
                else:
                    events = preflight(
                        model, replan_traj, actuator_id, robot_geom_ids, env_geom_ids
                    )
                    unexpected_detail = None
                    unexpected = [(t, n1, n2) for t, n1, n2 in events if is_unexpected(n1, n2)]
                if not unexpected:
                    candidate_traj = replan_traj
                    if replan_index > 1:
                        print(
                            f"Selected replan candidate {replan_index} after preflight rejections."
                        )
                    break
                rejected_traj = replan_traj
                rejected_detail = unexpected_detail
                rejected_unexpected = unexpected
                if replan_traj.carry is not None:
                    print(
                        f"  rejected replan candidate {replan_index}: "
                        f"carry={replan_traj.carry.mode} collision t={unexpected[0][0]:.3f}s "
                        f"{unexpected[0][1]} ↔ {unexpected[0][2]}"
                    )

            if candidate_traj is None:
                if rejected_traj is None:
                    print("Error: No feasible plan from current state. Aborting episode.")
                    reason = f"no feasible replan after {completed_phase_name}"
                else:
                    print("Error: All replan candidates failed preflight. Aborting episode.")
                    for t, n1, n2 in rejected_unexpected:
                        print(f"  collision t={t:.3f}s {n1} ↔ {n2}")
                    reason = f"all replan candidates failed preflight after {completed_phase_name}"
                    if failed_trajectory_path is not None and rejected_detail is not None:
                        path = failed_trajectory_path / (
                            f"replan_after_{completed_phase_name or 'start'}_failed.npz"
                        )
                        save_failed_preflight_trajectory(
                            path,
                            model,
                            rejected_traj,
                            actuator_id,
                            rejected_detail,
                        )
                        print(f"saved rejected replan trajectory: {path}")
                write_failed_trajectory_note(
                    failed_trajectory_path,
                    reason,
                    source=dynamic_source,
                    target=episode.target,
                )
                episode_status = "restart"
                return "restart"

            current_traj = candidate_traj
            # Loop on to execute the replanned remaining phases.

        if episode_status == "success" and record_rest_to_rest and viewer.is_running():
            print("Returning real arm to REST...")
            rest_real = clamp_and_warn(
                sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER, joint_offsets_deg),
                clamp_low,
                clamp_high,
                clip_warned,
            )
            ramp_to_resting(
                follower,
                rest_real,
                REST_ARM_JOINTS,
                REST_GRIPPER,
                actuator_id,
                model,
                data,
                viewer,
                clamp_low,
                clamp_high,
                clip_warned,
                on_tick=lambda commanded: record_tick(
                    commanded,
                    action_to_joints(follower.get_observation(), commanded),
                ),
            )

    except KeyboardInterrupt:
        # Let the caller park the arm; clean up cameras on the way out.
        print("\nInterrupted during episode.")
        raise
    finally:
        # Stop optional audio before draining the asynchronous video writer.  The
        # muxer pads/trims it to the frame-derived MP4 duration.
        if recording is not None:
            recording.stop_audio_capture()
        # Stop the readers first so no new frames are produced, then drain the
        # record queue with a sentinel so every queued tick is added before the
        # episode is committed or discarded.
        with servo_lock:
            servo_active = False
        if servo_thread is not None:
            servo_running = False
            servo_thread.join(timeout=1.0)
        for reader in (wrist_reader, overhead_reader, workspace_reader):
            if reader is not None:
                reader.close()
        if recording is not None:
            recording.stop_live_capture()
        if record_writer_thread is not None:
            record_queue.put(None)
            record_writer_thread.join(timeout=30.0)
        if episode_status == "success":
            episode_metadata = {}
            if pickup_metadata is not None:
                episode_metadata.update(pickup_metadata)
            if success_metadata is not None:
                episode_metadata.update(success_metadata())
            if not episode_metadata:
                episode_metadata = None
        _report_tracking(recorder)
        if recording is not None and recording.initialized and record_writer_thread is not None:
            # The writer thread has drained, so the drop count now covers the whole
            # episode. A dropped frame would desync the video from the recorded rows,
            # so fail before committing rather than save a corrupt episode.
            dropped = recording.dropped_frame_count()
            if dropped:
                raise RuntimeError(
                    f"Streaming video encoder dropped {dropped} frame(s): the encoder "
                    "cannot keep pace with capture, which would desync the video from the "
                    "recorded frames. Use a hardware vcodec (auto) or raise the encoder "
                    "queue size."
                )
            if episode_status == "success":
                recording.save_episode(episode_metadata)
                print(f"Saved episode to dataset ({len(recorder)} frames).")
            elif recording.has_pending_frames():
                recording.discard_episode()
                print(f"Discarded {episode_status} episode (not added to dataset).")
        if show_wrist_cam or show_wrist_mixed:
            import cv2

            cv2.destroyAllWindows()
        if wrist_renderer is not None:
            wrist_renderer.close()

    return "success" if episode_status == "success" else "restart"
