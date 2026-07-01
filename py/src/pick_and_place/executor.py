# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the physical SO-101 follower through a prepared pick-and-carry episode.

This is the home of the hardware execution path, split out of the sim-only
``pick_and_place/sim.py`` viewer. The sim is the plant: it integrates physics and
the trajectory's joint set points stream out to the real arm at ``CONTROL_HZ``.

Feedback is applied at two points, not continuously across the whole episode:

- **Descent (wrist-camera PBVS).** During the descent onto the cube, the wrist
  camera detects the cube each tick; the estimate is low-pass filtered into the
  live source pose, the grasp is re-derived for the locked face/elbow, and
  ``DescentPhase.evaluate`` re-solves IK toward the updated grasp, so the set
  points track the cube as it descends.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.episodes import (
    Episode,
    _preflight,
    _preflight_collision_is_unexpected,
    _save_failed_preflight_trajectory,
    _write_failed_trajectory_note,
    is_unexpected,
    jaw_floor_clearance,
    jaw_geom_ids,
    scan_contacts,
    set_joint,
)
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    GRIPPER_INDEX,
    GRIPPER_READBACK_CLOSED,
    JOINT_NAMES,
    action_to_joints,
    clamp_joints,
    joints_to_action,
    load_follower_joint_offsets,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.trajectory import replan_remaining_candidates
from pick_and_place.kinematics import So101Kinematics
from pick_and_place.recorder import EpisodeRecorder


# Wall-clock rate shared by physical control, motor readback, camera indexing,
# and dataset rows. MuJoCo takes multiple internal steps per control tick.
CONTROL_HZ = 30.0
# Simulation rate used by the hardware runner. This is an integer multiple of
# CONTROL_HZ, so sampling cannot drift onto the next MuJoCo step as it did with
# the stock 500 Hz timestep.
HARDWARE_SIMULATION_HZ = 600.0
# Seconds spent smoothly ramping the real arm onto the trajectory's start pose
# before playback begins, so there is no jump from wherever it was parked.
RAMP_DURATION = 2.0
# Default playback pace for the physical arm: run at the planner's nominal speed.
# Scaling the trajectory clock slows every phase uniformly without touching the
# planner. Override with ``speed``.
REAL_ARM_DEFAULT_SPEED = 1.0
# Logging-only pickup heuristic. A held cube should keep the physical gripper
# encoder noticeably more open than an empty close.
PICKUP_GRIPPER_MARGIN = 5.0


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
            data.ctrl[actuator_id[name]] = current_sim_joints[name] + alpha * (
                target_sim_joints[name] - current_sim_joints[name]
            )
        data.ctrl[actuator_id["gripper"]] = current_sim_gripper + alpha * (
            target_sim_gripper - current_sim_gripper
        )

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
    if "wall_t" in stacked and len(stacked["wall_t"]) > 1:
        wall_dt = np.diff(stacked["wall_t"])
        missed = int(np.count_nonzero(wall_dt > 1.5 / CONTROL_HZ))
        print(
            f"  wall cadence: median {np.median(wall_dt) * 1000:.1f} ms "
            f"({1.0 / np.median(wall_dt):.1f} Hz), "
            f"p95 {np.percentile(wall_dt, 95) * 1000:.1f} ms, "
            f"{missed} missed tick(s)"
        )
    print("  (with zero offsets, a joint's mean err is its sim→real calibration bias)")


@dataclass
class RecordingSession:
    """Holds the ``LeRobotDataset`` written across one collection run.

    The dataset is created lazily on the first recorded episode, once the camera
    frame shapes are known, and reused for every later episode. The runner owns
    it and calls :meth:`finalize` when the run ends. Episodes are added straight
    into the dataset during execution (one frame per control tick), so there are
    no intermediate video/motor files and no separate export step.
    """

    repo_id: str
    root: Path
    task: str
    fps: float
    vcodec: str = "auto"
    streaming_encoding: bool = True
    image_writer_threads: int = 4
    # Frames the streaming encoder may buffer per camera. The default of 30 (one
    # second at 30 Hz) overflows during the descent's visual-servo tick, when
    # AprilTag detection on the control thread briefly starves the encoder and
    # frames get dropped. A deeper buffer rides through that spike.
    encoder_queue_maxsize: int = 300
    dataset: Any = field(default=None, init=False)

    def create_dataset(self, wrist_shape: tuple, overhead_shape: tuple) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        joint_names = list(JOINT_NAMES)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": joint_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": joint_names,
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (wrist_shape[0], wrist_shape[1], 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.overhead": {
                "dtype": "video",
                "shape": (overhead_shape[0], overhead_shape[1], 3),
                "names": ["height", "width", "channels"],
            },
        }
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=int(round(self.fps)),
            features=features,
            root=self.root,
            robot_type="so101",
            use_videos=True,
            image_writer_threads=self.image_writer_threads,
            vcodec=self.vcodec,
            streaming_encoding=self.streaming_encoding,
            encoder_queue_maxsize=self.encoder_queue_maxsize,
            video_backend="pyav",
        )

    def dropped_frame_count(self) -> int:
        """Frames the streaming video encoder dropped in the current episode.

        The encoder silently drops a frame when its queue backs up (it can't keep
        pace with capture), which leaves the video shorter than the recorded rows
        and corrupts the episode. Returns 0 in PNG mode (no such queue) or before
        the dataset exists.
        """
        if self.dataset is None:
            return 0
        encoder = getattr(self.dataset.writer, "_streaming_encoder", None)
        if encoder is None:
            return 0
        return sum(encoder._dropped_frames.values())

    def save_episode(self, episode_metadata: dict[str, Any] | None = None) -> None:
        """Commit the pending LeRobot episode, optionally adding episode metadata.

        LeRobot stores frame features and episode metadata through separate paths:
        arbitrary metadata cannot be added to the per-frame buffer because it is
        validated against the dataset feature schema. The writer does accept
        extra episode metadata internally, so temporarily wrap that call and
        merge our run-specific fields into the episode row.
        """
        if self.dataset is None:
            raise RuntimeError("cannot save episode before the dataset exists")
        if not episode_metadata:
            self.dataset.save_episode()
            return

        meta = self.dataset.writer._meta
        original_save_episode = meta.save_episode

        def save_episode_with_metadata(
            episode_index,
            episode_length,
            episode_tasks,
            episode_stats,
            base_metadata,
        ):
            merged = dict(base_metadata)
            merged.update(episode_metadata)
            return original_save_episode(
                episode_index,
                episode_length,
                episode_tasks,
                episode_stats,
                merged,
            )

        meta.save_episode = save_episode_with_metadata
        try:
            self.dataset.save_episode()
        finally:
            meta.save_episode = original_save_episode

    def finalize(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()


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
    offsets_path: str | None = None,
    recording: RecordingSession | None = None,
    overhead_camera_cap=None,
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
    are required. Each camera runs a reader thread keeping a single-slot
    "latest frame" buffer fresh; once per control tick the loop snapshots both
    buffers and queues one frame for a dataset writer thread, which converts to
    RGB and calls ``add_frame``. Capture is therefore in lockstep with the
    ``CONTROL_HZ`` loop — exactly one dataset frame per camera per tick. With
    the session's streaming encoder, LeRobot encodes frames as they arrive. A
    completed episode is committed with ``save_episode``; a ``restart``/
    interrupted one is discarded with ``clear_episode_buffer``.

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
    failed_trajectory_path = (
        Path(failed_trajectory_dir) if failed_trajectory_dir is not None else None
    )
    if failed_trajectory_path is not None:
        failed_trajectory_path.mkdir(parents=True, exist_ok=True)

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
    simulation_steps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    control_period = 1.0 / CONTROL_HZ
    if not math.isclose(model.opt.timestep * simulation_steps_per_tick, control_period):
        raise ValueError(
            f"MuJoCo timestep {model.opt.timestep:g}s cannot produce {CONTROL_HZ:g} Hz exactly"
        )
    wrist_cam = None
    wrist_tracker = None
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    wrist_camera_matrix = None
    wrist_undistort_map = None

    cam_lock = threading.Lock()
    cam_frame = None
    cam_frame_id = 0
    last_processed_id = -1
    cam_running = False
    cam_thread = None
    wrist_renderer = None

    # Per-camera reader threads keep these single-slot "latest frame" buffers
    # fresh; the control loop snapshots both once per tick and queues one
    # (state, action, wrist_bgr, overhead_bgr) tuple onto record_queue, so the
    # dataset gets exactly one frame per camera per control tick.
    overhead_lock = threading.Lock()
    overhead_frame = None
    overhead_cam_running = False
    overhead_cam_thread = None

    record_queue: queue.Queue = queue.Queue()
    record_writer_thread = None

    def overhead_reader():
        nonlocal overhead_frame
        while overhead_cam_running:
            ok, frame = overhead_camera_cap.read()
            if ok:
                with overhead_lock:
                    overhead_frame = frame

    def record_writer():
        """Drain queued frames into the dataset until the ``None`` sentinel.

        Color conversion and ``add_frame`` run here, off the control loop, so a
        slow frame never delays a tick. Frames are enqueued one per tick in
        order, so the dataset episode keeps that order."""
        import cv2

        while True:
            item = record_queue.get()
            if item is None:
                return
            state, action, wrist_bgr, overhead_bgr = item
            recording.dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "observation.images.wrist": cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB),
                    "observation.images.overhead": cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2RGB),
                    "task": recording.task,
                }
            )

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
            _request_camera_fps(wrist_cam, "wrist")
            wrist_tracker = CubeTracker(smooth=0.95)

            intrinsics_path = wrist_intrinsics
            if intrinsics_path is None:
                from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR

                intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
            else:
                intrinsics_path = Path(intrinsics_path)

            if intrinsics_path.exists():
                from pick_and_place.camera_compare import load_intrinsics

                wrist_camera_matrix, wrist_undistort_map = load_intrinsics(
                    intrinsics_path, 1280, 720, cv2
                )
                rect_fy = float(wrist_camera_matrix[1, 1])
                model.cam_fovy[wrist_cam_id] = float(
                    np.degrees(2.0 * np.arctan((720 / 2.0) / rect_fy))
                )
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
    episode_status = "incomplete"
    episode_metadata: dict[str, Any] | None = None
    pickup_metadata: dict[str, Any] | None = None
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

        # Recording starts only after the unrecorded ramp reaches the start pose.
        # The wrist reader is already running from the open block above; the
        # overhead reader is started here. Recording needs both cameras.
        if recording is not None:
            if wrist_cam is None or overhead_camera_cap is None or not overhead_camera_cap.isOpened():
                raise RuntimeError(
                    "Recording requires both the wrist and overhead cameras to be open"
                )
            _request_camera_fps(overhead_camera_cap, "overhead")
            overhead_cam_running = True
            overhead_cam_thread = threading.Thread(target=overhead_reader, daemon=True)
            overhead_cam_thread.start()

            # Wait for both latest-frame buffers to fill so the first tick has a
            # real frame to log and the dataset features get true frame shapes.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with cam_lock:
                    first_wrist = cam_frame
                with overhead_lock:
                    first_overhead = overhead_frame
                if first_wrist is not None and first_overhead is not None:
                    break
                time.sleep(0.001)
            else:
                raise RuntimeError("Timed out waiting for the episode camera streams to start")

            if recording.dataset is None:
                recording.create_dataset(first_wrist.shape, first_overhead.shape)
            record_writer_thread = threading.Thread(target=record_writer, daemon=True)
            record_writer_thread.start()
        execution_wall_start = time.monotonic()

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
            from pick_and_place.trajectory import DescentPhase, _shortest_delta, grasp_candidates
            from pick_and_place.geometry import CubePose, CUBE_HALF_SIZE
            from scipy.spatial.transform import Rotation
            import dataclasses
            import cv2

            is_descent = isinstance(phase, DescentPhase)
            show_wrist = show_wrist_cam or show_wrist_mixed

            while viewer.is_running():
                phase_t = (data.time - playback_start) * speed

                bgr = None
                if wrist_cam is not None and (show_wrist or is_descent):
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

                        if show_wrist:
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
                            # Overlay the detected cube pose and TCP onto the wrist
                            # frame purely for the optional live windows. This is
                            # skipped when nothing is shown so the descent's
                            # per-tick vision work does not starve the video
                            # encoder and drop recorded frames.
                            if show_wrist:
                                CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
                                pos_mj_cam = cam_rot.T @ (estimate.position - cam_pos)
                                rot_mj_cam = cam_rot.T @ estimate.rotation
                                tvec = CV_TO_MJ @ pos_mj_cam
                                rmat = CV_TO_MJ @ rot_mj_cam
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

                                # Draw TCP dot
                                gripper_id = mujoco.mj_name2id(
                                    model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
                                )
                                if gripper_id >= 0:
                                    from pick_and_place.geometry import JAW_CONTACT_POSITION

                                    gripper_pos = data.xpos[gripper_id]
                                    gripper_mat = data.xmat[gripper_id].reshape(3, 3)
                                    tcp_world = gripper_pos + gripper_mat @ JAW_CONTACT_POSITION
                                    tcp_cam_mj = cam_rot.T @ (tcp_world - cam_pos)
                                    tcp_cam_cv = np.array(
                                        [tcp_cam_mj[0], -tcp_cam_mj[1], -tcp_cam_mj[2]]
                                    )
                                    if tcp_cam_cv[2] > 0.01:
                                        uv = tcp_cam_cv[:2] / tcp_cam_cv[2]
                                        uv_px = wrist_camera_matrix @ np.array([uv[0], uv[1], 1.0])
                                        px = (int(uv_px[0]), int(uv_px[1]))
                                        cv2.circle(bgr, px, 4, (0, 0, 255), -1, cv2.LINE_AA)
                                        cv2.circle(bgr, px, 4, (255, 255, 255), 1, cv2.LINE_AA)

                            roll, pitch, yaw = Rotation.from_matrix(estimate.rotation).as_euler(
                                "xyz"
                            )
                            new_source = CubePose(
                                x=float(estimate.position[0]),
                                y=float(estimate.position[1]),
                                z=CUBE_HALF_SIZE,
                                roll=0.0,
                                pitch=0.0,
                                yaw=float(yaw),
                            )

                            # Smoothly interpolate target to avoid arm jumps
                            alpha = 0.1
                            smoothed_x = dynamic_source.x * (1 - alpha) + new_source.x * alpha
                            smoothed_y = dynamic_source.y * (1 - alpha) + new_source.y * alpha
                            smoothed_yaw = (
                                dynamic_source.yaw
                                + _shortest_delta(dynamic_source.yaw, new_source.yaw) * alpha
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

                            # Update simulated cube to match camera detection
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
                    sim_frame_to_real(frame.joints, frame.gripper, offsets),
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

                # Queue exactly one dataset frame for this tick: the measured
                # joints as state, the commanded joints as action, and a snapshot
                # of each camera's latest frame.
                if record_writer_thread is not None:
                    with cam_lock:
                        wrist_snapshot = cam_frame
                    with overhead_lock:
                        overhead_snapshot = overhead_frame
                    if wrist_snapshot is not None and overhead_snapshot is not None:
                        record_queue.put(
                            (
                                actual.astype(np.float32),
                                commanded.astype(np.float32),
                                wrist_snapshot,
                                overhead_snapshot,
                            )
                        )

                viewer.sync()

                if phase_t >= phase.duration:
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
                    from pick_and_place.geometry import JAW_CONTACT_POSITION

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
                # Treat carry + release/lift as one contact-critical section.
                # At the low predrop pose, motor readback can map clear hardware
                # several millimetres through the sim floor. Release and lift
                # from the locked planned endpoint, then trust readback again.
                if len(current_traj.phases) > 1 and current_traj.phases[1].name == "release":
                    current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                    continue

            if completed_phase_name == "descent" and isinstance(phase, DescentPhase):
                if free_grasp:
                    dynamic_grasp = phase.grasp
                else:
                    from pick_and_place.trajectory import grasp_candidates

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
                from pick_and_place.trajectory import (
                    GRIPPER_OPEN,
                    GraspPhase,
                    LiftPhase,
                    RecoveryLiftPhase,
                )

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
                pickup_detected = gripper_delta >= pickup_gripper_margin
                confidence = (
                    gripper_delta / pickup_gripper_margin
                    if pickup_gripper_margin > 0.0
                    else float("inf")
                )
                pickup_metadata = {
                    "pickup_check_phase": completed_phase_name,
                    "pickup_detected": pickup_detected,
                    "pickup_gripper_position": gripper_position,
                    "pickup_empty_gripper_position": float(pickup_empty_gripper_position),
                    "pickup_gripper_margin": float(pickup_gripper_margin),
                    "pickup_gripper_delta": gripper_delta,
                    "pickup_confidence": confidence,
                }
                status = "held?" if pickup_detected else "empty?"
                print(
                    "Pickup check after "
                    f"{completed_phase_name}: gripper={gripper_position:.1f}, "
                    f"empty={pickup_empty_gripper_position:.1f}, "
                    f"delta={gripper_delta:+.1f} "
                    f"(margin {pickup_gripper_margin:.1f}) -> {status}"
                )

            # Checkpoint Replanning
            if len(current_traj.phases) <= 1:
                episode_status = "success"
                break  # All phases completed

            # Sense: get actual joints
            actual = action_to_joints(follower.get_observation(), commanded)
            measured_joints, measured_gripper = real_frame_to_sim(actual, offsets)
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
                    detail_events = _preflight(
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
                        if _preflight_collision_is_unexpected(event)
                    ]
                    unexpected = [
                        (event.time, event.geom1, event.geom2) for event in unexpected_detail
                    ]
                else:
                    events = _preflight(
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
                        _save_failed_preflight_trajectory(
                            path,
                            model,
                            rejected_traj,
                            actuator_id,
                            rejected_detail,
                        )
                        print(f"saved rejected replan trajectory: {path}")
                _write_failed_trajectory_note(
                    failed_trajectory_path,
                    reason,
                    source=dynamic_source,
                    target=episode.target,
                )
                episode_status = "restart"
                return "restart"

            current_traj = candidate_traj
            # Loop on to execute the replanned remaining phases.

    except KeyboardInterrupt:
        # Let the caller park the arm; clean up cameras on the way out.
        print("\nInterrupted during episode.")
        raise
    finally:
        # Stop the readers first so no new frames are produced, then drain the
        # record queue with a sentinel so every queued tick is added before the
        # episode is committed or discarded.
        if wrist_cam is not None:
            cam_running = False
            if cam_thread is not None:
                cam_thread.join(timeout=1.0)
            wrist_cam.release()
        if overhead_cam_thread is not None:
            overhead_cam_running = False
            overhead_cam_thread.join(timeout=1.0)
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
        if recording is not None and recording.dataset is not None and record_writer_thread is not None:
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
            elif recording.dataset.has_pending_frames():
                recording.dataset.clear_episode_buffer()
                print(f"Discarded {episode_status} episode (not added to dataset).")
        if show_wrist_cam or show_wrist_mixed:
            import cv2

            cv2.destroyAllWindows()
        if wrist_renderer is not None:
            wrist_renderer.close()

    return "success" if episode_status == "success" else "restart"
