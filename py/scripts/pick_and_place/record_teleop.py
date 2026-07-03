#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record teleoperated pick-and-place episodes on the physical SO-101.

The human-driven counterpart to ``real.py``. Instead of planning and executing
an analytic trajectory, this mirrors a physical SO-101 leader onto the follower
and records the resulting motion straight into a LeRobotDataset -- the same
dataset schema ``real.py`` writes (one frame per control tick: measured joints
as state, commanded leader joints as action, plus the wrist and overhead camera
frames).

Refuses to start unless the full rig is present: both the overhead and wrist
cameras open with calibrated intrinsics, and the overhead extrinsics solve from
the workspace-frame AprilTags (skippable with ``--no-recalibrate``).

The follower is *only ever driven by the leader*. After a one-time startup ramp
brings the follower onto the leader's pose, a background thread streams the
leader onto the follower continuously for the whole session; the script never
computes follower set points itself. Whenever it needs the arm somewhere -- out
of the camera's view, at a resting pose -- it asks the operator to move it there
with the leader.

Every prompt is a single keypress: SPACE to proceed, ESC to back out. There is
no Enter anywhere.

Per episode the flow is operator-gated:

1. The operator sets up the scene, positions the arm at a start pose, and presses
   SPACE when ready.
2. The cube and the drop-zone square are detected from the overhead camera. Both
   must be visible in the allowed zones; if either is missing the operator is
   advised and prompted to fix the scene and retry. The initial cube pose and
   target position are recorded. Recording then starts immediately -- no further
   keypress -- and a short chirp (no speech, which always lags) sounds at the
   instant capture actually begins (after the cameras warm up and, on the first
   episode, the dataset is created).
3. The operator teleoperates the pick-and-place; SPACE ends recording wherever
   the arm happens to be (it is not moved to any end pose).
4. The overhead camera is polled for the placed cube (the plate is not
   re-checked) to record the final placement; while it is hidden -- usually by
   the arm -- the operator is asked to move the arm clear. Once seen, the episode
   ends and the operator resets the scene for the next one.

At any point up to the end of recording the operator can press ESC to back out:
mid-scene it aborts the current scene (discarding an in-progress take) and drops
back to the ready gate; at the ready gate it quits the whole run.

Each completed episode is committed with its ``cube_start_*``/``target_*`` and
``cube_end_*``/``placement_detected`` metadata plus ``driver="teleop"``. To stop the run (ESC at
the ready gate, or Ctrl-C) the arm is left where it is -- not parked -- and, once
the operator confirms, torque is released so it ends limp.

``--gripper-mode`` controls how the leader gripper maps onto the follower:

- ``remap`` (default): linearly map the leader's full travel onto a custom
  ``[--gripper-closed-position, --gripper-open-position]`` window, so a full
  squeeze bottoms out at a *safe* cube grip (never crushing) and a full release
  only partly opens. Assumes the leader reads ~100 fully open and ~0 fully
  closed (verify the direction live once).
- ``match-analytic``: clamp the leader command to the follower's calibrated
  closed/open range ([2.3, 98.5]) so teleop gripper actions share the same scale
  as the analytic collector and the two sources can be trained on together.
- ``passthrough``: send the raw leader 0-100 command unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from pick_and_place.dataset_metadata import cube_pose_metadata, driver_metadata
from pick_and_place.episodes import _build_model
from pick_and_place.executor import (
    CONTROL_HZ,
    RAMP_DURATION,
    RecordingSession,
    clamp_and_warn,
    follower_clamp_limits,
)
from pick_and_place.follower import (
    GRIPPER_INDEX,
    GRIPPER_READBACK_CLOSED,
    GRIPPER_READBACK_OPEN,
    action_to_joints,
    joints_to_action,
    make_so101_follower,
    make_so101_leader,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.overhead_detection import (
    DEFAULT_ALERT_SOUND,
    OperatorNotifier,
    OverheadDetectionDebug,
    empty_overhead_debug,
    final_placement_metadata,
    track_cube,
    track_drop_zone_square,
    write_overhead_debug_image,
)
from pick_and_place.paper_detection import PaperTracker

# Seconds each detection attempt stares at the overhead feed before giving up.
DETECT_TIMEOUT = 2.0
# Key the operator presses to abort the current scene/take (back to the ready gate).
ESC = "\x1b"
# Short, chirpy sound played (no speech) at the instant recording actually starts.
DEFAULT_START_SOUND = "/System/Library/Sounds/Glass.aiff"


def make_gripper_transform(
    mode: str, open_position: float, closed_position: float
) -> Callable[[float], float] | None:
    """Build the leader->follower gripper command map for ``--gripper-mode``.

    ``passthrough`` sends the raw leader 0-100 command; ``match-analytic`` clamps
    it to the follower's calibrated closed/open range; ``remap`` linearly maps the
    leader's full 0-100 travel onto ``[closed_position, open_position]`` (leader
    fully open -> ``open_position``, fully closed -> ``closed_position``), so a
    full squeeze bottoms out at a safe grip rather than a hard close.
    """
    if mode == "passthrough":
        return None
    if mode == "match-analytic":
        low, high = GRIPPER_READBACK_CLOSED, GRIPPER_READBACK_OPEN
        return lambda g: float(np.clip(g, low, high))
    if mode == "remap":
        span = open_position - closed_position
        return lambda g: closed_position + float(np.clip(g, 0.0, 100.0)) / 100.0 * span
    raise ValueError(f"unknown gripper mode {mode!r}")


class CameraReader:
    """Background reader keeping a single-slot 'latest frame' buffer fresh.

    The record loop snapshots ``latest()`` once per tick rather than calling
    ``read()`` inline, so a slow capture never stalls the loop and every frame
    logged is the most recent one.
    """

    def __init__(self, cap, label: str) -> None:
        self._cap = cap
        self._label = label
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def latest(self):
        with self._lock:
            return self._frame

    def wait_for_first(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.001)
        raise RuntimeError(f"timed out waiting for the {self._label} camera stream to start")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def _ramp_follower(
    follower,
    target_real: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    warned: set[str],
) -> None:
    """Smoothstep the real arm from its current pose onto ``target_real``.

    The one-time startup ramp: the only follower motion this script commands
    itself. Everything after is driven by the leader through :class:`Teleop`.
    """
    current = action_to_joints(follower.get_observation(), target_real)
    delta = target_real - current
    steps = max(1, round(RAMP_DURATION * CONTROL_HZ))
    period = 1.0 / CONTROL_HZ
    for i in range(1, steps + 1):
        step_start = time.monotonic()
        interp = clamp_and_warn(current + _smoothstep(i / steps) * delta, low, high, warned)
        follower.send_action(joints_to_action(interp))
        remaining = period - (time.monotonic() - step_start)
        if remaining > 0:
            time.sleep(remaining)


class Teleop:
    """Continuously stream the leader's pose onto the follower.

    Started once, after the startup ramp has brought the follower onto the
    leader pose. From then on the follower is only ever driven by the leader:
    the background loop reads the leader every tick and sends it to the follower
    at ``CONTROL_HZ``, so whenever the operator moves the leader the follower
    tracks it live -- during detection prompts, between episodes, and while
    recording alike.

    Recording taps into that same stream: between :meth:`begin_recording` and
    :meth:`end_recording` each tick also snapshots the follower readback (state),
    the leader command (action), and both camera frames, and queues one dataset
    frame. If ``gripper_transform`` is set it is applied to the leader gripper
    command before it is sent and recorded (see :func:`make_gripper_transform`).
    """

    def __init__(
        self,
        *,
        leader,
        follower,
        clamp_low: np.ndarray,
        clamp_high: np.ndarray,
        clip_warned: set[str],
        gripper_transform: Callable[[float], float] | None,
        start_command: np.ndarray,
    ) -> None:
        self._leader = leader
        self._follower = follower
        self._low = clamp_low
        self._high = clamp_high
        self._warned = clip_warned
        self._gripper_transform = gripper_transform
        self._prev = start_command.copy()
        self._running = True
        self._lock = threading.Lock()
        self._recording = False
        self._wrist: CameraReader | None = None
        self._overhead: CameraReader | None = None
        self._queue: queue.Queue | None = None
        self._frames = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    def begin_recording(
        self, wrist: CameraReader, overhead: CameraReader, record_queue: queue.Queue
    ) -> None:
        with self._lock:
            self._wrist = wrist
            self._overhead = overhead
            self._queue = record_queue
            self._frames = 0
            self._recording = True

    def end_recording(self) -> int:
        """Stop tapping frames off the stream and return the recorded count.

        Safe to call more than once; the stream itself keeps running so the
        follower stays under the leader's control between episodes.
        """
        with self._lock:
            self._recording = False
            frames = self._frames
            self._wrist = None
            self._overhead = None
            self._queue = None
        return frames

    def _loop(self) -> None:
        period = 1.0 / CONTROL_HZ
        next_tick = time.monotonic()
        while self._running:
            commanded = action_to_joints(self._leader.get_action(), self._prev)
            commanded = clamp_and_warn(commanded, self._low, self._high, self._warned)
            if self._gripper_transform is not None:
                commanded[GRIPPER_INDEX] = self._gripper_transform(float(commanded[GRIPPER_INDEX]))
            self._follower.send_action(joints_to_action(commanded))
            self._prev = commanded

            with self._lock:
                recording = self._recording
                wrist = self._wrist
                overhead = self._overhead
                record_queue = self._queue
            if recording and wrist is not None and overhead is not None:
                measured = action_to_joints(self._follower.get_observation(), commanded)
                wrist_frame = wrist.latest()
                overhead_frame = overhead.latest()
                if wrist_frame is not None and overhead_frame is not None:
                    record_queue.put(
                        (
                            measured.astype(np.float32),
                            commanded.astype(np.float32),
                            wrist_frame,
                            overhead_frame,
                        )
                    )
                    with self._lock:
                        self._frames += 1

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                next_tick = time.monotonic()


def _read_key(accept: str, message: str) -> str:
    """Block until the operator presses one of ``accept`` and return that char.

    A single keypress in cbreak mode, so nothing needs an Enter unless asked for
    (handy while both hands are on the leader): space is ``" "`` and Enter is
    ``"\\n"`` (a CR is normalised to it). Other keys are ignored until an accepted
    one is pressed. Falls back to a line read when stdin is not an interactive
    terminal (a bare Enter reads as ``"\\n"``). Ctrl-C still interrupts.
    """
    print(message, end="", flush=True)
    if not sys.stdin.isatty():
        text = sys.stdin.readline().strip()
        ch = text[:1].lower() if text else "\n"
        print()
        return ch if ch in accept else next(iter(accept))
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            ch = "\n" if ch == "\r" else ch.lower()
            if ch in accept:
                print()
                return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def detect_scene(
    overhead_cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    drop_zone_tracker: PaperTracker,
    drop_zone_color: str,
    notifier: OperatorNotifier,
) -> tuple[CubePose, CubePose, OverheadDetectionDebug] | None:
    """Detect the initial cube pose and the drop-zone target for one episode.

    Loops until both are visible in their allowed zones, prompting the operator
    to fix the scene between attempts. Returns ``(source, target, debug)``, or
    ``None`` if the operator aborts the scene.
    """
    while True:
        print("Scanning the overhead camera for the cube and drop zone...")
        debug = empty_overhead_debug()
        target = track_drop_zone_square(
            overhead_cap,
            camera_name,
            model,
            data,
            drop_zone_tracker,
            drop_zone_color,
            timeout=DETECT_TIMEOUT,
            debug=debug,
        )
        source = track_cube(
            overhead_cap,
            camera_name,
            model,
            data,
            DETECT_TIMEOUT,
            debug=debug,
        )
        if source is not None and target is not None:
            print(
                f"Initial cube: ({source.x:.3f}, {source.y:.3f}, yaw {source.yaw:+.2f}); "
                f"target: ({target.x:.3f}, {target.y:.3f})."
            )
            return source, target, debug

        missing = []
        if source is None:
            missing.append("cube (in the pickup zone)")
        if target is None:
            missing.append(f"{drop_zone_color} drop-zone square")
        notifier.alert("Cannot see the " + " and the ".join(missing) + ".")
        reply = _read_key(
            " " + ESC,
            "Fix the scene, then press SPACE to retry (or ESC to abort this scene): ",
        )
        if reply == ESC:
            return None


def look_for_final_cube(
    overhead_cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_debug: OverheadDetectionDebug,
    notifier: OperatorNotifier,
) -> tuple[CubePose, OverheadDetectionDebug]:
    """Re-detect the cube after an episode to measure the final placement.

    The arm usually sits between the overhead camera and the placed cube, so this
    keeps polling -- asking the operator to move the arm clear -- until the cube
    comes into view, then returns it with a debug snapshot carrying the target
    overlay. Ctrl-C aborts the run if the cube can never be seen.
    """
    debug = OverheadDetectionDebug(
        bgr=initial_debug.bgr,
        camera_matrix=initial_debug.camera_matrix,
        camera_position=initial_debug.camera_position,
        camera_rotation=initial_debug.camera_rotation,
        target=initial_debug.target,
    )
    attempt = 0
    while True:
        cube = track_cube(
            overhead_cap,
            camera_name,
            model,
            data,
            DETECT_TIMEOUT,
            return_out_of_zone=True,
            debug=debug,
        )
        if cube is not None:
            return cube, debug
        # Each track_cube attempt already spent ~DETECT_TIMEOUT looking; alert on
        # the first miss and then only periodically so the voice isn't constant.
        if attempt % 4 == 0:
            notifier.alert(
                "Cannot see the cube for the placement check. "
                "Move the arm out of the camera's view."
            )
        attempt += 1


def record_episode(
    teleop: Teleop,
    overhead_cap,
    wrist_cap,
    recording: RecordingSession,
    task: str,
    notifier: OperatorNotifier,
    start_sound: str | None,
) -> tuple[bool, int]:
    """Record one teleoperated episode off the running leader->follower stream.

    Opens the per-episode camera reader threads, taps the ``teleop`` stream for
    one dataset frame per tick, and runs until the operator presses SPACE (end)
    or ESC (abort). Returns ``(aborted, frames)``; the caller commits the pending
    episode when not aborted, or discards it when aborted. The follower is not
    touched here -- it stays under the leader's control via ``teleop`` throughout.

    The go signal fires from here, at the instant capture actually starts (after
    the cameras warm up and, on the first episode, the dataset is created) -- not
    back when the scene was detected, which can be seconds earlier.

    The camera readers run only for the duration of the episode: between episodes
    the overhead cap is read directly by the detection helpers, so a reader
    thread must not also be running then (two threads reading one
    ``VideoCapture`` corrupt each other's frames).
    """
    import cv2

    overhead = CameraReader(overhead_cap, "overhead")
    wrist = CameraReader(wrist_cap, "wrist")
    record_queue: queue.Queue = queue.Queue()

    def record_writer() -> None:
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
                    "task": task,
                }
            )

    writer_thread = None
    frames = 0
    aborted = False
    try:
        first_wrist = wrist.wait_for_first()
        first_overhead = overhead.wait_for_first()
        # Create the dataset on the first episode, once the frame shapes are known.
        if recording.dataset is None:
            recording.create_dataset(first_wrist.shape, first_overhead.shape)

        writer_thread = threading.Thread(target=record_writer, daemon=True)
        writer_thread.start()

        teleop.begin_recording(wrist, overhead, record_queue)
        notifier.chirp(start_sound)
        key = _read_key(
            " " + ESC, "Recording... press SPACE to end the episode (or ESC to abort): "
        )
        frames = teleop.end_recording()
        aborted = key == ESC
    finally:
        teleop.end_recording()
        overhead.close()
        wrist.close()
        if writer_thread is not None:
            record_queue.put(None)
            writer_thread.join(timeout=30.0)
    return aborted, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True, help="serial port of the SO-101 leader")
    parser.add_argument("--leader-id", default="liddy", help="leader calibration id (default: liddy)")
    parser.add_argument("--follower-port", required=True, help="serial port of the SO-101 follower")
    parser.add_argument(
        "--follower-id", default="folly", help="follower calibration id (default: folly)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="number of episodes to record; 0 means loop until quit (default: 0)",
    )
    parser.add_argument(
        "--drop-zone-color",
        choices=("black", "white"),
        default="black",
        help="color of the drop-zone square to detect (default: black)",
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
        "--start-sound",
        default=DEFAULT_START_SOUND,
        help="short sound (no speech) played the instant recording starts "
        f"(default: {DEFAULT_START_SOUND})",
    )
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera")
    parser.add_argument("--camera-name", default="overhead_camera", help="camera name in the model")
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera")
    parser.add_argument(
        "--gripper-mode",
        choices=("remap", "match-analytic", "passthrough"),
        default="remap",
        help="how the leader gripper maps onto the follower: 'remap' linearly maps the leader's "
        "full travel onto [--gripper-closed-position, --gripper-open-position] so a full squeeze "
        "is a safe grip; 'match-analytic' clamps to the analytic collector's range "
        f"([{GRIPPER_READBACK_CLOSED}, {GRIPPER_READBACK_OPEN}]); 'passthrough' sends the raw "
        "leader 0-100 (default: remap)",
    )
    parser.add_argument(
        "--gripper-open-position",
        type=float,
        default=50.0,
        help="remap mode: follower gripper position (0-100) at leader fully open (default: 50)",
    )
    parser.add_argument(
        "--gripper-closed-position",
        type=float,
        default=10.0,
        help="remap mode: follower gripper position (0-100) at leader fully closed, a safe "
        "cube grip that never crushes (default: 10)",
    )
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve the overhead camera extrinsics live from the workspace-frame AprilTags at "
        "startup (default: on; --no-recalibrate uses the saved sidecar extrinsics)",
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
        "--save-overhead-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save initial/final overhead verification images into the run directory (default: on)",
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
        help="LeRobot video codec (default: auto = best available HW encoder)",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video in real time during capture (default: on)",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads LeRobot uses for PNG-then-encode mode",
    )
    args = parser.parse_args()
    if args.gripper_mode == "remap":
        for name, value in (
            ("--gripper-open-position", args.gripper_open_position),
            ("--gripper-closed-position", args.gripper_closed_position),
        ):
            if not 0.0 <= value <= 100.0:
                parser.error(f"{name} must be in [0, 100]")
        if args.gripper_closed_position >= args.gripper_open_position:
            parser.error("--gripper-closed-position must be less than --gripper-open-position")

    notifier = OperatorNotifier(enabled=args.operator_alerts, sound_path=args.alert_sound)

    import cv2

    from pick_and_place.cam_align_solve import (
        ExtrinsicsSolveError,
        apply_solve_result,
        check_solve_plausible,
        parse_index_or_path,
        solve_overhead_extrinsics,
    )
    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.workspace_overlays import PAN_AXIS

    def require_intrinsics(camera_name: str, override) -> None:
        path = (
            Path(override) if override is not None
            else LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
        )
        if not path.exists():
            raise SystemExit(f"Missing {camera_name} intrinsics at {path}. Calibrate the camera first.")

    require_intrinsics(args.camera_name, args.overhead_intrinsics)
    require_intrinsics("wrist_camera", None)

    print("Building scene...")
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(
        dummy_source,
        include_environment=True,
        paper_target_marker=True,
    )
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)

    kinematics = derive_kinematics(model)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    print("Opening overhead camera...")
    overhead_cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    overhead_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    overhead_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not overhead_cap.isOpened():
        overhead_cap.release()
        raise SystemExit(f"Could not open the overhead camera {args.camera!r}.")

    print("Opening wrist camera...")
    wrist_cap = cv2.VideoCapture(parse_index_or_path(args.wrist_camera), backend)
    wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not wrist_cap.isOpened():
        overhead_cap.release()
        wrist_cap.release()
        raise SystemExit(f"Could not open the wrist camera {args.wrist_camera!r}.")

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
            overhead_cap.release()
            wrist_cap.release()
            raise SystemExit(
                "Overhead calibration failed: never saw all four workspace-frame tags in one "
                "frame. Clear the camera view and check the tags."
            )
        try:
            check_solve_plausible(result)
        except ExtrinsicsSolveError as exc:
            overhead_cap.release()
            wrist_cap.release()
            raise SystemExit(f"Overhead calibration rejected: {exc}") from exc
        apply_solve_result(model, data, args.camera_name, result)
        print(
            f"Overhead extrinsics solved: {result.reprojection_error_px:.2f}px, "
            f"{result.nominal_delta.translation_m * 1000.0:.1f}mm / "
            f"{result.nominal_delta.rotation_deg:.2f}deg from nominal."
        )

    print("Connecting to leader...")
    leader = make_so101_leader(args.leader_port, args.leader_id)
    leader.connect(calibrate=True)

    print("Connecting to follower...")
    # Keep torque on a plain disconnect (crash / mid-loop exit) so the arm holds
    # rather than going limp; torque is only released deliberately at shutdown.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    gripper_transform = make_gripper_transform(
        args.gripper_mode, args.gripper_open_position, args.gripper_closed_position
    )

    # The one and only follower motion this script commands itself: ramp the
    # follower smoothly onto the leader's current pose. From here on the leader
    # drives the follower continuously through the Teleop stream.
    print("Ramping the follower onto the leader pose...")
    start_command = clamp_and_warn(
        action_to_joints(leader.get_action(), np.zeros(len(clamp_low))),
        clamp_low,
        clamp_high,
        clip_warned,
    )
    if gripper_transform is not None:
        start_command[GRIPPER_INDEX] = gripper_transform(float(start_command[GRIPPER_INDEX]))
    _ramp_follower(follower, start_command, clamp_low, clamp_high, clip_warned)

    teleop = Teleop(
        leader=leader,
        follower=follower,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
        clip_warned=clip_warned,
        gripper_transform=gripper_transform,
        start_command=start_command,
    )
    teleop.start()
    print("Leader is now driving the follower. Move the arm with the leader.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(__file__).resolve().parents[2] / "datasets" / timestamp
    )
    overhead_debug_dir = dataset_root / "overhead_debug"
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

    episode_index = 0
    try:
        while args.episodes == 0 or episode_index < args.episodes:
            print(
                f"\n=== Episode {episode_index + 1}"
                f"{f'/{args.episodes}' if args.episodes else ''} ==="
            )
            if _read_key(
                " " + ESC,
                "Set up the scene (cube in the pickup zone, plate in the drop zone) and put "
                "the arm at a start pose, then press SPACE when ready -- recording starts as "
                "soon as the cube and plate are found (or ESC to quit): ",
            ) == ESC:
                break

            scene = detect_scene(
                overhead_cap,
                args.camera_name,
                model,
                data,
                drop_zone_tracker,
                args.drop_zone_color,
                notifier,
            )
            if scene is None:
                print("Scene aborted.")
                continue
            source, target, initial_debug = scene

            print("Cube and plate found; starting the recording...")
            aborted, frames = record_episode(
                teleop, overhead_cap, wrist_cap, recording, args.task, notifier, args.start_sound
            )
            if aborted:
                if recording.dataset is not None and recording.dataset.has_pending_frames():
                    recording.dataset.clear_episode_buffer()
                print(f"Recording aborted after {frames} frames (discarded).")
                continue
            print(f"Episode captured {frames} frames.")

            # Write debug images only now: the overhead_debug dir lives under the
            # dataset root, and LeRobotDataset.create (run lazily inside
            # record_episode on the first episode) requires that root not to exist
            # yet, so nothing may create it beforehand.
            if args.save_overhead_debug and initial_debug.bgr.size:
                path = overhead_debug_dir / f"episode_{episode_index:05d}_initial.jpg"
                write_overhead_debug_image(path, initial_debug)
                print(f"Saved overhead initial debug image: {path}")

            dropped = recording.dropped_frame_count()
            if dropped:
                raise RuntimeError(
                    f"Streaming video encoder dropped {dropped} frame(s): the encoder cannot keep "
                    "pace with capture, which would desync the video from the recorded frames. Use "
                    "a hardware vcodec (auto) or raise the encoder queue size."
                )

            print("Checking final cube placement from the overhead camera...")
            final_cube, final_debug = look_for_final_cube(
                overhead_cap,
                args.camera_name,
                model,
                data,
                initial_debug,
                notifier,
            )
            if args.save_overhead_debug and final_debug.bgr.size:
                path = overhead_debug_dir / f"episode_{episode_index:05d}_final.jpg"
                write_overhead_debug_image(path, final_debug, show_distance=True)
                print(f"Saved overhead final debug image: {path}")

            metadata = cube_pose_metadata(source, target)
            metadata.update(final_placement_metadata(final_cube, target))
            metadata.update(driver_metadata("teleop"))
            recording.save_episode(metadata)
            episode_index += 1
            print(f"Saved episode {episode_index} to dataset. Reset the scene for the next one.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        overhead_cap.release()
        wrist_cap.release()
        if recording.dataset is not None:
            print("Finalizing dataset...")
            recording.finalize()
            print(f"Dataset written to {dataset_root}")
        # Stop streaming, then release torque so the arm ends limp (it is not
        # parked or rested). The follower holds its pose only until the operator
        # confirms, so a high pose does not drop unsupported.
        teleop.stop()
        _read_key(
            " ", "Support the arm, then press SPACE to release torque (it will go limp): "
        )
        try:
            follower.bus.disable_torque()
            print("Torque released.")
        except Exception as exc:  # noqa: BLE001 - best-effort torque release
            print(f"Warning: could not release torque: {exc}")
        print("Disconnecting hardware...")
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
