#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a LeRobot policy (ACT, SmolVLA, ...) on the physical SO-101, closed-loop.

The hardware counterpart to ``run_policy_sim.py``. Where the sim run renders MuJoCo
cameras and integrates physics, this reads two real cameras and a real arm: each
control tick it snapshots the latest overhead and wrist frames, reads the
follower's joints, runs ``select_action``, and streams the predicted joints back
to the arm as position targets.

It is *simpler* than the sim run on the proprioception side because no frame
conversion is needed. The follower already reports and accepts the exact real
frame the dataset was recorded in — arm joints in degrees, gripper as a 0-100
position — which is the frame SmolVLA's state and action live in. So the
follower's reading is the observation state verbatim, and the predicted action
is sent verbatim (clamped to the joint limits). There are no sim→real offsets
and no radians anywhere on the policy path; MuJoCo is loaded only to derive
those joint limits and the neutral start pose.

The cameras do need conversion: each raw, lens-distorted frame is undistorted
with its calibrated intrinsics, center-cropped to the policy's aspect ratio, and
resized to its input resolution every tick, via the same geometry
``convert_dataset_resolution.py`` applies to recorded datasets — so the live
frames fed to the policy match the ones it was fine-tuned on, pixel-geometry for
pixel-geometry. The resolution defaults to whatever the checkpoint was trained on.

``--checkpoint`` selects the policy; the default ``lerobot/smolvla_base`` is an
un-finetuned plumbing spike (the arm moves but does not solve the task). A
checkpoint fine-tuned on the project's dataset is the real use case.

Two recording modes exist. ``--save-video`` writes the exact per-tick frames fed
to the policy (cropped and resized). ``--record-video`` instead records the whole
run continuously at each camera's native rate and resolution — undistorted but
never cropped — on one shared clock, optionally including a third
``--workspace-camera`` view and, with ``--record-audio``, the audio input muxed
into every video. That is the mode for watchable footage of the policy performing.
``--action-log`` additionally writes one .npz per attempt with the per-tick
measured state, the action the policy returned (the ensembled one when temporal
ensembling is on), the command actually sent, and every raw action chunk the
model predicted — all in the real frame — so the freshest prediction can be
compared offline against the ensembled/executed motion.

The run is organised as a sequence of attempts. Each attempt locates the
drop-zone square (the success target) and the cube on the overhead camera —
panning the arm through random search poses to clear the view if either is
hidden, and asking the operator only if that dance comes up empty — then homes
the arm to a fresh randomish near-neutral start and runs the policy while
repeatedly scanning the overhead camera for the cube. During an attempt the
operator can press Enter to declare it failed and skip straight to the next one.
A timed-out or abandoned attempt returns the arm to neutral before the next one
begins. The cube counts as placed only once it sits at
the target both in xy (``--success-tolerance``) and near its resting height
(``--place-height-tolerance``) — i.e. actually set down, not just carried above
the target. From the moment it is first seen placed, two things run in parallel:
the placement is confirmed after ``--success-dwell`` seconds, and the arm's
slow-down is watched over the same window. The placement is the success; the
slow-down is soft — success fires as soon as the cube has held for the dwell and
the arm has slowed, or at ``--settle-timeout`` regardless, so the policy never
lingers in post-placement, off-distribution territory. If the cube moves again
the placement resets. If no placement happens within ``--attempt-timeout``, the
attempt is abandoned and retried from a new start (the policy is strongest early
on, especially with temporal ensembling). On success the run exits by default; with ``--loop`` it
instead alerts the operator to reset the cube and target (audibly and via an
Enter prompt) and continues with the next attempt. This needs the overhead
camera plus its calibrated intrinsics. By default the overhead extrinsics are
solved live from the workspace-frame AprilTags at startup (so the success scan
reads the cube against where the camera actually is) and re-checked periodically
between attempts, stopping the run if the camera has drifted; ``--no-recalibrate``
uses the saved sidecar extrinsics instead.

Safety: the arm ramps smoothly from wherever it is parked onto each start pose
before the policy takes over, and on exit (success, Ctrl-C or step budget) it
parks NEUTRAL -> REST and releases torque. Every command is clamped to the
model's joint limits.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

# Some SmolVLM backbone ops are not implemented for Apple MPS; fall back to CPU
# for just those ops instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.camera_intrinsics import (
    LOCAL_CAMERA_INTRINSICS_DIR,
    load_camera_intrinsics,
    load_local_camera_intrinsics,
)
from pick_and_place.episodes import sample_hunt_pose, sample_near_neutral
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.overhead_detection import DEFAULT_ALERT_SOUND, OperatorNotifier
from pick_and_place.executor import (
    CONTROL_HZ,
    RAMP_DURATION,
    clamp_and_warn,
    follower_clamp_limits,
)
from pick_and_place.follower import (
    GRIPPER_INDEX,
    JOINT_NAMES,
    action_to_joints,
    joints_to_action,
    make_so101_follower,
    sim_frame_to_real,
)
from pick_and_place.image_rectify import build_undistort_map, transform_frame
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.trajectory import (
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)
from pick_and_place.policy import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INSTRUCTION,
    make_policy,
    resolve_checkpoint_cameras,
    select_device,
)

# During the settle phase the arm's peak joint speed must stay below
# ``--settle-speed`` continuously for this long before the placement counts as
# finished, so a momentary pause mid-retreat does not end the attempt early.
SETTLE_STILL_HOLD = 1.0


def _drain_stdin_lines() -> bool:
    """Return True if the operator has typed a line on stdin, consuming all
    pending lines. Non-blocking, so it can be polled from the control loop
    without stalling a tick."""
    typed = False
    while select.select([sys.stdin], [], [], 0)[0]:
        if not sys.stdin.readline():
            break  # EOF
        typed = True
    return typed


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


class CameraReader:
    """Background reader keeping a single-slot 'latest frame' buffer fresh.

    The control loop snapshots ``frame`` once per tick rather than calling
    ``read()`` inline, so a slow capture never stalls the policy loop and the
    arm always acts on the most recent image rather than a buffered backlog.

    ``on_frame`` may be set to a callable taking ``(bgr, monotonic_time)``; it
    is invoked from the reader thread for every captured frame, so a recorder
    sees the camera's full native rate rather than the control loop's snapshots.
    """

    def __init__(self, source: str, width: int, height: int, label: str) -> None:
        import cv2

        from pick_and_place.cam_align_solve import parse_index_or_path

        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(parse_index_or_path(source), backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {label} camera {source!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self.on_frame = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                captured_at = time.monotonic()
                with self._lock:
                    self._frame = frame
                on_frame = self.on_frame
                if on_frame is not None:
                    on_frame(frame, captured_at)

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
        raise RuntimeError("timed out waiting for the camera stream to start")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class ActionLog:
    """Accumulate one attempt's per-tick actions and write them to
    ``<root>/attempt_NNN.npz`` when the attempt ends.

    Logged per tick: the measured joint state, the action the policy returned
    (the ensembled one when temporal ensembling is on), and the velocity-capped
    command actually sent. Whenever the model predicted a fresh chunk that tick
    (every tick under ensembling, every ``n_action_steps`` ticks otherwise), the
    whole chunk is logged too, keyed by the tick it arrived on. Everything is in
    the real frame (degrees, gripper 0-100). Row 0 of a chunk is the model's
    freshest prediction for its arrival tick, so ``chunks[i, 0] - action`` at
    ``chunk_tick[i]`` measures how far the ensemble lags the newest prediction,
    and comparing chunks across arrival ticks exposes mode flips.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempt = 0
        self._clear()

    def _clear(self) -> None:
        self._tick: list[int] = []
        self._t: list[float] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._commanded: list[np.ndarray] = []
        self._chunk_tick: list[int] = []
        self._chunks: list[np.ndarray] = []

    def start_attempt(self) -> None:
        self.attempt += 1
        self._clear()

    def log_tick(
        self,
        tick: int,
        t: float,
        state: np.ndarray,
        action: np.ndarray,
        commanded: np.ndarray,
        chunk: np.ndarray | None = None,
    ) -> None:
        self._tick.append(tick)
        self._t.append(t)
        self._state.append(np.asarray(state, dtype=np.float32))
        self._action.append(np.asarray(action, dtype=np.float32))
        self._commanded.append(np.asarray(commanded, dtype=np.float32))
        if chunk is not None:
            self._chunk_tick.append(tick)
            self._chunks.append(np.asarray(chunk, dtype=np.float32))

    def end_attempt(self, outcome: str) -> None:
        if not self._tick:
            return
        path = self.root / f"attempt_{self.attempt:03d}.npz"
        np.savez_compressed(
            path,
            tick=np.array(self._tick, dtype=np.int64),
            t=np.array(self._t, dtype=np.float64),
            state=np.stack(self._state),
            action=np.stack(self._action),
            commanded=np.stack(self._commanded),
            chunk_tick=np.array(self._chunk_tick, dtype=np.int64),
            chunks=(
                np.stack(self._chunks)
                if self._chunks
                else np.zeros((0, 0, 0), dtype=np.float32)
            ),
            outcome=np.array(outcome),
        )
        print(f"Wrote action log: {path}")


def _ramp_follower(
    follower, target_real: np.ndarray, low, high, warned, max_joint_speed: float
) -> None:
    """Smoothstep the real arm from its current pose onto ``target_real``.

    The ramp is stretched so no arm joint exceeds ``max_joint_speed`` (deg/s),
    so a large move from a far parked pose obeys the same velocity cap as the
    closed-loop run rather than snapping over in a fixed window.
    """
    current = action_to_joints(follower.get_observation(), target_real)
    delta = target_real - current
    arm_travel = float(np.max(np.abs(delta[:GRIPPER_INDEX]))) if GRIPPER_INDEX else 0.0
    capped_duration = arm_travel / max_joint_speed if max_joint_speed > 0 else 0.0
    duration = max(RAMP_DURATION, capped_duration)
    steps = max(1, round(duration * CONTROL_HZ))
    period = 1.0 / CONTROL_HZ
    for i in range(1, steps + 1):
        step_start = time.monotonic()
        interp = clamp_and_warn(current + _smoothstep(i / steps) * delta, low, high, warned)
        follower.send_action(joints_to_action(interp))
        remaining = period - (time.monotonic() - step_start)
        if remaining > 0:
            time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF policy checkpoint")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--follower-port", required=True, help="serial port of the SO-101 follower")
    parser.add_argument(
        "--follower-id", default="folly", help="follower calibration id (default: folly)"
    )
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera")
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera")
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="height fed to the policy (default: the checkpoint's training height, else 480)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="width fed to the policy (default: the checkpoint's training width, else 640)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after this many control ticks (0 = run until Ctrl-C)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=100,
        help=(
            "queued actions to execute before re-querying a chunked policy "
            "(default: 100; matches common ACT checkpoints; temporal ensembling uses 1)"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help=(
            "enable ACT temporal ensembling with this coefficient, e.g. 0.01; "
            "requires --n-action-steps 1"
        ),
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=10.0,
        help=(
            "hard per-joint velocity cap in deg/s. Each tick the command may move "
            "at most this far from the arm's measured pose, so a wild prediction "
            "can only ever crawl. Lower it (e.g. 3) to go really slow; <=0 disables"
        ),
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help=(
            "directory to write <dir>/wrist.mp4 and <dir>/overhead.mp4 with the exact "
            "frames fed to the policy each tick"
        ),
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        help=(
            "root directory for continuous native-rate MP4s of the whole run; each run "
            "writes into <dir>/<timestamp>/: undistorted full-resolution wrist_live.mp4 "
            "and overhead_live.mp4 (plus workspace_live.mp4 with --workspace-camera) on "
            "a shared clock — unlike --save-video's cropped per-tick policy-input frames"
        ),
    )
    parser.add_argument(
        "--action-log",
        type=Path,
        default=None,
        help=(
            "root directory for per-attempt action logs; each run writes "
            "<dir>/<timestamp>/attempt_NNN.npz with the per-tick state, returned "
            "(ensembled) action, sent command, and every raw predicted chunk, all "
            "in the real frame"
        ),
    )
    parser.add_argument(
        "--workspace-camera",
        default=None,
        help="optional OpenCV index/path of a workspace camera to include in --record-video",
    )
    parser.add_argument(
        "--record-audio",
        action="store_true",
        help="capture the audio input and mux it into every --record-video MP4",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="sounddevice input name or index (default: system input device)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "on success, alert the operator to reset the scene and continue with a "
            "new attempt instead of exiting; without it the run exits on the first "
            "success"
        ),
    )
    parser.add_argument(
        "--attempt-timeout",
        type=float,
        default=20.0,
        help="seconds before an unsuccessful attempt is abandoned and retried from a "
        "fresh randomish start; <=0 disables the timeout (default: 20)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=1.0,
        help="seconds between overhead success scans during an attempt (default: 1)",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=0.04,
        help="cube-to-target xy distance counted as placed, in metres (default: 0.04)",
    )
    parser.add_argument(
        "--place-height-tolerance",
        type=float,
        default=0.02,
        help="how far the cube centre may sit above its resting height and still count "
        "as placed (not carried), in metres (default: 0.02)",
    )
    parser.add_argument(
        "--success-dwell",
        type=float,
        default=1.0,
        help="seconds the cube must stay placed at the target before the placement is "
        "confirmed (default: 1)",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=3.0,
        help="max seconds after the cube is first seen placed before success fires "
        "regardless of arm motion; also caps how long the slow-down is awaited "
        "(default: 3)",
    )
    parser.add_argument(
        "--settle-speed",
        type=float,
        default=5.0,
        help="arm peak joint speed (deg/s) below which it counts as slowed, letting "
        "success fire as soon as the placement dwell is met (default: 5)",
    )
    parser.add_argument(
        "--max-hunt-tries",
        type=int,
        default=5,
        help="pan-around search poses to try while looking for the cube or drop zone "
        "before asking the operator for help (default: 5)",
    )
    parser.add_argument(
        "--camera-name", default="overhead_camera", help="overhead camera name in the model"
    )
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve the overhead camera extrinsics live from the workspace-frame AprilTags at "
        "startup and refuse to start if the solve is implausible; the success scan then reads "
        "the cube against where the camera actually is (--no-recalibrate uses the saved sidecar "
        "extrinsics instead)",
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
        default=None,
        help="overhead camera intrinsics JSON for the solve (default: local sidecar)",
    )
    parser.add_argument(
        "--recalibrate-check-interval",
        type=float,
        default=120.0,
        help="minimum seconds between periodic overhead drift checks, run at attempt boundaries "
        "while the arm is at neutral; the run stops if the camera has drifted past the limits "
        "below. <=0 disables (default: 120)",
    )
    parser.add_argument(
        "--recalibrate-drift-mm",
        type=float,
        default=10.0,
        help="translation drift from the startup solve that stops the run (default: 10 mm)",
    )
    parser.add_argument(
        "--recalibrate-drift-deg",
        type=float,
        default=2.0,
        help="rotation drift from the startup solve that stops the run (default: 2 deg)",
    )
    parser.add_argument(
        "--drop-zone-color",
        choices=("black", "white"),
        default="black",
        help="color of the drop-zone square to detect as the target (default: black)",
    )
    parser.add_argument(
        "--operator-alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play a sound and speak operator alerts on macOS (default: on)",
    )
    parser.add_argument(
        "--alert-sound",
        default=DEFAULT_ALERT_SOUND,
        help="sound file played before spoken operator alerts",
    )
    args = parser.parse_args()
    if args.workspace_camera is not None and args.record_video is None:
        parser.error("--workspace-camera requires --record-video")
    if args.record_audio and args.record_video is None:
        parser.error("--record-audio requires --record-video")

    override = (args.image_height, args.image_width)
    if any(override) and not all(override):
        parser.error("pass both --image-height and --image-width, or neither")
    (img_h, img_w), (overhead_key, wrist_key) = resolve_checkpoint_cameras(
        args.checkpoint, override_hw=(args.image_height, args.image_width) if all(override) else None
    )

    import cv2

    device = select_device(args.device)
    print(f"Loading {args.checkpoint} on {device} (first run downloads the weights)...")
    print(
        f"Feeding {img_w}x{img_h} (WxH) frames as {overhead_key!r} (overhead) "
        f"and {wrist_key!r} (wrist)."
    )

    # MuJoCo is used only for the joint limits (to clamp commands) and to map the
    # neutral sim pose into the real frame for the start ramp — never stepped.
    model = build_scene(include_environment=True).compile()
    kinematics = derive_kinematics(model)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    neutral_real = sim_frame_to_real(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)
    rest_real = sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER)

    from lerobot.utils.control_utils import predict_action

    intrinsics_by_camera = load_local_camera_intrinsics()
    missing = [cam for cam in ("overhead_camera", "wrist_camera") if cam not in intrinsics_by_camera]
    if missing:
        raise SystemExit(f"no calibrated intrinsics for {missing}; cannot undistort")
    workspace_intrinsics = None
    if args.workspace_camera is not None:
        workspace_intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / "workspace_camera.json"
        if not workspace_intrinsics_path.exists():
            raise SystemExit(
                f"no calibrated intrinsics at {workspace_intrinsics_path}; cannot undistort"
            )
        workspace_intrinsics = load_camera_intrinsics(workspace_intrinsics_path)

    print("Opening cameras...")
    overhead = CameraReader(args.camera, 1920, 1080, "overhead")
    wrist = CameraReader(args.wrist_camera, 1280, 720, "wrist")
    first_overhead = overhead.wait_for_first()
    first_wrist = wrist.wait_for_first()
    workspace = first_workspace = None
    if args.workspace_camera is not None:
        workspace = CameraReader(args.workspace_camera, 1920, 1080, "workspace")
        first_workspace = workspace.wait_for_first()

    # Every frame is rectified to the same pinhole view the offline dataset
    # conversion produces, at the policy's input resolution, so the policy loads
    # against a fixed shape regardless of either camera's native resolution.
    overhead_undistort_map = build_undistort_map(
        intrinsics_by_camera["overhead_camera"], first_overhead.shape[1], first_overhead.shape[0], cv2
    )
    wrist_undistort_map = build_undistort_map(
        intrinsics_by_camera["wrist_camera"], first_wrist.shape[1], first_wrist.shape[0], cv2
    )

    policy, preprocessor, postprocessor = make_policy(
        args.checkpoint,
        (img_h, img_w),
        (overhead_key, wrist_key),
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    policy.reset()
    if hasattr(policy.config, "chunk_size") and hasattr(policy.config, "n_action_steps"):
        print(
            f"Policy chunks: predicts {policy.config.chunk_size}, "
            f"executes {policy.config.n_action_steps} before re-query."
        )
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        print(f"Temporal ensembling coeff: {policy.config.temporal_ensemble_coeff}")

    # Action logging: capture every raw chunk the model predicts by wrapping
    # predict_action_chunk, which select_action calls both under temporal
    # ensembling (every tick, before the ensembler averages it away) and in
    # queued mode (once per re-query). The control loop drains the capture each
    # tick and unnormalizes it through the same postprocessor as the returned
    # action, so the log compares like with like in real units.
    action_log = None
    captured_chunk: list = [None]
    if args.action_log is not None:
        import datetime

        log_dir = args.action_log / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        action_log = ActionLog(log_dir)
        print(f"Logging per-attempt actions and raw chunks to {log_dir}")
        if hasattr(policy, "predict_action_chunk"):
            _predict_action_chunk = policy.predict_action_chunk

            def _capture_predict_action_chunk(batch):
                chunk = _predict_action_chunk(batch)
                captured_chunk[0] = chunk
                return chunk

            policy.predict_action_chunk = _capture_predict_action_chunk
        else:
            print(
                "Warning: policy has no predict_action_chunk; logging without raw chunks."
            )

    print("Connecting to follower...")
    # Keep torque on a plain disconnect so the arm holds rather than going limp;
    # torque is only released deliberately at REST in the finally block.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    wrist_writer = overhead_writer = None
    if args.save_video is not None:
        import imageio.v2 as imageio

        args.save_video.mkdir(parents=True, exist_ok=True)
        wrist_writer = imageio.get_writer(args.save_video / "wrist.mp4", fps=CONTROL_HZ)
        overhead_writer = imageio.get_writer(args.save_video / "overhead.mp4", fps=CONTROL_HZ)
        print(f"Saving observation frames to {args.save_video}/{{wrist,overhead}}.mp4")

    # Continuous run recording: every camera's full native-rate, undistorted
    # stream (no cropping/resizing) on a shared clock, with optional audio.
    # Frames are submitted from the reader threads, so the recording sees every
    # captured frame, not just the ones the control loop happened to snapshot.
    recorder = None
    record_dir = None
    if args.record_video is not None:
        import datetime

        from pick_and_place.episode_video import LiveVideoRecorder

        record_dir = args.record_video / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_maps = {"overhead": overhead_undistort_map, "wrist": wrist_undistort_map}
        if workspace is not None:
            record_maps["workspace"] = build_undistort_map(
                workspace_intrinsics, first_workspace.shape[1], first_workspace.shape[0], cv2
            )
        recorder = LiveVideoRecorder(
            record_dir,
            record_maps,
            audio=args.record_audio,
            audio_device=(
                int(args.audio_device)
                if args.audio_device is not None and args.audio_device.isdecimal()
                else args.audio_device
            ),
        )
        overhead.on_frame = lambda bgr, t: recorder.submit("overhead", bgr, t)
        wrist.on_frame = lambda bgr, t: recorder.submit("wrist", bgr, t)
        if workspace is not None:
            workspace.on_frame = lambda bgr, t: recorder.submit("workspace", bgr, t)
        cams = "/".join(record_maps)
        audio_note = " with audio" if args.record_audio else ""
        print(f"Recording the {cams} cameras{audio_note} to {record_dir}")

    # Overhead detection: the success scan reads the cube in world coordinates and
    # the drop-zone square gives the target. Attempts (timeout + retry, and the
    # success check) are the default run mode, so this is always built.
    rng = np.random.default_rng()
    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.cam_align_solve import (
        ExtrinsicsSolveError,
        apply_solve_result,
        check_solve_plausible,
        pose_delta_mm_deg,
        solve_overhead_extrinsics,
    )
    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )
    from pick_and_place.cube_detection import (
        cube_pose_to_world,
        estimate_cube_pose,
        make_cube_detector,
    )
    from pick_and_place.overhead_detection import (
        CUBE_LOOK_TIMEOUT,
        track_cube,
        track_drop_zone_square,
    )
    from pick_and_place.paper_detection import PaperTracker

    # The success scan reads the cube pose in world coordinates, so the model's
    # overhead camera must sit where the real one does. Start from the saved
    # extrinsics; unless --no-recalibrate, they are re-solved live from the
    # workspace-frame tags at startup (see solve_startup_extrinsics below).
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if cam_id < 0:
        raise SystemExit(f"No camera named {args.camera_name!r} in the model.")
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    # Set by the startup overhead solve; the periodic drift check compares against it.
    startup_extrinsics: tuple[np.ndarray, np.ndarray] | None = None
    last_drift_check = 0.0  # monotonic time of the last drift solve

    det_intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
    if not det_intrinsics.exists():
        raise SystemExit(f"Missing {args.camera_name} intrinsics at {det_intrinsics}.")
    det_matrix, det_map = load_intrinsics(det_intrinsics, 1920, 1080, cv2)
    detector = make_cube_detector()
    drop_zone_tracker = PaperTracker()
    notifier = OperatorNotifier(enabled=args.operator_alerts, sound_path=args.alert_sound)

    class _ReaderCap:
        """Adapt the overhead CameraReader's latest-frame buffer to the
        ``cap.read()`` interface the overhead detection helpers expect."""

        def read(self):
            frame = overhead.latest()
            return frame is not None, frame

    adapter = _ReaderCap()

    def scan_cube_world():
        """Detect the cube on the latest overhead frame and return its (x, y, z)
        world position, or None if it is not currently visible."""
        frame = overhead.latest()
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.remap(rgb, *det_map, cv2.INTER_LINEAR)
        estimate = estimate_cube_pose(rgb, detector, det_matrix)
        if estimate is None:
            return None
        _, pos = cube_pose_to_world(estimate, cam_pos, cam_rot)
        return float(pos[0]), float(pos[1]), float(pos[2])

    print(f"Instruction: {args.instruction!r}")
    if args.checkpoint == DEFAULT_CHECKPOINT:
        print("Running closed-loop. Actions are NOT task-calibrated (un-finetuned base).")
    else:
        print(f"Running closed-loop with fine-tuned checkpoint {args.checkpoint!r}.")

    period = 1.0 / CONTROL_HZ
    tick = 0
    parked = False

    def run_attempt(target_xy) -> str:
        """Drive the policy closed-loop for one attempt.

        Returns ``"steps"`` when the global ``--steps`` budget is hit, ``"timeout"``
        when ``--attempt-timeout`` elapses without a placement, ``"abandoned"``
        when the operator presses Enter to declare the attempt failed, or ``"success"``
        once the cube has been set down at the target (within ``--success-tolerance``
        in xy and ``--place-height-tolerance`` of its resting height). Placement is
        confirmed after ``--success-dwell`` seconds; the arm slow-down runs in
        parallel over the same window and is soft — success fires as soon as the
        cube has been placed for the dwell and the arm has slowed, or at
        ``--settle-timeout`` regardless. The timeout is disabled when
        ``--attempt-timeout`` is <= 0.
        """
        nonlocal tick
        # Discard any stale lines typed during the ramp/hunt so an old Enter
        # press cannot instantly abandon the attempt that is just starting.
        _drain_stdin_lines()
        print("Press Enter at any time to abandon this attempt and retry from neutral.")
        attempt_start = time.monotonic()
        next_tick = time.monotonic()
        report_time, report_tick, infer_seconds = next_tick, tick, 0.0

        # Run the overhead placement scan on its own thread: the AprilTag detection
        # on the full-resolution frame takes tens of milliseconds, which would
        # stall the 30 Hz control loop if done inline. The heavy work (detection,
        # remap) releases the GIL, so it overlaps the policy inference cleanly. The
        # scan keeps a single shared timestamp — when the cube was first seen
        # continuously placed — which the control loop reads to time out placement,
        # confirm it, and monitor slow-down all against the same clock.
        placement = SimpleNamespace(since=None)
        stop_scan = threading.Event()

        def scan_loop() -> None:
            # The cube counts as placed only when it is at the target in xy *and*
            # back near its resting height (set down, not still carried above the
            # target) — so a fly-through during the carry never counts. Once a
            # placement has started, a lost sighting does NOT clear it: the
            # retreating arm routinely occludes a cube it just set down. Only a
            # clear "visible but not placed" reading (moved away or lifted) clears
            # it, which also lets the control loop reset if the cube moves again.
            while not stop_scan.wait(args.scan_interval):
                pose = scan_cube_world()
                if pose is None:
                    print("success scan: cube not visible")
                    continue
                x, y, z = pose
                dist = float(np.hypot(x - target_xy[0], y - target_xy[1]))
                above = z - CUBE_HALF_SIZE
                at_target = dist <= args.success_tolerance
                set_down = abs(above) <= args.place_height_tolerance
                print(
                    f"success scan: cube ({x:.3f}, {y:.3f}) {dist * 100.0:.1f} cm from "
                    f"target, {above * 100.0:+.1f} cm above rest"
                )
                if at_target and set_down:
                    if placement.since is None:
                        placement.since = time.monotonic()
                else:
                    placement.since = None

        scanner = threading.Thread(target=scan_loop, daemon=True)
        scanner.start()
        if action_log is not None:
            action_log.start_attempt()
        outcome = "error"
        try:
            outcome = _control_loop(attempt_start, next_tick,
                                    report_time, report_tick, infer_seconds, placement)
            return outcome
        finally:
            stop_scan.set()
            scanner.join(timeout=2.0)
            if action_log is not None:
                action_log.end_attempt(outcome)

    def _control_loop(attempt_start, next_tick,
                      report_time, report_tick, infer_seconds, placement) -> str:
        nonlocal tick
        # Placement confirmation and arm slow-down run concurrently from the moment
        # the cube is first seen placed (``placement.since``): the policy keeps
        # driving so the arm retreats, but the placement is the success — the
        # slow-down is soft and only trims the tail, never a hard requirement.
        still_since = None
        prev_arm = None
        prev_t = None
        announced = False
        raw_lag = None  # newest chunk's first action vs the ensembled one, deg
        while True:
            if _drain_stdin_lines():
                return "abandoned"
            overhead_bgr = overhead.latest()
            wrist_bgr = wrist.latest()
            overhead_rgb = cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
            overhead_rgb = transform_frame(
                overhead_rgb, overhead_undistort_map, img_w, img_h, cv2
            )
            wrist_rgb = transform_frame(wrist_rgb, wrist_undistort_map, img_w, img_h, cv2)

            state = action_to_joints(follower.get_observation(), neutral_real).astype(np.float32)
            observation = {
                "observation.state": state,
                overhead_key: overhead_rgb,
                wrist_key: wrist_rgb,
            }
            if wrist_writer is not None:
                wrist_writer.append_data(wrist_rgb)
                overhead_writer.append_data(overhead_rgb)

            infer_start = time.monotonic()
            action = predict_action(
                observation,
                policy,
                device,
                preprocessor,
                postprocessor,
                use_amp=False,
                task=args.instruction,
                robot_type="so101",
            )
            infer_seconds += time.monotonic() - infer_start
            action_real = action.to("cpu").numpy().reshape(-1)[: len(JOINT_NAMES)]
            target = clamp_and_warn(action_real, clamp_low, clamp_high, clip_warned)
            # Velocity cap: never command an arm joint more than one tick's worth
            # of travel beyond where the arm actually is. This bounds both speed
            # and the servo's position error regardless of what the policy asks
            # for. The gripper passes through (open/close should stay timely).
            commanded = target.copy()
            if args.max_joint_speed > 0:
                max_step = args.max_joint_speed / CONTROL_HZ
                arm_delta = target[:GRIPPER_INDEX] - state[:GRIPPER_INDEX]
                commanded[:GRIPPER_INDEX] = state[:GRIPPER_INDEX] + np.clip(
                    arm_delta, -max_step, max_step
                )
            follower.send_action(joints_to_action(commanded))

            # Drain the chunk captured during this tick's inference (if any) and
            # unnormalize the whole (chunk, dim) sequence in one pass — the
            # postprocessor's stats broadcast per action dimension. Row 0 is the
            # model's freshest prediction for this very tick, so its gap to the
            # returned (ensembled) action is the ensemble lag.
            chunk_real = None
            if captured_chunk[0] is not None:
                chunk_real = (
                    postprocessor(captured_chunk[0].squeeze(0))
                    .to("cpu")
                    .numpy()[:, : len(JOINT_NAMES)]
                )
                captured_chunk[0] = None
                raw_lag = float(
                    np.max(np.abs(chunk_real[0, :GRIPPER_INDEX] - action_real[:GRIPPER_INDEX]))
                )
            if action_log is not None:
                action_log.log_tick(
                    tick, time.monotonic(), state, action_real, commanded, chunk_real
                )

            if tick % 10 == 0:
                np.set_printoptions(precision=2, suppress=True)
                now = time.monotonic()
                ticks = tick - report_tick
                rate = f"  {ticks / (now - report_time):5.1f} Hz" if ticks else ""
                infer = f"  infer {infer_seconds / ticks * 1000.0:5.1f} ms" if ticks else ""
                lag = f"  raw-ens {raw_lag:4.1f}deg" if raw_lag is not None else ""
                if args.attempt_timeout > 0:
                    clock = f"  {now - attempt_start:4.1f}/{args.attempt_timeout:.0f}s"
                else:
                    clock = f"  {now - attempt_start:4.1f}s"
                print(f"tick {tick:4d}  action={commanded}{rate}{infer}{lag}{clock}")
                report_time, report_tick, infer_seconds = now, tick, 0.0

            tick += 1
            if args.steps and tick >= args.steps:
                return "steps"

            now = time.monotonic()
            arm = state[:GRIPPER_INDEX]

            # Track arm slow-down every tick so it is already known the moment the
            # placement dwell completes, rather than measured only afterwards.
            arm_settled = False
            if prev_arm is not None and now > prev_t:
                speed = float(np.max(np.abs(arm - prev_arm))) / (now - prev_t)
                if speed <= args.settle_speed:
                    if still_since is None:
                        still_since = now
                else:
                    still_since = None
                arm_settled = still_since is not None and now - still_since >= SETTLE_STILL_HOLD
            prev_arm, prev_t = arm.copy(), now

            since = placement.since
            if since is None:
                # Cube not (yet, or no longer) placed. Give up on the attempt once
                # the placement timeout passes with nothing set down.
                announced = False
                if args.attempt_timeout > 0 and now - attempt_start >= args.attempt_timeout:
                    return "timeout"
            else:
                if not announced:
                    print("Cube placed. Confirming placement while the arm slows down...")
                    announced = True
                placed_for = now - since
                # The dwell confirms the placement; the slow-down runs over the same
                # window and only trims the tail. Finish once the cube has held for
                # the dwell and the arm has slowed, or at --settle-timeout regardless
                # (so the policy never lingers in post-placement, off-distribution
                # territory). A cube that moves again clears placement.since above.
                if placed_for >= args.success_dwell and (arm_settled or placed_for >= args.settle_timeout):
                    reason = "arm settled" if arm_settled else "settle timeout"
                    print(f"Cube placed for {placed_for:.1f}s ({reason}). Success.")
                    return "success"

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                # Don't issue a burst of catch-up commands after a long stall.
                next_tick = time.monotonic()

    def go_neutral() -> None:
        _ramp_follower(
            follower, neutral_real, clamp_low, clamp_high, clip_warned, args.max_joint_speed
        )

    def hunt(label, detect):
        """Look for something on the overhead camera, panning the arm through fresh
        random search poses (the 'dance') to clear the view between tries. The arm
        can sit between the fixed overhead camera and the cube or square, so a look
        from one pose may be blocked while another is clear. Returns the detection
        or None after ``--max-hunt-tries`` looks."""
        for i in range(args.max_hunt_tries):
            if i > 0:
                arm, grip = sample_hunt_pose(rng)
                print(f"{label} look {i + 1}/{args.max_hunt_tries}: panning to a new search pose...")
                _ramp_follower(
                    follower, sim_frame_to_real(arm, grip), clamp_low, clamp_high, clip_warned,
                    args.max_joint_speed,
                )
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"{label} look 1/{args.max_hunt_tries}: searching from the current pose...")
            result = detect()
            if result is not None:
                return result
        return None

    def find_or_prompt(label, detect, missing_message):
        """Hunt for a detection; if the dance comes up empty, ask the operator to
        make it visible and try again. Returns the detection or None on Ctrl-D."""
        while True:
            result = hunt(label, detect)
            if result is not None:
                return result
            notifier.alert(missing_message)
            try:
                input(f"Make the {label} visible, then press Enter (Ctrl-D to stop)...")
            except EOFError:
                return None

    def detect_target():
        return track_drop_zone_square(
            adapter, args.camera_name, model, data, drop_zone_tracker, args.drop_zone_color
        )

    def detect_cube():
        return track_cube(adapter, args.camera_name, model, data, CUBE_LOOK_TIMEOUT)

    def solve_startup_extrinsics() -> None:
        """Solve the overhead extrinsics live from the workspace-frame tags, validate
        them, and apply them to the model so the success scan back-projects the cube
        against where the camera actually is. Refuses to start on a failed or
        implausible solve."""
        nonlocal startup_extrinsics, last_drift_check, cam_pos, cam_rot
        print("Solving overhead camera extrinsics from the workspace-frame tags...")
        result = solve_overhead_extrinsics(
            model,
            data,
            adapter,
            camera_name=args.camera_name,
            intrinsics_path=args.overhead_intrinsics,
            samples=args.recalibrate_samples,
            max_seconds=args.recalibrate_max_seconds,
            cv2_module=cv2,
        )
        if result is None:
            raise SystemExit(
                "Overhead calibration failed: never saw all four workspace-frame tags "
                "in one frame. Clear the camera view and check the tags."
            )
        try:
            check_solve_plausible(result)
        except ExtrinsicsSolveError as exc:
            raise SystemExit(f"Overhead calibration rejected: {exc}") from exc
        apply_solve_result(model, data, args.camera_name, result)
        startup_extrinsics = (
            np.array(result.pos, dtype=float),
            np.array(result.quat, dtype=float),
        )
        # The success scan back-projects the cube through this pose, so refresh the
        # cached camera frame it reads to match the freshly solved extrinsics.
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
        last_drift_check = time.monotonic()
        print(
            f"Overhead extrinsics solved: {result.reprojection_error_px:.2f}px, "
            f"{result.nominal_delta.translation_m * 1000.0:.1f}mm / "
            f"{result.nominal_delta.rotation_deg:.2f}deg from nominal."
        )

    def check_overhead_drift() -> None:
        """Re-solve the overhead extrinsics from the current (near-neutral) pose and
        stop the run if the camera has drifted from the startup calibration. Skips
        quietly if the tags are occluded, and is rate-limited to
        --recalibrate-check-interval so it only runs occasionally between attempts."""
        nonlocal last_drift_check
        if (
            not args.recalibrate
            or startup_extrinsics is None
            or args.recalibrate_check_interval <= 0
            or time.monotonic() - last_drift_check < args.recalibrate_check_interval
        ):
            return
        print("Drift check: re-solving overhead extrinsics...")
        saved_pos = model.cam_pos[cam_id].copy()
        saved_quat = model.cam_quat[cam_id].copy()
        check = solve_overhead_extrinsics(
            model,
            data,
            adapter,
            camera_name=args.camera_name,
            intrinsics_path=args.overhead_intrinsics,
            samples=args.recalibrate_samples,
            max_seconds=args.recalibrate_max_seconds,
            cv2_module=cv2,
        )
        # The re-solve only decides whether to stop; the startup calibration stays
        # live and is never re-applied mid-run.
        model.cam_pos[cam_id] = saved_pos
        model.cam_quat[cam_id] = saved_quat
        mujoco.mj_forward(model, data)
        last_drift_check = time.monotonic()
        if check is None:
            print("Drift check skipped: could not see all four tags (occluded). Continuing.")
            return
        drift_mm, drift_deg = pose_delta_mm_deg(
            startup_extrinsics[0],
            startup_extrinsics[1],
            np.array(check.pos, dtype=float),
            np.array(check.quat, dtype=float),
        )
        print(f"Overhead drift vs startup: {drift_mm:.1f}mm / {drift_deg:.2f}deg.")
        if drift_mm > args.recalibrate_drift_mm or drift_deg > args.recalibrate_drift_deg:
            raise SystemExit(
                f"Overhead camera drifted {drift_mm:.1f}mm / {drift_deg:.2f}deg since startup "
                f"(limits {args.recalibrate_drift_mm:.0f}mm / {args.recalibrate_drift_deg:.1f}deg). "
                "Stopping so the operator can check the mount and recalibrate."
            )

    try:
        print("Homing to the neutral pose...")
        go_neutral()
        if args.recalibrate:
            solve_startup_extrinsics()
        attempt = 0
        while True:
            attempt += 1
            budget = f"timeout {args.attempt_timeout:.0f}s" if args.attempt_timeout > 0 else "no timeout"
            print(f"\n=== Attempt {attempt} ({budget}) ===")

            # The arm is at neutral here (homed on entry and after every attempt), so
            # the tags are unoccluded — a good moment for the periodic drift check.
            check_overhead_drift()

            target = find_or_prompt(
                "drop-zone square", detect_target, "Drop-zone square not visible."
            )
            if target is None:
                break
            target_xy = (float(target.x), float(target.y))
            print(f"Target drop zone at ({target_xy[0]:.3f}, {target_xy[1]:.3f}).")

            cube = find_or_prompt(
                "cube", detect_cube, "Cube not visible in the pickup zone. Please reset it."
            )
            if cube is None:
                break
            print(f"Cube at ({cube.x:.3f}, {cube.y:.3f}).")

            # Each attempt starts from a fresh randomish near-neutral pose: the
            # policy is strongest early on, so a timed-out attempt is abandoned and
            # simply retried from a new start rather than left to flail.
            arm, grip = sample_near_neutral(rng)
            print("Ramping to a randomish start pose...")
            _ramp_follower(
                follower, sim_frame_to_real(arm, grip), clamp_low, clamp_high, clip_warned,
                args.max_joint_speed,
            )
            policy.reset()

            outcome = run_attempt(target_xy)

            if outcome == "steps":
                break
            if outcome == "timeout":
                print(f"TIMEOUT — no success within {args.attempt_timeout:.0f}s. "
                      "Returning to neutral and retrying.")
                go_neutral()
                continue
            if outcome == "abandoned":
                print("ABANDONED — operator declared the attempt failed. "
                      "Returning to neutral and retrying.")
                go_neutral()
                continue

            # Success. Exit by default; with --loop, hand the scene back to the
            # operator to reset and keep going.
            print("SUCCESS — the cube reached the target.")
            if not args.loop:
                break
            notifier.alert("Success. Please reset the cube and target for the next attempt.")
            print("Parking to the neutral pose before the next attempt...")
            go_neutral()
            try:
                input("Reset the scene, then press Enter for the next attempt (Ctrl-C to stop)...")
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        overhead.close()
        wrist.close()
        if workspace is not None:
            workspace.close()
        if recorder is not None:
            print(f"Finalizing the run recording at {record_dir}...")
            recorder.close()
        if wrist_writer is not None:
            wrist_writer.close()
            overhead_writer.close()
        try:
            print("Parking to NEUTRAL then REST...")
            follower.bus.enable_torque()
            _ramp_follower(
                follower, neutral_real, clamp_low, clamp_high, clip_warned, args.max_joint_speed
            )
            _ramp_follower(
                follower, rest_real, clamp_low, clamp_high, clip_warned, args.max_joint_speed
            )
            parked = True
        except Exception as exc:  # noqa: BLE001 - best-effort park before release
            print(f"Warning: could not park the arm: {exc}")
        if parked:
            print("At REST — releasing torque.")
            try:
                follower.bus.disable_torque()
            except Exception as exc:  # noqa: BLE001 - best-effort torque release
                print(f"Warning: could not release torque: {exc}")
        print("Disconnecting hardware...")
        follower.disconnect()
    print(f"Ran {tick} control ticks.")


if __name__ == "__main__":
    main()
