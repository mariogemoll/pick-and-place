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

The run is organised as a sequence of attempts. Each attempt locates the
drop-zone square (the success target) and the cube on the overhead camera —
panning the arm through random search poses to clear the view if either is
hidden, and asking the operator only if that dance comes up empty — then homes
the arm to a fresh randomish near-neutral start and runs the policy while
repeatedly scanning the overhead camera for the cube. A timed-out attempt returns
the arm to neutral before the next one begins. The cube counts as placed only once it sits at
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
camera plus its calibrated intrinsics and the saved overhead extrinsics.

Safety: the arm ramps smoothly from wherever it is parked onto each start pose
before the policy takes over, and on exit (success, Ctrl-C or step budget) it
parks NEUTRAL -> REST and releases torque. Every command is clamped to the
model's joint limits.
"""

from __future__ import annotations

import argparse
import os
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
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
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


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


class CameraReader:
    """Background reader keeping a single-slot 'latest frame' buffer fresh.

    The control loop snapshots ``frame`` once per tick rather than calling
    ``read()`` inline, so a slow capture never stalls the policy loop and the
    arm always acts on the most recent image rather than a buffered backlog.
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
        raise RuntimeError("timed out waiting for the camera stream to start")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


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

    print("Opening cameras...")
    overhead = CameraReader(args.camera, 1920, 1080, "overhead")
    wrist = CameraReader(args.wrist_camera, 1280, 720, "wrist")
    first_overhead = overhead.wait_for_first()
    first_wrist = wrist.wait_for_first()

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

    # Overhead detection: the success scan reads the cube in world coordinates and
    # the drop-zone square gives the target. Attempts (timeout + retry, and the
    # success check) are the default run mode, so this is always built.
    rng = np.random.default_rng()
    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
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
    # overhead camera must sit where the real one does; apply the saved extrinsics
    # and read the resulting fixed camera frame.
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if cam_id < 0:
        raise SystemExit(f"No camera named {args.camera_name!r} in the model.")
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()

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
        when ``--attempt-timeout`` elapses without a placement, or ``"success"``
        once the cube has been set down at the target (within ``--success-tolerance``
        in xy and ``--place-height-tolerance`` of its resting height). Placement is
        confirmed after ``--success-dwell`` seconds; the arm slow-down runs in
        parallel over the same window and is soft — success fires as soon as the
        cube has been placed for the dwell and the arm has slowed, or at
        ``--settle-timeout`` regardless. The timeout is disabled when
        ``--attempt-timeout`` is <= 0.
        """
        nonlocal tick
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
        try:
            return _control_loop(attempt_start, next_tick,
                                 report_time, report_tick, infer_seconds, placement)
        finally:
            stop_scan.set()
            scanner.join(timeout=2.0)

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
        while True:
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

            if tick % 10 == 0:
                np.set_printoptions(precision=2, suppress=True)
                now = time.monotonic()
                ticks = tick - report_tick
                rate = f"  {ticks / (now - report_time):5.1f} Hz" if ticks else ""
                infer = f"  infer {infer_seconds / ticks * 1000.0:5.1f} ms" if ticks else ""
                if args.attempt_timeout > 0:
                    clock = f"  {now - attempt_start:4.1f}/{args.attempt_timeout:.0f}s"
                else:
                    clock = f"  {now - attempt_start:4.1f}s"
                print(f"tick {tick:4d}  action={commanded}{rate}{infer}{clock}")
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

    try:
        print("Homing to the neutral pose...")
        go_neutral()
        attempt = 0
        while True:
            attempt += 1
            budget = f"timeout {args.attempt_timeout:.0f}s" if args.attempt_timeout > 0 else "no timeout"
            print(f"\n=== Attempt {attempt} ({budget}) ===")

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
