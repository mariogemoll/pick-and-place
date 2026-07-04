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

Safety: the arm ramps smoothly from wherever it is parked onto NEUTRAL before the
policy takes over, and on exit (Ctrl-C or step budget) it parks NEUTRAL -> REST
and releases torque. Every command is clamped to the model's joint limits.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# Some SmolVLM backbone ops are not implemented for Apple MPS; fall back to CPU
# for just those ops instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from pick_and_place import build_scene
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
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

    print(f"Instruction: {args.instruction!r}")
    if args.checkpoint == DEFAULT_CHECKPOINT:
        print("Running closed-loop. Actions are NOT task-calibrated (un-finetuned base).")
    else:
        print(f"Running closed-loop with fine-tuned checkpoint {args.checkpoint!r}.")

    period = 1.0 / CONTROL_HZ
    tick = 0
    parked = False
    try:
        print("Ramping real arm to the neutral start pose...")
        _ramp_follower(
            follower, neutral_real, clamp_low, clamp_high, clip_warned, args.max_joint_speed
        )

        next_tick = time.monotonic()
        report_time, report_tick, infer_seconds = next_tick, 0, 0.0
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
                print(f"tick {tick:4d}  action={commanded}{rate}{infer}")
                report_time, report_tick, infer_seconds = now, tick, 0.0

            tick += 1
            if args.steps and tick >= args.steps:
                break

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                # Don't issue a burst of catch-up commands after a long stall.
                next_tick = time.monotonic()
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
