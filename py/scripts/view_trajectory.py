#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the SO-101 through the pick-and-place trajectory under real physics.

The arm is controlled through the model's position-servo actuators: each frame
the trajectory's joint set points are written to ``data.ctrl`` and the simulation
is stepped, so gravity and contact are live. The cube gets a free joint and rests
on the floor as a genuine rigid body.

Phases: (1) neutral -> hover, (2) hover -> pregrasp at cube center, (3) grasp,
(4) lift and carry the grasped cube over to the hover above the target,
(5) release, lift clear, and flow back to neutral.
"""

from __future__ import annotations

import argparse
import csv
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place.episodes import (
    is_unexpected,
    prepare_episode,
    scan_contacts,
)
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    GRIPPER_INDEX,
    JOINT_NAMES,
    action_to_joints,
    clamp_joints,
    joints_to_action,
    load_follower_joint_offsets,
    make_so101_follower,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import So101Kinematics


# Rate at which set points are streamed to the physical follower and the motors
# are read back. The sim steps far faster; follower I/O is throttled to this.
CONTROL_HZ = 50.0
# Seconds spent smoothly ramping the real arm onto the trajectory's start pose
# before playback begins, so there is no jump from wherever it was parked.
RAMP_DURATION = 4.0
# Default playback pace for the physical arm: a fraction of the nominal speed so
# the first hardware passes are gentle. Sim-only playback runs at nominal (1.0).
# Scaling the trajectory clock slows every phase uniformly without touching the
# planner. Override with --speed.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="source cube (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="target (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--follower-port",
        default=None,
        help="serial port of the SO-101 follower; engages the real arm when set",
    )
    parser.add_argument(
        "--follower-id",
        default="folly",
        help="follower calibration id used by lerobot (default: folly)",
    )
    parser.add_argument(
        "--offsets-path",
        default=None,
        help="JSON of per-joint sim→real degree offsets (default: zero offsets)",
    )
    parser.add_argument(
        "--record-path",
        default=None,
        help="CSV path for the per-tick desired-vs-actual motor log",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="playback speed multiplier of the nominal trajectory pace "
        f"(1.0 = nominal; default {REAL_ARM_DEFAULT_SPEED} when --follower-port is set, else 1.0)",
    )
    args = parser.parse_args()

    source = (
        CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
        if args.source is not None
        else None
    )
    target = (
        CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
        if args.target is not None
        else None
    )

    episode = prepare_episode(np.random.default_rng(), source, target, verbose=True)
    model = episode.model
    data = episode.data
    kinematics = episode.kinematics
    actuator_id = episode.actuator_id
    robot_geom_ids = episode.robot_geom_ids
    env_geom_ids = episode.env_geom_ids
    trajectory = episode.trajectory
    start_joints = episode.start_joints
    start_gripper = episode.start_gripper

    # The physical arm is opt-in: only when a follower port is given. With zero
    # offsets the real frame is just the sim frame in degrees, so this run also
    # measures each joint's sim→real calibration bias (Phase 2 input).
    offsets = load_follower_joint_offsets(args.offsets_path)
    clamp_low, clamp_high = _follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    follower = None
    if args.follower_port is not None:
        follower = make_so101_follower(
            args.follower_port,
            args.follower_id,
            disable_torque_on_disconnect=False,
        )
        follower.connect()

    # Playback pace: the trajectory clock runs at `speed` × wall time, so a
    # factor below 1.0 slows every phase uniformly. The sim still steps in real
    # time (the viewer shows real-time physics); only the set points evolve slower.
    speed = args.speed if args.speed is not None else (REAL_ARM_DEFAULT_SPEED if follower else 1.0)
    if speed <= 0.0:
        raise ValueError("--speed must be positive")
    if follower is not None:
        print(f"Playback speed: {speed:g}× nominal  (run ≈ {trajectory.duration / speed:.1f}s)")

    # Per-tick log of (trajectory time, commanded real joints, motor readback).
    log_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    control_period = 1.0 / CONTROL_HZ
    last_control_t = -math.inf

    prev_contacts: set[tuple[str, str]] = set()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Open the viewer at the start pose first, then ramp the real arm onto
            # it so both are visibly aligned before any playback motion begins.
            if follower is not None:
                start_real = _clamp_and_warn(
                    sim_frame_to_real(start_joints, start_gripper, offsets),
                    clamp_low,
                    clamp_high,
                    clip_warned,
                )
                print("Ramping real arm to the trajectory start pose...")
                _ramp_to_start(follower, start_real, model, data, viewer)
            # The ramp advances data.time, so anchor the trajectory clock here.
            playback_start = data.time
            while viewer.is_running():
                step_start = time.time()
                traj_t = (data.time - playback_start) * speed
                frame = trajectory.evaluate(traj_t)
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
                    print(f"collision t={traj_t:.3f}s  {pair[0]} ↔ {pair[1]}")
                prev_contacts = curr_contacts

                # Stream the same set points to the real arm and read the motors
                # back, throttled to CONTROL_HZ (the sim above steps far faster).
                if follower is not None and data.time - last_control_t >= control_period:
                    last_control_t = data.time
                    commanded = _clamp_and_warn(
                        sim_frame_to_real(frame.joints, frame.gripper, offsets),
                        clamp_low,
                        clamp_high,
                        clip_warned,
                    )
                    follower.send_action(joints_to_action(commanded))
                    actual = action_to_joints(follower.get_observation(), commanded)
                    log_rows.append((traj_t, commanded, actual))

                viewer.sync()
                # Driving the real arm: one pass through the planned trajectory,
                # then stop so we can report. Sim-only: stay open until closed.
                if follower is not None and traj_t >= trajectory.duration:
                    break
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if follower is not None:
            _report_tracking(log_rows)
            if args.record_path is not None:
                _write_record(args.record_path, log_rows)
            follower.disconnect()


if __name__ == "__main__":
    main()
