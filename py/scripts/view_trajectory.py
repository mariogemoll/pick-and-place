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
import dataclasses
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
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
from pick_and_place.kinematics import So101Kinematics, derive_kinematics
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    PickAndCarry,
    pick_and_carry_candidates,
)
from pick_and_place.workspace_overlays import (
    AZIMUTH_MAX,
    AZIMUTH_MIN,
    PAN_AXIS,
    WORKSPACE_OVERLAYS,
)

_CLEARANCE_OVERLAY = next(o for o in WORKSPACE_OVERLAYS if o.name == "workspace_clearance_pregrasp")


def _build_geom_sets(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """Return (robot_geom_ids, env_geom_ids).

    Robot geoms: all geoms on bodies other than the worldbody and the pick_cube.
    Environment geoms: floor and pick_cube — the things we check robot against.
    """
    world_body_id = 0
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    robot_geom_ids = {
        gid
        for gid in range(model.ngeom)
        if model.geom_bodyid[gid] not in (world_body_id, cube_body_id)
    }
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return robot_geom_ids, {cube_geom_id, floor_geom_id}


def _scan_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> list[tuple[str, str]]:
    """Return (name1, name2) for robot↔environment and robot↔robot contacts."""
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        g1_robot = g1 in robot_geom_ids
        g2_robot = g2 in robot_geom_ids
        if (g1_robot and g2 in env_geom_ids) or (g2_robot and g1 in env_geom_ids) or (g1_robot and g2_robot):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            hits.append((n1, n2))
    return hits


def _preflight(
    model: mujoco.MjModel,
    trajectory: PickAndCarry,
    actuator_id: dict[str, int],
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> list[tuple[float, str, str]]:
    """Simulate the full trajectory in a shadow MjData and return collision events."""
    shadow = mujoco.MjData(model)
    for name, value in NEUTRAL_ARM_JOINTS.items():
        _set_joint(model, shadow, name, value)
        shadow.ctrl[actuator_id[name]] = value
    shadow.ctrl[actuator_id["gripper"]] = NEUTRAL_GRIPPER

    # Compensate for the physical 2.8° (0.0486795 rad) arm twist.
    wrist_roll = math.radians(2.8 - 90)
    _set_joint(model, shadow, "wrist_roll", wrist_roll)
    shadow.ctrl[actuator_id["wrist_roll"]] = wrist_roll

    mujoco.mj_forward(model, shadow)

    events: list[tuple[float, str, str]] = []
    while shadow.time < trajectory.duration:
        frame = trajectory.evaluate(shadow.time)
        for name, value in frame.joints.items():
            shadow.ctrl[actuator_id[name]] = value
        shadow.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, shadow)
        for n1, n2 in _scan_contacts(model, shadow, robot_geom_ids, env_geom_ids):
            events.append((shadow.time, n1, n2))
    return events


_JAW_PREFIXES = ("fixed_jaw_col", "moving_jaw_col")


def _is_jaw(n: str) -> bool:
    return n.startswith(_JAW_PREFIXES)


def _is_unexpected(n1: str, n2: str) -> bool:
    """False only for jaw↔cube contacts, which are the intentional grasp."""
    return not ((_is_jaw(n1) and n2 == "pick_cube") or (_is_jaw(n2) and n1 == "pick_cube"))


def _random_cube() -> CubePose:
    """Sample a cube pose uniformly inside the clearance-pregrasp annular sector."""
    r_inner, r_outer = _CLEARANCE_OVERLAY.inner_radius, _CLEARANCE_OVERLAY.outer_radius
    # Uniform area sampling: draw r² uniformly so density is flat in 2-D.
    r = math.sqrt(np.random.uniform(r_inner**2, r_outer**2))
    theta = np.random.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
    yaw = np.random.uniform(0.0, 2 * math.pi)
    return CubePose(
        x=PAN_AXIS[0] + r * math.cos(theta),
        y=PAN_AXIS[1] + r * math.sin(theta),
        z=CUBE_HALF_SIZE,
        yaw=yaw,
    )


_NEAR_NEUTRAL_JOINT_SCALE = 0.4  # ±radians of random joint perturbation from neutral


def _random_near_neutral() -> tuple[dict[str, float], float]:
    """Return arm joints and gripper perturbed slightly from the neutral pose."""
    joints = {
        name: value + np.random.uniform(-_NEAR_NEUTRAL_JOINT_SCALE, _NEAR_NEUTRAL_JOINT_SCALE)
        for name, value in NEUTRAL_ARM_JOINTS.items()
    }
    gripper = float(np.random.uniform(0.0, GRIPPER_OPEN))
    return joints, gripper


def _set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


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

    fixed_source = args.source is not None
    fixed_target = args.target is not None

    attempt = 0
    while True:
        attempt += 1

        source = (
            CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
            if fixed_source
            else _random_cube()
        )
        target = (
            CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
            if fixed_target
            else _random_cube()
        )
        start_joints, start_gripper = _random_near_neutral()
        end_joints, end_gripper = _random_near_neutral()

        spec = build_scene()
        cube = spec.body("pick_cube")
        cube.pos = (source.x, source.y, source.z)
        half_yaw = source.yaw / 2.0
        cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        cube.add_freejoint()  # make the cube a real dynamic body
        model = spec.compile()
        data = mujoco.MjData(model)

        kinematics = derive_kinematics(model)

        actuator_id = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        for name, value in start_joints.items():
            _set_joint(model, data, name, value)
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = start_gripper

        mujoco.mj_forward(model, data)

        robot_geom_ids, env_geom_ids = _build_geom_sets(model)

        trajectory = None
        for traj in pick_and_carry_candidates(kinematics, source, target):
            grasp = traj.grasp
            events = _preflight(model, traj, actuator_id, robot_geom_ids, env_geom_ids)
            unexpected = [(t, n1, n2) for t, n1, n2 in events if _is_unexpected(n1, n2)]
            if not unexpected:
                trajectory = traj
                print(
                    f"source: x={source.x:.4f}  y={source.y:.4f}  yaw={math.degrees(source.yaw):.1f}°"
                    f"  target: x={target.x:.4f}  y={target.y:.4f}  yaw={math.degrees(target.yaw):.1f}°"
                )
                print(f"grasp: face={grasp.face}  elbow={grasp.elbow}  carry={traj.carry.mode}  (attempt {attempt})")
                break
            seen_pairs: set[tuple[str, str]] = set()
            for t, n1, n2 in unexpected:
                key = (min(n1, n2), max(n1, n2))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    print(f"skip {grasp.face}/{grasp.elbow}: collision t={t:.3f}s  {n1} ↔ {n2}")

        if trajectory is not None:
            break
        if fixed_source and fixed_target:
            raise ValueError("no collision-free pick-and-carry found for this source/target")
        print(f"attempt {attempt}: no trajectory found, resampling...")

    trajectory = dataclasses.replace(
        trajectory,
        start_joints=start_joints,
        start_gripper=start_gripper,
        end_joints=end_joints,
        end_gripper=end_gripper,
    )

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
                    for n1, n2 in _scan_contacts(model, data, robot_geom_ids, env_geom_ids)
                    if _is_unexpected(n1, n2)
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
