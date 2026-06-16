#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Batch-generate pick-and-place episodes under real physics and store them raw.

For each episode a random source/target cube and near-neutral start/end arm pose
are sampled and a collision-free pick-and-carry is found (same logic as
``view_trajectory``), then the trajectory is run headless — no viewer, no
rendering — and sampled at a fixed control rate into a ``.npz`` per episode.

No camera frames are stored: the full per-frame ``qpos``/``qvel`` is logged, so
the exact run can be reconstructed and any camera rendered later (see
``replay_episode.py``). Each file holds per-frame ``action``/``state`` (the
proprioceptive stream for IL/RL) plus the start/end poses of both the cube and
the robot, and a ``meta.json`` written alongside captures the run-wide layout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.episodes import (
    SUCCESS_XY_TOLERANCE,
    SUCCESS_YAW_TOLERANCE,
    SUCCESS_Z_TOLERANCE,
    EpisodeSamplingError,
    Episode,
    is_unexpected,
    placement_errors,
    prepare_episode,
    quat_yaw,
    scan_contacts,
)
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import CubePose

# Default rate at which the trajectory is sampled into the dataset (Hz). The sim
# steps far faster; frames are emitted on this slower clock — the same cadence a
# real follower would be commanded at.
DEFAULT_CONTROL_HZ = 50.0
# Per-episode budget of pose resamples before that episode is abandoned.
DEFAULT_MAX_ATTEMPTS = 50


def _cube_qpos_adr(model: mujoco.MjModel) -> int:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    return int(model.jnt_qposadr[model.body_jntadr[body]])


def _joint_qpos_adr(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]


def _action_vector(frame) -> np.ndarray:
    """The frame's joint set points laid out in ``JOINT_NAMES`` order (radians)."""
    values = [frame.joints[name] for name in JOINT_NAMES[:-1]]
    values.append(frame.gripper)
    return np.asarray(values, dtype=np.float64)


def run_episode(episode: Episode, control_hz: float) -> dict[str, np.ndarray]:
    """Step the prepared episode under physics, sampling the trajectory at
    ``control_hz`` into per-frame ``action``/``state``/``qpos``/``qvel`` arrays.

    Returns a dict of stacked arrays plus the start/end poses of the cube and
    robot and the realized final cube pose, ready to hand to ``np.savez``.
    """
    model, data = episode.model, episode.data
    trajectory = episode.trajectory
    actuator_id = episode.actuator_id
    cube_adr = _cube_qpos_adr(model)
    joint_adr = _joint_qpos_adr(model)
    control_period = 1.0 / control_hz

    times: list[float] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    qpos_log: list[np.ndarray] = []
    qvel_log: list[np.ndarray] = []
    collisions: list[tuple[float, str, str]] = []

    last_sample_t = -math.inf
    while True:
        traj_t = data.time
        frame = trajectory.evaluate(traj_t)
        for name, value in frame.joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = frame.gripper

        if traj_t - last_sample_t >= control_period:
            last_sample_t = traj_t
            times.append(traj_t)
            actions.append(_action_vector(frame))
            states.append(data.qpos[joint_adr].copy())
            qpos_log.append(data.qpos.copy())
            qvel_log.append(data.qvel.copy())
            for n1, n2 in scan_contacts(model, data, episode.robot_geom_ids, episode.env_geom_ids):
                if is_unexpected(n1, n2):
                    collisions.append((traj_t, n1, n2))

        if traj_t >= trajectory.duration:
            break
        mujoco.mj_step(model, data)

    cube_pos = data.qpos[cube_adr : cube_adr + 3]
    cube_quat = data.qpos[cube_adr + 3 : cube_adr + 7]
    cube_end = np.array([cube_pos[0], cube_pos[1], cube_pos[2], quat_yaw(cube_quat)])

    def _pose4(p: CubePose) -> np.ndarray:
        return np.array([p.x, p.y, p.z, p.yaw])

    def _robot6(joints: dict[str, float], gripper: float) -> np.ndarray:
        return np.array([joints[n] for n in JOINT_NAMES[:-1]] + [gripper])

    target = episode.target
    xy_err, z_err, yaw_err = placement_errors(cube_end, target)
    # The preflight vets the whole trajectory collision-free, so an accepted
    # episode should run without unexpected contacts; if any slip through, the run
    # clipped the floor or itself and is not a clean demonstration regardless of
    # where the cube landed, so it does not count as a success.
    success = (
        xy_err <= SUCCESS_XY_TOLERANCE
        and z_err <= SUCCESS_Z_TOLERANCE
        and yaw_err <= SUCCESS_YAW_TOLERANCE
        and len(collisions) == 0
    )

    return {
        "time": np.asarray(times),
        "action": np.asarray(actions),
        "state": np.asarray(states),
        "qpos": np.asarray(qpos_log),
        "qvel": np.asarray(qvel_log),
        "cube_start": _pose4(episode.source),
        "cube_target": _pose4(episode.target),
        "cube_end": cube_end,
        "robot_start": _robot6(episode.start_joints, episode.start_gripper),
        "robot_end": _robot6(episode.end_joints, episode.end_gripper),
        "grasp_face": np.array(episode.grasp.face),
        "grasp_elbow": np.array(episode.grasp.elbow),
        "carry_mode": np.array(episode.trajectory.carry.mode),
        "success": np.array(success),
        "final_xy_error": np.array(xy_err),
        "final_yaw_error": np.array(yaw_err),
        "duration": np.array(trajectory.duration),
        "control_hz": np.array(control_hz),
        "attempts": np.array(episode.attempts),
        "n_collisions": np.array(len(collisions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-episodes", type=int, default=10, help="episodes to record")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "episodes",
        help="directory for the .npz episodes and meta.json (default: py/out/episodes)",
    )
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed for reproducibility")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=DEFAULT_CONTROL_HZ,
        help=f"trajectory sampling rate (default {DEFAULT_CONTROL_HZ:g})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"pose resamples per episode before giving up (default {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print the grasp search log")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    successes = 0
    written = 0
    for index in range(args.num_episodes):
        try:
            episode = prepare_episode(rng, max_attempts=args.max_attempts, verbose=args.verbose)
        except EpisodeSamplingError as exc:
            print(f"episode {index}: skipped ({exc})")
            continue
        record = run_episode(episode, args.control_hz)
        path = args.out_dir / f"episode_{index:05d}.npz"
        np.savez_compressed(path, episode_index=np.array(index), seed=np.array(args.seed), **record)
        written += 1
        ok = bool(record["success"])
        successes += ok
        print(
            f"episode {index}: {len(record['time'])} frames, "
            f"{record['grasp_face']}/{record['grasp_elbow']} {record['carry_mode']}, "
            f"xy_err={float(record['final_xy_error']):.3f}m, "
            f"yaw_err={math.degrees(float(record['final_yaw_error'])):.1f}°, "
            f"{'success' if ok else 'MISS'} -> {path.name}"
        )

    meta = {
        "joint_names": list(JOINT_NAMES),
        "control_hz": args.control_hz,
        "seed": args.seed,
        "num_requested": args.num_episodes,
        "num_written": written,
        "num_success": successes,
        "success_xy_tolerance": SUCCESS_XY_TOLERANCE,
        "success_z_tolerance": SUCCESS_Z_TOLERANCE,
        "success_yaw_tolerance_deg": math.degrees(SUCCESS_YAW_TOLERANCE),
        "qpos_layout": "JOINT_NAMES (6) then pick_cube free joint (pos[3] + quat[4])",
        "fields": {
            "time": "(T,) trajectory time in seconds",
            "action": "(T,6) joint set points, JOINT_NAMES order, radians",
            "state": "(T,6) measured joint positions, JOINT_NAMES order, radians",
            "qpos": "(T,nq) full sim qpos for exact reconstruction",
            "qvel": "(T,nv) full sim qvel",
            "cube_start/cube_target/cube_end": "(x, y, z, yaw)",
            "robot_start/robot_end": "(6,) JOINT_NAMES order set points",
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"\nWrote {written}/{args.num_episodes} episodes "
        f"({successes} successful) to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
