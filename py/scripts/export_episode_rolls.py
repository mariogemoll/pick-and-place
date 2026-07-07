#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export sim pick-and-place episodes as ``.bin`` files for the web viewer.

Each output file holds one episode's per-frame ``qpos`` — the 6 arm/gripper
joint angles (radians) followed by the cube's free-joint pose (pos[3] +
quat[4]), sampled from a fresh physics rollout (see ``pick_and_place.episodes``)
at a fixed frame rate. That is the complete, minimal state needed to replay
the episode: the browser viewer drives the existing joint-hierarchy robot
model directly from these values, the same way ``robot.ts``/``grasp-and-lift.ts``
already do for a single static or procedural pose — no additional forward
kinematics format or geometry table is needed.

Only episodes where the cube actually lands on target (no unexpected contacts,
same definition as ``record_episodes.py``) are kept; failures are silently
retried with the next seed. The default 5 grip positions are hand-picked to
span the pickup sector's radius (from near the base to its outer edge) at
alternating azimuths, with the last one deliberately the hardest combination
reachable — near-max radius *and* near-max azimuth for that radius.

Binary layout ("PPRL" format, little-endian)::

    magic    4 bytes   b"PPRL"
    version  u32       1
    fps      u32       sampling rate the frames were recorded at
    nframes  u32       number of frames
    nq       u32       floats per frame (6 joints + 7 cube pose = 13)
    target_x f32       drop target position on the floor (meters)
    target_y f32
    qpos     f32[nframes * nq]

Usage::

    python -m scripts.export_episode_rolls -n 5 --fps 60 \\
        --out-dir ../ts/public/episodes
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.episodes import (
    EpisodeSamplingError,
    is_unexpected,
    pickup_yaw_from_azimuth,
    prepare_episode,
    scan_contacts,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.workspace_overlays import PAN_AXIS

from scripts.record_episodes import SUCCESS_XY_TOLERANCE, SUCCESS_Z_TOLERANCE

MAGIC = b"PPRL"
VERSION = 1
DEFAULT_FPS = 60.0

# (radius, azimuth_deg) around PAN_AXIS for each demo episode's grip position,
# spanning the pickup sector from near the base to its outer edge; the last
# entry is the hardest reachable combination (near-max radius and azimuth).
DEFAULT_GRIP_PRESETS = (
    (0.13, -85.0),
    (0.20, 55.0),
    (0.27, -45.0),
    (0.34, 40.0),
    (0.42, -30.0),
)


def source_from_preset(radius: float, azimuth_deg: float) -> CubePose:
    azimuth = math.radians(azimuth_deg)
    return CubePose(
        x=PAN_AXIS[0] + radius * math.cos(azimuth),
        y=PAN_AXIS[1] + radius * math.sin(azimuth),
        z=CUBE_HALF_SIZE,
        yaw=pickup_yaw_from_azimuth(azimuth),
    )


def run_and_sample(
    rng: np.random.Generator,
    fps: float,
    max_attempts: int,
    source: CubePose | None,
) -> tuple[np.ndarray, float, float, bool]:
    """Prepare one episode and sample its physics rollout's ``qpos`` at ``fps``.

    Returns the sampled frames, the episode's drop target (x, y), and whether
    the cube actually landed on target with no unexpected contacts (the same
    success definition ``record_episodes.py`` uses).
    """
    episode = prepare_episode(rng, source=source, max_attempts=max_attempts)
    model, data = episode.model, episode.data
    trajectory = episode.trajectory
    actuator_id = episode.actuator_id
    period = 1.0 / fps

    frames: list[np.ndarray] = []
    collisions = 0
    last_sample_t = -np.inf
    while True:
        traj_t = data.time
        frame = trajectory.evaluate(traj_t)
        for name, value in frame.joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = frame.gripper

        if traj_t - last_sample_t >= period:
            last_sample_t = traj_t
            frames.append(data.qpos.copy())
            for n1, n2 in scan_contacts(model, data, episode.robot_geom_ids, episode.env_geom_ids):
                if is_unexpected(n1, n2):
                    collisions += 1

        if traj_t >= trajectory.duration:
            break
        mujoco.mj_step(model, data)

    qpos = np.stack(frames).astype(np.float32)

    cube_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    cube_qpos_adr = int(model.jnt_qposadr[model.body_jntadr[cube_adr]])
    cube_end = data.qpos[cube_qpos_adr : cube_qpos_adr + 3]
    xy_err = math.hypot(cube_end[0] - episode.target.x, cube_end[1] - episode.target.y)
    z_err = abs(cube_end[2] - CUBE_HALF_SIZE)
    success = xy_err <= SUCCESS_XY_TOLERANCE and z_err <= SUCCESS_Z_TOLERANCE and collisions == 0

    return qpos, float(episode.target.x), float(episode.target.y), success


def write_episode(
    qpos: np.ndarray, fps: float, target_x: float, target_y: float, out_path: Path
) -> None:
    nframes, nq = qpos.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<IIII", VERSION, round(fps), nframes, nq))
        file.write(struct.pack("<ff", target_x, target_y))
        file.write(qpos.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-episodes", type=int, default=len(DEFAULT_GRIP_PRESETS))
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument(
        "--max-attempts", type=int, default=50, help="pose resamples per episode"
    )
    parser.add_argument(
        "--max-seed-retries",
        type=int,
        default=30,
        help="fresh seeds to try per grip position before giving up on it",
    )
    parser.add_argument(
        "--random-grip",
        action="store_true",
        help="sample grip positions randomly instead of using the spanning presets "
        "(presets only cover --num-episodes <= 5; use this for more)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ts" / "public" / "episodes",
        help="directory to write episode_NN.bin files (default: ts/public/episodes)",
    )
    args = parser.parse_args()

    if not args.random_grip and args.num_episodes > len(DEFAULT_GRIP_PRESETS):
        parser.error(
            f"--num-episodes {args.num_episodes} exceeds the {len(DEFAULT_GRIP_PRESETS)} "
            "built-in grip presets; pass --random-grip for more"
        )

    written = 0
    seed = args.seed
    while written < args.num_episodes:
        source = (
            None if args.random_grip
            else source_from_preset(*DEFAULT_GRIP_PRESETS[written])
        )

        episode_result = None
        for _ in range(args.max_seed_retries):
            rng = np.random.default_rng(seed)
            try:
                result = run_and_sample(rng, args.fps, args.max_attempts, source)
            except EpisodeSamplingError as exc:
                print(f"seed {seed}: skipped ({exc})")
                seed += 1
                continue
            seed += 1
            if result[3]:
                episode_result = result
                break
            print(f"seed {seed - 1}: skipped (cube missed target or collided)")

        if episode_result is None:
            raise RuntimeError(
                f"No successful episode for grip preset {written} within "
                f"{args.max_seed_retries} seeds"
            )

        qpos, target_x, target_y, _ = episode_result
        out_path = args.out_dir / f"episode_{written:02d}.bin"
        write_episode(qpos, args.fps, target_x, target_y, out_path)
        print(f"{qpos.shape[0]} frames -> {out_path}")
        written += 1

    print(f"\nWrote {written} episodes to {args.out_dir}")


if __name__ == "__main__":
    main()
