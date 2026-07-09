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

Pass ``--continuous-chain`` to make each episode start from the previous
episode's final cube pose while still sampling new random targets. Add
``--closed-loop`` to make the last episode place the cube back at the first
source.

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

    python -m scripts.export_episode_rolls -n 5 --fps 60 \\
        --continuous-chain --closed-loop --out-dir ../ts/public/episodes
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.episodes import (
    EpisodeSamplingError,
    is_unexpected,
    pickup_yaw_from_azimuth,
    prepare_episode,
    sample_cube,
    sample_near_neutral,
    scan_contacts,
)
from pick_and_place.follower import ARM_JOINT_NAMES
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.workspace_overlays import (
    CUBE_PLACEMENT_BOUNDS,
    PAN_AXIS,
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
)

from scripts.record_episodes import SUCCESS_XY_TOLERANCE, SUCCESS_Z_TOLERANCE

MAGIC = b"PPRL"
VERSION = 1
DEFAULT_FPS = 60.0
DEFAULT_END_SETTLE_SECONDS = 1.0

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


def pose_from_qpos(qpos: np.ndarray, cube_qpos_adr: int) -> CubePose:
    x, y, z = (float(v) for v in qpos[cube_qpos_adr : cube_qpos_adr + 3])
    qw, qx, qy, qz = (float(v) for v in qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 0.0:
        qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch_sin = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, pitch_sin)))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return CubePose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)


def sample_chain_pose(rng: np.random.Generator) -> CubePose:
    """Sample a pose that can be both a drop target and the next pickup source."""
    while True:
        pose = sample_cube(rng)
        if is_cube_drop_allowed(pose.x, pose.y):
            return pose


def robot_pose_from_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
) -> tuple[dict[str, float], float]:
    joints = {}
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joints[name] = float(qpos[model.jnt_qposadr[joint_id]])
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper = float(qpos[model.jnt_qposadr[gripper_id]])
    return joints, gripper


def run_and_sample(
    rng: np.random.Generator,
    fps: float,
    max_attempts: int,
    end_settle_seconds: float,
    source: CubePose | None,
    target: CubePose | None = None,
    start_joints: dict[str, float] | None = None,
    start_gripper: float | None = None,
    end_joints: dict[str, float] | None = None,
    end_gripper: float | None = None,
) -> tuple[np.ndarray, float, float, bool, CubePose, CubePose, dict[str, float], float]:
    """Prepare one episode and sample its physics rollout's ``qpos`` at ``fps``.

    Returns the sampled frames, the episode's drop target (x, y), and whether
    the cube actually landed on target with no unexpected contacts (the same
    success definition ``record_episodes.py`` uses).
    """
    episode = prepare_episode(
        rng,
        source=source,
        target=target,
        start_joints=start_joints,
        start_gripper=start_gripper,
        end_joints=end_joints,
        end_gripper=end_gripper,
        max_attempts=max_attempts,
    )
    model, data = episode.model, episode.data
    trajectory = episode.trajectory
    actuator_id = episode.actuator_id
    period = 1.0 / fps

    frames: list[np.ndarray] = []
    collisions = 0
    last_sample_t = -np.inf

    def apply_frame_controls(frame) -> None:
        for name, value in frame.joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = frame.gripper

    def sample_if_due() -> None:
        nonlocal collisions, last_sample_t
        if data.time - last_sample_t < period:
            return
        last_sample_t = data.time
        frames.append(data.qpos.copy())
        for n1, n2 in scan_contacts(model, data, episode.robot_geom_ids, episode.env_geom_ids):
            if is_unexpected(n1, n2):
                collisions += 1

    while True:
        traj_t = data.time
        frame = trajectory.evaluate(traj_t)
        apply_frame_controls(frame)
        sample_if_due()

        if traj_t >= trajectory.duration:
            break
        mujoco.mj_step(model, data)

    final_frame = trajectory.evaluate(trajectory.duration)
    settle_until = data.time + end_settle_seconds
    while data.time < settle_until:
        apply_frame_controls(final_frame)
        mujoco.mj_step(model, data)
        sample_if_due()

    qpos = np.stack(frames).astype(np.float32)

    cube_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    cube_qpos_adr = int(model.jnt_qposadr[model.body_jntadr[cube_adr]])
    cube_start_pose = pose_from_qpos(qpos[0], cube_qpos_adr)
    cube_end = data.qpos[cube_qpos_adr : cube_qpos_adr + 3]
    xy_err = math.hypot(cube_end[0] - episode.target.x, cube_end[1] - episode.target.y)
    z_err = abs(cube_end[2] - CUBE_HALF_SIZE)
    success = xy_err <= SUCCESS_XY_TOLERANCE and z_err <= SUCCESS_Z_TOLERANCE and collisions == 0
    cube_final_pose = pose_from_qpos(qpos[-1], cube_qpos_adr)
    robot_final_joints, robot_final_gripper = robot_pose_from_qpos(model, qpos[-1])

    return (
        qpos,
        float(episode.target.x),
        float(episode.target.y),
        success,
        cube_start_pose,
        cube_final_pose,
        robot_final_joints,
        robot_final_gripper,
    )


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


def _allowed_mask(predicate, resolution: int = 320) -> tuple[np.ndarray, list[float]]:
    x_min, x_max, y_min, y_max = CUBE_PLACEMENT_BOUNDS
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    mask = np.array([[predicate(float(x), float(y)) for x in xs] for y in ys], dtype=float)
    return mask, [x_min, x_max, y_min, y_max]


def write_layout_png(
    episodes: list[tuple[CubePose, CubePose, CubePose]],
    out_path: Path,
) -> None:
    """Write a top-down PNG of exported source/target/final cube positions."""
    cache_dir = Path(tempfile.gettempdir()) / "pick_place_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = CUBE_PLACEMENT_BOUNDS
    margin = 0.035
    extent = [
        bounds[0] - margin,
        bounds[1] + margin,
        bounds[2] - margin,
        bounds[3] + margin,
    ]
    drop_mask, mask_extent = _allowed_mask(is_cube_drop_allowed)
    pickup_mask, _ = _allowed_mask(is_cube_pickup_allowed)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(
        drop_mask,
        extent=mask_extent,
        origin="lower",
        cmap="Greens",
        alpha=0.16,
        interpolation="nearest",
    )
    ax.imshow(
        pickup_mask,
        extent=mask_extent,
        origin="lower",
        cmap="Blues",
        alpha=0.16,
        interpolation="nearest",
    )
    ax.add_patch(
        plt.Rectangle(
            (bounds[0], bounds[2]),
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            fill=False,
            edgecolor="0.35",
            linewidth=1.0,
        )
    )
    ax.scatter([PAN_AXIS[0]], [PAN_AXIS[1]], marker="+", s=80, color="0.2", label="pan axis")

    for index, (source, target, final) in enumerate(episodes):
        color = plt.cm.tab10(index % 10)
        ax.annotate(
            "",
            xy=(target.x, target.y),
            xytext=(source.x, source.y),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.6,
                "shrinkA": 4,
                "shrinkB": 4,
            },
        )
        ax.scatter([source.x], [source.y], s=44, color=color, marker="o")
        ax.scatter([target.x], [target.y], s=58, color=color, marker="x", linewidths=2.0)
        ax.scatter([final.x], [final.y], s=36, facecolors="none", edgecolors=color, marker="s")
        ax.text(source.x, source.y, f" {index}", color=color, fontsize=8, va="center")
        ax.text(target.x, target.y, f" T{index}", color=color, fontsize=8, va="center")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.82", linewidth=0.6)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Exported episode cube positions")
    ax.plot([], [], marker="o", color="0.2", linestyle="None", label="source")
    ax.plot([], [], marker="x", color="0.2", linestyle="None", label="target")
    ax.plot([], [], marker="s", markerfacecolor="none", color="0.2", linestyle="None", label="final")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def chain_pose(index: int) -> CubePose:
    return source_from_preset(*DEFAULT_GRIP_PRESETS[index])


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
        "--end-settle-seconds",
        type=float,
        default=DEFAULT_END_SETTLE_SECONDS,
        help=f"seconds to keep recording with final controls held (default {DEFAULT_END_SETTLE_SECONDS:g})",
    )
    parser.add_argument(
        "--random-grip",
        action="store_true",
        help="sample grip positions randomly instead of using the spanning presets "
        "(presets only cover --num-episodes <= 5; use this for more)",
    )
    parser.add_argument(
        "--continuous-chain",
        action="store_true",
        help="sample random targets while starting each episode from the previous "
        "episode's final cube pose",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="with --continuous-chain, set the last episode target to the first "
        "sampled source so the replay loops continuously",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ts" / "public" / "episodes",
        help="directory to write episode_NN.bin files (default: ts/public/episodes)",
    )
    parser.add_argument(
        "--layout-png",
        type=Path,
        default=None,
        help="top-down PNG of source/target/final cube dots (default: OUT_DIR/episode_layout.png)",
    )
    parser.add_argument(
        "--no-layout-png",
        action="store_true",
        help="do not write the top-down source/target/final cube dot PNG",
    )
    args = parser.parse_args()

    if args.continuous_chain and args.random_grip:
        parser.error("--continuous-chain already samples random chain poses; do not combine it with --random-grip")
    if args.end_settle_seconds < 0:
        parser.error("--end-settle-seconds must be non-negative")
    if args.closed_loop and not args.continuous_chain:
        parser.error("--closed-loop requires --continuous-chain")
    if not args.random_grip and not args.continuous_chain and args.num_episodes > len(DEFAULT_GRIP_PRESETS):
        parser.error(
            f"--num-episodes {args.num_episodes} exceeds the {len(DEFAULT_GRIP_PRESETS)} "
            "built-in grip presets; pass --random-grip for more"
        )

    written = 0
    seed = args.seed
    layout_episodes: list[tuple[CubePose, CubePose, CubePose]] = []
    first_source: CubePose | None = None
    first_start_joints: dict[str, float] | None = None
    first_start_gripper: float | None = None
    next_source: CubePose | None = None
    next_start_joints: dict[str, float] | None = None
    next_start_gripper: float | None = None
    if args.continuous_chain:
        first_source = sample_chain_pose(
            np.random.default_rng(np.random.SeedSequence([args.seed, 0x5150]))
        )
        next_source = first_source
        first_start_joints, first_start_gripper = sample_near_neutral(
            np.random.default_rng(np.random.SeedSequence([args.seed, 0xC0FFEE]))
        )
        next_start_joints = dict(first_start_joints)
        next_start_gripper = first_start_gripper

    while written < args.num_episodes:
        source = (
            next_source
            if args.continuous_chain
            else (None if args.random_grip else chain_pose(written))
        )
        target = None
        if args.continuous_chain:
            target = first_source if args.closed_loop and written == args.num_episodes - 1 else None
        end_joints = (
            first_start_joints
            if args.continuous_chain and args.closed_loop and written == args.num_episodes - 1
            else None
        )
        end_gripper = (
            first_start_gripper
            if args.continuous_chain and args.closed_loop and written == args.num_episodes - 1
            else None
        )

        episode_result = None
        for _ in range(args.max_seed_retries):
            rng = np.random.default_rng(seed)
            episode_target = target
            if args.continuous_chain and episode_target is None:
                episode_target = sample_chain_pose(rng)
            try:
                result = run_and_sample(
                    rng,
                    args.fps,
                    args.max_attempts,
                    args.end_settle_seconds,
                    source,
                    episode_target,
                    next_start_joints,
                    next_start_gripper,
                    end_joints,
                    end_gripper,
                )
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
                f"No successful episode {written} within "
                f"{args.max_seed_retries} seeds"
            )

        (
            qpos,
            target_x,
            target_y,
            _,
            cube_start_pose,
            cube_final_pose,
            next_end_joints,
            next_end_gripper,
        ) = episode_result
        out_path = args.out_dir / f"episode_{written:02d}.bin"
        write_episode(qpos, args.fps, target_x, target_y, out_path)
        layout_episodes.append((
            cube_start_pose,
            CubePose(x=target_x, y=target_y, z=CUBE_HALF_SIZE),
            cube_final_pose,
        ))
        chain_note = "" if not args.continuous_chain else f" target=({target_x:.3f}, {target_y:.3f})"
        print(f"{qpos.shape[0]} frames -> {out_path}{chain_note}")
        if args.continuous_chain:
            next_source = cube_final_pose
            next_start_joints = next_end_joints
            next_start_gripper = next_end_gripper
        written += 1

    if not args.no_layout_png:
        layout_path = args.layout_png or (args.out_dir / "episode_layout.png")
        write_layout_png(layout_episodes, layout_path)
        print(f"Wrote layout plot -> {layout_path}")

    print(f"\nWrote {written} episodes to {args.out_dir}")


if __name__ == "__main__":
    main()
