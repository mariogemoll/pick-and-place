#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""View the SO-101 pick-and-place trajectory in the sim under real physics.

The arm is controlled through the model's position-servo actuators: each frame
the trajectory's joint set points are written to ``data.ctrl`` and the simulation
is stepped, so gravity and contact are live. The cube gets a free joint and rests
on the floor as a genuine rigid body. Unexpected collisions are flagged as they
happen; the viewer loops the trajectory until closed.

Phases: (1) neutral -> hover, (2) hover -> grasp at cube center, (3) grasp,
(4) lift and carry the grasped cube over to the hover above the target,
(5) release, lift clear, and flow back to neutral.

This is sim-only. To run on the physical SO-101 follower, use
``real.py`` (``pick_and_place.executor``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place.episodes import (
    Episode,
    EpisodeSamplingError,
    is_unexpected,
    prepare_episode,
    scan_contacts,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose


def _play(episode: Episode, speed: float) -> None:
    """Loop the trajectory in a passive viewer, flagging unexpected collisions.

    The sim steps in real time; the trajectory clock runs at ``speed`` × wall time,
    so a factor below 1.0 slows every phase uniformly for closer inspection.
    """
    model = episode.model
    data = episode.data
    actuator_id = episode.actuator_id
    robot_geom_ids = episode.robot_geom_ids
    env_geom_ids = episode.env_geom_ids
    trajectory = episode.trajectory

    prev_contacts: set[tuple[str, str]] = set()
    with mujoco.viewer.launch_passive(model, data) as viewer:
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

            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


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
        "--drop-orientation",
        choices=("free", "target"),
        default="free",
        help="free searches any reachable drop orientation; target preserves target yaw",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier of the nominal trajectory pace (1.0 = nominal)",
    )
    parser.add_argument(
        "--environment",
        action="store_true",
        help="include the calibration workspace_frame and overhead camera mount in the scene",
    )
    parser.add_argument(
        "--preflight-debug",
        action="store_true",
        help="print detailed collision diagnostics for rejected trajectory candidates",
    )
    parser.add_argument(
        "--preflight-debug-limit",
        type=int,
        default=12,
        help="maximum detailed contact rows to print per rejected candidate",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="prepare and preflight the episode, then exit without opening the viewer",
    )
    parser.add_argument(
        "--save-failed-trajectories",
        type=Path,
        default=None,
        help="directory for replayable .npz rollouts of rejected preflight candidates",
    )
    parser.add_argument(
        "--failed-trajectory-limit",
        type=int,
        default=8,
        help="maximum rejected candidates to save",
    )
    args = parser.parse_args()

    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")

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

    try:
        episode = prepare_episode(
            np.random.default_rng(),
            source,
            target,
            verbose=True,
            include_environment=args.environment,
            drop_orientation=args.drop_orientation,
            preflight_debug=args.preflight_debug,
            preflight_debug_limit=args.preflight_debug_limit,
            failed_trajectory_dir=args.save_failed_trajectories,
            failed_trajectory_limit=args.failed_trajectory_limit,
        )
    except EpisodeSamplingError as exc:
        raise SystemExit(str(exc)) from exc

    if args.plan_only:
        print(
            f"planned source=({episode.source.x:.3f}, {episode.source.y:.3f}) "
            f"target=({episode.target.x:.3f}, {episode.target.y:.3f}) "
            f"grasp={episode.grasp.face}/{episode.grasp.elbow} "
            f"carry={episode.trajectory.carry.mode} attempts={episode.attempts}"
        )
        return

    _play(episode, args.speed)


if __name__ == "__main__":
    main()
