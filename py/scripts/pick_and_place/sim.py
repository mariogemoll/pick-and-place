#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""View SO-101 pick-and-place episodes in the sim under real physics.

The arm is controlled through the model's position-servo actuators: each frame
the trajectory's joint set points are written to ``data.ctrl`` and the simulation
is stepped, so gravity and contact are live. The cube gets a free joint and rests
on the floor as a genuine rigid body, and a square marks the drop target on the
floor. Unexpected collisions are flagged as they happen.

By default the viewer loops forever, planning a fresh episode each time the
previous one finishes; ``--episodes N`` stops after N. Press Enter in the
terminal to skip to the next episode immediately, or close the viewer to stop.

Phases: (1) neutral -> hover, (2) hover -> grasp at cube center, (3) grasp,
(4) lift and carry the grasped cube over to the hover above the target,
(5) release, lift clear, and flow back to neutral.

This is sim-only. To run on the physical SO-101 follower, use
``real.py`` (``pick_and_place.executor``).
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place.episodes import (
    Episode,
    EpisodeSamplingError,
    _build_model,
    is_unexpected,
    placement_error,
    prepare_episode,
    sample_target,
    scan_contacts,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    place_paper_target_marker,
)
from pick_and_place.workspace_overlays import PAN_AXIS, is_cube_drop_allowed

# How long the final pose is held after a trajectory finishes, before the next
# episode is planned, so the placed cube is visible for a beat.
END_DWELL = 0.8

# GLFW key codes the viewer's key_callback reports for the Return / keypad-Enter
# keys, used to skip to the next episode from the viewer window.
_ENTER_KEYS = frozenset({257, 335})


def _watch_for_skip(skip_event: threading.Event) -> None:
    """Set ``skip_event`` whenever the user presses Enter in the terminal."""
    for _ in iter(sys.stdin.readline, ""):
        skip_event.set()


def _plate_corners_allowed(cx: float, cy: float, yaw: float, half_size: float) -> bool:
    """Whether every corner of a ``yaw``-rotated square plate centered at
    ``(cx, cy)`` still lands inside the allowed drop zone."""
    c, s = math.cos(yaw), math.sin(yaw)
    for lx, ly in ((half_size, half_size), (half_size, -half_size), (-half_size, half_size), (-half_size, -half_size)):
        if not is_cube_drop_allowed(cx + lx * c - ly * s, cy + lx * s + ly * c):
            return False
    return True


def _sample_marker_yaw(rng: np.random.Generator, cx: float, cy: float) -> float:
    """Sample a marker yaw in [0, 90) degrees whose plate corners stay in bounds.

    The plate is square, so any yaw outside [0, 90) is equivalent to one inside
    it. Falls back to yaw 0 (axis-aligned, the smallest possible footprint) if
    no sampled yaw fits after enough tries. Used for a CLI-pinned ``--target``,
    where the position itself cannot be resampled.
    """
    for _ in range(200):
        yaw = rng.uniform(0.0, math.pi / 2.0)
        if _plate_corners_allowed(cx, cy, yaw, DROP_ZONE_HALF_SIZE):
            return yaw
    return 0.0


def _sample_target_plate(rng: np.random.Generator, max_attempts: int = 200) -> tuple[CubePose, float]:
    """Jointly sample a target position and marker yaw whose plate footprint
    fully fits the allowed drop zone; the plate's center is the placement
    target. Falls back to a freshly sampled position at yaw 0 (the smallest
    possible footprint) if nothing fits within ``max_attempts``.
    """
    for _ in range(max_attempts):
        candidate = sample_target(rng)
        yaw = rng.uniform(0.0, math.pi / 2.0)
        if _plate_corners_allowed(candidate.x, candidate.y, yaw, DROP_ZONE_HALF_SIZE):
            return candidate, yaw
    return sample_target(rng), 0.0


class _MarkerTargetSampler:
    """``prepare_episode`` target sampler that also records the marker yaw
    chosen for the most recently sampled target, so the caller can render a
    fully-fitting plate around the same point once planning succeeds."""

    def __init__(self) -> None:
        self.yaw = 0.0

    def __call__(self, rng: np.random.Generator) -> CubePose:
        target, self.yaw = _sample_target_plate(rng)
        return target


def _show_target_marker(model: mujoco.MjModel, target: CubePose, yaw: float) -> None:
    """Place the drop-zone square on the floor at the episode target."""
    place_paper_target_marker(
        model,
        (target.x, target.y),
        yaw,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(target.x, target.y),
    )


def _play(
    episode: Episode,
    speed: float,
    viewer: mujoco.viewer.Handle,
    skip_event: threading.Event,
) -> bool:
    """Play one episode's trajectory once, flagging unexpected collisions.

    The sim steps in real time; the trajectory clock runs at ``speed`` × wall time,
    so a factor below 1.0 slows every phase uniformly for closer inspection.
    Returns whether the trajectory finished normally after the final dwell.
    """
    model = episode.model
    data = episode.data
    actuator_id = episode.actuator_id
    robot_geom_ids = episode.robot_geom_ids
    env_geom_ids = episode.env_geom_ids
    trajectory = episode.trajectory

    prev_contacts: set[tuple[str, str]] = set()
    playback_start = data.time
    while viewer.is_running():
        if skip_event.is_set():
            skip_event.clear()
            return False
        step_start = time.time()
        traj_t = (data.time - playback_start) * speed
        if traj_t > trajectory.duration + END_DWELL:
            return True
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
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="number of episodes to play; 0 means loop until the viewer is closed (default: 0)",
    )
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="source cube (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--source-yaw",
        type=float,
        default=0.0,
        help="source cube yaw in degrees, only used with --source (default: 0.0)",
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
        help="prepare and preflight a single episode, then exit without opening the viewer",
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
    if args.episodes < 0:
        raise ValueError("--episodes must be non-negative")

    source = (
        CubePose(
            x=args.source[0],
            y=args.source[1],
            z=CUBE_HALF_SIZE,
            yaw=math.radians(args.source_yaw),
        )
        if args.source is not None
        else None
    )
    target = (
        CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
        if args.target is not None
        else None
    )

    rng = np.random.default_rng()

    # One persistent scene for the whole run: the cube is a freejoint that
    # prepare_episode repositions per episode, so a single viewer stays bound.
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(
        dummy_source,
        include_environment=args.environment,
        paper_target_marker=True,
    )
    mujoco.mj_forward(model, data)

    marker_sampler = _MarkerTargetSampler()

    def plan(verbose: bool) -> Episode:
        return prepare_episode(
            rng,
            source,
            target,
            model=model,
            data=data,
            verbose=verbose,
            include_environment=args.environment,
            preflight_debug=args.preflight_debug,
            preflight_debug_limit=args.preflight_debug_limit,
            failed_trajectory_dir=args.save_failed_trajectories,
            failed_trajectory_limit=args.failed_trajectory_limit,
            target_sampler=marker_sampler if target is None else None,
        )

    if args.plan_only:
        try:
            episode = plan(verbose=True)
        except EpisodeSamplingError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"planned source=({episode.source.x:.3f}, {episode.source.y:.3f}) "
            f"target=({episode.target.x:.3f}, {episode.target.y:.3f}) "
            f"grasp={episode.grasp.face}/{episode.grasp.elbow} "
            f"carry={episode.trajectory.carry.mode} attempts={episode.attempts}"
        )
        return

    skip_event = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(target=_watch_for_skip, args=(skip_event,), daemon=True).start()
    print("Press Enter (in the viewer or the terminal) to skip to the next episode; "
          "close the viewer to stop.")

    def on_key(keycode: int) -> None:
        if keycode in _ENTER_KEYS:
            skip_event.set()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        index = 0
        while viewer.is_running():
            if args.episodes and index >= args.episodes:
                break
            index += 1
            print(
                f"\n--- Episode {index}"
                f"{f'/{args.episodes}' if args.episodes else ''} ---"
            )
            try:
                episode = plan(verbose=True)
            except EpisodeSamplingError as exc:
                print(str(exc))
                # A pinned source/target with no feasible plan fails every time,
                # so there is nothing to retry — stop rather than spin.
                break
            marker_yaw = (
                _sample_marker_yaw(rng, episode.target.x, episode.target.y)
                if target is not None
                else marker_sampler.yaw
            )
            _show_target_marker(model, episode.target, marker_yaw)
            skip_event.clear()
            if _play(episode, args.speed, viewer, skip_event):
                print(placement_error(model, data, episode.target).summary())


if __name__ == "__main__":
    main()
