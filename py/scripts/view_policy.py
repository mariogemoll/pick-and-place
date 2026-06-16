#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Watch a policy drive the arm closed-loop in the MuJoCo viewer.

Samples analytically-feasible episodes (the same sampler the recorder and eval
harness use) and rolls a policy out live, in real time, looping episode after
episode. Pass ``--analytic`` to watch the planner baseline, or ``--checkpoint`` to
watch a trained BC policy — the identical control path
:mod:`pick_and_place.il.rollout` scores headlessly, just rendered.

One persistent model/data backs the viewer; because the cube is a free body, each
new episode is staged by rewriting the arm and cube ``qpos`` rather than rebuilding
the scene (which the passive viewer cannot swap under itself).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.episodes import (
    Episode,
    EpisodeSamplingError,
    is_placed,
    placement_errors,
    prepare_episode,
    set_joint,
)
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.il.bc import BCPolicy, resolve_device
from pick_and_place.il.observations import build_observation, cube_pose_from_qpos
from pick_and_place.il.rollout import (
    SETTLE_SECONDS,
    AnalyticPolicy,
    _joint_dof_adr,
    _joint_qpos_adr,
)

_DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "out" / "bc_policy.pt"


def _build_viewer_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """A standalone scene with the cube as a free body, reused for every episode."""
    spec = build_scene()
    spec.body("pick_cube").add_freejoint()
    model = spec.compile()
    return model, mujoco.MjData(model)


def _cube_adr(model: mujoco.MjModel) -> int:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    return int(model.jnt_qposadr[model.body_jntadr[body]])


def _stage(model: mujoco.MjModel, data: mujoco.MjData, episode: Episode, actuator_id: dict) -> None:
    """Reset the persistent sim to this episode's start: arm pose + cube at source."""
    mujoco.mj_resetData(model, data)
    for name, value in episode.start_joints.items():
        set_joint(model, data, name, value)
        data.ctrl[actuator_id[name]] = value
    set_joint(model, data, "gripper", episode.start_gripper)
    data.ctrl[actuator_id["gripper"]] = episode.start_gripper
    src = episode.source
    adr = _cube_adr(model)
    half_yaw = src.yaw / 2.0
    data.qpos[adr : adr + 3] = (src.x, src.y, src.z)
    data.qpos[adr + 3 : adr + 7] = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    mujoco.mj_forward(model, data)


def _run_episode(viewer, policy, episode, model, data, actuator_id, control_hz, speed,
                 horizon_scale) -> None:
    joint_adr = _joint_qpos_adr(model)
    dof_adr = _joint_dof_adr(model)
    target = np.array([episode.target.x, episode.target.y, episode.target.z, episode.target.yaw])
    control_period = 1.0 / control_hz
    horizon = episode.trajectory.duration * horizon_scale + SETTLE_SECONDS

    _stage(model, data, episode, actuator_id)
    policy.reset()
    viewer.sync()

    last_sample_t = -math.inf
    wall0 = time.time()
    while data.time < horizon and viewer.is_running():
        t = data.time
        if t - last_sample_t >= control_period:
            last_sample_t = t
            joints = data.qpos[joint_adr].copy()
            joint_vel = data.qvel[dof_adr].copy()
            obs = build_observation(joints, joint_vel, cube_pose_from_qpos(data.qpos), target)
            action = policy.act(obs)
            for i, name in enumerate(JOINT_NAMES):
                data.ctrl[actuator_id[name]] = float(action[i])
        mujoco.mj_step(model, data)
        viewer.sync()
        # Pace to wall clock so the motion plays at (scaled) real time.
        remaining = (wall0 + data.time / max(speed, 1e-6)) - time.time()
        if remaining > 0:
            time.sleep(remaining)

    cube_end = cube_pose_from_qpos(data.qpos)
    xy, _z, yaw = placement_errors(cube_end, episode.target)
    placed = is_placed(cube_end, episode.target)
    print(f"  {'PLACED' if placed else 'miss  '}  xy={xy*1000:5.1f}mm  yaw={math.degrees(yaw):5.1f}°")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=Path, default=None, help="BC checkpoint (.pt)")
    group.add_argument("--analytic", action="store_true", help="watch the analytic baseline")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    parser.add_argument(
        "--horizon-scale",
        type=float,
        default=1.0,
        help="run each episode this many times the planned duration (give a stalled policy more time)",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    if args.analytic:
        make_policy = lambda ep: AnalyticPolicy(ep.trajectory, args.control_hz)  # noqa: E731
        label = "analytic baseline"
    else:
        ckpt = args.checkpoint or _DEFAULT_CKPT
        policy = BCPolicy.load(ckpt, device=resolve_device(args.device))
        make_policy = lambda ep: policy  # noqa: E731, ARG005
        label = f"BC checkpoint {ckpt.name}"

    model, data = _build_viewer_model()
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    rng = np.random.default_rng(args.seed)
    print(f"Viewing {label}. Close the window to stop; each episode loops automatically.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        index = 0
        while viewer.is_running():
            try:
                episode = prepare_episode(rng, max_attempts=50)
            except EpisodeSamplingError:
                continue
            index += 1
            print(f"episode {index}:")
            _run_episode(viewer, make_policy(episode), episode, model, data, actuator_id,
                         args.control_hz, args.speed, args.horizon_scale)


if __name__ == "__main__":
    main()
