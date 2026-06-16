# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The shared closed-loop eval harness: run any :class:`Policy` under real physics.

A policy is rolled out on a freshly sampled, analytically-feasible episode (same
``prepare_episode`` the recorder uses, so the pose distribution and start pose
match training). At each 50 Hz control tick the live observation is built with the
*same* :func:`build_observation` used to make the dataset, the policy's action is
written to the actuators, and the sim is stepped to the next tick. After the
planned horizon the cube is left to settle and judged by the shared placement test
(:func:`is_placed`) — exactly the rule the demos were vetted against.

The analytic planner is wrapped as :class:`AnalyticPolicy`, an open-loop policy
that ignores its observation and replays the precomputed trajectory. Running it
through this harness is the harness's own sanity check: it should reproduce the
demos' success rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from pick_and_place.episodes import (
    Episode,
    EpisodeSamplingError,
    is_placed,
    is_unexpected,
    placement_errors,
    prepare_episode,
    scan_contacts,
    set_joint,
)
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.il.observations import build_observation, cube_pose_from_qpos
from pick_and_place.il.policy import Policy
from pick_and_place.trajectory import PickAndCarry

# Extra seconds the sim is stepped (holding the last command) after the planned
# horizon, so a just-released cube can settle before it is measured.
SETTLE_SECONDS = 0.5


class AnalyticPolicy:
    """Open-loop policy replaying a precomputed trajectory on its own clock.

    The analytic planner *is* a policy that ignores its observation. Wrapping it
    here lets the planner run through the identical harness as the learned ones —
    the embodiment of the guide's "every approach yields the same object" claim.
    """

    def __init__(self, trajectory: PickAndCarry, control_hz: float) -> None:
        self._trajectory = trajectory
        self._dt = 1.0 / control_hz
        self._t = 0.0

    def reset(self) -> None:
        self._t = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:  # noqa: ARG002 - open loop
        frame = self._trajectory.evaluate(self._t)
        self._t += self._dt
        return np.array([frame.joints[n] for n in JOINT_NAMES[:-1]] + [frame.gripper])


@dataclass
class RolloutResult:
    success: bool  # placed within tolerance AND no unexpected collisions
    placed: bool  # placed within tolerance (ignores collisions)
    xy_error: float
    z_error: float
    yaw_error: float
    n_collisions: int
    cube_end: np.ndarray


def _joint_qpos_adr(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]


def _joint_dof_adr(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]


def _reset_to_start(episode: Episode) -> None:
    """Return the episode's sim to its recorded start pose (cube at source)."""
    model, data = episode.model, episode.data
    mujoco.mj_resetData(model, data)
    for name, value in episode.start_joints.items():
        set_joint(model, data, name, value)
        data.ctrl[episode.actuator_id[name]] = value
    set_joint(model, data, "gripper", episode.start_gripper)
    data.ctrl[episode.actuator_id["gripper"]] = episode.start_gripper

    src = episode.source
    cube_adr = int(
        model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")]]
    )
    half_yaw = src.yaw / 2.0
    data.qpos[cube_adr : cube_adr + 3] = (src.x, src.y, src.z)
    data.qpos[cube_adr + 3 : cube_adr + 7] = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    mujoco.mj_forward(model, data)


def rollout_policy(
    policy: Policy, episode: Episode, control_hz: float, settle_seconds: float = SETTLE_SECONDS
) -> RolloutResult:
    """Run ``policy`` closed-loop on ``episode`` and judge the placement."""
    model, data = episode.model, episode.data
    actuator_id = episode.actuator_id
    joint_adr = _joint_qpos_adr(model)
    dof_adr = _joint_dof_adr(model)
    target = np.array([episode.target.x, episode.target.y, episode.target.z, episode.target.yaw])
    control_period = 1.0 / control_hz
    horizon = episode.trajectory.duration

    _reset_to_start(episode)
    policy.reset()

    n_collisions = 0
    last_sample_t = -math.inf
    while True:
        t = data.time
        if t - last_sample_t >= control_period:
            last_sample_t = t
            joints = data.qpos[joint_adr].copy()
            joint_vel = data.qvel[dof_adr].copy()
            cube_pose = cube_pose_from_qpos(data.qpos)
            obs = build_observation(joints, joint_vel, cube_pose, target)
            action = policy.act(obs)
            for i, name in enumerate(JOINT_NAMES):
                data.ctrl[actuator_id[name]] = float(action[i])
            for n1, n2 in scan_contacts(model, data, episode.robot_geom_ids, episode.env_geom_ids):
                if is_unexpected(n1, n2):
                    n_collisions += 1
        if t >= horizon:
            break
        mujoco.mj_step(model, data)

    settle_until = data.time + settle_seconds
    while data.time < settle_until:
        mujoco.mj_step(model, data)

    cube_end = cube_pose_from_qpos(data.qpos)
    xy_err, z_err, yaw_err = placement_errors(cube_end, episode.target)
    placed = is_placed(cube_end, episode.target)
    return RolloutResult(
        success=placed and n_collisions == 0,
        placed=placed,
        xy_error=xy_err,
        z_error=z_err,
        yaw_error=yaw_err,
        n_collisions=n_collisions,
        cube_end=cube_end,
    )


def evaluate(
    make_policy: Callable[[Episode], Policy],
    *,
    n_episodes: int,
    seed: int,
    control_hz: float = 50.0,
    max_attempts: int = 50,
    verbose: bool = False,
) -> list[RolloutResult]:
    """Sample ``n_episodes`` feasible episodes and roll out a policy on each.

    ``make_policy`` is given the prepared episode and returns the policy to run —
    a constant learned policy ignores it; :class:`AnalyticPolicy` needs the
    episode's own trajectory. Episodes the sampler cannot solve are skipped.
    """
    rng = np.random.default_rng(seed)
    results: list[RolloutResult] = []
    index = 0
    while len(results) < n_episodes:
        index += 1
        try:
            episode = prepare_episode(rng, max_attempts=max_attempts)
        except EpisodeSamplingError as exc:
            if verbose:
                print(f"episode {index}: skipped ({exc})")
            continue
        result = rollout_policy(make_policy(episode), episode, control_hz)
        results.append(result)
        if verbose:
            tag = "OK " if result.success else ("placed" if result.placed else "MISS")
            print(
                f"episode {len(results):3d}/{n_episodes}: {tag} "
                f"xy={result.xy_error:.3f}m yaw={math.degrees(result.yaw_error):4.1f}° "
                f"col={result.n_collisions}"
            )
    return results
