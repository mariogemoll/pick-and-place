# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Gymnasium environment for the 'approach to hover' phase.

.. deprecated::
    Reference / smoke test only. This is the throwaway position-only "hover
    milestone" that validated the SB3/MuJoCo/PPO loop. It is **not** on the
    frozen 31-dim RL contract (it is 23-dim with delta actions) and is **not**
    part of the curriculum — nothing here transfers to the unified
    ``PickPlaceEnv``. Kept only as a smoke test of the training machinery; the
    living replacement is chapter 1's reach-to-hover sub-step on the frozen
    schema. See ``docs/rl-curriculum-roadmap.md``.

One episode: the arm starts at a near-neutral pose; the goal is to move the EE
tip to the hover point above the cube with the gripper wrist-roll aligned to one
of the four cube faces.  At episode setup the most natural IK-feasible face is
selected (hover IK only — no descent check) and its hover pose becomes the
target for this episode.  Any of the four faces would be geometrically valid
for hovering; we fix one per episode to give the policy an unambiguous target.

Observation (23-dim):
    [0:6]   joint positions (rad), JOINT_NAMES order
    [6:12]  joint velocities (rad/s)
    [12:18] previous commanded action
    [18:21] hover tip target (x, y, z) in world frame
    [21:23] target wrist-roll (cos, sin)

Action (6-dim): delta fractions in [-1, 1].  Each value is multiplied by
MAX_DELTA and added to the previous command, so ±1 means the largest allowed
step and 0 means hold.  The resulting absolute command is what is sent to the
actuators (and to real hardware).

Reward: positive distance shaping (1 at the hover point, decaying with tip
distance) + weighted wrist-roll error, plus a per-step bonus for holding the tip
inside the success shell. Small action-smoothness penalty. Terminal collision
penalty per crash.

The distance term is deliberately non-negative so that surviving a step is never
worse than ending the episode: combined with the collision penalty this removes
the incentive to terminate early (e.g. slamming the arm into the floor) just to
stop accumulating negative reward.  Success is likewise NOT a terminating
condition — the only early termination is a crash.  Otherwise the policy would
tag the nearest point of the success shell, collect a one-off bonus and quit,
parking ~1.5 cm short of the target instead of centring on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, SupportsFloat

import gymnasium as gym
import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.episodes import (
    MIN_START_CLEARANCE,
    build_geom_sets,
    jaw_floor_clearance,
    jaw_geom_ids,
    sample_cube,
    sample_near_neutral,
    scan_contacts,
    set_joint,
)
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import VERTICAL_FACES, CubePose, pregrasp_matrix
from pick_and_place.ik import solve_simple_pregrasp_ik
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics, derive_kinematics
from pick_and_place.trajectory import SOURCE_HOVER_TIP_Z

# Max absolute joint change per 50 Hz control step (~2.5 rad/s upper bound).
MAX_DELTA: float = 0.05

# Reward weights and thresholds.
# Length scale of the exp distance shaping.  Kept comparable to SUCCESS_TIP_DIST
# so the reward still climbs steeply inside the success shell — otherwise the
# centre (tip_dist 0) is barely better than the 1.5 cm boundary and the policy
# parks at the edge of the sphere.
DIST_SCALE: float = 0.02
VERTICAL_WEIGHT: float = 1.0      # penalise non-vertical approach axis [0, 2]
GRIPPER_FLOOR_PENALTY: float = 5.0  # per metre any jaw geom is below the floor
GRIPPER_FLOOR: float = 0.04      # 4 cm — keeps jaws above the 3 cm cube
ACTION_SMOOTH_WEIGHT: float = 0.1
SUCCESS_BONUS: float = 5.0
SUCCESS_TIP_DIST: float = 0.015   # 1.5 cm EE tip distance
# Terminal penalty for a crash.  Must exceed the best return still reachable by
# surviving, otherwise the policy learns to collide on purpose to end the
# episode (collision is a terminating condition).
COLLISION_PENALTY: float = 50.0

CONTROL_HZ: float = 50.0
MAX_STEPS: int = 200              # 4 s at 50 Hz

OBS_DIM: int = 23                 # [21:23] wrist-roll target (cos, sin)
ACT_DIM: int = 6


def _select_hover_wrist_roll(
    k: So101Kinematics, source: CubePose, hover_offset: float
) -> float | None:
    """Return wrist-roll for the most natural hover-IK-feasible face (no descent check)."""
    dx = k.pan_axis[0] - source.x
    dy = k.pan_axis[1] - source.y
    dist = math.hypot(dx, dy)
    c, s = math.cos(source.yaw), math.sin(source.yaw)
    face_normals: dict[str, tuple[float, float]] = {
        "+x": (c, s), "-x": (-c, -s), "+y": (-s, c), "-y": (s, -c)
    }

    def naturalness(face: str) -> float:
        nx, ny = face_normals[face]
        return (nx * dx + ny * dy) / dist if dist > 1e-6 else 0.0

    for face in sorted(VERTICAL_FACES, key=naturalness, reverse=True):
        mat = pregrasp_matrix(face, source, hover_offset)
        if mat is None:
            continue
        branches = solve_simple_pregrasp_ik(k, mat)
        if branches:
            return float(branches[0].joints["wrist_roll"])
    return None


def _build_base_model() -> mujoco.MjModel:
    """Compile the scene once.  Cube position is updated in-place on each reset."""
    return build_scene(wrist_camera=False).compile()


def _joint_addresses(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    qpos = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in JOINT_NAMES
    ]
    dof = [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in JOINT_NAMES
    ]
    return qpos, dof


def _actuator_map(model: mujoco.MjModel) -> dict[str, int]:
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}


class ApproachToHoverEnv(gym.Env):
    """Gymnasium env: move the arm from a near-neutral pose to the hover point.

    The cube pose and start arm pose are randomised each episode.  The hover
    target (tip position + wrist-roll) is derived analytically via IK and held
    fixed for the episode.  Only the hover IK feasibility is checked, not the
    full pick-and-place trajectory.
    """

    metadata = {"render_modes": ["human"], "render_fps": int(CONTROL_HZ)}

    def __init__(
        self,
        render_mode: str | None = None,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        self._key_callback = key_callback
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # [-1, 1] delta fractions; multiplied by MAX_DELTA in step().
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )
        self.render_mode = render_mode

        # Built once per worker process — reused across all episodes.
        self._model: mujoco.MjModel = _build_base_model()
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._k: So101Kinematics = derive_kinematics(self._model)
        self._actuator_id: dict[str, int] = _actuator_map(self._model)
        self._robot_geom_ids: set[int]
        self._env_geom_ids: set[int]
        self._robot_geom_ids, self._env_geom_ids = build_geom_sets(self._model)
        self._qpos_adr: list[int]
        self._dof_adr: list[int]
        self._qpos_adr, self._dof_adr = _joint_addresses(self._model)
        self._jaw_ids: list[int] = jaw_geom_ids(self._model)
        self._cube_body_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube"
        )

        # Episode state — updated in reset().
        self._hover_tip: np.ndarray = np.zeros(3, dtype=np.float32)
        self._wrist_roll_target: float = 0.0
        self._prev_action: np.ndarray = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count: int = 0
        self._viewer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0:6] = self._data.qpos[self._qpos_adr]
        obs[6:12] = self._data.qvel[self._dof_adr]
        obs[12:18] = self._prev_action
        obs[18:21] = self._hover_tip
        obs[21] = math.cos(self._wrist_roll_target)
        obs[22] = math.sin(self._wrist_roll_target)
        return obs

    def _reward_and_info(self, action: np.ndarray) -> tuple[float, dict]:
        arm_joints = {
            n: float(self._data.qpos[self._qpos_adr[i]])
            for i, n in enumerate(JOINT_NAMES)
            if n in ARM_JOINT_NAMES
        }
        tip = np.asarray(self._k.tip_position(arm_joints))
        tip_dist = float(np.linalg.norm(tip - self._hover_tip))

        # Vertical approach: 0 when pointing straight down, up to 2 otherwise.
        # tool_pitch = -π/2 → pointing down; sin(-π/2) = -1 → 1 + (-1) = 0.
        tool_pitch = -(
            arm_joints["shoulder_lift"] + arm_joints["elbow_flex"] + arm_joints["wrist_flex"]
        )
        vertical_deviation = 1.0 + math.sin(tool_pitch)

        # Gripper floor: penalise any jaw geom below 4 cm (keeps jaws above the cube).
        gripper_clearance = jaw_floor_clearance(self._model, self._data, self._jaw_ids)
        gripper_floor_violation = max(0.0, GRIPPER_FLOOR - gripper_clearance)

        smooth = float(np.sum((action - self._prev_action) ** 2))

        contacts = scan_contacts(self._model, self._data, self._robot_geom_ids, self._env_geom_ids)
        n_col = len(contacts)
        collision = n_col > 0
        success = tip_dist < SUCCESS_TIP_DIST

        # Non-negative distance shaping: 1 at the hover point, decaying with
        # tip distance.  Keeps every surviving step worth >= 0 (before the
        # bounded penalties) so the policy can't profit by ending early.
        reach_reward = math.exp(-tip_dist / DIST_SCALE)

        reward = (
            reach_reward
            - VERTICAL_WEIGHT * vertical_deviation
            - GRIPPER_FLOOR_PENALTY * gripper_floor_violation
            - ACTION_SMOOTH_WEIGHT * smooth
        )
        # Per-step bonus for holding the tip inside the success shell.  Success
        # is NOT a terminating condition: ending on first contact with the
        # 1.5 cm boundary would let the policy claim a one-off bonus at the
        # nearest edge and quit, so instead it is paid every step the tip stays
        # inside, rewarding reaching the centre and holding there.
        if success:
            reward += SUCCESS_BONUS
        if collision:
            reward -= COLLISION_PENALTY

        info = {
            "tip_dist": tip_dist,
            "tip_z": float(tip[2]),
            "vertical_deviation": vertical_deviation,
            "gripper_clearance": gripper_clearance,
            "gripper_floor_violation": gripper_floor_violation,
            "n_collisions": n_col,
            "collision": collision,
            "success": success,
        }
        return reward, info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        for _ in range(50):
            source = sample_cube(self.np_random)

            # Reposition the static cube body in-place — no recompile.
            half_yaw = source.yaw / 2.0
            self._model.body_pos[self._cube_body_id] = (source.x, source.y, source.z)
            self._model.body_quat[self._cube_body_id] = (
                math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)
            )

            start_joints, start_gripper = sample_near_neutral(self.np_random)
            mujoco.mj_resetData(self._model, self._data)
            for name, value in start_joints.items():
                set_joint(self._model, self._data, name, value)
                self._data.ctrl[self._actuator_id[name]] = value
            set_joint(self._model, self._data, "gripper", start_gripper)
            self._data.ctrl[self._actuator_id["gripper"]] = start_gripper
            mujoco.mj_forward(self._model, self._data)

            if jaw_floor_clearance(self._model, self._data, self._jaw_ids) < MIN_START_CLEARANCE:
                continue

            # Hover target: directly above cube center at safe height.
            hover_offset = SOURCE_HOVER_TIP_Z - source.z
            wrist_roll = _select_hover_wrist_roll(self._k, source, hover_offset)
            if wrist_roll is None:
                continue

            self._hover_tip = np.array(
                [source.x, source.y, SOURCE_HOVER_TIP_Z], dtype=np.float32
            )
            self._wrist_roll_target = wrist_roll
            self._prev_action = np.array(
                [start_joints[n] if n != "gripper" else start_gripper for n in JOINT_NAMES],
                dtype=np.float32,
            )
            self._step_count = 0

            info = {"source": source, "hover_tip": self._hover_tip.tolist()}
            return self._build_obs(), info

        raise RuntimeError("Could not sample a valid hover episode in 50 attempts")

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        # action ∈ [-1, 1]: delta fraction.  Convert to absolute command.
        action = np.asarray(action, dtype=np.float32)
        clipped = np.clip(self._prev_action + action * MAX_DELTA, -np.pi, np.pi)

        for i, name in enumerate(JOINT_NAMES):
            self._data.ctrl[self._actuator_id[name]] = float(clipped[i])

        target_t = self._data.time + 1.0 / CONTROL_HZ
        while self._data.time < target_t:
            mujoco.mj_step(self._model, self._data)

        reward, info = self._reward_and_info(clipped)
        self._prev_action = clipped
        self._step_count += 1

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        # Only a crash ends the episode early.  Success is held, not terminal
        # (see _reward_and_info), so the policy is rewarded for staying on the
        # hover point rather than tagging the success shell and quitting.
        terminated = info["collision"]
        truncated = self._step_count >= MAX_STEPS
        return self._build_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer  # lazy import — not needed for headless training

            self._viewer = mujoco.viewer.launch_passive(
                self._model, self._data, key_callback=self._key_callback
            )
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
