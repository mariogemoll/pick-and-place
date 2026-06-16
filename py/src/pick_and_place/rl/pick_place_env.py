# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Unified pick-and-place Gymnasium environment (curriculum skeleton).

This is build-order step 2 of ``docs/rl-curriculum-roadmap.md``: the single env
the whole curriculum (position -> orientation -> regrasp -> robustness) finetunes
on. Every chapter is a **reward-weight change on this same env**, never a fork and
never an observation-dimension change — the I/O is the frozen 31-dim contract in
:mod:`pick_and_place.rl.contract`.

What the skeleton provides
--------------------------
* **Reset** samples a source *and* a target cube pose (:func:`episodes.sample_cube`)
  and a near-neutral start arm pose (:func:`episodes.sample_near_neutral`), with the
  cube as a real **dynamic free body** so it can be grasped, carried and dropped.
* **Observation** is the frozen 31-dim contract (confidence held at ``1.0`` for the
  pure-sim chapters); the current cube pose is read straight from the simulated free
  body (the privileged state the sim-to-real plan later replaces with an estimator).
* **Action** is an **absolute** 6-vector joint set point; the per-step change is
  clamped by :func:`contract.clamp_setpoint` and then clipped to the model's joint
  limits (limit clipping is the env's job — it owns the model).
* **Reward** is a weighted sum of **composable, named terms** (:func:`reward_terms`).
  A chapter is a different ``reward_weights`` dict — e.g. chapter 1 (position) leaves
  ``yaw`` at weight 0; chapter 2 (orientation) turns it on. The terms are computed
  from a plain :class:`RewardMetrics` snapshot so they are unit-testable without
  physics, and so the YAML curriculum runner (step 4) can drive them from config.

The reward weights here are deliberately *placeholder* magnitudes (mirroring the
hover milestone) — the actual chapter-1 reward shaping / staging is build-order
step 5, not this skeleton. Grasp / contact / success scaffolding is wired up; the
tuning is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.episodes import (
    MIN_START_CLEARANCE,
    SUCCESS_XY_TOLERANCE,
    SUCCESS_YAW_TOLERANCE,
    SUCCESS_Z_TOLERANCE,
    build_geom_sets,
    is_unexpected,
    jaw_floor_clearance,
    jaw_geom_ids,
    placement_errors,
    quat_yaw,
    sample_cube,
    sample_near_neutral,
    scan_contacts,
    set_joint,
)
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics, derive_kinematics
from pick_and_place.rl import contract

CONTROL_HZ: float = 50.0
# Pick-place is a longer task than the hover milestone (reach -> grasp -> carry ->
# place), so give it more room: 8 s at 50 Hz.
MAX_STEPS: int = 400

# Reward shaping length scales (metres / radians). Placeholders mirroring the
# hover milestone; chapter-1 tuning (build-order step 5) owns the real values.
REACH_SCALE: float = 0.05    # tip -> pre-grasp hover exp shaping
CARRY_SCALE: float = 0.10    # cube -> target xy exp shaping
YAW_SCALE: float = 0.50      # yaw-error exp shaping (only used once weighted in)
LIFT_REF: float = 0.10       # cube height (above resting) that saturates the lift term

# Flat hold-shell on top of the reach exponential: a constant bonus paid every step
# the tip is within REACH_HOLD_TOL of the hover point. Without it the bare exp keeps
# rewarding the last-millimetre approach, so the marginal gain of nudging closer beats
# the smoothness penalty and the policy hunts the optimum in a limit cycle (visible as
# wobble). The flat top removes that marginal gain inside the shell, so the smoothness
# terms win and the policy settles and holds — same mechanism as the hover milestone.
REACH_HOLD_TOL: float = 0.015    # tip -> hover distance that counts as "on target" (m)
REACH_HOLD_BONUS: float = 4.0    # per-step reach bonus while inside the hold shell

# The reach term aims the gripper at a pre-grasp *hover* point this high above the
# cube's xy, not the cube centre. Pointing it at the cube (which rests on the floor)
# made the reward optimum physically unreachable: closing the last few cm forces the
# jaws onto the floor, which is a terminating collision. With the target lifted clear
# of the floor the reach optimum is attainable without a crash, and "fly straight up"
# stops being the safe local optimum it otherwise collapses to.
REACH_TARGET_HEIGHT: float = 0.06

# Default reward weights = **chapter 1 (position only)**: yaw alignment is present
# in the term set but zero-weighted, exactly as the roadmap specifies. A later
# chapter is this dict with ``yaw`` (and friends) turned up — not a code change.
DEFAULT_WEIGHTS: dict[str, float] = {
    "reach": 1.0,
    "grasp": 1.0,
    "lift": 1.0,
    "carry": 1.0,
    "place": 5.0,
    "yaw": 0.0,
    # Smoothness shaping. ``action_smooth`` penalises set-point *speed* (first
    # difference) and ``jerk`` penalises set-point *acceleration* (second
    # difference). Jerk carries the weight: it punishes chattering / abrupt
    # direction changes without penalising steady, purposeful motion toward the
    # cube, so it can be pushed harder than ``action_smooth`` (which fights reach).
    "action_smooth": -0.5,
    "jerk": -2.0,
    "collision": -50.0,
}


@dataclass(frozen=True)
class RewardMetrics:
    """Physics-free snapshot the reward terms are computed from.

    Separated from the env so :func:`reward_terms` is unit-testable without a
    MuJoCo step, and so the curriculum runner can reason about terms from config.
    """

    tip_to_hover: float       # EE-tip -> pre-grasp hover point (above the cube) (m)
    grasped: bool             # a gripper jaw is in contact with the cube
    cube_height: float        # cube centre height above its resting height (m)
    cube_to_target_xy: float  # cube -> target distance in the floor plane (m)
    placed: bool              # cube within all placement tolerances of the target
    yaw_error: float          # shortest-arc cube-vs-target yaw error (rad)
    action_change_sq: float   # sum of squared per-joint set-point change (speed)
    action_jerk_sq: float     # sum of squared per-joint change-in-change (jerk)
    collision: bool           # an unexpected (non jaw<->cube) contact occurred


def reward_terms(m: RewardMetrics) -> dict[str, float]:
    """The composable per-chapter reward terms, each "higher is better".

    Penalties (``action_smooth``, ``collision``) are returned as non-negative
    magnitudes; they become penalties through their **negative weight** in the
    weights dict, so the total is always ``sum(weight[k] * term[k])``.
    """
    return {
        "reach": math.exp(-m.tip_to_hover / REACH_SCALE)
        + (REACH_HOLD_BONUS if m.tip_to_hover < REACH_HOLD_TOL else 0.0),
        "grasp": 1.0 if m.grasped else 0.0,
        # Lift only pays once the cube is actually held, so the policy can't farm
        # it by batting the cube up off the floor.
        "lift": (
            min(max(m.cube_height, 0.0), LIFT_REF) / LIFT_REF if m.grasped else 0.0
        ),
        "carry": math.exp(-m.cube_to_target_xy / CARRY_SCALE),
        "place": 1.0 if m.placed else 0.0,
        "yaw": math.exp(-m.yaw_error / YAW_SCALE),
        "action_smooth": m.action_change_sq,
        "jerk": m.action_jerk_sq,
        "collision": 1.0 if m.collision else 0.0,
    }


def weighted_reward(terms: dict[str, float], weights: dict[str, float]) -> float:
    """Combine reward terms by their per-chapter weights (missing weight = 0)."""
    return float(sum(weights.get(name, 0.0) * value for name, value in terms.items()))


def _build_base_model() -> mujoco.MjModel:
    """Compile the scene once, with the cube as a dynamic free body.

    The cube's pose is rewritten in ``qpos`` on every reset (no recompile), and it
    is a real free body so it falls, can be grasped, carried and dropped.
    """
    spec = build_scene(wrist_camera=False)
    spec.body("pick_cube").add_freejoint()
    return spec.compile()


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


def _joint_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint (low, high) absolute set-point bounds in ``JOINT_NAMES`` order.

    Limited joints use the model range; unlimited ones fall back to ``±π``.
    """
    low = np.empty(contract.N_JOINTS, dtype=np.float32)
    high = np.empty(contract.N_JOINTS, dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            low[i], high[i] = model.jnt_range[jid]
        else:
            low[i], high[i] = -math.pi, math.pi
    return low, high


def _actuator_map(model: mujoco.MjModel) -> dict[str, int]:
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }


class PickPlaceEnv(gym.Env):
    """The one env the whole curriculum finetunes on (frozen 31-dim contract).

    Reset randomises the source cube pose, the target cube pose and the start arm
    pose. The cube is a dynamic free body. The observation is the frozen contract;
    the action is an absolute joint set point (clamped + limit-clipped). The reward
    is a weighted sum of composable terms — pass ``reward_weights`` to select a
    chapter (default = chapter 1, position only, ``yaw`` weight 0).
    """

    metadata = {"render_modes": ["human"], "render_fps": int(CONTROL_HZ)}

    def __init__(
        self,
        render_mode: str | None = None,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.reward_weights = dict(DEFAULT_WEIGHTS if reward_weights is None else reward_weights)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(contract.OBS_DIM,), dtype=np.float32
        )
        self.render_mode = render_mode

        # Built once per worker process — reused across all episodes.
        self._model: mujoco.MjModel = _build_base_model()
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._k: So101Kinematics = derive_kinematics(self._model)
        self._actuator_id: dict[str, int] = _actuator_map(self._model)
        self._robot_geom_ids, self._env_geom_ids = build_geom_sets(self._model)
        self._qpos_adr, self._dof_adr = _joint_addresses(self._model)
        self._jaw_ids: list[int] = jaw_geom_ids(self._model)
        self._cube_body_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube"
        )
        self._cube_qpos_adr: int = int(
            self._model.jnt_qposadr[self._model.body_jntadr[self._cube_body_id]]
        )

        # The action is an absolute joint set point; bound it by the joint limits.
        low, high = _joint_bounds(self._model)
        self._joint_low, self._joint_high = low, high
        self.action_space = gym.spaces.Box(
            low=low, high=high, shape=(contract.ACT_DIM,), dtype=np.float32
        )

        # Episode state — updated in reset().
        self._target: CubePose = CubePose(0.0, 0.0, CUBE_HALF_SIZE)
        self._target_pose_vec: np.ndarray = np.zeros(contract.POSE_DIM, dtype=np.float64)
        self._prev_setpoint: np.ndarray = np.zeros(contract.ACT_DIM, dtype=np.float64)
        # Previous per-joint set-point change, for the jerk (second-difference) term.
        self._prev_delta: np.ndarray = np.zeros(contract.ACT_DIM, dtype=np.float64)
        self._step_count: int = 0
        self._viewer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_cube(self, pose: CubePose) -> None:
        """Write a cube pose into the free-body ``qpos`` (and zero its velocity)."""
        adr = self._cube_qpos_adr
        half_yaw = pose.yaw / 2.0
        self._data.qpos[adr : adr + 3] = (pose.x, pose.y, pose.z)
        self._data.qpos[adr + 3 : adr + 7] = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

    def _cube_pose_vec(self) -> np.ndarray:
        """Current cube pose as a 9-vector (position + 6D rotation), from the sim.

        Read straight from the free body — the privileged state the modular
        sim-to-real path later replaces with a fused vision/dead-reckoning estimate.
        """
        position = self._data.xpos[self._cube_body_id]
        rotation = self._data.xmat[self._cube_body_id].reshape(3, 3)
        return contract.pose_to_vec(position, rotation)

    def _cube_xyz_yaw(self) -> np.ndarray:
        """Current cube ``(x, y, z, yaw)`` from its free-body ``qpos``."""
        adr = self._cube_qpos_adr
        pos = self._data.qpos[adr : adr + 3]
        yaw = quat_yaw(self._data.qpos[adr + 3 : adr + 7])
        return np.array([pos[0], pos[1], pos[2], yaw])

    def _build_obs(self) -> np.ndarray:
        return contract.build_observation(
            joint_pos=self._data.qpos[self._qpos_adr],
            joint_vel=self._data.qvel[self._dof_adr],
            cube_pose=self._cube_pose_vec(),
            target_pose=self._target_pose_vec,
            confidence=contract.DEFAULT_CONFIDENCE,
        )

    def _metrics(self, setpoint: np.ndarray) -> tuple[RewardMetrics, dict]:
        arm_joints = {
            n: float(self._data.qpos[self._qpos_adr[i]])
            for i, n in enumerate(JOINT_NAMES)
            if n in ARM_JOINT_NAMES
        }
        tip = np.asarray(self._k.tip_position(arm_joints))
        cube = self._cube_xyz_yaw()
        # Reach aims at a hover point above the cube, not the cube itself (see
        # REACH_TARGET_HEIGHT) — keeps the reward optimum off the floor.
        reach_target = np.array([cube[0], cube[1], REACH_TARGET_HEIGHT])
        tip_to_hover = float(np.linalg.norm(tip - reach_target))

        # Split contacts: jaw<->cube is the intentional grasp (``is_unexpected`` is
        # False for it); everything else robot<->env / robot<->robot is a crash.
        contacts = scan_contacts(
            self._model, self._data, self._robot_geom_ids, self._env_geom_ids
        )
        grasp_contacts = [(a, b) for a, b in contacts if not is_unexpected(a, b)]
        collisions = [(a, b) for a, b in contacts if is_unexpected(a, b)]

        delta = setpoint - self._prev_setpoint

        xy_err, z_err, yaw_err = placement_errors(cube, self._target)
        placed = (
            xy_err <= SUCCESS_XY_TOLERANCE
            and z_err <= SUCCESS_Z_TOLERANCE
            and yaw_err <= SUCCESS_YAW_TOLERANCE
        )

        metrics = RewardMetrics(
            tip_to_hover=tip_to_hover,
            grasped=len(grasp_contacts) > 0,
            cube_height=float(cube[2]) - CUBE_HALF_SIZE,
            cube_to_target_xy=xy_err,
            placed=placed,
            yaw_error=yaw_err,
            action_change_sq=float(np.sum(delta**2)),
            action_jerk_sq=float(np.sum((delta - self._prev_delta) ** 2)),
            collision=len(collisions) > 0,
        )
        info = {
            "tip_to_hover": metrics.tip_to_hover,
            "grasped": metrics.grasped,
            "cube_height": metrics.cube_height,
            "cube_to_target_xy": metrics.cube_to_target_xy,
            "yaw_error": metrics.yaw_error,
            "placed": metrics.placed,
            "success": metrics.placed and not metrics.collision,
            "n_collisions": len(collisions),
            "collision": metrics.collision,
        }
        return metrics, info

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
            target = sample_cube(self.np_random)
            start_joints, start_gripper = sample_near_neutral(self.np_random)

            mujoco.mj_resetData(self._model, self._data)
            for name, value in start_joints.items():
                set_joint(self._model, self._data, name, value)
                self._data.ctrl[self._actuator_id[name]] = value
            set_joint(self._model, self._data, "gripper", start_gripper)
            self._data.ctrl[self._actuator_id["gripper"]] = start_gripper
            self._place_cube(source)
            mujoco.mj_forward(self._model, self._data)

            if (
                jaw_floor_clearance(self._model, self._data, self._jaw_ids)
                < MIN_START_CLEARANCE
            ):
                continue

            self._target = target
            self._target_pose_vec = contract.pose_vec_from_xyz_yaw(
                target.x, target.y, target.z, target.yaw
            )
            self._prev_setpoint = np.array(
                [start_joints[n] if n != "gripper" else start_gripper for n in JOINT_NAMES],
                dtype=np.float64,
            )
            self._prev_delta = np.zeros(contract.ACT_DIM, dtype=np.float64)
            self._step_count = 0

            info = {"source": source, "target": target}
            return self._build_obs(), info

        raise RuntimeError("Could not sample a valid pick-place episode in 50 attempts")

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        # Absolute joint set point: clamp the per-step change (contract, model-free)
        # then clip to the joint limits (the env owns the model).
        setpoint = contract.clamp_setpoint(self._prev_setpoint, np.asarray(action))
        setpoint = np.clip(setpoint, self._joint_low, self._joint_high)

        for i, name in enumerate(JOINT_NAMES):
            self._data.ctrl[self._actuator_id[name]] = float(setpoint[i])

        target_t = self._data.time + 1.0 / CONTROL_HZ
        while self._data.time < target_t:
            mujoco.mj_step(self._model, self._data)

        metrics, info = self._metrics(setpoint)
        terms = reward_terms(metrics)
        reward = weighted_reward(terms, self.reward_weights)
        info["reward_terms"] = terms

        self._prev_delta = setpoint - self._prev_setpoint
        self._prev_setpoint = setpoint
        self._step_count += 1

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        # Only a crash ends the episode early; success is held, not terminal (same
        # rationale as the hover milestone — otherwise the policy tags the success
        # shell once and quits instead of settling the cube on the target).
        terminated = metrics.collision
        truncated = self._step_count >= MAX_STEPS
        return self._build_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer  # lazy import — not needed for headless training

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
