# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Gymnasium environment for the 'lift the cube' sub-task.

This is the natural next step after the approach-to-hover milestone
(:mod:`pick_and_place.rl.hover_env`): the arm starts at a near-neutral pose and
must reach the cube, close the gripper on it and lift it off the floor. It shares
the hover env's **23-dim observation and 6-dim delta-action layout exactly**, so a
policy trained by ``train_rl_hover`` can be used as a warm start — its learned
"move the EE tip toward the target in obs[18:21]" behaviour transfers directly,
and only the reward changes (lift the cube instead of hovering above it).

The two differences from the hover env that matter:

* The cube is a real **dynamic free body** (it can be grasped, lifted and
  dropped), not the static decoration of the hover milestone.
* The reach target fed in ``obs[18:21]`` is the **live cube position** (not a
  fixed hover point), so reaching for it means descending onto the cube to grasp.

Observation (23-dim, identical layout to the hover env):
    [0:6]   joint positions (rad), JOINT_NAMES order
    [6:12]  joint velocities (rad/s)
    [12:18] previous commanded action (absolute joint set point)
    [18:21] grasp target — current cube position (x, y, z) in world frame
    [21:23] target wrist-roll (cos, sin)

Action (6-dim): delta fractions in [-1, 1]. Each value is multiplied by
``max_delta`` and added to the previous command, so ±1 is the largest allowed
step and 0 holds. The resulting absolute command is what is sent to the
actuators.

The action regime — ``control_hz`` (decisions per second), ``max_delta`` (largest
per-decision joint change) and ``max_steps`` (episode horizon) — is configurable
per env so it can be A/B-tested without editing code. ``max_delta`` is purely a
**sim training** knob: keep it large enough that exploration can actually reach
and grasp within an episode, and reconcile real-hardware safety with a deploy-time
action down-scale instead. The module-level ``MAX_DELTA`` / ``CONTROL_HZ`` /
``MAX_STEPS`` are just the defaults.

Reward (``reward_mode``): the bare, unshaped signal — no reach/grasp shaping, no
smoothness penalty, no collision handling. Paid every step.

* ``"lift"`` (default): the cube's height above its resting height, so it rewards
  both raising the cube and holding it up.
* ``"move"``: the 3D displacement of the cube centre from its reset position. A
  much denser bootstrap — bumping or sliding the cube already pays — and since
  lifting straight up is just one way to displace it, ``"lift"`` is a special case
  of ``"move"``, so a move-trained policy warm-starts a lift run cleanly. Note it
  is maximised by *whacking* the cube away, so it is a contact/diagnostic rung,
  not a reward to train lifting to convergence on.

Nothing terminates the episode early — not even a crash; it always runs the full
horizon, so the reward is the only thing shaping behaviour. ``grasped`` and
``collision`` are recorded in ``info`` for monitoring only.
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
    is_unexpected,
    jaw_floor_clearance,
    jaw_geom_ids,
    sample_cube,
    sample_near_neutral,
    scan_contacts,
    set_joint,
)
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import CUBE_HALF_SIZE, VERTICAL_FACES, CubePose, pregrasp_matrix
from pick_and_place.ik import solve_simple_pregrasp_ik
from pick_and_place.kinematics import So101Kinematics, derive_kinematics
from pick_and_place.trajectory import SOURCE_HOVER_TIP_Z

# Default max absolute joint change per control step at the default 50 Hz
# (~2.5 rad/s upper bound). Per-env override via the ``max_delta`` __init__ param.
MAX_DELTA: float = 0.05

# Reward: the cube's height above its resting height (m), and nothing else — the
# bare "lift the cube" signal, with no reach/grasp shaping, no smoothness penalty
# and no collision handling. The point is to see what an unshaped lift reward
# produces. The height that counts as a successful lift (info only — not terminal).
SUCCESS_LIFT_HEIGHT: float = 0.05  # 5 cm clear of the floor

# Defaults for the action regime; per-env overrides via the ``control_hz`` and
# ``max_steps`` __init__ params.
CONTROL_HZ: float = 50.0
MAX_STEPS: int = 300               # 6 s at 50 Hz

OBS_DIM: int = 23                  # [18:21] live cube pos, [21:23] wrist-roll (cos, sin)
ACT_DIM: int = 6


def _select_grasp_wrist_roll(
    k: So101Kinematics, source: CubePose, hover_offset: float
) -> float | None:
    """Return wrist-roll for the most natural grasp-feasible cube face (hover IK).

    Picks the vertical cube face whose outward normal best faces the arm base and
    is reachable by the simple pre-grasp IK, then returns that branch's wrist-roll
    so the gripper is pre-aligned to the cube. (Same selection as the hover
    milestone — alignment is what lets the grasp close cleanly.)
    """
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
    """Compile the scene once, with the cube as a dynamic free body.

    The cube's pose is rewritten in ``qpos`` on every reset (no recompile), and it
    is a real free body so it can be grasped, lifted and dropped.
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


def _actuator_map(model: mujoco.MjModel) -> dict[str, int]:
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}


class LiftCubeEnv(gym.Env):
    """Gymnasium env: reach the cube, grasp it and lift it off the floor.

    The cube pose and start arm pose are randomised each episode. The cube is a
    dynamic free body. The observation and action share the hover milestone's
    23-dim / 6-dim layout so a ``train_rl_hover`` checkpoint warm-starts training.

    ``reward_mode`` selects the (unshaped) reward: ``"lift"`` (cube height) or
    ``"move"`` (cube displacement from its reset pose) — see the module docstring.
    """

    metadata = {"render_modes": ["human"], "render_fps": int(CONTROL_HZ)}

    REWARD_MODES = ("lift", "move")

    def __init__(
        self,
        render_mode: str | None = None,
        reward_mode: str = "lift",
        control_hz: float = CONTROL_HZ,
        max_delta: float = MAX_DELTA,
        max_steps: int = MAX_STEPS,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        if reward_mode not in self.REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {self.REWARD_MODES}, got {reward_mode!r}"
            )
        self._reward_mode = reward_mode
        self._control_hz = float(control_hz)
        self._max_delta = float(max_delta)
        self._max_steps = int(max_steps)
        # render_fps follows the control rate so viewers play back at task speed.
        self.metadata = {**self.metadata, "render_fps": int(round(self._control_hz))}
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
        self._cube_qpos_adr: int = int(
            self._model.jnt_qposadr[self._model.body_jntadr[self._cube_body_id]]
        )
        self._cube_dof_adr: int = int(
            self._model.jnt_dofadr[self._model.body_jntadr[self._cube_body_id]]
        )

        # Episode state — updated in reset().
        self._wrist_roll_target: float = 0.0
        self._cube_start: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_action: np.ndarray = np.zeros(ACT_DIM, dtype=np.float32)
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
        self._data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

    def _cube_position(self) -> np.ndarray:
        """Current cube centre position (x, y, z) from its free-body ``qpos``."""
        adr = self._cube_qpos_adr
        return np.asarray(self._data.qpos[adr : adr + 3], dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0:6] = self._data.qpos[self._qpos_adr]
        obs[6:12] = self._data.qvel[self._dof_adr]
        obs[12:18] = self._prev_action
        obs[18:21] = self._cube_position()
        obs[21] = math.cos(self._wrist_roll_target)
        obs[22] = math.sin(self._wrist_roll_target)
        return obs

    def _reward_and_info(self) -> tuple[float, dict]:
        cube = self._cube_position()
        cube_height = float(cube[2]) - CUBE_HALF_SIZE
        cube_moved = float(np.linalg.norm(cube - self._cube_start))

        # The whole reward, paid every step (see REWARD_MODES / the module docstring):
        # "lift" rewards raising and holding the cube; "move" rewards any displacement.
        reward = cube_height if self._reward_mode == "lift" else cube_moved

        # Recorded for monitoring only — neither term affects the reward or ends
        # the episode (so we can see whether the policy makes a real grasp or just
        # bats / flips the cube, and whether it crashes).
        contacts = scan_contacts(self._model, self._data, self._robot_geom_ids, self._env_geom_ids)
        grasped = any(not is_unexpected(a, b) for a, b in contacts)
        collision = any(is_unexpected(a, b) for a, b in contacts)

        info = {
            "cube_height": cube_height,
            "cube_moved": cube_moved,
            "grasped": grasped,
            "collision": collision,
            "success": cube_height >= SUCCESS_LIFT_HEIGHT,
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
            start_joints, start_gripper = sample_near_neutral(self.np_random)

            mujoco.mj_resetData(self._model, self._data)
            for name, value in start_joints.items():
                set_joint(self._model, self._data, name, value)
                self._data.ctrl[self._actuator_id[name]] = value
            set_joint(self._model, self._data, "gripper", start_gripper)
            self._data.ctrl[self._actuator_id["gripper"]] = start_gripper
            self._place_cube(source)
            mujoco.mj_forward(self._model, self._data)

            if jaw_floor_clearance(self._model, self._data, self._jaw_ids) < MIN_START_CLEARANCE:
                continue

            # Wrist-roll pre-aligned to a grasp-feasible cube face (hover IK only).
            hover_offset = SOURCE_HOVER_TIP_Z - source.z
            wrist_roll = _select_grasp_wrist_roll(self._k, source, hover_offset)
            if wrist_roll is None:
                continue

            self._wrist_roll_target = wrist_roll
            # Reset position the "move" reward measures cube displacement against.
            self._cube_start = self._cube_position()
            self._prev_action = np.array(
                [start_joints[n] if n != "gripper" else start_gripper for n in JOINT_NAMES],
                dtype=np.float32,
            )
            self._step_count = 0

            info = {"source": source}
            return self._build_obs(), info

        raise RuntimeError("Could not sample a valid lift episode in 50 attempts")

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        # action ∈ [-1, 1]: delta fraction. Convert to absolute command.
        action = np.asarray(action, dtype=np.float32)
        clipped = np.clip(self._prev_action + action * self._max_delta, -np.pi, np.pi)

        for i, name in enumerate(JOINT_NAMES):
            self._data.ctrl[self._actuator_id[name]] = float(clipped[i])

        target_t = self._data.time + 1.0 / self._control_hz
        while self._data.time < target_t:
            mujoco.mj_step(self._model, self._data)

        reward, info = self._reward_and_info()
        self._prev_action = clipped
        self._step_count += 1

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        # Nothing ends the episode early — not even a crash. The episode always
        # runs the full horizon so the only thing driving behaviour is the lift
        # reward (every step the cube is up adds to the return).
        truncated = self._step_count >= self._max_steps
        return self._build_obs(), reward, False, truncated, info

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
