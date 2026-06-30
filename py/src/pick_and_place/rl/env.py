# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Gymnasium env for reverse-curriculum pick-and-place.

The env wraps the same MuJoCo scene the rest of the project builds, with a reset
that restores a full-state snapshot drawn from a pool of recorded successful
episodes (see :mod:`pick_and_place.rl.episode_pool`). The curriculum *stage*
selects which scripted phase the reset starts from; stages run in reverse so the
policy first learns to finish the task and the reset distribution then walks
backward toward the neutral start one phase at a time.

The obs/action interface is identical for every stage, so the network is trained
once and reused as the reset distribution moves backward:

  observation (29-dim, world frame)
    joint positions   6   5 arm joints + gripper
    joint velocities  6
    cube position     3
    cube orientation  6   6D rotation (first two rotation-matrix columns)
    cube lin+ang vel  6
    target xy         2

  action (6-dim)
    absolute joint position targets, JOINT_NAMES order, applied at control_hz.

The reward is a terminal placement score: once the cube has settled on the floor,
the episode ends and receives a reward based on the cube-target xy distance.
Within the success tolerance this is 1.0; farther clean floor placements receive
partial credit. An unexpected collision or the cube leaving the workspace ends
the episode as a failure; running past a per-episode step budget (scaled to the
scripted duration remaining from the reset frame) truncates it.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from pick_and_place import build_scene
from pick_and_place.episodes import build_geom_sets, is_unexpected, scan_contacts
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.rl.episode_pool import EpisodePool, ResetSnapshot

# Reverse-curriculum stages: stage k resets at the start of this scripted phase
# (and anywhere through the end of the trajectory). Stage 0 is the easy tail.
CURRICULUM_PHASES: tuple[str, ...] = (
    "release",  # drop the carried cube onto the target
    "carry",    # carry to the target hover, then drop
    "lift",     # lift the grasped cube, carry, drop
    "grasp",    # close on the cube, lift, carry, drop
    "descent",  # descend onto the cube, grasp, ..., drop
    "approach",  # the full task from the neutral start
)

# Success oracle, matching record_episodes.py: the cube has settled within this
# far (m) of the target in the floor plane, sits at cube-half-size above the
# floor, and is at rest.
SUCCESS_XY_TOLERANCE = 0.04
SUCCESS_Z_TOLERANCE = 0.01
# Cube speed (m/s and rad/s) below which it counts as settled rather than still
# falling or rolling — required so a cube passing through the target mid-bounce
# is not scored as a successful placement.
SETTLED_LIN_SPEED = 0.02
SETTLED_ANG_SPEED = 0.2

# Generous bounds beyond which the cube is considered knocked out of the
# workspace (a failure): more than this far from the arm base horizontally, or
# lifted/sunk past these heights.
_MAX_CUBE_RADIUS = 0.8
_MAX_CUBE_HEIGHT = 0.6
_MIN_CUBE_HEIGHT = -0.05

# Per-episode step budget = remaining scripted frames from the reset point times
# this slack, floored at a small minimum so a reset near the very end still has
# room to act.
_BUDGET_SLACK = 1.5
_MIN_BUDGET_STEPS = 20

# Terminal shaping once the cube has settled on the floor. A valid but off-target
# floor placement receives a small base reward plus a linear xy-distance score,
# while anything within the success tolerance receives the full reward.
FLOOR_PLACEMENT_REWARD = 0.2
PLACEMENT_REWARD_MAX_XY_ERROR = 0.20


class ReverseCurriculumEnv(gym.Env):
    """Pick-and-place env whose reset distribution is a recorded-episode pool."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        pool: EpisodePool | Path | str,
        *,
        stage: int = 0,
        phase_fraction: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.pool = pool if isinstance(pool, EpisodePool) else EpisodePool(Path(pool))
        for phase in CURRICULUM_PHASES:
            if phase not in self.pool.phase_names:
                raise ValueError(
                    f"curriculum phase {phase!r} not in pool phases {self.pool.phase_names}"
                )
        self.set_stage(stage)
        self.phase_fraction = phase_fraction
        self.render_mode = render_mode

        self.model, self.data, self._marker_mocapid = _build_model()
        if self.model.nq != self.pool.nq:
            raise ValueError(
                f"model nq {self.model.nq} != recorded qpos width {self.pool.nq}; "
                "scene mismatch with the episode pool"
            )

        actuator_id = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }
        self._ctrl_index = np.array([actuator_id[name] for name in JOINT_NAMES])
        self._joint_qpos_adr = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in JOINT_NAMES
        ])
        self._joint_dof_adr = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in JOINT_NAMES
        ])
        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[cube_body]])
        self._cube_dof_adr = int(self.model.jnt_dofadr[self.model.body_jntadr[cube_body]])
        self._robot_geom_ids, self._env_geom_ids = build_geom_sets(self.model)

        ctrl_range = self.model.actuator_ctrlrange[self._ctrl_index]
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )

        control_hz = self.pool.control_hz or 50.0
        self._sim_steps = max(1, round((1.0 / control_hz) / float(self.model.opt.timestep)))

        self._target_xy = np.zeros(2)
        self._step_count = 0
        self._max_steps = _MIN_BUDGET_STEPS
        self._viewer = None
        self._renderer: mujoco.Renderer | None = None

    # -- curriculum ---------------------------------------------------------

    def set_stage(self, stage: int) -> None:
        """Select the reverse-curriculum stage (0 = easy tail, drop-only)."""
        if not 0 <= stage < len(CURRICULUM_PHASES):
            raise ValueError(f"stage {stage} out of range 0..{len(CURRICULUM_PHASES) - 1}")
        self.stage = stage
        self.phase = CURRICULUM_PHASES[stage]

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        snapshot = self.pool.sample_reset(
            self.np_random, self.phase, phase_fraction=self.phase_fraction
        )
        self._restore(snapshot)
        remaining = snapshot.total_frames - snapshot.frame
        self._max_steps = max(_MIN_BUDGET_STEPS, math.ceil(remaining * _BUDGET_SLACK))
        self._step_count = 0
        return self._observation(), self._info(snapshot)

    def step(self, action):
        ctrl = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self._ctrl_index] = ctrl
        for _ in range(self._sim_steps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        collided = self._has_unexpected_collision()
        out_of_bounds = self._cube_out_of_bounds()
        settled_on_floor = self._cube_settled_on_floor()
        xy_error = self._cube_xy_error()
        success = (
            (not collided)
            and (not out_of_bounds)
            and settled_on_floor
            and xy_error <= SUCCESS_XY_TOLERANCE
        )

        terminated = settled_on_floor or collided or out_of_bounds
        truncated = (not terminated) and self._step_count >= self._max_steps
        reward = (
            self._placement_reward(xy_error)
            if settled_on_floor and not (collided or out_of_bounds)
            else 0.0
        )

        info = {
            "success": success,
            "collision": collided,
            "out_of_bounds": out_of_bounds,
            "settled_on_floor": settled_on_floor,
            "xy_error": xy_error,
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    # -- state restore / readout -------------------------------------------

    def _restore(self, snapshot: ResetSnapshot) -> None:
        # Clear solver warm-start, contacts, and time from the previous episode so
        # physics restarts cleanly from this snapshot rather than continuing the
        # last one.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.ctrl[self._ctrl_index] = snapshot.ctrl
        self._target_xy = snapshot.target_xy.copy()
        self.data.mocap_pos[self._marker_mocapid] = (
            float(self._target_xy[0]),
            float(self._target_xy[1]),
            0.0,
        )
        mujoco.mj_forward(self.model, self.data)

    def _cube_xyz(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3]

    def _cube_orientation_6d(self) -> np.ndarray:
        quat = self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        # First two columns of the rotation matrix (column-major slice).
        return mat.reshape(3, 3)[:, :2].flatten(order="F")

    def _cube_velocity(self) -> np.ndarray:
        return self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6]

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos[self._joint_qpos_adr],
                self.data.qvel[self._joint_dof_adr],
                self._cube_xyz(),
                self._cube_orientation_6d(),
                self._cube_velocity(),
                self._target_xy,
            ]
        ).astype(np.float32)

    def _info(self, snapshot: ResetSnapshot) -> dict:
        return {
            "stage": self.stage,
            "phase": self.phase,
            "reset_frame": snapshot.frame,
            "max_steps": self._max_steps,
            "source": snapshot.source.name,
        }

    # -- termination predicates --------------------------------------------

    def _has_unexpected_collision(self) -> bool:
        for n1, n2 in scan_contacts(
            self.model, self.data, self._robot_geom_ids, self._env_geom_ids
        ):
            if is_unexpected(n1, n2):
                return True
        return False

    def _cube_out_of_bounds(self) -> bool:
        x, y, z = self._cube_xyz()
        return (
            math.hypot(float(x), float(y)) > _MAX_CUBE_RADIUS
            or not (_MIN_CUBE_HEIGHT < float(z) < _MAX_CUBE_HEIGHT)
        )

    def _cube_xy_error(self) -> float:
        x, y, _ = self._cube_xyz()
        return math.hypot(float(x) - self._target_xy[0], float(y) - self._target_xy[1])

    def _cube_settled_on_floor(self) -> bool:
        _, _, z = self._cube_xyz()
        if abs(float(z) - CUBE_HALF_SIZE) > SUCCESS_Z_TOLERANCE:
            return False
        vel = self._cube_velocity()
        return (
            float(np.linalg.norm(vel[:3])) < SETTLED_LIN_SPEED
            and float(np.linalg.norm(vel[3:])) < SETTLED_ANG_SPEED
        )

    def _placement_reward(self, xy_error: float) -> float:
        if xy_error <= SUCCESS_XY_TOLERANCE:
            return 1.0
        distance_span = PLACEMENT_REWARD_MAX_XY_ERROR - SUCCESS_XY_TOLERANCE
        if distance_span <= 0.0:
            return FLOOR_PLACEMENT_REWARD
        distance_score = 1.0 - min(
            1.0,
            max(0.0, (xy_error - SUCCESS_XY_TOLERANCE) / distance_span),
        )
        return FLOOR_PLACEMENT_REWARD + (1.0 - FLOOR_PLACEMENT_REWARD) * distance_score

    # -- rendering ----------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _build_model() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Compile the scene once with a dynamic cube and a mocap drop-target marker.

    The cube is a free body so any episode's full ``qpos`` restores into it, and
    the target is a mocap marker repositioned per reset, so a single model/data
    serves the whole pool without recompiling.
    """
    spec = build_scene(include_environment=True)
    marker = spec.worldbody.add_body(name="reset_target_marker")
    marker.mocap = True
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.002),
        size=(CUBE_HALF_SIZE, CUBE_HALF_SIZE, 0.001),
        rgba=(0.0, 0.95, 0.35, 0.7),
        contype=0,
        conaffinity=0,
    )
    spec.body("pick_cube").add_freejoint()
    model = spec.compile()
    marker_mocapid = int(
        model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reset_target_marker")
        ]
    )
    return model, mujoco.MjData(model), marker_mocapid
