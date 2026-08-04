# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One pick-and-place episode in DPPO's observation and action convention.

This plays the part the vendored ``RobomimicImageWrapper`` and ``MultiStep``
wrappers play for robomimic tasks, against :class:`PolicySimEnv`:

- observations are ``{"state": (cond_steps, 6), "rgb": (cond_steps, 6, H, W)}``,
  state min-max normalized to ``[-1, 1]`` with the training export's bounds and
  the two cameras concatenated overhead-then-wrist on the channel axis, exactly
  as ``diffusion_policy_server`` feeds the same checkpoint in closed loop;
- an action is a chunk of ``act_steps`` normalized six-dimensional joint
  commands, executed one control tick apart;
- the reward is the sparse full-task one: ``1.0`` on the step where the success
  oracle confirms a settled placement, ``0.0`` everywhere else.

Rendering goes straight to the recording resolution the dataset's videos were
written at, so the policy sees the same single area-downsample hop to 96x96 that
its training frames went through.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.dppo_rl.scenes import (
    TRAINING_CONTROL_HZ,
    TRAINING_MAX_STEPS,
    TRAINING_SEED_BASE,
    SceneStream,
)
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policy_sim import PolicySimEnv
from pick_and_place.scene_appearance import SceneAppearance

# The resolution the blue-cube dataset's videos were recorded at
# (``source_video_hw`` in its export manifest).
RECORDING_HW = (720, 960)
SUCCESS_REWARD = 1.0


def normalize_state(
    state: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Apply the exporter's per-dimension min-max map to ``[-1, 1]``."""
    return 2.0 * (state - minimum) / (maximum - minimum + 1e-6) - 1.0


def unnormalize_action(
    action: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Invert :func:`normalize_state` for a predicted action."""
    return (action + 1.0) / 2.0 * (maximum - minimum + 1e-6) + minimum


@dataclass(frozen=True)
class EnvConfig:
    """Everything a worker needs to build an identical environment."""

    normalization_path: Path
    image_hw: tuple[int, int] = (96, 96)
    render_hw: tuple[int, int] = RECORDING_HW
    cond_steps: int = 2
    act_steps: int = 8
    control_hz: float = TRAINING_CONTROL_HZ
    max_steps: int = TRAINING_MAX_STEPS
    seed_base: int = TRAINING_SEED_BASE
    # The appearance the checkpoint was trained in. The blue-cube policies were
    # trained on re-rendered episodes whose cube is solid blue; rolling them out
    # against the compiled AprilTag cube leaves them unable to see it at all.
    scene_appearance: SceneAppearance | None = None
    # Weight of the potential-based shaping term, in reward per metre. Zero
    # keeps the reward purely the sparse settled placement. Shaping is
    # potential-based (gamma * phi(s') - phi(s)), which leaves the optimal
    # policy unchanged, and the terminal reward still dominates: the potential
    # spans about 0.5 m, so a weight of 0.5 contributes at most ~0.25 against
    # the terminal 1.0.
    shaping_weight: float = 0.0
    # Discount used by the shaping term. Must match the trainer's gamma for the
    # policy-invariance guarantee to hold.
    gamma: float = 0.999
    # Diagnostic only. Replaces the task reward with a trivially learnable
    # function of the action itself: the negative mean absolute value of the
    # first arm joint's command, so the optimal policy simply shifts that output
    # toward zero. If PPO cannot improve *this*, the fault is in the gradient
    # path rather than in the task, the reward or the exploration -- and the
    # setup becomes a minimal reproduction. Never use for a real run.
    debug_action_reward: bool = False
    # Emit a privileged observation for an asymmetric critic. The actor never
    # reads it -- ``VisionUnet1D`` consumes only ``rgb`` and ``state`` -- so the
    # deployed policy is unchanged. A critic that must regress a sparse return
    # from 96x96 pixels learns almost nothing (measured correlation 0.33 with
    # realized returns), and its noise is what degrades the policy.
    privileged_obs: bool = False
    # Overridden by tests with a stub renderer; production always renders.
    renderer_factory: Callable[..., Any] | None = None


@dataclass
class EpisodeSummary:
    """Per-episode facts the trainer logs but never shows the policy."""

    scenario_id: str = ""
    control_steps: int = 0
    success: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)
    final_xy_error_m: float = float("nan")
    unexpected_collision: bool = False
    out_of_bounds: bool = False


class DppoTaskEnv:
    """A DPPO-facing episode loop over :class:`PolicySimEnv`."""

    def __init__(self, config: EnvConfig, *, offset: int = 0, stride: int = 1) -> None:
        self.config = config
        # The stream is built here rather than passed in so an episode's step
        # budget can only ever come from one place.
        self.scene_stream = SceneStream(
            offset=offset,
            stride=stride,
            seed_base=config.seed_base,
            control_hz=config.control_hz,
            max_steps=config.max_steps,
        )
        bounds = np.load(config.normalization_path)
        self.obs_min = bounds["obs_min"].astype(np.float32)
        self.obs_max = bounds["obs_max"].astype(np.float32)
        self.action_min = bounds["action_min"].astype(np.float32)
        self.action_max = bounds["action_max"].astype(np.float32)
        renderer = (
            {} if config.renderer_factory is None
            else {"renderer_factory": config.renderer_factory}
        )
        self._env = PolicySimEnv(
            image_hw=config.image_hw,
            render_hw=config.render_hw,
            scene_appearance=config.scene_appearance,
            **renderer,
        )
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=config.cond_steps)
        self._summary = EpisodeSummary()
        self._step = 0
        self._video_frames: list[np.ndarray] | None = None
        self._video_path: Path | None = None

    # -- observations ----------------------------------------------------

    def _record(self, observation: dict[str, np.ndarray]) -> None:
        overhead = np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8)
        wrist = np.asarray(observation[WRIST_FEATURE], dtype=np.uint8)
        self._history.append({
            "state": normalize_state(
                np.asarray(observation[STATE_FEATURE], dtype=np.float32),
                self.obs_min,
                self.obs_max,
            ).astype(np.float32),
            # HWC per camera to the CHW pair the encoder expects.
            "rgb": np.concatenate(
                [overhead.transpose(2, 0, 1), wrist.transpose(2, 0, 1)], axis=0
            ),
        })
        if self.config.privileged_obs:
            self._history[-1]["privileged"] = self._privileged()
        if self._video_frames is not None:
            self._video_frames.append(np.concatenate([overhead, wrist], axis=1))

    def _privileged(self) -> np.ndarray:
        """Simulator facts that make the value function learnable.

        Never shown to the actor. Distances and positions are in metres, already
        on a scale a small MLP handles without further normalization.
        """
        state = self._env._last_task_state
        cube_x, cube_y, cube_z = state.cube_position_m
        target_x, target_y = state.target_xy_m
        tcp_to_cube = self._env.tcp_to_cube_distance_m()
        return np.array([
            cube_x, cube_y, cube_z,
            target_x, target_y,
            cube_x - target_x, cube_y - target_y,
            math.hypot(cube_x - target_x, cube_y - target_y),
            tcp_to_cube,
            float(state.robot_cube_contact),
            float(state.grasped),
            float(state.gripper_open),
            self._step / max(self.config.max_steps, 1),
        ], dtype=np.float32)

    def _debug_step(
        self, chunk: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Diagnostic step: reward is -mean|j1| over the executed chunk."""
        reward = -float(np.abs(chunk[:, 0]).mean())
        for action in chunk:
            observation, _, terminated, truncated, info = self._env.step(
                unnormalize_action(action, self.action_min, self.action_max)
            )
            self._step = int(info["control_steps"])
            self._record(observation)
            if terminated or truncated:
                break
        summary = self._summary
        stacked = self._stacked()
        if terminated or truncated:
            stacked = self.reset()
        return stacked, reward, terminated or truncated, truncated, {"episode": summary}

    def _potential(self) -> float:
        """Negative remaining distance: reach the cube, then carry it to target.

        Summing both legs keeps the potential continuous through the grasp,
        which a phase-switched potential would not be.
        """
        if self.config.shaping_weight == 0.0:
            return 0.0
        state = self._env._last_task_state
        cube_x, cube_y, _ = state.cube_position_m
        target_x, target_y = state.target_xy_m
        cube_to_target = math.hypot(cube_x - target_x, cube_y - target_y)
        return -self.config.shaping_weight * (
            self._env.tcp_to_cube_distance_m() + cube_to_target
        )

    def _stacked(self) -> dict[str, np.ndarray]:
        """The last ``cond_steps`` observations, padding a short history with its
        oldest entry the way the closed-loop controller does at episode start."""
        history = list(self._history)
        padding = [history[0]] * (self.config.cond_steps - len(history))
        history = padding + history
        keys = ("state", "rgb", "privileged") if self.config.privileged_obs else ("state", "rgb")
        return {key: np.stack([item[key] for item in history]) for key in keys}

    # -- episode loop ----------------------------------------------------

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        options = options or {}
        self._flush_video()
        video_path = options.get("video_path")
        self._video_path = Path(video_path) if video_path is not None else None
        self._video_frames = [] if self._video_path is not None else None

        scenario = self.scene_stream.next()
        self._step = 0
        observation, _ = self._env.reset(options={"scenario": scenario})
        self._history.clear()
        self._record(observation)
        self._summary = EpisodeSummary(scenario_id=scenario.scenario_id)
        return self._stacked()

    def step(
        self, action_chunk: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one chunk of normalized actions, one control tick apart."""
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None]
        potential_before = self._potential()
        reward = 0.0
        if self.config.debug_action_reward:
            # Depends only on the commanded action, not on the environment.
            return self._debug_step(chunk)
        terminated = truncated = False
        for action in chunk:
            observation, step_reward, terminated, truncated, info = self._env.step(
                unnormalize_action(action, self.action_min, self.action_max)
            )
            reward += float(step_reward)
            self._step = int(info["control_steps"])
            self._record(observation)
            if terminated or truncated:
                break
        if self.config.shaping_weight != 0.0:
            # Progress shaping: phi(s') - phi(s), with no discount factor on the
            # first term. The textbook potential-based form gamma*phi(s') - phi(s)
            # is policy-invariant, but for gamma < 1 it pays (gamma - 1) * phi per
            # step for standing still -- and phi is negative here, so that is a
            # *positive* reward for remaining far from the goal. At gamma 0.9 with
            # weight 5 it came to +0.25 a step, about +4.75 an episode, which was
            # the entire apparent "shaped return" of that run and rewarded not
            # finishing the task. Dropping the factor makes standing still worth
            # exactly zero and approaching worth the distance closed.
            potential_after = 0.0 if (terminated or truncated) else self._potential()
            reward += potential_after - potential_before

        task_state = info["task_state"]
        cube_x, cube_y, _ = task_state["cube_position_m"]
        target_x, target_y = task_state["target_xy_m"]
        self._summary = EpisodeSummary(
            scenario_id=self._summary.scenario_id,
            control_steps=int(info["control_steps"]),
            success=bool(info["success"]),
            milestones=dict(info["milestones"]),
            final_xy_error_m=float(math.hypot(cube_x - target_x, cube_y - target_y)),
            unexpected_collision=bool(task_state["unexpected_collision"]),
            out_of_bounds=bool(task_state["out_of_bounds"]),
        )
        observation = self._stacked()
        summary = self._summary
        if terminated or truncated:
            # DPPO's rollout loop expects the env to be ready for the next step,
            # so start the next episode here and hand back its first observation.
            observation = self.reset()
        # Report the step limit as terminal, not merely truncated. DPPO's
        # advantage estimator masks bootstrapping with `1 - terminated`, so a
        # truncated step is credited with `gamma * V(next observation)` -- and
        # because the line above has already reset, that observation belongs to
        # the *next* episode. Its value is that of a fresh neutral start, which
        # this policy solves most of the time, so every timeout (a failure) would
        # be credited with a large future return and PPO would learn that running
        # out the clock pays. Here the step budget is part of the task, so a
        # timed-out episode genuinely has zero future reward and cutting the
        # bootstrap is the correct estimate rather than a convenient one.
        return observation, reward, terminated or truncated, truncated, {"episode": summary}

    # -- diagnostics -----------------------------------------------------

    def _flush_video(self) -> None:
        if self._video_path is None or not self._video_frames:
            self._video_frames = None
            return
        import imageio.v2 as imageio

        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            self._video_path, fps=int(round(self.config.control_hz))
        ) as writer:
            for frame in self._video_frames:
                writer.append_data(frame)
        self._video_frames = None

    def close(self) -> None:
        self._flush_video()
        self._env.close()
