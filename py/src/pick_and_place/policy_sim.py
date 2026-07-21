# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reusable visual-policy MuJoCo environment for closed-loop evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, fields, replace
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pick_and_place import build_scene
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.domain_randomization import (
    DomainRandomizer,
    DomainSample,
    reload_renderer_textures,
)
from pick_and_place.episodes import build_geom_sets, is_unexpected, scan_contacts
from pick_and_place.executor import CONTROL_HZ, HARDWARE_SIMULATION_HZ
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    GRIPPER_INDEX,
    JOINT_NAMES,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.miscalibration import MiscalibrationDraw
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    add_paper_target_marker,
    place_paper_target_marker,
)
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    PolicyController,
    PolicyObservation,
)
from pick_and_place.policy_evaluation import (
    EpisodeResult,
    EvaluationScenario,
    TaskState,
    TaskSuccessOracle,
)
from pick_and_place.robot_dynamics import set_actuator_activation
from pick_and_place.sim_recorder import resize_and_center_crop
from pick_and_place.workspace_overlays import is_cube_drop_allowed

_MAX_CUBE_RADIUS_M = 0.8
_MIN_CUBE_HEIGHT_M = -0.05
_MAX_CUBE_HEIGHT_M = 0.6
_GRIPPER_OPENING_THRESHOLD_RAD = 0.5
_JAW_PREFIXES = ("fixed_jaw_col", "moving_jaw_col")

RendererFactory = Callable[..., Any]


def build_policy_sim_model(render_height: int, render_width: int) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the calibrated visual-policy scene with a free pick cube."""
    if render_height < 1 or render_width < 1:
        raise ValueError("render dimensions must be positive")
    spec = build_scene(include_environment=True)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, render_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, render_height)
    apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    intrinsics = load_local_camera_intrinsics()
    for camera in spec.cameras:
        if camera.name in intrinsics and "fovy_deg" in intrinsics[camera.name]:
            camera.fovy = float(intrinsics[camera.name]["fovy_deg"])

    spec.body("pick_cube").add_freejoint()
    add_paper_target_marker(spec)
    model = spec.compile()
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    return model, mujoco.MjData(model)


def joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINT_NAMES
    ])


def sim_state_to_real(
    qpos_rad: np.ndarray,
    joint_offsets_rad: dict[str, float] | None = None,
) -> np.ndarray:
    """Convert true MuJoCo joints to servo-style hardware-frame readback."""
    offsets = joint_offsets_rad or {}
    arm = {
        name: float(qpos_rad[index]) - offsets.get(name, 0.0)
        for index, name in enumerate(ARM_JOINT_NAMES)
    }
    return sim_frame_to_real(arm, float(qpos_rad[GRIPPER_INDEX])).astype(np.float32)


def real_action_to_sim_ctrl(action_real: np.ndarray) -> np.ndarray:
    """Convert a six-dimensional hardware-frame policy action to radians."""
    action = np.asarray(action_real, dtype=np.float32).reshape(-1)
    if action.shape != (len(JOINT_NAMES),):
        raise ValueError(f"action must have shape ({len(JOINT_NAMES)},), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("action must contain only finite values")
    arm_rad, gripper_rad = real_frame_to_sim(action)
    return np.array([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad])


def _miscalibration_from_scenario(scenario: EvaluationScenario) -> MiscalibrationDraw:
    payload = scenario.miscalibration_sample
    expected = {"joint_offsets_deg"}
    if set(payload) != expected:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} has invalid miscalibration fields; "
            f"missing={sorted(expected - set(payload))}, unknown={sorted(set(payload) - expected)}"
        )
    raw_offsets = payload["joint_offsets_deg"]
    if not isinstance(raw_offsets, dict):
        raise ValueError("joint_offsets_deg must be a JSON object")
    unknown_joints = set(raw_offsets) - set(ARM_JOINT_NAMES)
    if unknown_joints:
        raise ValueError(f"joint_offsets_deg contains unknown joints: {sorted(unknown_joints)}")
    joint_offsets = {str(name): float(value) for name, value in raw_offsets.items()}
    if not all(math.isfinite(value) for value in joint_offsets.values()):
        raise ValueError("joint_offsets_deg must contain only finite numbers")
    return MiscalibrationDraw(
        base_offsets_deg=joint_offsets,
        pan_jitter=None,
        cube_belief_error=(0.0, 0.0, 0.0, 0.0),
        target_belief_error=(0.0, 0.0),
    )


def _domain_sample_from_scenario(
    scenario: EvaluationScenario,
    miscalibration: MiscalibrationDraw,
) -> DomainSample | None:
    payload = dict(scenario.domain_randomization_sample)
    enabled = payload.pop("enabled", scenario.domain_randomization_preset is not None)
    if not enabled:
        if payload:
            raise ValueError(
                f"disabled domain sample for {scenario.scenario_id!r} contains sampled values"
            )
        return None
    expected = {field.name for field in fields(DomainSample)} - {"miscalibration"}
    if set(payload) != expected:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} has invalid domain sample fields; "
            f"missing={sorted(expected - set(payload))}, unknown={sorted(set(payload) - expected)}"
        )
    payload["material_factors"] = {
        name: tuple(float(value) for value in factors)
        for name, factors in payload["material_factors"].items()
    }
    for name in (
        "key_light_position",
        "key_light_target",
        "overhead_camera_position_m",
        "overhead_camera_rotation_deg",
        "wrist_camera_position_m",
        "wrist_camera_rotation_deg",
        "background_rgb",
        "table_rgb",
        "white_balance",
    ):
        payload[name] = tuple(float(value) for value in payload[name])
    return DomainSample(**payload, miscalibration=miscalibration)


class PolicySimEnv(gym.Env):
    """Headless visual policy environment reset by an explicit frozen scenario."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(CONTROL_HZ)}

    def __init__(
        self,
        *,
        image_hw: tuple[int, int],
        render_hw: tuple[int, int] = (1080, 1920),
        renderer_factory: RendererFactory = mujoco.Renderer,
    ) -> None:
        super().__init__()
        image_height, image_width = image_hw
        render_height, render_width = render_hw
        if min(image_height, image_width, render_height, render_width) < 1:
            raise ValueError("image and render dimensions must be positive")
        if render_height < image_height or render_width < image_width:
            raise ValueError("render dimensions must be at least the policy image dimensions")
        self.image_hw = image_hw
        self.render_hw = render_hw
        self.model, self.data = build_policy_sim_model(render_height, render_width)
        self._renderer_factory = renderer_factory
        self._renderer: Any | None = None
        self._randomizer = DomainRandomizer(self.model)

        self._joint_qpos_adr = joint_qpos_addresses(self.model)
        actuator_ids = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index): index
            for index in range(self.model.nu)
        }
        self._ctrl_index = np.array([actuator_ids[name] for name in JOINT_NAMES])
        ctrl_range = self.model.actuator_ctrlrange[self._ctrl_index]
        low_arm = {name: ctrl_range[index, 0] for index, name in enumerate(ARM_JOINT_NAMES)}
        high_arm = {name: ctrl_range[index, 1] for index, name in enumerate(ARM_JOINT_NAMES)}
        action_low = sim_frame_to_real(low_arm, ctrl_range[GRIPPER_INDEX, 0]).astype(np.float32)
        action_high = sim_frame_to_real(high_arm, ctrl_range[GRIPPER_INDEX, 1]).astype(np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            STATE_FEATURE: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(JOINT_NAMES),),
                dtype=np.float32,
            ),
            OVERHEAD_FEATURE: spaces.Box(
                low=0,
                high=255,
                shape=(image_height, image_width, 3),
                dtype=np.uint8,
            ),
            WRIST_FEATURE: spaces.Box(
                low=0,
                high=255,
                shape=(image_height, image_width, 3),
                dtype=np.uint8,
            ),
        })

        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        cube_joint = self.model.body_jntadr[cube_body]
        self._cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint])
        self._cube_dof_adr = int(self.model.jnt_dofadr[cube_joint])
        self._cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube"
        )
        self._robot_geom_ids, self._env_geom_ids = build_geom_sets(self.model)
        self._oracle = TaskSuccessOracle()
        self._scenario: EvaluationScenario | None = None
        self._miscalibration: MiscalibrationDraw | None = None
        self._domain_sample: DomainSample | None = None
        self._target_xy = np.zeros(2)
        self._step_count = 0
        self._substeps = 1
        self._last_task_state: TaskState | None = None

    @property
    def oracle(self) -> TaskSuccessOracle:
        return self._oracle

    @property
    def scenario(self) -> EvaluationScenario | None:
        return self._scenario

    def _renderer_instance(self):
        if self._renderer is None:
            height, width = self.render_hw
            self._renderer = self._renderer_factory(self.model, height=height, width=width)
        return self._renderer

    def _joint_offsets_rad(self) -> dict[str, float]:
        if self._miscalibration is None:
            return {}
        return self._miscalibration.offsets_rad(self._step_count / self._scenario.control_hz)

    def _set_robot_state(self, state_real: tuple[float, ...]) -> None:
        base_ctrl = real_action_to_sim_ctrl(np.asarray(state_real, dtype=np.float32))
        offsets = self._joint_offsets_rad()
        offset_vector = np.array([offsets.get(name, 0.0) for name in JOINT_NAMES])
        true_ctrl = np.clip(
            base_ctrl + offset_vector,
            self.model.actuator_ctrlrange[self._ctrl_index, 0],
            self.model.actuator_ctrlrange[self._ctrl_index, 1],
        )
        self.data.qpos[self._joint_qpos_adr] = true_ctrl
        self.data.ctrl[self._ctrl_index] = true_ctrl
        for actuator_id, value in zip(self._ctrl_index, true_ctrl, strict=True):
            set_actuator_activation(self.model, self.data, int(actuator_id), float(value))

    def _set_cube(self, scenario: EvaluationScenario) -> None:
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = (
            scenario.source_position_m
        )
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = (
            scenario.source_orientation_wxyz
        )
        self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None or not isinstance(options.get("scenario"), EvaluationScenario):
            raise ValueError('reset requires options={"scenario": EvaluationScenario(...)}')
        scenario = options["scenario"]
        self._scenario = scenario
        self._step_count = 0
        self._miscalibration = _miscalibration_from_scenario(scenario)
        self._domain_sample = _domain_sample_from_scenario(scenario, self._miscalibration)

        mujoco.mj_resetData(self.model, self.data)
        if self._domain_sample is None:
            self._randomizer.reset()
        else:
            self._randomizer.apply(self._domain_sample)
        if self._renderer is not None:
            reload_renderer_textures(self._renderer, self._randomizer.texture_ids)
        self._set_robot_state(scenario.initial_robot_state_real)
        self._set_cube(scenario)
        self._target_xy[:] = scenario.target_position_m[:2]
        place_paper_target_marker(
            self.model,
            tuple(self._target_xy),
            0.0,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*self._target_xy),
            alpha=1.0,
        )
        if self._domain_sample is not None:
            self._randomizer.tint_episode_markers()
        self._substeps = max(
            1,
            round((1.0 / scenario.control_hz) / float(self.model.opt.timestep)),
        )
        self._oracle.reset()
        mujoco.mj_forward(self.model, self.data)
        self._last_task_state = self._task_state()
        observation = self._observation()
        return observation, self._info(self._last_task_state)

    def _render_camera(self, camera: str) -> np.ndarray:
        renderer = self._renderer_instance()
        renderer.update_scene(self.data, camera=camera)
        image = resize_and_center_crop(renderer.render(), *self.image_hw)
        if self._domain_sample is not None:
            image = self._randomizer.postprocess(image)
        return image

    def _observation(self) -> dict[str, np.ndarray]:
        # Preserve the existing interactive runner's render order so deterministic
        # per-frame domain noise maps to the same camera on both code paths.
        wrist = self._render_camera("wrist_camera")
        overhead = self._render_camera("overhead_camera")
        return {
            STATE_FEATURE: sim_state_to_real(
                self.data.qpos[self._joint_qpos_adr], self._joint_offsets_rad()
            ),
            OVERHEAD_FEATURE: overhead,
            WRIST_FEATURE: wrist,
        }

    def _contact_facts(self) -> tuple[bool, bool]:
        robot_cube_contact = False
        fixed_jaw_contact = False
        moving_jaw_contact = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom[0]), int(contact.geom[1])
            if self._cube_geom_id not in (first, second):
                continue
            other = second if first == self._cube_geom_id else first
            if other not in self._robot_geom_ids:
                continue
            robot_cube_contact = True
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            fixed_jaw_contact |= name.startswith(_JAW_PREFIXES[0])
            moving_jaw_contact |= name.startswith(_JAW_PREFIXES[1])
        return robot_cube_contact, fixed_jaw_contact and moving_jaw_contact

    def _task_state(self) -> TaskState:
        cube_position = self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3]
        cube_velocity = self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6]
        robot_cube_contact, grasped = self._contact_facts()
        unexpected_collision = any(
            is_unexpected(first, second)
            for first, second in scan_contacts(
                self.model,
                self.data,
                self._robot_geom_ids,
                self._env_geom_ids,
            )
        )
        x, y, z = cube_position
        out_of_bounds = (
            math.hypot(float(x), float(y)) > _MAX_CUBE_RADIUS_M
            or not (_MIN_CUBE_HEIGHT_M < float(z) < _MAX_CUBE_HEIGHT_M)
        )
        return TaskState(
            cube_position_m=tuple(float(value) for value in cube_position),
            cube_linear_velocity_m_s=tuple(float(value) for value in cube_velocity[:3]),
            cube_angular_velocity_rad_s=tuple(float(value) for value in cube_velocity[3:]),
            target_xy_m=tuple(float(value) for value in self._target_xy),
            robot_cube_contact=robot_cube_contact,
            grasped=grasped,
            gripper_open=(
                float(self.data.qpos[self._joint_qpos_adr[GRIPPER_INDEX]])
                > _GRIPPER_OPENING_THRESHOLD_RAD
            ),
            unexpected_collision=unexpected_collision,
            out_of_bounds=out_of_bounds,
        )

    def _info(self, task_state: TaskState) -> dict[str, Any]:
        return {
            "scenario_id": self._scenario.scenario_id,
            "task_state": asdict(task_state),
            "milestones": asdict(self._oracle.milestones),
            "success": self._oracle.success,
            "control_steps": self._step_count,
        }

    def step(self, action):
        if self._scenario is None:
            raise RuntimeError("reset must be called before step")
        base_ctrl = real_action_to_sim_ctrl(action)
        offsets = self._joint_offsets_rad()
        offset_vector = np.array([offsets.get(name, 0.0) for name in JOINT_NAMES])
        self.data.ctrl[self._ctrl_index] = np.clip(
            base_ctrl + offset_vector,
            self.model.actuator_ctrlrange[self._ctrl_index, 0],
            self.model.actuator_ctrlrange[self._ctrl_index, 1],
        )
        mujoco.mj_step(self.model, self.data, nstep=self._substeps)
        self._step_count += 1

        self._last_task_state = self._task_state()
        success = self._oracle.update(
            self._last_task_state,
            step_duration_s=1.0 / self._scenario.control_hz,
        )
        terminated = (
            success
            or self._last_task_state.unexpected_collision
            or self._last_task_state.out_of_bounds
        )
        truncated = not terminated and self._step_count >= self._scenario.max_steps
        return (
            self._observation(),
            1.0 if success else 0.0,
            terminated,
            truncated,
            self._info(self._last_task_state),
        )

    def episode_result(self) -> EpisodeResult:
        if self._scenario is None or self._last_task_state is None:
            raise RuntimeError("no episode has been run")
        timed_out = self._step_count >= self._scenario.max_steps and not self._oracle.success
        return EpisodeResult(
            scenario_id=self._scenario.scenario_id,
            group=self._scenario.group,
            workspace_region=self._scenario.workspace_region,
            success=self._oracle.success,
            milestones=self._oracle.milestones,
            failures=self._oracle.failure_flags(timed_out=timed_out),
            final_xy_error_m=self._last_task_state.xy_error_m,
            control_steps=self._step_count,
            simulated_time_s=self._step_count / self._scenario.control_hz,
            time_to_success_s=self._oracle.success_time_s,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


ObservationCallback = Callable[[int, PolicyObservation], None]


def evaluate_policy_episode(
    env: PolicySimEnv,
    controller: PolicyController,
    scenario: EvaluationScenario,
    *,
    observation_callback: ObservationCallback | None = None,
) -> EpisodeResult:
    """Run one controller episode, resetting all controller state first."""
    controller.reset()
    observation, _ = env.reset(options={"scenario": scenario})
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        if observation_callback is not None:
            observation_callback(step, observation)
        action = controller.act(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        step += 1
        if getattr(controller, "failure", None) is not None:
            break
    result = env.episode_result()
    controller_failure = getattr(controller, "failure", None)
    if controller_failure is not None:
        result = replace(result, controller_failure=asdict(controller_failure))
    return result
