# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Observation-driven incremental analytic pick-and-place controller."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.cube_detection import CubeTracker
from pick_and_place.episodes import (
    Episode,
    _preflight,
    is_unexpected,
    prepare_episode,
    sample_hunt_pose,
    set_joint,
)
from pick_and_place.follower import (
    JOINT_NAMES,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.overhead_localization import OverheadLocalizer
from pick_and_place.paper_detection import PaperTarget
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    ControllerFailure,
    PolicyObservation,
)
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    DescentPhase,
    GraspPhase,
    LiftPhase,
    RecoveryLiftPhase,
    Trajectory,
    _shortest_delta,
    fold_cube_yaw,
    grasp_candidates,
    replan_remaining_candidates,
)
from pick_and_place.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DescentServoConvergence,
    DescentServoRetryState,
)

PlanEpisode = Callable[..., Episode]
WristLocalization = Callable[
    [np.ndarray, dict[str, float], float, CubePose], CubePose | None
]
ReplanCandidates = Callable[..., Iterable[Trajectory]]
TrajectoryPreflight = Callable[[Episode, Trajectory], bool]


class WristCameraLocalizer:
    """Map wrist RGB into world poses through fixed nominal calibration.

    The model is a controller-owned kinematic mirror. Reported joints pose its
    nominal wrist camera each tick, so physical or simulated camera-mount
    perturbations remain hidden in the observation image instead of leaking
    through extrinsics supplied by the environment.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_matrix: np.ndarray,
        *,
        camera_name: str = "wrist_camera",
        tracker_factory: Callable[[], Any] | None = None,
        free_grasp: bool = False,
    ) -> None:
        matrix = np.asarray(camera_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must have finite shape (3, 3)")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"model has no camera named {camera_name!r}")
        self.model = model
        self.camera_matrix = matrix.copy()
        self.camera_id = camera_id
        self.free_grasp = free_grasp
        self._tracker_factory = tracker_factory or (lambda: CubeTracker(smooth=0.95))
        self._shadow = mujoco.MjData(model)
        self.reset()

    def reset(self) -> None:
        """Clear tracking history between controller episodes."""
        self._tracker = self._tracker_factory()

    def __call__(
        self,
        image: np.ndarray,
        reported_joints: dict[str, float],
        reported_gripper: float,
        prior: CubePose,
    ) -> CubePose | None:
        del prior
        for name, value in reported_joints.items():
            set_joint(self.model, self._shadow, name, value)
        set_joint(self.model, self._shadow, "gripper", reported_gripper)
        mujoco.mj_forward(self.model, self._shadow)
        estimate = self._tracker.update_frame(
            image,
            self.camera_matrix,
            self._shadow.cam_xpos[self.camera_id],
            self._shadow.cam_xmat[self.camera_id].reshape(3, 3),
            dist=None,
        )
        if estimate is None:
            return None
        roll, pitch, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
        return CubePose(
            x=float(estimate.position[0]),
            y=float(estimate.position[1]),
            z=CUBE_HALF_SIZE,
            roll=float(roll) if self.free_grasp else 0.0,
            pitch=float(pitch) if self.free_grasp else 0.0,
            yaw=float(yaw),
        )


class ScriptedPolicyState(str, Enum):
    """Externally inspectable phase of the scripted controller."""

    LOCALIZING = "localizing"
    READY = "ready"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ScriptedPolicy:
    """Localize, plan, servo, and execute from deployable observations.

    Each call to :meth:`act` is one fixed-rate control tick. The environment
    supplies only RGB images and reported hardware-frame joints; camera
    calibration, wrist localization, planning, preflight, and all execution
    state remain fixed controller configuration.
    """

    def __init__(
        self,
        localizer: OverheadLocalizer,
        workspace_corners_world: np.ndarray,
        *,
        target_color: str = "black",
        max_localization_steps: int = 60,
        localization_steps_per_search: int = 15,
        rng_seed: int = 0,
        control_hz: float = 30.0,
        wrist_localizer: WristLocalization | None = None,
        plan_episode: PlanEpisode = prepare_episode,
        replan_candidates: ReplanCandidates = replan_remaining_candidates,
        trajectory_preflight: TrajectoryPreflight | None = None,
    ) -> None:
        if target_color not in {"black", "white"}:
            raise ValueError("target_color must be 'black' or 'white'")
        if max_localization_steps < 1:
            raise ValueError("max_localization_steps must be at least 1")
        if localization_steps_per_search < 1:
            raise ValueError("localization_steps_per_search must be at least 1")
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be positive and finite")
        corners = np.asarray(workspace_corners_world, dtype=float)
        if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
            raise ValueError("workspace_corners_world must have finite shape (4, 3)")

        self.localizer = localizer
        self.workspace_corners_world = corners.copy()
        self.target_color = target_color
        self.max_localization_steps = max_localization_steps
        self.localization_steps_per_search = localization_steps_per_search
        self.rng_seed = rng_seed
        self.control_hz = float(control_hz)
        self.wrist_localizer = wrist_localizer
        self._plan_episode = plan_episode
        self._replan_candidates = replan_candidates
        self._trajectory_preflight = trajectory_preflight or self._default_preflight
        self.reset()

    def reset(self) -> None:
        """Forget detections, planning state, random draws, and failures."""
        self.localizer.reset()
        reset_wrist = getattr(self.wrist_localizer, "reset", None)
        if reset_wrist is not None:
            reset_wrist()
        self.state = ScriptedPolicyState.LOCALIZING
        self.cube_pose: CubePose | None = None
        self.drop_target: PaperTarget | None = None
        self.episode: Episode | None = None
        self.failure: ControllerFailure | None = None
        self._localization_steps = 0
        self._search_target: np.ndarray | None = None
        self._rng = np.random.default_rng(self.rng_seed)
        self._trajectory: Trajectory | None = None
        self._phase_elapsed = 0.0
        self._phase_complete = False
        self._dynamic_source: CubePose | None = None
        self._dynamic_grasp = None
        self._descent_convergence: DescentServoConvergence | None = None
        self._descent_retry: DescentServoRetryState | None = None
        self._descent_saw_detection = False

    @property
    def terminal(self) -> bool:
        return self.state in (ScriptedPolicyState.SUCCEEDED, ScriptedPolicyState.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.state is ScriptedPolicyState.SUCCEEDED

    @staticmethod
    def _hold_action(observation: PolicyObservation) -> np.ndarray:
        if STATE_FEATURE not in observation:
            raise KeyError(f"observation is missing {STATE_FEATURE!r}; cannot issue a safe hold")
        action = np.asarray(observation[STATE_FEATURE], dtype=np.float32).reshape(-1)
        expected_shape = (len(JOINT_NAMES),)
        if action.shape != expected_shape:
            raise ValueError(
                f"{STATE_FEATURE} must have shape {expected_shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{STATE_FEATURE} must contain only finite values")
        return action.copy()

    @staticmethod
    def _image(observation: PolicyObservation, feature: str) -> np.ndarray:
        if feature not in observation:
            raise KeyError(f"observation is missing {feature!r}")
        image = np.asarray(observation[feature])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{feature} must have shape (height, width, 3), got {image.shape}")
        return image

    def _fail(self, code: str, message: str) -> None:
        self.state = ScriptedPolicyState.FAILED
        self.failure = ControllerFailure(code=code, message=message)

    @staticmethod
    def _default_preflight(episode: Episode, trajectory: Trajectory) -> bool:
        events = _preflight(
            episode.model,
            trajectory,
            episode.actuator_id,
            episode.robot_geom_ids,
            episode.env_geom_ids,
        )
        return not any(is_unexpected(name1, name2) for _, name1, name2 in events)

    def _search_action(self) -> np.ndarray:
        arm_joints, gripper = sample_hunt_pose(self._rng)
        return sim_frame_to_real(arm_joints, gripper).astype(np.float32)

    def _plan(self, reported_joints: np.ndarray) -> None:
        assert self.cube_pose is not None
        assert self.drop_target is not None
        start_joints, start_gripper = real_frame_to_sim(reported_joints)
        target_xy = np.asarray(self.drop_target.xy, dtype=float).reshape(-1)
        if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
            raise ValueError(
                "localized drop target xy must have finite shape (2,), "
                f"got {target_xy.shape}"
            )
        target = CubePose(
            x=float(target_xy[0]),
            y=float(target_xy[1]),
            z=CUBE_HALF_SIZE,
        )
        self.episode = self._plan_episode(
            self._rng,
            self.cube_pose,
            target,
            start_joints=start_joints,
            start_gripper=start_gripper,
            max_attempts=1,
            include_environment=True,
        )

    def _begin_execution(self) -> None:
        assert self.episode is not None
        self._trajectory = self.episode.trajectory
        if not self._trajectory.phases:
            self.state = ScriptedPolicyState.SUCCEEDED
            return
        self._dynamic_source = self.cube_pose
        self._dynamic_grasp = self._trajectory.grasp
        self._start_phase()
        self.state = ScriptedPolicyState.EXECUTING

    def _start_phase(self) -> None:
        self._phase_elapsed = 0.0
        self._phase_complete = False
        assert self._trajectory is not None and self._trajectory.phases
        if isinstance(self._trajectory.phases[0], DescentPhase):
            self._descent_convergence = DescentServoConvergence()
            self._descent_retry = DescentServoRetryState()
            self._descent_saw_detection = False
        else:
            self._descent_convergence = None
            self._descent_retry = None
            self._descent_saw_detection = False

    def _update_descent(
        self,
        phase: DescentPhase,
        wrist: np.ndarray,
        reported_joints: np.ndarray,
    ) -> DescentPhase:
        if self.wrist_localizer is None:
            return phase
        assert self._dynamic_source is not None
        joints, gripper = real_frame_to_sim(reported_joints)
        estimate = self.wrist_localizer(wrist, joints, gripper, self._dynamic_source)
        if estimate is None:
            return phase

        folded_yaw = fold_cube_yaw(self._dynamic_source.yaw, estimate.yaw)
        estimate = dataclasses.replace(estimate, yaw=folded_yaw)
        alpha = 0.1
        source = dataclasses.replace(
            estimate,
            x=self._dynamic_source.x * (1.0 - alpha) + estimate.x * alpha,
            y=self._dynamic_source.y * (1.0 - alpha) + estimate.y * alpha,
            yaw=self._dynamic_source.yaw
            + _shortest_delta(self._dynamic_source.yaw, estimate.yaw) * alpha,
        )
        if phase.grasp.face != "free":
            updated_grasp = next(
                (
                    grasp
                    for grasp in grasp_candidates(self.episode.kinematics, source)
                    if grasp.face == phase.grasp.face and grasp.elbow == phase.grasp.elbow
                ),
                None,
            )
            if updated_grasp is not None:
                phase = dataclasses.replace(phase, grasp=updated_grasp)
        self._dynamic_source = source
        self._descent_saw_detection = True
        assert self._descent_convergence is not None
        self._descent_convergence.observe(source)
        return phase

    def _descent_finished(self, phase: DescentPhase, phase_t: float) -> bool:
        if self.wrist_localizer is None:
            return phase_t >= phase.duration
        assert self._descent_retry is not None
        assert self._descent_convergence is not None
        retry = self._descent_retry
        if retry.is_backing_up():
            if retry.backup_complete(self._phase_elapsed):
                retry.finish_backup()
                self._descent_convergence = DescentServoConvergence()
                self._descent_saw_detection = False
                self._phase_elapsed = 0.0
            return False
        if (
            not self._descent_saw_detection
            and phase_t >= phase.duration
            and retry.can_retry()
        ):
            retry.start_backup(self._phase_elapsed)
            return False
        if self._phase_elapsed >= max(phase.duration, DESCENT_SERVO_MAX_DURATION):
            detail = "before settling" if self._descent_saw_detection else "without a detection"
            self._fail(
                "descent_servo_timeout",
                f"wrist visual servo reached its duration cap {detail}",
            )
            return False
        return phase_t >= phase.duration and self._descent_convergence.is_stable()

    def _advance_locked_section(self, completed: str) -> bool:
        assert self._trajectory is not None
        phases = self._trajectory.phases
        next_name = phases[1].name if len(phases) > 1 else None
        locked_pair = (
            (completed == "approach" and next_name == "descent")
            or (completed == "grasp" and next_name in ("lift", "recovery_lift"))
            or (completed == "carry" and next_name == "drop_descent")
            or (completed == "drop_descent" and next_name == "release")
        )
        if not locked_pair:
            return False
        self._trajectory = dataclasses.replace(self._trajectory, phases=phases[1:])
        self._start_phase()
        return True

    def _rebuild_after_descent(self, phase: DescentPhase) -> None:
        assert self.episode is not None
        assert self._trajectory is not None
        assert self._dynamic_source is not None
        phases = self._trajectory.phases
        if phase.grasp.face == "free":
            self._dynamic_grasp = phase.grasp
        else:
            self._dynamic_grasp = next(
                (
                    grasp
                    for grasp in grasp_candidates(self.episode.kinematics, self._dynamic_source)
                    if grasp.face == phase.face and grasp.elbow == phase.elbow
                ),
                phase.grasp,
            )
        lift_class = RecoveryLiftPhase if isinstance(phases[2], RecoveryLiftPhase) else LiftPhase
        grasp_phase = GraspPhase(self._dynamic_grasp.grasp_joints, start_gripper=GRIPPER_OPEN)
        lift_phase = lift_class(
            self.episode.kinematics,
            self._dynamic_grasp.grasp_joints,
            self._dynamic_grasp.lift_joints,
        )
        self._trajectory = dataclasses.replace(
            self._trajectory,
            phases=(grasp_phase, lift_phase, *phases[3:]),
            grasp=self._dynamic_grasp,
        )
        self._start_phase()

    def _complete_phase(self, reported_joints: np.ndarray) -> None:
        assert self.episode is not None
        assert self._trajectory is not None and self._trajectory.phases
        phase = self._trajectory.phases[0]
        completed = phase.name
        if isinstance(phase, DescentPhase):
            self._rebuild_after_descent(phase)
            return
        if self._advance_locked_section(completed):
            return
        if len(self._trajectory.phases) <= 1:
            self.state = ScriptedPolicyState.SUCCEEDED
            return

        measured_joints, measured_gripper = real_frame_to_sim(reported_joints)
        assert self._dynamic_source is not None
        free_grasp = self._dynamic_grasp is not None and self._dynamic_grasp.face == "free"
        for candidate in self._replan_candidates(
            self.episode.kinematics,
            measured_joints,
            measured_gripper,
            completed,
            self._dynamic_source,
            self.episode.target,
            self._dynamic_grasp,
            self.episode.end_joints,
            self.episode.end_gripper,
            free_grasp=free_grasp,
        ):
            if self._trajectory_preflight(self.episode, candidate):
                self._trajectory = candidate
                self._start_phase()
                return
        self._fail(
            "replanning_error",
            f"no collision-free remaining trajectory after {completed}",
        )

    def _execute(self, hold: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        if self._phase_complete:
            try:
                self._complete_phase(hold)
            except Exception as exc:
                self._fail("replanning_error", str(exc))
                return hold
            if self.terminal:
                return hold

        assert self._trajectory is not None and self._trajectory.phases
        phase = self._trajectory.phases[0]
        phase_t = self._phase_elapsed
        try:
            if isinstance(phase, DescentPhase):
                phase = self._update_descent(phase, wrist, hold)
                self._trajectory = dataclasses.replace(
                    self._trajectory,
                    phases=(phase, *self._trajectory.phases[1:]),
                )
                assert self._descent_retry is not None or self.wrist_localizer is None
                if self._descent_retry is not None:
                    phase_t = self._descent_retry.command_phase_t(
                        self._phase_elapsed, phase.duration
                    )
            frame = phase.evaluate(min(phase_t, phase.duration))
        except Exception as exc:
            self._fail("execution_error", str(exc))
            return hold

        if isinstance(phase, DescentPhase):
            self._phase_complete = self._descent_finished(phase, phase_t)
        else:
            self._phase_complete = phase_t >= phase.duration
        if self.state is ScriptedPolicyState.FAILED:
            return hold
        self._phase_elapsed += 1.0 / self.control_hz
        return sim_frame_to_real(frame.joints, frame.gripper).astype(np.float32)

    def act(self, observation: PolicyObservation) -> np.ndarray:
        hold = self._hold_action(observation)
        if self.terminal:
            return hold

        try:
            wrist = self._image(observation, WRIST_FEATURE)
        except Exception as exc:
            code = (
                "localization_error"
                if self.state is ScriptedPolicyState.LOCALIZING
                else "observation_error"
            )
            self._fail(code, str(exc))
            return hold

        if self.state is ScriptedPolicyState.READY:
            try:
                self._begin_execution()
            except Exception as exc:
                self._fail("execution_error", str(exc))
                return hold
            if self.terminal:
                return hold
        if self.state is ScriptedPolicyState.EXECUTING:
            return self._execute(hold, wrist)

        try:
            overhead = self._image(observation, OVERHEAD_FEATURE)
            if self.cube_pose is None:
                self.cube_pose = self.localizer.localize_cube(overhead)
            if self.drop_target is None:
                self.drop_target = self.localizer.localize_drop_target(
                    overhead,
                    target_color=self.target_color,
                    workspace_corners_world=self.workspace_corners_world,
                )
        except Exception as exc:
            self._fail("localization_error", str(exc))
            return hold

        self._localization_steps += 1
        if self.cube_pose is not None and self.drop_target is not None:
            try:
                self._plan(hold)
            except Exception as exc:
                self._fail("planning_error", str(exc))
                return hold
            self.state = ScriptedPolicyState.READY
            return hold

        if self._localization_steps >= self.max_localization_steps:
            missing = []
            if self.cube_pose is None:
                missing.append("cube")
            if self.drop_target is None:
                missing.append("drop target")
            self._fail(
                "localization_timeout",
                f"could not localize {' and '.join(missing)} in "
                f"{self.max_localization_steps} control steps",
            )
            return hold

        if self._localization_steps % self.localization_steps_per_search == 0:
            self._search_target = self._search_action()
        if self._search_target is not None:
            return self._search_target.copy()
        return hold
