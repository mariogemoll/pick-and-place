# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Observation-driven incremental analytic pick-and-place controller.

The expert. It localizes, plans, servos the descent onto what it sees and
replans the rest from measured state — and it does all of that from images and
reported joints, which is the whole point: it is drivable from exactly what a
learned policy is drivable from.

It *consumes* sightings rather than producing them. Detection, preflight and
episode preparation are injected, because each needs a capability the controller
has no other use for — a tag detector, a compiled scene, live physics — and
reaching for any of them would make the expert a thing that can only run where
those happen to be.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Protocol

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import real_frame_to_sim, sim_frame_to_real
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.scripted.episode_sampling import (
    CARRY_HUNT_PAN_SCALE,
    sample_hunt_pose,
    sample_target,
)
from pick_and_place.scripted.grasp import fold_cube_yaw, grasp_candidates
from pick_and_place.scripted.motion import shortest_delta, smoothstep
from pick_and_place.scripted.replan import replan_remaining_candidates
from pick_and_place.scripted.trajectory import (
    DescentPhase,
    GraspPhase,
    LiftPhase,
    RecoveryLiftPhase,
    Trajectory,
)
from pick_and_place.scripted.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DescentServoConvergence,
    DescentServoRetryState,
)
from pick_and_place.spec.controller import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    ControllerFailure,
    PolicyObservation,
)
from pick_and_place.spec.drop_zone import PaperTarget
from pick_and_place.spec.robot import GRIPPER_GRASP, GRIPPER_OPEN, JOINT_NAMES, NEUTRAL_ARM_JOINTS
from pick_and_place.spec.workspace import CUBE_HALF_SIZE


class PlannedEpisode(Protocol):
    """The plan-level facts the controller needs from a prepared episode.

    A structural type rather than an import: preparing an episode means
    compiling a scene and vetting a trajectory against live physics, and a
    controller that could reach for either would only run where they exist.
    """

    trajectory: Trajectory
    kinematics: So101Kinematics
    target: CubePose
    end_joints: dict[str, float]
    end_gripper: float


class SceneLocalizer(Protocol):
    """What the controller needs from whatever looks at the overhead image.

    Declared here, by the consumer, because it is a contract about *readings* —
    where the cube is, where the plate is — and says nothing about detectors.
    """

    def reset(self) -> None:
        """Forget detections from the previous episode."""
        ...

    def localize_cube(
        self, frame_rgb: np.ndarray, *, free_grasp: bool = False
    ) -> CubePose | None:
        """Where the cube is, from one overhead frame."""
        ...

    def localize_drop_target(
        self,
        frame_rgb: np.ndarray,
        *,
        target_color: str,
        workspace_corners_world: np.ndarray,
    ) -> PaperTarget | None:
        """Where the drop plate is, from one overhead frame."""
        ...


PlanEpisode = Callable[..., PlannedEpisode]
TargetSampler = Callable[[np.random.Generator], CubePose]
WristLocalization = Callable[
    [np.ndarray, dict[str, float], float, CubePose], CubePose | None
]
ReplanCandidates = Callable[..., Iterable[Trajectory]]
TrajectoryPreflight = Callable[[PlannedEpisode, Trajectory], bool]


class ScriptedPolicyState(str, Enum):
    """Externally inspectable phase of the scripted controller."""

    LOCALIZING = "localizing"
    READY = "ready"
    EXECUTING = "executing"
    FINDING_PLATE = "finding_plate"
    REVEALING_PLACEMENT = "revealing_placement"
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
        localizer: SceneLocalizer,
        workspace_corners_world: np.ndarray,
        plan_episode: PlanEpisode,
        trajectory_preflight: TrajectoryPreflight,
        *,
        target_color: str = "black",
        max_localization_steps: int = 60,
        localization_steps_per_search: int = 15,
        max_search_poses: int = 8,
        max_grasp_retries: int = 1,
        planning_max_attempts: int = 40,
        planning_verbose: bool = False,
        rng_seed: int = 0,
        control_hz: float = 30.0,
        wrist_localizer: WristLocalization | None = None,
        replan_candidates: ReplanCandidates = replan_remaining_candidates,
        target_sampler: TargetSampler | None = None,
        drop_target_xy: tuple[float, float] | None = None,
        cache_drop_target_early: bool = False,
        free_grasp: bool = False,
    ) -> None:
        if target_color not in {"black", "white"}:
            raise ValueError("target_color must be 'black' or 'white'")
        if max_localization_steps < 1:
            raise ValueError("max_localization_steps must be at least 1")
        if localization_steps_per_search < 1:
            raise ValueError("localization_steps_per_search must be at least 1")
        if max_search_poses < 1:
            raise ValueError("max_search_poses must be at least 1")
        if max_grasp_retries < 0:
            raise ValueError("max_grasp_retries must be nonnegative")
        if planning_max_attempts < 1:
            raise ValueError("planning_max_attempts must be at least 1")
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
        self.max_search_poses = max_search_poses
        self.max_grasp_retries = max_grasp_retries
        self.planning_max_attempts = planning_max_attempts
        self.planning_verbose = planning_verbose
        self.rng_seed = rng_seed
        self.control_hz = float(control_hz)
        self.wrist_localizer = wrist_localizer
        self._plan_episode = plan_episode
        self._replan_candidates = replan_candidates
        self._trajectory_preflight = trajectory_preflight
        self.target_sampler = target_sampler
        # Pinning the drop target says "this run does not localize the plate":
        # the pose is known, so no overhead search runs and planning takes the
        # same fixed-target path a detection would have produced. Survives
        # reset() so one controller can be re-pinned per episode.
        self.drop_target_xy = drop_target_xy
        self.cache_drop_target_early = cache_drop_target_early
        self.free_grasp = free_grasp
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
        self.episode: PlannedEpisode | None = None
        self.failure: ControllerFailure | None = None
        self._localization_steps = 0
        self._search_target: np.ndarray | None = None
        self._search_start: np.ndarray | None = None
        self._search_progress = 0
        self._search_poses = 0
        self._grasp_retries = 0
        self._rng = np.random.default_rng(self.rng_seed)
        self._trajectory: Trajectory | None = None
        self._phase_elapsed = 0.0
        self._phase_complete = False
        self._dynamic_source: CubePose | None = None
        self._dynamic_grasp = None
        self._descent_convergence: DescentServoConvergence | None = None
        self._descent_retry: DescentServoRetryState | None = None
        self._descent_saw_detection = False

    def close(self) -> None:
        """Release controller-owned background workers, if any."""
        close_localizer = getattr(self.localizer, "close", None)
        if close_localizer is not None:
            close_localizer()
        close_wrist = getattr(self.wrist_localizer, "close", None)
        if close_wrist is not None:
            close_wrist()

    @property
    def terminal(self) -> bool:
        return self.state in (ScriptedPolicyState.SUCCEEDED, ScriptedPolicyState.FAILED)

    @property
    def placement_target_xy(self) -> tuple[float, float] | None:
        """Where this episode is placing, however the target was obtained.

        A pinned target and a localized plate are different fields, and
        :meth:`_plan` prefers the pinned one. Anything downstream that scores
        or records the placement wants that same answer rather than reaching
        for whichever field it happens to know about — a chained run pins the
        target and never localizes a plate, so reading ``drop_target`` alone
        reports every placement as a failure.
        """
        if self.drop_target_xy is not None:
            return (float(self.drop_target_xy[0]), float(self.drop_target_xy[1]))
        if self.drop_target is not None:
            return (float(self.drop_target.xy[0]), float(self.drop_target.xy[1]))
        return None

    @property
    def succeeded(self) -> bool:
        return self.state is ScriptedPolicyState.SUCCEEDED

    @property
    def phase_name(self) -> str | None:
        """Current trajectory phase, for physical verification and diagnostics."""
        if self.state is ScriptedPolicyState.LOCALIZING:
            return "find_cube"
        if self.state is ScriptedPolicyState.FINDING_PLATE:
            return "find_plate"
        if self.state is ScriptedPolicyState.REVEALING_PLACEMENT:
            return "reveal_placement"
        if self._trajectory is None or not self._trajectory.phases:
            return None
        return self._trajectory.phases[0].name

    def begin_execution(self) -> None:
        """Enter trajectory execution after external localization/planning preflight."""
        if self.state is not ScriptedPolicyState.READY:
            raise RuntimeError(f"policy is not ready for execution: {self.state.value}")
        self._begin_execution()

    def report_pickup_result(self, succeeded: bool) -> None:
        """Continue after a verified lift or locally retry a missed grasp."""
        if succeeded:
            return
        if self.state is not ScriptedPolicyState.FINDING_PLATE:
            raise RuntimeError(
                f"pickup result is only valid after lift, not while {self.state.value}"
            )
        if self._grasp_retries >= self.max_grasp_retries:
            self._fail(
                "grasp_retry_exhausted",
                f"grasp still missed after {self.max_grasp_retries} retries",
            )
            return
        self._grasp_retries += 1
        self.localizer.reset()
        self.state = ScriptedPolicyState.LOCALIZING
        self.cube_pose = None
        if not self.cache_drop_target_early:
            self.drop_target = None
        self.episode = None
        self._trajectory = None
        self._dynamic_source = None
        self._dynamic_grasp = None
        self._localization_steps = 0
        self._search_start = None
        self._search_target = None
        self._search_progress = 0
        self._search_poses = 0

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

    def _begin_search_motion(self, hold: np.ndarray, *, carrying: bool) -> None:
        if self._search_poses >= self.max_search_poses:
            self._fail(
                "search_budget_exhausted",
                f"could not make the object visible in {self.max_search_poses} search poses",
            )
            return
        if carrying:
            arm_joints, gripper = real_frame_to_sim(hold)
            arm_joints["shoulder_pan"] = NEUTRAL_ARM_JOINTS["shoulder_pan"] + self._rng.uniform(
                -CARRY_HUNT_PAN_SCALE, CARRY_HUNT_PAN_SCALE
            )
            gripper = GRIPPER_GRASP
        else:
            arm_joints, gripper = sample_hunt_pose(self._rng)
        self._search_start = hold.astype(np.float32, copy=True)
        self._search_target = sim_frame_to_real(arm_joints, gripper).astype(np.float32)
        self._search_progress = 0
        self._search_poses += 1

    def _search_action(self, hold: np.ndarray, *, carrying: bool) -> np.ndarray:
        if self._search_target is None:
            self._begin_search_motion(hold, carrying=carrying)
            if self.terminal:
                return hold
        assert self._search_start is not None and self._search_target is not None
        self._search_progress += 1
        alpha = smoothstep(
            min(1.0, self._search_progress / self.localization_steps_per_search)
        )
        action = self._search_start + (self._search_target - self._search_start) * alpha
        if self._search_progress >= self.localization_steps_per_search:
            self._search_start = None
            self._search_target = None
            self._search_progress = 0
        return action.astype(np.float32)

    def _plan(self, reported_joints: np.ndarray) -> None:
        assert self.cube_pose is not None
        start_joints, start_gripper = real_frame_to_sim(reported_joints)
        target = None
        planning_target_sampler = self.target_sampler
        if self.drop_target_xy is not None or self.drop_target is not None:
            if self.drop_target_xy is not None:
                target_xy = np.asarray(self.drop_target_xy, dtype=float).reshape(-1)
            else:
                assert self.drop_target is not None
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
        elif self.target_sampler is None:
            # Pickup geometry does not depend on the eventual drop target. Use a
            # disposable valid target to obtain a vetted approach/grasp/lift;
            # carry is rebuilt from a fresh post-lift plate sighting.
            planning_target_sampler = sample_target
        self.episode = self._plan_episode(
            self._rng,
            self.cube_pose,
            target,
            start_joints=start_joints,
            start_gripper=start_gripper,
            max_attempts=self.planning_max_attempts,
            verbose=self.planning_verbose,
            include_environment=True,
            free_grasp=self.free_grasp,
            target_sampler=planning_target_sampler,
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
            + shortest_delta(self._dynamic_source.yaw, estimate.yaw) * alpha,
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
        if (
            completed in ("lift", "recovery_lift")
            and self.drop_target_xy is None
            and self.target_sampler is None
        ):
            self.state = ScriptedPolicyState.FINDING_PLATE
            if not self.cache_drop_target_early:
                self.drop_target = None
            self._search_start = None
            self._search_target = None
            self._search_progress = 0
            self._search_poses = 0
            return
        if len(self._trajectory.phases) <= 1:
            if completed == "retreat":
                self.state = ScriptedPolicyState.REVEALING_PLACEMENT
                self._search_start = None
                self._search_target = None
                self._search_progress = 0
                self._search_poses = 0
            else:
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
            if self.state is not ScriptedPolicyState.EXECUTING:
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

        if (
            self.cache_drop_target_early
            and self.drop_target is None
            and self.drop_target_xy is None
        ):
            try:
                overhead = self._image(observation, OVERHEAD_FEATURE)
                self.drop_target = self.localizer.localize_drop_target(
                    overhead,
                    target_color=self.target_color,
                    workspace_corners_world=self.workspace_corners_world,
                )
            except Exception as exc:
                self._fail("localization_error", str(exc))
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

        if self.state is ScriptedPolicyState.FINDING_PLATE:
            try:
                if self.drop_target is None:
                    overhead = self._image(observation, OVERHEAD_FEATURE)
                    self.drop_target = self.localizer.localize_drop_target(
                        overhead,
                        target_color=self.target_color,
                        workspace_corners_world=self.workspace_corners_world,
                    )
            except Exception as exc:
                self._fail("localization_error", str(exc))
                return hold
            if self.drop_target is None:
                return self._search_action(hold, carrying=True)
            target = CubePose(*self.drop_target.xy, CUBE_HALF_SIZE)
            assert self.episode is not None
            self.episode.target = target
            measured_joints, measured_gripper = real_frame_to_sim(hold)
            assert self._dynamic_source is not None
            for candidate in self._replan_candidates(
                self.episode.kinematics,
                measured_joints,
                measured_gripper,
                "lift",
                self._dynamic_source,
                target,
                self._dynamic_grasp,
                self.episode.end_joints,
                self.episode.end_gripper,
                free_grasp=False,
            ):
                if self._trajectory_preflight(self.episode, candidate):
                    self._trajectory = candidate
                    self._start_phase()
                    self.state = ScriptedPolicyState.EXECUTING
                    return hold
            self._fail("replanning_error", "no collision-free carry after finding plate")
            return hold

        if self.state is ScriptedPolicyState.REVEALING_PLACEMENT:
            try:
                overhead = self._image(observation, OVERHEAD_FEATURE)
                visible_cube = self.localizer.localize_cube(overhead) is not None
            except Exception as exc:
                self._fail("localization_error", str(exc))
                return hold
            if visible_cube:
                self.state = ScriptedPolicyState.SUCCEEDED
                return hold
            return self._search_action(hold, carrying=False)

        try:
            overhead = self._image(observation, OVERHEAD_FEATURE)
            if self.cube_pose is None:
                if self.free_grasp:
                    self.cube_pose = self.localizer.localize_cube(overhead, free_grasp=True)
                else:
                    self.cube_pose = self.localizer.localize_cube(overhead)
        except Exception as exc:
            self._fail("localization_error", str(exc))
            return hold

        self._localization_steps += 1
        if self.cube_pose is not None:
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
            self._fail(
                "localization_timeout",
                f"could not localize {' and '.join(missing)} in "
                f"{self.max_localization_steps} control steps",
            )
            return hold

        if self._search_target is not None or (
            self._localization_steps % self.localization_steps_per_search == 0
        ):
            return self._search_action(hold, carrying=False)
        return hold
