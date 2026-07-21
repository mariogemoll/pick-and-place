# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Safety boundary between policy controllers and the physical robot."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from pick_and_place.follower import ARM_JOINT_NAMES, JOINT_NAMES, action_to_joints, joints_to_action
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    ControllerFailure,
    PolicyController,
    PolicyObservation,
)


class PhysicalEpisodeOutcome(str, Enum):
    """Mutually exclusive physical result of an attempted episode."""

    SUCCESS = "success"
    CONTROLLER_FAILURE = "controller_failure"
    TIMEOUT = "timeout"
    PICKUP_FAILURE = "pickup_failure"
    PLACEMENT_FAILURE = "placement_failure"
    OPERATOR_ABORT = "operator_abort"


@dataclass(frozen=True)
class PhysicalPolicyTick:
    """One synchronized observation, policy action, and hardware command."""

    index: int
    scheduled_at: float
    observed_at: float
    observation: PolicyObservation
    requested_action: np.ndarray
    command: np.ndarray
    clamped: bool
    slew_limited: bool


@dataclass(frozen=True)
class PhysicalPolicyEpisodeResult:
    """Outcome of one controller episode on the physical adapter."""

    control_steps: int
    outcome: PhysicalEpisodeOutcome
    controller_failure: ControllerFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome is PhysicalEpisodeOutcome.SUCCESS

    @property
    def timed_out(self) -> bool:
        return self.outcome is PhysicalEpisodeOutcome.TIMEOUT


class PhysicalEpisodeRecording(Protocol):
    """Transactional recording owned by the physical orchestration layer."""

    def record_tick(self, tick: PhysicalPolicyTick) -> None: ...

    def commit(self) -> None: ...

    def discard(self) -> None: ...


def calibrated_state(raw_state: np.ndarray, offsets_deg: Mapping[str, float]) -> np.ndarray:
    """Map raw follower readback into the calibrated policy hardware frame."""
    state = np.asarray(raw_state, dtype=float).copy()
    for index, name in enumerate(ARM_JOINT_NAMES):
        state[index] += float(offsets_deg.get(name, 0.0))
    return state


def raw_command(calibrated_command: np.ndarray, offsets_deg: Mapping[str, float]) -> np.ndarray:
    """Map a calibrated policy command into raw follower servo coordinates."""
    command = np.asarray(calibrated_command, dtype=float).copy()
    for index, name in enumerate(ARM_JOINT_NAMES):
        command[index] -= float(offsets_deg.get(name, 0.0))
    return command


def _limit_slew(
    requested: np.ndarray,
    previous: np.ndarray,
    max_slew_per_second: float | np.ndarray | None,
    period: float,
) -> tuple[np.ndarray, bool]:
    if max_slew_per_second is None:
        return requested, False
    slew = np.broadcast_to(np.asarray(max_slew_per_second, dtype=float), requested.shape)
    if not np.all(np.isfinite(slew)) or np.any(slew <= 0.0):
        raise ValueError("max_slew_per_second must be positive and finite")
    limited = previous + np.clip(requested - previous, -slew * period, slew * period)
    return limited, not np.allclose(limited, requested)


def classify_physical_outcome(
    *,
    controller_succeeded: bool,
    controller_failure: ControllerFailure | None,
    timed_out: bool,
    pickup_verified: bool | None,
    placement_verified: bool | None,
) -> PhysicalEpisodeOutcome:
    """Combine controller completion with independent physical verification."""
    if timed_out:
        return PhysicalEpisodeOutcome.TIMEOUT
    if controller_failure is not None or not controller_succeeded:
        return PhysicalEpisodeOutcome.CONTROLLER_FAILURE
    if pickup_verified is False:
        return PhysicalEpisodeOutcome.PICKUP_FAILURE
    if placement_verified is False:
        return PhysicalEpisodeOutcome.PLACEMENT_FAILURE
    return PhysicalEpisodeOutcome.SUCCESS


def prepare_physical_policy_episode(
    controller: PolicyController,
    *,
    follower: Any,
    overhead_rgb: Callable[[], np.ndarray],
    wrist_rgb: Callable[[], np.ndarray],
    clamp_low: np.ndarray,
    clamp_high: np.ndarray,
    joint_zero_offsets: Mapping[str, float] | None = None,
    max_steps: int,
    sleep: Callable[[float], None] = time.sleep,
) -> PhysicalPolicyEpisodeResult | None:
    """Localize and plan before starting the deadline-sensitive command loop.

    Search-pose commands are still clamped and sent at 30 Hz, but planning time
    does not consume execution ticks or shift the execution schedule.
    """
    offsets = joint_zero_offsets or {}
    low = np.asarray(clamp_low, dtype=float)
    high = np.asarray(clamp_high, dtype=float)
    controller.reset()
    previous_raw = action_to_joints(follower.get_observation(), (low + high) / 2.0)
    for step in range(1, max_steps + 1):
        state = calibrated_state(
            action_to_joints(follower.get_observation(), previous_raw), offsets
        ).astype(np.float32)
        observation = {
            STATE_FEATURE: state,
            OVERHEAD_FEATURE: np.asarray(overhead_rgb()),
            WRIST_FEATURE: np.asarray(wrist_rgb()),
        }
        action = np.asarray(controller.act(observation), dtype=float).reshape(-1)
        command = np.clip(action, low, high)
        follower.send_action(joints_to_action(raw_command(command, offsets)))
        previous_raw = raw_command(command, offsets)
        if bool(getattr(controller, "terminal", False)):
            return PhysicalPolicyEpisodeResult(
                step,
                PhysicalEpisodeOutcome.CONTROLLER_FAILURE,
                getattr(controller, "failure", None),
            )
        state_value = getattr(getattr(controller, "state", None), "value", None)
        if state_value == "ready":
            controller.begin_execution()
            return None
        sleep(1.0 / 30.0)
    return PhysicalPolicyEpisodeResult(max_steps, PhysicalEpisodeOutcome.TIMEOUT)


def run_physical_policy_episode(
    controller: PolicyController,
    *,
    follower: Any,
    overhead_rgb: Callable[[], np.ndarray],
    wrist_rgb: Callable[[], np.ndarray],
    clamp_low: np.ndarray,
    clamp_high: np.ndarray,
    control_hz: float,
    max_steps: int,
    joint_zero_offsets: Mapping[str, float] | None = None,
    max_slew_per_second: float | np.ndarray | None = None,
    pickup_verifier: Callable[[PhysicalPolicyTick], bool | None] | None = None,
    placement_verifier: Callable[[], bool] | None = None,
    recording: PhysicalEpisodeRecording | None = None,
    observation_callback: Callable[[PolicyObservation], None] | None = None,
    tick_callback: Callable[[PhysicalPolicyTick], None] | None = None,
    reset_controller: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PhysicalPolicyEpisodeResult:
    """Run a controller at a fixed command rate and verify physical success.

    Camera reads, controller inference, and commands form one synchronized tick.
    Localization and planning should be completed before entering this function.
    The adapter never treats trajectory completion alone as physical success.
    """
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be positive and finite")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    low = np.asarray(clamp_low, dtype=float).reshape(-1)
    high = np.asarray(clamp_high, dtype=float).reshape(-1)
    expected_shape = (len(JOINT_NAMES),)
    if low.shape != expected_shape or high.shape != expected_shape:
        raise ValueError(f"joint limits must have shape {expected_shape}")
    if np.any(low > high):
        raise ValueError("each lower joint limit must not exceed its upper limit")

    offsets = joint_zero_offsets or {}
    if reset_controller:
        controller.reset()
    period = 1.0 / control_hz
    next_tick = clock()
    raw_previous = action_to_joints(follower.get_observation(), np.clip((low + high) / 2.0, low, high))
    previous = calibrated_state(raw_previous, offsets)
    steps = 0
    pickup_verified: bool | None = None

    try:
        for step in range(1, max_steps + 1):
            steps = step
            observed_at = clock()
            raw_state = action_to_joints(follower.get_observation(), raw_previous)
            state = calibrated_state(raw_state, offsets).astype(np.float32)
            observation = {
                STATE_FEATURE: state,
                OVERHEAD_FEATURE: np.asarray(overhead_rgb()),
                WRIST_FEATURE: np.asarray(wrist_rgb()),
            }
            if observation_callback is not None:
                observation_callback(observation)
            action = np.asarray(controller.act(observation), dtype=float).reshape(-1)
            if action.shape != expected_shape or not np.all(np.isfinite(action)):
                raise ValueError(
                    f"controller action must be finite with shape {expected_shape}, got {action.shape}"
                )
            slew_command, slew_limited = _limit_slew(action, previous, max_slew_per_second, period)
            command = np.clip(slew_command, low, high)
            clamped = not np.allclose(command, slew_command)
            if clamped:
                warnings.warn("physical policy action exceeded joint limits and was clamped", stacklevel=2)
            tick = PhysicalPolicyTick(
                index=step,
                scheduled_at=next_tick,
                observed_at=observed_at,
                observation=observation,
                requested_action=action.copy(),
                command=command.copy(),
                clamped=clamped,
                slew_limited=slew_limited,
            )
            if tick_callback is not None:
                tick_callback(tick)
            if recording is not None:
                recording.record_tick(tick)
            follower.send_action(joints_to_action(raw_command(command, offsets)))
            raw_previous = raw_state
            previous = command

            if pickup_verifier is not None and pickup_verified is None:
                pickup_verified = pickup_verifier(tick)
                if pickup_verified is False:
                    result = PhysicalPolicyEpisodeResult(
                        step, PhysicalEpisodeOutcome.PICKUP_FAILURE
                    )
                    break

            if bool(getattr(controller, "terminal", False)):
                failure = getattr(controller, "failure", None)
                outcome = classify_physical_outcome(
                    controller_succeeded=bool(getattr(controller, "succeeded", False)),
                    controller_failure=failure,
                    timed_out=False,
                    pickup_verified=pickup_verified,
                    placement_verified=placement_verifier() if placement_verifier is not None else None,
                )
                result = PhysicalPolicyEpisodeResult(step, outcome, failure)
                break

            next_tick += period
            remaining = next_tick - clock()
            if remaining > 0.0:
                sleep(remaining)
            elif remaining < -period:
                next_tick = clock()
        else:
            result = PhysicalPolicyEpisodeResult(max_steps, PhysicalEpisodeOutcome.TIMEOUT)
    except KeyboardInterrupt:
        result = PhysicalPolicyEpisodeResult(steps, PhysicalEpisodeOutcome.OPERATOR_ABORT)

    if recording is not None:
        if result.succeeded:
            recording.commit()
        else:
            recording.discard()
    return result
