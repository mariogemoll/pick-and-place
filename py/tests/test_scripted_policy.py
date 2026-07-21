# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from dataclasses import dataclass
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pick_and_place.follower import ARM_JOINT_NAMES, real_frame_to_sim, sim_frame_to_real
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.scripted_policy import (
    AsyncWristLocalization,
    ScriptedPolicy,
    ScriptedPolicyState,
)
from pick_and_place.trajectory import DescentPhase, Frame, GraspChoice, Trajectory


class StubLocalizer:
    def __init__(self, cubes=(), targets=()):
        self.cubes = iter(cubes)
        self.targets = iter(targets)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def localize_cube(self, image, **kwargs):
        del image
        self.cube_kwargs = kwargs
        return next(self.cubes, None)

    def localize_drop_target(self, image, **kwargs):
        del image
        self.target_kwargs = kwargs
        return next(self.targets, None)


def _observation(state=None):
    return {
        STATE_FEATURE: np.arange(6, dtype=np.float32) if state is None else state,
        OVERHEAD_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
        WRIST_FEATURE: np.ones((8, 8, 3), dtype=np.uint8),
        "privileged.cube_position": np.array([9.0, 9.0, 9.0]),
    }


def _policy(localizer, *, max_steps=2):
    return ScriptedPolicy(
        localizer,
        np.ones((4, 3)),
        max_localization_steps=max_steps,
        plan_episode=lambda *args, **kwargs: SimpleNamespace(trajectory="planned"),
    )


def test_async_wrist_localization_returns_the_latest_completed_result():
    completed = threading.Event()
    estimate = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    received = []

    def localize(frame, joints, gripper, prior):
        received.append((frame, joints, gripper, prior))
        completed.set()
        return estimate

    localizer = AsyncWristLocalization(localize)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    prior = CubePose(0.0, 0.0, CUBE_HALF_SIZE)
    try:
        first = localizer(frame, {"shoulder_pan": 0.1}, 2.0, prior)
        assert completed.wait(timeout=1.0)
        second = localizer(frame, {"shoulder_pan": 0.2}, 3.0, prior)
    finally:
        localizer.close()

    assert first is None
    assert second is estimate
    assert received[0][0] is not frame
    np.testing.assert_array_equal(received[0][0], frame)


def test_scripted_policy_localizes_from_images_and_holds_reported_state():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    localizer = StubLocalizer(cubes=[cube], targets=[target])
    policy = _policy(localizer)

    action = policy.act(_observation())

    np.testing.assert_array_equal(action, np.arange(6, dtype=np.float32))
    assert policy.state is ScriptedPolicyState.READY
    assert policy.cube_pose is cube
    assert policy.drop_target is target
    assert policy.episode.trajectory == "planned"
    assert localizer.target_kwargs["target_color"] == "black"
    np.testing.assert_array_equal(
        localizer.target_kwargs["workspace_corners_world"], np.ones((4, 3))
    )


def test_scripted_policy_times_out_explicitly_and_keeps_holding():
    policy = _policy(StubLocalizer(), max_steps=2)

    policy.act(_observation())
    second_action = policy.act(_observation())
    failed_action = policy.act(_observation(state=np.full(6, 3.0, dtype=np.float32)))

    np.testing.assert_array_equal(second_action, np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(failed_action, np.full(6, 3.0, dtype=np.float32))
    assert policy.terminal
    assert policy.failure is not None
    assert policy.failure.code == "localization_timeout"
    assert "cube and drop target" in policy.failure.message


def test_scripted_policy_reset_clears_episode_state_and_resets_localizer():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    localizer = StubLocalizer(cubes=[cube], targets=[target])
    policy = _policy(localizer)
    policy.act(_observation())

    policy.reset()

    assert policy.state is ScriptedPolicyState.LOCALIZING
    assert policy.cube_pose is None
    assert policy.drop_target is None
    assert policy.episode is None
    assert policy.failure is None
    assert localizer.reset_count == 2


def test_scripted_policy_turns_bad_image_observation_into_terminal_failure():
    policy = _policy(StubLocalizer())
    observation = _observation()
    del observation[WRIST_FEATURE]

    action = policy.act(observation)

    np.testing.assert_array_equal(action, observation[STATE_FEATURE])
    assert policy.terminal
    assert policy.failure is not None
    assert policy.failure.code == "localization_error"
    assert WRIST_FEATURE in policy.failure.message


def test_scripted_policy_searches_from_fixed_rng_and_reset_replays_search():
    policy = ScriptedPolicy(
        StubLocalizer(),
        np.ones((4, 3)),
        max_localization_steps=3,
        localization_steps_per_search=1,
        rng_seed=42,
    )

    first_search = policy.act(_observation())
    second_search = policy.act(_observation())
    policy.reset()
    reset_search = policy.act(_observation())

    assert not np.array_equal(first_search, _observation()[STATE_FEATURE])
    assert not np.array_equal(second_search, first_search)
    np.testing.assert_array_equal(reset_search, first_search)


def test_scripted_policy_holds_search_target_until_the_next_search_draw():
    policy = ScriptedPolicy(
        StubLocalizer(),
        np.ones((4, 3)),
        max_localization_steps=5,
        localization_steps_per_search=2,
    )

    initial_hold = policy.act(_observation())
    first_search = policy.act(_observation())
    retained_search = policy.act(_observation())
    second_search = policy.act(_observation())

    np.testing.assert_array_equal(initial_hold, _observation()[STATE_FEATURE])
    np.testing.assert_array_equal(retained_search, first_search)
    assert not np.array_equal(second_search, first_search)


def test_scripted_policy_plans_from_localized_poses_and_latest_reported_joints():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.25, -0.15))
    calls = []

    def plan_episode(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(trajectory="planned")

    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        plan_episode=plan_episode,
    )
    reported = np.array([10.0, -20.0, 30.0, -40.0, 50.0, 51.0], dtype=np.float32)

    policy.act(_observation(state=reported))

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1] is cube
    assert args[2] == CubePose(0.25, -0.15, CUBE_HALF_SIZE)
    np.testing.assert_allclose(
        list(kwargs["start_joints"].values()),
        np.radians(reported[:5]),
    )
    assert kwargs["start_gripper"] > 0.0
    assert kwargs["max_attempts"] == 40
    assert kwargs["include_environment"] is True


def test_scripted_policy_recovery_samples_target_without_a_source_override():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    localizer = StubLocalizer(cubes=[cube])
    calls = []

    def sample_target(rng):
        del rng
        return CubePose(0.2, -0.1, CUBE_HALF_SIZE)

    policy = ScriptedPolicy(
        localizer,
        np.ones((4, 3)),
        target_sampler=sample_target,
        free_grasp=True,
        plan_episode=lambda *args, **kwargs: calls.append((args, kwargs))
        or SimpleNamespace(trajectory="planned"),
    )

    policy.act(_observation())

    args, kwargs = calls[0]
    assert args[1] is cube
    assert args[2] is None
    assert kwargs["target_sampler"] is sample_target
    assert kwargs["free_grasp"] is True
    assert localizer.cube_kwargs == {"free_grasp": True}
    assert not hasattr(localizer, "target_kwargs")


def test_scripted_policy_forwards_planning_diagnostics():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    calls = []

    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        planning_max_attempts=7,
        planning_verbose=True,
        preflight_debug=True,
        preflight_debug_limit=3,
        failed_trajectory_limit=2,
        plan_episode=lambda *args, **kwargs: calls.append(kwargs)
        or SimpleNamespace(trajectory="planned"),
    )

    policy.act(_observation())

    assert calls[0]["max_attempts"] == 7
    assert calls[0]["verbose"] is True
    assert calls[0]["preflight_debug"] is True
    assert calls[0]["preflight_debug_limit"] == 3
    assert calls[0]["failed_trajectory_limit"] == 2


def test_scripted_policy_reports_planning_failure_and_holds():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))

    def fail_plan(*args, **kwargs):
        raise RuntimeError("no feasible trajectory")

    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        plan_episode=fail_plan,
    )

    action = policy.act(_observation())

    np.testing.assert_array_equal(action, _observation()[STATE_FEATURE])
    assert policy.terminal
    assert policy.failure is not None
    assert policy.failure.code == "planning_error"
    assert policy.failure.message == "no feasible trajectory"


@dataclass(frozen=True)
class StubPhase:
    name: str
    duration: float
    value: float

    def evaluate(self, t):
        return Frame(
            joints={name: self.value + t for name in ARM_JOINT_NAMES},
            gripper=0.1 + t,
        )


def _planned_episode(trajectory):
    return SimpleNamespace(
        trajectory=trajectory,
        kinematics=object(),
        target=CubePose(0.2, -0.1, CUBE_HALF_SIZE),
        end_joints={name: 0.0 for name in ARM_JOINT_NAMES},
        end_gripper=0.0,
    )


def test_scripted_policy_executes_one_trajectory_sample_per_control_tick():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    phase = StubPhase("retreat", duration=1.0, value=0.25)
    trajectory = Trajectory((phase,))
    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        control_hz=2.0,
        plan_episode=lambda *args, **kwargs: _planned_episode(trajectory),
    )

    planning_action = policy.act(_observation())
    first = policy.act(_observation())
    second = policy.act(_observation())
    final = policy.act(_observation())
    held = policy.act(_observation(state=np.full(6, 7.0, dtype=np.float32)))

    np.testing.assert_array_equal(planning_action, _observation()[STATE_FEATURE])
    np.testing.assert_allclose(
        first,
        sim_frame_to_real(phase.evaluate(0.0).joints, phase.evaluate(0.0).gripper),
    )
    np.testing.assert_allclose(
        second,
        sim_frame_to_real(phase.evaluate(0.5).joints, phase.evaluate(0.5).gripper),
    )
    np.testing.assert_allclose(
        final,
        sim_frame_to_real(phase.evaluate(1.0).joints, phase.evaluate(1.0).gripper),
    )
    np.testing.assert_array_equal(held, np.full(6, 7.0, dtype=np.float32))
    assert policy.succeeded


def test_scripted_policy_replans_at_checkpoint_from_latest_reported_joints():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    first_phase = StubPhase("lift", duration=0.0, value=0.1)
    unused_phase = StubPhase("release", duration=1.0, value=0.2)
    replanned_phase = StubPhase("retreat", duration=1.0, value=0.3)
    initial = Trajectory((first_phase, unused_phase))
    replanned = Trajectory((replanned_phase,))
    calls = []

    def replan(*args, **kwargs):
        calls.append((args, kwargs))
        return [replanned]

    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        plan_episode=lambda *args, **kwargs: _planned_episode(initial),
        replan_candidates=replan,
        trajectory_preflight=lambda episode, trajectory: True,
    )
    policy.act(_observation())
    policy.act(_observation())
    reported = np.array([12.0, -23.0, 34.0, -45.0, 56.0, 67.0], dtype=np.float32)

    action = policy.act(_observation(state=reported))

    assert len(calls) == 1
    expected_joints, expected_gripper = real_frame_to_sim(reported)
    assert calls[0][0][1] == expected_joints
    assert calls[0][0][2] == expected_gripper
    assert calls[0][0][3] == "lift"
    np.testing.assert_allclose(
        action,
        sim_frame_to_real(
            replanned_phase.evaluate(0.0).joints,
            replanned_phase.evaluate(0.0).gripper,
        ),
    )


def test_scripted_policy_wrist_servo_uses_only_image_and_reported_state():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE, yaw=0.1)
    estimate = CubePose(0.12, 0.18, CUBE_HALF_SIZE, yaw=0.2)
    target = SimpleNamespace(xy=(0.2, -0.1))
    joints = {name: 0.0 for name in ARM_JOINT_NAMES}
    identity = np.eye(4)
    grasp = GraspChoice(
        face="free",
        elbow="up",
        pitch=0.0,
        roll_offset=0.0,
        closing_azimuth=0.0,
        camera_outward=0.0,
        hover_joints=joints,
        grasp_joints=joints,
        hover_matrix=identity,
        grasp_matrix=identity,
        lift_joints=joints,
        lift_matrix=identity,
        inward_normal=np.zeros(3),
    )
    descent = DescentPhase(object(), grasp)
    trajectory = Trajectory((descent,), grasp=grasp)
    calls = []

    def wrist_localizer(image, measured_joints, measured_gripper, prior):
        calls.append((image.copy(), measured_joints, measured_gripper, prior))
        return estimate

    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        wrist_localizer=wrist_localizer,
        plan_episode=lambda *args, **kwargs: _planned_episode(trajectory),
    )
    observation = _observation(
        state=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
    )
    policy.act(observation)

    policy.act(observation)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], observation[WRIST_FEATURE])
    expected_joints, expected_gripper = real_frame_to_sim(observation[STATE_FEATURE])
    assert calls[0][1] == expected_joints
    assert calls[0][2] == expected_gripper
    assert calls[0][3] is cube
    assert policy._dynamic_source.x == pytest.approx(0.102)
    assert policy._dynamic_source.y == pytest.approx(0.198)


def test_scripted_policy_safely_holds_after_wrist_servo_timeout():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    joints = {name: 0.0 for name in ARM_JOINT_NAMES}
    identity = np.eye(4)
    grasp = GraspChoice(
        face="free",
        elbow="up",
        pitch=0.0,
        roll_offset=0.0,
        closing_azimuth=0.0,
        camera_outward=0.0,
        hover_joints=joints,
        grasp_joints=joints,
        hover_matrix=identity,
        grasp_matrix=identity,
        lift_joints=joints,
        lift_matrix=identity,
        inward_normal=np.zeros(3),
    )
    trajectory = Trajectory((DescentPhase(object(), grasp),), grasp=grasp)
    policy = ScriptedPolicy(
        StubLocalizer(cubes=[cube], targets=[target]),
        np.ones((4, 3)),
        control_hz=1.0,
        wrist_localizer=lambda *args: None,
        plan_episode=lambda *args, **kwargs: _planned_episode(trajectory),
    )
    observation = _observation(state=np.full(6, 4.0, dtype=np.float32))
    policy.act(observation)

    for _ in range(20):
        action = policy.act(observation)
        if policy.terminal:
            break

    assert policy.state is ScriptedPolicyState.FAILED
    assert policy.failure.code == "descent_servo_timeout"
    np.testing.assert_array_equal(action, observation[STATE_FEATURE])
