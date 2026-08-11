# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The fitted servo tracking bias, and that injecting it moves the physics."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from pick_and_place.core.robot_dynamics import (
    load_robot_dynamics_config,
    set_actuator_activation,
    tracking_bias_rad,
    tracking_bias_vector,
)
from pick_and_place.sim.scene import build_scene
from pick_and_place.spec.robot import ARM_JOINT_NAMES, JOINT_NAMES

# A loaded reach-down pose, where the arm carries the most gravity torque.
REACH_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.9,
    "elbow_flex": 1.2,
    "wrist_flex": 0.7,
    "wrist_roll": 0.0,
    "gripper": 0.3,
}


@pytest.fixture(scope="module")
def config() -> dict:
    return load_robot_dynamics_config()


def test_bias_matches_the_fit_it_is_derived_from(config: dict) -> None:
    """steady_state_bias is beta / alpha, and the artifact records both."""
    bias = tracking_bias_rad(config)
    for name in ARM_JOINT_NAMES:
        joint = config["joints"][name]
        expected = joint["beta_per_frame"] / joint["alpha_per_frame"]
        assert joint["steady_state_bias"] == pytest.approx(expected, rel=1e-6)
        assert bias[name] == pytest.approx(math.radians(expected), rel=1e-9)


def test_the_loaded_joints_carry_the_bias_and_the_vertical_axis_does_not(config: dict) -> None:
    """Gravity droop, not follower lag: pan holds no gravity torque and fits at ~0."""
    bias = tracking_bias_rad(config)
    assert abs(math.degrees(bias["shoulder_lift"])) > 2.0
    assert abs(math.degrees(bias["elbow_flex"])) > 1.0
    assert abs(math.degrees(bias["shoulder_pan"])) < 0.1


def test_the_gripper_is_excluded_from_the_bias(config: dict) -> None:
    """Its fit is in 0-100 hardware units, so converting it as an angle is wrong."""
    assert "gripper" not in tracking_bias_rad(config)


def test_scale_zero_removes_the_bias(config: dict) -> None:
    assert all(value == 0.0 for value in tracking_bias_rad(config, scale=0.0).values())


def test_scale_is_linear(config: dict) -> None:
    single = tracking_bias_rad(config)
    doubled = tracking_bias_rad(config, scale=2.0)
    for name, value in single.items():
        assert doubled[name] == pytest.approx(2.0 * value)


def test_vector_orders_by_joint_and_zeroes_the_absent(config: dict) -> None:
    vector = tracking_bias_vector(tracking_bias_rad(config), JOINT_NAMES)
    assert vector.shape == (len(JOINT_NAMES),)
    assert vector[JOINT_NAMES.index("gripper")] == 0.0
    assert vector[JOINT_NAMES.index("shoulder_lift")] != 0.0


def settle(model, data, ctrl: dict[str, float]) -> dict[str, float]:
    """Command a pose, run to steady state, return achieved minus commanded."""
    mujoco.mj_resetData(model, data)
    for name, value in ctrl.items():
        actuator = model.actuator(name).id
        data.ctrl[actuator] = value
        set_actuator_activation(model, data, actuator, value)
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    for _ in range(4000):
        mujoco.mj_step(model, data)
    return {
        name: data.qpos[model.jnt_qposadr[model.joint(name).id]] - value
        for name, value in ctrl.items()
    }


@pytest.fixture(scope="module")
def settled_scene():
    model = build_scene().compile()
    return model, mujoco.MjData(model)


def test_simulation_tracks_its_command_almost_exactly(settled_scene) -> None:
    """The gap this exists to close: kp is 998, so sim has no droop of its own."""
    model, data = settled_scene
    error = settle(model, data, REACH_POSE)
    assert max(abs(math.degrees(value)) for value in error.values()) < 0.1


def test_a_biased_command_settles_where_the_fit_says(settled_scene, config: dict) -> None:
    """Commanding ``target + bias`` must land the joint on ``target + bias``."""
    model, data = settled_scene
    bias = tracking_bias_rad(config)
    commanded = {name: value + bias.get(name, 0.0) for name, value in REACH_POSE.items()}
    achieved = settle(model, data, commanded)
    for name in ARM_JOINT_NAMES:
        landed = commanded[name] + achieved[name] - REACH_POSE[name]
        assert math.degrees(landed) == pytest.approx(
            math.degrees(bias[name]), abs=0.1
        ), f"{name} did not settle a bias away from the unbiased target"


def test_the_bias_moves_the_gripper_in_space(settled_scene, config: dict) -> None:
    """A degree at the shoulder is millimetres at the fingertip -- grasp scale."""
    model, data = settled_scene
    bias = tracking_bias_rad(config)
    positions = []
    for biased in (False, True):
        ctrl = {
            name: value + (bias.get(name, 0.0) if biased else 0.0)
            for name, value in REACH_POSE.items()
        }
        settle(model, data, ctrl)
        positions.append(data.body("gripper").xpos.copy())
    displacement = float(np.linalg.norm(positions[1] - positions[0]))
    assert displacement > 0.005, "the fitted bias should move the gripper by millimetres"
