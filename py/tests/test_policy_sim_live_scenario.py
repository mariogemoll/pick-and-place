# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Freezing an interactively sampled scene into a scenario the env can reset on.

`run_policy_sim.py` samples its own cube and drop zone, but `PolicySimEnv` is
reset by a fully materialized scenario -- the frozen manifest being the only way
in is what makes two scored runs comparable. `live_scenario` is the bridge, and
it is only worth having if what it declares actually reaches the simulator.
"""

import math

import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.miscalibration import MiscalibrationModel
from pick_and_place.core.physics import PhysicsDraw
from pick_and_place.runtime.policy_sim import (
    PolicySimEnv,
    joint_qpos_addresses,
    live_scenario,
)
from pick_and_place.spec.controller import STATE_FEATURE
from pick_and_place.spec.robot import ARM_JOINT_NAMES
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

RENDER_HW = (64, 64)
IMAGE_HW = (32, 32)
NEUTRAL_STATE = (0.0, 0.0, 0.0, 0.0, -90.0, 39.3)


class DummyRenderer:
    def __init__(self, model, *, height, width):
        del model
        self.height = height
        self.width = width

    def update_scene(self, data, *, camera):
        del data, camera

    def render(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        pass


def _source():
    return CubePose(x=0.22, y=0.01, z=CUBE_HALF_SIZE, yaw=0.3)


def _scenario(**overrides):
    fields = dict(
        source=_source(),
        target_xy=(0.34, -0.05),
        target_plate_yaw_rad=0.4,
        initial_robot_state_real=NEUTRAL_STATE,
        control_hz=30.0,
        max_steps=450,
    )
    fields.update(overrides)
    return live_scenario(**fields)


def test_the_scene_it_declares_is_the_scene_the_env_resets_onto():
    scenario = _scenario()
    env = PolicySimEnv(
        image_hw=IMAGE_HW, render_hw=RENDER_HW, renderer_factory=DummyRenderer
    )
    try:
        _, info = env.reset(options={"scenario": scenario})
        cube = info["task_state"]["cube_position_m"]
        assert cube == pytest.approx((_source().x, _source().y, _source().z), abs=1e-9)
        assert info["task_state"]["target_xy_m"] == pytest.approx((0.34, -0.05), abs=1e-9)
    finally:
        env.close()


def test_an_unseeded_run_records_a_seed_rather_than_failing():
    """The env never reads it, but the field is an int and a run may have none."""
    assert _scenario(seed=None).seed == 0
    assert _scenario(seed=7).seed == 7


def test_a_scene_without_draws_declares_nominal_ones():
    scenario = _scenario()
    assert scenario.domain_randomization_preset is None
    assert scenario.domain_randomization_sample == {"enabled": False}
    assert scenario.miscalibration_sample["pan_jitter"] is None
    assert scenario.miscalibration_sample["joint_offsets_deg"] == {}
    assert scenario.physics_sample == PhysicsDraw().__dict__


def test_the_drawn_miscalibration_reaches_the_env():
    """The offsets have to move the arm, not merely be recorded on the scenario.

    A joint zero is a *sensing* error, so at reset it cancels on the way out:
    the arm is placed at ``commanded + offset`` and read back at ``measured -
    offset``, leaving the observation identical. Where it is visible is the true
    joint the physics runs on, which is the point of injecting it at all.
    """
    draw = MiscalibrationModel(pan_jitter_sigma_deg=0.0).sample(np.random.default_rng(4))
    env = PolicySimEnv(
        image_hw=IMAGE_HW, render_hw=RENDER_HW, renderer_factory=DummyRenderer
    )
    try:
        plain, _ = env.reset(options={"scenario": _scenario()})
        nominal_qpos = env.data.qpos[joint_qpos_addresses(env.model)].copy()
        offset, _ = env.reset(options={"scenario": _scenario(miscalibration=draw)})
        drawn_qpos = env.data.qpos[joint_qpos_addresses(env.model)].copy()
    finally:
        env.close()

    np.testing.assert_allclose(offset[STATE_FEATURE], plain[STATE_FEATURE], atol=1e-5)
    shift_deg = np.degrees(drawn_qpos - nominal_qpos)
    for index, name in enumerate(ARM_JOINT_NAMES):
        assert shift_deg[index] == pytest.approx(
            draw.base_offsets_deg.get(name, 0.0), abs=1e-3
        )


def test_the_wander_survives_the_trip_through_the_scenario():
    """Serialized by seed, so the env's fresh jitter must retrace the same path."""
    draw = MiscalibrationModel().sample(np.random.default_rng(9))
    env = PolicySimEnv(
        image_hw=IMAGE_HW, render_hw=RENDER_HW, renderer_factory=DummyRenderer
    )
    try:
        env.reset(options={"scenario": _scenario(miscalibration=draw)})
        rebuilt = env._miscalibration
        times = [0.0, 0.5, 2.0, 7.0]
        assert [rebuilt.offsets_deg(t) for t in times] == [draw.offsets_deg(t) for t in times]
    finally:
        env.close()


def test_the_target_plate_yaw_must_be_one_the_square_plate_has():
    with pytest.raises(ValueError, match="target_plate_yaw_rad"):
        _scenario(target_plate_yaw_rad=math.pi)
