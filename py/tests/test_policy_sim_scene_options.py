# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The scene options `PolicySimEnv` has to carry to host the sim runner.

`run_policy_sim.py` compiles its own model so it can ask `build_scene` for the
finite-floor scene and for raw upstream actuators. Moving it onto `PolicySimEnv`
means the env has to offer the same three choices, and offering them is only
useful if they reach the compiled model -- a passthrough that silently does
nothing would be worse than not having one.
"""

import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.runtime.policy_sim import (
    PolicySimEnv,
    build_policy_sim_model,
    live_scenario,
)
from pick_and_place.spec.controller import WRIST_FEATURE
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.variants.scene import scene_texture_ids

RENDER_HW = (240, 320)


def _table():
    return np.full((32, 32, 3), 128, dtype=np.uint8)


def _panorama():
    return np.full((32, 64, 3), 64, dtype=np.uint8)


def test_the_default_scene_is_the_groundplane_one():
    """No textures to repaint, which is what the evaluator has always compiled."""
    model, _ = build_policy_sim_model(*RENDER_HW)
    assert scene_texture_ids(model) == ()


def test_textures_produce_the_finite_floor_scene():
    model, _ = build_policy_sim_model(
        *RENDER_HW, table_texture=_table(), background_panorama=_panorama()
    )
    # A skybox and a table surface, which is what a background draw repaints.
    assert len(scene_texture_ids(model)) == 2


def test_robot_dynamics_reaches_the_actuators():
    fitted, _ = build_policy_sim_model(*RENDER_HW)
    upstream, _ = build_policy_sim_model(*RENDER_HW, robot_dynamics=False)
    assert not (
        np.array_equal(fitted.actuator_gainprm, upstream.actuator_gainprm)
        and np.array_equal(fitted.actuator_dynprm, upstream.actuator_dynprm)
    ), "--no-robot-dynamics must reach the compiled actuators"


def test_the_env_passes_all_three_through():
    env = PolicySimEnv(
        image_hw=(96, 128),
        render_hw=RENDER_HW,
        table_texture=_table(),
        background_panorama=_panorama(),
        robot_dynamics=False,
    )
    try:
        assert len(scene_texture_ids(env.model)) == 2
        upstream, _ = build_policy_sim_model(*RENDER_HW, robot_dynamics=False)
        np.testing.assert_array_equal(env.model.actuator_gainprm, upstream.actuator_gainprm)
    finally:
        env.close()


def test_the_env_default_is_unchanged():
    """The evaluator constructs it with none of these, and must get what it always got."""
    env = PolicySimEnv(image_hw=(96, 128), render_hw=RENDER_HW)
    try:
        expected, _ = build_policy_sim_model(*RENDER_HW)
        assert scene_texture_ids(env.model) == ()
        np.testing.assert_array_equal(env.model.actuator_gainprm, expected.actuator_gainprm)
        assert env.model.opt.timestep == pytest.approx(expected.opt.timestep)
    finally:
        env.close()


class _GradientRenderer:
    """A fixed noisy render, so the two reductions cannot agree by accident."""

    def __init__(self, model, *, height, width):
        del model
        self.height = height
        self.width = width

    def update_scene(self, data, *, camera):
        del data, camera

    def render(self):
        # Fine detail with no structure that survives averaging unchanged: a
        # regular stripe would average to the same grey on either schedule.
        return np.random.default_rng(0).integers(
            0, 256, size=(self.height, self.width, 3), dtype=np.uint8
        )

    def close(self):
        pass


def _observation_through(recording_hw):
    env = PolicySimEnv(
        image_hw=(24, 24),
        render_hw=(240, 320),
        recording_hw=recording_hw,
        renderer_factory=_GradientRenderer,
    )
    try:
        observation, _ = env.reset(
            options={
                "scenario": live_scenario(
                    source=CubePose(x=0.22, y=0.0, z=CUBE_HALF_SIZE, yaw=0.0),
                    target_xy=(0.34, -0.05),
                    target_plate_yaw_rad=0.4,
                    initial_robot_state_real=(0.0, 0.0, 0.0, 0.0, -90.0, 39.3),
                    control_hz=30.0,
                    max_steps=10,
                )
            }
        )
        return observation[WRIST_FEATURE].copy()
    finally:
        env.close()


def test_the_recording_resolution_is_a_hop_the_frame_actually_takes():
    """A policy trained on video saw two reductions, and they do not compose away.

    Without this the sim runner's ``--recording-hw`` would have nowhere to land
    once it is hosted here, and a learned policy would be fed one-hop frames it
    was never trained on -- a silent change, since the images stay the right size.
    """
    one_hop = _observation_through(None)
    two_hop = _observation_through((60, 80))
    assert one_hop.shape == two_hop.shape == (24, 24, 3)
    assert not np.array_equal(one_hop, two_hop)


def test_the_default_recording_resolution_is_the_policy_size():
    """One hop, which is what a controller that never saw a video wants."""
    env = PolicySimEnv(image_hw=(24, 24), render_hw=RENDER_HW, renderer_factory=_GradientRenderer)
    try:
        assert env.recording_hw == (24, 24)
    finally:
        env.close()
    np.testing.assert_array_equal(_observation_through(None), _observation_through((24, 24)))


def test_a_recording_resolution_above_the_render_is_refused():
    with pytest.raises(ValueError, match="at least the recording dimensions"):
        PolicySimEnv(image_hw=(24, 24), render_hw=(64, 64), recording_hw=(128, 128))
