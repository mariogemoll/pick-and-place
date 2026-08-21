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

from pick_and_place.runtime.policy_sim import PolicySimEnv, build_policy_sim_model
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
