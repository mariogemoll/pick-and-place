# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drawing an arm that is not quite the one the plan assumed.

The claim under test is narrow and load-bearing: the dial does nothing at zero,
does something at one, and never leaves anything behind. A scale that is not
undone compounds onto the next episode, which would widen the envelope silently
and be almost impossible to notice in the data.
"""

import numpy as np
import pytest

from pick_and_place.core.physics import NOMINAL, PhysicsModel
from pick_and_place.rollout.sim import build_recording_scene
from pick_and_place.sim.physics import PhysicsRandomizer, actuator_gains, tracking_bias_offsets
from pick_and_place.spec.robot import ARM_JOINT_NAMES, JOINT_NAMES

WIDE = PhysicsModel(amount=1.0)


@pytest.fixture(scope="module")
def model():
    return build_recording_scene(render_width=64, render_height=64)[0]


def _snapshot(model) -> dict[str, np.ndarray]:
    return {
        "gain": actuator_gains(model),
        "mass": model.body_mass.copy(),
        "inertia": model.body_inertia.copy(),
        "friction": model.geom_friction.copy(),
        "damping": model.dof_damping.copy(),
        "frictionloss": model.dof_frictionloss.copy(),
        "timeconst": model.actuator_dynprm[:, 0].copy(),
    }


def test_the_dial_at_zero_is_the_nominal_arm():
    assert PhysicsModel().sample(np.random.default_rng(0)) is NOMINAL
    assert NOMINAL.is_nominal


def test_a_negative_amount_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        PhysicsModel(amount=-0.1)


def test_a_draw_is_a_function_of_its_seed():
    first = WIDE.sample(np.random.default_rng(5))
    assert first == WIDE.sample(np.random.default_rng(5))
    assert first != WIDE.sample(np.random.default_rng(6))


def test_no_draw_can_turn_a_scale_negative():
    """Log-normal, so a doubling and a halving are equally likely and zero is out of reach."""
    wide = PhysicsModel(amount=4.0)
    for seed in range(50):
        draw = wide.sample(np.random.default_rng(seed))
        assert draw.mass_scale > 0.0
        assert draw.friction_scale > 0.0
        assert draw.damping_scale > 0.0
        assert all(value > 0.0 for value in draw.joint_gain_scale.values())


def test_applying_a_draw_changes_the_arm_and_resetting_puts_it_back(model):
    before = _snapshot(model)
    randomizer = PhysicsRandomizer(model)

    randomizer.apply(WIDE.sample(np.random.default_rng(1)))
    after = _snapshot(model)
    for name in ("gain", "mass", "friction", "damping", "frictionloss", "timeconst"):
        assert not np.array_equal(before[name], after[name]), name

    randomizer.reset()
    for name, value in _snapshot(model).items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)


def test_a_second_draw_does_not_compound_onto_the_first(model):
    randomizer = PhysicsRandomizer(model)

    randomizer.apply(WIDE.sample(np.random.default_rng(1)))
    first = _snapshot(model)
    randomizer.apply(WIDE.sample(np.random.default_rng(2)))
    randomizer.apply(WIDE.sample(np.random.default_rng(1)))

    for name, value in _snapshot(model).items():
        np.testing.assert_allclose(value, first[name], err_msg=name)


def test_the_servo_stays_a_servo(model):
    """A position actuator's gain and its bias have to move together."""
    randomizer = PhysicsRandomizer(model)

    randomizer.apply(WIDE.sample(np.random.default_rng(3)))

    for name in JOINT_NAMES:
        index = [
            i
            for i in range(model.nu)
            if model.actuator(i).name == name
        ]
        if not index:
            continue
        gain = model.actuator_gainprm[index[0], 0]
        assert model.actuator_biasprm[index[0], 1] == pytest.approx(-gain)
    randomizer.reset()


def test_stiction_is_added_to_what_the_joint_already_has(model):
    """Overwriting it would make a randomized arm freer than the nominal one."""
    before = model.dof_frictionloss.copy()
    randomizer = PhysicsRandomizer(model)

    randomizer.apply(WIDE.sample(np.random.default_rng(4)))

    assert np.all(model.dof_frictionloss >= before - 1e-12)
    assert np.any(model.dof_frictionloss > before)
    randomizer.reset()


def test_mass_and_inertia_scale_together(model):
    """Otherwise the links weigh more but spin as though they did not."""
    before = _snapshot(model)
    randomizer = PhysicsRandomizer(model)
    draw = WIDE.sample(np.random.default_rng(9))

    randomizer.apply(draw)

    heavy = model.body_mass > 0
    np.testing.assert_allclose(
        model.body_mass[heavy] / before["mass"][heavy], draw.mass_scale
    )
    spinning = before["inertia"] > 0
    np.testing.assert_allclose(
        model.body_inertia[spinning] / before["inertia"][spinning], draw.mass_scale
    )
    randomizer.reset()


def test_the_tracking_bias_is_a_fraction_of_the_fitted_droop():
    fitted = {name: 0.02 for name in ARM_JOINT_NAMES}

    assert tracking_bias_offsets(fitted, NOMINAL) == {}

    draw = WIDE.sample(np.random.default_rng(7))
    offsets = tracking_bias_offsets(fitted, draw)
    assert offsets == pytest.approx(
        {name: 0.02 * draw.tracking_bias_scale for name in ARM_JOINT_NAMES}
    )


def test_the_draw_is_recorded_so_a_dataset_can_be_split_by_it():
    metadata = WIDE.sample(np.random.default_rng(8)).as_metadata()

    assert metadata["physics_mass_scale"] > 0.0
    assert "physics_gain_scale_shoulder_pan" in metadata
    assert "physics_extra_joint_friction_elbow_flex" in metadata
    assert all(isinstance(value, float) for value in metadata.values())
