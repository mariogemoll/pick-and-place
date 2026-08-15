# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Locating the cube and the drop plate by actually looking at them.

These render, so they are slower than most of the suite. They earn it: the claim
being checked is that the simulated overhead chain misses the way the rig's
does, and nothing short of running the detector on a rendered frame can say
whether it holds.
"""

import math

import mujoco
import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.miscalibration import (
    MiscalibrationModel,
    OverheadCameraError,
    OverheadCameraModel,
)
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed
from pick_and_place.plant.overhead import SimOverheadPerception, believed_camera_pose
from pick_and_place.plant.overhead_check import fold_cube_symmetry
from pick_and_place.rollout.localized_episode import prepare_localized_episode
from pick_and_place.rollout.sim import build_recording_scene
from pick_and_place.sim.model import set_cube_pose, set_joint
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.spec.workspace import CUBE_HALF_SIZE, DROP_ZONE_HALF_SIZE

NO_ERROR = OverheadCameraError((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

#: A cube and a plate far enough apart that the plate's square stays whole. A
#: cube resting on it breaks the contour, which is a real failure and has its
#: own test rather than being allowed to leak into every other one.
CUBE = CubePose(x=0.26, y=-0.13, z=CUBE_HALF_SIZE, yaw=0.4)
TARGET_XY = (0.19, 0.09)

#: An arm pose clear of both, so a look is about localization rather than luck.
CLEAR_ARM = {
    "shoulder_pan": -1.2,
    "shoulder_lift": -1.4,
    "elbow_flex": 1.4,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
}


@pytest.fixture(scope="module")
def scene():
    model, data = build_recording_scene(render_width=1920, render_height=1080)
    _place(model, data, CUBE, TARGET_XY, CLEAR_ARM)
    return model, data


@pytest.fixture(scope="module")
def perception(scene):
    model, data = scene
    built = SimOverheadPerception(model, data)
    yield built
    built.close()


def _place(model, data, cube, target_xy, arm) -> None:
    set_cube_pose(model, data, cube)
    place_paper_target_marker(
        model,
        target_xy,
        0.25,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(*target_xy),
        alpha=1.0,
    )
    for name, value in arm.items():
        set_joint(model, data, name, value)
    set_joint(model, data, "gripper", 0.5)
    mujoco.mj_forward(model, data)


def test_a_clean_scene_localizes_almost_exactly(scene, perception):
    """Which is the problem, not the achievement.

    An honest render-and-detect in a scene where the extrinsics are exact beats
    the rig by more than an order of magnitude. Injecting the causes is what
    brings it back to something demonstrations can be generated from.
    """
    perception.set_error(NO_ERROR)
    perception.reset()

    reading = perception.look()

    assert reading.complete
    assert math.hypot(reading.cube.x - CUBE.x, reading.cube.y - CUBE.y) < 0.002
    seen_x, seen_y = reading.target.xy
    assert math.hypot(seen_x - TARGET_XY[0], seen_y - TARGET_XY[1]) < 0.005


def test_a_calibration_that_is_wrong_makes_the_localization_wrong(scene, perception):
    """The whole mechanism: the gap between the two camera poses becomes the miss."""
    perception.set_error(OverheadCameraError((0.02, -0.015, 0.0), (0.0, 0.0, 0.0)))
    perception.reset()

    reading = perception.look()

    assert reading.complete
    miss = math.hypot(reading.cube.x - CUBE.x, reading.cube.y - CUBE.y)
    assert 0.015 < miss < 0.045
    perception.set_error(NO_ERROR)


def test_the_arm_can_stand_in_the_way(scene, perception):
    """Occlusion is an outcome, not a bug — it is why the rig hunts."""
    model, data = scene
    perception.set_error(NO_ERROR)
    _place(model, data, CUBE, TARGET_XY, {name: 0.0 for name in CLEAR_ARM})
    perception.reset()

    hidden = perception.look()

    _place(model, data, CUBE, TARGET_XY, CLEAR_ARM)
    perception.reset()
    assert perception.look().complete
    assert not hidden.complete


def test_the_believed_camera_pose_is_the_true_one_plus_the_error():
    position = np.array([0.1, 0.2, 0.8])
    rotation = np.eye(3)

    same_position, same_rotation = believed_camera_pose(position, rotation, NO_ERROR)
    np.testing.assert_allclose(same_position, position)
    np.testing.assert_allclose(same_rotation, rotation)

    moved, turned = believed_camera_pose(
        position, rotation, OverheadCameraError((0.01, 0.0, 0.0), (0.0, 0.0, 5.0))
    )
    np.testing.assert_allclose(moved, position + np.array([0.01, 0.0, 0.0]))
    assert not np.allclose(turned, rotation)


def test_the_frame_placement_error_is_part_of_the_position_error():
    """Two causes, one effect: they compose rather than acting separately."""
    error = OverheadCameraModel().sample(np.random.default_rng(4))

    assert error.frame_placement_m[2] == 0.0  # a flat fixture lies on the table
    assert any(abs(value) > 0.0 for value in error.frame_placement_m[:2])
    # The frame's contribution is inside the total, not alongside it.
    assert abs(error.position_m[0]) <= abs(error.frame_placement_m[0]) + 0.05


def test_a_quarter_turn_of_the_cube_is_not_an_error():
    """A cube face is indistinguishable from the next one round."""
    assert fold_cube_symmetry(math.pi / 2) == pytest.approx(0.0, abs=1e-9)
    assert fold_cube_symmetry(-math.pi / 2) == pytest.approx(0.0, abs=1e-9)
    assert fold_cube_symmetry(math.radians(3.0)) == pytest.approx(math.radians(3.0))


def test_a_localized_episode_plans_against_what_it_saw(scene):
    """End to end: the belief is measured, and it is not the truth."""
    model, data = scene
    perception = SimOverheadPerception(model, data)
    rng = np.random.default_rng(3)
    draw = MiscalibrationModel().sample(rng)
    perception.set_error(OverheadCameraModel().sample(rng))
    try:
        localized = prepare_localized_episode(
            rng,
            model,
            data,
            perception,
            max_attempts=8,
            include_environment=True,
            miscalibration=draw,
        )
    finally:
        perception.close()

    episode = localized.episode
    assert episode.trajectory.phases
    cube_miss = math.hypot(
        episode.believed_source.x - episode.source.x,
        episode.believed_source.y - episode.source.y,
    )
    # Measured, so neither exact nor arbitrary: a few millimetres, like the rig.
    assert 0.0 < cube_miss < 0.030
    # The plate was placed before the look, which is the only order that works.
    assert -math.pi <= localized.target_plate_yaw <= math.pi
