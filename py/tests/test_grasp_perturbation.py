# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The deliberate-fumble draw, and that it only ever moves the *believed* cube."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.grasp_perturbation import (
    DEFAULT_MAGNITUDE_M,
    GraspPerturbation,
)
from pick_and_place.spec.workspace import CUBE_HALF_SIZE


def test_sample_has_the_requested_magnitude_in_a_random_direction():
    rng = np.random.default_rng(0)
    draws = [GraspPerturbation.sample(rng) for _ in range(64)]
    # Fixed magnitude is the point: "did the grasp miss" must not itself be a
    # coin flip, which a Gaussian would make it.
    for draw in draws:
        assert draw.magnitude_m == pytest.approx(DEFAULT_MAGNITUDE_M)
    bearings = [math.atan2(d.dy_m, d.dx_m) for d in draws]
    assert max(bearings) - min(bearings) > math.pi, "directions should span widely"


def test_default_magnitude_clears_the_jaws():
    # An offset inside the cube half-width lets the jaws catch it anyway, which
    # would produce an episode labelled perturbed that succeeded regardless.
    assert DEFAULT_MAGNITUDE_M > CUBE_HALF_SIZE
    assert GraspPerturbation.sample(np.random.default_rng(1)).clears_jaws()
    assert not GraspPerturbation(dx_m=0.001, dy_m=0.0).clears_jaws()


def test_sample_is_a_pure_function_of_the_seed():
    a = GraspPerturbation.sample(np.random.default_rng(7))
    b = GraspPerturbation.sample(np.random.default_rng(7))
    assert a == b


def test_rejects_a_nonpositive_magnitude():
    with pytest.raises(ValueError):
        GraspPerturbation.sample(np.random.default_rng(0), magnitude_m=0.0)


def test_apply_displaces_only_the_plane_and_leaves_height_alone():
    believed = CubePose(x=0.2, y=-0.1, z=CUBE_HALF_SIZE, yaw=0.3)
    draw = GraspPerturbation(dx_m=0.02, dy_m=-0.01, dyaw_rad=0.05)
    moved = draw.apply(believed)
    assert moved.x == pytest.approx(0.22)
    assert moved.y == pytest.approx(-0.11)
    # The cube rests on the table in both frames; a z error would make the
    # descent stop short or drive into the floor, which is a different failure.
    assert moved.z == pytest.approx(believed.z)
    assert moved.yaw == pytest.approx(0.35)


def test_apply_does_not_mutate_its_input():
    believed = CubePose(x=0.2, y=-0.1, z=CUBE_HALF_SIZE)
    GraspPerturbation(dx_m=0.02, dy_m=0.0).apply(believed)
    assert (believed.x, believed.y) == (0.2, -0.1)


def test_metadata_is_json_safe_and_carries_the_kind():
    draw = GraspPerturbation.sample(np.random.default_rng(3))
    meta = draw.as_metadata()
    assert meta["kind"] == "believed_cube_offset"
    assert meta["magnitude_m"] == pytest.approx(DEFAULT_MAGNITUDE_M)
    for key, value in meta.items():
        assert isinstance(value, (str, float)), key
