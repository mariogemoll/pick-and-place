# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The predicate that decides whether a contact rejects an episode."""

from pick_and_place.sim.collisions import is_unexpected


def test_a_jaw_touching_the_cube_is_the_grasp() -> None:
    assert not is_unexpected("fixed_jaw_col_0", "pick_cube")
    assert not is_unexpected("pick_cube", "moving_jaw_col_1")


def test_anything_else_touching_the_cube_is_a_collision() -> None:
    """Only the jaws may touch the cube — an elbow nudging it is a failed episode."""
    assert is_unexpected("upper_arm_col_0", "pick_cube")


def test_a_jaw_touching_the_floor_is_a_collision() -> None:
    assert is_unexpected("fixed_jaw_col_0", "floor")


def test_the_robot_touching_itself_is_a_collision() -> None:
    assert is_unexpected("upper_arm_col_0", "base_col_1")
