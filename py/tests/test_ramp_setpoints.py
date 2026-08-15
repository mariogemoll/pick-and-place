# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Splitting one setpoint change into several equal sends."""

import numpy as np
import pytest

from pick_and_place.scripted.motion import ramp_setpoints


def test_single_step_is_the_undivided_target() -> None:
    start = np.array([0.0, 10.0])
    target = np.array([3.0, -5.0])
    (only,) = ramp_setpoints(start, target, 1)
    np.testing.assert_allclose(only, target)


def test_last_send_lands_exactly_on_target() -> None:
    """No residual error, so nothing accumulates across policy periods."""
    start = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, -7.25, 3.0])
    for steps in (1, 2, 3, 5, 8):
        np.testing.assert_allclose(ramp_setpoints(start, target, steps)[-1], target)


def test_sends_are_equally_spaced() -> None:
    start = np.zeros(2)
    target = np.array([9.0, -3.0])
    sends = ramp_setpoints(start, target, 3)
    deltas = np.diff(np.vstack([start, *sends]), axis=0)
    for delta in deltas[1:]:
        np.testing.assert_allclose(delta, deltas[0])


def test_per_send_jump_shrinks_with_more_steps() -> None:
    """The point of the split: same travel, smaller individual steps."""
    start = np.zeros(1)
    target = np.array([12.0])
    one = np.abs(ramp_setpoints(start, target, 1)[0] - start).max()
    three = np.abs(ramp_setpoints(start, target, 3)[0] - start).max()
    assert three == pytest.approx(one / 3.0)


def test_no_movement_when_already_on_target() -> None:
    start = np.array([4.0, -1.0])
    for send in ramp_setpoints(start, start.copy(), 3):
        np.testing.assert_allclose(send, start)


def test_rejects_non_positive_steps() -> None:
    start, target = np.zeros(1), np.ones(1)
    for steps in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            ramp_setpoints(start, target, steps)
