# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.cli.project_dvd import LOGO_ASPECT, advance

HALF_EXTENT = 0.2626  # the workspace frame's inner rail
LOGO_WIDTH = 0.085


def _limit(half_extent: float = HALF_EXTENT, logo_width: float = LOGO_WIDTH) -> np.ndarray:
    half_w = logo_width / 2.0
    return np.array([half_extent - half_w, half_extent - half_w / LOGO_ASPECT])


def _simulate(steps: int, speed: float = 0.25, fps: float = 30.0, seed_heading=(1.0, 0.72)):
    limit = _limit()
    period = 1.0 / fps
    heading = np.array(seed_heading)
    velocity = heading / np.linalg.norm(heading) * speed
    position = np.zeros(2)
    track = []
    bounces = 0
    for _ in range(steps):
        position, velocity, bounced = advance(position, velocity, limit, period)
        bounces += bounced
        track.append(position.copy())
    return np.array(track), bounces, limit


def test_the_logo_never_crosses_the_rail():
    track, bounces, limit = _simulate(4000)

    assert bounces > 10, "the run should have turned many times"
    assert np.all(np.abs(track) <= limit + 1e-12)


def test_the_logo_edge_reaches_the_rail_exactly_when_it_turns():
    """The point of the whole exercise: it must touch, not stop short."""
    track, _, limit = _simulate(4000)

    # Some tick sits exactly on each limit, on both sides of both axes.
    for axis in (0, 1):
        assert np.isclose(track[:, axis].max(), limit[axis], atol=1e-12)
        assert np.isclose(track[:, axis].min(), -limit[axis], atol=1e-12)

    # And the logo's outer edge is then exactly on the rail itself.
    half_w = LOGO_WIDTH / 2.0
    assert track[:, 0].max() + half_w == pytest.approx(HALF_EXTENT, abs=1e-12)
    assert track[:, 1].max() + half_w / LOGO_ASPECT == pytest.approx(HALF_EXTENT, abs=1e-12)


def test_the_turning_point_does_not_drift_with_frame_rate():
    """Clamping on bounce is what makes this frame-rate independent."""
    slow, _, limit = _simulate(6000, fps=12.0)
    fast, _, _ = _simulate(6000, fps=90.0)

    assert np.isclose(slow[:, 0].max(), limit[0], atol=1e-12)
    assert np.isclose(fast[:, 0].max(), limit[0], atol=1e-12)


def test_a_bounce_reverses_only_the_axis_that_hit():
    limit = np.array([0.2, 0.2])
    position = np.array([0.199, 0.0])
    velocity = np.array([1.0, 1.0])

    _, reflected, bounced = advance(position, velocity, limit, 0.01)

    assert bounced == 1
    assert reflected[0] == -1.0
    assert reflected[1] == 1.0


def test_a_corner_reverses_both_axes():
    limit = np.array([0.2, 0.2])
    position = np.array([0.199, 0.199])
    velocity = np.array([1.0, 1.0])

    landed, reflected, bounced = advance(position, velocity, limit, 0.01)

    assert bounced == 2
    assert reflected == pytest.approx([-1.0, -1.0])
    assert landed == pytest.approx([0.2, 0.2])


def test_advance_does_not_mutate_its_inputs():
    """The loop reuses these arrays; aliasing them would corrupt the next tick."""
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 1.0])

    advance(position, velocity, np.array([0.2, 0.2]), 0.5)

    assert position == pytest.approx([0.0, 0.0])
    assert velocity == pytest.approx([1.0, 1.0])
