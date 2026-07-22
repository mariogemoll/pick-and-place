# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Target-plate yaw sampling.

The recorded drop-zone plate used to be pinned to yaw 0. ``sample_target``
only guarantees that *some* yaw in [0, 90) fits at a target centre, so a pinned
yaw 0 rendered plates that physically overlapped a calibration AprilTag.
"""

import math

import numpy as np
import pytest

from pick_and_place.episodes import sample_target
from pick_and_place.paper_detection import DROP_ZONE_HALF_SIZE
from pick_and_place.workspace_overlays import (
    is_target_plate_allowed,
    is_target_plate_position_allowed,
    sample_target_plate_yaw,
)


def test_sampled_yaw_always_fits_the_plate_at_sampled_targets():
    """Every drawn yaw must clear the rails and tags, at every legal centre."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        target = sample_target(rng)
        yaw = sample_target_plate_yaw(
            rng, target.x, target.y, half_size=DROP_ZONE_HALF_SIZE
        )
        assert is_target_plate_allowed(
            target.x, target.y, yaw, half_size=DROP_ZONE_HALF_SIZE
        ), f"yaw {math.degrees(yaw):.1f} deg does not fit at {target.x, target.y}"


def test_sampled_yaw_spans_the_full_quarter_turn():
    """The plate must actually vary; a constant yaw is the bug being fixed."""
    rng = np.random.default_rng(1)
    yaws = [
        sample_target_plate_yaw(rng, t.x, t.y, half_size=DROP_ZONE_HALF_SIZE)
        for t in (sample_target(rng) for _ in range(500))
    ]

    assert all(0.0 <= yaw < math.pi / 2.0 + 1e-9 for yaw in yaws)
    # A square plate is symmetric under 90 deg, so [0, 90) is the whole space.
    assert min(yaws) < math.radians(10.0)
    assert max(yaws) > math.radians(80.0)
    assert len(set(yaws)) > 400, "yaw is not being resampled per episode"


def test_yaw_zero_is_not_a_safe_default():
    """Regression: the pinned yaw-0 plate overlapped tags at real targets.

    Documents *why* the fallback is a scan rather than 0. If this ever finds no
    such centre, the exclusion geometry changed and the fallback can be
    simplified.
    """
    rng = np.random.default_rng(0)
    targets = [sample_target(rng) for _ in range(4000)]
    illegal_at_zero = [
        t
        for t in targets
        if not is_target_plate_allowed(t.x, t.y, 0.0, half_size=DROP_ZONE_HALF_SIZE)
    ]

    assert illegal_at_zero, "expected some centres where an axis-aligned plate clips"
    # Every such centre is still a legal target, so a fitting yaw must exist.
    for target in illegal_at_zero:
        assert is_target_plate_position_allowed(target.x, target.y)


def test_fallback_scan_is_used_when_random_draws_all_miss():
    """With no random attempts left, the scan must still return a legal yaw."""
    rng = np.random.default_rng(2)
    target = next(
        t
        for t in (sample_target(rng) for _ in range(4000))
        if not is_target_plate_allowed(t.x, t.y, 0.0, half_size=DROP_ZONE_HALF_SIZE)
    )

    yaw = sample_target_plate_yaw(
        rng, target.x, target.y, half_size=DROP_ZONE_HALF_SIZE, max_attempts=0
    )

    assert yaw != 0.0
    assert is_target_plate_allowed(
        target.x, target.y, yaw, half_size=DROP_ZONE_HALF_SIZE
    )


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_yaw_draw_is_reproducible_for_a_seeded_stream(seed):
    """Episodes stay index-addressable: same seed and centre, same plate."""
    target = sample_target(np.random.default_rng(seed))
    first = sample_target_plate_yaw(
        np.random.default_rng(seed), target.x, target.y, half_size=DROP_ZONE_HALF_SIZE
    )
    repeated = sample_target_plate_yaw(
        np.random.default_rng(seed), target.x, target.y, half_size=DROP_ZONE_HALF_SIZE
    )

    assert first == repeated
