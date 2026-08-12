# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Cube-position bin selection for the session joint-zero calibration."""

from types import SimpleNamespace

from pick_and_place.calibration.session_calibration import (
    _AZIMUTH_EDGES_DEG,
    _RADIUS_EDGES,
    _bin_center_pose,
    _suggest_bin,
)
from pick_and_place.core.workspace_bounds import PAN_AXIS


def _kinematics() -> SimpleNamespace:
    """Bin selection reads nothing but the pan axis."""
    return SimpleNamespace(pan_axis=PAN_AXIS)


def test_first_position_is_the_outermost_reachable_radius() -> None:
    """With nothing visited the fit has no anchor, so it wants a long lever."""
    kinematics = _kinematics()
    first = _suggest_bin(set(), kinematics)
    assert first is not None
    reachable_radii = {
        r
        for r in range(len(_RADIUS_EDGES) - 1)
        for a in range(len(_AZIMUTH_EDGES_DEG) - 1)
        if _bin_center_pose(kinematics, (r, a)) is not None
    }
    assert first[0] == max(reachable_radii)


def test_later_positions_spread_in_radius_away_from_visited() -> None:
    """Radius spread conditions the parallel-axis lift/elbow/wrist_flex split."""
    kinematics = _kinematics()
    first = _suggest_bin(set(), kinematics)
    assert first is not None
    second = _suggest_bin({first}, kinematics)
    assert second is not None
    assert second[0] != first[0]


def test_bins_are_exhausted_rather_than_repeated() -> None:
    """Returns None once every reachable bin has been visited."""
    kinematics = _kinematics()
    visited: set[tuple[int, int]] = set()
    while (bin_ := _suggest_bin(visited, kinematics)) is not None:
        assert bin_ not in visited
        visited.add(bin_)
    assert visited
