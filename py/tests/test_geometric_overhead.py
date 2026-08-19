# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pick_and_place.plant.geometric_overhead import (
    DEFAULT_CUBE_VISIBILITY_FRACTION,
    DEFAULT_PLATE_VISIBILITY_FRACTION,
    SimGeometricOverheadLocalizer,
)


def test_geometric_visibility_uses_separate_cube_and_plate_thresholds():
    localizer = object.__new__(SimGeometricOverheadLocalizer)
    localizer.visibility_fraction_thresholds = {
        "cube": DEFAULT_CUBE_VISIBILITY_FRACTION,
        "plate": DEFAULT_PLATE_VISIBILITY_FRACTION,
    }
    localizer.visibility_fraction = lambda object_name: 0.5

    assert not localizer._visible("cube")
    assert localizer._visible("plate")
