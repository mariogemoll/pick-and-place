# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The shape of a drop-zone reading.

This is a contract rather than a capability. The detector produces one, the
controller consumes one, and neither has any business knowing how the other
works — so what they agree on lives here instead of in either of them.

Purely geometric: where the square is in the image, where it is on the table,
and how confidently it was called a square. Nothing about how it was found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PaperTarget:
    """Detected drop-zone geometry in the image and on a horizontal world plane."""

    center_px: NDArray
    corners_px: NDArray
    center_world: NDArray
    corners_world: NDArray
    area_px: float
    rectangularity: float

    @property
    def xy(self) -> tuple[float, float]:
        return float(self.center_world[0]), float(self.center_world[1])

    @property
    def yaw(self) -> float:
        """Yaw angle (radians) of the square's first edge in world XY."""
        edge = self.corners_world[1] - self.corners_world[0]
        return float(np.arctan2(edge[1], edge[0]))

    @property
    def half_extent(self) -> tuple[float, float]:
        """Half side lengths (metres) along the square's own two edge directions."""
        edge_x = self.corners_world[1] - self.corners_world[0]
        edge_y = self.corners_world[2] - self.corners_world[1]
        return (
            float(np.linalg.norm(edge_x[:2])) / 2.0,
            float(np.linalg.norm(edge_y[:2])) / 2.0,
        )
