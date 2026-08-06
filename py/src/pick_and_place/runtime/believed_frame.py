# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The gap between where the arm is told to be and where it actually is.

A real servo commanded ``theta`` rests at model angle ``theta + offset``, and
reports ``theta`` back. Everything the controller sees — the plan, the recorded
``observation.state``, the state a checkpoint replans from — therefore lives in
the *believed* frame, while physics runs in the *true* one.

Playing an episode with a :class:`~pick_and_place.core.miscalibration.MiscalibrationDraw`
means holding both at once, and this is the one place that converts between
them. Without a draw it is the identity: the two frames are the same frame, and
the whole closed loop above degenerates to feedforward playback.

The offsets are read at the *episode's* elapsed time rather than the simulator's,
because the drawn pan jitter wanders over a session and a scene reused across
episodes carries the previous one's clock forward.
"""

from __future__ import annotations

import dataclasses

import mujoco

from pick_and_place.core.miscalibration import MiscalibrationDraw
from pick_and_place.sim.model import get_joint
from pick_and_place.spec.robot import ARM_JOINT_NAMES


@dataclasses.dataclass(frozen=True)
class BelievedFrame:
    """What the arm believes about itself, given the miscalibration in effect."""

    model: mujoco.MjModel
    data: mujoco.MjData
    draw: MiscalibrationDraw | None
    time_origin: float

    @property
    def closed_loop(self) -> bool:
        """Whether there is a belief error to correct for at all."""
        return self.draw is not None

    def offsets_deg(self) -> dict[str, float] | None:
        """Joint-zero offsets in degrees, or ``None`` when nothing was drawn.

        ``None`` rather than an empty mapping because that is what the frame
        conversions take to mean "no feed-forward correction".
        """
        return None if self.draw is None else self.draw.offsets_deg(self.elapsed())

    def offsets_rad(self) -> dict[str, float]:
        """Joint-zero offsets in radians, empty when nothing was drawn."""
        return {} if self.draw is None else self.draw.offsets_rad(self.elapsed())

    def arm_joints(self) -> dict[str, float]:
        """The servo-style readback: true joints minus the offsets in effect."""
        offsets = self.offsets_rad()
        return {
            name: get_joint(self.model, self.data, name) - offsets.get(name, 0.0)
            for name in ARM_JOINT_NAMES
        }

    def elapsed(self) -> float:
        return self.data.time - self.time_origin
