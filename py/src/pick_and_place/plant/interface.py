# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What a controller commands and observes, whether it is a rig or a simulator.

Real and sim are already structurally identical — **a true world plus a believed
shadow**. On the rig the true world is the physical arm and the shadow is a
MuJoCo model stepped at the commanded joints; in sim the true world is a MuJoCo
model and the shadow is a second one over it. Both step MuJoCo, both take the
wrist camera pose from forward kinematics of the believed shadow, and both solve
tag detection against it.

The differences are narrow: where the image comes from, what receives the
commands, whether the detector runs on a thread or inline, and what drives the
clock. All four fit behind the three operations below — command joints, read
back joints, give me the latest cube sighting — which is why one episode runner
can drive either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from pick_and_place.core.geometry import CubePose


@dataclass(frozen=True)
class Sighting:
    """One look at the cube from the wrist camera.

    ``pose`` is the world-frame answer the descent steers by, pinned to the
    cube's resting height, or ``None`` when nothing was solved this tick.

    ``fresh`` is the asynchronous/synchronous difference made explicit. The
    simulator renders and solves inline, once per tick, so every sighting it
    returns is new. The rig's detector runs on its own thread and the loop
    receives whatever it has most recently produced, which is often the same
    reading twice — and folding the same reading in twice would let one
    detection pull the grasp further than it should.
    """

    pose: CubePose | None = None
    fresh: bool = False
    detections: list = field(default_factory=list)
    estimate: Any = None

    @property
    def usable(self) -> bool:
        """Whether this tick has something new for the descent to steer by."""
        return self.pose is not None and self.fresh


#: What a tick sees when nothing looked.
NOTHING_SEEN = Sighting()


class Plant(Protocol):
    """One tick's worth of commanding and observing.

    Implementations own everything that differs between a rig and a simulator,
    including the clock: ``time`` is what trajectory time advances on, so a phase
    runs at the rate its world does.
    """

    @property
    def time(self) -> float:
        """The clock trajectory time is measured against, in seconds."""
        ...

    def step(self, joints: Mapping[str, float], gripper: float) -> np.ndarray:
        """Apply one tick's set point, let the tick elapse, and return what was sent.

        The return is the real-frame six-vector actually issued — clamped, and
        with any feed-forward offsets folded in — because that, not the planner's
        sim-frame set point, is what a dataset row's ``action`` has to hold.
        """
        ...

    def measured(self) -> tuple[dict[str, float], float]:
        """The believed readback: arm joints and gripper, in the sim frame (radians).

        What the servos report rather than where the arm physically is, because
        that is what the controller has to plan from. Both come back together
        because reading them is one I/O call on the rig, and two calls could
        straddle a tick and disagree.
        """
        ...

    def sighting(self, believed_cube: CubePose) -> Sighting:
        """The latest look at the cube, given where the controller believes it is.

        The argument is not a hint about where to look — the camera points where
        the arm points. It poses the believed shadow, which is what the solve is
        projected through, so the estimate inherits the whole hand-eye error.
        """
        ...

    def set_believed_cube(self, pose: CubePose) -> None:
        """Move the cube in the believed shadow, after a sighting has moved it."""
        ...

    def new_contacts(self) -> set[tuple[str, str]]:
        """Unexpected robot/environment contact pairs touching right now."""
        ...

    def close(self) -> None:
        """Release renderers, detector processes and camera threads."""
        ...
