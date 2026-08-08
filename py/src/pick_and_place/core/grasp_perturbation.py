# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A deliberate one-shot error in where the cube is believed to be.

The analytic planner never fails, so no recorded demonstration contains a retry:
the cloned policy has never seen an arm recover from a fumbled grasp, and
`docs/POLICY_FAILURE_ANATOMY.md` section 4 measures the consequence -- a missed
grasp is terminal, the arm retreats to a pose no demonstration passes through and
drifts outward until timeout. This makes the planner fumble on purpose so the
recovery it then plans can be recorded as a demonstration.

**Why the belief and not the command.** The obvious alternative is to disturb the
arm directly: add a transient offset to `data.ctrl`, or shove the cube's `qpos`
mid-approach. Both are worse.

- Moving the cube's `qpos` teleports it. Consecutive frames then show a
  discontinuity no real scene can produce, and it lands inside the observation
  history the policy conditions on.
- Offsetting `data.ctrl` makes physics diverge from the command, so the recorded
  `observation.state` says the arm is somewhere the images show it is not. At the
  ~1 degree scale :mod:`pick_and_place.core.miscalibration` draws that is a
  faithful model of a real miscalibrated servo; at the centimetre scale needed to
  actually miss a grasp it is a mismatch no robot exhibits.

Offsetting the *believed* cube pose instead leaves the arm executing its plan
accurately. State, action and images stay mutually consistent; the grasp misses
because the target was wrong, and the jaws shove the cube exactly the way scene
17 records it. The displacement is then a physical consequence rather than an
injected event.

**Why it is not part of a MiscalibrationDraw.** That draw means "error the real
system actually exhibits", is calibrated to measured residuals (6-9 mm of cube
belief error), and is recorded into episode metadata where the sim2real work
reads it. Folding deliberate sabotage into the same field would make the
domain-randomization sample uninterpretable, and would also inherit the wrong
lifetime: a draw is drawn per *episode*, and a persistent bias is unrecoverable
by construction, because the retry inherits it and mis-aims identically. What
recovery data needs is a *transient* error -- wrong for the first grasp, gone for
the retry, which re-localizes from a fresh sighting and can therefore succeed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

#: Planar magnitude of the injected belief error, metres.
#:
#: The cube is 30 mm across (``CUBE_HALF_SIZE`` 0.015), so an error much below
#: the half-size still lets the jaws catch it and the fumble does not happen;
#: this is set past that so the grasp reliably misses. The upper bound worth
#: caring about is plausibility rather than physics: 22 mm reads as a bad
#: localization, which is the failure being modelled, and matches the ~25 mm
#: shove measured in the scene-17 trace. For reference the *measured* real belief
#: error is 6-9 mm, which the planner survives -- that is why it cannot be reused
#: here by simply turning it up a little.
DEFAULT_MAGNITUDE_M = 0.022


@dataclass(frozen=True)
class GraspPerturbation:
    """Where the planner will think the cube is, relative to where it is.

    Applied to the believed pose only, and only for the first grasp. Recorded on
    the episode so a dataset can be filtered or reweighted by perturbation type
    later, which is what makes the perturbed fraction a sweepable parameter
    rather than a property baked into the data.
    """

    dx_m: float
    dy_m: float
    dyaw_rad: float = 0.0
    #: Names the mechanism, so a dataset carrying several can be split by it.
    kind: str = "believed_cube_offset"

    @classmethod
    def sample(
        cls,
        rng: np.random.Generator,
        *,
        magnitude_m: float = DEFAULT_MAGNITUDE_M,
        yaw_error_rad: float = 0.0,
    ) -> "GraspPerturbation":
        """Draw a displacement of fixed magnitude in a uniformly random direction.

        The magnitude is fixed rather than drawn so that "did the grasp miss" is
        not itself a random variable -- a Gaussian would put a chunk of its mass
        inside the jaw width, silently producing unperturbed episodes labelled as
        perturbed. The *direction* is uniform so the fumble is not biased to one
        approach side, which would teach a recovery specific to that side.
        """
        if magnitude_m <= 0.0:
            raise ValueError("magnitude_m must be positive")
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        return cls(
            dx_m=magnitude_m * math.cos(bearing),
            dy_m=magnitude_m * math.sin(bearing),
            dyaw_rad=float(rng.uniform(-yaw_error_rad, yaw_error_rad))
            if yaw_error_rad
            else 0.0,
        )

    @property
    def magnitude_m(self) -> float:
        return math.hypot(self.dx_m, self.dy_m)

    def clears_jaws(self) -> bool:
        """Whether the offset is big enough that the grasp should actually miss.

        Not enforced -- a caller may deliberately want a marginal fumble -- but
        worth being able to assert in a test, since an offset inside the cube
        half-width produces a perturbed episode that succeeds anyway.
        """
        return self.magnitude_m > CUBE_HALF_SIZE

    def apply(self, believed: CubePose) -> CubePose:
        """Displace a believed cube pose. The true pose is never touched."""
        return CubePose(
            x=believed.x + self.dx_m,
            y=believed.y + self.dy_m,
            z=believed.z,
            roll=believed.roll,
            pitch=believed.pitch,
            yaw=believed.yaw + self.dyaw_rad,
        )

    def as_metadata(self) -> dict[str, float | str]:
        """Flat, JSON-safe form for episode metadata."""
        return {
            "kind": self.kind,
            "dx_m": self.dx_m,
            "dy_m": self.dy_m,
            "dyaw_rad": self.dyaw_rad,
            "magnitude_m": self.magnitude_m,
        }
