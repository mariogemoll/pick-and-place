# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One episode's arm, as distinct from the nominal one its plan was made for.

The joint *zeros* are randomized per episode from a measured day-to-day spread.
The joint *response* is not: there is one fitted dynamics config, applied or
not. Everything below the zeros — how hard a servo pulls, how fast it gets
there, how much the arm weighs, how much it rubs — is a single fixed set of
numbers that every demonstration shares, so nothing in the data ever shows the
planner meeting an arm that behaves differently from the one it assumed.

**The spread here is a judgment, not a measurement.** The joint-zero sigmas come
from per-day fits across four real sessions. There is only one fitted dynamics
config, so there is no observed day-to-day spread to draw from and these start
as knobs. They can be *made* measured the same way the zeros were — fit dynamics
per session across the existing recordings and use the observed spread — and
that is worth doing before leaning on the numbers.

Until then, :attr:`PhysicsModel.amount` is what makes them honest: one dial for
the whole set, zero by default, so a run that has not thought about it gets the
nominal arm and a run that wants variation says how much. Expect demonstration
yield to fall as it goes up — the planner assumes nominal dynamics, which is
the entire reason the knob exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pick_and_place.spec.robot import JOINT_NAMES

#: Fractional spreads at ``amount = 1``. Chosen, not fitted: wide enough that a
#: policy meets arms it has to cope with, narrow enough that the planner's
#: nominal assumption still usually holds.
DEFAULT_GAIN_SIGMA = 0.15
DEFAULT_TIME_CONSTANT_SIGMA = 0.20
DEFAULT_MASS_SIGMA = 0.10
DEFAULT_FRICTION_SIGMA = 0.25
DEFAULT_DAMPING_SIGMA = 0.30
#: Fraction of the *fitted* tracking bias, so an episode droops more or less
#: than the arm that was measured rather than in some unrelated amount.
DEFAULT_TRACKING_BIAS_SIGMA = 0.5
#: Extra dry friction in a joint, in the model's own torque units and *on top of*
#: the 0.052 the stock arm already carries. A ceiling rather than a spread,
#: because it only goes one way: a joint can be stickier than the model says,
#: not slipperier than frictionless. It stands in for backlash — what a policy
#: actually meets is a dead band around the command, and stiction produces one
#: without adding a degree of freedom to the model.
DEFAULT_EXTRA_JOINT_FRICTION_MAX = 0.02


@dataclass(frozen=True)
class PhysicsDraw:
    """How this episode's arm differs from the nominal one.

    Everything is a multiplier on what the compiled model already carries,
    except ``extra_joint_friction``, which is added to it, and
    ``tracking_bias_scale``, which is a fraction of the fitted droop. A draw of
    all ones and zeros is the nominal arm, which is what ``amount = 0``
    produces — so the randomized path and the plain path run the same code.
    """

    joint_gain_scale: dict[str, float] = field(default_factory=dict)
    joint_time_constant_scale: dict[str, float] = field(default_factory=dict)
    extra_joint_friction: dict[str, float] = field(default_factory=dict)
    tracking_bias_scale: float = 0.0
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    damping_scale: float = 1.0

    @property
    def is_nominal(self) -> bool:
        """Whether this draw leaves the model exactly as compiled."""
        return (
            self.mass_scale == 1.0
            and self.friction_scale == 1.0
            and self.damping_scale == 1.0
            and self.tracking_bias_scale == 0.0
            and all(value == 1.0 for value in self.joint_gain_scale.values())
            and all(value == 1.0 for value in self.joint_time_constant_scale.values())
            and all(value == 0.0 for value in self.extra_joint_friction.values())
        )

    def as_metadata(self) -> dict[str, float]:
        """Flatten for an episode row, so a dataset can be split by what it ran under."""
        metadata: dict[str, float] = {
            "physics_mass_scale": self.mass_scale,
            "physics_friction_scale": self.friction_scale,
            "physics_damping_scale": self.damping_scale,
            "physics_tracking_bias_scale": self.tracking_bias_scale,
        }
        for name, value in self.joint_gain_scale.items():
            metadata[f"physics_gain_scale_{name}"] = value
        for name, value in self.joint_time_constant_scale.items():
            metadata[f"physics_time_constant_scale_{name}"] = value
        for name, value in self.extra_joint_friction.items():
            metadata[f"physics_extra_joint_friction_{name}"] = value
        return metadata


#: The arm exactly as the model compiles it.
NOMINAL = PhysicsDraw()


@dataclass(frozen=True)
class PhysicsModel:
    """The envelope an episode's arm is drawn from.

    ``amount`` scales every spread at once, which is the point: the individual
    sigmas are guesses, so what a run should be able to say is "vary the
    physics this much", not "vary the elbow's damping by exactly this".
    """

    amount: float = 0.0
    gain_sigma: float = DEFAULT_GAIN_SIGMA
    time_constant_sigma: float = DEFAULT_TIME_CONSTANT_SIGMA
    mass_sigma: float = DEFAULT_MASS_SIGMA
    friction_sigma: float = DEFAULT_FRICTION_SIGMA
    damping_sigma: float = DEFAULT_DAMPING_SIGMA
    tracking_bias_sigma: float = DEFAULT_TRACKING_BIAS_SIGMA
    extra_joint_friction_max: float = DEFAULT_EXTRA_JOINT_FRICTION_MAX

    def __post_init__(self) -> None:
        if self.amount < 0.0:
            raise ValueError("physics randomization amount must not be negative")

    def sample(self, rng: np.random.Generator) -> PhysicsDraw:
        """Draw one episode's arm. At ``amount = 0`` this is the nominal one."""
        if self.amount == 0.0:
            return NOMINAL

        def scale(sigma: float) -> float:
            # Log-normal, so a doubling and a halving are equally likely and no
            # draw can turn a mass or a gain negative.
            return float(np.exp(rng.normal(0.0, sigma * self.amount)))

        return PhysicsDraw(
            joint_gain_scale={name: scale(self.gain_sigma) for name in JOINT_NAMES},
            joint_time_constant_scale={
                name: scale(self.time_constant_sigma) for name in JOINT_NAMES
            },
            extra_joint_friction={
                name: float(rng.uniform(0.0, self.extra_joint_friction_max * self.amount))
                for name in JOINT_NAMES
            },
            tracking_bias_scale=float(
                rng.normal(0.0, self.tracking_bias_sigma * self.amount)
            ),
            mass_scale=scale(self.mass_sigma),
            friction_scale=scale(self.friction_sigma),
            damping_scale=scale(self.damping_sigma),
        )
