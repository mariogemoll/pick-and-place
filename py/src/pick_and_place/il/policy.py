# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The one interface every approach produces: ``observation -> action`` at 50 Hz.

The learning guide's central claim is that analytic, IL, RL and VLA all ultimately
yield the same object — a closed-loop policy. :class:`Policy` is that object. The
analytic planner is wrapped as an (open-loop) policy too
(:class:`~pick_and_place.il.rollout.AnalyticPolicy`), so every method runs through
the identical eval harness and the comparisons become real rather than anecdotal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """A closed-loop controller mapping an observation to a 6-vector of joint
    set points (``JOINT_NAMES`` order, radians)."""

    def reset(self) -> None:
        """Clear any per-episode state. Called once at the start of each rollout."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the action for ``observation`` (shape ``(OBS_DIM,)`` -> ``(6,)``)."""
        ...
