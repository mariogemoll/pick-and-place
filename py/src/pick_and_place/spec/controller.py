# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The contract between a closed-loop controller and whatever is driving it.

A controller is asked to ``reset`` at the start of an episode and to ``act`` on
an observation once per control tick. Scripted planners, learned policies and
the evaluation harness all speak this, which is why it lives below all of them
rather than beside any one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

STATE_FEATURE = "observation.state"
OVERHEAD_FEATURE = "observation.images.overhead"
WRIST_FEATURE = "observation.images.wrist"

PolicyObservation = dict[str, np.ndarray]


@runtime_checkable
class PolicyController(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: PolicyObservation) -> np.ndarray: ...


@dataclass(frozen=True)
class ControllerFailure:
    """A terminal failure reported by a controller without unsafe motion."""

    code: str
    message: str
