# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Decide, between phases, whether to trust the plan or re-derive it.

The arm runs a planned trajectory open-loop within a phase, but a plan made from
a cube pose measured before the episode drifts: servos have backlash, the model's
zero is not the rig's zero, and the cube moves when it is touched. So at phase
boundaries the executor senses where the arm actually is and replans the
remainder from there — sense, plan, execute, re-seed.

Not at *every* boundary, though. A checkpoint costs a pause and re-derives the
plan from motor readback, which is itself noisy; taken at the wrong moment that
is worse than the drift it corrects. Two situations call for skipping it:

- **The next phase corrects for itself.** The descent is a visual servo that
  re-solves IK toward the cube every tick, so replanning the hover it starts from
  is wasted work.
- **Readback is not trustworthy here.** Right after the jaws close, and again at
  the low drop pose, mapping measured joints back into the sim puts the jaws or
  the cube slightly *through* the floor. Nothing useful comes of replanning from
  a state physics considers impossible, so those pairs are flown from the locked
  plan and measured once safely clear.

:data:`FUSED_PHASES` is that list, and it is the whole of the policy — which is
why it lives here, with the controller, and not with the machinery that carries
a replan out.
"""

from __future__ import annotations

#: Phase pairs run back to back off the locked plan, with no checkpoint between
#: them. Each entry is (just completed, what may follow without a replan).
FUSED_PHASES: frozenset[tuple[str, str]] = frozenset(
    {
        # The descent's visual servo corrects any hover-pose error on its own.
        ("approach", "descent"),
        # Contact-critical: readback right after closing maps the jaws through
        # the floor. Lift from the locked grasp pose, then measure.
        ("grasp", "lift"),
        ("grasp", "recovery_lift"),
        # The cruise waypoint is elevated and non-contact; nothing risky has
        # happened yet, so a checkpoint is pure cost and one more chance for
        # sensor noise to abort a fine episode.
        ("carry", "drop_descent"),
        # Contact-critical again: at the low drop pose readback can map clear
        # hardware several millimetres through the floor.
        ("drop_descent", "release"),
    }
)

#: Below this the measured pose has the jaws essentially on the floor, which is
#: worth saying out loud before replanning from it.
JAW_CLEARANCE_WARN_M = 0.005


def fuses_into_next(completed: str | None, upcoming: str) -> bool:
    """Whether ``upcoming`` runs straight off the locked plan after ``completed``."""
    return (completed, upcoming) in FUSED_PHASES
