# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The analytic expert: plan a pick-and-place, and fly it.

Tier: capability branch.

Everything the scripted system decides lives here — where to take hold, how to
get the cube across, when to replan, how to servo a descent onto what the wrist
camera sees. What it deliberately does *not* contain is a way to see, a way to
step physics, or a loop: it consumes sightings rather than producing them, and
its planning and preflight are injected. That is what lets it import nothing but
the physical facts and pure geometry, and what makes it drivable from exactly
the observations a learned policy is drivable from.
"""
