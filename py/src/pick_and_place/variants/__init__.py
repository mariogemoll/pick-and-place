# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turning one recorded trajectory into many rendered datasets.

Tier: convergence.

Everything here answers "no" to the question that organizes the whole
sim/real split: if I change this, does the correct action change? Lighting,
materials, colours, backgrounds, camera viewpoint, exposure and noise move
pixels and nothing else, so they can be drawn *after* an episode exists, as
often as wanted, against a trajectory that already succeeded. The action labels
stay correct because the arm really did execute them.

The input is a trajectory artifact, which is why none of this needs the
planner, the detectors or physics.
"""
