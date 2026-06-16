# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Imitation learning on the analytic pick-and-place demonstrations.

Rung 1 of the learning guide (``docs/learning-approaches.md``): plain behavior
cloning of a small MLP from state observations (joint positions + privileged cube
and target pose) to the 6 joint set points the analytic planner commands.

Everything is wrapped behind one :class:`~pick_and_place.il.policy.Policy`
interface and run through one shared eval harness
(:mod:`pick_and_place.il.rollout`), so later rungs (ACT/LeRobot, RL, VLA) become
drop-in replacements measured the same way.
"""
