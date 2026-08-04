# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The facts and contracts every part of the project has to agree on.

Nothing here imports anything else in ``pick_and_place``, and nothing here needs
MuJoCo, OpenCV, lerobot or Torch. That is the point: the cube's size, the tag
ids on its faces and the shape of a controller's answer are properties of the
task, not of whichever component happens to touch them first.

Keeping them here is what lets the simulator and the detector agree by
construction rather than by two modules importing into each other's internals.
"""
