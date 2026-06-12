#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Open the composed SO-101 model in the interactive MuJoCo viewer.

Toggle geom group 3 in the viewer (key '3') to show the collision boxes,
group 2 (key '2') to hide the visual meshes.
"""

from __future__ import annotations

import argparse

import mujoco.viewer

from pick_and_place import build_robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    model = build_robot().compile()
    mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
