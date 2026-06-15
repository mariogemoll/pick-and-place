#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Open the composed SO-101 model in the interactive MuJoCo viewer.

Toggle geom group 3 in the viewer (key '3') to show the collision boxes,
group 2 (key '2') to hide the visual meshes.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import mujoco.viewer

from pick_and_place import build_robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-wrist-camera",
        action="store_true",
        help="omit the wrist-camera mount and module",
    )
    args = parser.parse_args()

    model = build_robot(wrist_camera=not args.no_wrist_camera).compile()
    data = mujoco.MjData(model)

    # Compensate for the physical 2.8° (0.0486795 rad) arm twist.
    wrist_roll = math.radians(2.8 - 90)
    data.joint("wrist_roll").qpos = wrist_roll
    data.actuator("wrist_roll").ctrl = wrist_roll

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
