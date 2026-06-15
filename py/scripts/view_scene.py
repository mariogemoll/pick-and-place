#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Open or export the SO-101 robot with a floor, light, and 3 cm cube.

The model is composed on the fly from the stock robot, hand-tuned collision
boxes, workspace overlays, one floor plane, one light, and one cube. Toggle
geom group 4 in the viewer (key '4') to show or hide the workspace overlays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco.viewer

from pick_and_place import build_scene, export_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-wrist-camera",
        action="store_true",
        help="omit the wrist-camera mount and module",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="XML",
        help="write the composed scene to XML before opening the viewer",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="write the XML without opening the viewer (requires --export)",
    )
    args = parser.parse_args()

    if args.export_only and args.export is None:
        parser.error("--export-only requires --export")

    wrist_camera = not args.no_wrist_camera
    if args.export is not None:
        output = export_scene(args.export, wrist_camera=wrist_camera)
        print(f"Wrote {output}")
    if not args.export_only:
        model = build_scene(wrist_camera=wrist_camera).compile()
        mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
