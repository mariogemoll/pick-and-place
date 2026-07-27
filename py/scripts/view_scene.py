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
import math
from pathlib import Path

import mujoco
import mujoco.viewer

from pick_and_place import build_scene, export_scene
from pick_and_place.camera_extrinsics import (
    add_camera_extrinsics_markers,
    apply_camera_extrinsics_to_spec,
    camera_pose_delta_mm_deg,
    load_local_camera_extrinsics,
)


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
    parser.add_argument(
        "--environment",
        action="store_true",
        help="include the calibration workspace_frame and overhead camera mount in the scene",
    )
    parser.add_argument(
        "--apriltag-cube",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "use the AprilTag-stickered pick cube (perception target) instead of "
            "the plain red cube; defaults to on with --environment, off otherwise"
        ),
    )
    parser.add_argument(
        "--camera-extrinsics",
        action="store_true",
        help=(
            "apply the locally calibrated camera poses from config/camera_extrinsics "
            "and mark both poses in the scene: red ball and sight line where the scene "
            "authors the camera, green where it was measured. Toggle geom group 5 in "
            "the viewer (key '5') to hide them"
        ),
    )
    args = parser.parse_args()

    if args.export_only and args.export is None:
        parser.error("--export-only requires --export")

    wrist_camera = not args.no_wrist_camera
    if args.export is not None:
        output = export_scene(
            args.export,
            wrist_camera=wrist_camera,
            include_environment=args.environment,
            apriltag_cube=args.apriltag_cube,
        )
        print(f"Wrote {output}")
    if not args.export_only:
        spec = build_scene(
            wrist_camera=wrist_camera,
            include_environment=args.environment,
            apriltag_cube=args.apriltag_cube,
        )
        if args.camera_extrinsics:
            extrinsics = load_local_camera_extrinsics()
            if not extrinsics:
                print("no local camera extrinsics found in config/camera_extrinsics")
            for camera in spec.cameras:
                measured = extrinsics.get(camera.name)
                if measured is None:
                    continue
                authored = (camera.pos, camera.quat)
                millimetres, degrees = camera_pose_delta_mm_deg(authored, measured)
                print(f"{camera.name}: measured pose is {millimetres:.2f} mm and "
                      f"{degrees:.2f} deg from the authored one")
            add_camera_extrinsics_markers(spec, extrinsics)
            apply_camera_extrinsics_to_spec(spec, extrinsics)
        model = spec.compile()
        data = mujoco.MjData(model)

        # Compensate for the physical 2.8° (0.0486795 rad) arm twist.
        wrist_roll = math.radians(2.8 - 90)
        data.joint("wrist_roll").qpos = wrist_roll
        data.actuator("wrist_roll").ctrl = wrist_roll

        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
