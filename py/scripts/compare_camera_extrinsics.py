#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compare camera extrinsics against the MuJoCo model and visualize them."""

from __future__ import annotations

import argparse
import numpy as np
import mujoco
import mujoco.viewer

from pick_and_place.scene import build_scene
from pick_and_place.camera_extrinsics import load_local_camera_extrinsics
from pick_and_place.builder import STOCK_ASSETS_DIR

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true", help="Do not launch the viewer")
    args = parser.parse_args()

    # Load environment
    spec = build_scene(include_environment=True)
    spec.meshdir = str(STOCK_ASSETS_DIR)
    
    # Load extrinsics
    extrinsics = load_local_camera_extrinsics()
    
    if not extrinsics:
        print("No extrinsics found in 'config/camera_extrinsics'. Will only show model cameras.")
        
    print("Visualizing cameras...")
    for cam in spec.cameras:
        camera_name = cam.name
        print(f"Camera: {camera_name}")
        print(f"  Model pos: {cam.pos}")

        parent_body = None
        for b in list(spec.bodies) + [spec.worldbody]:
            if cam in list(b.cameras):
                parent_body = b
                break
        
        if parent_body:
            # Add dots to visualize positions
            parent_body.add_geom(
                name=f"{camera_name}_dot_model",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0.0, 0.0],
                pos=cam.pos,
                rgba=[0, 0, 1, 1], # Blue for Model
                contype=0,
                conaffinity=0,
            )
            
            ext = extrinsics.get(camera_name)
            if ext:
                print(f"  Extr pos:  {ext['pos']}")
                print(f"  Diff pos:  {np.array(cam.pos) - np.array(ext['pos'])}")
                
                # Look for the lens cylinder geom in the same body
                lens_name = f"{camera_name}_lens_visual"
                for geom in parent_body.geoms:
                    if geom.name == lens_name:
                        print(f"  Lens pos:  {geom.pos}")
                        print(f"  Diff lens: {np.array(geom.pos) - np.array(ext['pos'])}")
                        break

                # Add calibrated camera to the mount (parent body)
                calibrated_name = f"{camera_name}_calibrated"
                parent_body.add_camera(
                    name=calibrated_name,
                    pos=ext['pos'],
                    quat=ext['quat'],
                )
                print(f"  Added: {calibrated_name}")
                
                parent_body.add_geom(
                    name=f"{camera_name}_dot_real",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0.0, 0.0],
                    pos=ext['pos'],
                    rgba=[0, 1, 0, 1], # Green for Real
                    contype=0,
                    conaffinity=0,
                )
            else:
                print("  No extrinsics found for comparison.")
        else:
            print(f"  Warning: parent body not found for {camera_name}")

    for camera_name in extrinsics:
        if not any(cam.name == camera_name for cam in spec.cameras):
            print(f"Camera {camera_name} from extrinsics not found in model cameras.")

    # Visualize
    if not args.no_viewer:
        print("Launching viewer...")
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
