# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export matching MJCF and web-manifest files for the composed SO-101.

For consumers that need a file on disk (MuJoCo's ``simulate`` viewer, the
Isaac MJCF importer, and the TypeScript viewer). The exported XML carries an
absolute meshdir, so both outputs are machine-local build artifacts.

Usage::

    python -m pick_and_place.export [-o OUTPUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco

from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot

_GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}
_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def _values(values: Any) -> list[float]:
    return [float(value) for value in values]


def web_manifest(spec: mujoco.MjSpec, model: mujoco.MjModel) -> dict[str, Any]:
    """Return a web representation of a composed and compiled MuJoCo model.

    Body and geom poses come from ``spec`` because compiled mesh geom poses
    include MuJoCo's internal mesh recentering and principal-axis rotation.
    Web GLBs retain the source mesh coordinate frames.
    """
    bodies: list[dict[str, Any]] = []
    joints_by_body: dict[int, list[dict[str, Any]]] = {}
    geoms_by_body: dict[int, list[dict[str, Any]]] = {}

    for joint_id in range(model.njnt):
        joint_type = mujoco.mjtJoint(int(model.jnt_type[joint_id]))
        joint = {
            "name": model.joint(joint_id).name,
            "type": _JOINT_TYPES[joint_type],
            "position": _values(model.jnt_pos[joint_id]),
            "axis": _values(model.jnt_axis[joint_id]),
            "limited": bool(model.jnt_limited[joint_id]),
        }
        if joint["limited"]:
            joint["range"] = _values(model.jnt_range[joint_id])
        joints_by_body.setdefault(int(model.jnt_bodyid[joint_id]), []).append(joint)

    for geom_id in range(model.ngeom):
        geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
        spec_geom = spec.geoms[geom_id]
        geom: dict[str, Any] = {
            "name": model.geom(geom_id).name or f"geom_{geom_id}",
            "role": "visual" if int(model.geom_group[geom_id]) == 2 else "collision",
            "type": _GEOM_TYPES[geom_type],
            "position": _values(spec_geom.pos),
            "quaternion": _values(spec_geom.quat),
            "rgba": _values(model.geom_rgba[geom_id]),
        }
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            geom["mesh"] = f"{model.mesh(mesh_id).name}.glb"
        else:
            geom["size"] = _values(model.geom_size[geom_id])
        geoms_by_body.setdefault(int(model.geom_bodyid[geom_id]), []).append(geom)

    for body_id in range(1, model.nbody):
        spec_body = spec.bodies[body_id]
        body: dict[str, Any] = {
            "name": model.body(body_id).name,
            "parent": model.body(int(model.body_parentid[body_id])).name,
            "position": _values(spec_body.pos),
            "quaternion": _values(spec_body.quat),
            "joints": joints_by_body.get(body_id, []),
            "geometries": geoms_by_body.get(body_id, []),
        }
        bodies.append(body)

    cameras = [
        {
            "name": model.camera(camera_id).name,
            "body": model.body(int(model.cam_bodyid[camera_id])).name,
            "position": _values(model.cam_pos[camera_id]),
            "quaternion": _values(model.cam_quat[camera_id]),
            "fovy": float(model.cam_fovy[camera_id]),
        }
        for camera_id in range(model.ncam)
    ]
    return {"format": "pick-and-place-web-model", "version": 1, "bodies": bodies, "cameras": cameras}


def export_robot(output: Path, *, wrist_camera: bool = True) -> tuple[Path, Path]:
    """Write matching XML and JSON outputs from one composed robot."""
    spec = build_robot(wrist_camera=wrist_camera)
    # The spec was loaded relative to the stock model; rewrite meshdir so the
    # exported file resolves meshes from wherever it is saved.
    spec.meshdir = str(STOCK_ASSETS_DIR)
    model = spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    manifest_output = output.with_suffix(".json")
    manifest_output.write_text(json.dumps(web_manifest(spec, model), indent=2) + "\n")
    return output, manifest_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None, help="output XML path")
    parser.add_argument(
        "--no-wrist-camera",
        action="store_true",
        help="omit the wrist-camera mount and module",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parents[2] / "out" / "so101.xml"

    paths = export_robot(output, wrist_camera=not args.no_wrist_camera)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
