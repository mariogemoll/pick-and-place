# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export matching MJCF and web-manifest files for the composed SO-101.

For consumers that need a file on disk. The exported XML carries an absolute
meshdir, so both outputs are machine-local build artifacts.

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
from pick_and_place.camera_intrinsics import (
    CAMERA_INTRINSICS_BY_NAME,
    load_camera_intrinsics,
    load_local_camera_intrinsics,
)
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.materials import MaterialConfig
from pick_and_place.scene import ROBOT_BASE_Z_OFFSET, build_environment, build_scene

_GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
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

# The mesh pipeline (mesh_optimization/scripts/simplify_meshes.py) packs meshes
# into three named-node GLBs along these same body-subtree boundaries.
GRIPPER_ROOT_BODY = "gripper"
ROBOT_ROOT_BODY = "base"


def _values(values: Any) -> list[float]:
    return [float(value) for value in values]


def _body_descendants(model: mujoco.MjModel, root_name: str) -> set[int]:
    try:
        root_id = model.body(root_name).id
    except KeyError:
        return set()
    descendants = {root_id}
    frontier = [root_id]
    while frontier:
        parent = frontier.pop()
        children = [
            body_id for body_id in range(model.nbody)
            if int(model.body_parentid[body_id]) == parent and body_id not in descendants
        ]
        descendants.update(children)
        frontier.extend(children)
    return descendants


def _assign_mesh_files(manifest: dict[str, Any], model: mujoco.MjModel) -> None:
    """Tag each mesh geometry with the packed GLB (see module comment above) that contains it.

    Bodies under ``gripper`` go to ``gripper.glb``, the rest of the robot
    (rooted at ``base``) to ``arm.glb``, and anything else (workspace frame,
    overhead mount) to ``environment.glb``.
    """
    gripper_ids = _body_descendants(model, GRIPPER_ROOT_BODY)
    robot_ids = _body_descendants(model, ROBOT_ROOT_BODY)
    body_ids_by_name = {model.body(body_id).name: body_id for body_id in range(model.nbody)}
    for body in manifest["bodies"]:
        body_id = body_ids_by_name.get(body["name"])
        for geom in body["geometries"]:
            if geom.get("mesh") is None:
                continue
            if body_id in gripper_ids:
                geom["meshFile"] = "gripper.glb"
            elif body_id in robot_ids:
                geom["meshFile"] = "arm.glb"
            else:
                geom["meshFile"] = "environment.glb"


def web_manifest(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    materials: MaterialConfig | None = None,
    camera_intrinsics_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a web representation of a composed and compiled MuJoCo model.

    Geom poses come from ``spec`` because compiled mesh geom poses include
    MuJoCo's internal mesh recentering and principal-axis rotation; web GLBs
    retain the source mesh coordinate frames. Body poses come from the
    compiled ``model`` instead, since ``spec`` body poses don't fold in the
    transform of any ``<frame>`` inserted by ``MjSpec.attach`` (e.g. for an
    attached end effector), while the compiled model's are already resolved
    relative to the body's actual parent.
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
        mat_id = int(model.geom_matid[geom_id])
        geom_group = int(model.geom_group[geom_id])
        geom: dict[str, Any] = {
            "name": model.geom(geom_id).name or f"geom_{geom_id}",
            "role": "visual" if geom_group in (2, 4) else "collision",
            "type": _GEOM_TYPES[geom_type],
            "position": _values(spec_geom.pos),
            "quaternion": _values(spec_geom.quat),
            "rgba": _values(model.geom_rgba[geom_id]),
        }
        if mat_id != -1:
            geom["material"] = model.mat(mat_id).name
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            geom["mesh"] = model.mesh(mesh_id).name
        else:
            geom["size"] = _values(model.geom_size[geom_id])
        geoms_by_body.setdefault(int(model.geom_bodyid[geom_id]), []).append(geom)

    for body_id in range(model.nbody):
        body: dict[str, Any] = {
            "name": model.body(body_id).name,
            "parent": model.body(int(model.body_parentid[body_id])).name if body_id > 0 else "world",
            "position": _values(model.body_pos[body_id]),
            "quaternion": _values(model.body_quat[body_id]),
            "joints": joints_by_body.get(body_id, []),
            "geometries": geoms_by_body.get(body_id, []),
        }
        bodies.append(body)

    camera_intrinsics_by_name = camera_intrinsics_by_name or CAMERA_INTRINSICS_BY_NAME
    cameras = []
    for camera_id in range(model.ncam):
        name = model.camera(camera_id).name
        camera = {
            "name": model.camera(camera_id).name,
            "body": model.body(int(model.cam_bodyid[camera_id])).name,
            "position": _values(model.cam_pos[camera_id]),
            "quaternion": _values(model.cam_quat[camera_id]),
            "fovy": float(model.cam_fovy[camera_id]),
        }
        intrinsics = camera_intrinsics_by_name.get(name)
        if intrinsics is not None:
            camera["intrinsics"] = intrinsics
        cameras.append(camera)
    materials_dict: dict[str, list[float]] = {
        model.mat(mat_id).name: _values(model.mat_rgba[mat_id])
        for mat_id in range(model.nmat)
    }

    return {
        "format": "pick-and-place-web-model",
        "version": 2,
        "materials": materials_dict,
        "bodies": bodies,
        "cameras": cameras,
    }


def _write_outputs(
    spec: mujoco.MjSpec,
    output: Path,
    materials: MaterialConfig | None,
    camera_intrinsics_by_name: dict[str, dict[str, Any]] | None = None,
    *,
    include_local_camera_extrinsics: bool = True,
) -> tuple[Path, Path]:
    """Write matching XML and JSON web-manifest outputs for a composed spec."""
    # The spec was loaded relative to the stock model; rewrite meshdir so the
    # exported file resolves meshes from wherever it is saved.
    camera_intrinsics_by_name = camera_intrinsics_by_name or CAMERA_INTRINSICS_BY_NAME
    _apply_camera_intrinsics(spec, camera_intrinsics_by_name)
    if include_local_camera_extrinsics:
        apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    spec.meshdir = str(STOCK_ASSETS_DIR)
    model = spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    manifest_output = output.with_suffix(".json")
    manifest = web_manifest(
        spec,
        model,
        materials,
        camera_intrinsics_by_name=camera_intrinsics_by_name,
    )
    _assign_mesh_files(manifest, model)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output, manifest_output


def _apply_camera_intrinsics(
    spec: mujoco.MjSpec,
    camera_intrinsics_by_name: dict[str, dict[str, Any]],
) -> None:
    for camera in spec.cameras:
        intrinsics = camera_intrinsics_by_name.get(camera.name)
        if intrinsics is not None and "fovy_deg" in intrinsics:
            camera.fovy = float(intrinsics["fovy_deg"])


def _camera_intrinsics_with_overrides(
    overrides: dict[str, dict[str, Any]] | None,
    *,
    include_local: bool = True,
) -> dict[str, dict[str, Any]]:
    camera_intrinsics_by_name = {
        name: dict(intrinsics)
        for name, intrinsics in CAMERA_INTRINSICS_BY_NAME.items()
    }
    if include_local:
        camera_intrinsics_by_name.update(load_local_camera_intrinsics())
    if overrides:
        camera_intrinsics_by_name.update(overrides)
    return camera_intrinsics_by_name


def export_robot(
    output: Path,
    *,
    wrist_camera: bool = True,
    include_environment: bool = False,
    materials: MaterialConfig | None = None,
    camera_intrinsics: dict[str, dict[str, Any]] | None = None,
    include_local_camera_intrinsics: bool = True,
    include_local_camera_extrinsics: bool = True,
) -> tuple[Path, Path]:
    """Write matching XML and JSON outputs from one composed robot."""
    if include_environment:
        spec = build_scene(
            wrist_camera=wrist_camera,
            materials=materials,
            include_environment=True,
        )
    else:
        spec = build_robot(wrist_camera=wrist_camera, materials=materials)
        spec.body("base").pos = (0.0, 0.0, ROBOT_BASE_Z_OFFSET)
    return _write_outputs(
        spec,
        output,
        materials,
        _camera_intrinsics_with_overrides(
            camera_intrinsics,
            include_local=include_local_camera_intrinsics,
        ),
        include_local_camera_extrinsics=include_local_camera_extrinsics,
    )


def export_environment(
    output: Path,
    *,
    materials: MaterialConfig | None = None,
    camera_intrinsics: dict[str, dict[str, Any]] | None = None,
    include_local_camera_intrinsics: bool = True,
    include_local_camera_extrinsics: bool = True,
) -> tuple[Path, Path]:
    """Write matching XML and JSON outputs for the robot-free environment.

    The web viewer overlays this on the standalone ``so101`` model, so the
    robot lives in exactly one place instead of being duplicated here.
    """
    spec = build_environment(materials=materials)
    return _write_outputs(
        spec,
        output,
        materials,
        _camera_intrinsics_with_overrides(
            camera_intrinsics,
            include_local=include_local_camera_intrinsics,
        ),
        include_local_camera_extrinsics=include_local_camera_extrinsics,
    )


def _parse_camera_intrinsics_args(values: list[str]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected CAMERA=PATH for --camera-intrinsics, got {value!r}")
        camera_name, path = value.split("=", 1)
        if not camera_name:
            raise ValueError(f"missing camera name in --camera-intrinsics {value!r}")
        overrides[camera_name] = load_camera_intrinsics(Path(path))
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None, help="output XML path")
    parser.add_argument(
        "--no-wrist-camera",
        action="store_true",
        help="omit the wrist-camera mount and module",
    )
    parser.add_argument(
        "--include-environment",
        action="store_true",
        help="include the workspace frame and overhead mount in the robot model",
    )
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="export only the robot-free environment (floor, cube, frame, overhead mount)",
    )
    parser.add_argument(
        "--camera-intrinsics",
        action="append",
        default=[],
        metavar="CAMERA=PATH",
        help=(
            "replace nominal intrinsics for a camera with values from a JSON file; "
            "local config/camera_intrinsics/<camera>.json files are loaded automatically"
        ),
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        name = "environment.xml" if args.environment_only else "so101.xml"
        output = Path(__file__).resolve().parents[2] / "out" / name

    try:
        camera_intrinsics = _parse_camera_intrinsics_args(args.camera_intrinsics)
    except ValueError as exc:
        parser.error(str(exc))

    if args.environment_only:
        paths = export_environment(output, camera_intrinsics=camera_intrinsics)
    else:
        paths = export_robot(
            output,
            wrist_camera=not args.no_wrist_camera,
            include_environment=args.include_environment,
            camera_intrinsics=camera_intrinsics,
        )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
