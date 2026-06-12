# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco
import numpy as np

from pick_and_place import add_camera_module, build_robot
from pick_and_place.camera_module import BOARD_HALF_SIZE, LENS_HALF_LENGTH, LENS_POS, LENS_RADIUS
from pick_and_place.collision_boxes import COLLISION_BOXES
from pick_and_place.wrist_camera_mount_collision_boxes import (
    WRIST_CAMERA_MOUNT_COLLISION_BOXES,
)
from pick_and_place.wrist_camera import _MOUNT_VISUAL_POS


def box_names():
    return [box.name for boxes in COLLISION_BOXES.values() for box in boxes]


def test_robot_has_box_collisions_only():
    model = build_robot().compile()
    names = [model.geom(i).name for i in range(model.ngeom)]
    for name in box_names():
        assert name in names
    colliding_meshes = [
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[i] != 0
    ]
    assert colliding_meshes == []


def test_wrist_camera_is_on_by_default_and_can_be_disabled():
    model = build_robot().compile()

    assert model.body("wrist_camera_mount").id >= 0
    assert model.body("wrist_camera_module").id >= 0
    assert model.geom("wrist_camera_mount_visual").id >= 0
    assert model.camera("wrist_camera").id >= 0
    for box in WRIST_CAMERA_MOUNT_COLLISION_BOXES:
        assert model.geom(box.name).id >= 0

    plain_model = build_robot(wrist_camera=False).compile()
    names = [plain_model.body(i).name for i in range(plain_model.nbody)]
    assert "wrist_camera_mount" not in names
    assert "wrist_camera_module" not in names


def test_wrist_camera_mount_mesh_is_canonical_before_placement():
    spec = build_robot()
    mount_mesh = spec.mesh("SO-ARM101_camera_wrist_mount")
    mount_geom = spec.geom("wrist_camera_mount_visual")

    assert mount_mesh.file == ""
    np.testing.assert_allclose(mount_mesh.scale, (1.0, 1.0, 1.0))
    vertices = np.asarray(mount_mesh.uservert).reshape(-1, 3)
    np.testing.assert_allclose(
        np.ptp(vertices, axis=0),
        (0.035, 0.066093338, 0.042592957),
        atol=1e-7,
    )
    np.testing.assert_allclose(mount_geom.pos, _MOUNT_VISUAL_POS)
    np.testing.assert_allclose(mount_geom.pos, (-0.015086, 0.024001, -0.031666))
    np.testing.assert_allclose(mount_geom.quat, (1.0, 0.0, 0.0, 0.0))


def test_grip_geoms_have_contact_params():
    model = build_robot().compile()
    for body_name, boxes in COLLISION_BOXES.items():
        for box in boxes:
            if not box.grip:
                continue
            gid = model.geom(box.name).id
            assert model.geom_condim[gid] == 4, box.name
            assert model.geom_friction[gid][0] == 2.0, box.name


def test_camera_module_adds_shared_visual_and_collision_geometry():
    spec = mujoco.MjSpec()
    add_camera_module(
        spec.worldbody,
        prefix="wrist_",
        pos=(0.1, 0.2, 0.3),
        quat=(0.70710678, 0.70710678, 0.0, 0.0),
    )
    model = spec.compile()

    body = model.body("wrist_camera_module")
    np.testing.assert_allclose(model.body_pos[body.id], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(
        model.body_quat[body.id], (0.70710678, 0.70710678, 0.0, 0.0), atol=1e-7
    )

    board_visual = model.geom("wrist_camera_board_visual").id
    board_collision = model.geom("wrist_camera_board_collision").id
    lens_visual = model.geom("wrist_camera_lens_visual").id
    lens_collision = model.geom("wrist_camera_lens_collision").id

    np.testing.assert_allclose(model.geom_size[board_visual], BOARD_HALF_SIZE)
    np.testing.assert_allclose(model.geom_size[board_collision], BOARD_HALF_SIZE)
    np.testing.assert_allclose(model.geom_pos[lens_visual], LENS_POS)
    np.testing.assert_allclose(model.geom_pos[lens_collision], LENS_POS)
    np.testing.assert_allclose(model.geom_size[lens_visual, :2], (LENS_RADIUS, LENS_HALF_LENGTH))
    np.testing.assert_allclose(model.geom_size[lens_collision, :2], (LENS_RADIUS, LENS_HALF_LENGTH))
    assert model.geom_contype[board_visual] == 0
    assert model.geom_contype[lens_visual] == 0
    assert model.geom_contype[board_collision] != 0
    assert model.geom_contype[lens_collision] != 0
    assert model.site("wrist_camera_frame").id >= 0


def test_camera_module_collision_can_be_disabled():
    spec = mujoco.MjSpec()
    add_camera_module(spec.worldbody, prefix="overhead_", collision=False)
    model = spec.compile()
    names = [model.geom(i).name for i in range(model.ngeom)]

    assert names == ["overhead_camera_board_visual", "overhead_camera_lens_visual"]
