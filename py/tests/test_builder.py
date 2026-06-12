# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco

from pick_and_place import build_robot
from pick_and_place.collision_boxes import COLLISION_BOXES


def box_names():
    return [box.name for boxes in COLLISION_BOXES.values() for box in boxes]


def test_plain_robot_has_box_collisions_only():
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


def test_grip_geoms_have_contact_params():
    model = build_robot().compile()
    for body_name, boxes in COLLISION_BOXES.items():
        for box in boxes:
            if not box.grip:
                continue
            gid = model.geom(box.name).id
            assert model.geom_condim[gid] == 4, box.name
            assert model.geom_friction[gid][0] == 2.0, box.name


