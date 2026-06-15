# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco
import numpy as np

from pick_and_place import build_scene, export_scene
from pick_and_place.workspace_overlays import WORKSPACE_OVERLAYS, WORKSPACE_OVERLAY_GROUP


def test_scene_contains_robot_floor_light_and_cube():
    model = build_scene().compile()

    assert model.body("base").id >= 0
    assert model.body("pick_cube").id >= 0
    assert model.nbody == build_scene(wrist_camera=False).compile().nbody + 2
    floor = model.geom("floor").id
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    assert tuple(model.geom_size[floor, :2]) == (0.0, 0.0)
    assert model.mat(model.geom_matid[floor]).name == "groundplane"
    assert model.texture("groundplane").id >= 0
    assert model.light("scene_light").id >= 0
    assert model.nlight == 1
    assert model.body_jntnum[model.body("pick_cube").id] == 0
    assert tuple(model.geom_size[model.geom("pick_cube").id]) == (0.015, 0.015, 0.015)


def test_scene_contains_non_colliding_workspace_overlays():
    model = build_scene().compile()

    for overlay in WORKSPACE_OVERLAYS:
        geom = model.geom(overlay.name).id
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_group[geom] == WORKSPACE_OVERLAY_GROUP
        assert model.geom_contype[geom] == 0
        assert model.geom_conaffinity[geom] == 0
        assert model.geom_bodyid[geom] == model.body("base").id
        np.testing.assert_allclose(model.geom_rgba[geom], (1.0, 0.4667, 0.0, 0.22), atol=1e-6)


def test_workspace_overlays_follow_robot_base():
    original_model = build_scene().compile()
    original_data = mujoco.MjData(original_model)
    mujoco.mj_forward(original_model, original_data)

    spec = build_scene()
    base = spec.body("base")
    base.pos = (1.0, 2.0, 0.1)
    base.quat = (2**-0.5, 0.0, 0.0, 2**-0.5)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_id = model.body("base").id
    np.testing.assert_allclose(data.xpos[base_id], (1.0, 2.0, 0.1))
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    for overlay in WORKSPACE_OVERLAYS:
        geom = model.geom(overlay.name).id
        original_geom = original_model.geom(overlay.name).id
        assert model.geom_bodyid[geom] == base_id
        np.testing.assert_allclose(
            data.geom_xpos[geom],
            np.array((1.0, 2.0, 0.1)) + rotation @ original_data.geom_xpos[original_geom],
            atol=1e-7,
        )


def test_export_scene_writes_compilable_xml(tmp_path):
    output = export_scene(tmp_path / "scene.xml")

    assert output == tmp_path / "scene.xml"
    assert output.exists()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.body("pick_cube").id >= 0
    assert model.geom("workspace_global").id >= 0
