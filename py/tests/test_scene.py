# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco
import numpy as np

from pick_and_place import build_environment, build_scene, export_scene
from pick_and_place.environment import WORKSPACE_FRAME_APRILTAG_PLATES
from pick_and_place.workspace_overlays import WORKSPACE_OVERLAY_GROUP, WORKSPACE_OVERLAYS


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


def test_robot_visual_geoms_are_visible():
    model = build_scene().compile()
    base_id = model.body("base").id
    visual_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == base_id and int(model.geom_group[geom_id]) == 2
    ]

    assert visual_geoms
    for geom_id in visual_geoms:
        assert model.geom_rgba[geom_id, 3] > 0


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


def test_environment_contains_textured_workspace_frame_apriltags():
    model = build_environment().compile()
    frame_id = model.body("workspace_frame_frame").id

    for tag_id, corner_name, pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        geom = model.geom(f"workspace_frame_tag_{corner_name}").id
        material = model.geom_matid[geom]
        texture = model.texture(f"workspace_frame_apriltag_{tag_id:02d}").id

        assert model.tex_type[texture] == mujoco.mjtTexture.mjTEXTURE_CUBE
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
        assert model.geom_group[geom] == 2
        assert model.geom_bodyid[geom] == frame_id
        assert model.geom_contype[geom] == 0
        assert model.geom_conaffinity[geom] == 0
        np.testing.assert_allclose(model.geom_size[geom], (0.03, 0.03, 0.0025))
        np.testing.assert_allclose(model.geom_pos[geom], pos)
        assert model.mat(material).name == f"workspace_frame_apriltag_{tag_id:02d}_material"
        assert model.mat_texid[material][1] == texture


def test_export_scene_writes_compilable_xml(tmp_path):
    output = export_scene(tmp_path / "scene.xml")

    assert output == tmp_path / "scene.xml"
    assert output.exists()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.body("pick_cube").id >= 0
    assert model.geom("workspace_global").id >= 0
