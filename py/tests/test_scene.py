# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco
import numpy as np

from pick_and_place import build_environment, build_scene, export_scene
from pick_and_place.scene import (
    BACKDROP_WALL_DISTANCE,
    BACKDROP_WALL_THICKNESS,
    TABLE_BACKGROUND_RGBA,
    TABLE_EAST_EDGE_Y,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_NORTH_EDGE_X,
    TABLE_THICKNESS,
    TABLE_WEST_EDGE_Y,
    TABLE_WIDTH,
)
from pick_and_place.environment import WORKSPACE_FRAME_APRILTAG_PLATES
from pick_and_place.episodes import _build_model
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import PAPER_TARGET_MARKER_NAME
from pick_and_place.workspace_overlays import WORKSPACE_OVERLAY_GROUP, WORKSPACE_OVERLAYS


def test_scene_contains_robot_floor_light_and_cube():
    model = build_scene().compile()

    assert model.body("base").id >= 0
    assert model.body("pick_cube").id >= 0
    assert model.nbody == build_scene(wrist_camera=False).compile().nbody + 2
    floor = model.geom("floor").id
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    assert tuple(model.geom_size[floor, :2]) == (0.0, 0.0)
    groundplane = model.mat(model.geom_matid[floor])
    assert groundplane.name == "groundplane"
    light_id = model.light("scene_light").id
    assert light_id >= 0
    warm_spotlight_id = model.light("warm_spotlight").id
    assert warm_spotlight_id >= 0
    assert model.nlight == 2
    assert model.light_castshadow[light_id] == 0
    assert model.light_castshadow[warm_spotlight_id] == 0
    np.testing.assert_allclose(model.light_specular[light_id], (0.22, 0.22, 0.22))
    np.testing.assert_allclose(
        model.light_diffuse[warm_spotlight_id], (0.4, 0.28, 0.17)
    )
    assert model.body_jntnum[model.body("pick_cube").id] == 0
    assert tuple(model.geom_size[model.geom("pick_cube").id]) == (0.015, 0.015, 0.015)


def test_tabletop_starts_at_the_north_frame_edge_and_extends_east():
    model = build_scene(tabletop=True).compile()

    table = model.geom("tabletop").id
    background = model.geom("table_background").id
    backdrop_wall = model.geom("backdrop_wall").id
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    np.testing.assert_allclose(
        model.geom_pos[table],
        (
            TABLE_NORTH_EDGE_X + TABLE_LENGTH / 2,
            (TABLE_WEST_EDGE_Y + TABLE_EAST_EDGE_Y) / 2,
            -TABLE_THICKNESS / 2,
        ),
    )
    np.testing.assert_allclose(
        model.geom_size[table],
        (TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_THICKNESS / 2),
    )
    np.testing.assert_allclose(model.geom_pos[table][1] - model.geom_size[table][1], TABLE_WEST_EDGE_Y)
    np.testing.assert_allclose(model.geom_pos[table][1] + model.geom_size[table][1], TABLE_EAST_EDGE_Y)
    assert model.geom_type[background] == mujoco.mjtGeom.mjGEOM_PLANE
    np.testing.assert_allclose(model.geom_pos[background][2], -TABLE_HEIGHT)
    np.testing.assert_allclose(model.geom_rgba[background], TABLE_BACKGROUND_RGBA)
    np.testing.assert_allclose(
        model.geom_pos[backdrop_wall][0] + model.geom_size[backdrop_wall][0],
        TABLE_NORTH_EDGE_X - BACKDROP_WALL_DISTANCE,
    )
    np.testing.assert_allclose(model.geom_size[backdrop_wall][0], BACKDROP_WALL_THICKNESS / 2)
    np.testing.assert_allclose(model.geom_rgba[backdrop_wall], TABLE_BACKGROUND_RGBA)


def test_environment_tabletop_uses_the_replay_lighting():
    model = build_environment(tabletop=True).compile()

    assert model.geom("tabletop").id >= 0
    assert model.geom("table_background").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    light_id = model.light("scene_light").id
    np.testing.assert_allclose(model.vis.headlight.diffuse, (0.6, 0.6, 0.6))
    np.testing.assert_allclose(model.vis.headlight.ambient, (0.3, 0.3, 0.3))
    np.testing.assert_allclose(model.light_diffuse[light_id], (0.35, 0.35, 0.35))
    np.testing.assert_allclose(model.light_ambient[light_id], (0.15, 0.15, 0.15))


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
        assert model.geom_bodyid[geom] == 0
        np.testing.assert_allclose(model.geom_rgba[geom], (1.0, 0.4667, 0.0, 0.22), atol=1e-6)


def test_workspace_overlays_stay_on_worldbody_floor():
    spec = build_scene()
    base = spec.body("base")
    base.pos = (1.0, 2.0, 0.1)
    base.quat = (2**-0.5, 0.0, 0.0, 2**-0.5)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for overlay in WORKSPACE_OVERLAYS:
        geom = model.geom(overlay.name).id
        assert model.geom_bodyid[geom] == 0
        np.testing.assert_allclose(data.geom_xpos[geom][2], overlay.z, atol=1e-7)


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


def test_workspace_frame_board_visuals_are_primitive_boxes():
    model = build_environment().compile()
    frame_id = model.body("workspace_frame_frame").id
    board_names = (
        "north_01",
        "north_02",
        "north_04",
        "north_05",
        "east_01",
        "east_02",
        "east_03",
        "east_04",
        "east_05",
        "south_01",
        "south_02",
        "south_04",
        "south_05",
        "west_01",
        "west_02",
        "west_03",
        "west_04",
        "west_05",
    )

    for name in board_names:
        visual = model.geom(f"workspace_frame_{name}_visual").id
        collision = model.geom(f"workspace_frame_{name}_collision").id

        assert model.geom_type[visual] == mujoco.mjtGeom.mjGEOM_BOX
        assert model.geom_group[visual] == 2
        assert model.geom_bodyid[visual] == frame_id
        assert model.geom_contype[visual] == 0
        assert model.geom_conaffinity[visual] == 0
        assert model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_BOX
        assert model.geom_group[collision] == 3

    north_02_visual = model.geom("workspace_frame_north_02_visual").id
    north_02_collision = model.geom("workspace_frame_north_02_collision").id
    np.testing.assert_allclose(model.geom_pos[north_02_visual], (-0.1325, 0.2813, 0.0036))
    np.testing.assert_allclose(model.geom_size[north_02_visual], (0.063, 0.0187, 0.0036))
    np.testing.assert_allclose(model.geom_pos[north_02_collision], (-0.1325, 0.2813, 0.0036))
    np.testing.assert_allclose(model.geom_size[north_02_collision], (0.053, 0.0187, 0.0036))
    assert model.geom_type[model.geom("workspace_frame_north_03_visual").id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_type[model.geom("overhead_mount_bottom_visual").id] == mujoco.mjtGeom.mjGEOM_MESH


def test_export_scene_writes_compilable_xml(tmp_path):
    output = export_scene(tmp_path / "scene.xml")

    assert output == tmp_path / "scene.xml"
    assert output.exists()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.body("pick_cube").id >= 0
    assert model.geom("workspace_global").id >= 0


def test_episode_model_can_include_drop_zone_marker():
    model, _ = _build_model(
        CubePose(x=0.2, y=-0.1, z=CUBE_HALF_SIZE),
        paper_target_marker=True,
    )

    assert model.body(PAPER_TARGET_MARKER_NAME).id >= 0
    assert model.geom(PAPER_TARGET_MARKER_NAME + "_geom").id >= 0
