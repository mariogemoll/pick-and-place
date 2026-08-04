# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import mujoco

from pick_and_place.spec.workspace import CUBE_APRILTAG_IDS
from pick_and_place.sim.scene import build_environment, build_scene


def _texture(spec: mujoco.MjSpec, name: str) -> mujoco.MjsTexture:
    return next(tex for tex in spec.textures if tex.name == name)


def test_pick_cube_faces_carry_their_apriltag_textures():
    spec = build_scene(include_environment=True)

    cube_geom = spec.geom("pick_cube")
    assert cube_geom.material == "pick_cube_apriltags"

    texture = _texture(spec, "pick_cube_apriltags")
    assert texture.type == mujoco.mjtTexture.mjTEXTURE_CUBE
    # cubefiles are ordered right, left, up, down, front, back.
    stems = [file.rsplit("/", 1)[-1] for file in texture.cubefiles]
    expected = [
        f"tagStandard41h12_{tag_id:05d}_30x30mm_tag20mm.png"
        for tag_id in CUBE_APRILTAG_IDS
    ]
    assert stems == expected


def test_pick_cube_apriltags_survive_into_compiled_models():
    for spec in (build_scene(include_environment=True), build_environment()):
        model = spec.compile()
        geom_id = model.geom("pick_cube").id
        mat_id = int(model.geom_matid[geom_id])
        assert mat_id != -1
        assert model.mat(mat_id).name == "pick_cube_apriltags"
        # A cube texture is bound in one of the material's texture-role slots.
        assert any(tex_id != -1 for tex_id in model.mat_texid[mat_id])


def test_apriltag_cube_defaults_to_scene_type():
    # Simple scene -> plain cube; standard scene -> tagged cube.
    simple = build_scene(include_environment=False)
    assert simple.geom("pick_cube").material == ""
    standard = build_scene(include_environment=True)
    assert standard.geom("pick_cube").material == "pick_cube_apriltags"


def test_apriltag_cube_flag_overrides_default():
    tagged_simple = build_scene(include_environment=False, apriltag_cube=True)
    assert tagged_simple.geom("pick_cube").material == "pick_cube_apriltags"
    plain_standard = build_scene(include_environment=True, apriltag_cube=False)
    assert plain_standard.geom("pick_cube").material == ""
    # A plain cube must not leave a dangling apriltag texture/material behind.
    assert not any(mat.name == "pick_cube_apriltags" for mat in plain_standard.materials)
