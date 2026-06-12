# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

from pick_and_place.export import export_robot


def test_export_robot_writes_matching_xml_and_web_manifest(tmp_path):
    xml_path, json_path = export_robot(tmp_path / "so101.xml")

    assert xml_path.exists()
    assert json_path == tmp_path / "so101.json"
    manifest = json.loads(json_path.read_text())

    assert manifest["format"] == "pick-and-place-web-model"
    bodies = {body["name"]: body for body in manifest["bodies"]}
    assert bodies["shoulder"]["parent"] == "base"
    assert bodies["shoulder"]["joints"][0]["name"] == "shoulder_pan"
    assert bodies["shoulder"]["joints"][0]["range"] == [
        -1.9198621771937616,
        1.9198621771937634,
    ]
    assert bodies["upper_arm"]["quaternion"] == [0.5, -0.5, -0.5, -0.5]

    all_geometries = [
        geometry for body in manifest["bodies"] for geometry in body["geometries"]
    ]
    assert any(geometry["mesh"] == "base_so101_v2.glb" for geometry in all_geometries)
    base_motor_holder = next(
        geometry
        for geometry in all_geometries
        if geometry.get("mesh") == "base_motor_holder_so101_v1.glb"
    )
    assert base_motor_holder["position"] == [-0.00636471, -9.94414e-05, -0.0024]
    assert base_motor_holder["quaternion"] == [0.5, 0.5, 0.5, 0.5]
    assert "scale" not in base_motor_holder
    assert any(geometry["name"] == "base_col0" for geometry in all_geometries)
    assert any(geometry["name"] == "wrist_camera_board_visual" for geometry in all_geometries)
    assert any(geometry["name"] == "wrist_camera_board_collision" for geometry in all_geometries)
    wrist_camera_mount = next(
        geometry
        for geometry in all_geometries
        if geometry.get("mesh") == "SO-ARM101_camera_wrist_mount.glb"
    )
    assert wrist_camera_mount["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert all("scale" not in geometry for geometry in all_geometries)
    assert manifest["cameras"][0]["name"] == "wrist_camera"


def test_export_robot_can_omit_wrist_camera(tmp_path):
    _, json_path = export_robot(tmp_path / "so101.xml", wrist_camera=False)
    manifest = json.loads(json_path.read_text())

    body_names = {body["name"] for body in manifest["bodies"]}
    assert "wrist_camera_mount" not in body_names
    assert manifest["cameras"] == []
