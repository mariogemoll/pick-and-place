# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.export import export_environment, export_robot


def test_export_robot_writes_matching_xml_and_web_manifest(tmp_path):
    xml_path, json_path = export_robot(
        tmp_path / "so101.xml",
        include_local_camera_intrinsics=False,
        include_local_camera_extrinsics=False,
    )

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
    assert manifest["cameras"][0]["fovy"] == 47.0
    assert manifest["cameras"][0]["intrinsics"]["camera_matrix"] == [
        [1240.0, 0.0, 907.0],
        [0.0, 1240.0, 522.0],
        [0.0, 0.0, 1.0],
    ]
    assert manifest["cameras"][0]["intrinsics"]["dist_coeffs"] == [
        -0.428,
        0.203,
        0.0,
        -0.001,
        -0.049,
    ]
    assert manifest["cameras"][0]["intrinsics"]["approximate"] is True
    assert manifest["cameras"][0]["intrinsics"]["calibration_required"] is True


def test_export_robot_can_omit_wrist_camera(tmp_path):
    _, json_path = export_robot(
        tmp_path / "so101.xml",
        wrist_camera=False,
        include_local_camera_intrinsics=False,
        include_local_camera_extrinsics=False,
    )
    manifest = json.loads(json_path.read_text())

    body_names = {body["name"] for body in manifest["bodies"]}
    assert "wrist_camera_mount" not in body_names
    assert manifest["cameras"] == []


def test_export_environment_includes_overhead_camera_intrinsics(tmp_path):
    _, json_path = export_environment(
        tmp_path / "environment.xml",
        include_local_camera_intrinsics=False,
        include_local_camera_extrinsics=False,
    )
    manifest = json.loads(json_path.read_text())

    assert manifest["cameras"][0]["name"] == "overhead_camera"
    assert manifest["cameras"][0]["fovy"] == 47.0
    assert manifest["cameras"][0]["intrinsics"]["camera_matrix"] == [
        [1240.0, 0.0, 907.0],
        [0.0, 1240.0, 522.0],
        [0.0, 0.0, 1.0],
    ]
    assert manifest["cameras"][0]["intrinsics"]["dist_coeffs"] == [
        -0.428,
        0.203,
        0.0,
        -0.001,
        -0.049,
    ]
    assert manifest["cameras"][0]["intrinsics"]["approximate"] is True
    assert manifest["cameras"][0]["intrinsics"]["calibration_required"] is True


def test_export_robot_can_override_camera_intrinsics_from_json(tmp_path):
    intrinsics_path = tmp_path / "wrist_intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "model": "standard",
                "width": 1920,
                "height": 1080,
                "camera_matrix": [
                    [1200.0, 0.0, 900.0],
                    [0.0, 1205.0, 520.0],
                    [0.0, 0.0, 1.0],
                ],
                "dist_coeffs": [-0.4, 0.2, 0.0, 0.0, -0.05],
                "fovy_deg": 48.0,
            }
        )
    )

    _, json_path = export_robot(
        tmp_path / "so101.xml",
        camera_intrinsics={"wrist_camera": json.loads(intrinsics_path.read_text())},
        include_local_camera_intrinsics=False,
        include_local_camera_extrinsics=False,
    )
    manifest = json.loads(json_path.read_text())

    assert manifest["cameras"][0]["name"] == "wrist_camera"
    assert manifest["cameras"][0]["fovy"] == 48.0
    assert manifest["cameras"][0]["intrinsics"]["camera_matrix"] == [
        [1200.0, 0.0, 900.0],
        [0.0, 1205.0, 520.0],
        [0.0, 0.0, 1.0],
    ]


def test_load_local_camera_intrinsics_reads_known_camera_files(tmp_path):
    camera_intrinsics_dir = tmp_path / "camera_intrinsics"
    camera_intrinsics_dir.mkdir()
    (camera_intrinsics_dir / "wrist_camera.json").write_text(
        json.dumps(
            {
                "model": "standard",
                "width": 1920,
                "height": 1080,
                "camera_matrix": [
                    [1210.0, 0.0, 910.0],
                    [0.0, 1215.0, 530.0],
                    [0.0, 0.0, 1.0],
                ],
                "dist_coeffs": [-0.41, 0.21, 0.0, 0.0, -0.04],
                "fovy_deg": 47.5,
            }
        )
    )

    intrinsics = load_local_camera_intrinsics(camera_intrinsics_dir)

    assert set(intrinsics) == {"wrist_camera"}
    assert intrinsics["wrist_camera"]["fovy_deg"] == 47.5
