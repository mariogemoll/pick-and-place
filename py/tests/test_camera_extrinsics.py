# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

import numpy as np

from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
    save_camera_extrinsics,
)
from pick_and_place.scene import build_environment


def test_save_and_load_camera_extrinsics_sidecar(tmp_path):
    model = build_environment().compile()
    camera_id = model.camera("overhead_camera").id
    model.cam_pos[camera_id] = np.array((0.001, -0.002, 0.003))
    model.cam_quat[camera_id] = np.array((0.99, 0.01, 0.02, 0.03))

    output = save_camera_extrinsics(
        model,
        "overhead_camera",
        path=tmp_path / "camera_extrinsics" / "overhead_camera.json",
        meta={"method": "test"},
    )

    data = json.loads(output.read_text())
    assert data["method"] == "test"
    assert data["cameras"]["overhead_camera"]["pos"] == [0.001, -0.002, 0.003]
    assert load_local_camera_extrinsics(output.parent)["overhead_camera"]["quat"] == [
        0.99,
        0.01,
        0.02,
        0.03,
    ]


def test_apply_camera_extrinsics_to_spec_and_model():
    extrinsics = {
        "overhead_camera": {
            "pos": [0.004, -0.005, 0.006],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    }
    spec = build_environment()

    assert apply_camera_extrinsics_to_spec(spec, extrinsics) == ["overhead_camera"]
    assert tuple(spec.camera("overhead_camera").pos) == (0.004, -0.005, 0.006)

    model = build_environment().compile()
    assert apply_camera_extrinsics_to_model(model, extrinsics) == ["overhead_camera"]
    np.testing.assert_allclose(model.cam_pos[model.camera("overhead_camera").id], (0.004, -0.005, 0.006))
