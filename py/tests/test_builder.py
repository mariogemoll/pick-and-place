# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco
import numpy as np

from pick_and_place import add_camera_module, build_robot
from pick_and_place.camera_module import BOARD_HALF_SIZE, LENS_HALF_LENGTH, LENS_POS, LENS_RADIUS
from pick_and_place.collision_boxes import COLLISION_BOXES
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.robot_dynamics import load_robot_dynamics_config
from pick_and_place.wrist_camera import _MOUNT_VISUAL_POS
from pick_and_place.wrist_camera_mount_collision_boxes import (
    WRIST_CAMERA_MOUNT_COLLISION_BOXES,
)


def box_names():
    return [box.name for boxes in COLLISION_BOXES.values() for box in boxes]


def test_shoulder_uses_two_collision_boxes():
    boxes = COLLISION_BOXES["shoulder"]

    assert [box.name for box in boxes] == ["shoulder_col0", "shoulder_col1"]
    np.testing.assert_allclose(
        [box.size for box in boxes], [(0.025, 0.031, 0.028), (0.028, 0.019, 0.027)]
    )
    np.testing.assert_allclose(
        [box.pos for box in boxes],
        [
            (-0.0181991995, 0.000162462614, 0.0188999997),
            (-0.0321991995, 0.00116246261, -0.0381000003),
        ],
    )


def test_upper_arm_uses_two_collision_boxes():
    boxes = COLLISION_BOXES["upper_arm"]

    assert [box.name for box in boxes] == ["upper_arm_col0", "upper_arm_col1"]
    np.testing.assert_allclose(
        [box.size for box in boxes], [(0.012, 0.034, 0.052), (0.026, 0.025, 0.019)]
    )
    np.testing.assert_allclose(
        [box.pos for box in boxes],
        [
            (-0.039084999, 0.000899999723, 0.0201499995),
            (-0.112084999, -0.0131000003, 0.0191499995),
        ],
    )


def test_lower_arm_uses_two_collision_boxes():
    boxes = COLLISION_BOXES["lower_arm"]

    assert [box.name for box in boxes] == ["lower_arm_col0", "lower_arm_col1"]
    np.testing.assert_allclose(
        [box.size for box in boxes], [(0.012, 0.032, 0.050), (0.018, 0.028, 0.027)]
    )
    np.testing.assert_allclose(
        [box.pos for box in boxes],
        [
            (-0.0385499965, -0.000550141265, 0.0201997877),
            (-0.117549996, 0.00344985871, 0.0202001922),
        ],
    )


def test_wrist_uses_three_collision_boxes():
    boxes = COLLISION_BOXES["wrist"]

    assert [box.name for box in boxes] == ["wrist_col0", "wrist_col1", "wrist_col2"]
    np.testing.assert_allclose(
        [box.size for box in boxes],
        [(0.012, 0.032, 0.017), (0.016, 0.026, 0.007), (0.018, 0.032, 0.012)],
    )
    np.testing.assert_allclose(
        [box.pos for box in boxes],
        [
            (-0.000795616758, -0.00584594182, 0.0221499983),
            (-0.0000924933516, -0.0578411875, 0.028150002),
            (-0.00237626471, -0.0368701509, 0.0221500008),
        ],
    )


def test_fixed_jaw_refits_only_holder_boxes():
    boxes = COLLISION_BOXES["gripper"]
    holder_boxes = [box for box in boxes if box.name in {"fixed_jaw_col0a", "fixed_jaw_col0b"}]
    tuned_boxes = [box for box in boxes if box.name.startswith("fixed_jaw_col")][2:]

    assert [box.name for box in holder_boxes] == ["fixed_jaw_col0a", "fixed_jaw_col0b"]
    np.testing.assert_allclose(
        [box.size for box in holder_boxes], [(0.02325, 0.024, 0.012), (0.03225, 0.02625, 0.00825)]
    )
    assert [(box.name, box.pos, box.size) for box in tuned_boxes] == [
        ("fixed_jaw_col1", (-0.0244, 0.0, -0.041025), (0.00941, 0.016, 0.00412)),
        ("fixed_jaw_col2", (-0.02291, 0.0, -0.05485), (0.00962, 0.01134, 0.00981)),
        ("fixed_jaw_col3", (-0.0176, 0.0, -0.0745), (0.00601, 0.00805, 0.0098)),
        ("fixed_jaw_col4", (-0.01492, -0.00018, -0.0893), (0.00503, 0.00703, 0.005)),
        ("fixed_jaw_col5", (-0.01189, -0.00015, -0.099363), (0.004, 0.00545, 0.005063)),
    ]


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


def test_robot_applies_fitted_actuator_time_constants_by_default():
    model = build_robot().compile()
    dynamics = load_robot_dynamics_config()

    for name in JOINT_NAMES:
        actuator_id = model.actuator(name).id
        assert model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_FILTEREXACT
        assert model.actuator_dynprm[actuator_id, 0] == dynamics["joints"][name]["time_constant_s"]


def test_robot_dynamics_can_be_disabled_for_stock_actuators():
    model = build_robot(robot_dynamics=False).compile()

    for name in JOINT_NAMES:
        actuator_id = model.actuator(name).id
        assert model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_NONE


def test_robot_shoulder_pan_range_has_no_self_contacts():
    model = build_robot().compile()
    data = mujoco.MjData(model)
    shoulder_pan_qpos = model.joint("shoulder_pan").qposadr[0]

    for angle in np.linspace(*model.jnt_range[model.joint("shoulder_pan").id], 17):
        data.qpos[shoulder_pan_qpos] = angle
        mujoco.mj_forward(model, data)

        assert data.ncon == 0


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
