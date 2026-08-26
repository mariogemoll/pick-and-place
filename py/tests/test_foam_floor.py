# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The foam sheet: where it is, what it clears, and what now rests on it."""

import mujoco
import numpy as np

from pick_and_place.core.workspace_bounds import (
    CUBE_PLACEMENT_BOUNDS,
    WORKSPACE_FRAME_INNER_HALF_EXTENT,
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
    world_to_frame_xy,
)
from pick_and_place.sim.foam_floor import (
    FOAM_FLOOR_BODY_NAME,
    base_cutout_half_width,
    foam_floor_boxes,
    foam_floor_wedges,
)
from pick_and_place.sim.model import cube_qpos_address
from pick_and_place.sim.scene import build_scene
from pick_and_place.spec.workspace import (
    CUBE_HALF_SIZE,
    CUBE_REST_Z,
    FOAM_FLOOR_BASE_CUTOUT_TIP_Y,
    FOAM_FLOOR_CORNER_CUTOUT_INNER,
    FOAM_FLOOR_THICKNESS,
    WORKSPACE_FLOOR_Z,
    WORKSPACE_FRAME_APRILTAG_PLATE_HALF_SIZE,
    WORKSPACE_FRAME_APRILTAG_PLATES,
)


def _on_foam(x: float, y: float) -> bool:
    """Is the frame-local point (x, y) covered by one of the sheet's pieces?"""
    for _, (x_min, x_max, y_min, y_max) in foam_floor_boxes():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
    for _, corners in foam_floor_wedges():
        xs = [corner[0] for corner in corners]
        if not (min(xs) <= x <= max(xs)):
            continue
        if not (corners[0][1] <= y <= corners[2][1]):
            continue
        if abs(x) >= base_cutout_half_width(y):
            return True
    return False


def test_the_sheet_fills_the_square_except_where_something_stands_on_the_table():
    edge = WORKSPACE_FRAME_INNER_HALF_EXTENT
    assert _on_foam(0.0, 0.0)
    assert _on_foam(edge - 0.001, 0.0) and _on_foam(0.0, -edge + 0.001)
    # the corner plates, and the strip between plate and rail, are cut away
    for _, _, (x, y, _) in WORKSPACE_FRAME_APRILTAG_PLATES:
        assert not _on_foam(x, y)
        assert not _on_foam(
            np.sign(x) * (edge - 0.001), np.sign(y) * (edge - 0.001)
        )
        assert _on_foam(x, np.sign(y) * (FOAM_FLOOR_CORNER_CUTOUT_INNER - 0.001))
    # the wedge under the robot follows the base plate, and only the base plate
    for y in np.arange(FOAM_FLOOR_BASE_CUTOUT_TIP_Y + 0.001, WORKSPACE_FRAME_INNER_HALF_EXTENT, 0.005):
        half = base_cutout_half_width(float(y))
        assert not _on_foam(0.0, float(y))
        assert not _on_foam(half - 0.001, float(y))
        assert _on_foam(half + 0.001, float(y))
    assert _on_foam(0.0, FOAM_FLOOR_BASE_CUTOUT_TIP_Y - 0.001)


def test_every_pose_the_cube_may_be_given_is_over_the_foam():
    x_min, x_max, y_min, y_max = CUBE_PLACEMENT_BOUNDS
    checked = 0
    for x in np.linspace(x_min, x_max, 60):
        for y in np.linspace(y_min, y_max, 60):
            if not (is_cube_pickup_allowed(x, y) or is_cube_drop_allowed(x, y)):
                continue
            # the cube's own footprint, not just its centre, must be supported
            for dx in (-CUBE_HALF_SIZE, CUBE_HALF_SIZE):
                for dy in (-CUBE_HALF_SIZE, CUBE_HALF_SIZE):
                    assert _on_foam(*world_to_frame_xy(x + dx, y + dy))
            checked += 1
    assert checked > 500


def test_the_sheet_lies_on_the_table_inside_the_frame():
    model = build_scene().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body = model.body(FOAM_FLOOR_BODY_NAME)
    geoms = [gid for gid in range(model.ngeom) if model.geom_bodyid[gid] == body.id]
    visual = [gid for gid in geoms if model.geom_group[gid] == 2]
    collision = [gid for gid in geoms if model.geom_group[gid] == 3]
    assert len(visual) == len(collision) == len(foam_floor_boxes()) + len(foam_floor_wedges())
    assert all(model.geom_contype[gid] == 0 for gid in visual)
    assert all(model.geom_contype[gid] == 1 for gid in collision)

    corners = []
    for gid in collision:
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            local = model.mesh_vert[
                model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
            ]
        else:
            half = model.geom_size[gid]
            local = np.array(
                [[a, b, c] for a in (-half[0], half[0]) for b in (-half[1], half[1])
                 for c in (-half[2], half[2])]
            )
        corners.append(local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    corners = np.vstack(corners)

    np.testing.assert_allclose(corners[:, 2].min(), 0.0, atol=1e-9)
    np.testing.assert_allclose(corners[:, 2].max(), FOAM_FLOOR_THICKNESS, atol=1e-9)
    frame_xy = np.array([world_to_frame_xy(x, y) for x, y in corners[:, :2]])
    # the surveyed frame quaternion is given to six digits, hence the tolerance
    assert np.abs(frame_xy).max() <= WORKSPACE_FRAME_INNER_HALF_EXTENT + 1e-6


def test_a_dropped_cube_comes_to_rest_at_the_height_the_code_assumes():
    spec = build_scene()
    spec.body("pick_cube").add_freejoint()
    model = spec.compile()
    data = mujoco.MjData(model)
    adr = cube_qpos_address(model)
    data.qpos[adr : adr + 3] = (0.2, -0.12, CUBE_REST_Z + 0.02)
    mujoco.mj_forward(model, data)
    for _ in range(600):
        mujoco.mj_step(model, data)

    resting_z = float(data.qpos[adr + 2])
    np.testing.assert_allclose(resting_z, CUBE_REST_Z, atol=5e-4)
    assert resting_z - CUBE_HALF_SIZE > WORKSPACE_FLOOR_Z - 5e-4


def test_the_corner_plates_still_stand_on_the_bare_table():
    model = build_scene().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for _, corner, _ in WORKSPACE_FRAME_APRILTAG_PLATES:
        plate = model.geom(f"workspace_frame_tag_{corner}").id
        bottom = data.geom_xpos[plate][2] - model.geom_size[plate][2]
        np.testing.assert_allclose(bottom, 0.0, atol=1e-9)
        assert model.geom_size[plate][0] == WORKSPACE_FRAME_APRILTAG_PLATE_HALF_SIZE
