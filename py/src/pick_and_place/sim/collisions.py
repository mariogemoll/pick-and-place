# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Which contacts count as a collision, and how close the jaws come to the floor.

Every check here reads a compiled model: which geoms belong to the robot, which
belong to the things it must not hit, and whether a given contact is the
intentional grasp or a real collision. The cheap per-configuration checker
screens candidate motions during planning; stepping the dynamics to vet a whole
trajectory is :mod:`pick_and_place.runtime.preflight`, which is the
authoritative check.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np

from pick_and_place.spec.robot import ARM_JOINT_NAMES


def build_geom_sets(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """Return (robot_geom_ids, env_geom_ids).

    Robot geoms: all geoms on bodies other than the worldbody and the pick_cube.
    Environment geoms: floor and pick_cube — the things we check the robot against.
    """
    world_body_id = 0
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    robot_geom_ids = {
        gid
        for gid in range(model.ngeom)
        if model.geom_bodyid[gid] not in (world_body_id, cube_body_id)
    }
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return robot_geom_ids, {cube_geom_id, floor_geom_id}


def scan_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> list[tuple[str, str]]:
    """Return (name1, name2) for robot↔environment and robot↔robot contacts."""
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        g1_robot = g1 in robot_geom_ids
        g2_robot = g2 in robot_geom_ids
        if (g1_robot and g2 in env_geom_ids) or (g2_robot and g1 in env_geom_ids) or (g1_robot and g2_robot):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            hits.append((n1, n2))
    return hits


_JAW_PREFIXES = ("fixed_jaw_col", "moving_jaw_col")


def _is_jaw(n: str) -> bool:
    return n.startswith(_JAW_PREFIXES)


def is_unexpected(n1: str, n2: str) -> bool:
    """False only for jaw↔cube contacts, which are the intentional grasp."""
    return not ((_is_jaw(n1) and n2 == "pick_cube") or (_is_jaw(n2) and n1 == "pick_cube"))


def make_carry_collision_checker(
    model: mujoco.MjModel,
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> Callable[[dict[str, float]], bool]:
    """Build a cheap per-configuration collision check for screening carry
    candidates during planning, before committing to the much more expensive
    full-trajectory preflight (``preflight``, which steps real dynamics).

    Uses pure kinematics + collision detection -- no integration, no contact
    dynamics -- so it's fast enough to run on every candidate. The cube isn't
    positioned here (its pose isn't tracked by this cheap check), so it's
    excluded from the environment set entirely; the full preflight remains the
    authoritative check for anything cube-related.
    """
    shadow = mujoco.MjData(model)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    env_geom_ids_no_cube = {gid for gid in env_geom_ids if model.geom_bodyid[gid] != cube_body_id}
    qpos_adr = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ARM_JOINT_NAMES
    }

    def check(joints: dict[str, float]) -> bool:
        for name, value in joints.items():
            shadow.qpos[qpos_adr[name]] = value
        mujoco.mj_kinematics(model, shadow)
        mujoco.mj_collision(model, shadow)
        for i in range(shadow.ncon):
            contact = shadow.contact[i]
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            g1_robot = g1 in robot_geom_ids
            g2_robot = g2 in robot_geom_ids
            if not (
                (g1_robot and g2 in env_geom_ids_no_cube)
                or (g2_robot and g1 in env_geom_ids_no_cube)
                or (g1_robot and g2_robot)
            ):
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            if is_unexpected(n1, n2):
                return False
        return True

    return check


def jaw_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Geom ids of the gripper-jaw collision boxes — the lowest-reaching robot
    parts near neutral, hence what the start-pose floor clearance is measured on."""
    return [
        gid
        for gid in range(model.ngeom)
        if _is_jaw(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
    ]


def jaw_floor_clearance(
    model: mujoco.MjModel, data: mujoco.MjData, jaw_ids: list[int]
) -> float:
    """Height of the lowest gripper-jaw corner above the floor for the current
    (already forward-kinematics'd) configuration. The jaws are oriented boxes, so
    each box's lowest corner is its center minus the world-z extent of its half
    sizes; the minimum over all jaw geoms is the clearance."""
    clearance = math.inf
    for gid in jaw_ids:
        half_sizes = model.geom_size[gid]
        world_z_row = data.geom_xmat[gid].reshape(3, 3)[2]
        half_height = float(np.abs(world_z_row) @ half_sizes)
        clearance = min(clearance, float(data.geom_xpos[gid][2]) - half_height)
    return clearance
