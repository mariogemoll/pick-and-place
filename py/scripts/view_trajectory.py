#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the SO-101 through the pick-and-place trajectory under real physics.

The arm is controlled through the model's position-servo actuators: each frame
the trajectory's joint set points are written to ``data.ctrl`` and the simulation
is stepped, so gravity and contact are live. The cube gets a free joint and rests
on the floor as a genuine rigid body.

Phases: (1) neutral -> hover, (2) hover -> pregrasp at cube center, (3) grasp,
(4) lift and carry the grasped cube over to the hover above the target.
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.trajectory import (
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    PickAndCarry,
    pick_and_carry_candidates,
)
from pick_and_place.workspace_overlays import (
    AZIMUTH_MAX,
    AZIMUTH_MIN,
    PAN_AXIS,
    WORKSPACE_OVERLAYS,
)

_CLEARANCE_OVERLAY = next(o for o in WORKSPACE_OVERLAYS if o.name == "workspace_clearance_pregrasp")


def _build_geom_sets(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """Return (robot_geom_ids, env_geom_ids).

    Robot geoms: all geoms on bodies other than the worldbody and the pick_cube.
    Environment geoms: floor and pick_cube — the things we check robot against.
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


def _scan_contacts(
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


def _preflight(
    model: mujoco.MjModel,
    trajectory: PickAndCarry,
    actuator_id: dict[str, int],
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> list[tuple[float, str, str]]:
    """Simulate the full trajectory in a shadow MjData and return collision events."""
    shadow = mujoco.MjData(model)
    for name, value in NEUTRAL_ARM_JOINTS.items():
        _set_joint(model, shadow, name, value)
        shadow.ctrl[actuator_id[name]] = value
    shadow.ctrl[actuator_id["gripper"]] = NEUTRAL_GRIPPER
    mujoco.mj_forward(model, shadow)

    events: list[tuple[float, str, str]] = []
    while shadow.time < trajectory.duration:
        frame = trajectory.evaluate(shadow.time)
        for name, value in frame.joints.items():
            shadow.ctrl[actuator_id[name]] = value
        shadow.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, shadow)
        for n1, n2 in _scan_contacts(model, shadow, robot_geom_ids, env_geom_ids):
            events.append((shadow.time, n1, n2))
    return events


_JAW_PREFIXES = ("fixed_jaw_col", "moving_jaw_col")


def _is_jaw(n: str) -> bool:
    return n.startswith(_JAW_PREFIXES)


def _is_unexpected(n1: str, n2: str) -> bool:
    """False only for jaw↔cube contacts, which are the intentional grasp."""
    return not ((_is_jaw(n1) and n2 == "pick_cube") or (_is_jaw(n2) and n1 == "pick_cube"))


def _random_cube() -> CubePose:
    """Sample a cube pose uniformly inside the clearance-pregrasp annular sector."""
    r_inner, r_outer = _CLEARANCE_OVERLAY.inner_radius, _CLEARANCE_OVERLAY.outer_radius
    # Uniform area sampling: draw r² uniformly so density is flat in 2-D.
    r = math.sqrt(np.random.uniform(r_inner**2, r_outer**2))
    theta = np.random.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
    yaw = np.random.uniform(0.0, 2 * math.pi)
    return CubePose(
        x=PAN_AXIS[0] + r * math.cos(theta),
        y=PAN_AXIS[1] + r * math.sin(theta),
        z=CUBE_HALF_SIZE,
        yaw=yaw,
    )


def _set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="source cube (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="target (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    args = parser.parse_args()

    if args.source is not None:
        source = CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
    else:
        source = _random_cube()
        print(f"source: x={source.x:.4f}  y={source.y:.4f}  yaw={math.degrees(source.yaw):.1f}°")

    if args.target is not None:
        target = CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
    else:
        target = _random_cube()
        print(f"target: x={target.x:.4f}  y={target.y:.4f}  yaw={math.degrees(target.yaw):.1f}°")

    spec = build_scene()
    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    half_yaw = source.yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()  # make the cube a real dynamic body
    model = spec.compile()
    data = mujoco.MjData(model)

    kinematics = derive_kinematics(model)

    # Start at the neutral pose, gripper closed, holding via the servos.
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    for name, value in NEUTRAL_ARM_JOINTS.items():
        _set_joint(model, data, name, value)
        data.ctrl[actuator_id[name]] = value
    data.ctrl[actuator_id["gripper"]] = NEUTRAL_GRIPPER
    mujoco.mj_forward(model, data)

    robot_geom_ids, env_geom_ids = _build_geom_sets(model)

    trajectory = None
    for traj in pick_and_carry_candidates(kinematics, source, target):
        grasp = traj.grasp
        events = _preflight(model, traj, actuator_id, robot_geom_ids, env_geom_ids)
        unexpected = [(t, n1, n2) for t, n1, n2 in events if _is_unexpected(n1, n2)]
        if not unexpected:
            trajectory = traj
            print(f"grasp: face={grasp.face}  elbow={grasp.elbow}  carry={traj.carry.mode}  (pre-flight clean)")
            break
        seen_pairs: set[tuple[str, str]] = set()
        for t, n1, n2 in unexpected:
            key = (min(n1, n2), max(n1, n2))
            if key not in seen_pairs:
                seen_pairs.add(key)
                print(f"skip {grasp.face}/{grasp.elbow}: collision t={t:.3f}s  {n1} ↔ {n2}")
    if trajectory is None:
        raise ValueError("no collision-free pick-and-carry found for this source/target")

    prev_contacts: set[tuple[str, str]] = set()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            frame = trajectory.evaluate(data.time)
            for name, value in frame.joints.items():
                data.ctrl[actuator_id[name]] = value
            data.ctrl[actuator_id["gripper"]] = frame.gripper
            mujoco.mj_step(model, data)
            curr_contacts = {
                (min(n1, n2), max(n1, n2))
                for n1, n2 in _scan_contacts(model, data, robot_geom_ids, env_geom_ids)
                if _is_unexpected(n1, n2)
            }
            for pair in curr_contacts - prev_contacts:
                print(f"collision t={data.time:.3f}s  {pair[0]} ↔ {pair[1]}")
            prev_contacts = curr_contacts
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
