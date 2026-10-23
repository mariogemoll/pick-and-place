# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export a web-manifest JSON for an arbitrary robot_descriptions model.

Unlike ``pick_and_place.sim.export``, this loads the model's stock MJCF directly
instead of composing the project's own SO-101 scene, so it works for any
robot_descriptions ``*_mj_description`` package.

Usage::

    pap export-generic-robot ur5e_mj_description -o public/ur5e.json

A local MJCF file works in place of a robot_descriptions module; its file stem
names the mesh bundle::

    pap export-generic-robot path/to/robot.xml -o public/robot.json

An end effector can be attached to a named site on the base robot::

    pap export-generic-robot ur5e_mj_description \\
        --gripper robotiq_2f85_mj_description -o public/ur5e.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.cli.common import add_output_argument
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.sim.export import web_manifest

GRIPPER_PREFIX = "gripper_"
SETTLE_STEPS = 500
CTRL_SAMPLES = 10
_MIMIC_JOINT_TYPES = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}


def _actuator_joint_ids(model: mujoco.MjModel, act_id: int) -> list[int]:
    """Return the joint(s) an actuator transmits force to directly."""
    trntype = mujoco.mjtTrn(int(model.actuator_trntype[act_id]))
    if trntype == mujoco.mjtTrn.mjTRN_JOINT:
        return [int(model.actuator_trnid[act_id, 0])]
    if trntype == mujoco.mjtTrn.mjTRN_TENDON:
        tendon_id = int(model.actuator_trnid[act_id, 0])
        adr, num = int(model.tendon_adr[tendon_id]), int(model.tendon_num[tendon_id])
        return [int(joint_id) for joint_id in model.wrap_objid[adr:adr + num]]
    return []


def _body_ancestors(model: mujoco.MjModel, body_id: int) -> list[int]:
    chain = [body_id]
    while chain[-1] != 0:
        chain.append(int(model.body_parentid[chain[-1]]))
    return chain


def _lowest_common_ancestor(model: mujoco.MjModel, body_ids: set[int]) -> int:
    chains = [_body_ancestors(model, body_id) for body_id in body_ids]
    common = set.intersection(*(set(chain) for chain in chains))
    return next(body_id for body_id in chains[0] if body_id in common)


def _joints_in_subtree(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    descendants = {root_body_id}
    frontier = [root_body_id]
    while frontier:
        parent = frontier.pop()
        children = [
            body_id for body_id in range(model.nbody)
            if int(model.body_parentid[body_id]) == parent and body_id not in descendants
        ]
        descendants.update(children)
        frontier.extend(children)
    return [
        joint_id for joint_id in range(model.njnt)
        if int(model.jnt_bodyid[joint_id]) in descendants
        and mujoco.mjtJoint(int(model.jnt_type[joint_id])) in _MIMIC_JOINT_TYPES
    ]


def equality_joint_mimics(model: mujoco.MjModel) -> dict[str, dict[str, Any]]:
    """Read linear mimic relations stated directly as joint equality constraints.

    A joint equality with polycoef (c0, c1, 0, 0, 0) means joint1 = c0 + c1 * joint2,
    which is exactly a mimic. Models converted from URDF <mimic> tags state every
    coupling this way, and the exact coefficients beat a fitted approximation.
    Chains (a follower of a follower) are folded into one relation on the chain's
    root joint, since the viewer applies a single level of mimics. Only relations
    from a passive joint to an actuated root are kept: an equality between two
    driven joints (e.g. a gripper's two driver joints) is not a mimic.
    """
    actuated = {
        model.joint(joint_id).name
        for act_id in range(model.nu)
        for joint_id in _actuator_joint_ids(model, act_id)
    }
    direct: dict[str, tuple[str, float, float]] = {}
    for eq_id in range(model.neq):
        if mujoco.mjtEq(int(model.eq_type[eq_id])) != mujoco.mjtEq.mjEQ_JOINT:
            continue
        follower_id, leader_id = int(model.eq_obj1id[eq_id]), int(model.eq_obj2id[eq_id])
        offset, multiplier, *higher = (float(v) for v in model.eq_data[eq_id, :5])
        if leader_id < 0 or any(higher):
            continue
        direct[model.joint(follower_id).name] = (model.joint(leader_id).name, multiplier, offset)

    mimics: dict[str, dict[str, Any]] = {}
    for follower in direct:
        leader, multiplier, offset = direct[follower]
        seen = {follower}
        while leader in direct and leader not in seen:
            seen.add(leader)
            next_leader, next_multiplier, next_offset = direct[leader]
            leader, multiplier, offset = (
                next_leader, multiplier * next_multiplier, offset + multiplier * next_offset
            )
        if follower not in actuated and leader in actuated:
            mimics[follower] = {"joint": leader, "multiplier": multiplier, "offset": offset}
    return mimics


def derive_joint_mimics(model: mujoco.MjModel) -> dict[str, dict[str, Any]]:
    """Derive linear mimic relations for underactuated joint groups.

    Grippers like the Robotiq 2F-85 (and Franka's own hand) expose one joint
    per link of a closed finger linkage, but drive them through a single
    actuator; the rest follow via equality constraints MuJoCo enforces during
    simulation (e.g. the 2F-85's coupler/spring_link/follower joints, which
    aren't even part of the actuator's tendon — they're linked to the driven
    joints only through the closed-loop geometry). The web viewer has no
    physics engine, so for every actuator, this settles its mechanism (the
    actuator's directly transmitted joints, plus any other joint in the same
    body subtree that isn't independently driven by a different actuator) at
    several setpoints and fits each non-primary joint's position as a linear
    function of the directly driven ("primary") joint, to be replayed as a
    simple multiply-and-add in the browser.

    Joints coupled to an actuated joint by an explicit linear joint equality
    take its exact coefficients (see ``equality_joint_mimics``) and are left out
    of the fit, which in a multi-actuator hand also keeps each finger from being
    fitted against the other fingers' actuators.
    """
    explicit = equality_joint_mimics(model)
    mimics = dict(explicit)
    actuated_joint_ids = {
        joint_id
        for act_id in range(model.nu)
        for joint_id in _actuator_joint_ids(model, act_id)
    }

    for act_id in range(model.nu):
        transmitted = [
            joint_id for joint_id in _actuator_joint_ids(model, act_id)
            if mujoco.mjtJoint(int(model.jnt_type[joint_id])) in _MIMIC_JOINT_TYPES
        ]
        if not transmitted:
            continue
        primary_id = transmitted[0]
        primary_name = model.joint(primary_id).name

        body_ids = {int(model.jnt_bodyid[joint_id]) for joint_id in transmitted}
        ancestor = (
            _lowest_common_ancestor(model, body_ids) if len(body_ids) > 1
            else int(model.body_parentid[next(iter(body_ids))])
        )
        passive_ids = [
            joint_id for joint_id in _joints_in_subtree(model, ancestor)
            if joint_id not in actuated_joint_ids and model.joint(joint_id).name not in explicit
        ]
        joint_ids = transmitted + passive_ids
        if len(joint_ids) < 2:
            continue

        data = mujoco.MjData(model)
        saved_gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0
        lo, hi = (float(v) for v in model.actuator_ctrlrange[act_id])
        samples: dict[int, list[tuple[float, float]]] = {joint_id: [] for joint_id in joint_ids}
        for ctrl in np.linspace(lo, hi, CTRL_SAMPLES):
            data.ctrl[act_id] = ctrl
            for _ in range(SETTLE_STEPS):
                mujoco.mj_step(model, data)
            primary_value = float(data.qpos[model.jnt_qposadr[primary_id]])
            for joint_id in joint_ids:
                value = float(data.qpos[model.jnt_qposadr[joint_id]])
                samples[joint_id].append((primary_value, value))
        model.opt.gravity[:] = saved_gravity

        for joint_id in joint_ids[1:]:
            xs, ys = zip(*samples[joint_id])
            multiplier, offset = np.polyfit(xs, ys, 1)
            mimics[model.joint(joint_id).name] = {
                "joint": primary_name,
                "multiplier": float(multiplier),
                "offset": float(offset),
            }
    return mimics


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the generic robot exporter."""
    parser = SuggestingArgumentParser(description=__doc__)
    parser.add_argument(
        "robot",
        help="robot_descriptions module name, e.g. ur5e_mj_description, or an MJCF .xml path",
    )
    add_output_argument(parser, required=True, short=True, help="output JSON path")
    parser.add_argument(
        "--gripper",
        help="robot_descriptions module name for an end effector to attach, e.g. robotiq_2f85_mj_description",
    )
    parser.add_argument(
        "--site",
        default="attachment_site",
        help="site on the base robot to attach --gripper to (default: attachment_site)",
    )
    return parser


def _mjcf_path(robot: str) -> Path:
    """MJCF of a robot_descriptions module, or `robot` itself when it is an .xml path."""
    if robot.endswith(".xml"):
        return Path(robot).resolve()
    return Path(importlib.import_module(f"robot_descriptions.{robot}").MJCF_PATH)


def _robot_name(robot: str) -> str:
    """Web-facing name: the MJCF file stem, or the module name without "_mj_description"."""
    if robot.endswith(".xml"):
        return Path(robot).stem
    return robot.removesuffix("_mj_description")


def run(args: argparse.Namespace) -> None:
    """Write the web manifest for the named robot."""
    spec = mujoco.MjSpec.from_file(str(_mjcf_path(args.robot)))

    if args.gripper:
        gripper_module = importlib.import_module(f"robot_descriptions.{args.gripper}")
        gripper_spec = mujoco.MjSpec.from_file(str(gripper_module.MJCF_PATH))
        spec.attach(gripper_spec, prefix=GRIPPER_PREFIX, site=spec.site(args.site))

    model = spec.compile()

    manifest = web_manifest(spec, model)
    # The mesh pipeline packs all of this robot's meshes into one GLB as named
    # nodes; point every mesh geometry at it.
    mesh_file = f"{_robot_name(args.robot)}.glb"
    for body in manifest["bodies"]:
        for geom in body["geometries"]:
            if geom.get("mesh") is not None:
                geom["meshFile"] = mesh_file

    mimics = derive_joint_mimics(model)
    for body in manifest["bodies"]:
        for joint in body["joints"]:
            mimic = mimics.get(joint["name"])
            if mimic is not None:
                joint["mimic"] = mimic

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
