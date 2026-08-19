#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export the scene and the runtime contract the browser policy page steps.

Writes two files:

``policy-scene.mjb``
    The compiled, asset-free policy scene, saved as a MuJoCo binary model. The
    binary form is what makes the browser rollout *exact*: MuJoCo's XML writer
    rounds every number to six significant figures, while the binary carries the
    compiled model verbatim, so the WebAssembly engine steps the same values
    Python does. It costs about 40 KB more than the XML and removes a whole
    class of "nearly the same" from the comparison.

``policy-scene.json``
    Everything the browser needs that is not in the model: which qpos slot each
    joint occupies, the tracking bias the environment holds the arm against, the
    rates, and the constants the gripper conversion is calibrated on.

Because a binary model is tied to the engine that wrote it, the manifest records
the MuJoCo version. The browser refuses to run against a different one rather
than loading a model it cannot trust.

Usage::

    python scripts/export_web_policy_scene.py -o ../ts/public/policy-scene
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from pick_and_place.core.joint_frames import (
    GRIPPER_READBACK_CLOSED,
    GRIPPER_READBACK_OPEN,
    GRIPPER_RENDER_CLOSED_DEG,
    GRIPPER_RENDER_OPEN_DEG,
)
from pick_and_place.core.physics import PhysicsDraw
from pick_and_place.core.robot_dynamics import (
    load_robot_dynamics_config,
    tracking_bias_rad,
    tracking_bias_vector,
)
from pick_and_place.sim.physics import tracking_bias_offsets
from pick_and_place.sim.physics_export import physics_only_spec
from pick_and_place.spec.robot import (
    HARDWARE_SIMULATION_HZ,
    JOINT_NAMES,
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
)
from pick_and_place.spec.workspace import CUBE_HALF_SIZE, DROP_ZONE_HALF_SIZE

#: Rate the state flow policy was exported at, and therefore the rate one
#: environment step covers. ``PolicySimEnv`` takes its substep count from the
#: scenario's control rate, and the flow export declares 10 Hz.
POLICY_HZ = 10.0

PICK_CUBE_BODY = "pick_cube"


def scene_manifest(model: mujoco.MjModel) -> dict[str, object]:
    """Describe the compiled model in the terms the browser runtime needs."""
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    actuator_ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index): index
        for index in range(model.nu)
    }
    cube_joint = model.body(PICK_CUBE_BODY).jntadr[0]
    # The droop the environment holds the arm against, taken through the same
    # scaling the environment applies rather than straight from the fit. The
    # nominal draw scales it to zero -- the fitted bias belongs to a *physics
    # randomized* episode, and a page that added it unconditionally would push
    # every command past its target by a couple of degrees at the shoulder.
    bias = tracking_bias_vector(
        tracking_bias_offsets(tracking_bias_rad(load_robot_dynamics_config()), PhysicsDraw()),
        JOINT_NAMES,
    )
    neutral_sim = [NEUTRAL_ARM_JOINTS[name] for name in JOINT_NAMES[:-1]] + [NEUTRAL_GRIPPER]
    return {
        "format": "pick-and-place-policy-scene",
        "version": 1,
        "mujocoVersion": mujoco.__version__,
        "model": "policy-scene.mjb",
        "timestep": 1.0 / HARDWARE_SIMULATION_HZ,
        "simulationHz": HARDWARE_SIMULATION_HZ,
        "policyHz": POLICY_HZ,
        "substeps": round((1.0 / POLICY_HZ) * HARDWARE_SIMULATION_HZ),
        "jointNames": list(JOINT_NAMES),
        "jointQposAdr": [int(model.jnt_qposadr[i]) for i in joint_ids],
        "ctrlIndex": [int(actuator_ids[name]) for name in JOINT_NAMES],
        "trackingBiasRad": [float(v) for v in bias],
        "neutralJointsRad": [float(v) for v in neutral_sim],
        "cubeQposAdr": int(model.jnt_qposadr[cube_joint]),
        "cubeDofAdr": int(model.jnt_dofadr[cube_joint]),
        "cubeHalfSize": CUBE_HALF_SIZE,
        "dropZoneHalfSize": DROP_ZONE_HALF_SIZE,
        "gripperCalibration": {
            "readbackClosed": GRIPPER_READBACK_CLOSED,
            "readbackOpen": GRIPPER_READBACK_OPEN,
            "renderClosedDeg": GRIPPER_RENDER_CLOSED_DEG,
            "renderOpenDeg": GRIPPER_RENDER_OPEN_DEG,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="path prefix; '.mjb' and '.json' are appended",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = physics_only_spec().compile()
    binary = args.output.with_suffix(".mjb")
    mujoco.mj_saveModel(model, str(binary), None)

    manifest = scene_manifest(model)
    manifest["model"] = binary.name
    with args.output.with_suffix(".json").open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print(f"wrote {binary} ({binary.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {args.output.with_suffix('.json')}")
    print(f"  bodies {model.nbody} geoms {model.ngeom} meshes {model.nmesh} textures {model.ntex}")


if __name__ == "__main__":
    main()
