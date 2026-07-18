#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Bench sign check for the sim->real arm-joint mapping.

Parks the real arm at a clear inspection pose, then commands one arm joint at a
time to ``base + delta`` (in the *sim* frame) and back. Before each move it
prints the model's forward-kinematics prediction for how the wrist camera should
physically move (up/down and horizontally), so the operator can confirm the arm
moves the predicted way.

If every joint matches its prediction, the sim->real mapping carries no
per-joint sign flip and the fitted joint-zero offsets apply feed-forward as
``real_deg = degrees(sim_rad) - offset_deg`` (offsets are the amount to *add* to
the sim joints). If a joint moves the opposite way, that joint's correction must
be applied with the flipped sign.

Everything runs open loop; no cube or cameras are needed.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from pick_and_place.episodes import _build_model
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    make_so101_follower,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.overhead_detection import MockViewer
from pick_and_place.session_calibration import CalibrationConfig, _move_arm_to
from pick_and_place.trajectory import GRIPPER_OPEN
from pick_and_place.workspace_overlays import PAN_AXIS

# A clear, mid-range pose: arm lifted, forearm angled up, wrist camera upright.
INSPECTION_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": math.radians(-50.0),
    "elbow_flex": math.radians(55.0),
    "wrist_flex": math.radians(20.0),
    "wrist_roll": -math.pi / 2.0,
}


def _joint_ranges(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        ranges[name] = (float(lo), float(hi))
    return ranges


def _wrist_cam_pos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addrs: dict[str, int],
    pose: dict[str, float],
    cam_id: int,
) -> np.ndarray:
    for name in ARM_JOINT_NAMES:
        data.qpos[qpos_addrs[name]] = pose[name]
    mujoco.mj_forward(model, data)
    return data.cam_xpos[cam_id].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", required=True, help="follower serial port")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id")
    parser.add_argument(
        "--joints",
        default=",".join(ARM_JOINT_NAMES[:4]),
        help="comma-separated arm joints to sweep (default: the four fitted joints)",
    )
    parser.add_argument(
        "--delta-deg", type=float, default=8.0, help="sim-frame perturbation per joint (deg)"
    )
    parser.add_argument("--cycles", type=int, default=2, help="base<->offset oscillations per joint")
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    args = parser.parse_args()

    joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    for j in joints:
        if j not in ARM_JOINT_NAMES:
            raise SystemExit(f"Unknown joint {j!r}; choose from {ARM_JOINT_NAMES}.")

    print("Building scene...")
    dummy = CubePose(x=PAN_AXIS[0] + 0.24, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(dummy, include_environment=False)
    mujoco.mj_forward(model, data)

    qpos_addrs = {
        n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in (*ARM_JOINT_NAMES, "gripper")
    }
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    ranges = _joint_ranges(model)

    delta = math.radians(args.delta_deg)
    base = dict(INSPECTION_POSE)

    print("Connecting to follower...")
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    config = CalibrationConfig()
    viewer_ctx = MockViewer() if not args.viewer else mujoco.viewer.launch_passive(model, data)
    try:
        with viewer_ctx as viewer:
            input("Clear the arm's workspace, then press Enter to move to the inspection pose...")
            _move_arm_to(follower, base, GRIPPER_OPEN, model, data, qpos_addrs, viewer, config)

            base_cam = _wrist_cam_pos(model, data, qpos_addrs, base, cam_id)
            for name in joints:
                lo, hi = ranges[name]
                target = float(np.clip(base[name] + delta, lo, hi))
                applied_deg = math.degrees(target - base[name])
                if abs(applied_deg) < 0.5:
                    print(f"\n[{name}] base already at a limit; skipping.")
                    continue

                perturbed = dict(base)
                perturbed[name] = target
                pert_cam = _wrist_cam_pos(model, data, qpos_addrs, perturbed, cam_id)
                d = pert_cam - base_cam
                vertical = (
                    "UP" if d[2] > 0.003 else "DOWN" if d[2] < -0.003 else "≈level"
                )
                # Real-frame command deltas (what the servos are told): a pure
                # radians->degrees map, so this equals the sim-frame delta.
                real_base = sim_frame_to_real(base, GRIPPER_OPEN)
                real_pert = sim_frame_to_real(perturbed, GRIPPER_OPEN)
                cmd_deg = (real_pert - real_base)[ARM_JOINT_NAMES.index(name)]

                print(f"\n[{name}]  sim +{applied_deg:+.1f}deg  ->  servo command {cmd_deg:+.1f}deg")
                print(
                    f"  model predicts the wrist camera moves {vertical}: "
                    f"dz={d[2] * 1000:+.0f}mm, dx={d[0] * 1000:+.0f}mm, dy={d[1] * 1000:+.0f}mm "
                    f"(world x=+{PAN_AXIS[0]:.2f} axis forward)"
                )
                input("  Press Enter to run the oscillation; watch the arm...")

                for _ in range(max(1, args.cycles)):
                    _move_arm_to(follower, perturbed, GRIPPER_OPEN, model, data, qpos_addrs, viewer, config)
                    _move_arm_to(follower, base, GRIPPER_OPEN, model, data, qpos_addrs, viewer, config)

                answer = input(f"  Did {name} move {vertical} (as predicted)? [y/n/skip]: ").strip().lower()
                if answer.startswith("n"):
                    print(f"  *** {name}: MISMATCH — this joint's correction needs the FLIPPED sign. ***")
                elif answer.startswith("y"):
                    print(f"  {name}: OK — no sign flip; subtract offset feed-forward.")

            print("\nReturning to the inspection pose. Sweep complete.")
            _move_arm_to(follower, base, GRIPPER_OPEN, model, data, qpos_addrs, viewer, config)
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
