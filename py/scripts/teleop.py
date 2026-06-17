#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Teleoperate the simulated SO-101 arm using a physical SO-101 leader."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    action_to_joints,
    joints_to_action,
    load_follower_joint_offsets,
    make_so101_leader,
    make_so101_follower,
    real_frame_to_sim,
)

def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leader-port", required=True, help="Serial port of the SO-101 leader"
    )
    parser.add_argument(
        "--leader-id", default="liddy", help="Leader ID (default: liddy)"
    )
    parser.add_argument(
        "--follower-port", default=None, help="Optional serial port of the SO-101 follower to mirror to"
    )
    parser.add_argument(
        "--follower-id", default="folly", help="Follower ID (default: folly)"
    )
    parser.add_argument(
        "--offsets-path", default=None, help="JSON of per-joint sim→real degree offsets"
    )
    parser.add_argument(
        "--sim-tracks", choices=["leader", "follower"], default="leader",
        help="Whether the sim displays the leader's target pose or the follower's actual readback pose."
    )
    parser.add_argument(
        "--kinematic", action="store_true",
        help="Teleport the simulation joints instantly rather than driving them with actuators. (Forced true if --sim-tracks=follower)."
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Teleop loop rate (Hz)")
    args = parser.parse_args()

    offsets = load_follower_joint_offsets(args.offsets_path)

    print(f"Connecting to leader on {args.leader_port}...")
    leader = make_so101_leader(args.leader_port, args.leader_id)
    # Perform an interactive calibration if this is the first time, 
    # otherwise load existing files.
    leader.connect(calibrate=True)
    print("Leader connected.")

    follower = None
    follower_start_joints = None
    if args.follower_port is not None:
        print(f"Connecting to follower on {args.follower_port}...")
        follower = make_so101_follower(
            args.follower_port,
            args.follower_id,
            disable_torque_on_disconnect=False,
        )
        follower.connect(calibrate=True)
        # Capture the follower's initial parked pose so we can ramp from it
        follower_obs = follower.get_observation()
        follower_start_joints = action_to_joints(follower_obs, np.zeros(6, dtype=float))
        print("Follower connected. Torque will remain on after disconnect.")

    spec = build_scene(include_environment=True)
    # Add a dynamic pick cube if we want something to interact with
    cube = spec.body("pick_cube")
    cube.add_freejoint()
    
    model = spec.compile()
    data = mujoco.MjData(model)

    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i 
        for i in range(model.nu)
    }

    # Initialize to the leader's actual pose so the sim doesn't snap abruptly
    leader_action = leader.get_action()
    real_joints = action_to_joints(leader_action, np.zeros(6, dtype=float))
    arm_rad, gripper_rad = real_frame_to_sim(real_joints, offsets)

    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = arm_rad[name]
        data.ctrl[actuator_id[name]] = arm_rad[name]
        
    g_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[model.jnt_qposadr[g_jid]] = gripper_rad
    data.ctrl[actuator_id["gripper"]] = gripper_rad
    
    mujoco.mj_forward(model, data)

    dt = 1.0 / args.fps
    ramp_duration = 4.0
    print("Starting simulation teleop. Press Ctrl+C to stop.")
    loop_start_time = time.perf_counter()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()
                elapsed_total = step_start - loop_start_time

                obs = leader.get_action()
                leader_joints = action_to_joints(obs, real_joints)
                
                follower_read_joints = None
                # Mirror to follower with a smooth ramp from its starting position
                if follower is not None:
                    if elapsed_total < ramp_duration:
                        alpha = _smoothstep(elapsed_total / ramp_duration)
                        follower_target = follower_start_joints + alpha * (leader_joints - follower_start_joints)
                    else:
                        follower_target = leader_joints
                    follower.send_action(joints_to_action(follower_target))

                    # Read back actual follower position
                    follower_obs = follower.get_observation()
                    follower_read_joints = action_to_joints(follower_obs, follower_target)

                # Determine what the sim should visualize
                if args.sim_tracks == "follower" and follower_read_joints is not None:
                    sim_target_joints = follower_read_joints
                else:
                    sim_target_joints = leader_joints

                # Update real_joints for the next iteration's previous state
                real_joints = leader_joints

                arm_rad, gripper_rad = real_frame_to_sim(sim_target_joints, offsets)

                # Use kinematic teleportation if explicitly requested, or if tracking 
                # the actual follower readback (so physics doesn't lag the real arm).
                is_kinematic = args.kinematic or (args.sim_tracks == "follower" and follower_read_joints is not None)

                if is_kinematic:
                    # Kinematic teleport
                    for name in ARM_JOINT_NAMES:
                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        data.qpos[model.jnt_qposadr[jid]] = arm_rad[name]
                        data.ctrl[actuator_id[name]] = arm_rad[name]
                        data.qvel[model.jnt_dofadr[jid]] = 0.0

                    g_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
                    data.qpos[model.jnt_qposadr[g_jid]] = gripper_rad
                    data.ctrl[actuator_id["gripper"]] = gripper_rad
                    data.qvel[model.jnt_dofadr[g_jid]] = 0.0
                else:
                    # Actuator driven: we are commanding a target pose
                    for name in ARM_JOINT_NAMES:
                        data.ctrl[actuator_id[name]] = arm_rad[name]
                    data.ctrl[actuator_id["gripper"]] = gripper_rad

                # We could step multiple times per frame (e.g. 10 substeps) if desired,
                # but a simple 1:1 control works decently for basic teleop.
                # Actually, running a few substeps helps physics stability:
                substeps = max(1, int((1.0 / args.fps) / model.opt.timestep))
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                    
                viewer.sync()

                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if follower is not None:
            follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
