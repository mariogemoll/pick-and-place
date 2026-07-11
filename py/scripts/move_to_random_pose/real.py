#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Move-to-random-pose demo (real): the hardware counterpart of ``sim.py``.

Same task, same toolbox (``episode_loop``/``EpisodeRecorder``/``recover_on``)
and the same pose sampling/easing (``pick_and_place.move_to_random_pose``) as
``sim.py`` — only the execution backend differs. Each episode samples a random
near-neutral arm pose and moves the real SO-101 follower there, recording the
real-frame commanded set point vs. the encoder read-back at each control tick.
No cube, no camera, no grasp, no checkpoint replanning: this reuses
``pick_and_place.follower``/``pick_and_place.executor``'s connect/ramp/clamp
plumbing, the same plumbing ``pick_and_place/real.py`` uses, but
none of its grasp-specific machinery.

The sim model is kept alongside the real arm purely as the kinematics source
(joint limits) and an optional viewer; it is not the safety-critical path —
the real arm's start pose for every move is always read fresh from the
encoders, never assumed from the sim.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.episode_loop import episode_loop
from pick_and_place.executor import clamp_and_warn, follower_clamp_limits
from pick_and_place.follower import (
    JOINT_NAMES,
    action_to_joints,
    joints_to_action,
    load_follower_joint_offsets,
    make_so101_follower,
    sim_frame_to_real,
)
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.move_to_random_pose import sample_reachable_pose, smoothstep
from pick_and_place.recorder import EpisodeRecorder
from pick_and_place.safety import recover_on
from pick_and_place.trajectory import (
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)


def move_to_real(
    follower,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: dict[str, int],
    viewer,
    target_joints: dict[str, float],
    target_gripper: float,
    duration: float,
    control_hz: float,
    offsets: np.ndarray,
    clamp_low: np.ndarray,
    clamp_high: np.ndarray,
    clip_warned: set[str],
    recorder: EpisodeRecorder | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothstep-ease the real arm and the sim onto ``target_joints``/
    ``target_gripper`` together.

    The real arm's start is read fresh from the encoders (never assumed from
    the sim), so this is safe to call regardless of whether the sim and the
    physical arm currently agree — the same approach ``executor.ramp_to_resting``
    uses. If ``recorder`` is given, logs the real-frame commanded set point vs.
    the encoder read-back each control tick. Returns ``(start_real, target_real)``
    for the caller to log as episode metadata.
    """
    target_real = clamp_and_warn(
        sim_frame_to_real(target_joints, target_gripper, offsets), clamp_low, clamp_high, clip_warned
    )
    start_real = action_to_joints(follower.get_observation(), target_real)
    delta_real = target_real - start_real

    start_sim_joints = {name: data.ctrl[actuator_id[name]] for name in target_joints}
    start_sim_gripper = data.ctrl[actuator_id["gripper"]]

    control_period = 1.0 / control_hz
    t0 = data.time
    last_control_t = -math.inf
    while True:
        wall_start = time.time()
        elapsed = data.time - t0
        alpha = smoothstep(elapsed / duration) if duration > 0 else 1.0

        for name in target_joints:
            data.ctrl[actuator_id[name]] = (
                start_sim_joints[name] + alpha * (target_joints[name] - start_sim_joints[name])
            )
        data.ctrl[actuator_id["gripper"]] = (
            start_sim_gripper + alpha * (target_gripper - start_sim_gripper)
        )
        mujoco.mj_step(model, data)

        if data.time - last_control_t >= control_period:
            last_control_t = data.time
            commanded = start_real + alpha * delta_real
            follower.send_action(joints_to_action(commanded))
            actual = action_to_joints(follower.get_observation(), commanded)
            if recorder is not None:
                recorder.log(commanded=commanded, measured=actual, t=data.time)

        if viewer is not None:
            viewer.sync()

        if elapsed >= duration:
            break
        remaining = model.opt.timestep - (time.time() - wall_start)
        if remaining > 0:
            time.sleep(remaining)

    return start_real, target_real


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="number of moves to run; 0 means loop until Ctrl-C/viewer closed (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the sampled poses")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "out" / "move_to_random_pose",
        help="directory for the recorded .npz episodes (default: py/out/move_to_random_pose)",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=2.0,
        help="seconds per move, smoothstep-eased (default: 2.0)",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
        help="rate set points are streamed to the follower and read back (default: 30.0)",
    )
    parser.add_argument(
        "--rest-every",
        type=int,
        default=10,
        help="moves between torque-off cooldown rests; 0 to disable (default: 10)",
    )
    parser.add_argument(
        "--rest-duration",
        type=float,
        default=30.0,
        help="cooldown rest duration in seconds, torque off at REST (default: 30.0)",
    )
    parser.add_argument("--follower-port", required=True, help="serial port of the SO-101 follower")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id (default: folly)")
    parser.add_argument(
        "--offsets-path",
        default=None,
        help="JSON of per-joint sim->real degree offsets (default: zero offsets)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="run headless (no 3D MuJoCo viewer)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    spec = build_scene(include_environment=False)
    model = spec.compile()
    data = mujoco.MjData(model)
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    kinematics = derive_kinematics(model)
    offsets = load_follower_joint_offsets(args.offsets_path)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    for name, value in NEUTRAL_ARM_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
        data.ctrl[actuator_id[name]] = value
    gripper_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[model.jnt_qposadr[gripper_jid]] = NEUTRAL_GRIPPER
    data.ctrl[actuator_id["gripper"]] = NEUTRAL_GRIPPER
    mujoco.mj_forward(model, data)

    print("Connecting to follower...")
    # Keep torque on a plain disconnect (crash / mid-loop exit) so the arm holds
    # rather than going limp; torque is only released deliberately at REST.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    def move(viewer, target_joints, target_gripper, recorder=None) -> tuple[np.ndarray, np.ndarray]:
        return move_to_real(
            follower, model, data, actuator_id, viewer,
            target_joints, target_gripper, args.move_duration, args.control_hz,
            offsets, clamp_low, clamp_high, clip_warned, recorder,
        )

    def cooldown(viewer) -> None:
        print(f"Cooldown: resting with torque off for {args.rest_duration:.0f}s...")
        move(viewer, REST_ARM_JOINTS, REST_GRIPPER)
        follower.bus.disable_torque()
        time.sleep(args.rest_duration)
        follower.bus.enable_torque()
        move(viewer, NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)

    ended_at_rest = False

    def park_from_interrupt() -> None:
        # The real viewer has already been torn down by the `with` below, so
        # park headless (`viewer=None`); `move_to_real` is already null-safe on
        # that. Make sure torque is on first — a Ctrl-C during the cooldown
        # sleep leaves it off, and parking commands would be ignored.
        nonlocal ended_at_rest
        print("\nCtrl-C: parking to NEUTRAL then REST...")
        try:
            follower.bus.enable_torque()
        except Exception as exc:  # noqa: BLE001 - best-effort re-enable before parking
            print(f"Warning: could not enable torque before parking: {exc}")
        move(None, NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)
        move(None, REST_ARM_JOINTS, REST_GRIPPER)
        ended_at_rest = True

    viewer_ctx = nullcontext(None) if args.no_viewer else mujoco.viewer.launch_passive(model, data)

    episode_index = 0
    try:
        with recover_on(KeyboardInterrupt, recover=park_from_interrupt):
            with viewer_ctx as viewer:
                print("Homing to neutral...")
                move(viewer, NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)

                should_continue = (lambda: True) if viewer is None else viewer.is_running
                for ep in episode_loop(
                    target=args.episodes,
                    rest_every=args.rest_every,
                    cooldown=lambda: cooldown(viewer),
                    should_continue=should_continue,
                ):
                    target_joints, target_gripper = sample_reachable_pose(rng)

                    recorder = EpisodeRecorder()
                    print(f"\n--- Move {ep.index}{f'/{args.episodes}' if args.episodes else ''} ---")
                    start_real, target_real = move(viewer, target_joints, target_gripper, recorder)

                    episode_index += 1
                    path = args.out_dir / f"episode_{episode_index:05d}.npz"
                    record = recorder.save(
                        path,
                        episode_index=np.array(episode_index),
                        seed=np.array(args.seed),
                        joint_names=np.array(JOINT_NAMES),
                        start_joints=start_real,
                        target_joints=target_real,
                        duration=np.array(args.move_duration),
                        control_hz=np.array(args.control_hz),
                    )
                    print(f"  {len(record['t'])} frames -> {path.name}")
                    ep.complete()

                if should_continue():
                    print("Loop done. Moving to REST...")
                    move(viewer, REST_ARM_JOINTS, REST_GRIPPER)
                    ended_at_rest = True
    finally:
        if ended_at_rest:
            print("At REST — releasing torque.")
            try:
                follower.bus.disable_torque()
            except Exception as exc:  # noqa: BLE001 - best-effort torque release
                print(f"Warning: could not release torque: {exc}")
        print("Disconnecting hardware...")
        follower.disconnect()

    print(f"Wrote {episode_index} move(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
