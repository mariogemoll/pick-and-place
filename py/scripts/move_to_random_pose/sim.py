#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Move-to-random-pose demo (sim): the second task built on the episode toolbox.

Deliberately minimal. Each episode samples a random near-neutral arm pose and
moves there from wherever the arm currently is, recording the full joint
trajectory of the move, then returns to neutral once the run ends. No cube, no
detection, no grasp: this task shares nothing with pick-and-place except the
episode loop, the recorder, and Ctrl-C recovery — proof that those three are
the toolbox's actual seams, not an artifact of the one task it was extracted
from. See ``docs/episode-toolkit-plan.md``.

The arm is driven through the model's position actuators under real physics,
same execution pattern as ``view_trajectory``. Sim-only; see ``real.py`` for
the hardware counterpart.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.episode_loop import episode_loop
from pick_and_place.follower import ARM_JOINT_NAMES
from pick_and_place.move_to_random_pose import (
    current_pose,
    joint_qpos_adr as compute_joint_qpos_adr,
    lerp_joints,
    sample_reachable_pose,
    smoothstep,
)
from pick_and_place.recorder import EpisodeRecorder
from pick_and_place.safety import recover_on
from pick_and_place.trajectory import NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER


def move_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: dict[str, int],
    viewer,
    start_joints: dict[str, float],
    start_gripper: float,
    target_joints: dict[str, float],
    target_gripper: float,
    duration: float,
    control_hz: float,
    recorder: EpisodeRecorder | None,
    joint_qpos_adr: list[int],
    realtime: bool,
) -> None:
    """Smoothstep-ease the arm from start to target over ``duration`` seconds.

    Steps physics throughout; if ``recorder`` is given, logs commanded vs.
    measured joints at ``control_hz``. Paces to wall-clock time when
    ``realtime`` (so a viewer can keep up).
    """
    control_period = 1.0 / control_hz
    t0 = data.time
    last_sample_t = -float("inf")
    while True:
        wall_start = time.time()
        elapsed = data.time - t0
        alpha = smoothstep(elapsed / duration) if duration > 0 else 1.0
        joints = lerp_joints(start_joints, target_joints, alpha)
        gripper = start_gripper + (target_gripper - start_gripper) * alpha
        for name, value in joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = gripper

        if recorder is not None and elapsed - last_sample_t >= control_period:
            last_sample_t = elapsed
            commanded = np.array([joints[n] for n in ARM_JOINT_NAMES] + [gripper])
            recorder.log(
                commanded=commanded, measured=data.qpos[joint_qpos_adr].copy(), time=data.time
            )

        if viewer is not None:
            viewer.sync()

        if elapsed >= duration:
            break
        mujoco.mj_step(model, data)
        if realtime:
            remaining = model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


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
        default=50.0,
        help="trajectory sampling rate for the recorder (default: 50.0)",
    )
    parser.add_argument(
        "--rest-every",
        type=int,
        default=0,
        help="moves between neutral-pose cooldown pauses; 0 to disable (default: 0)",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=3.0,
        help="pause duration at neutral during a cooldown (default: 3.0)",
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
    joint_qpos_adr = compute_joint_qpos_adr(model)

    for name, value in NEUTRAL_ARM_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
        data.ctrl[actuator_id[name]] = value
    gripper_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[model.jnt_qposadr[gripper_jid]] = NEUTRAL_GRIPPER
    data.ctrl[actuator_id["gripper"]] = NEUTRAL_GRIPPER
    mujoco.mj_forward(model, data)

    def park_to_neutral(viewer) -> None:
        print("Parking to neutral...")
        joints, gripper = current_pose(data, joint_qpos_adr)
        move_to(
            model, data, actuator_id, viewer,
            joints, gripper, NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER,
            args.move_duration, args.control_hz, None, joint_qpos_adr,
            realtime=viewer is not None,
        )

    def cooldown(viewer) -> None:
        print(f"Cooldown: pausing at neutral for {args.cooldown_seconds:.0f}s...")
        park_to_neutral(viewer)
        deadline = time.time() + args.cooldown_seconds
        while time.time() < deadline:
            wall_start = time.time()
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            remaining = model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)

    viewer_ctx = nullcontext(None) if args.no_viewer else mujoco.viewer.launch_passive(model, data)

    episode_index = 0
    # Run-level session: whatever ends the run (episode budget met, viewer
    # closed, or Ctrl-C) flows the arm back to neutral. A KeyboardInterrupt
    # inside `with viewer_ctx` closes the real viewer before this recovers, so
    # `park_to_neutral` is called with `viewer=None` (headless, no rendering).
    with recover_on(KeyboardInterrupt, recover=lambda: park_to_neutral(None)):
        with viewer_ctx as viewer:
            should_continue = (lambda: True) if viewer is None else viewer.is_running
            for ep in episode_loop(
                target=args.episodes,
                rest_every=args.rest_every,
                cooldown=lambda: cooldown(viewer),
                should_continue=should_continue,
            ):
                start_joints, start_gripper = current_pose(data, joint_qpos_adr)
                target_joints, target_gripper = sample_reachable_pose(rng)

                recorder = EpisodeRecorder()
                print(f"\n--- Move {ep.index}{f'/{args.episodes}' if args.episodes else ''} ---")
                move_to(
                    model, data, actuator_id, viewer,
                    start_joints, start_gripper, target_joints, target_gripper,
                    args.move_duration, args.control_hz, recorder, joint_qpos_adr,
                    realtime=viewer is not None,
                )

                episode_index += 1
                path = args.out_dir / f"episode_{episode_index:05d}.npz"
                record = recorder.save(
                    path,
                    episode_index=np.array(episode_index),
                    seed=np.array(args.seed),
                    start_joints=np.array([start_joints[n] for n in ARM_JOINT_NAMES] + [start_gripper]),
                    target_joints=np.array(
                        [target_joints[n] for n in ARM_JOINT_NAMES] + [target_gripper]
                    ),
                    duration=np.array(args.move_duration),
                    control_hz=np.array(args.control_hz),
                )
                print(f"  {len(record['time'])} frames -> {path.name}")
                ep.complete()

            if should_continue():
                print("Loop done. Parking to neutral...")
                park_to_neutral(viewer)

    print(f"Wrote {episode_index} move(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
