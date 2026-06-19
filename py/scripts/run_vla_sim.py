#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a SmolVLA policy in the sim, closed-loop.

Pass ``--checkpoint`` to evaluate a fine-tuned policy; the default is the base
``lerobot/smolvla_base``. The base is a plumbing spike, not a working manipulator
— it has never seen this robot, these cameras, or this instruction, so its
actions are not meaningful (the arm moves but does not solve the task). A policy
fine-tuned on the project's dataset is the real use case. Either way the loop is
the same: render the sim cameras, build the observation a LeRobot policy expects
(two images + proprio state + a language instruction), run ``select_action``, and
feed the result back into the sim as position targets.

The policy speaks the real (hardware) frame the dataset was recorded in — arm
joints in degrees, gripper as a 0-100 position — while MuJoCo speaks radians. The
two boundaries convert accordingly: sim ``qpos`` -> ``sim_frame_to_real`` for the
observation state, and the predicted action -> ``real_frame_to_sim`` before it is
written to ``data.ctrl``. Normalization stats live inside the policy's processor
and load from the checkpoint, so the dataset is left in raw physical units.

The sim is the plant: the cube is a free rigid body, the arm is driven through
its position-servo actuators, and physics integrates live. SmolVLA predicts an
action chunk (``n_action_steps``) and ``select_action`` serves one step per call,
only re-running the network when the chunk drains.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Some SmolVLM backbone ops are not implemented for Apple MPS; fall back to CPU
# for just those ops instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    GRIPPER_INDEX,
    JOINT_NAMES,
    load_follower_joint_offsets,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.episodes import sample_target
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    add_paper_target_marker,
    place_paper_target_marker,
)
from pick_and_place.trajectory import GRIPPER_OPEN, NEUTRAL_ARM_JOINTS
from pick_and_place.workspace_overlays import is_cube_drop_allowed
from pick_and_place.vla import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INSTRUCTION,
    OVERHEAD_FEATURE,
    WRIST_FEATURE,
    make_policy,
    select_device,
)

# Control/render rate. The sim steps at the model timestep; one policy query and
# one camera render happen per control tick.
CONTROL_HZ = 10.0


def _build_model(
    source_xy: tuple[float, float],
    source_yaw: float,
    target_xy: tuple[float, float],
    target_yaw: float,
    render_size: int,
):
    """Compile the standard (AprilTag, calibrated-camera) scene with the pick cube
    placed as a free rigid body at the requested pose. Mirrors the layout used by
    the episode tooling so the cameras and cube match what a policy would see.

    The black drop-zone square is rendered at ``target_xy``/``target_yaw`` so the
    frames match a real recording, where a physical paper square on the table marks
    where the cube must be placed; without it the policy sees no target.

    ``render_size`` enlarges the offscreen framebuffer so the camera renders fed to
    the policy fit (MuJoCo defaults to 640x480, too small for a 512 square)."""
    spec = build_scene(include_environment=True)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, render_size)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, render_size)
    apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    intrinsics = load_local_camera_intrinsics()
    for camera in spec.cameras:
        if camera.name in intrinsics and "fovy_deg" in intrinsics[camera.name]:
            camera.fovy = float(intrinsics[camera.name]["fovy_deg"])

    cube = spec.body("pick_cube")
    cube.pos = (source_xy[0], source_xy[1], CUBE_HALF_SIZE)
    half_yaw = source_yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()

    add_paper_target_marker(spec)

    model = spec.compile()
    place_paper_target_marker(
        model,
        target_xy,
        target_yaw,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(target_xy[0], target_xy[1]),
        alpha=1.0,
    )
    return model, mujoco.MjData(model)


def _joint_qpos_adr(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]


def _sim_state_to_real(qpos_rad: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sim joint positions (radians, ``JOINT_NAMES`` order) -> real-frame state
    vector (arm degrees + gripper 0-100), matching the dataset convention."""
    arm = {name: float(qpos_rad[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
    return sim_frame_to_real(arm, float(qpos_rad[GRIPPER_INDEX]), offsets).astype(np.float32)


def _real_action_to_ctrl(action_real: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Real-frame action vector from the policy -> sim ctrl (radians,
    ``JOINT_NAMES`` order, which the actuators follow)."""
    arm_rad, gripper_rad = real_frame_to_sim(action_real, offsets)
    return np.array([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad])


def _set_neutral(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Park the arm at the neutral pose with the gripper open, and hold it there
    by initialising the position-servo set points to the same values."""
    actuator_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES
    }
    targets = dict(NEUTRAL_ARM_JOINTS)
    targets["gripper"] = GRIPPER_OPEN
    for name, value in targets.items():
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[adr] = value
        data.ctrl[actuator_id[name]] = value
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF policy checkpoint")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--image-size", type=int, default=512, help="square render size fed to the VLA")
    parser.add_argument("--source", type=float, nargs=2, metavar=("X", "Y"), default=(0.22, 0.0))
    parser.add_argument("--source-yaw", type=float, default=0.0, help="cube yaw (radians)")
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the drop-zone center (x, y); omit to sample one randomly like the recording",
    )
    parser.add_argument("--target-yaw", type=float, default=0.0, help="drop-zone yaw (radians)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for random target sampling")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after this many control ticks (0 = run until the viewer is closed)",
    )
    parser.add_argument("--headless", action="store_true", help="no viewer; render only for the policy")
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help=(
            "directory to write <dir>/wrist.mp4 and <dir>/overhead.mp4 with the exact "
            "frames fed to the policy each tick"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "live-preview the observation frames in OpenCV windows (requires --headless: "
            "the MuJoCo viewer runs its own GUI loop under mjpython and conflicts with it)"
        ),
    )
    args = parser.parse_args()
    if args.show and not args.headless:
        parser.error("--show requires --headless")

    from lerobot.utils.control_utils import predict_action

    device = select_device(args.device)
    print(f"Loading {args.checkpoint} on {device} (first run downloads the weights)...")

    # Sample a random drop zone the same way the recording does, unless pinned.
    if args.target is not None:
        target_xy, target_yaw = tuple(args.target), args.target_yaw
    else:
        sampled = sample_target(np.random.default_rng(args.seed))
        target_xy, target_yaw = (sampled.x, sampled.y), sampled.yaw
    print(f"Drop zone at ({target_xy[0]:.4f}, {target_xy[1]:.4f}), yaw {target_yaw:.3f} rad")

    model, data = _build_model(
        tuple(args.source),
        args.source_yaw,
        target_xy,
        target_yaw,
        args.image_size,
    )
    _set_neutral(model, data)
    joint_adr = _joint_qpos_adr(model)
    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()
    # Zero sim->real offsets: the real frame is sim degrees with no calibration
    # bias, which is what an uncalibrated fine-tune is trained against.
    offsets = load_follower_joint_offsets(None)

    hw = (args.image_size, args.image_size)
    policy, preprocessor, postprocessor = make_policy(args.checkpoint, hw, hw, device)
    policy.reset()

    renderer = mujoco.Renderer(model, height=hw[0], width=hw[1])

    def render(camera: str) -> np.ndarray:
        renderer.update_scene(data, camera=camera)
        return renderer.render()  # (H, W, 3) uint8 RGB

    substeps = max(1, round((1.0 / CONTROL_HZ) / model.opt.timestep))
    period = 1.0 / CONTROL_HZ

    wrist_writer = overhead_writer = None
    if args.save_video is not None:
        import imageio.v2 as imageio

        args.save_video.mkdir(parents=True, exist_ok=True)
        wrist_writer = imageio.get_writer(args.save_video / "wrist.mp4", fps=CONTROL_HZ)
        overhead_writer = imageio.get_writer(args.save_video / "overhead.mp4", fps=CONTROL_HZ)
        print(f"Saving observation frames to {args.save_video}/{{wrist,overhead}}.mp4")

    if args.show:
        import cv2

        cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
        cv2.namedWindow("overhead", cv2.WINDOW_NORMAL)

    viewer_ctx = None
    if not args.headless:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None

    print(f"Instruction: {args.instruction!r}")
    if args.checkpoint == DEFAULT_CHECKPOINT:
        print("Running closed-loop. Actions are NOT task-calibrated (un-finetuned base).")
    else:
        print(f"Running closed-loop with fine-tuned checkpoint {args.checkpoint!r}.")
    tick = 0
    try:
        while viewer is None or viewer.is_running():
            tick_start = time.time()

            wrist_frame = render("wrist_camera")
            overhead_frame = render("overhead_camera")
            observation = {
                "observation.state": _sim_state_to_real(data.qpos[joint_adr], offsets),
                WRIST_FEATURE: wrist_frame,
                OVERHEAD_FEATURE: overhead_frame,
            }
            if wrist_writer is not None:
                wrist_writer.append_data(wrist_frame)
                overhead_writer.append_data(overhead_frame)
            if args.show:
                cv2.imshow("wrist", cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow("overhead", cv2.cvtColor(overhead_frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            action = predict_action(
                observation,
                policy,
                device,
                preprocessor,
                postprocessor,
                use_amp=False,
                task=args.instruction,
                robot_type="so101",
            )
            action_real = action.to("cpu").numpy().reshape(-1)[: len(JOINT_NAMES)]
            ctrl = _real_action_to_ctrl(action_real, offsets)
            data.ctrl[:] = np.clip(ctrl, ctrl_low, ctrl_high)

            mujoco.mj_step(model, data, nstep=substeps)
            if viewer is not None:
                viewer.sync()

            if tick % 10 == 0:
                np.set_printoptions(precision=3, suppress=True)
                print(f"tick {tick:4d}  ctrl(rad)={data.ctrl[:]}")

            tick += 1
            if args.steps and tick >= args.steps:
                break

            remaining = period - (time.time() - tick_start)
            if remaining > 0 and viewer is not None:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        renderer.close()
        if wrist_writer is not None:
            wrist_writer.close()
            overhead_writer.close()
        if args.show:
            cv2.destroyAllWindows()
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
    print(f"Ran {tick} control ticks.")


if __name__ == "__main__":
    main()
