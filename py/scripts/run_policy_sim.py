#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a LeRobot policy (ACT, SmolVLA, ...) in the sim, closed-loop.

Pass ``--checkpoint`` to evaluate a fine-tuned policy; the default is the base
``lerobot/smolvla_base``. The base is a plumbing spike, not a working manipulator
— it has never seen this robot, these cameras, or this instruction, so its
actions are not meaningful (the arm moves but does not solve the task). A policy
fine-tuned on the project's dataset is the real use case. Either way the loop is
the same: render the sim cameras, build the observation a LeRobot policy expects
(two images + proprio state + a language instruction), run ``select_action``, and
feed the result back into the sim as position targets. The concrete policy class
is resolved from the checkpoint, so the same script serves whatever was trained.

The policy speaks the real (hardware) frame the dataset was recorded in — arm
joints in degrees, gripper as a 0-100 position — while MuJoCo speaks radians. The
two boundaries convert accordingly: sim ``qpos`` -> ``sim_frame_to_real`` for the
observation state, and the predicted action -> ``real_frame_to_sim`` before it is
written to ``data.ctrl``. Normalization stats live inside the policy's processor
and load from the checkpoint, so the dataset is left in raw physical units.

The sim is the plant: the cube is a free rigid body, the arm is driven through
its position-servo actuators, and physics integrates live. Chunked policies
predict a horizon of actions and ``select_action`` serves one step per call,
only re-running the network after ``n_action_steps`` queued actions.
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
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.episodes import cube_quat_from_pose, sample_cube, sample_target
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    DomainRandomizer,
    domain_seed,
    generate_procedural_appearance,
    orient_cube,
    reload_renderer_textures,
)
from pick_and_place.miscalibration import MiscalibrationDraw, MiscalibrationModel
from pick_and_place.sim_recorder import resize_and_center_crop
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    add_paper_target_marker,
    place_paper_target_marker,
)
from pick_and_place.trajectory import GRIPPER_OPEN, NEUTRAL_ARM_JOINTS
from pick_and_place.workspace_overlays import is_cube_drop_allowed
from pick_and_place.policy import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INSTRUCTION,
    make_policy,
    resolve_checkpoint_cameras,
    select_device,
)

# One policy query and one camera render happen per control tick; the sim steps
# at the model timestep in between. The rate matches the real rig's control loop
# (and the dataset fps), so a chunked policy's action spacing plays back true.
from pick_and_place.executor import CONTROL_HZ, HARDWARE_SIMULATION_HZ


def _build_model(
    source: CubePose,
    target_xy: tuple[float, float],
    render_h: int,
    render_w: int,
    background_panorama: Path | np.ndarray | None = None,
    table_texture: Path | np.ndarray | None = None,
):
    """Compile the standard (AprilTag, calibrated-camera) scene with the pick cube
    placed as a free rigid body at the requested pose. Mirrors the layout used by
    the episode tooling so the cameras and cube match what a policy would see.

    The black drop-zone square is rendered at ``target_xy`` so the frames match a
    real recording, where a physical paper square on the table marks where the
    cube must be placed; without it the policy sees no target.

    ``render_h``/``render_w`` enlarge the offscreen framebuffer so the camera
    renders fed to the policy fit whatever resolution the checkpoint expects."""
    spec = build_scene(
        include_environment=True,
        background_panorama=background_panorama,
        table_texture=table_texture,
    )
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, render_w)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, render_h)
    apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    intrinsics = load_local_camera_intrinsics()
    for camera in spec.cameras:
        if camera.name in intrinsics and "fovy_deg" in intrinsics[camera.name]:
            camera.fovy = float(intrinsics[camera.name]["fovy_deg"])

    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    cube.quat = cube_quat_from_pose(source)
    cube.add_freejoint()

    add_paper_target_marker(spec)

    model = spec.compile()
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    place_paper_target_marker(
        model,
        target_xy,
        0.0,
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


def _sim_state_to_real(
    qpos_rad: np.ndarray, joint_offsets_rad: dict[str, float] | None = None
) -> np.ndarray:
    """Sim joint positions (radians, ``JOINT_NAMES`` order) -> real-frame state
    vector (arm degrees + gripper 0-100), matching the dataset convention.

    With a miscalibration draw, the observation is servo-style readback: the
    physically true joint angle less the injected joint-zero offset.
    """
    offsets = joint_offsets_rad or {}
    arm = {
        name: float(qpos_rad[i]) - offsets.get(name, 0.0)
        for i, name in enumerate(ARM_JOINT_NAMES)
    }
    return sim_frame_to_real(arm, float(qpos_rad[GRIPPER_INDEX])).astype(np.float32)


def _real_action_to_ctrl(action_real: np.ndarray) -> np.ndarray:
    """Real-frame action vector from the policy -> sim ctrl (radians,
    ``JOINT_NAMES`` order, which the actuators follow)."""
    arm_rad, gripper_rad = real_frame_to_sim(action_real)
    return np.array([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad])


def _set_neutral(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_offsets_rad: dict[str, float] | None = None,
) -> None:
    """Park the arm at the neutral pose with the gripper open, and hold it there
    by initialising the position-servo set points to the same values."""
    actuator_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES
    }
    offsets = joint_offsets_rad or {}
    targets = dict(NEUTRAL_ARM_JOINTS)
    targets["gripper"] = GRIPPER_OPEN
    for name, value in targets.items():
        true_value = value + offsets.get(name, 0.0)
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[adr] = true_value
        data.ctrl[actuator_id[name]] = true_value
    mujoco.mj_forward(model, data)


def _cube_freejoint_addrs(model: mujoco.MjModel) -> tuple[int, int]:
    """Return the (qpos, qvel) base addresses of the pick cube's free joint."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    return int(model.jnt_qposadr[model.body_jntadr[body_id]]), int(model.body_dofadr[body_id])


def _place_cube(data: mujoco.MjData, qadr: int, dofadr: int, pose: CubePose) -> None:
    """Drop a cube pose onto the table with zero velocity."""
    data.qpos[qadr : qadr + 3] = (pose.x, pose.y, pose.z)
    data.qpos[qadr + 3 : qadr + 7] = cube_quat_from_pose(pose)
    data.qvel[dofadr : dofadr + 6] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF policy checkpoint")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="render height fed to the policy (default: the checkpoint's training height, else 480)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="render width fed to the policy (default: the checkpoint's training width, else 640)",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1920,
        help="MuJoCo source render width before downsampling/cropping (default: 1920)",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=1080,
        help="MuJoCo source render height before downsampling/cropping (default: 1080)",
    )
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
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for random target sampling")
    parser.add_argument(
        "--miscalibration",
        action="store_true",
        help=(
            "inject a fresh measured joint-zero miscalibration draw for the initial "
            "scene and every Enter resample; observations use servo-style readback "
            "and actions are shifted into the true physical joint frame"
        ),
    )
    parser.add_argument(
        "--domain-randomization",
        type=Path,
        default=None,
        help=(
            "strict per-episode sim randomization preset; includes measured "
            "miscalibration, cameras, lighting, materials, cube orientation, and appearance"
        ),
    )
    parser.add_argument(
        "--background-panorama",
        type=Path,
        default=None,
        help="equirectangular room panorama to render as a skybox behind the scene",
    )
    parser.add_argument(
        "--table-texture",
        type=Path,
        default=None,
        help="top-down table texture (from reconstruct_table_texture.py) for the floor",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after this many control ticks (0 = run until the viewer is closed)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=100,
        help=(
            "queued actions to execute before re-querying a chunked policy "
            "(default: 100; matches common ACT checkpoints; temporal ensembling uses 1)"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help=(
            "enable ACT temporal ensembling with this coefficient, e.g. 0.01; "
            "requires --n-action-steps 1"
        ),
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

    override = (args.image_height, args.image_width)
    if any(override) and not all(override):
        parser.error("pass both --image-height and --image-width, or neither")
    image_hw, (overhead_key, wrist_key) = resolve_checkpoint_cameras(
        args.checkpoint, override_hw=(args.image_height, args.image_width) if all(override) else None
    )
    if args.render_width < image_hw[1] or args.render_height < image_hw[0]:
        parser.error("--render-width and --render-height must be at least the policy image size")

    from lerobot.utils.control_utils import predict_action

    device = select_device(args.device)
    print(f"Loading {args.checkpoint} on {device} (first run downloads the weights)...")
    print(
        f"Feeding {image_hw[1]}x{image_hw[0]} (WxH) frames as {overhead_key!r} (overhead) "
        f"and {wrist_key!r} (wrist)."
    )

    rng = np.random.default_rng(args.seed)
    preset = (
        DomainRandomizationPreset.load(args.domain_randomization)
        if args.domain_randomization is not None
        else None
    )
    domain_episode = 0
    active_sample = (
        preset.sample(domain_seed(args.seed, domain_episode)) if preset is not None else None
    )
    miscalibration_model = MiscalibrationModel() if args.miscalibration and preset is None else None
    draw: MiscalibrationDraw | None = (
        active_sample.miscalibration
        if active_sample is not None
        else (miscalibration_model.sample(rng) if miscalibration_model is not None else None)
    )

    # Sample a random drop zone the same way the recording does, unless pinned.
    if args.target is not None:
        target_xy = tuple(args.target)
    else:
        sampled = sample_target(rng)
        target_xy = (sampled.x, sampled.y)
    print(f"Drop zone at ({target_xy[0]:.4f}, {target_xy[1]:.4f})")

    source_pose = CubePose(
        x=float(args.source[0]),
        y=float(args.source[1]),
        z=CUBE_HALF_SIZE,
        yaw=args.source_yaw,
    )
    if active_sample is not None:
        source_pose = orient_cube(source_pose, active_sample.cube_orientation_index)
        appearance = generate_procedural_appearance(active_sample)
        background_panorama = appearance.background_rgb
        table_texture = appearance.table_rgb
    else:
        background_panorama = args.background_panorama
        table_texture = args.table_texture

    model, data = _build_model(
        source_pose,
        target_xy,
        args.render_height,
        args.render_width,
        background_panorama=background_panorama,
        table_texture=table_texture,
    )
    randomizer = DomainRandomizer(model) if active_sample is not None else None
    if randomizer is not None:
        randomizer.apply(active_sample)
        randomizer.tint_episode_markers()
        print(
            f"Domain sample episode {domain_episode}: seed={active_sample.seed}, "
            f"cube_orientation={active_sample.cube_orientation_index}"
        )
    episode_time_origin = data.time

    def offsets_rad_now() -> dict[str, float]:
        if draw is None:
            return {}
        return draw.offsets_rad(data.time - episode_time_origin)

    _set_neutral(model, data, offsets_rad_now())
    joint_adr = _joint_qpos_adr(model)
    cube_qadr, cube_dofadr = _cube_freejoint_addrs(model)
    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    hw = image_hw
    policy, preprocessor, postprocessor = make_policy(
        args.checkpoint,
        hw,
        (overhead_key, wrist_key),
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    policy.reset()
    if hasattr(policy.config, "chunk_size") and hasattr(policy.config, "n_action_steps"):
        print(
            f"Policy chunks: predicts {policy.config.chunk_size}, "
            f"executes {policy.config.n_action_steps} before re-query."
        )
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        print(f"Temporal ensembling coeff: {policy.config.temporal_ensemble_coeff}")

    renderer = mujoco.Renderer(
        model, height=args.render_height, width=args.render_width
    )

    def render(camera: str) -> np.ndarray:
        renderer.update_scene(data, camera=camera)
        image = resize_and_center_crop(renderer.render(), hw[0], hw[1])
        return randomizer.postprocess(image) if randomizer is not None else image

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

    def resample_scene() -> None:
        """Restart the episode: park the arm, drop a freshly sampled cube, draw a
        freshly sampled drop zone, and reset the policy's action chunk."""
        nonlocal active_sample, domain_episode, draw, episode_time_origin, target_xy
        domain_episode += 1
        if preset is not None:
            active_sample = preset.sample(domain_seed(args.seed, domain_episode))
            randomizer.apply(active_sample)
            reload_renderer_textures(renderer, randomizer.texture_ids)
            draw = active_sample.miscalibration
            print(
                f"Domain sample episode {domain_episode}: seed={active_sample.seed}, "
                f"cube_orientation={active_sample.cube_orientation_index}"
            )
        else:
            draw = (
                miscalibration_model.sample(rng)
                if miscalibration_model is not None
                else None
            )
        episode_time_origin = data.time
        _set_neutral(model, data, offsets_rad_now())
        cube = sample_cube(rng)
        if active_sample is not None:
            cube = orient_cube(cube, active_sample.cube_orientation_index)
        _place_cube(data, cube_qadr, cube_dofadr, cube)
        target = sample_target(rng)
        target_xy = (target.x, target.y)
        place_paper_target_marker(
            model,
            (target.x, target.y),
            0.0,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(target.x, target.y),
            alpha=1.0,
        )
        if randomizer is not None:
            randomizer.tint_episode_markers()
        mujoco.mj_forward(model, data)
        policy.reset()
        if draw is not None:
            offsets = ", ".join(
                f"{name}={value:+.2f}°" for name, value in sorted(draw.base_offsets_deg.items())
            )
            print(f"Injected joint-zero offsets: {offsets}")
        print(
            f"Resampled: cube ({cube.x:.4f}, {cube.y:.4f}) yaw {cube.yaw:.3f}, "
            f"drop zone ({target.x:.4f}, {target.y:.4f})"
        )

    # Press Enter (in the viewer or a --show window) to resample the scene. Every
    # letter key is already bound to a MuJoCo viewer visualization toggle, so a
    # non-letter key is needed to avoid colliding with one.
    pending_resample = {"flag": False}
    GLFW_KEY_ENTER = 257

    def key_callback(keycode: int) -> None:
        if keycode == GLFW_KEY_ENTER:
            pending_resample["flag"] = True

    viewer_ctx = None
    if not args.headless:
        viewer_ctx = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None

    print(f"Instruction: {args.instruction!r}")
    if args.checkpoint == DEFAULT_CHECKPOINT:
        print("Running closed-loop. Actions are NOT task-calibrated (un-finetuned base).")
    else:
        print(f"Running closed-loop with fine-tuned checkpoint {args.checkpoint!r}.")
    print("Press Enter to resample the cube and drop zone and restart the scene.")
    if draw is not None:
        offsets = ", ".join(
            f"{name}={value:+.2f}°" for name, value in sorted(draw.base_offsets_deg.items())
        )
        print(f"Injected joint-zero offsets: {offsets}")
    tick = 0
    try:
        while viewer is None or viewer.is_running():
            tick_start = time.time()

            if pending_resample["flag"]:
                pending_resample["flag"] = False
                resample_scene()

            wrist_frame = render("wrist_camera")
            overhead_frame = render("overhead_camera")
            observation = {
                "observation.state": _sim_state_to_real(
                    data.qpos[joint_adr], offsets_rad_now()
                ),
                overhead_key: overhead_frame,
                wrist_key: wrist_frame,
            }
            if wrist_writer is not None:
                wrist_writer.append_data(wrist_frame)
                overhead_writer.append_data(overhead_frame)
            if args.show:
                cv2.imshow("wrist", cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow("overhead", cv2.cvtColor(overhead_frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (13, 10):  # Enter / keypad Enter
                    pending_resample["flag"] = True
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
            ctrl = _real_action_to_ctrl(action_real)
            offsets = offsets_rad_now()
            offset_ctrl = np.array([offsets.get(name, 0.0) for name in JOINT_NAMES])
            data.ctrl[:] = np.clip(ctrl + offset_ctrl, ctrl_low, ctrl_high)

            mujoco.mj_step(model, data, nstep=substeps)
            if viewer is not None:
                viewer.sync()

            if tick % 10 == 0:
                np.set_printoptions(precision=3, suppress=True)
                cube_xyz = data.qpos[cube_qadr : cube_qadr + 3]
                dist = math.hypot(cube_xyz[0] - target_xy[0], cube_xyz[1] - target_xy[1])
                print(
                    f"tick {tick:4d}  ctrl(rad)={data.ctrl[:]}  "
                    f"cube=({cube_xyz[0]:+.3f}, {cube_xyz[1]:+.3f}, {cube_xyz[2]:+.3f})  "
                    f"to-target={dist * 100:.1f}cm"
                )

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
    cube_xyz = data.qpos[cube_qadr : cube_qadr + 3]
    dist = math.hypot(cube_xyz[0] - target_xy[0], cube_xyz[1] - target_xy[1])
    print(
        f"Ran {tick} control ticks. Final cube ({cube_xyz[0]:+.4f}, {cube_xyz[1]:+.4f}, "
        f"{cube_xyz[2]:+.4f}), {dist * 100:.1f}cm from the drop-zone center."
    )


if __name__ == "__main__":
    main()
