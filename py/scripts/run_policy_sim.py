#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a policy in the sim, closed-loop.

Two leaves, because what a checkpoint is and how it is queried is not something
the two controllers share:

``lerobot`` loads a LeRobot checkpoint (ACT, SmolVLA, pi0.5, ...); pass
``--checkpoint`` to evaluate a fine-tune, else the base ``lerobot/smolvla_base``.
The base is a plumbing spike, not a working manipulator — it has never seen this
robot, these cameras, or this instruction, so its actions are not meaningful (the
arm moves but does not solve the task). A policy fine-tuned on the project's
dataset is the real use case.

``scripted`` runs the expert, which is what watching a working episode looks
like: it localizes the cube from the overhead frame, plans an eight-phase
trajectory, servos the descent off the wrist camera and replans at every phase
boundary. It takes no checkpoint, and reads the same two images and reported
joints a learned policy does, which is what makes the two comparable.
``--scripted-perception detector`` swaps the simulator's own pose for the real
optical pipeline; ``--miscalibration`` gives it a rig worth correcting for.

Everything about the world and the run — scene, cube pose, render size,
randomization, seed — is declared once and shared by both leaves, so a number
produced under one is comparable to a number produced under the other.

The loop is: render the sim cameras, build the observation
(two images + proprio state), ask the controller for the next hardware-frame
action, and feed the result back into the sim as position targets.

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
import json
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

from pick_and_place.scripted.episode_sampling import sample_cube, sample_target
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.spec.workspace import CUBE_HALF_SIZE, DROP_ZONE_HALF_SIZE
from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    domain_seed,
    generate_procedural_appearance,
    orient_cube,
)
from pick_and_place.core.miscalibration import MiscalibrationDraw, MiscalibrationModel
from pick_and_place.variants.appearance import parse_appearance
from pick_and_place.spec.robot import GRIPPER_OPEN, NEUTRAL_ARM_JOINTS
from pick_and_place.core.workspace_bounds import sample_target_plate_yaw
from pick_and_place.policies.policy import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE_HW,
    resolve_checkpoint_cameras,
    select_device,
)
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policies.policy_controllers import LeRobotPolicyController
from pick_and_place.runtime.policy_sim import PolicySimEnv, live_scenario

# One policy query and one camera render happen per control tick; the sim steps
# at the model timestep in between. The rate matches the real rig's control loop
# (and the dataset fps), so a chunked policy's action spacing plays back true.
from pick_and_place.cli.policy import (
    add_lerobot_arguments,
    add_policy_image_arguments,
)
from pick_and_place.cli.scene import (
    add_cube_pose_arguments,
    add_preflight_debug_arguments,
    add_render_size_arguments,
    add_scene_appearance_arguments,
    add_scene_texture_arguments,
    preflight_debug_from_args,
)
from pick_and_place.rollout.scripted_sim import (
    SCRIPTED_PERCEPTION_MODES,
    sim_scripted_controller,
)
from pick_and_place.spec.robot import CONTROL_HZ


# An interactive run with neither --steps nor --resample-every has no episode
# length at all -- it flies until the viewer is closed. A scenario still has to
# name a budget, so this stands in for one: large enough that the env's
# truncation never arrives before the person watching does.
UNBOUNDED_EPISODE_STEPS = 1_000_000


def _resolve_recording_hw(
    parser: argparse.ArgumentParser, args: argparse.Namespace, image_hw: tuple[int, int]
) -> tuple[int, int]:
    """Validate the recording resolution rendered frames are reduced through.

    A learned policy has to be fed frames that went through the same downsampling
    its training video did, so the ``lerobot`` leaf requires the resolution and
    cannot guess it. The expert never saw a video and the leaf does not declare
    the flag: it reads the frames it is given, and passing them through a resize
    that models nothing would only cost detail its localizers use.
    """
    recording_hw = getattr(args, "recording_hw", None)
    if recording_hw is None:
        return image_hw
    height, width = recording_hw
    if height < 1 or width < 1:
        parser.error("--recording-hw must be positive")
    return (height, width)


def main() -> None:
    # Everything about the world and the run, shared by every leaf so that two
    # controllers are driven through literally the same declaration. What a
    # checkpoint is and how it is queried belongs to the leaf that has one.
    common = argparse.ArgumentParser(add_help=False)
    parser = common
    add_policy_image_arguments(parser)
    add_render_size_arguments(parser)
    add_cube_pose_arguments(parser, source_default=(0.22, 0.0))
    parser.add_argument(
        "--no-robot-dynamics",
        action="store_true",
        help="use raw upstream MuJoCo actuators instead of fitted actuator time constants",
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
    add_scene_texture_arguments(parser)
    add_scene_appearance_arguments(parser)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after this many control ticks (0 = run until the viewer is closed)",
    )
    parser.add_argument("--headless", action="store_true", help="no viewer; render only for the policy")
    parser.add_argument(
        "--resample-every",
        type=int,
        default=None,
        help=(
            "resample the cube and drop zone every N control ticks; the headless "
            "equivalent of pressing Enter, for sweeping many scenes in one run"
        ),
    )
    parser.add_argument(
        "--trajectory-json",
        type=Path,
        default=None,
        help=(
            "write a per-tick record of joints, cube pose and drop-zone pose to this "
            "path, for offline analysis of where the arm aims"
        ),
    )
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
    parser = argparse.ArgumentParser(description=__doc__)
    leaves = parser.add_subparsers(dest="controller", required=True, metavar="CONTROLLER")

    lerobot = leaves.add_parser(
        "lerobot",
        parents=[common],
        help="a LeRobot checkpoint (ACT, SmolVLA, pi0.5, ...)",
        description="Run a LeRobot checkpoint in the sim, closed-loop.",
    )
    # Default to the checkpoint's own n_action_steps. The 100 that
    # add_lerobot_arguments otherwise supplies matches ACT's chunk and is larger
    # than pi0.5's 50, so it turned every pi0.5 rollout into an argument error
    # unless the flag was passed by hand.
    add_lerobot_arguments(lerobot, n_action_steps_default=None)
    lerobot.add_argument(
        "--recording-hw",
        type=int,
        nargs=2,
        required=True,
        metavar=("HEIGHT", "WIDTH"),
        help="resolution the training videos were recorded at, which rendered frames "
        "are downsampled through on the way to the policy's input size",
    )

    scripted = leaves.add_parser(
        "scripted",
        parents=[common],
        help="the expert: localize, plan, servo the descent, replan at each phase",
        description="Run the expert in the sim, closed-loop. It takes no checkpoint.",
    )
    scripted.add_argument(
        "--scripted-perception",
        choices=SCRIPTED_PERCEPTION_MODES,
        default="geometric",
        help=(
            "overhead perception: geometric reads the simulator's own pose behind a "
            "visibility gate, detector runs the real optical pipeline "
            "(default: geometric)"
        ),
    )
    add_preflight_debug_arguments(scripted)

    args = parser.parse_args()
    if args.show and not args.headless:
        parser.error("--show requires --headless")

    try:
        appearance_name, scene_appearance = (
            parse_appearance(args.scene_appearance)
            if args.scene_appearance is not None
            else (None, None)
        )
    except ValueError as exc:
        parser.error(str(exc))

    override = (args.image_height, args.image_width)
    if any(override) and not all(override):
        parser.error("pass both --image-height and --image-width, or neither")
    override_hw = (args.image_height, args.image_width) if all(override) else None

    controller = None
    image_hw = (
        override_hw or DEFAULT_IMAGE_HW
        if args.controller == "scripted"
        else resolve_checkpoint_cameras(args.checkpoint, override_hw=override_hw)[0]
    )
    if args.render_width < image_hw[1] or args.render_height < image_hw[0]:
        parser.error("--render-width and --render-height must be at least the policy image size")

    recording_hw = _resolve_recording_hw(parser, args, image_hw)
    if args.render_width < recording_hw[1] or args.render_height < recording_hw[0]:
        parser.error("--render-width and --render-height must be at least the recording resolution")

    print(
        f"Feeding {image_hw[1]}x{image_hw[0]} (WxH) overhead and wrist frames, "
        f"downsampled through the {recording_hw[1]}x{recording_hw[0]} recording resolution."
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
    target_yaw = sample_target_plate_yaw(
        rng,
        target_xy[0],
        target_xy[1],
        half_size=DROP_ZONE_HALF_SIZE,
    )
    print(
        f"Drop zone at ({target_xy[0]:.4f}, {target_xy[1]:.4f}), "
        f"yaw {target_yaw:.3f}"
    )

    source_pose = CubePose(
        x=float(args.source[0]),
        y=float(args.source[1]),
        z=CUBE_HALF_SIZE,
        yaw=math.radians(args.source_yaw),
    )
    if active_sample is not None:
        source_pose = orient_cube(source_pose, active_sample.cube_orientation_index)
        appearance = generate_procedural_appearance(active_sample.appearance())
        background_panorama = appearance.background_rgb
        table_texture = appearance.table_rgb
    else:
        background_panorama = args.background_panorama
        table_texture = args.table_texture

    env = PolicySimEnv(
        image_hw=image_hw,
        render_hw=(args.render_height, args.render_width),
        recording_hw=recording_hw,
        scene_appearance=scene_appearance,
        # An interactive run is an inspection tool rather than a scored one: it
        # keeps flying after a placement lands, and --steps and --resample-every
        # are what end a scene. The oracle still reports, it just does not stop.
        terminate_on_success=False,
        background_panorama=background_panorama,
        table_texture=table_texture,
        robot_dynamics=not args.no_robot_dynamics,
    )
    if scene_appearance is not None:
        print(f"Scene appearance: {appearance_name}.")
    if active_sample is not None:
        print(
            f"Domain sample episode {domain_episode}: seed={active_sample.seed}, "
            f"cube_orientation={active_sample.cube_orientation_index}"
        )

    # The pose the arm is parked at, in the frame a scenario declares it in.
    initial_robot_state_real = sim_frame_to_real(NEUTRAL_ARM_JOINTS, GRIPPER_OPEN)
    # One episode is however long a scene is left running before the next
    # resample. The env truncates at this budget, which the loop below treats as
    # a scene ending rather than a stop, so it only has to not fire early.
    episode_max_steps = args.resample_every or args.steps or UNBOUNDED_EPISODE_STEPS

    def reset_episode(source: CubePose, target: tuple[float, float], yaw: float) -> tuple:
        """Freeze the sampled scene into a scenario and reset the env onto it."""
        scenario = live_scenario(
            source=source,
            target_xy=target,
            target_plate_yaw_rad=yaw,
            initial_robot_state_real=initial_robot_state_real,
            control_hz=CONTROL_HZ,
            max_steps=episode_max_steps,
            seed=args.seed,
            miscalibration=draw,
            domain_sample=active_sample,
            domain_preset_name=preset.name if preset is not None else None,
        )
        return env.reset(options={"scenario": scenario})


    hw = image_hw
    if args.controller == "scripted":
        controller, _ = sim_scripted_controller(
            image_hw=hw,
            render_hw=(args.render_height, args.render_width),
            control_hz=CONTROL_HZ,
            # The live scene, so the geometric localizer reads the cube where
            # physics actually put it. The controller's own nominal cameras are
            # compiled inside the factory and are deliberately not these.
            scene_model=env.model,
            scene_data=env.data,
            perception=args.scripted_perception,
            # The env owns the draw once it has been reset onto a scenario, so
            # the belief error is read from there rather than from the runner's
            # copy -- one of them advances the wander, and it is not this one.
            cube_belief_error=lambda: env.cube_belief_error,
            debug=preflight_debug_from_args(args),
        )
    if controller is None:
        device = select_device(args.device)
        print(f"Loading {args.checkpoint} on {device} (first run downloads the weights)...")
        controller = LeRobotPolicyController.from_checkpoint(
            args.checkpoint,
            device=device,
            image_hw=hw,
            instruction=args.instruction,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            base_checkpoint=args.base_checkpoint,
        )
        config = controller.policy.config
        if hasattr(config, "chunk_size") and hasattr(config, "n_action_steps"):
            print(
                f"Policy chunks: predicts {config.chunk_size}, "
                f"executes {config.n_action_steps} before re-query."
            )
        if getattr(config, "temporal_ensemble_coeff", None) is not None:
            print(f"Temporal ensembling coeff: {config.temporal_ensemble_coeff}")
    control_hz = CONTROL_HZ
    print(f"Policy control rate: {control_hz:g} Hz.")
    controller.reset()

    period = 1.0 / control_hz

    wrist_writer = overhead_writer = None
    if args.save_video is not None:
        import imageio.v2 as imageio

        args.save_video.mkdir(parents=True, exist_ok=True)
        wrist_writer = imageio.get_writer(args.save_video / "wrist.mp4", fps=control_hz)
        overhead_writer = imageio.get_writer(args.save_video / "overhead.mp4", fps=control_hz)
        print(f"Saving observation frames to {args.save_video}/{{wrist,overhead}}.mp4")

    if args.show:
        import cv2

        cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
        cv2.namedWindow("overhead", cv2.WINDOW_NORMAL)

    def resample_scene() -> tuple:
        """Restart the episode: draw a fresh scene, and reset the env onto it.

        The draws happen here because sampling interactively is what this runner
        is for; everything that applies them to the simulator belongs to the env,
        which is reached through the scenario the draws are frozen into."""
        nonlocal active_sample, domain_episode, draw, target_xy
        domain_episode += 1
        if preset is not None:
            active_sample = preset.sample(domain_seed(args.seed, domain_episode))
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
        cube = sample_cube(rng)
        if active_sample is not None:
            cube = orient_cube(cube, active_sample.cube_orientation_index)
        target = sample_target(rng)
        target_xy = (target.x, target.y)
        target_yaw = sample_target_plate_yaw(
            rng,
            target.x,
            target.y,
            half_size=DROP_ZONE_HALF_SIZE,
        )
        reset = reset_episode(cube, target_xy, target_yaw)
        controller.reset()
        if draw is not None:
            offsets = ", ".join(
                f"{name}={value:+.2f}°" for name, value in sorted(draw.base_offsets_deg.items())
            )
            print(f"Injected joint-zero offsets: {offsets}")
        print(
            f"Resampled: cube ({cube.x:.4f}, {cube.y:.4f}) yaw {cube.yaw:.3f}, "
            f"drop zone ({target.x:.4f}, {target.y:.4f}) yaw {target_yaw:.3f}"
        )
        return reset

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
        viewer_ctx = mujoco.viewer.launch_passive(
            env.model, env.data, key_callback=key_callback
        )
    viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None

    if args.controller == "lerobot":
        print(f"Instruction: {args.instruction!r}")
    if args.controller == "scripted":
        print(
            "Running closed-loop with the expert, overhead perception "
            f"{args.scripted_perception!r}."
        )
    elif args.checkpoint == DEFAULT_CHECKPOINT:
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
    # One entry per control tick; `segment` increments on every resample so an
    # analysis can treat each sampled scene as an independent rollout.
    trajectory: list[dict] = []
    segment = 0
    # The first scene was drawn before the controller existed, so the env is
    # reset onto it here rather than in `resample_scene`.
    observation, info = reset_episode(source_pose, target_xy, target_yaw)
    try:
        while viewer is None or viewer.is_running():
            tick_start = time.time()

            if args.resample_every and tick and tick % args.resample_every == 0:
                pending_resample["flag"] = True
            if pending_resample["flag"]:
                pending_resample["flag"] = False
                observation, info = resample_scene()
                segment += 1

            wrist_frame = observation[WRIST_FEATURE]
            overhead_frame = observation[OVERHEAD_FEATURE]
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
            action_real = controller.act(observation)
            if args.trajectory_json is not None:
                # Recorded before the step, so joints and cube pose are exactly
                # the state the policy conditioned on for this tick.
                cube_now = info["task_state"]["cube_position_m"]
                trajectory.append(
                    {
                        "tick": tick,
                        "segment": segment,
                        "state_real": [float(v) for v in observation[STATE_FEATURE]],
                        "action_real": [float(v) for v in action_real],
                        "cube_xyz": [float(v) for v in cube_now],
                        "target_xy": [float(target_xy[0]), float(target_xy[1])],
                    }
                )
            observation, _, terminated, truncated, info = env.step(action_real)
            if viewer is not None:
                viewer.sync()
            # This does not end the run: the scene keeps flying until --steps or
            # the next resample, which is what makes this a thing to watch
            # rather than a scored episode. It is worth saying out loud, though.
            # Truncation is not, being this runner's own budget arriving on
            # schedule -- it lands on the resample that was coming anyway.
            if terminated:
                reason = (
                    "collision" if info["task_state"]["unexpected_collision"]
                    else "cube out of bounds" if info["task_state"]["out_of_bounds"]
                    else "placement confirmed"
                )
                print(f"tick {tick:4d}  a scored episode would have ended here: {reason}")

            if tick % 10 == 0:
                np.set_printoptions(precision=3, suppress=True)
                cube_xyz = info["task_state"]["cube_position_m"]
                dist = math.hypot(cube_xyz[0] - target_xy[0], cube_xyz[1] - target_xy[1])
                print(
                    f"tick {tick:4d}  ctrl(rad)={env.data.ctrl[:]}  "
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
        env.close()
        close_controller = getattr(controller, "close", None)
        if close_controller is not None:
            close_controller()
        if wrist_writer is not None:
            wrist_writer.close()
            overhead_writer.close()
        if args.show:
            cv2.destroyAllWindows()
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
    if args.trajectory_json is not None:
        args.trajectory_json.parent.mkdir(parents=True, exist_ok=True)
        # A leaf that takes no checkpoint declares neither flag, so both are
        # absent from the namespace rather than unset.
        checkpoint = getattr(args, "checkpoint", None)
        with args.trajectory_json.open("w") as file:
            json.dump(
                {
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "seed": args.seed,
                    "n_action_steps": getattr(args, "n_action_steps", None),
                    "scene_appearance": args.scene_appearance,
                    "resample_every": args.resample_every,
                    "segments": segment + 1,
                    "ticks": trajectory,
                },
                file,
            )
        print(f"Wrote {len(trajectory)} ticks over {segment + 1} segments to {args.trajectory_json}")

    cube_xyz = info["task_state"]["cube_position_m"]
    dist = math.hypot(cube_xyz[0] - target_xy[0], cube_xyz[1] - target_xy[1])
    print(
        f"Ran {tick} control ticks. Final cube ({cube_xyz[0]:+.4f}, {cube_xyz[1]:+.4f}, "
        f"{cube_xyz[2]:+.4f}), {dist * 100:.1f}cm from the drop-zone center."
    )


if __name__ == "__main__":
    main()
