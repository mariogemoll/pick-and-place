#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Roll out an image-conditioned flow policy in the pick-and-place simulator.

The operating point is the state policy's: predict 16 actions, execute 8, and
integrate the flow with 10 Euler steps, which a paired check found matched
100-step integration at 7.65 times the speed.

A policy must be rolled out in the appearance it was trained on, and at the
image size it was trained on -- both are part of the model contract, and both
have silently produced zero-success evaluations in this project before.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.policies.flow_image_policy import FlowImagePolicyController
from pick_and_place.runtime.policy_sim import PolicySimEnv
from pick_and_place.runtime.training_scenes import training_scenario
from pick_and_place.sim.scene_appearance import parse_appearance
from pick_and_place.spec.controller import OVERHEAD_FEATURE, WRIST_FEATURE

ENTER_KEYS = frozenset({257, 335})


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Encode a stack of RGB frames as h264."""
    import imageio_ffmpeg

    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def observation_frame(observation: dict[str, np.ndarray]) -> np.ndarray:
    """The two policy views side by side, overhead left and wrist right."""
    return np.concatenate(
        (
            np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8),
            np.asarray(observation[WRIST_FEATURE], dtype=np.uint8),
        ),
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="matching image export")
    parser.add_argument(
        "--scenarios",
        type=int,
        default=50,
        help="how many scenes of the seed stream to run (0 = keep going until the viewer "
        "is closed or Ctrl-C)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=6_000_000,
        help="scene stream seed; scene i is drawn from seed-base + i",
    )
    parser.add_argument("--act-steps", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene-appearance", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="watch the rollouts in the MuJoCo viewer, throttled to the control rate "
        "(run under mjpython)",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="directory to write one mp4 per scenario of the frames the policy sees",
    )
    args = parser.parse_args()

    policy = FlowImagePolicyController.from_export(
        args.checkpoint,
        args.export,
        act_steps=args.act_steps,
        integration_steps=args.integration_steps,
        device=torch.device(args.device),
        seed=args.policy_seed,
    )
    appearance = None
    if args.scene_appearance:
        _, appearance = parse_appearance(args.scene_appearance)

    env = PolicySimEnv(
        image_hw=policy.image_hw, render_hw=(1080, 1920), scene_appearance=appearance
    )

    if args.save_video is not None:
        args.save_video.mkdir(parents=True, exist_ok=True)

    skip_requested = threading.Event()
    viewer_context: Any = nullcontext(None)
    if args.viewer:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=lambda keycode: (
                skip_requested.set() if keycode in ENTER_KEYS else None
            ),
        )
        print("Enter in the viewer skips to the next scene")

    endless = args.scenarios == 0
    scene_indices = itertools.count() if endless else range(args.scenarios)
    total = "inf" if endless else str(args.scenarios)

    records = []
    successes = 0
    try:
        with viewer_context as viewer:
            for index in scene_indices:
                if viewer is not None and not viewer.is_running():
                    print(f"viewer closed after {index} scenarios")
                    break
                scenario = training_scenario(index, seed_base=args.seed_base)
                observation, info = env.reset(options={"scenario": scenario})
                policy.reset()
                skip_requested.clear()
                frames: list[np.ndarray] = []
                for _ in range(scenario.max_steps):
                    if skip_requested.is_set():
                        break
                    started = time.perf_counter()
                    if args.save_video is not None:
                        frames.append(observation_frame(observation))
                    action = policy.act(observation)
                    observation, _, terminated, truncated, info = env.step(action)
                    if viewer is not None:
                        if not viewer.is_running():
                            break
                        viewer.sync()
                        time.sleep(
                            max(0.0, 1 / scenario.control_hz - (time.perf_counter() - started))
                        )
                    if terminated or truncated:
                        break
                if skip_requested.is_set():
                    # An abandoned rollout says nothing about the policy, so it is
                    # left out of the tally entirely.
                    print(f"{scenario.scenario_id}: skipped", flush=True)
                    continue
                if frames:
                    write_video(
                        args.save_video / f"{scenario.scenario_id}.mp4",
                        frames,
                        scenario.control_hz,
                    )
                milestones = info["milestones"]
                success = bool(info["success"])
                successes += success
                records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "success": success,
                        "control_steps": int(info["control_steps"]),
                        "milestones": milestones,
                        "clipped_fraction": policy.clipped_fraction,
                    }
                )
                print(
                    f"{index + 1}/{total} {scenario.scenario_id}: "
                    f"{'SUCCESS' if success else 'failure'}  running {successes}/{len(records)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)

    env.close()
    if not records:
        print("no scenarios completed")
        return
    completed = len(records)
    summary = {
        "checkpoint": str(args.checkpoint),
        "export": str(args.export),
        "scenarios": completed,
        "seed_base": args.seed_base,
        "act_steps": args.act_steps,
        "integration_steps": args.integration_steps,
        "scene_appearance": args.scene_appearance or "as-compiled",
        "successes": successes,
        "success_rate": successes / completed,
        "lifted": sum(r["milestones"]["cube_lifted"] for r in records),
        "contact_attempted": sum(r["milestones"]["pickup_contact_attempted"] for r in records),
        "episodes": records,
    }
    print(f"\n{successes}/{completed} = {successes / completed:.1%} success")
    print(f"lifted {summary['lifted']}, contact attempted {summary['contact_attempted']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
