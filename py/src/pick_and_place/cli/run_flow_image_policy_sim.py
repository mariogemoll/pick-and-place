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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.analysis.flow_trace_recording import FlowTraceRecording, encode
from pick_and_place.cli.run_flow_image_policy_sim_parser import build_parser
from pick_and_place.policies.flow_image_policy import (
    FlowImagePolicyController,
    summarize_smoothness,
)
from pick_and_place.runtime.policy_sim import PolicySimEnv
from pick_and_place.runtime.training_scenes import training_scenario
from pick_and_place.variants.appearance import parse_appearance
from pick_and_place.spec.controller import OVERHEAD_FEATURE, WRIST_FEATURE
from pick_and_place.spec.robot import GRIPPER_INDEX

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


@dataclass
class TraceCollector:
    """Accumulate one rollout's replay state and integration paths."""

    qpos: list[np.ndarray] = field(default_factory=list)
    ticks: list[int] = field(default_factory=list)
    paths: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)

    def add_horizon(self, tick: int, policy: FlowImagePolicyController) -> None:
        """Keep the horizon this tick generated, if it generated one at all."""
        if policy.latest_path is None or policy.latest_prediction is None:
            return
        self.ticks.append(tick)
        self.paths.append(policy.latest_path)
        self.commands.append(policy.latest_prediction)

    def build(self, *, fps: float, act_steps: int, target_xy: tuple[float, float]):
        return FlowTraceRecording(
            fps=fps,
            act_steps=act_steps,
            target_xy=target_xy,
            qpos=np.stack(self.qpos),
            chunk_ticks=np.array(self.ticks, dtype=np.uint32),
            path=np.stack(self.paths),
            commands=np.stack(self.commands),
        )


def observation_frame(observation: dict[str, np.ndarray]) -> np.ndarray:
    """The two policy views side by side, overhead left and wrist right."""
    return np.concatenate(
        (
            np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8),
            np.asarray(observation[WRIST_FEATURE], dtype=np.uint8),
        ),
        axis=1,
    )


def run(args: argparse.Namespace) -> None:
    """Run the image-flow policy over the seed stream of scenes."""

    policy = FlowImagePolicyController.from_export(
        args.checkpoint,
        args.export,
        act_steps=args.act_steps,
        integration_steps=args.integration_steps,
        device=torch.device(args.device),
        seed=args.policy_seed,
        noise_correlation=args.noise_correlation,
    )
    appearance = None
    if args.scene_appearance:
        _, appearance = parse_appearance(args.scene_appearance)

    env = PolicySimEnv(
        image_hw=policy.image_hw,
        render_hw=(1080, 1920),
        scene_appearance=appearance,
    )

    if args.save_video is not None:
        args.save_video.mkdir(parents=True, exist_ok=True)
    if args.record_trace is not None:
        args.record_trace.mkdir(parents=True, exist_ok=True)

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
                commands: list[np.ndarray] = []
                requeried: list[bool] = []
                trace = TraceCollector() if args.record_trace is not None else None
                for tick in range(scenario.max_steps):
                    if skip_requested.is_set():
                        break
                    started = time.perf_counter()
                    if args.save_video is not None:
                        frames.append(observation_frame(observation))
                    if trace is not None:
                        trace.qpos.append(env.replay_qpos())
                    action = policy.act(observation)
                    commands.append(action)
                    requeried.append(policy.latest_prediction is not None)
                    if trace is not None:
                        trace.add_horizon(tick, policy)
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
                if trace is not None and trace.paths:
                    recording = trace.build(
                        fps=scenario.control_hz,
                        act_steps=args.act_steps,
                        target_xy=tuple(float(v) for v in scenario.target_position_m[:2]),
                    )
                    path = args.record_trace / f"{scenario.scenario_id}.bin"
                    path.write_bytes(encode(recording))
                    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
                if frames:
                    write_video(
                        args.save_video / f"{scenario.scenario_id}.mp4",
                        frames,
                        scenario.control_hz,
                    )
                milestones = info["milestones"]
                success = bool(info["success"])
                successes += success
                smoothness = summarize_smoothness(
                    np.stack(commands), np.array(requeried), joints=GRIPPER_INDEX
                )
                records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "success": success,
                        "control_steps": int(info["control_steps"]),
                        "milestones": milestones,
                        "clipped_fraction": policy.clipped_fraction,
                        "smoothness": smoothness,
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
        "noise_correlation": args.noise_correlation,
        "successes": successes,
        "success_rate": successes / completed,
        "lifted": sum(r["milestones"]["cube_lifted"] for r in records),
        "contact_attempted": sum(r["milestones"]["pickup_contact_attempted"] for r in records),
        "smoothness": {
            key: float(np.mean([r["smoothness"][key] for r in records]))
            for key in records[0]["smoothness"]
        },
        "episodes": records,
    }
    print(f"\n{successes}/{completed} = {successes / completed:.1%} success")
    print(f"lifted {summary['lifted']}, contact attempted {summary['contact_attempted']}")
    smoothness = summary["smoothness"]
    print(
        f"per-tick joint step: {smoothness['mean_step_deg']:.2f} deg mean, "
        f"{smoothness['interior_step_deg']:.2f} within a chunk vs "
        f"{smoothness['boundary_step_deg']:.2f} at replan boundaries "
        f"(worst {smoothness['max_step_deg']:.2f})"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(summary, file, indent=2)


def main() -> None:
    run(build_parser(description=__doc__).parse_args())


if __name__ == "__main__":
    main()
