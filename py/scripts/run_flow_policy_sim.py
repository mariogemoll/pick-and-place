#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a state-conditioned flow policy in the pick-and-place simulator."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.policies.flow_matching import (
    VelocityModel,
    generate,
    load_model,
)
from pick_and_place.policies.flow_policy import (
    load_export,
    normalize,
    pack_observation,
    unnormalize,
)
from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D
from pick_and_place.runtime.policy_sim import PolicySimEnv
from pick_and_place.runtime.recorded_scenes import recorded_episode_scenario
from pick_and_place.runtime.training_scenes import training_scenario

ENTER_KEYS = frozenset({257, 335})


class StateFlowPolicy:
    """Turn simulator observations into a queue of generated joint commands."""

    def __init__(
        self,
        model: VelocityModel,
        manifest: dict[str, Any],
        bounds: dict[str, np.ndarray],
        *,
        act_steps: int,
        integration_steps: int,
        seed: int,
    ) -> None:
        if manifest.get("endpoint_semantics") != "absolute joint command":
            raise ValueError("the simulator runner requires absolute joint commands")
        self.model = model
        self.observation_steps = int(manifest["observation_steps"])
        self.prediction_steps = int(manifest["prediction_steps"])
        self.observation_dim = int(manifest["observation_dim"])
        self.endpoint_dim = int(manifest["endpoint_dim"])
        expected_input = (
            self.prediction_steps * self.endpoint_dim
            + model.time_embedding_dim
            + self.observation_steps * self.observation_dim
        )
        if (model.input_dim, model.output_dim) != (
            expected_input,
            self.prediction_steps * self.endpoint_dim,
        ):
            raise ValueError("checkpoint dimensions do not match the export")
        if isinstance(model, FlowConditionalUnet1D) and (
            model.prediction_steps != self.prediction_steps or model.action_dim != self.endpoint_dim
        ):
            raise ValueError("U-Net temporal dimensions do not match the export")
        if not 1 <= act_steps <= self.prediction_steps or integration_steps < 1:
            raise ValueError("invalid action or integration horizon")
        self.minimum = bounds["observation_min"]
        self.maximum = bounds["observation_max"]
        self.endpoint_minimum = bounds["endpoint_min"]
        self.endpoint_maximum = bounds["endpoint_max"]
        self.act_steps, self.integration_steps, self.seed = act_steps, integration_steps, seed
        self.history: deque[np.ndarray] = deque(maxlen=self.observation_steps)
        self.actions: deque[np.ndarray] = deque()
        self.clipped_fraction = 0.0
        self.reset()

    def reset(self) -> None:
        self.history.clear()
        self.actions.clear()
        torch.manual_seed(self.seed)

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        current = pack_observation(observation, info)
        if current.shape != (self.observation_dim,):
            raise ValueError(f"packed observation must have shape ({self.observation_dim},)")
        self.history.append(normalize(current, self.minimum, self.maximum))
        while len(self.history) < self.observation_steps:
            self.history.appendleft(self.history[0].copy())
        if not self.actions:
            generated = (
                generate(
                    self.model,
                    torch.from_numpy(np.stack(self.history).reshape(1, -1)),
                    self.integration_steps,
                )[0]
                .cpu()
                .numpy()
            )
            clipped = np.clip(generated, -1, 1)
            self.clipped_fraction = float(np.mean(generated != clipped))
            commands = unnormalize(
                clipped.reshape(self.prediction_steps, self.endpoint_dim),
                self.endpoint_minimum,
                self.endpoint_maximum,
            )
            self.actions.extend(commands[: self.act_steps])
        return self.actions.popleft()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def format_episode_result(result: Any) -> str:
    """Format a compact result summary while cycling through scenes."""
    milestones = asdict(result.milestones)
    failures = [name for name, enabled in asdict(result.failures).items() if enabled]
    progress = " ".join(
        f"{name}={'yes' if milestones[key] else 'no'}"
        for name, key in (
            ("lifted", "cube_lifted"),
            ("reached", "target_reached_while_holding"),
            ("released", "cube_released"),
            ("settled", "cube_settled"),
        )
    )
    return (
        f"Result {result.scenario_id}: {'SUCCESS' if result.success else 'FAILURE'} | "
        f"xy error={result.final_xy_error_m * 1000:.1f} mm | steps={result.control_steps} | "
        f"{progress} | flags={','.join(failures) if failures else 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--act-steps", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=100)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-width", type=int, default=960)
    args = parser.parse_args()
    manifest, bounds = load_export(args.export)
    recorded_indices = sorted(
        set(manifest["training_episode_indices"]) | set(manifest["validation_episode_indices"])
    )
    if args.episode_index is not None and args.episode_index not in recorded_indices:
        raise ValueError(f"episode {args.episode_index} is not included in the export")

    def make_scenario(index: int):
        if args.episode_index is None:
            scenario = training_scenario(index)
            return replace(scenario, max_steps=args.steps) if args.steps is not None else scenario
        return recorded_episode_scenario(
            manifest["source_dataset"],
            index,
            control_hz=float(manifest["policy_hz"]),
            max_steps=args.steps,
        )

    scenario_index = args.episode_index if args.episode_index is not None else args.scenario_index
    recorded_position = (
        recorded_indices.index(scenario_index) if args.episode_index is not None else None
    )
    scenario = make_scenario(scenario_index)
    if scenario.control_hz != float(manifest["policy_hz"]):
        raise ValueError("export and scenario control rates do not match")
    policy = StateFlowPolicy(
        load_model(args.checkpoint, select_device(args.device)),
        manifest,
        bounds,
        act_steps=args.act_steps,
        integration_steps=args.integration_steps,
        seed=args.seed,
    )
    env = PolicySimEnv(
        image_hw=(96, 96), render_hw=(args.render_height, args.render_width), include_images=False
    )
    observation, info = env.reset(options={"scenario": scenario})
    resample_requested = threading.Event()
    context = nullcontext(None)
    if not args.headless:
        import mujoco.viewer

        context = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=lambda keycode: (
                resample_requested.set() if keycode in ENTER_KEYS else None
            ),
        )
    try:
        with context as viewer:
            terminated = truncated = False
            while viewer is None or viewer.is_running():
                if resample_requested.is_set():
                    resample_requested.clear()
                    if recorded_position is None:
                        scenario_index += 1
                    else:
                        recorded_position = (recorded_position + 1) % len(recorded_indices)
                        scenario_index = recorded_indices[recorded_position]
                    scenario = make_scenario(scenario_index)
                    policy.reset()
                    observation, info = env.reset(options={"scenario": scenario})
                    terminated = truncated = False
                    print(f"Resampled: {scenario.scenario_id}")
                if terminated or truncated:
                    if viewer is None:
                        break
                    viewer.sync()
                    time.sleep(1 / scenario.control_hz)
                    continue
                started = time.perf_counter()
                observation, _, terminated, truncated, info = env.step(
                    policy.act(observation, info)
                )
                if terminated or truncated:
                    print(format_episode_result(env.episode_result()))
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0, 1 / scenario.control_hz - (time.perf_counter() - started)))
        print(json.dumps(asdict(env.episode_result()), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
