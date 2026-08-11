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
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.policies.flow_image_encoder import FlowImageUnet1D
from pick_and_place.runtime.policy_sim import PolicySimEnv
from pick_and_place.runtime.training_scenes import training_scenario
from pick_and_place.sim.scene_appearance import parse_appearance
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def normalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    span = maximum - minimum
    return np.where(
        span > 1e-6, 2 * (values - minimum) / np.where(span > 1e-6, span, 1) - 1, 0
    ).astype(np.float32)


def unnormalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    span = maximum - minimum
    return np.where(span > 1e-6, (values + 1) / 2 * span + minimum, minimum).astype(np.float32)


def load_checkpoint(path: Path, device: torch.device) -> FlowImageUnet1D:
    contents = torch.load(path, map_location=device, weights_only=False)
    if contents.get("model_type") != "flow_image_unet1d":
        raise ValueError(f"{path} is not an image flow checkpoint")
    model = FlowImageUnet1D(**contents["model_config"]).to(device)
    model.load_state_dict(contents["model"])
    model.eval()
    return model


class ImageFlowPolicy:
    """Turn simulator observations into a queue of generated joint commands."""

    def __init__(
        self,
        model: FlowImageUnet1D,
        bounds: dict[str, np.ndarray],
        *,
        act_steps: int,
        integration_steps: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model
        self.device = device
        self.act_steps = act_steps
        self.integration_steps = integration_steps
        self.seed = seed
        self.observation_min = bounds["obs_min"]
        self.observation_max = bounds["obs_max"]
        self.action_min = bounds["action_min"]
        self.action_max = bounds["action_max"]
        self.mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
        self.images: deque[np.ndarray] = deque(maxlen=model.observation_steps)
        self.states: deque[np.ndarray] = deque(maxlen=model.observation_steps)
        self.actions: deque[np.ndarray] = deque()
        self.clipped_fraction = 0.0
        self.reset()

    def reset(self) -> None:
        self.images.clear()
        self.states.clear()
        self.actions.clear()
        torch.manual_seed(self.seed)

    def _pack(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # The export concatenates cameras on the channel axis in
        # camera_features order: overhead first, then wrist.
        overhead = np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8)
        wrist = np.asarray(observation[WRIST_FEATURE], dtype=np.uint8)
        stacked = np.concatenate(
            (np.moveaxis(overhead, -1, 0), np.moveaxis(wrist, -1, 0)), axis=0
        )
        state = normalize(
            np.asarray(observation[STATE_FEATURE], dtype=np.float32),
            self.observation_min,
            self.observation_max,
        )
        return stacked, state

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        del info
        images, state = self._pack(observation)
        self.images.append(images)
        self.states.append(state)
        while len(self.images) < self.model.observation_steps:
            self.images.appendleft(self.images[0].copy())
            self.states.appendleft(self.states[0].copy())

        if not self.actions:
            image_batch = torch.from_numpy(np.stack(self.images)[None]).to(self.device)
            steps, channels = image_batch.shape[1], image_batch.shape[2]
            floats = image_batch.float().div_(255.0).reshape(-1, 3, *image_batch.shape[-2:])
            floats = ((floats - self.mean) / self.std).reshape(
                1, steps, channels, *image_batch.shape[-2:]
            )
            state_batch = torch.from_numpy(np.stack(self.states)[None]).to(self.device)

            values = torch.randn(
                1, self.model.prediction_steps, self.model.action_dim, device=self.device
            )
            time = torch.zeros(1, 1, device=self.device)
            with torch.no_grad():
                condition = self.model.encode_observation(floats, state_batch)
                for _ in range(self.integration_steps):
                    velocity = self.model.unet(values, time, condition)
                    values = values + velocity / self.integration_steps
                    time = time + 1 / self.integration_steps
            generated = values[0].cpu().numpy()
            clipped = np.clip(generated, -1, 1)
            self.clipped_fraction = float(np.mean(generated != clipped))
            commands = unnormalize(clipped, self.action_min, self.action_max)
            self.actions.extend(commands[: self.act_steps])
        return self.actions.popleft()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="matching image export")
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=6_000_000)
    parser.add_argument("--act-steps", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene-appearance", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with (args.export / "export.json").open() as file:
        manifest = json.load(file)
    image_hw = tuple(int(value) for value in manifest["image_size"])
    with np.load(args.export / "normalization.npz", allow_pickle=False) as archive:
        bounds = {
            name: np.asarray(archive[name], dtype=np.float32)
            for name in ("obs_min", "obs_max", "action_min", "action_max")
        }

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    appearance = None
    if args.scene_appearance:
        _, appearance = parse_appearance(args.scene_appearance)

    env = PolicySimEnv(image_hw=image_hw, render_hw=(1080, 1920), scene_appearance=appearance)
    policy = ImageFlowPolicy(
        model,
        bounds,
        act_steps=args.act_steps,
        integration_steps=args.integration_steps,
        device=device,
        seed=args.policy_seed,
    )

    records = []
    successes = 0
    for index in range(args.scenarios):
        scenario = training_scenario(index, seed_base=args.seed_base)
        observation, info = env.reset(options={"scenario": scenario})
        policy.reset()
        for _ in range(scenario.max_steps):
            action = policy.act(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
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
            f"{index + 1}/{args.scenarios} {scenario.scenario_id}: "
            f"{'SUCCESS' if success else 'failure'}  running {successes}/{index + 1}",
            flush=True,
        )

    env.close()
    summary = {
        "checkpoint": str(args.checkpoint),
        "export": str(args.export),
        "scenarios": args.scenarios,
        "seed_base": args.seed_base,
        "act_steps": args.act_steps,
        "integration_steps": args.integration_steps,
        "scene_appearance": args.scene_appearance or "as-compiled",
        "successes": successes,
        "success_rate": successes / args.scenarios,
        "lifted": sum(r["milestones"]["cube_lifted"] for r in records),
        "contact_attempted": sum(r["milestones"]["pickup_contact_attempted"] for r in records),
        "episodes": records,
    }
    print(f"\n{successes}/{args.scenarios} = {successes / args.scenarios:.1%} success")
    print(f"lifted {summary['lifted']}, contact attempted {summary['contact_attempted']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
