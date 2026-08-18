#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record a closed-loop flow-policy rollout for the browser to reproduce.

The browser page reimplements two things Python already does: the environment's
reset and step, and the contract around the flow policy -- how an observation is
packed and normalized, and how a horizon becomes executed actions. Neither is
hard, and both are the kind of thing that can be subtly wrong for an entire
episode without throwing.

So this runs the real ``PolicySimEnv`` against the real checkpoint and writes
down everything the browser needs to be checked against: the scene it started
from, the noise draw behind every horizon, and the observation, action and full
qpos at every tick. ``ts/src/visualizations/live-policy/parity.test.ts`` replays
the same noise through the TypeScript stack and has to land on the same
trajectory.

Noise is drawn here from a seeded NumPy generator and handed to the sampler
explicitly, rather than left to ``generate``'s internal ``torch.randn``. That is
the only way the two languages can be on the same draws, and it is the same
sampler either way -- ``FlowSampler`` is what the ONNX export traces.

The fixture lands in ``ts/public/`` because it needs a checkpoint and a compiled
scene to produce, which is exactly why it is generated rather than committed.

Usage::

    python scripts/export_policy_parity_fixture.py \\
        --checkpoint .../checkpoint.pt --export .../flow-policy-state-.../ \\
        -o ../ts/public/policy-parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pick_and_place.policies.flow_matching import load_model
from pick_and_place.policies.flow_onnx import FlowSampler
from pick_and_place.policies.flow_policy import load_export, normalize, pack_observation, unnormalize
from pick_and_place.runtime.policy_sim import PolicySimEnv
from pick_and_place.runtime.training_scenes import training_scenario

DEFAULT_ACT_STEPS = 8
DEFAULT_INTEGRATION_STEPS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--scenario-index", type=int, default=6_000_000)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--act-steps", type=int, default=DEFAULT_ACT_STEPS)
    parser.add_argument("--integration-steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest, bounds = load_export(args.export)
    sampler = FlowSampler(load_model(args.checkpoint, "cpu"), args.integration_steps).eval()
    observation_steps = int(manifest["observation_steps"])
    observation_dim = int(manifest["observation_dim"])
    prediction_steps = int(manifest["prediction_steps"])
    endpoint_dim = int(manifest["endpoint_dim"])

    scenario = training_scenario(args.scenario_index)
    env = PolicySimEnv(image_hw=(96, 96), render_hw=(96, 96), include_images=False)
    observation, info = env.reset(options={"scenario": scenario})

    generator = np.random.default_rng(args.seed)
    history: list[np.ndarray] = []
    queue: list[np.ndarray] = []
    noise_draws: list[list[float]] = []
    ticks: list[dict[str, object]] = []

    for _ in range(args.ticks):
        packed = pack_observation(observation, info)
        history.append(normalize(packed, bounds["observation_min"], bounds["observation_max"]))
        while len(history) < observation_steps:
            history.insert(0, history[0].copy())
        history = history[-observation_steps:]

        drew_noise = False
        if not queue:
            noise = generator.standard_normal(prediction_steps * endpoint_dim).astype(np.float32)
            noise_draws.append([float(v) for v in noise])
            drew_noise = True
            with torch.no_grad():
                generated = sampler(
                    torch.from_numpy(np.stack(history).reshape(1, -1)),
                    torch.from_numpy(noise.reshape(1, -1)),
                )[0].numpy()
            commands = unnormalize(
                np.clip(generated, -1, 1).reshape(prediction_steps, endpoint_dim),
                bounds["endpoint_min"],
                bounds["endpoint_max"],
            )
            queue = [row.copy() for row in commands[: args.act_steps]]

        action = queue.pop(0)
        ticks.append(
            {
                "packedObservation": [float(v) for v in packed],
                "drewNoise": drew_noise,
                "action": [float(v) for v in action],
                "qposBefore": [float(v) for v in env.replay_qpos()],
            }
        )
        observation, _, terminated, truncated, info = env.step(action)
        ticks[-1]["qposAfter"] = [float(v) for v in env.replay_qpos()]
        if terminated or truncated:
            break

    payload = {
        "SPDX-FileCopyrightText": "2026 Mario Gemoll",
        "SPDX-License-Identifier": "0BSD",
        "description": (
            "A closed-loop state flow-policy rollout recorded from PolicySimEnv, for the "
            "browser page to reproduce tick for tick."
        ),
        "generator": "py/scripts/export_policy_parity_fixture.py",
        "scenarioId": scenario.scenario_id,
        "actSteps": args.act_steps,
        "integrationSteps": args.integration_steps,
        "observationDim": observation_dim,
        "setup": {
            "cube": {
                "position": [float(v) for v in scenario.source_position_m],
                "quaternion": [float(v) for v in scenario.source_orientation_wxyz],
            },
            "targetXy": [float(v) for v in scenario.target_position_m[:2]],
            "initialJointsReal": [float(v) for v in scenario.initial_robot_state_real],
        },
        "noiseDraws": noise_draws,
        "ticks": ticks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(payload, file)
        file.write("\n")
    env.close()
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(ticks)} ticks, {len(noise_draws)} horizons, scenario {scenario.scenario_id}")


if __name__ == "__main__":
    main()
