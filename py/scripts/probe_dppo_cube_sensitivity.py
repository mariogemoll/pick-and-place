#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure whether a DPPO checkpoint changes its action for a moved visual cue.

The probe freezes robot state, camera calibration, and diffusion noise. It
moves either the source cube or target plate, so action deltas isolate
sensitivity to that visual location rather than closed-loop behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pick_and_place.dppo_policy import DppoPolicyController
from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policy_evaluation import ScenarioManifest
from pick_and_place.policy_sim import PolicySimEnv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "evaluation" / "smoke_v1.json"
DEFAULT_DPPO_CONFIG = (
    REPOSITORY_ROOT / "config" / "diffusion_policy" / "pretrain_so101_unet_img.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dppo-normalization", type=Path, required=True)
    parser.add_argument("--dppo-python", type=Path, default=os.environ.get("DPPO_PYTHON"))
    parser.add_argument("--dppo-config", type=Path, default=DEFAULT_DPPO_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario-id", default="smoke_v1-000")
    parser.add_argument(
        "--initial-robot-state",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "GRIPPER"),
        help="replace the scenario's neutral state, for an approach-state probe",
    )
    parser.add_argument(
        "--previous-robot-state",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "GRIPPER"),
        help="previous control-tick state; enables a true two-frame history probe",
    )
    parser.add_argument(
        "--vary",
        choices=("cube", "target"),
        default="cube",
        help="visual cue to displace while all other scene state remains fixed",
    )
    parser.add_argument(
        "--offset-m",
        type=float,
        default=0.10,
        help="displacement in each cardinal direction (default: 0.10)",
    )
    parser.add_argument("--seed", type=int, default=0, help="fixed diffusion sampling seed")
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=10,
        help="DDIM steps for this fast diagnostic (default: 10)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render-height", type=int, default=1080)
    parser.add_argument("--render-width", type=int, default=1920)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dppo_python is None:
        parser.error("--dppo-python (or $DPPO_PYTHON) is required")
    for name in ("checkpoint", "dppo_normalization", "dppo_python", "dppo_config", "manifest"):
        if not Path(getattr(args, name)).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    if args.offset_m <= 0.0:
        parser.error("--offset-m must be positive")
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    return args


def main() -> None:
    args = _parse_args()
    manifest = ScenarioManifest.load(args.manifest)
    try:
        base_scenario = next(
            scenario for scenario in manifest.scenarios if scenario.scenario_id == args.scenario_id
        )
    except StopIteration as exc:
        raise ValueError(f"scenario not found in manifest: {args.scenario_id}") from exc
    if args.initial_robot_state is not None:
        base_scenario = replace(
            base_scenario,
            initial_robot_state_real=tuple(args.initial_robot_state),
        )

    controller = DppoPolicyController.launch(
        python=args.dppo_python,
        checkpoint=args.checkpoint,
        config=args.dppo_config,
        normalization=args.dppo_normalization,
        device=args.device,
        seed=args.seed,
        ddim_steps=args.ddim_steps,
    )
    probes = (
        ("center", 0.0, 0.0),
        (f"{args.vary}_plus_x", args.offset_m, 0.0),
        (f"{args.vary}_minus_x", -args.offset_m, 0.0),
        (f"{args.vary}_plus_y", 0.0, args.offset_m),
        (f"{args.vary}_minus_y", 0.0, -args.offset_m),
    )
    args.output.mkdir(parents=True)
    env = PolicySimEnv(
        image_hw=controller.image_hw,
        render_hw=(args.render_height, args.render_width),
    )
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "scenario_id": base_scenario.scenario_id,
        "varied_feature": args.vary,
        "source_position_m": base_scenario.source_position_m,
        "target_position_m": base_scenario.target_position_m,
        "offset_m": args.offset_m,
        "sampling_seed": args.seed,
        "sampler": controller.handshake["sampler"],
        "probes": [],
    }
    try:
        for name, dx, dy in probes:
            if args.vary == "cube":
                source = (
                    base_scenario.source_position_m[0] + dx,
                    base_scenario.source_position_m[1] + dy,
                    base_scenario.source_position_m[2],
                )
                scenario = replace(base_scenario, source_position_m=source)
                varied_position = source
            else:
                target = (
                    base_scenario.target_position_m[0] + dx,
                    base_scenario.target_position_m[1] + dy,
                    base_scenario.target_position_m[2],
                )
                scenario = replace(base_scenario, target_position_m=target)
                varied_position = target
            observation, _ = env.reset(options={"scenario": scenario})
            if args.previous_robot_state is None:
                actions = controller.predict_horizon(observation, sampling_seed=args.seed)
            else:
                previous_scenario = replace(
                    scenario,
                    initial_robot_state_real=tuple(args.previous_robot_state),
                )
                previous_observation, _ = env.reset(options={"scenario": previous_scenario})
                actions = controller.predict_horizon_from_history(
                    [previous_observation, observation],
                    sampling_seed=args.seed,
                )
            iio.imwrite(args.output / f"{name}-overhead.png", observation[OVERHEAD_FEATURE])
            iio.imwrite(args.output / f"{name}-wrist.png", observation[WRIST_FEATURE])
            np.savez_compressed(
                args.output / f"{name}-input.npz",
                state=observation[STATE_FEATURE],
                overhead=observation[OVERHEAD_FEATURE],
                wrist=observation[WRIST_FEATURE],
            )
            report["probes"].append({
                "name": name,
                f"{args.vary}_position_m": varied_position,
                "first_action": actions[0].tolist(),
                "action_horizon": actions.tolist(),
            })
    finally:
        env.close()
        controller.close()

    probes_report = report["probes"]
    assert isinstance(probes_report, list)
    baseline = np.asarray(probes_report[0]["first_action"], dtype=np.float32)
    for probe in probes_report:
        first_action = np.asarray(probe["first_action"], dtype=np.float32)
        delta = first_action - baseline
        probe["first_action_delta_from_center"] = delta.tolist()
        probe["arm_delta_l2_deg"] = float(np.linalg.norm(delta[:5]))
        probe["gripper_delta"] = float(delta[5])
        print(
            f"{probe['name']}: first action={np.array2string(first_action, precision=2)}, "
            f"arm_delta={probe['arm_delta_l2_deg']:.3f}°, "
            f"gripper_delta={probe['gripper_delta']:+.3f}"
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote exact inputs and report to {args.output}")


if __name__ == "__main__":
    main()
