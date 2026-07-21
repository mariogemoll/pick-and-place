#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a LeRobot checkpoint on a frozen visual-policy simulator manifest."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
from pathlib import Path

from pick_and_place.policy import (
    DEFAULT_INSTRUCTION,
    resolve_checkpoint_cameras,
    select_device,
)
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    WRIST_FEATURE,
    LeRobotPolicyController,
)
from pick_and_place.policy_evaluation import (
    ScenarioManifest,
    TaskOracleConfig,
    fingerprint_checkpoint,
    git_provenance,
    package_versions,
    write_evaluation_artifacts,
)
from pick_and_place.policy_sim import PolicySimEnv, evaluate_policy_episode

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "evaluation" / "smoke_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local LeRobot checkpoint or repository ID")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"frozen scenario manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--output", type=Path, required=True, help="new evaluation run directory")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--render-height", type=int, default=1080)
    parser.add_argument("--render-width", type=int, default=1920)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="override queued actions executed per policy query",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="enable ACT temporal ensembling (requires --n-action-steps 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N scenarios for a non-headline wiring check",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="save the exact overhead and wrist policy frames for every scenario",
    )
    args = parser.parse_args()
    if (args.image_height is None) != (args.image_width is None):
        parser.error("pass both --image-height and --image-width, or neither")
    if args.image_height is not None and min(args.image_height, args.image_width) < 1:
        parser.error("image dimensions must be positive")
    if min(args.render_height, args.render_width) < 1:
        parser.error("render dimensions must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    return args


class _EpisodeVideoWriters:
    def __init__(self, directory: Path, scenario_id: str, fps: float) -> None:
        import imageio.v2 as imageio

        directory.mkdir(parents=True, exist_ok=True)
        self._overhead = imageio.get_writer(directory / f"{scenario_id}-overhead.mp4", fps=fps)
        self._wrist = imageio.get_writer(directory / f"{scenario_id}-wrist.mp4", fps=fps)

    def append(self, step: int, observation) -> None:
        del step
        self._overhead.append_data(observation[OVERHEAD_FEATURE])
        self._wrist.append_data(observation[WRIST_FEATURE])

    def close(self) -> None:
        self._overhead.close()
        self._wrist.close()


def _policy_metadata(controller: LeRobotPolicyController) -> dict:
    config = controller.policy.config
    return {
        "type": getattr(config, "type", type(controller.policy).__name__),
        "image_features": {
            "overhead": controller.image_keys[0],
            "wrist": controller.image_keys[1],
        },
        "checkpoint_image_feature_order": list(getattr(config, "image_features", [])),
        "action_horizon": getattr(config, "chunk_size", None),
        "executed_action_steps": getattr(config, "n_action_steps", None),
        "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
    }


def main() -> None:
    args = _parse_args()
    started_at = dt.datetime.now(dt.UTC)
    manifest = ScenarioManifest.load(args.manifest)
    scenarios = manifest.scenarios[: args.limit] if args.limit is not None else manifest.scenarios
    override_hw = (
        (args.image_height, args.image_width) if args.image_height is not None else None
    )
    image_hw, _ = resolve_checkpoint_cameras(args.checkpoint, override_hw=override_hw)
    if args.render_height < image_hw[0] or args.render_width < image_hw[1]:
        raise ValueError("render dimensions must be at least the checkpoint image dimensions")

    device = select_device(args.device)
    print(f"Loading {args.checkpoint} on {device}...")
    controller = LeRobotPolicyController.from_checkpoint(
        args.checkpoint,
        device=device,
        image_hw=image_hw,
        instruction=args.instruction,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    print(
        f"Evaluating {len(scenarios)}/{len(manifest.scenarios)} {manifest.suite!r} scenarios "
        f"at {image_hw[1]}x{image_hw[0]}."
    )

    env = PolicySimEnv(
        image_hw=image_hw,
        render_hw=(args.render_height, args.render_width),
    )
    results = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            writers = None
            if args.save_videos:
                writers = _EpisodeVideoWriters(
                    args.output / "videos",
                    scenario.scenario_id,
                    scenario.control_hz,
                )
            try:
                result = evaluate_policy_episode(
                    env,
                    controller,
                    scenario,
                    observation_callback=writers.append if writers is not None else None,
                )
            finally:
                if writers is not None:
                    writers.close()
            results.append(result)
            status = "SUCCESS" if result.success else "failure"
            print(
                f"[{index:02d}/{len(scenarios):02d}] {scenario.scenario_id}: {status}, "
                f"steps={result.control_steps}, final_xy={result.final_xy_error_m * 100:.1f} cm"
            )
    finally:
        env.close()

    run = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "checkpoint": {
            "path_or_repository_id": args.checkpoint,
            "fingerprint": fingerprint_checkpoint(args.checkpoint),
        },
        "policy": _policy_metadata(controller),
        "instruction": args.instruction,
        "scenario_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest.sha256(),
            "suite": manifest.suite,
            "selected_scenario_ids": [scenario.scenario_id for scenario in scenarios],
            "complete_suite": len(scenarios) == len(manifest.scenarios),
        },
        "environment": {
            "image_height": image_hw[0],
            "image_width": image_hw[1],
            "render_height": args.render_height,
            "render_width": args.render_width,
            "control_hz": sorted({scenario.control_hz for scenario in scenarios}),
            "episode_step_limits": sorted({scenario.max_steps for scenario in scenarios}),
            "domain_randomization_presets": sorted({
                scenario.domain_randomization_preset or "none" for scenario in scenarios
            }),
            "oracle": asdict(TaskOracleConfig()),
            "state_frame": "hardware (arm degrees, gripper position 0-100)",
            "action_frame": "hardware (arm degrees, gripper position 0-100)",
        },
        "device": str(device),
        "code": git_provenance(REPOSITORY_ROOT),
        "package_versions": package_versions(["gymnasium", "lerobot", "mujoco", "numpy", "torch"]),
        "videos_saved": args.save_videos,
    }
    summary = write_evaluation_artifacts(args.output, run, results)
    print(
        f"Wrote {args.output}: {summary['success_count']}/{summary['episode_count']} "
        f"successes ({summary['success_rate']:.1%})."
    )


if __name__ == "__main__":
    main()
