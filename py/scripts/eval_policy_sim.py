#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a learned or scripted controller on a frozen simulator manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from pick_and_place.policies.policy import (
    DEFAULT_IMAGE_HW,
    resolve_checkpoint_cameras,
    select_device,
)
from pick_and_place.policies.flow_image_policy import FlowImagePolicyController
from pick_and_place.spec.controller import OVERHEAD_FEATURE, WRIST_FEATURE
from pick_and_place.policies.policy_controllers import LeRobotPolicyController
from pick_and_place.policies.policy_evaluation import (
    ScenarioManifest,
    TaskOracleConfig,
    fingerprint_checkpoint,
    git_provenance,
    package_versions,
    write_evaluation_artifacts,
)
from pick_and_place.runtime.policy_sim import (
    evaluate_policy_episode,
    PolicySimEnv,
)
from pick_and_place.runtime.replay_rollout import write_rollout
from pick_and_place.rollout.scripted_sim import sim_scripted_controller
from pick_and_place.cli.policy import (
    add_checkpoint_argument,
    add_device_argument,
    add_flow_image_arguments,
    add_lerobot_arguments,
    add_policy_image_arguments,
)
from pick_and_place.cli.scene import add_render_size_arguments, add_scene_appearance_arguments
from pick_and_place.variants.appearance import parse_appearance

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "evaluation" / "smoke_v1.json"
SCRIPTED_IMAGE_HW = DEFAULT_IMAGE_HW


def _parse_args() -> argparse.Namespace:
    # The manifest, the output and the world the scenarios run in: shared, so a
    # flow number and a SmolVLA number are produced under one declaration.
    common = argparse.ArgumentParser(add_help=False)
    parser = common
    add_policy_image_arguments(parser)
    add_render_size_arguments(parser)
    add_scene_appearance_arguments(parser)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"frozen scenario manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--output", type=Path, required=True, help="new evaluation run directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N scenarios for a non-headline wiring check",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "skip the first N scenarios, applied before --limit. Together the two shard one "
            "suite across concurrent workers; the shards stay comparable because each scenario "
            "is independent and carries its own seed"
        ),
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help=("cap each scenario's simulated duration; useful for fast approach-only diagnostics"),
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="save the exact overhead and wrist policy frames for every scenario",
    )
    parser.add_argument(
        "--save-rollouts",
        action="store_true",
        help=(
            "save every scenario's per-frame qpos as a PPRL file the browser episode "
            "viewer replays; a few kilobytes an episode against megabytes of video"
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    leaves = parser.add_subparsers(dest="controller", required=True, metavar="CONTROLLER")

    lerobot = leaves.add_parser(
        "lerobot",
        parents=[common],
        help="a LeRobot checkpoint (ACT, SmolVLA, pi0.5, ...)",
        description="Score a LeRobot checkpoint against a frozen scenario manifest.",
    )
    add_lerobot_arguments(
        lerobot, checkpoint_default=None, checkpoint_required=True, n_action_steps_default=None
    )

    flow_image = leaves.add_parser(
        "flow-image",
        parents=[common],
        help="the image-conditioned flow-matching policy",
        description="Score an image-flow export against a frozen scenario manifest.",
    )
    add_checkpoint_argument(
        flow_image, default=None, required=True, help="flow-policy checkpoint-*.pt file"
    )
    add_device_argument(flow_image)
    # recording_hw=False: the evaluator renders at the target resolution directly,
    # so the flag the live runners need would parse here and do nothing.
    add_flow_image_arguments(flow_image, recording_hw=False, flow_export_required=True)

    scripted = leaves.add_parser(
        "scripted",
        parents=[common],
        help="the expert: localize, plan, servo the descent, replan at each phase",
        description="Score the expert against a frozen scenario manifest.",
    )
    scripted.add_argument(
        "--scripted-perception",
        choices=("geometric", "detector"),
        default="geometric",
        help=(
            "simulated overhead perception: geometric uses the 80%% segmentation "
            "visibility gate and controlled pose beliefs; detector runs the real "
            "optical pipeline (default: geometric)"
        ),
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
    if args.offset < 0:
        parser.error("--offset must not be negative")
    if args.scene_appearance is not None:
        try:
            parse_appearance(args.scene_appearance)
        except ValueError as exc:
            parser.error(str(exc))
    if args.max_episode_seconds is not None and (
        not math.isfinite(args.max_episode_seconds) or args.max_episode_seconds <= 0.0
    ):
        parser.error("--max-episode-seconds must be a positive finite number")
    if args.controller == "flow-image":
        for name, path in (("checkpoint", args.checkpoint), ("flow-export", args.flow_export)):
            if not Path(path).exists():
                parser.error(f"--{name} does not exist: {path}")
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


class _EpisodeRolloutWriter:
    """Collect one episode's replay state, then write it as a PPRL file.

    The frames come from the environment rather than the observation the
    callback carries: an image is what the policy saw, and this is where the
    scene actually was.
    """

    def __init__(self, directory: Path, scenario_id: str, env, target_xy, fps: float) -> None:
        self._path = directory / f"{scenario_id}.bin"
        self._env = env
        self._target_xy = target_xy
        self._fps = fps
        self._frames: list[np.ndarray] = []

    def append(self, step: int, observation) -> None:
        del step, observation
        self._frames.append(self._env.replay_qpos())

    def close(self) -> None:
        if not self._frames:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_rollout(self._path, np.stack(self._frames), self._fps, self._target_xy)


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lerobot_metadata(controller: LeRobotPolicyController) -> dict:
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


def _flow_image_metadata(controller: FlowImagePolicyController, args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    export = Path(args.flow_export)
    return {
        "type": "flow-image",
        "model_type": checkpoint["model_type"],
        "model_config": checkpoint["model_config"],
        "image_features": {
            "overhead": OVERHEAD_FEATURE,
            "wrist": WRIST_FEATURE,
        },
        "observation_steps": controller.observation_steps,
        "action_horizon": controller.prediction_steps,
        "executed_action_steps": controller.act_steps,
        "policy_hz": controller.policy_hz,
        "integration": "euler",
        "integration_steps": controller.integration_steps,
        "sampling_seed": controller.seed,
        "noise_correlation": controller.noise_correlation,
        "image_augmentation_at_rollout": False,
        "training": {
            "update": checkpoint.get("update"),
            "seed": checkpoint.get("seed"),
            "random_shift": checkpoint.get("random_shift"),
            "random_scale_pct": checkpoint.get("random_scale_pct"),
            "photometric_augmentation": checkpoint.get("photometric_augmentation"),
        },
        "export": {
            "path": str(export.resolve()),
            "manifest_sha256": _sha256_of_file(export / "export.json"),
            "normalization_sha256": _sha256_of_file(export / "normalization.npz"),
        },
    }


def _camera_base_metadata(model: mujoco.MjModel) -> dict:
    """The compiled scene's camera geometry, recorded with every scored run.

    Domain randomization is a displacement applied to these poses, so two runs
    that agree on a scenario manifest but disagree here are not comparable --
    and until this was written down, nothing in a policy run's record would have
    shown it. The manifest pins the jitter; this pins what the jitter is
    relative to.
    """
    cameras = {}
    for name in ("overhead_camera", "wrist_camera"):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            continue
        cameras[name] = {
            "position_m": [float(value) for value in model.cam_pos[camera_id]],
            "quat_wxyz": [float(value) for value in model.cam_quat[camera_id]],
            "fovy_deg": float(model.cam_fovy[camera_id]),
        }
    return {"source": "authored", "cameras": cameras}


def main() -> None:
    args = _parse_args()
    started_at = dt.datetime.now(dt.UTC)
    manifest = ScenarioManifest.load(args.manifest)
    if args.offset >= len(manifest.scenarios):
        raise SystemExit(
            f"--offset {args.offset} skips the whole {len(manifest.scenarios)}-scenario suite"
        )
    selected = manifest.scenarios[args.offset :]
    scenarios = selected[: args.limit] if args.limit is not None else selected
    override_hw = (args.image_height, args.image_width) if args.image_height is not None else None
    appearance_name, scene_appearance = (
        parse_appearance(args.scene_appearance)
        if args.scene_appearance is not None
        else (None, None)
    )
    if args.controller == "lerobot":
        image_hw, _ = resolve_checkpoint_cameras(args.checkpoint, override_hw=override_hw)
    elif args.controller == "flow-image":
        flow_device = select_device(args.device)
        print(f"Loading {args.checkpoint} on {flow_device}...")
        flow_image_controller = FlowImagePolicyController.from_export(
            args.checkpoint,
            args.flow_export,
            act_steps=args.flow_act_steps,
            integration_steps=args.flow_integration_steps,
            device=flow_device,
            seed=args.flow_seed,
            noise_correlation=args.flow_noise_correlation,
        )
        if override_hw is not None and override_hw != flow_image_controller.image_hw:
            raise ValueError(
                f"--image-height/--image-width {override_hw} do not match the "
                f"model's trained image size {flow_image_controller.image_hw}"
            )
        image_hw = flow_image_controller.image_hw
        scenarios = tuple(
            replace(
                scenario,
                control_hz=flow_image_controller.policy_hz,
                max_steps=max(
                    1,
                    round(
                        scenario.max_steps * flow_image_controller.policy_hz / scenario.control_hz
                    ),
                ),
            )
            for scenario in scenarios
        )
    else:
        image_hw = override_hw or SCRIPTED_IMAGE_HW
    if args.render_height < image_hw[0] or args.render_width < image_hw[1]:
        raise ValueError("render dimensions must be at least the controller image dimensions")

    if args.max_episode_seconds is not None:
        scenarios = tuple(
            replace(
                scenario,
                max_steps=min(
                    scenario.max_steps,
                    max(1, round(args.max_episode_seconds * scenario.control_hz)),
                ),
            )
            for scenario in scenarios
        )

    env = PolicySimEnv(
        image_hw=image_hw,
        render_hw=(args.render_height, args.render_width),
        scene_appearance=scene_appearance,
    )

    control_hz_values = {scenario.control_hz for scenario in scenarios}
    if args.controller == "scripted" and len(control_hz_values) != 1:
        raise ValueError("scripted evaluation requires one control frequency per run")
    if args.controller == "lerobot":
        device = select_device(args.device)
        print(f"Loading {args.checkpoint} on {device}...")
        controller = LeRobotPolicyController.from_checkpoint(
            args.checkpoint,
            device=device,
            image_hw=image_hw,
            instruction=args.instruction,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            base_checkpoint=args.base_checkpoint,
        )
        controller_metadata = _lerobot_metadata(controller)
    elif args.controller == "flow-image":
        device = flow_device
        controller = flow_image_controller
        controller_metadata = _flow_image_metadata(flow_image_controller, args)
    else:
        device = None
        controller, controller_metadata = sim_scripted_controller(
            image_hw=image_hw,
            render_hw=(args.render_height, args.render_width),
            control_hz=next(iter(control_hz_values)),
            scene_model=env.model,
            scene_data=env.data,
            perception=args.scripted_perception,
            cube_belief_error=lambda: env.cube_belief_error,
        )
    print(
        f"Evaluating {len(scenarios)}/{len(manifest.scenarios)} {manifest.suite!r} scenarios "
        f"with {args.controller} "
        f"at {image_hw[1]}x{image_hw[0]} and {next(iter(control_hz_values)):g} Hz, "
        f"scene appearance {args.scene_appearance or 'as-compiled'}."
    )

    camera_base = _camera_base_metadata(env.model)
    results = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            writers = []
            if args.save_videos:
                writers.append(_EpisodeVideoWriters(
                    args.output / "videos",
                    scenario.scenario_id,
                    scenario.control_hz,
                ))
            if args.save_rollouts:
                writers.append(_EpisodeRolloutWriter(
                    args.output / "rollouts",
                    scenario.scenario_id,
                    env,
                    scenario.target_position_m[:2],
                    scenario.control_hz,
                ))

            def record(step: int, observation, _writers=writers) -> None:
                for writer in _writers:
                    writer.append(step, observation)

            try:
                result = evaluate_policy_episode(
                    env,
                    controller,
                    scenario,
                    observation_callback=record if writers else None,
                )
            finally:
                for writer in writers:
                    writer.close()
            results.append(result)
            status = "SUCCESS" if result.success else "failure"
            failure_detail = (
                f", controller_failure={result.controller_failure['code']}"
                if result.controller_failure is not None
                else ""
            )
            print(
                f"[{index:02d}/{len(scenarios):02d}] {scenario.scenario_id}: {status}, "
                f"steps={result.control_steps}, final_xy={result.final_xy_error_m * 100:.1f} cm, "
                f"closest_tcp={result.min_tcp_to_cube_distance_m * 100:.1f} cm, "
                f"approach={result.tcp_to_cube_distance_reduction_m * 100:.1f} cm"
                f"{failure_detail}"
            )
    finally:
        env.close()
        close_controller = getattr(controller, "close", None)
        if close_controller is not None:
            close_controller()

    checkpoint = getattr(args, "checkpoint", None)
    run = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        # A leaf that takes no checkpoint does not declare the flag, so absent
        # and unset are the same thing here and both mean "not a checkpoint run".
        "checkpoint": (
            {
                "path_or_repository_id": checkpoint,
                "fingerprint": fingerprint_checkpoint(checkpoint),
            }
            if checkpoint is not None
            else None
        ),
        "controller": controller_metadata,
        "instruction": getattr(args, "instruction", None),
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
            "requested_max_episode_seconds": args.max_episode_seconds,
            "domain_randomization_presets": sorted(
                {scenario.domain_randomization_preset or "none" for scenario in scenarios}
            ),
            "scene_appearance": appearance_name,
            "scene_appearance_fields": (
                asdict(scene_appearance) if scene_appearance is not None else None
            ),
            "camera_base": camera_base,
            "oracle": asdict(TaskOracleConfig()),
            "state_frame": "hardware (arm degrees, gripper position 0-100)",
            "action_frame": "hardware (arm degrees, gripper position 0-100)",
        },
        "device": str(device) if device is not None else None,
        "code": git_provenance(REPOSITORY_ROOT),
        "package_versions": package_versions(
            ["gymnasium", "mujoco", "numpy"]
            + (
                ["lerobot", "torch"]
                if args.controller == "lerobot"
                else ["torch"]
                if args.controller == "flow-image"
                else []
            )
        ),
        "videos_saved": args.save_videos,
        "rollouts_saved": args.save_rollouts,
    }
    summary = write_evaluation_artifacts(args.output, run, results)
    print(
        f"Wrote {args.output}: {summary['success_count']}/{summary['episode_count']} "
        f"successes ({summary['success_rate']:.1%})."
    )


if __name__ == "__main__":
    main()
