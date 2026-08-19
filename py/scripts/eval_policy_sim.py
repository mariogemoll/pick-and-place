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
    build_policy_sim_model,
    evaluate_policy_episode,
    PolicySimEnv,
)
from pick_and_place.perception.overhead_localization import OverheadLocalizer
from pick_and_place.plant.wrist_localizer import AsyncWristLocalization, WristCameraLocalizer
from pick_and_place.rollout.scripted import scripted_policy
from pick_and_place.scripted.policy import ScriptedPolicy
from pick_and_place.perception.cube_detection import CubeTracker
from pick_and_place.perception.detector_process import DetectorProcess
from pick_and_place.cli.policy import (
    add_flow_image_arguments,
    add_policy_arguments,
)
from pick_and_place.cli.scene import add_render_size_arguments, add_scene_appearance_arguments
from pick_and_place.variants.appearance import parse_appearance
from pick_and_place.core.workspace_bounds import workspace_interior_corners_world

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "evaluation" / "smoke_v1.json"
SCRIPTED_IMAGE_HW = DEFAULT_IMAGE_HW


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_arguments(
        parser,
        controllers=("lerobot", "scripted", "flow-image"),
        checkpoint_default=None,
        n_action_steps_default=None,
    )
    add_flow_image_arguments(parser)
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
    if args.controller in ("lerobot", "flow-image") and args.checkpoint is None:
        parser.error(f"--checkpoint is required for the {args.controller} controller")
    if args.controller == "scripted" and args.checkpoint is not None:
        parser.error("--checkpoint does not apply to the scripted controller")
    if args.controller == "flow-image":
        if args.flow_export is None:
            parser.error("--flow-export is required for the flow-image controller")
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


def _camera_matrix_for_output(
    model: mujoco.MjModel,
    camera_name: str,
    *,
    render_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Return the intrinsics of a MuJoCo render after resize-and-center-crop."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"model has no camera named {camera_name!r}")
    render_height, render_width = render_hw
    image_height, image_width = image_hw
    scale = max(image_width / render_width, image_height / render_height)
    resized_width = max(image_width, round(render_width * scale))
    resized_height = max(image_height, round(render_height * scale))
    scale_x = resized_width / render_width
    scale_y = resized_height / render_height
    left = (resized_width - image_width) // 2
    top = (resized_height - image_height) // 2
    focal = (render_height / 2.0) / math.tan(math.radians(model.cam_fovy[camera_id]) / 2.0)
    return np.array(
        [
            [focal * scale_x, 0.0, render_width * scale_x / 2.0 - left],
            [0.0, focal * scale_y, render_height * scale_y / 2.0 - top],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_scripted_controller(
    *,
    image_hw: tuple[int, int],
    render_hw: tuple[int, int],
    control_hz: float,
) -> tuple[ScriptedPolicy, dict]:
    """Build the controller-owned nominal camera and kinematic models."""
    model, data = build_policy_sim_model(*render_hw)
    mujoco.mj_forward(model, data)
    camera_matrices = {
        name: _camera_matrix_for_output(
            model,
            name,
            render_hw=render_hw,
            image_hw=image_hw,
        )
        for name in ("overhead_camera", "wrist_camera")
    }
    overhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    overhead_position = data.cam_xpos[overhead_id].copy()
    overhead_rotation = data.cam_xmat[overhead_id].reshape(3, 3).copy()
    workspace_corners = workspace_interior_corners_world()
    # Run AprilTag detection out-of-process. The pupil_apriltags C destructor
    # (apriltag_detector_destroy) segfaults when a Detector is garbage-collected
    # inside a multiprocessing pool worker, killing the worker outright
    # (BrokenProcessPool) and taking every banked episode with it. DetectorProcess
    # is the same containment sim_recorder.py uses; detection is bit-identical
    # across the process boundary, so the controller behaves exactly as it does on
    # real hardware -- only the detector's address space changes.
    overhead_detector = DetectorProcess(nthreads=1)
    wrist_detector = DetectorProcess(nthreads=1)
    # Both localizers rebuild their detector every episode via reset(). Hand each
    # one the single persistent handle so no child process leaks per episode --
    # a DetectorProcess holds no per-frame state and is built for reuse across a
    # long run.
    controller = scripted_policy(
        OverheadLocalizer(
            camera_matrices["overhead_camera"],
            overhead_position,
            overhead_rotation,
            detector_factory=lambda: overhead_detector,
        ),
        workspace_corners,
        control_hz=control_hz,
        wrist_localizer=AsyncWristLocalization(
            WristCameraLocalizer(
                model,
                camera_matrices["wrist_camera"],
                tracker_factory=lambda: CubeTracker(smooth=0.95, detector=wrist_detector),
            )
        ),
    )
    # ScriptedPolicy.close() only reaches the async wrist wrapper, so tear the two
    # detector children down alongside it.
    _controller_close = controller.close

    def _close_with_detectors() -> None:
        try:
            _controller_close()
        finally:
            overhead_detector.close()
            wrist_detector.close()

    controller.close = _close_with_detectors  # type: ignore[method-assign]
    metadata = {
        "type": "scripted",
        "class": f"{type(controller).__module__}.{type(controller).__name__}",
        "image_features": {
            "overhead": OVERHEAD_FEATURE,
            "wrist": WRIST_FEATURE,
        },
        "control_hz": controller.control_hz,
        "wrist_localization": "asynchronous_latest_completed",
        "apriltag_detection": "out-of-process (DetectorProcess, nthreads=1)",
        "target_color": controller.target_color,
        "max_localization_steps": controller.max_localization_steps,
        "localization_steps_per_search": controller.localization_steps_per_search,
        "rng_seed": controller.rng_seed,
        "nominal_camera_calibration": {
            "overhead_camera": {
                "camera_matrix": camera_matrices["overhead_camera"].tolist(),
                "position_m": overhead_position.tolist(),
                "rotation_world_from_camera": overhead_rotation.tolist(),
            },
            "wrist_camera": {
                "camera_matrix": camera_matrices["wrist_camera"].tolist(),
                "kinematic_model": "controller-owned nominal MuJoCo model",
            },
        },
        "workspace_corners_world_m": workspace_corners.tolist(),
    }
    return controller, metadata


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
        controller, controller_metadata = _make_scripted_controller(
            image_hw=image_hw,
            render_hw=(args.render_height, args.render_width),
            control_hz=next(iter(control_hz_values)),
        )
    print(
        f"Evaluating {len(scenarios)}/{len(manifest.scenarios)} {manifest.suite!r} scenarios "
        f"with {args.controller} "
        f"at {image_hw[1]}x{image_hw[0]} and {next(iter(control_hz_values)):g} Hz, "
        f"scene appearance {args.scene_appearance or 'as-compiled'}."
    )

    appearance_name, scene_appearance = (
        parse_appearance(args.scene_appearance)
        if args.scene_appearance is not None
        else (None, None)
    )
    env = PolicySimEnv(
        image_hw=image_hw,
        render_hw=(args.render_height, args.render_width),
        scene_appearance=scene_appearance,
    )
    camera_base = _camera_base_metadata(env.model)
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

    run = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "checkpoint": (
            {
                "path_or_repository_id": args.checkpoint,
                "fingerprint": fingerprint_checkpoint(args.checkpoint),
            }
            if args.checkpoint is not None
            else None
        ),
        "controller": controller_metadata,
        "instruction": args.instruction if args.controller == "lerobot" else None,
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
    }
    summary = write_evaluation_artifacts(args.output, run, results)
    print(
        f"Wrote {args.output}: {summary['success_count']}/{summary['episode_count']} "
        f"successes ({summary['success_rate']:.1%})."
    )


if __name__ == "__main__":
    main()
