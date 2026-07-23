#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a learned or scripted controller on a frozen simulator manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import os
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.policy import (
    DEFAULT_IMAGE_HW,
    DEFAULT_INSTRUCTION,
    resolve_checkpoint_cameras,
    select_device,
)
from pick_and_place.dppo_policy import DppoPolicyController
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
from pick_and_place.policy_sim import build_policy_sim_model
from pick_and_place.overhead_localization import OverheadLocalizer
from pick_and_place.scripted_policy import (
    AsyncWristLocalization,
    ScriptedPolicy,
    WristCameraLocalizer,
)
from pick_and_place.cube_detection import CubeTracker
from pick_and_place.detector_process import DetectorProcess
from pick_and_place.workspace_overlays import workspace_interior_corners_world

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "evaluation" / "smoke_v1.json"
DEFAULT_DPPO_CONFIG = (
    REPOSITORY_ROOT / "config" / "diffusion_policy" / "pretrain_so101_mlp_img.yaml"
)
SCRIPTED_IMAGE_HW = DEFAULT_IMAGE_HW


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=("lerobot", "scripted", "dppo"),
        default="lerobot",
        help="controller implementation (default: lerobot)",
    )
    parser.add_argument(
        "--checkpoint",
        help="local LeRobot checkpoint or repository ID, or a DPPO state_*.pt file",
    )
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
        "--dppo-python",
        type=Path,
        default=os.environ.get("DPPO_PYTHON"),
        help="interpreter of the DPPO virtual environment (default: $DPPO_PYTHON)",
    )
    parser.add_argument(
        "--dppo-config",
        type=Path,
        default=DEFAULT_DPPO_CONFIG,
        help=f"DPPO training configuration YAML (default: {DEFAULT_DPPO_CONFIG})",
    )
    parser.add_argument(
        "--dppo-normalization",
        type=Path,
        help="normalization.npz written by the Diffusion Policy dataset export",
    )
    parser.add_argument(
        "--dppo-act-steps",
        type=int,
        default=None,
        help="executed actions per policy query (default: the full prediction horizon)",
    )
    parser.add_argument(
        "--dppo-seed",
        type=int,
        default=0,
        help="Torch seed for DDPM action sampling (default: 0)",
    )
    parser.add_argument(
        "--dppo-ddim-steps",
        type=int,
        default=None,
        help=(
            "sample with DDIM using this many steps instead of the trained DDPM "
            "schedule; faster but not the training sampler, so not for headline runs"
        ),
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
    if args.controller in ("lerobot", "dppo") and args.checkpoint is None:
        parser.error(f"--checkpoint is required for the {args.controller} controller")
    if args.controller == "scripted" and args.checkpoint is not None:
        parser.error("--checkpoint does not apply to the scripted controller")
    if args.controller == "dppo":
        if args.dppo_python is None:
            parser.error("--dppo-python (or $DPPO_PYTHON) is required for the dppo controller")
        if args.dppo_normalization is None:
            parser.error("--dppo-normalization is required for the dppo controller")
        for name in ("checkpoint", "dppo_config", "dppo_normalization", "dppo_python"):
            path = Path(getattr(args, name))
            if not path.exists():
                parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
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


def _dppo_metadata(controller: DppoPolicyController, args: argparse.Namespace) -> dict:
    return {
        "type": "dppo-diffusion",
        "image_features": {
            "overhead": OVERHEAD_FEATURE,
            "wrist": WRIST_FEATURE,
        },
        "action_horizon": controller.horizon_steps,
        "executed_action_steps": controller.act_steps,
        "denoising_steps": controller.handshake["denoising_steps"],
        "sampler": controller.handshake["sampler"],
        "checkpoint_epoch": controller.handshake["epoch"],
        "weights": "ema",
        "image_augmentation": False,
        "sampling_seed": controller.handshake["seed"],
        "server_device": controller.handshake["device"],
        "server_torch_version": controller.handshake["torch_version"],
        "config": {
            "path": str(args.dppo_config.resolve()),
            "sha256": _sha256_of_file(args.dppo_config),
        },
        "normalization": {
            "path": str(args.dppo_normalization.resolve()),
            "sha256": _sha256_of_file(args.dppo_normalization),
        },
    }


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
    overhead_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera"
    )
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
    controller = ScriptedPolicy(
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
    scenarios = manifest.scenarios[: args.limit] if args.limit is not None else manifest.scenarios
    override_hw = (
        (args.image_height, args.image_width) if args.image_height is not None else None
    )
    if args.controller == "lerobot":
        image_hw, _ = resolve_checkpoint_cameras(args.checkpoint, override_hw=override_hw)
    elif args.controller == "dppo":
        print(f"Starting the DPPO policy server for {args.checkpoint}...")
        dppo_controller = DppoPolicyController.launch(
            python=args.dppo_python,
            checkpoint=args.checkpoint,
            config=args.dppo_config,
            normalization=args.dppo_normalization,
            device=args.device,
            seed=args.dppo_seed,
            act_steps=args.dppo_act_steps,
            ddim_steps=args.dppo_ddim_steps,
        )
        if override_hw is not None and override_hw != dppo_controller.image_hw:
            raise ValueError(
                f"--image-height/--image-width {override_hw} do not match the "
                f"model's trained image size {dppo_controller.image_hw}"
            )
        image_hw = dppo_controller.image_hw
    else:
        image_hw = override_hw or SCRIPTED_IMAGE_HW
    if args.render_height < image_hw[0] or args.render_width < image_hw[1]:
        raise ValueError("render dimensions must be at least the controller image dimensions")

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
        )
        controller_metadata = _lerobot_metadata(controller)
    elif args.controller == "dppo":
        device = dppo_controller.handshake["device"]
        controller = dppo_controller
        controller_metadata = _dppo_metadata(dppo_controller, args)
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
            failure_detail = (
                f", controller_failure={result.controller_failure['code']}"
                if result.controller_failure is not None
                else ""
            )
            print(
                f"[{index:02d}/{len(scenarios):02d}] {scenario.scenario_id}: {status}, "
                f"steps={result.control_steps}, final_xy={result.final_xy_error_m * 100:.1f} cm"
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
            "domain_randomization_presets": sorted({
                scenario.domain_randomization_preset or "none" for scenario in scenarios
            }),
            "oracle": asdict(TaskOracleConfig()),
            "state_frame": "hardware (arm degrees, gripper position 0-100)",
            "action_frame": "hardware (arm degrees, gripper position 0-100)",
        },
        "device": str(device) if device is not None else None,
        "code": git_provenance(REPOSITORY_ROOT),
        "package_versions": package_versions(
            ["gymnasium", "mujoco", "numpy"]
            + (["lerobot", "torch"] if args.controller == "lerobot" else [])
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
