# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Score a controller against a frozen scenario manifest.

The evaluation as a function, so it can be called rather than shelled out to:
:class:`EvaluationRun` says what a scored run is, and :func:`run_evaluation`
performs one and returns its summary. ``scripts/eval_policy_sim.py`` is a
parser and two lines on top of this.

Splitting it this way is what makes the run testable at all. Everything here
used to live inside one ``main``, where the only way to exercise it was to
launch a process and read the files it left behind.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import torch

from pick_and_place.cli.eval_policy_sim import REPOSITORY_ROOT
from pick_and_place.policies.policy import (
    DEFAULT_IMAGE_HW,
    resolve_checkpoint_cameras,
    select_device,
)
from pick_and_place.policies.flow_image_policy import FlowImagePolicyController
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
from pick_and_place.spec.controller import OVERHEAD_FEATURE, WRIST_FEATURE
from pick_and_place.variants.appearance import parse_appearance

SCRIPTED_IMAGE_HW = DEFAULT_IMAGE_HW


@dataclass(frozen=True)
class EvaluationRun:
    """What a scored run is: a controller, a world, and which scenarios to take.

    A leaf declares only the flags its controller understands, so the fields
    belonging to the other two arrive as ``None``. That is the same thing the
    parser expresses, carried in one typed object instead of a namespace whose
    contents depend on which subcommand was used.
    """

    controller: str
    output: Path
    render_height: int
    render_width: int
    manifest: Path | None = None
    image_height: int | None = None
    image_width: int | None = None
    scene_appearance: str | None = None
    limit: int | None = None
    offset: int = 0
    max_episode_seconds: float | None = None
    save_videos: bool = False
    save_rollouts: bool = False
    # The lerobot leaf.
    checkpoint: str | None = None
    device: str | None = None
    instruction: str | None = None
    base_checkpoint: str | None = None
    n_action_steps: int | None = None
    temporal_ensemble_coeff: float | None = None
    # The flow-image leaf.
    flow_export: Path | None = None
    flow_act_steps: int | None = None
    flow_integration_steps: int | None = None
    flow_seed: int | None = None
    flow_noise_correlation: float | None = None
    # The scripted leaf.
    scripted_perception: str | None = None

    @classmethod
    def from_args(cls, args: Any) -> "EvaluationRun":
        """Build a run from a parsed namespace, defaulting absent leaf flags to None."""
        return cls(**{field.name: getattr(args, field.name, None) for field in fields(cls)})

    @property
    def override_hw(self) -> tuple[int, int] | None:
        return (self.image_height, self.image_width) if self.image_height is not None else None

    @property
    def render_hw(self) -> tuple[int, int]:
        return (self.render_height, self.render_width)


class EpisodeVideoWriters:
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


class EpisodeRolloutWriter:
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


def _flow_image_metadata(controller: FlowImagePolicyController, cfg: EvaluationRun) -> dict:
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    export = Path(cfg.flow_export)
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


def run_evaluation(cfg: EvaluationRun, *, report: Callable[[str], None] = print) -> dict:
    """Score ``cfg``'s controller over the manifest's scenarios, and write the run.

    Returns the summary :func:`write_evaluation_artifacts` produced. ``report``
    takes the progress lines, so a caller that is not a terminal can silence
    them without the evaluation behaving differently.
    """
    started_at = dt.datetime.now(dt.UTC)
    manifest = ScenarioManifest.load(cfg.manifest)
    if cfg.offset >= len(manifest.scenarios):
        raise SystemExit(
            f"--offset {cfg.offset} skips the whole {len(manifest.scenarios)}-scenario suite"
        )
    selected = manifest.scenarios[cfg.offset :]
    scenarios = selected[: cfg.limit] if cfg.limit is not None else selected
    override_hw = cfg.override_hw
    appearance_name, scene_appearance = (
        parse_appearance(cfg.scene_appearance) if cfg.scene_appearance is not None else (None, None)
    )
    if cfg.controller == "lerobot":
        image_hw, _ = resolve_checkpoint_cameras(cfg.checkpoint, override_hw=override_hw)
    elif cfg.controller == "flow-image":
        flow_device = select_device(cfg.device)
        report(f"Loading {cfg.checkpoint} on {flow_device}...")
        flow_image_controller = FlowImagePolicyController.from_export(
            cfg.checkpoint,
            cfg.flow_export,
            act_steps=cfg.flow_act_steps,
            integration_steps=cfg.flow_integration_steps,
            device=flow_device,
            seed=cfg.flow_seed,
            noise_correlation=cfg.flow_noise_correlation,
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
    if cfg.render_height < image_hw[0] or cfg.render_width < image_hw[1]:
        raise ValueError("render dimensions must be at least the controller image dimensions")

    if cfg.max_episode_seconds is not None:
        scenarios = tuple(
            replace(
                scenario,
                max_steps=min(
                    scenario.max_steps,
                    max(1, round(cfg.max_episode_seconds * scenario.control_hz)),
                ),
            )
            for scenario in scenarios
        )

    env = PolicySimEnv(
        image_hw=image_hw,
        render_hw=cfg.render_hw,
        scene_appearance=scene_appearance,
    )

    control_hz_values = {scenario.control_hz for scenario in scenarios}
    if cfg.controller == "scripted" and len(control_hz_values) != 1:
        raise ValueError("scripted evaluation requires one control frequency per run")
    if cfg.controller == "lerobot":
        device = select_device(cfg.device)
        report(f"Loading {cfg.checkpoint} on {device}...")
        controller = LeRobotPolicyController.from_checkpoint(
            cfg.checkpoint,
            device=device,
            image_hw=image_hw,
            instruction=cfg.instruction,
            n_action_steps=cfg.n_action_steps,
            temporal_ensemble_coeff=cfg.temporal_ensemble_coeff,
            base_checkpoint=cfg.base_checkpoint,
        )
        controller_metadata = _lerobot_metadata(controller)
    elif cfg.controller == "flow-image":
        device = flow_device
        controller = flow_image_controller
        controller_metadata = _flow_image_metadata(flow_image_controller, cfg)
    else:
        device = None
        controller, controller_metadata = sim_scripted_controller(
            image_hw=image_hw,
            render_hw=cfg.render_hw,
            control_hz=next(iter(control_hz_values)),
            scene_model=env.model,
            scene_data=env.data,
            perception=cfg.scripted_perception,
            cube_belief_error=lambda: env.cube_belief_error,
        )
    report(
        f"Evaluating {len(scenarios)}/{len(manifest.scenarios)} {manifest.suite!r} scenarios "
        f"with {cfg.controller} "
        f"at {image_hw[1]}x{image_hw[0]} and {next(iter(control_hz_values)):g} Hz, "
        f"scene appearance {cfg.scene_appearance or 'as-compiled'}."
    )

    camera_base = _camera_base_metadata(env.model)
    results = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            writers = []
            if cfg.save_videos:
                writers.append(EpisodeVideoWriters(
                    cfg.output / "videos",
                    scenario.scenario_id,
                    scenario.control_hz,
                ))
            if cfg.save_rollouts:
                writers.append(EpisodeRolloutWriter(
                    cfg.output / "rollouts",
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
            report(
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
        # A leaf that takes no checkpoint leaves the field None, so absent and
        # unset are the same thing here and both mean "not a checkpoint run".
        "checkpoint": (
            {
                "path_or_repository_id": cfg.checkpoint,
                "fingerprint": fingerprint_checkpoint(cfg.checkpoint),
            }
            if cfg.checkpoint is not None
            else None
        ),
        "controller": controller_metadata,
        "instruction": cfg.instruction,
        "scenario_manifest": {
            "path": str(cfg.manifest.resolve()),
            "sha256": manifest.sha256(),
            "suite": manifest.suite,
            "selected_scenario_ids": [scenario.scenario_id for scenario in scenarios],
            "complete_suite": len(scenarios) == len(manifest.scenarios),
        },
        "environment": {
            "image_height": image_hw[0],
            "image_width": image_hw[1],
            "render_height": cfg.render_height,
            "render_width": cfg.render_width,
            "control_hz": sorted({scenario.control_hz for scenario in scenarios}),
            "episode_step_limits": sorted({scenario.max_steps for scenario in scenarios}),
            "requested_max_episode_seconds": cfg.max_episode_seconds,
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
                if cfg.controller == "lerobot"
                else ["torch"]
                if cfg.controller == "flow-image"
                else []
            )
        ),
        "videos_saved": cfg.save_videos,
        "rollouts_saved": cfg.save_rollouts,
    }
    return write_evaluation_artifacts(cfg.output, run, results)
