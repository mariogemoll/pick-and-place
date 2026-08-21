# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Wire the expert to a simulated scene, the way a deployed run wires it to a rig.

The expert is drivable from images and reported joints alone, which is what makes
it comparable to a learned policy. Everything it needs beyond that — camera
intrinsics, where the overhead camera sits, which localizer answers "where is the
cube" — is configuration, and in sim it has to be *built* rather than measured.
That is what this does, once, for every sim entry point that runs the expert.

Two models are in play and they are not the same model. The controller owns a
nominal one, compiled here at the authored camera poses, and solves through its
intrinsics; the scene passed in is the live one the episode actually runs in,
which domain randomization may have displaced. Handing the controller the live
scene's cameras would quietly delete the calibration error the sim exists to
reproduce.

``perception`` picks how the overhead answer is obtained: ``geometric`` reads the
simulator's own pose behind a visibility gate, ``detector`` runs the real optical
pipeline. The wrist always runs the detector, since the descent servo is what a
miscalibrated rig is tested against.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.core.workspace_bounds import workspace_interior_corners_world
from pick_and_place.perception.cube_detection import CubeTracker
from pick_and_place.perception.detector_process import DetectorProcess
from pick_and_place.perception.overhead_localization import OverheadLocalizer
from pick_and_place.plant.geometric_overhead import SimGeometricOverheadLocalizer
from pick_and_place.plant.wrist_localizer import AsyncWristLocalization, WristCameraLocalizer
from pick_and_place.rollout.scripted import scripted_policy
from pick_and_place.runtime.policy_sim import build_policy_sim_model
from pick_and_place.scripted.policy import ScriptedPolicy
from pick_and_place.spec.controller import OVERHEAD_FEATURE, WRIST_FEATURE

SCRIPTED_PERCEPTION_MODES = ("geometric", "detector")

#: The overhead segmentation pass only decides visibility, so it renders small.
_SEGMENTATION_WIDTH = 320
_SEGMENTATION_HEIGHT = 240


def camera_matrix_for_output(
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


def sim_scripted_controller(
    *,
    image_hw: tuple[int, int],
    render_hw: tuple[int, int],
    control_hz: float,
    scene_model: mujoco.MjModel,
    scene_data: mujoco.MjData,
    perception: str = "geometric",
    cube_belief_error: Callable[[], tuple[float, float, float, float]] | None = None,
    **kwargs: Any,
) -> tuple[ScriptedPolicy, dict]:
    """Build the expert against ``scene_model``/``scene_data``, and describe it.

    The returned metadata is what a scored run records about the controller it
    ran, so it names the perception mode and the nominal calibration the
    controller solved through rather than the scene's own.
    """
    if perception not in SCRIPTED_PERCEPTION_MODES:
        raise ValueError(
            f"perception must be one of {SCRIPTED_PERCEPTION_MODES}, got {perception!r}"
        )

    model, data = build_policy_sim_model(*render_hw)
    mujoco.mj_forward(model, data)
    camera_matrices = {
        name: camera_matrix_for_output(
            model, name, render_hw=render_hw, image_hw=image_hw
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
    overhead_detector = DetectorProcess(nthreads=1) if perception == "detector" else None
    wrist_detector = DetectorProcess(nthreads=1)
    # Both localizers rebuild their detector every episode via reset(). Hand each
    # one the single persistent handle so no child process leaks per episode --
    # a DetectorProcess holds no per-frame state and is built for reuse across a
    # long run.
    overhead_localizer = (
        SimGeometricOverheadLocalizer(
            scene_model,
            scene_data,
            width=min(_SEGMENTATION_WIDTH, render_hw[1]),
            height=min(_SEGMENTATION_HEIGHT, render_hw[0]),
            cube_belief_error=cube_belief_error,
        )
        if perception == "geometric"
        else OverheadLocalizer(
            camera_matrices["overhead_camera"],
            overhead_position,
            overhead_rotation,
            detector_factory=lambda: overhead_detector,
        )
    )
    controller = scripted_policy(
        overhead_localizer,
        workspace_corners,
        control_hz=control_hz,
        wrist_localizer=AsyncWristLocalization(
            WristCameraLocalizer(
                model,
                camera_matrices["wrist_camera"],
                tracker_factory=lambda: CubeTracker(smooth=0.95, detector=wrist_detector),
            )
        ),
        **kwargs,
    )
    # ScriptedPolicy.close() only reaches the async wrist wrapper, so tear the two
    # detector children down alongside it.
    _controller_close = controller.close

    def _close_with_detectors() -> None:
        try:
            _controller_close()
        finally:
            if overhead_detector is not None:
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
        "overhead_perception": perception,
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
