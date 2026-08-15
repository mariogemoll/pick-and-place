# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Re-render recorded scene states and measure object visibility at 96x96.

Reconstructs one recorded control tick in the calibrated visual-policy scene —
robot at the recorded hardware-frame joints, cube at its recorded true pose,
target plate at its recorded pose and yaw — and renders RGB and segmentation
through the same camera pipeline and aspect-fill/center-crop transform as
training images. Coverage maps are fractional per-pixel object areas after
that exact transform, so they align with the images the policy actually sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pick_and_place.data.trajectory_artifact import ARTIFACT_FILENAME, load_trajectory
from pick_and_place.spec.workspace import DROP_ZONE_HALF_SIZE
from pick_and_place.sim.paper_target_marker import PAPER_TARGET_MARKER_NAME, place_paper_target_marker
from pick_and_place.runtime.policy_sim import (
    build_policy_sim_model,
    joint_qpos_addresses,
    real_action_to_sim_ctrl,
)
from pick_and_place.core.image_ops import resize_and_center_crop
from pick_and_place.core.task_phases import coarse_phase_labels
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed

# Thresholds on the fractional pixel coverage after the exact 96x96 transform:
# >= 0.5 counts as an object pixel, > 0.1 is excluded from background rings so
# mixed border pixels do not dilute contrast or leak into inpainting fills.
OBJECT_COVERAGE = 0.5
RING_EXCLUSION_COVERAGE = 0.1
RING_DILATION_PX = 5


@dataclass(frozen=True)
class EpisodeTruth:
    """Per-frame ground truth needed to re-render a recorded episode."""

    name: str
    states: np.ndarray  # (N, 6) hardware-frame true arm joints
    cube_poses: np.ndarray  # (N, 7) true cube position + wxyz quaternion
    target_xy: tuple[float, float]
    target_plate_yaw: float
    coarse_phases: np.ndarray  # (N,) coarse phase name per 30 Hz frame


def load_episode_truth(root: Path) -> EpisodeTruth:
    """Load one staged episode's per-frame ground truth and phase labels.

    Read from the episode's trajectory artifact, which is the only place the
    *true* arm pose is stored. The dataset's ``observation.state`` is the
    servo-style readback, so under a miscalibration draw it is a different pose —
    put the arm there and it lands in the wrong place in the picture.
    """
    artifact = load_trajectory(root / ARTIFACT_FILENAME)
    return EpisodeTruth(
        name=root.name,
        states=artifact.frames.true_state,
        cube_poses=artifact.frames.true_cube_pose.astype(np.float64),
        target_xy=artifact.facts.target_xy,
        target_plate_yaw=artifact.facts.target_plate_yaw,
        coarse_phases=coarse_phase_labels(artifact.facts.phase_spans, len(artifact.frames)),
    )


def video_render_hw(root: Path) -> tuple[int, int]:
    """Return the (height, width) the episode's camera videos were rendered at."""
    with (root / "meta" / "info.json").open() as file:
        info = json.load(file)
    shapes = {
        tuple(feature["shape"][:2])
        for feature in info["features"].values()
        if feature.get("dtype") == "video"
    }
    if len(shapes) != 1:
        raise ValueError(f"{root.name} cameras must share one resolution, got {shapes}")
    return next(iter(shapes))


class SceneMeasurer:
    """Re-render recorded scene states and measure object visibility."""

    def __init__(self, render_hw: tuple[int, int], image_size: int) -> None:
        height, width = render_hw
        self.image_size = image_size
        self.model, self.data = build_policy_sim_model(height, width)
        self.rgb_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.seg_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.seg_renderer.enable_segmentation_rendering()
        self._joint_qpos_adr = joint_qpos_addresses(self.model)
        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[cube_body]])
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")
        self.plate_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, PAPER_TARGET_MARKER_NAME + "_geom"
        )
        if min(self.cube_geom_id, self.plate_geom_id) < 0:
            raise ValueError("scene is missing the pick_cube or target plate geom")
        self.robot_geom_ids = self._subtree_geom_ids("base")
        # The workspace frame's AprilTag calibration plates: white tagged squares
        # at roughly the cube's apparent scale, and the scene's nearest visual
        # confuser for it. They exist only to calibrate the overhead camera.
        self.frame_tag_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
                "workspace_frame_tag_"
            )
        }

    def _subtree_geom_ids(self, root_body: str) -> set[int]:
        """Geom ids on the body named ``root_body`` or any of its descendants.

        Walking the body tree is what distinguishes the arm from the workspace
        frame: both hang off the worldbody, so a "not worldbody" test would sweep
        the frame in with the robot.
        """
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_body)
        if root < 0:
            raise ValueError(f"scene is missing the {root_body!r} body")
        geom_ids: set[int] = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id == root:
                    geom_ids.add(geom_id)
                    break
                body_id = int(self.model.body_parentid[body_id])
        return geom_ids

    def set_target_plate(self, target_xy: tuple[float, float], yaw: float) -> None:
        place_paper_target_marker(
            self.model,
            target_xy,
            yaw,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*target_xy),
            alpha=1.0,
        )

    def set_frame(self, state_real: np.ndarray, cube_pose: np.ndarray) -> None:
        self.data.qpos[self._joint_qpos_adr] = real_action_to_sim_ctrl(state_real)
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_pose
        mujoco.mj_forward(self.model, self.data)

    def coverage_maps(
        self, camera: str, *, include_robot: bool = False, include_frame_tags: bool = False
    ) -> dict[str, np.ndarray]:
        """Fractional cube/plate pixel coverage after the exact 96x96 transform.

        With ``include_robot`` the result also carries a ``robot`` map covering
        every arm geom, for measuring how distinguishable the cube is from the
        arm that reaches for it; ``include_frame_tags`` adds a ``frame_tags`` map
        over the workspace frame's calibration plates. Segmentation ignores
        transparency, so the ``frame_tags`` map still covers the plates' footprint
        when they have been made invisible — the RGB there is then whatever the
        frame shows underneath, which is what a peeled-off sticker looks like.
        """
        self.seg_renderer.update_scene(self.data, camera=camera)
        seg = self.seg_renderer.render()
        geom_ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        maps = {
            name: resize_and_center_crop(
                (geom_ids == geom_id).astype(np.float32), self.image_size, self.image_size
            )
            for name, geom_id in (("cube", self.cube_geom_id), ("plate", self.plate_geom_id))
        }
        for enabled, name, ids in (
            (include_robot, "robot", self.robot_geom_ids),
            (include_frame_tags, "frame_tags", self.frame_tag_geom_ids),
        ):
            if enabled:
                maps[name] = resize_and_center_crop(
                    np.isin(geom_ids, list(ids)).astype(np.float32),
                    self.image_size,
                    self.image_size,
                )
        return maps

    def render(self, camera: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return the transformed RGB image and per-object coverage maps."""
        self.rgb_renderer.update_scene(self.data, camera=camera)
        rgb = resize_and_center_crop(self.rgb_renderer.render(), self.image_size, self.image_size)
        return rgb, self.coverage_maps(camera)

    def close(self) -> None:
        self.rgb_renderer.close()
        self.seg_renderer.close()


def contrast(rgb: np.ndarray, coverage: np.ndarray, other_coverage: np.ndarray) -> float | None:
    """Michelson luminance contrast of an object against its background ring."""
    luminance = rgb.astype(np.float64) @ (0.2126, 0.7152, 0.0722)
    object_mask = coverage >= OBJECT_COVERAGE
    if not object_mask.any():
        return None
    kernel = np.ones((2 * RING_DILATION_PX + 1,) * 2, np.uint8)
    dilated = cv2.dilate(object_mask.astype(np.uint8), kernel).astype(bool)
    ring = (
        dilated
        & (coverage <= RING_EXCLUSION_COVERAGE)
        & (other_coverage <= RING_EXCLUSION_COVERAGE)
    )
    if not ring.any():
        return None
    object_level = float(luminance[object_mask].mean())
    ring_level = float(luminance[ring].mean())
    if object_level + ring_level <= 0.0:
        return None
    return abs(object_level - ring_level) / (object_level + ring_level)


def inpaint_object(image: np.ndarray, coverage: np.ndarray) -> tuple[np.ndarray, bool]:
    """Replace an object's pixels with the median color of its background ring.

    Returns the (possibly unchanged) image and whether anything was masked.
    """
    mask = (coverage > RING_EXCLUSION_COVERAGE).astype(np.uint8)
    if not mask.any():
        return image, False
    grown = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
    ring_kernel = np.ones((2 * RING_DILATION_PX + 1,) * 2, np.uint8)
    ring = cv2.dilate(mask, ring_kernel).astype(bool) & ~grown
    if not ring.any():
        return image, False
    fill = np.median(image[ring], axis=0).astype(image.dtype)
    result = image.copy()
    result[grown] = fill
    return result, True
