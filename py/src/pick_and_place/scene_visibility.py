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

import cv2
import mujoco
import numpy as np

from pick_and_place.episodes import build_geom_sets
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    PAPER_TARGET_MARKER_NAME,
    place_paper_target_marker,
)
from pick_and_place.policy_sim import (
    build_policy_sim_model,
    joint_qpos_addresses,
    real_action_to_sim_ctrl,
)
from pick_and_place.sim_recorder import resize_and_center_crop
from pick_and_place.workspace_overlays import is_cube_drop_allowed

# Thresholds on the fractional pixel coverage after the exact 96x96 transform:
# >= 0.5 counts as an object pixel, > 0.1 is excluded from background rings so
# mixed border pixels do not dilute contrast or leak into inpainting fills.
OBJECT_COVERAGE = 0.5
RING_EXCLUSION_COVERAGE = 0.1
RING_DILATION_PX = 5


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
        self.robot_geom_ids, _ = build_geom_sets(self.model)

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

    def coverage_maps(self, camera: str) -> dict[str, np.ndarray]:
        """Fractional cube/plate pixel coverage after the exact 96x96 transform."""
        self.seg_renderer.update_scene(self.data, camera=camera)
        seg = self.seg_renderer.render()
        geom_ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        return {
            name: resize_and_center_crop(
                (geom_ids == geom_id).astype(np.float32), self.image_size, self.image_size
            )
            for name, geom_id in (("cube", self.cube_geom_id), ("plate", self.plate_geom_id))
        }

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
