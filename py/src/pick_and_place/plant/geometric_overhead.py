# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Detector-free overhead observations for scripted simulation.

The scripted policy consumes the same localizer contract in simulation and on
the rig.  The rig implements it with optical detection.  Dataset generation
does not need to rediscover poses the simulator already owns, but it must still
make the policy search when the arm hides an object.  This adapter therefore
uses segmentation only for a visibility gate and returns controlled pose
beliefs once the gate opens.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from pick_and_place.core.camera_projection import project_to_pixel
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.image_ops import resize_and_center_crop
from pick_and_place.plant.overhead import OVERHEAD_CAMERA, camera_matrix_for
from pick_and_place.spec.drop_zone import PaperTarget
from pick_and_place.spec.workspace import DROP_ZONE_HALF_SIZE
from pick_and_place.sim.model import get_cube_pose

DEFAULT_CUBE_VISIBILITY_FRACTION = 0.80
DEFAULT_PLATE_VISIBILITY_FRACTION = 0.20


class SimGeometricOverheadLocalizer:
    """Return simulator-owned poses only when enough object pixels are visible."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        width: int,
        height: int,
        cube_visibility_fraction: float = DEFAULT_CUBE_VISIBILITY_FRACTION,
        plate_visibility_fraction: float = DEFAULT_PLATE_VISIBILITY_FRACTION,
        cube_belief_error: Callable[[], tuple[float, float, float, float]] | None = None,
    ) -> None:
        for name, value in (
            ("cube_visibility_fraction", cube_visibility_fraction),
            ("plate_visibility_fraction", plate_visibility_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self.model = model
        self.data = data
        self.width = int(width)
        self.height = int(height)
        self.visibility_fraction_thresholds = {
            "cube": float(cube_visibility_fraction),
            "plate": float(plate_visibility_fraction),
        }
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, OVERHEAD_CAMERA)
        self.cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")
        self.plate_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "paper_target_marker_geom"
        )
        if min(self.camera_id, self.cube_geom_id, self.plate_geom_id) < 0:
            raise ValueError("scene must contain the overhead camera, cube, and plate")
        self.robot_geom_ids = self._subtree_geom_ids("base")
        self._renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        self._renderer.enable_segmentation_rendering()
        self._cube_error = np.zeros(4, dtype=float)
        self._cube_belief_error = cube_belief_error

    def _subtree_geom_ids(self, root_body: str) -> np.ndarray:
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_body)
        if root < 0:
            raise ValueError(f"scene is missing body {root_body!r}")
        result = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id == root:
                    result.append(geom_id)
                    break
                body_id = int(self.model.body_parentid[body_id])
        return np.asarray(result, dtype=np.int32)

    def reset(self) -> None:
        """Match the optical localizer contract; geometric observations are stateless."""

    def set_cube_belief_error(self, error: tuple[float, float, float, float]) -> None:
        values = np.asarray(error, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("cube belief error must contain four finite values")
        self._cube_error = values.copy()

    def _mask(self, geom_id: int, *, hide_robot: bool) -> np.ndarray:
        original_groups = None
        if hide_robot:
            original_groups = self.model.geom_group[self.robot_geom_ids].copy()
            self.model.geom_group[self.robot_geom_ids] = 5
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        if hide_robot:
            option.geomgroup[5] = 0
        try:
            self._renderer.update_scene(
                self.data,
                camera=self.camera_id,
                scene_option=option,
            )
        finally:
            if original_groups is not None:
                self.model.geom_group[self.robot_geom_ids] = original_groups
        segmentation = self._renderer.render()
        ids = np.where(
            segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
            segmentation[..., 0],
            -1,
        )
        return resize_and_center_crop((ids == geom_id).astype(np.float32), self.height, self.width)

    def visibility_fraction(self, object_name: str) -> float:
        """Visible pixels divided by the object's robot-unoccluded pixels."""
        geom_id = {"cube": self.cube_geom_id, "plate": self.plate_geom_id}[object_name]
        visible = float(self._mask(geom_id, hide_robot=False).sum())
        reference = float(self._mask(geom_id, hide_robot=True).sum())
        return 0.0 if reference <= 0.0 else min(1.0, visible / reference)

    def _visible(self, object_name: str) -> bool:
        return (
            self.visibility_fraction(object_name)
            >= self.visibility_fraction_thresholds[object_name]
        )

    def localize_cube(self, frame_rgb: np.ndarray, *, free_grasp: bool = False) -> CubePose | None:
        del frame_rgb, free_grasp
        if not self._visible("cube"):
            return None
        true_pose = get_cube_pose(self.model, self.data)
        error = (
            np.asarray(self._cube_belief_error(), dtype=float)
            if self._cube_belief_error is not None
            else self._cube_error
        )
        dx, dy, dz, dyaw = error
        return CubePose(
            x=true_pose.x + float(dx),
            y=true_pose.y + float(dy),
            z=true_pose.z + float(dz),
            roll=true_pose.roll,
            pitch=true_pose.pitch,
            yaw=true_pose.yaw + float(dyaw),
        )

    def localize_drop_target(
        self,
        frame_rgb: np.ndarray,
        *,
        target_color: str,
        workspace_corners_world: np.ndarray,
    ) -> PaperTarget | None:
        del frame_rgb, target_color, workspace_corners_world
        if not self._visible("plate"):
            return None
        center = self.data.geom_xpos[self.plate_geom_id].copy()
        rotation = self.data.geom_xmat[self.plate_geom_id].reshape(3, 3)
        local = np.array(
            [
                [-DROP_ZONE_HALF_SIZE, -DROP_ZONE_HALF_SIZE, 0.0],
                [DROP_ZONE_HALF_SIZE, -DROP_ZONE_HALF_SIZE, 0.0],
                [DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE, 0.0],
                [-DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE, 0.0],
            ]
        )
        corners_world = local @ rotation.T + center
        camera_position = self.data.cam_xpos[self.camera_id]
        camera_rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        corners_px = project_to_pixel(
            corners_world,
            camera_matrix_for(self.model, self.camera_id, self.width, self.height),
            camera_position,
            camera_rotation,
        )
        return PaperTarget(
            center_px=corners_px.mean(axis=0),
            corners_px=corners_px,
            center_world=center,
            corners_world=corners_world,
            area_px=float(
                abs(
                    np.linalg.det(
                        np.stack((corners_px[1] - corners_px[0], corners_px[2] - corners_px[1]))
                    )
                )
            ),
            rectangularity=1.0,
        )

    def close(self) -> None:
        self._renderer.close()
