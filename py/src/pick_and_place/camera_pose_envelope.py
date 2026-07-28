# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Which overhead camera poses could the real rig actually be calibrated at?

``run_policy_real`` solves the overhead extrinsics from the four workspace-frame
AprilTags at startup, and re-solves them periodically to catch drift. A pose the
real camera cannot see all four tags from is therefore one the rig can never
run at -- so randomizing the sim camera over such a pose trains the policy on a
situation that cannot occur.

That makes tag visibility a constraint on the randomization, and it is not a box:
the overhead poses measured from the real recordings each clear the limit by
~75 px, but the corners of any box wide enough to contain them do not, because
extremes on all six axes compound. ``DomainRandomizationPreset.sample`` therefore
draws from a generous box and rejects the draws this module refuses, which keeps
the full per-axis range while dropping only the impossible corners.

The check runs against the *raw* sensor -- full 1920x1080, calibrated intrinsics,
barrel distortion -- because that is the image ``cam_align_solve`` detects on;
the rectified dataset frames are a downstream artifact nothing solves from. The
limit is not the sensor border but :func:`calibrated_radius_px`: the distortion
polynomial was fit from views that never reached the frame corners and stops
inverting at r~810 px, well inside the corner at r~1173 px, so a tag out past
that is imaged but not usefully measurable.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.cam_align_solve import tag_world_corners
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR

#: The camera whose pose the workspace-frame tags constrain.
CAMERA_NAME = "overhead_camera"

#: The real sensor the extrinsics solve reads.
SENSOR_W, SENSOR_H = 1920, 1080

#: The recorded overhead image, whose frustum the camera's own hardware must clear.
OUT_W, OUT_H = 960, 720

#: Corner clearance (px) a tag keeps from the usable-image boundary. The detector
#: needs the whole black border plus its quad-fit neighbourhood, so merely being
#: inside is not enough.
DEFAULT_TAG_MARGIN_PX = 20.0


def calibrated_radius_px(
    matrix: np.ndarray, dist: np.ndarray, *, tolerance_px: float = 1.0
) -> float:
    """Largest image radius at which the calibrated distortion model still inverts.

    Past some radius the fitted polynomial stops being monotonic:
    ``undistortPoints`` diverges there, and so would any PnP relying on it. Found
    by round-tripping a radial fan of pixels and taking the last radius whose
    residual stays under ``tolerance_px``.
    """
    center = matrix[:2, 2]
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = np.arange(50.0, 2000.0, 5.0)
    fan = center + radii[:, None, None] * np.stack([np.cos(angles), np.sin(angles)], axis=-1)[None]
    pixels = fan.reshape(-1, 1, 2).astype(np.float64)
    rays = cv2.undistortPoints(pixels, matrix, dist).reshape(-1, 2)
    with np.errstate(over="ignore", invalid="ignore"):
        back, _ = cv2.projectPoints(
            np.column_stack([rays, np.ones(len(rays))]), np.zeros(3), np.zeros(3), matrix, dist
        )
    residual = np.linalg.norm(back.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1)
    residual = np.nan_to_num(residual, nan=np.inf, posinf=np.inf)
    ok = (residual.reshape(len(radii), len(angles)) < tolerance_px).all(axis=1)
    failed = np.flatnonzero(~ok)
    return float(radii[failed[0] - 1]) if len(failed) else float(radii[-1])


def camera_module_geoms(model: mujoco.MjModel, camera: int) -> tuple[int, ...]:
    """Geoms rigidly attached to a camera -- its lens barrel and its board.

    Both cameras hang off their own ``*_camera_module`` body carrying nothing
    else, so the camera's body is exactly its hardware and nothing structural
    gets dragged along.
    """
    body = model.cam_bodyid[camera]
    return tuple(geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == body)


def apply_camera_jitter(
    model: mujoco.MjModel,
    camera: int,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    geom_base: dict[int, tuple[np.ndarray, np.ndarray]],
    position_m: np.ndarray,
    rotation_deg: np.ndarray,
) -> None:
    """Move a camera *and its hardware* by a jitter, as one rigid assembly.

    A real camera cannot slide backwards out of its own lens: sensor, barrel and
    board are one part. Displacing ``cam_pos``/``cam_quat`` alone detaches the
    optical centre from the housing, and since the centre sits exactly on the
    lens's front face, backing it up swings the barrel into frame as a hard black
    wedge. The geoms therefore take the same rigid transform, taken about the
    camera's nominal centre so the camera's own pose comes out unchanged by the
    rotation term.
    """
    delta = Rotation.from_euler("xyz", rotation_deg, degrees=True)
    model.cam_pos[camera] = base_pos + position_m
    model.cam_quat[camera] = (delta * Rotation.from_quat(base_quat[[1, 2, 3, 0]])).as_quat()[
        [3, 0, 1, 2]
    ]
    for geom, (geom_pos, geom_quat) in geom_base.items():
        # The compiler flags geoms whose frame coincides with their body's, and
        # mj_kinematics then copies the body frame and ignores geom_pos/geom_quat
        # entirely -- so a moved geom silently stays put. Clearing the flag makes
        # the fields authoritative; they are written in full every call, so the
        # canonical pose still comes back exactly on reset.
        model.geom_sameframe[geom] = 0
        model.geom_pos[geom] = base_pos + position_m + delta.apply(geom_pos - base_pos)
        model.geom_quat[geom] = (delta * Rotation.from_quat(geom_quat[[1, 2, 3, 0]])).as_quat()[
            [3, 0, 1, 2]
        ]


def _geom_surface_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Geom-local points dense enough to catch it entering the frustum."""
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        ring = np.stack([size[0] * np.cos(angle), size[0] * np.sin(angle)], axis=1)
        return np.vstack(
            [np.column_stack([ring, np.full(len(angle), z)]) for z in (-size[1], 0.0, size[1])]
        )
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array(
            [
                (sx * size[0], sy * size[1], sz * size[2])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            float,
        )
    return np.zeros((0, 3))


class OverheadPoseFilter:
    """Whether a jittered overhead pose produces a usable, solvable frame.

    Two independent rejections:

    * all four workspace-frame tags must stay usable, so the real rig could
      solve its extrinsics from the pose;
    * the camera must not see its own hardware. :func:`apply_camera_jitter` now
      carries the lens and board along, so this should never fire -- it is kept
      as a guard, because when the assembly did come apart the symptom was a hard
      black wedge across the frame rather than anything that looked like a bug.

    The tags are fixed to the workspace and the camera hangs off a static mount,
    so both tests are functions of the pose offset alone -- no rendering. One
    instance is reusable across episodes; build it through
    :func:`overhead_pose_filter`, which caches it.
    """

    def __init__(self, *, camera_name: str = CAMERA_NAME, intrinsics_path: Path | None = None):
        from pick_and_place.scene import build_environment

        model = build_environment().compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera < 0:
            raise ValueError(f"unknown camera {camera_name!r}")

        if intrinsics_path is None:
            intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
        if not Path(intrinsics_path).exists():
            raise FileNotFoundError(
                f"frame-tag visibility needs the calibrated intrinsics at {intrinsics_path}, "
                "which are machine-local and gitignored. Restore them, or set the preset's "
                "overhead_camera_frame_tag_margin_px to 0 to skip the check (which will "
                "randomize over poses the real rig could not be calibrated at)."
            )
        intrinsics = json.loads(Path(intrinsics_path).read_text())
        self.matrix = np.array(intrinsics["camera_matrix"], float)
        self.dist = np.array(intrinsics["dist_coeffs"], float).ravel()
        self.radius = calibrated_radius_px(self.matrix, self.dist)

        corners = tag_world_corners(model, data)
        if len(corners) < 4:
            raise ValueError(f"need all four workspace-frame tags, found {sorted(corners)}")
        self.corners = np.concatenate([corners[tag] for tag in sorted(corners)])
        self._model = model
        self._data = data
        self.nominal_pos = model.cam_pos[self.camera].copy()
        self.nominal_quat = model.cam_quat[self.camera].copy()

        # The recorder overrides cam_fovy from the calibration, so match it here
        # rather than trusting the scene's authored value.
        self.fovy = math.degrees(
            2.0 * math.atan((float(intrinsics["height"]) / 2.0) / self.matrix[1, 1])
        )
        module = camera_module_geoms(model, self.camera)
        self._geom_base = {
            geom: (model.geom_pos[geom].copy(), model.geom_quat[geom].copy()) for geom in module
        }
        self._own_geoms = {
            geom: points for geom in module if len(points := _geom_surface_points(model, geom))
        }

    def _camera_frame(
        self, position_m: np.ndarray, rotation_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the camera under a jitter, applied as the randomizer does."""
        apply_camera_jitter(
            self._model,
            self.camera,
            self.nominal_pos,
            self.nominal_quat,
            self._geom_base,
            np.asarray(position_m, float),
            np.asarray(rotation_deg, float),
        )
        mujoco.mj_forward(self._model, self._data)
        return (
            self._data.cam_xpos[self.camera].copy(),
            self._data.cam_xmat[self.camera].reshape(3, 3).copy(),
        )

    def margin_px(self, position_m: np.ndarray, rotation_deg: np.ndarray) -> float:
        """Clearance of the worst tag corner from the usable image, in pixels.

        Negative means at least one corner has left the sensor or the radius the
        lens calibration covers, either of which makes the pose unsolvable.
        """
        center, rotation = self._camera_frame(position_m, rotation_deg)
        # MuJoCo camera (x right, y up, z back) -> OpenCV (x right, y down, z fwd).
        rotation_cv = np.diag([1.0, -1.0, -1.0]) @ rotation.T
        rvec, _ = cv2.Rodrigues(rotation_cv)
        pixels, _ = cv2.projectPoints(
            self.corners, rvec, -rotation_cv @ center, self.matrix, self.dist
        )
        pixels = pixels.reshape(-1, 2)
        depth = (self.corners - center) @ rotation_cv[2]
        if np.any(depth <= 0):
            return -np.inf
        border = np.minimum.reduce(
            [pixels[:, 0], pixels[:, 1], SENSOR_W - pixels[:, 0], SENSOR_H - pixels[:, 1]]
        )
        calibrated = self.radius - np.linalg.norm(pixels - self.matrix[:2, 2], axis=1)
        return float(np.minimum(border, calibrated).min())

    def sees_own_hardware(
        self, position_m: np.ndarray, rotation_deg: np.ndarray, aspect: float = OUT_W / OUT_H
    ) -> bool:
        """Whether the camera's own lens or board falls inside the recorded frame.

        Checked against the recorded crop's frustum rather than the full render,
        since that is the image an episode keeps. Conservative by design: a point
        merely in front of the optical centre counts, even where backface culling
        would have hidden it, because the camera being inside its own barrel is
        not a pose worth recording either way.
        """
        self._camera_frame(position_m, rotation_deg)
        center = self._data.cam_xpos[self.camera]
        rotation = self._data.cam_xmat[self.camera].reshape(3, 3)
        tan_y = math.tan(math.radians(self.fovy) / 2.0)
        tan_x = tan_y * aspect
        for geom, local in self._own_geoms.items():
            world = self._data.geom_xpos[geom] + local @ self._data.geom_xmat[geom].reshape(3, 3).T
            camera = (world - center) @ rotation
            depth = -camera[:, 2]
            ahead = depth > 1e-6
            if not ahead.any():
                continue
            u = np.abs(camera[ahead, 0] / depth[ahead])
            v = np.abs(camera[ahead, 1] / depth[ahead])
            if np.any((u <= tan_x) & (v <= tan_y)):
                return True
        return False

    def accepts(
        self,
        position_m: np.ndarray,
        rotation_deg: np.ndarray,
        min_margin_px: float = DEFAULT_TAG_MARGIN_PX,
    ) -> bool:
        """Whether this pose yields a usable frame the real rig could solve from."""
        if self.sees_own_hardware(position_m, rotation_deg):
            return False
        return self.margin_px(position_m, rotation_deg) >= min_margin_px


@functools.lru_cache(maxsize=4)
def overhead_pose_filter(camera_name: str = CAMERA_NAME) -> OverheadPoseFilter:
    """Cached :class:`OverheadPoseFilter`; building one compiles the scene."""
    return OverheadPoseFilter(camera_name=camera_name)
