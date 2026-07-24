#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""How far may the overhead camera pose move before something breaks?

Domain randomization jitters the overhead camera by
``overhead_camera_position_mm`` / ``overhead_camera_rotation_deg``. The measured
calibration sits 9.13 mm and 1.76 deg from the authored pose, which the original
+-5 mm / +-1 deg box did not contain. Widening the box needs an upper bound, and
there are two independent ones.

**Calibratability.** ``cam_align_solve`` recovers the pose from the four
workspace-frame AprilTags, so a pose it cannot image all four tags from is one
the real rig could never be calibrated at. Its own plausibility gate is a second,
usually tighter ceiling: it rejects solves further than
``DEFAULT_MAX_NOMINAL_DELTA_MM`` / ``_DEG`` from nominal.

**Usable frames.** The recorded overhead image is a 960x720 center crop of a
1920x1080 render, and the cube or the drop zone leaving it makes an episode
useless regardless of what the camera could be calibrated at.

The two are measured against different images. The tags are checked on the
*real* sensor — full 1920x1080, calibrated intrinsics, barrel distortion — since
that is the image the solve runs on, and the distortion is what lets the real
camera see corners a distortion-free pinhole clips. (At the nominal pose the
sim's own render already cuts the two near tags off by 16 px; nothing in the sim
loop reads them, so this is only a statement about the real rig.) The workspace
sectors are checked on the sim's recorded 960x720 crop, which is what the policy
actually sees.

The tag limit is not the sensor border but ``calibrated_radius_px``: the
distortion polynomial was fit from views that never reached the frame corners
and stops being invertible at r~810 px, well inside the corner at r~1173 px. A
tag out past that is imaged but not usefully measurable, so that radius — not
the edge of the sensor — is what bounds a solve.

Both measures are pure forward projection, so a whole box is certified in
seconds. ``--render`` then confirms the worst sampled draw the expensive way: it
renders that pose, warps the render into the calibrated camera model, and runs
the same ``pupil_apriltags`` detector the real solve uses. The two have agreed
to the pixel at every pose tried. Note that the warp inverts the distortion and
so is itself only valid inside that radius; it prints a warning, and beyond it
only the forward projection should be believed.

Examples:

    cd py
    # nominal + measured margins, and the per-axis envelope
    python scripts/probe_camera_pose_envelope.py

    # certify a candidate randomization box
    python scripts/probe_camera_pose_envelope.py --box 25 3 --samples 20000

    # confirm the worst draw of that box with the real detector
    python scripts/probe_camera_pose_envelope.py --box 25 3 --render

    # certify a box centered where the real camera actually sits, not where the
    # scene authors it
    python scripts/probe_camera_pose_envelope.py \
        --center 12.7 -3.8 -11.0 -1.64 0.82 -1.25 --box 13 1.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.cam_align_solve import (
    DEFAULT_MAX_NOMINAL_DELTA_DEG,
    DEFAULT_MAX_NOMINAL_DELTA_MM,
    tag_world_corners,
)
from pick_and_place.camera_extrinsics import LOCAL_CAMERA_EXTRINSICS_DIR
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.camera_pose_envelope import (
    apply_camera_jitter,
    calibrated_radius_px,
    camera_module_geoms,
    overhead_pose_filter,
)
from pick_and_place.episodes import sample_cube, sample_target
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.scene import build_environment
from pick_and_place.sim_recorder import (
    configure_render_quality,
    fovy_from_intrinsics,
    resize_and_center_crop,
)
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY,
    CUBE_PLACEMENT_OVERLAY,
    PAN_AXIS,
    is_cube_drop_allowed,
    is_cube_placement_allowed,
)

CAMERA_NAME = "overhead_camera"

#: The real sensor, which ``cam_align_solve`` reads.
REAL_W, REAL_H = 1920, 1080
#: The sim recording chain: render this, then resize-and-center-crop to output.
RENDER_W, RENDER_H = 1920, 1080
OUT_W, OUT_H = 960, 720

#: Oversized pinhole render that ``--render`` warps into the calibrated model.
#: Wide enough that the whole distorted frame is covered, dense enough that tags
#: land on at least as many pixels as they would on the real sensor.
WIDE_W, WIDE_H = 2560, 1600

#: Corner clearance (px) a tag must keep from the sensor border to count as
#: usable. The detector needs the whole black border plus its quad-fit
#: neighbourhood, so touching the edge is not enough.
DEFAULT_TAG_MARGIN_PX = 20.0
#: Same idea for the workspace sectors in the recorded crop.
DEFAULT_SECTOR_MARGIN_PX = 0.0
#: Half-edge (m) of the drop-zone plate marker, from ``environment.py``.
PLATE_HALF_SIZE = 0.03


def _resize_crop_geometry() -> tuple[float, int, int]:
    """Scale and crop offsets of ``sim_recorder.resize_and_center_crop``."""
    scale = max(OUT_W / RENDER_W, OUT_H / RENDER_H)
    resized_w = max(OUT_W, round(RENDER_W * scale))
    resized_h = max(OUT_H, round(RENDER_H * scale))
    return scale, (resized_w - OUT_W) // 2, (resized_h - OUT_H) // 2


SCALE, CROP_L, CROP_T = _resize_crop_geometry()


class Scene:
    """The environment with the overhead camera's pose under our control."""

    def __init__(self, *, wide_render: bool = False) -> None:
        spec = build_environment()
        self.model = spec.compile()
        if wide_render:
            configure_render_quality(self.model)
            self.model.vis.global_.offwidth = WIDE_W
            self.model.vis.global_.offheight = WIDE_H
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if self.camera < 0:
            raise SystemExit(f"unknown camera {CAMERA_NAME!r}")

        intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / f"{CAMERA_NAME}.json"
        if not intrinsics_path.exists():
            raise SystemExit(
                f"missing calibrated intrinsics {intrinsics_path}; the envelope is "
                "meaningless without the real camera's focal length and distortion"
            )
        intrinsics = json.loads(intrinsics_path.read_text())
        self.matrix = np.array(intrinsics["camera_matrix"], float)
        self.dist = np.array(intrinsics["dist_coeffs"], float).ravel()
        self.fovy = fovy_from_intrinsics(intrinsics)
        self.model.cam_fovy[self.camera] = self.fovy

        self.nominal_pos = self.model.cam_pos[self.camera].copy()
        self.nominal_quat = self.model.cam_quat[self.camera].copy()
        self._geom_base = {
            geom: (self.model.geom_pos[geom].copy(), self.model.geom_quat[geom].copy())
            for geom in camera_module_geoms(self.model, self.camera)
        }
        self.pose(np.zeros(3), np.zeros(3))

    def pose(self, position_m: np.ndarray, rotation_deg: np.ndarray) -> None:
        """Apply a jitter exactly as ``DomainRandomizer._apply_camera`` does.

        Through the shared transform, so the rendered grid shows what recording
        actually produces -- the camera's lens and board move with it.
        """
        apply_camera_jitter(
            self.model,
            self.camera,
            self.nominal_pos,
            self.nominal_quat,
            self._geom_base,
            np.asarray(position_m, float),
            np.asarray(rotation_deg, float),
        )
        mujoco.mj_forward(self.model, self.data)

    # -- projections ----------------------------------------------------

    def _camera_frame(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.cam_xpos[self.camera], self.data.cam_xmat[self.camera].reshape(3, 3)

    def project_real(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points to real-sensor pixels, through the calibrated model."""
        center, rotation = self._camera_frame()
        # MuJoCo camera (x right, y up, z back) -> OpenCV (x right, y down, z fwd).
        rotation_cv = np.diag([1.0, -1.0, -1.0]) @ rotation.T
        rvec, _ = cv2.Rodrigues(rotation_cv)
        pixels, _ = cv2.projectPoints(
            points.astype(float), rvec, -rotation_cv @ center, self.matrix, self.dist
        )
        return pixels.reshape(-1, 2), (points - center) @ rotation_cv[2]

    def project_sim(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points to recorded-crop pixels, through MuJoCo's pinhole."""
        center, rotation = self._camera_frame()
        local = (points - center) @ rotation
        depth = -local[:, 2]
        focal = (RENDER_H / 2.0) / math.tan(math.radians(self.fovy) / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = focal * (local[:, 0] / depth) + RENDER_W / 2.0
            v = RENDER_H / 2.0 - focal * (local[:, 1] / depth)
        return np.column_stack([u * SCALE - CROP_L, v * SCALE - CROP_T]), depth

    # -- measures -------------------------------------------------------

    def tag_corners(self) -> tuple[list[int], np.ndarray]:
        corners = tag_world_corners(self.model, self.data)
        tags = sorted(corners)
        return tags, np.concatenate([corners[tag] for tag in tags])


def _border_margin(pixels: np.ndarray, depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Signed distance from each point to the image border; behind camera = -inf."""
    margin = np.minimum.reduce(
        [pixels[:, 0], pixels[:, 1], width - pixels[:, 0], height - pixels[:, 1]]
    )
    return np.where(depth > 0, margin, -np.inf)


def sector_points() -> np.ndarray:
    """Cube bounding-box corners over every allowed cube and drop position.

    A margin computed on these says the cube is *wholly* inside the recorded
    image anywhere the episode sampler could put it.
    """
    radius = max(CANONICAL_PICKUP_OVERLAY.outer_radius, CUBE_PLACEMENT_OVERLAY.outer_radius)
    step = 0.01
    axis = np.arange(-radius, radius + step, step)
    grid_x, grid_y = np.meshgrid(PAN_AXIS[0] + axis, PAN_AXIS[1] + axis)
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    allowed = np.array(
        [is_cube_placement_allowed(x, y) or is_cube_drop_allowed(x, y) for x, y in flat]
    )
    centers = flat[allowed]
    offsets = CUBE_HALF_SIZE * np.array(
        [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (1.0, 3.0)  # cube sits on the floor: z from 0 to 2*half
        ]
    )
    points = np.repeat(np.column_stack([centers, np.zeros(len(centers))]), len(offsets), axis=0)
    return points + np.tile(offsets, (len(centers), 1))


def episode_points(count: int = 1500, seed: int = 12345) -> np.ndarray:
    """Per-episode object extents, drawn from the episode samplers themselves.

    ``sector_points`` asks whether the *whole* allowed region stays in frame,
    which the extreme radii dominate; those corners are rarely sampled. This
    asks the question that decides whether a dataset is usable: for a given
    camera pose, what fraction of episodes would lose sight of their cube or
    their drop zone?  Returns ``(count, points_per_episode, 3)``.
    """
    rng = np.random.default_rng(seed)
    cube_offsets = CUBE_HALF_SIZE * np.array(
        [(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (1.0, 3.0)]
    )
    plate_offsets = np.array(
        [
            (sx * PLATE_HALF_SIZE, sy * PLATE_HALF_SIZE, 0.0)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
    )
    episodes = []
    for _ in range(count):
        cube = sample_cube(rng)
        target = sample_target(rng)
        episodes.append(
            np.concatenate(
                [
                    np.array((cube.x, cube.y, 0.0)) + cube_offsets,
                    np.array((target.x, target.y, 0.0)) + plate_offsets,
                    np.array((target.x, target.y, 0.0)) + cube_offsets,
                ]
            )
        )
    return np.stack(episodes)


class Envelope:
    """Both bounds, evaluated at an arbitrary camera-pose offset."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.sector = sector_points()
        self.episodes = episode_points()
        self.calibrated_radius = calibrated_radius_px(scene.matrix, scene.dist)

    def evaluate(self, position_m: np.ndarray, rotation_deg: np.ndarray) -> dict[str, Any]:
        self.scene.pose(position_m, rotation_deg)

        tags, corners = self.scene.tag_corners()
        pixels, depth = self.scene.project_real(corners)
        border = _border_margin(pixels, depth, REAL_W, REAL_H)
        # Distance to the edge of the calibrated region, which sits well inside
        # the sensor border and is what actually limits a usable solve.
        radius = np.linalg.norm(pixels - self.scene.matrix[:2, 2], axis=1)
        calibrated = np.where(depth > 0, self.calibrated_radius - radius, -np.inf)
        tag_margin = np.minimum(border, calibrated)
        per_tag = {tag: float(tag_margin[4 * i : 4 * i + 4].min()) for i, tag in enumerate(tags)}

        pixels, depth = self.scene.project_sim(self.sector)
        sector_margin = float(_border_margin(pixels, depth, OUT_W, OUT_H).min())

        shape = self.episodes.shape
        pixels, depth = self.scene.project_sim(self.episodes.reshape(-1, 3))
        per_point = _border_margin(pixels, depth, OUT_W, OUT_H).reshape(shape[0], shape[1])
        episode_margin = per_point.min(axis=1)

        rotation = Rotation.from_euler("xyz", rotation_deg, degrees=True)
        return {
            "tag_margin_px": float(tag_margin.min()),
            "per_tag_margin_px": per_tag,
            "sector_margin_px": sector_margin,
            "episode_loss_rate": float((episode_margin < 0).mean()),
            "episode_margin_p01_px": float(np.percentile(episode_margin, 1)),
            "delta_mm": float(np.linalg.norm(position_m) * 1000.0),
            "delta_deg": float(np.degrees(rotation.magnitude())),
        }

    def limit(
        self,
        direction_pos: np.ndarray,
        direction_rot: np.ndarray,
        *,
        measure: str,
        min_margin: float,
        ceiling: float,
    ) -> float:
        """Largest scale of ``direction`` whose margin still clears ``min_margin``."""

        def ok(scale: float) -> bool:
            result = self.evaluate(direction_pos * scale, direction_rot * scale)
            return result[measure] >= min_margin

        if not ok(0.0):
            return 0.0
        if ok(ceiling):
            return ceiling
        low, high = 0.0, ceiling
        for _ in range(48):
            mid = (low + high) / 2.0
            if ok(mid):
                low = mid
            else:
                high = mid
        return low


def measured_offset(scene: Scene) -> tuple[np.ndarray, np.ndarray] | None:
    """The jitter that would reproduce the calibrated pose, if one is on disk."""
    path = LOCAL_CAMERA_EXTRINSICS_DIR / f"{CAMERA_NAME}.json"
    if not path.exists():
        return None
    camera = json.loads(path.read_text())["cameras"][CAMERA_NAME]
    position = np.array(camera["pos"], float) - scene.nominal_pos
    measured = Rotation.from_quat(np.array(camera["quat"], float)[[1, 2, 3, 0]])
    nominal = Rotation.from_quat(scene.nominal_quat[[1, 2, 3, 0]])
    return position, (measured * nominal.inv()).as_euler("xyz", degrees=True)


AXES = (
    ("+x", (1e-3, 0, 0), (0, 0, 0), "mm"),
    ("-x", (-1e-3, 0, 0), (0, 0, 0), "mm"),
    ("+y", (0, 1e-3, 0), (0, 0, 0), "mm"),
    ("-y", (0, -1e-3, 0), (0, 0, 0), "mm"),
    ("+z", (0, 0, 1e-3), (0, 0, 0), "mm"),
    ("-z", (0, 0, -1e-3), (0, 0, 0), "mm"),
    ("+rx", (0, 0, 0), (1, 0, 0), "deg"),
    ("-rx", (0, 0, 0), (-1, 0, 0), "deg"),
    ("+ry", (0, 0, 0), (0, 1, 0), "deg"),
    ("-ry", (0, 0, 0), (0, -1, 0), "deg"),
    ("+rz", (0, 0, 0), (0, 0, 1), "deg"),
    ("-rz", (0, 0, 0), (0, 0, -1), "deg"),
)


def sample_rejected(
    envelope: Envelope,
    position_mm: float,
    rotation_deg: float,
    samples: int,
    seed: int,
    min_tag_margin: float,
    center: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Draw from a box, keep only poses where all four frame tags stay visible.

    A box is the wrong shape for this constraint. The real camera poses measured
    from the recorded sessions each clear the tag limit by ~75 px, but the
    corners of any box large enough to contain them do not: extremes on all six
    axes at once compound into poses the rig could never calibrate at. Squaring
    the region off therefore either clips the poses the camera actually takes or
    invents ones it cannot. Rejecting the draws that lose a tag keeps the real
    ones and drops only the impossible corners, which is what the randomization
    is for -- every pose it trains on is one the real rig could be in.
    """
    rng = np.random.default_rng(seed)
    center_pos, center_rot = center if center is not None else (np.zeros(3), np.zeros(3))
    accepted: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for _ in range(samples):
        position = center_pos / 1000.0 + rng.uniform(-position_mm, position_mm, size=3) / 1000.0
        rotation = center_rot + rng.uniform(-rotation_deg, rotation_deg, size=3)
        result = envelope.evaluate(position, rotation)
        if result["tag_margin_px"] >= min_tag_margin:
            accepted.append((position * 1000.0, rotation, result))
    if not accepted:
        return {"position_mm": position_mm, "rotation_deg": rotation_deg, "accepted": 0}
    positions = np.array([a[0] for a in accepted])
    rotations = np.array([a[1] for a in accepted])
    losses = np.array([a[2]["episode_loss_rate"] for a in accepted])
    return {
        "position_mm": position_mm,
        "rotation_deg": rotation_deg,
        "samples": samples,
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / samples,
        "accepted_position_range_mm": [
            positions.min(axis=0).tolist(),
            positions.max(axis=0).tolist(),
        ],
        "accepted_rotation_range_deg": [
            rotations.min(axis=0).tolist(),
            rotations.max(axis=0).tolist(),
        ],
        "accepted_position_std_mm": positions.std(axis=0).tolist(),
        "accepted_rotation_std_deg": rotations.std(axis=0).tolist(),
        "episode_loss_rate_mean": float(losses.mean()),
        "episode_loss_rate_p99": float(np.percentile(losses, 99)),
        "episode_loss_rate_max": float(losses.max()),
    }


def sample_box(
    envelope: Envelope,
    position_mm: float,
    rotation_deg: float,
    samples: int,
    seed: int,
    center: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Worst case over draws from the randomizer's uniform box.

    ``center`` offsets the box from the authored pose. The real camera does not
    sit at the authored pose and never has, so a box centered there spends its
    corners on poses the rig is never in while still having to reach the ones it
    is: recentering buys back most of the envelope.
    """
    rng = np.random.default_rng(seed)
    center_pos, center_rot = center if center is not None else (np.zeros(3), np.zeros(3))
    worst: dict[str, Any] = {"tag_margin_px": math.inf, "sector_margin_px": math.inf}
    tag_fail = sector_fail = gate_fail = 0
    losses = []
    for index in range(samples):
        position = center_pos / 1000.0 + rng.uniform(-position_mm, position_mm, size=3) / 1000.0
        rotation = center_rot + rng.uniform(-rotation_deg, rotation_deg, size=3)
        result = envelope.evaluate(position, rotation)
        for measure in ("tag_margin_px", "sector_margin_px"):
            if result[measure] < worst[measure]:
                worst[measure] = result[measure]
                worst[f"{measure}_at"] = {
                    "index": index,
                    "position_mm": (position * 1000.0).tolist(),
                    "rotation_deg": rotation.tolist(),
                    **result,
                }
        losses.append(result["episode_loss_rate"])
        tag_fail += result["tag_margin_px"] < DEFAULT_TAG_MARGIN_PX
        sector_fail += result["sector_margin_px"] < DEFAULT_SECTOR_MARGIN_PX
        gate_fail += (
            result["delta_mm"] > DEFAULT_MAX_NOMINAL_DELTA_MM
            or result["delta_deg"] > DEFAULT_MAX_NOMINAL_DELTA_DEG
        )
    losses = np.array(losses)
    return {
        "position_mm": position_mm,
        "rotation_deg": rotation_deg,
        "center_position_mm": center_pos.tolist(),
        "center_rotation_deg": center_rot.tolist(),
        "samples": samples,
        "seed": seed,
        "worst": worst,
        "tag_violation_rate": tag_fail / samples,
        "sector_violation_rate": sector_fail / samples,
        "solve_gate_violation_rate": gate_fail / samples,
        # Fraction of (camera draw, episode draw) pairs losing the cube or the
        # drop zone off the recorded image -- the share of a dataset wasted.
        "episode_loss_rate_mean": float(losses.mean()),
        "episode_loss_rate_p99": float(np.percentile(losses, 99)),
        "episode_loss_rate_max": float(losses.max()),
    }


def render_and_detect(
    position_m: np.ndarray, rotation_deg: np.ndarray, output: Path | None
) -> dict[str, Any]:
    """Render the pose, warp it into the real camera model, run the detector."""
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("--render needs pupil-apriltags installed") from exc

    scene = Scene(wide_render=True)
    wide_fovy = 75.0
    scene.model.cam_fovy[scene.camera] = wide_fovy

    grid_u, grid_v = np.meshgrid(
        np.arange(REAL_W, dtype=np.float32), np.arange(REAL_H, dtype=np.float32)
    )
    rays = cv2.undistortPoints(
        np.stack([grid_u.ravel(), grid_v.ravel()], axis=1).reshape(-1, 1, 2),
        scene.matrix,
        scene.dist,
    ).reshape(-1, 2)
    focal = (WIDE_H / 2.0) / math.tan(math.radians(wide_fovy) / 2.0)
    map_x = (focal * rays[:, 0] + WIDE_W / 2.0).reshape(REAL_H, REAL_W).astype(np.float32)
    map_y = (focal * rays[:, 1] + WIDE_H / 2.0).reshape(REAL_H, REAL_W).astype(np.float32)
    if map_x.min() < 0 or map_y.min() < 0 or map_x.max() >= WIDE_W or map_y.max() >= WIDE_H:
        print(
            "  note: the warp samples outside the wide render at the frame corners; "
            "tags near the border may be clipped by the probe, not by the camera"
        )

    scene.pose(position_m, rotation_deg)
    renderer = mujoco.Renderer(scene.model, height=WIDE_H, width=WIDE_W)
    renderer.update_scene(scene.data, camera=CAMERA_NAME)
    image = cv2.remap(renderer.render(), map_x, map_y, cv2.INTER_LINEAR)

    detector = Detector(families="tagStandard41h12", nthreads=4, refine_edges=True)
    detections = detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    frame_tags = {}
    for detection in detections:
        if detection.tag_id not in (12, 13, 14, 15):
            continue
        corners = np.asarray(detection.corners)
        frame_tags[int(detection.tag_id)] = {
            "margin_px": float(
                min(
                    corners[:, 0].min(),
                    corners[:, 1].min(),
                    REAL_W - corners[:, 0].max(),
                    REAL_H - corners[:, 1].max(),
                )
            ),
            "edge_px": float(np.linalg.norm(corners[0] - corners[1])),
            "decision_margin": float(detection.decision_margin),
        }
        cv2.polylines(image, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {"detected": sorted(frame_tags), "tags": frame_tags, "panel": str(output or "")}


def render_pose_grid(
    envelope: Envelope,
    size: int,
    position_mm: float,
    rotation_deg: float,
    seed: int,
    min_tag_margin: float,
    reference: list[tuple[str, np.ndarray, np.ndarray]],
    output: Path,
) -> Path:
    """Tile ``size``x``size`` recorded-crop views over sampled camera poses.

    The reference poses lead, outlined, so the randomization can be read against
    the pose the scene authors and the ones the real camera was measured at --
    the question being whether the spread looks like plausible camera placements
    rather than merely whether it is wide.
    """
    tile_w, tile_h = 384, 288
    pose_filter = overhead_pose_filter()
    scene = Scene()
    scene.model.vis.global_.offwidth = RENDER_W
    scene.model.vis.global_.offheight = RENDER_H
    configure_render_quality(scene.model)

    poses = list(reference[: size * size])
    rng = np.random.default_rng(seed)
    while len(poses) < size * size:
        position = rng.uniform(-position_mm, position_mm, size=3)
        rotation = rng.uniform(-rotation_deg, rotation_deg, size=3)
        if not pose_filter.accepts(position / 1000.0, rotation, min_tag_margin):
            continue
        label = (
            f"{np.linalg.norm(position):.0f}mm "
            f"{np.degrees(np.linalg.norm(np.radians(rotation))):.1f}deg"
        )
        poses.append((label, position, rotation))

    renderer = mujoco.Renderer(scene.model, height=RENDER_H, width=RENDER_W)
    tiles = []
    for index, (label, position, rotation) in enumerate(poses):
        scene.pose(position / 1000.0, rotation)
        renderer.update_scene(scene.data, camera=CAMERA_NAME)
        frame = resize_and_center_crop(renderer.render(), OUT_H, OUT_W)
        tile = cv2.cvtColor(
            cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2BGR,
        )
        highlight = index < len(reference)
        cv2.rectangle(tile, (0, 0), (tile_w - 1, 20), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255) if highlight else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if highlight:
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (0, 220, 255), 2)
        tiles.append(tile)

    grid = np.vstack([np.hstack(tiles[row * size : (row + 1) * size]) for row in range(size)])
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--box",
        nargs=2,
        type=float,
        metavar=("POSITION_MM", "ROTATION_DEG"),
        action="append",
        help="certify a randomization box (repeatable)",
    )
    parser.add_argument("--samples", type=int, default=5000, help="draws per box")
    parser.add_argument(
        "--center",
        nargs=6,
        type=float,
        metavar=("X_MM", "Y_MM", "Z_MM", "RX_DEG", "RY_DEG", "RZ_DEG"),
        help="offset the certified box from the authored pose (default: centered on it)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tag-margin-px",
        type=float,
        default=DEFAULT_TAG_MARGIN_PX,
        help="corner clearance a frame tag must keep from the sensor border",
    )
    parser.add_argument(
        "--reject-invisible",
        action="store_true",
        help="keep only draws where all four frame tags stay visible, and report "
        "how much variety survives",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="confirm the worst draw of the last box with the real detector",
    )
    parser.add_argument(
        "--grid",
        nargs="?",
        type=int,
        const=5,
        metavar="N",
        help="render an NxN grid of overhead views sampled from the last box "
        "(default 5), with the authored pose and any measured one on the top row",
    )
    parser.add_argument("--output", type=Path, help="write JSON and any render panels here")
    args = parser.parse_args()

    scene = Scene()
    envelope = Envelope(scene)
    report: dict[str, Any] = {
        "camera": CAMERA_NAME,
        "fovy_deg": scene.fovy,
        "real_sensor": [REAL_W, REAL_H],
        "recorded_image": [OUT_W, OUT_H],
        "tag_margin_px": args.tag_margin_px,
        "solve_gate": {
            "max_delta_mm": DEFAULT_MAX_NOMINAL_DELTA_MM,
            "max_delta_deg": DEFAULT_MAX_NOMINAL_DELTA_DEG,
        },
    }

    nominal = envelope.evaluate(np.zeros(3), np.zeros(3))
    report["nominal"] = nominal
    report["calibrated_radius_px"] = envelope.calibrated_radius
    corner_radius = float(
        np.hypot(
            max(scene.matrix[0, 2], REAL_W - scene.matrix[0, 2]),
            max(scene.matrix[1, 2], REAL_H - scene.matrix[1, 2]),
        )
    )
    print(f"camera fovy {scene.fovy:.3f} deg")
    print(
        f"distortion model inverts out to r={envelope.calibrated_radius:.0f} px "
        f"(sensor corner is at r={corner_radius:.0f} px), so that is the tag limit"
    )
    print(
        f"nominal pose: tag margin {nominal['tag_margin_px']:+.0f} px on the real sensor "
        f"({', '.join(f'{t}:{v:+.0f}' for t, v in nominal['per_tag_margin_px'].items())}), "
        f"workspace margin {nominal['sector_margin_px']:+.0f} px in the recorded crop"
    )

    offset = measured_offset(scene)
    if offset is not None:
        measured = envelope.evaluate(*offset)
        report["measured"] = {
            "position_mm": (offset[0] * 1000.0).tolist(),
            "rotation_deg": offset[1].tolist(),
            **measured,
        }
        print(
            f"measured pose ({measured['delta_mm']:.2f} mm, {measured['delta_deg']:.2f} deg off "
            f"nominal): tag margin {measured['tag_margin_px']:+.0f} px, "
            f"workspace margin {measured['sector_margin_px']:+.0f} px"
        )
    else:
        print("measured pose: no local extrinsics on disk, skipping")

    print(
        f"\nper-axis envelope (tag clearance >= {args.tag_margin_px:.0f} px | workspace in frame):"
    )
    print(f"  {'axis':5s} {'tags':>12s} {'workspace':>12s}")
    axis_limits: dict[str, dict[str, float]] = {}
    for name, position, rotation, unit in AXES:
        ceiling = 300.0 if unit == "mm" else 45.0
        limits = {
            measure: envelope.limit(
                np.array(position, float),
                np.array(rotation, float),
                measure=measure,
                min_margin=args.tag_margin_px
                if measure == "tag_margin_px"
                else DEFAULT_SECTOR_MARGIN_PX,
                ceiling=ceiling,
            )
            for measure in ("tag_margin_px", "sector_margin_px")
        }
        axis_limits[name] = {**limits, "unit": unit, "ceiling": ceiling}

        def show(value: float, ceiling: float = ceiling, unit: str = unit) -> str:
            return f">{ceiling:.0f} {unit}" if value >= ceiling else f"{value:.1f} {unit}"

        print(
            f"  {name:5s} {show(limits['tag_margin_px']):>12s} "
            f"{show(limits['sector_margin_px']):>12s}"
        )
    report["axis_limits"] = axis_limits

    if args.box:
        center = (
            (np.array(args.center[:3], float), np.array(args.center[3:], float))
            if args.center
            else None
        )
        if center is not None:
            print(
                f"\nbox certification ({args.samples} draws each), centered on "
                f"{center[0].tolist()} mm / {center[1].tolist()} deg:"
            )
        else:
            print(f"\nbox certification ({args.samples} draws each):")
        boxes = []
        for position_mm, rotation_deg in args.box:
            if args.reject_invisible:
                kept = sample_rejected(
                    envelope,
                    position_mm,
                    rotation_deg,
                    args.samples,
                    args.seed,
                    args.tag_margin_px,
                    center,
                )
                boxes.append(kept)
                if not kept["accepted"]:
                    print(
                        f"  +-{position_mm:5g} mm / +-{rotation_deg:4g} deg: no draw kept all 4 tags"
                    )
                    continue
                lo, hi = kept["accepted_position_range_mm"]
                rlo, rhi = kept["accepted_rotation_range_deg"]
                print(
                    f"  +-{position_mm:5g} mm / +-{rotation_deg:4g} deg: "
                    f"kept {kept['acceptance_rate'] * 100:5.1f}% | "
                    f"spread mm x[{lo[0]:+.0f},{hi[0]:+.0f}] y[{lo[1]:+.0f},{hi[1]:+.0f}] "
                    f"z[{lo[2]:+.0f},{hi[2]:+.0f}] | "
                    f"deg rx[{rlo[0]:+.1f},{rhi[0]:+.1f}] ry[{rlo[1]:+.1f},{rhi[1]:+.1f}] "
                    f"rz[{rlo[2]:+.1f},{rhi[2]:+.1f}] | "
                    f"episodes lost {kept['episode_loss_rate_mean'] * 100:.2f}%"
                )
                continue
            result = sample_box(
                envelope, position_mm, rotation_deg, args.samples, args.seed, center
            )
            boxes.append(result)
            worst = result["worst"]
            print(
                f"  +-{position_mm:5g} mm / +-{rotation_deg:4g} deg: "
                f"tags {worst['tag_margin_px']:+5.0f} px | "
                f"sector {worst['sector_margin_px']:+5.0f} px | "
                f"episodes lost mean {result['episode_loss_rate_mean'] * 100:5.2f}% "
                f"p99 {result['episode_loss_rate_p99'] * 100:5.2f}% "
                f"max {result['episode_loss_rate_max'] * 100:5.2f}% | "
                f"solve gate {result['solve_gate_violation_rate'] * 100:4.1f}%"
            )
        report["boxes"] = boxes

        if args.render:
            worst = boxes[-1]["worst"]
            report["render"] = {}
            for measure in ("tag_margin_px", "sector_margin_px"):
                draw = worst[f"{measure}_at"]
                panel = (args.output / f"worst_{measure[:-3]}.png") if args.output else None
                print(
                    f"\nrendering the draw with the worst {measure[:-3]}: "
                    f"{np.round(draw['position_mm'], 1).tolist()} mm, "
                    f"{np.round(draw['rotation_deg'], 2).tolist()} deg"
                )
                detection = render_and_detect(
                    np.array(draw["position_mm"]) / 1000.0, np.array(draw["rotation_deg"]), panel
                )
                report["render"][measure] = {"draw": draw, **detection}
                print(f"  detector found frame tags {detection['detected']}")
                for tag, info in sorted(detection["tags"].items()):
                    print(
                        f"    tag {tag}: margin {info['margin_px']:+.0f} px, "
                        f"edge {info['edge_px']:.0f} px, "
                        f"decision margin {info['decision_margin']:.0f}"
                    )
                if panel is not None:
                    print(f"  wrote {panel}")

        if args.grid:
            reference: list[tuple[str, np.ndarray, np.ndarray]] = [
                ("SIM AUTHORED", np.zeros(3), np.zeros(3))
            ]
            if offset is not None:
                reference.append(("measured", offset[0] * 1000.0, offset[1]))
            grid_path = (args.output or Path(".")) / "pose_grid.png"
            position_mm, rotation_deg = args.box[-1]
            print(f"\nrendering a {args.grid}x{args.grid} pose grid...")
            written = render_pose_grid(
                envelope,
                args.grid,
                position_mm,
                rotation_deg,
                args.seed,
                args.tag_margin_px,
                reference,
                grid_path,
            )
            report["grid"] = str(written)
            print(f"  wrote {written}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "camera_pose_envelope.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
