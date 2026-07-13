# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Solve the overhead camera extrinsics from workspace-frame AprilTags.

The workspace frame carries four fixed tagStandard41h12 tags:

- id 12: ``workspace_frame_tag_ne``
- id 13: ``workspace_frame_tag_nw``
- id 14: ``workspace_frame_tag_sw``
- id 15: ``workspace_frame_tag_se``

This command detects those tags in a real overhead-camera frame, uses their known
3-D positions in the generated MuJoCo scene, runs OpenCV PnP, converts the result
to MuJoCo's camera frame convention, and saves a local sidecar:

    config/camera_extrinsics/overhead_camera.json

Example:

    cd py
    python -m pick_and_place.cam_align_solve \
        --camera 0 \
        --intrinsics ../config/camera_intrinsics/overhead_camera.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.camera_extrinsics import LOCAL_CAMERA_EXTRINSICS_DIR, save_camera_extrinsics
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.scene import build_environment

TAG_GEOMS: dict[int, tuple[str, tuple[int, int] | None]] = {
    12: ("workspace_frame_tag_ne", (2, +1)),
    13: ("workspace_frame_tag_nw", (2, +1)),
    14: ("workspace_frame_tag_sw", (2, +1)),
    15: ("workspace_frame_tag_se", (2, +1)),
}

# The printed workspace tags have a 40 mm graphic on a 60 mm sticker.  AprilTag
# detects the black border, which is 5/9 of the graphic edge for this family.
WORKSPACE_TAG_DETECTED_EDGE_M = 0.040 * (5.0 / 9.0)

# Startup-solve plausibility gate. A good solve reprojects to a couple of px and
# sits ~1 cm / ~2 deg from the model's nominal camera pose; swapped or rotated
# tags (or wrong intrinsics) push the reprojection error and/or the nominal delta
# far past these limits, so they are set generously to pass any honest solve while
# rejecting a mislabelled workspace frame.
DEFAULT_MAX_REPROJ_PX = 8.0
DEFAULT_MAX_NOMINAL_DELTA_MM = 40.0
DEFAULT_MAX_NOMINAL_DELTA_DEG = 6.0


class ExtrinsicsSolveError(RuntimeError):
    """The overhead extrinsics could not be solved or failed a plausibility check."""


@dataclass(frozen=True)
class NominalDelta:
    translation_m: float
    rotation_deg: float


@dataclass(frozen=True)
class SolveResult:
    used_tags: tuple[int, ...]
    reprojection_error_px: float
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    nominal_delta: NominalDelta


def parse_index_or_path(value: str) -> int | str:
    """Parse an OpenCV camera selector as an int index or device path."""
    try:
        return int(value)
    except ValueError:
        return value


def mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a canonical MuJoCo wxyz quaternion."""
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(matrix).as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    return quat if quat[0] >= 0.0 else -quat


def quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """Return the shortest angular distance between two wxyz quaternions."""
    from scipy.spatial.transform import Rotation

    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    return float(np.degrees((r0.inv() * r1).magnitude()))


def average_quaternions_wxyz(quaternions: list[np.ndarray]) -> np.ndarray:
    """Average same-hemisphere wxyz quaternions and normalize the result."""
    from scipy.spatial.transform import Rotation

    if not quaternions:
        raise ValueError("cannot average zero quaternions")
    quats_xyzw = [[q[1], q[2], q[3], q[0]] for q in quaternions]
    xyzw = Rotation.from_quat(quats_xyzw).mean().as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    return quat if quat[0] >= 0.0 else -quat


def average_results(
    results: list[SolveResult],
    *,
    nominal_pos: np.ndarray,
    nominal_quat: np.ndarray,
) -> SolveResult:
    """Average solved poses and reprojection statistics over multiple frames."""
    if not results:
        raise ValueError("cannot average zero solve results")
    pos = np.mean(np.array([result.pos for result in results], dtype=float), axis=0)
    quat = average_quaternions_wxyz(
        [np.array(result.quat, dtype=float) for result in results]
    )
    used_tags = tuple(sorted(set.intersection(*(set(result.used_tags) for result in results))))
    delta = NominalDelta(
        translation_m=float(np.linalg.norm(pos - nominal_pos)),
        rotation_deg=quat_angle_deg(quat, nominal_quat),
    )
    return SolveResult(
        used_tags=used_tags,
        reprojection_error_px=float(
            np.mean([result.reprojection_error_px for result in results])
        ),
        pos=tuple(float(v) for v in pos),
        quat=tuple(float(v) for v in quat),
        nominal_delta=delta,
    )


def pose_delta_mm_deg(
    pos_a: np.ndarray,
    quat_a: np.ndarray,
    pos_b: np.ndarray,
    quat_b: np.ndarray,
) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two parent-relative camera poses."""
    mm = float(np.linalg.norm(np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float)) * 1000.0)
    deg = quat_angle_deg(np.asarray(quat_a, dtype=float), np.asarray(quat_b, dtype=float))
    return mm, deg


def tag_world_points(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, np.ndarray]:
    """Return visible-face centers for the fixed workspace-frame tag geoms."""
    points: dict[int, np.ndarray] = {}
    mujoco.mj_forward(model, data)
    for tag_id, (geom_name, axis) in TAG_GEOMS.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        point = data.geom_xpos[geom_id].copy()
        if axis is not None:
            axis_index, sign = axis
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            point = point + sign * rotation[:, axis_index] * model.geom_size[geom_id][axis_index]
        points[tag_id] = point
    if len(points) < 4:
        raise ValueError(f"need all 4 workspace-frame tags, found {sorted(points)}")
    return points


def tag_world_corners(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, np.ndarray]:
    """Return AprilTag-detected corners in pupil-apriltags corner order.

    Each tag supplies four known PnP points, allowing a usable pose estimate
    when just one workspace tag is visible.
    """
    half_edge = WORKSPACE_TAG_DETECTED_EDGE_M / 2.0
    local_corners = half_edge * np.array(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
    )
    corners: dict[int, np.ndarray] = {}
    mujoco.mj_forward(model, data)
    for tag_id, (geom_name, axis) in TAG_GEOMS.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        center = data.geom_xpos[geom_id].copy()
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if axis is not None:
            axis_index, sign = axis
            center += sign * rotation[:, axis_index] * model.geom_size[geom_id][axis_index]
        corners[tag_id] = center + local_corners @ rotation.T
    return corners


def camera_matrix_from_intrinsics(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and scale calibrated camera intrinsics for the solve resolution."""
    data = json.loads(path.read_text())
    matrix = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(data["dist_coeffs"], dtype=float)
    sx = width / float(data["width"])
    sy = height / float(data["height"])
    matrix[0, :] *= sx
    matrix[1, :] *= sy
    return matrix, dist


def default_camera_matrix(width: int, height: int, fovy_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Fallback pinhole intrinsics from the MuJoCo camera fovy."""
    focal = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return matrix, np.zeros(5)


def opencv_camera_pose_to_mujoco_parent_pose(
    rotation_camera_world: np.ndarray,
    translation_camera_world: np.ndarray,
    parent_rotation_world: np.ndarray,
    parent_position_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an OpenCV solvePnP pose to parent-relative MuJoCo camera pose."""
    rotation_world_camera = rotation_camera_world.T
    camera_center_world = (-rotation_world_camera @ translation_camera_world).ravel()

    # OpenCV camera: x right, y down, z forward.
    # MuJoCo camera: x right, y up, z back.
    rotation_world_mujoco_camera = np.column_stack(
        [
            rotation_world_camera[:, 0],
            -rotation_world_camera[:, 1],
            -rotation_world_camera[:, 2],
        ]
    )

    parent_rotation = np.asarray(parent_rotation_world, dtype=float)
    parent_position = np.asarray(parent_position_world, dtype=float)
    pos = parent_rotation.T @ (camera_center_world - parent_position)
    quat = mat_to_quat_wxyz(parent_rotation.T @ rotation_world_mujoco_camera)
    return pos, quat


def solve_camera_pose(
    *,
    frame_rgb: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    matrix: np.ndarray,
    dist: np.ndarray,
    detector: Any,
    detections: list[Any] | None = None,
    min_workspace_tags: int = 4,
    cv2_module: Any,
    nominal_pos: np.ndarray,
    nominal_quat: np.ndarray,
) -> SolveResult | None:
    """Solve/apply the camera pose from detected workspace-frame tags.

    When ``detections`` is omitted, tags are detected from ``frame_rgb``.  A
    caller that also draws or otherwise consumes detections can supply them to
    avoid running the detector twice. ``min_workspace_tags=1`` uses each
    detected tag's four corners to provide a live estimate from a partial view;
    the default four-tag solve retains the established tag-center method.
    """
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"unknown camera {camera_name!r}")

    if detections is None:
        gray = cv2_module.cvtColor(frame_rgb, cv2_module.COLOR_RGB2GRAY)
        detections = detector.detect(gray)
    if not 1 <= min_workspace_tags <= len(TAG_GEOMS):
        raise ValueError(f"min_workspace_tags must be between 1 and {len(TAG_GEOMS)}")

    if min_workspace_tags == len(TAG_GEOMS):
        points = tag_world_points(model, data)
        matched = [det for det in detections if det.tag_id in points]
        if len(matched) < min_workspace_tags:
            return None
        object_points = np.array([points[det.tag_id] for det in matched], dtype=float)
        image_points = np.array([det.center for det in matched], dtype=float)
        flags = (
            cv2_module.SOLVEPNP_IPPE
            if np.ptp(object_points[:, 2]) < 1e-3
            else cv2_module.SOLVEPNP_ITERATIVE
        )
    else:
        corners = tag_world_corners(model, data)
        matched = [det for det in detections if det.tag_id in corners]
        if len(matched) < min_workspace_tags:
            return None
        object_points = np.concatenate([corners[det.tag_id] for det in matched]).astype(float)
        image_points = np.concatenate([det.corners for det in matched]).astype(float)
        flags = cv2_module.SOLVEPNP_IPPE if len(matched) == 1 else cv2_module.SOLVEPNP_SQPNP

    if not matched:
        return None
    ok, rvec, tvec = cv2_module.solvePnP(object_points, image_points, matrix, dist, flags=flags)
    if not ok:
        return None

    projected, _ = cv2_module.projectPoints(object_points, rvec, tvec, matrix, dist)
    reprojection_error = float(
        np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean()
    )

    rotation_camera_world, _ = cv2_module.Rodrigues(rvec)
    parent_id = int(model.cam_bodyid[camera_id])
    parent_rotation = data.xmat[parent_id].reshape(3, 3)
    parent_position = data.xpos[parent_id]
    pos, quat = opencv_camera_pose_to_mujoco_parent_pose(
        rotation_camera_world,
        tvec,
        parent_rotation,
        parent_position,
    )

    model.cam_pos[camera_id] = pos
    model.cam_quat[camera_id] = quat
    mujoco.mj_forward(model, data)

    delta = NominalDelta(
        translation_m=float(np.linalg.norm(pos - nominal_pos)),
        rotation_deg=quat_angle_deg(quat, nominal_quat),
    )
    return SolveResult(
        used_tags=tuple(sorted(det.tag_id for det in matched)),
        reprojection_error_px=reprojection_error,
        pos=tuple(float(v) for v in pos),
        quat=tuple(float(v) for v in quat),
        nominal_delta=delta,
    )


def solve_averaged_from_camera(
    cap: Any,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    matrix: np.ndarray,
    dist: np.ndarray,
    detector: Any,
    cv2_module: Any,
    nominal_pos: np.ndarray,
    nominal_quat: np.ndarray,
    samples: int,
    max_seconds: float,
    width: int,
    height: int,
    preview: bool = False,
) -> SolveResult | None:
    """Read frames until ``samples`` solve or ``max_seconds`` elapses, then average.

    Returns the averaged pose, or ``None`` if not one frame yielded a solve (all
    four workspace-frame tags visible) before the loop ended."""
    start = time.monotonic()
    results: list[SolveResult] = []
    while True:
        frame_rgb = read_camera_frame(cap, cv2_module)
        if frame_rgb is None:
            continue
        if frame_rgb.shape[1] != width or frame_rgb.shape[0] != height:
            frame_rgb = cv2_module.resize(frame_rgb, (width, height), interpolation=cv2_module.INTER_AREA)
        result = solve_camera_pose(
            frame_rgb=frame_rgb,
            model=model,
            data=data,
            camera_name=camera_name,
            matrix=matrix,
            dist=dist,
            detector=detector,
            cv2_module=cv2_module,
            nominal_pos=nominal_pos,
            nominal_quat=nominal_quat,
        )
        if result is not None:
            results.append(result)
            if samples > 1:
                print(
                    f"Sample {len(results)}/{samples}: "
                    f"{result.reprojection_error_px:.3f} px, "
                    f"{result.nominal_delta.translation_m * 1000.0:.1f} mm, "
                    f"{result.nominal_delta.rotation_deg:.2f} deg",
                    flush=True,
                )
        if preview:
            preview_bgr = cv2_module.cvtColor(frame_rgb, cv2_module.COLOR_RGB2BGR)
            cv2_module.imshow("cam_align_solve", preview_bgr)
            key = cv2_module.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        if len(results) >= samples:
            break
        if max_seconds > 0.0 and time.monotonic() - start > max_seconds:
            break
    if not results:
        return None
    return average_results(results, nominal_pos=nominal_pos, nominal_quat=nominal_quat)


def apply_solve_result(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    result: SolveResult,
) -> None:
    """Write an (averaged) solved pose onto the compiled model and re-forward."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ExtrinsicsSolveError(f"unknown camera {camera_name!r}")
    model.cam_pos[camera_id] = np.array(result.pos, dtype=float)
    model.cam_quat[camera_id] = np.array(result.quat, dtype=float)
    mujoco.mj_forward(model, data)


def check_solve_plausible(
    result: SolveResult,
    *,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
    max_nominal_delta_mm: float = DEFAULT_MAX_NOMINAL_DELTA_MM,
    max_nominal_delta_deg: float = DEFAULT_MAX_NOMINAL_DELTA_DEG,
) -> None:
    """Raise ``ExtrinsicsSolveError`` if a solved pose looks wrong.

    Catches a workspace frame whose tags are swapped, rotated, or placed in the
    wrong corners (large reprojection error), and a camera that has moved far from
    where the model expects it (large nominal delta)."""
    if set(result.used_tags) != set(TAG_GEOMS):
        raise ExtrinsicsSolveError(
            f"solved with tags {list(result.used_tags)}; need all four "
            f"workspace-frame tags {sorted(TAG_GEOMS)}"
        )
    if result.reprojection_error_px > max_reproj_px:
        raise ExtrinsicsSolveError(
            f"reprojection error {result.reprojection_error_px:.1f}px exceeds "
            f"{max_reproj_px:.1f}px — workspace-frame tags may be swapped, rotated, "
            "or in the wrong corners, or the intrinsics are wrong"
        )
    delta_mm = result.nominal_delta.translation_m * 1000.0
    delta_deg = result.nominal_delta.rotation_deg
    if delta_mm > max_nominal_delta_mm or delta_deg > max_nominal_delta_deg:
        raise ExtrinsicsSolveError(
            f"solved pose is {delta_mm:.0f}mm / {delta_deg:.1f}deg from nominal "
            f"(limits {max_nominal_delta_mm:.0f}mm / {max_nominal_delta_deg:.0f}deg) — "
            "check the camera mount and the tag placement"
        )


def solve_overhead_extrinsics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cap: Any,
    *,
    camera_name: str = "overhead_camera",
    intrinsics_path: Path | None = None,
    width: int = 1920,
    height: int = 1080,
    samples: int = 10,
    max_seconds: float = 10.0,
    flush_frames: int = 5,
    cv2_module: Any | None = None,
) -> SolveResult | None:
    """Solve the overhead camera extrinsics live from the workspace-frame tags.

    Returns the averaged ``SolveResult`` (not applied to ``model`` — call
    ``apply_solve_result`` after validating it), or ``None`` if all four tags were
    never visible in one frame within ``max_seconds`` (e.g. the arm is occluding
    them). Raises ``ExtrinsicsSolveError`` if the camera or apriltag dependency is
    missing."""
    if cv2_module is None:
        import cv2 as cv2_module  # type: ignore[no-redef]
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise ExtrinsicsSolveError(
            "solving overhead extrinsics requires opencv-python and pupil-apriltags"
        ) from exc

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ExtrinsicsSolveError(f"unknown camera {camera_name!r}")
    nominal_pos = model.cam_pos[camera_id].copy()
    nominal_quat = model.cam_quat[camera_id].copy()

    if intrinsics_path is None:
        local = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
        intrinsics_path = local if local.exists() else None
    if intrinsics_path is not None:
        matrix, dist = camera_matrix_from_intrinsics(Path(intrinsics_path), width, height)
    else:
        matrix, dist = default_camera_matrix(width, height, float(model.cam_fovy[camera_id]))

    # Drop buffered frames so we solve against what the camera sees now.
    for _ in range(flush_frames):
        cap.read()

    detector = Detector(families="tagStandard41h12", nthreads=4, refine_edges=True)
    return solve_averaged_from_camera(
        cap,
        model=model,
        data=data,
        camera_name=camera_name,
        matrix=matrix,
        dist=dist,
        detector=detector,
        cv2_module=cv2_module,
        nominal_pos=nominal_pos,
        nominal_quat=nominal_quat,
        samples=samples,
        max_seconds=max_seconds,
        width=width,
        height=height,
    )


def read_real_image(path: Path, cv2_module: Any) -> np.ndarray:
    """Read an image file as RGB."""
    frame_bgr = cv2_module.imread(str(path))
    if frame_bgr is None:
        raise SystemExit(f"could not read image {path}")
    return cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2RGB)


def open_camera(camera: int | str, width: int, height: int, fps: int, cv2_module: Any) -> Any:
    """Open an OpenCV VideoCapture with the requested stream settings."""
    cap = cv2_module.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera!r}")
    cap.set(cv2_module.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2_module.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2_module.CAP_PROP_FPS, fps)
    return cap


def read_camera_frame(cap: Any, cv2_module: Any) -> np.ndarray | None:
    """Read one OpenCV VideoCapture frame as RGB."""
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2RGB)


def print_result(result: SolveResult) -> None:
    """Print a concise solve summary."""
    print(f"Tags        : {list(result.used_tags)}")
    print(f"Reprojection: {result.reprojection_error_px:.3f} px")
    print(
        "Nominal delta: "
        f"{result.nominal_delta.translation_m * 1000.0:.1f} mm, "
        f"{result.nominal_delta.rotation_deg:.2f} deg"
    )
    print(f"Position    : {[round(v, 8) for v in result.pos]}")
    print(f"Quaternion  : {[round(v, 8) for v in result.quat]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--real-image", type=Path, help="captured real overhead frame")
    source.add_argument("--camera", help="OpenCV camera index or device path")
    source.add_argument("--self-test", action="store_true", help="render sim camera and solve it")
    parser.add_argument("--camera-name", default="overhead_camera")
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show", action="store_true", help="show a live camera window while solving")
    parser.add_argument("--no-save", action="store_true", help="report the solve without writing JSON")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means wait forever for live camera")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="number of solved live-camera frames to average before reporting/saving",
    )
    parser.add_argument(
        "--self-test-fovy",
        type=float,
        default=60.0,
        help="synthetic render fovy used only with --self-test, chosen to show all four tags",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    try:
        import cv2
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise SystemExit(
            "camera extrinsic solving requires opencv-python and pupil-apriltags"
        ) from exc

    spec = build_environment()
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if camera_id < 0:
        raise SystemExit(f"unknown camera {args.camera_name!r}")
    nominal_pos = model.cam_pos[camera_id].copy()
    nominal_quat = model.cam_quat[camera_id].copy()

    intrinsics = args.intrinsics
    if intrinsics is None:
        local_intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
        intrinsics = local_intrinsics if local_intrinsics.exists() else None
    if intrinsics is not None:
        matrix, dist = camera_matrix_from_intrinsics(intrinsics, args.width, args.height)
        print(f"Intrinsics  : {intrinsics}")
    else:
        matrix, dist = default_camera_matrix(
            args.width,
            args.height,
            float(model.cam_fovy[camera_id]),
        )
        print("Intrinsics  : nominal MuJoCo fovy (calibrated JSON recommended)")

    detector = Detector(families="tagStandard41h12", nthreads=4, refine_edges=True)

    renderer = None
    cap = None
    result: SolveResult | None = None
    try:
        if args.self_test:
            args.width = min(args.width, int(model.vis.global_.offwidth))
            args.height = min(args.height, int(model.vis.global_.offheight))
            model.cam_fovy[camera_id] = args.self_test_fovy
            matrix, dist = default_camera_matrix(
                args.width,
                args.height,
                float(model.cam_fovy[camera_id]),
            )
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            renderer.update_scene(data, camera=args.camera_name)
            frame_rgb = renderer.render()
            result = solve_camera_pose(
                frame_rgb=frame_rgb,
                model=model,
                data=data,
                camera_name=args.camera_name,
                matrix=matrix,
                dist=dist,
                detector=detector,
                cv2_module=cv2,
                nominal_pos=nominal_pos,
                nominal_quat=nominal_quat,
            )
        elif args.real_image is not None:
            frame_rgb = read_real_image(args.real_image, cv2)
            if frame_rgb.shape[1] != args.width or frame_rgb.shape[0] != args.height:
                frame_rgb = cv2.resize(frame_rgb, (args.width, args.height), interpolation=cv2.INTER_AREA)
            result = solve_camera_pose(
                frame_rgb=frame_rgb,
                model=model,
                data=data,
                camera_name=args.camera_name,
                matrix=matrix,
                dist=dist,
                detector=detector,
                cv2_module=cv2,
                nominal_pos=nominal_pos,
                nominal_quat=nominal_quat,
            )
        else:
            cap = open_camera(parse_index_or_path(args.camera), args.width, args.height, args.fps, cv2)
            result = solve_averaged_from_camera(
                cap,
                model=model,
                data=data,
                camera_name=args.camera_name,
                matrix=matrix,
                dist=dist,
                detector=detector,
                cv2_module=cv2,
                nominal_pos=nominal_pos,
                nominal_quat=nominal_quat,
                samples=args.samples,
                max_seconds=args.max_seconds,
                width=args.width,
                height=args.height,
                preview=args.show,
            )
    finally:
        if renderer is not None:
            renderer.close()
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if result is None:
        raise SystemExit("no pose solved; need all four workspace-frame tags visible")

    model.cam_pos[camera_id] = np.array(result.pos, dtype=float)
    model.cam_quat[camera_id] = np.array(result.quat, dtype=float)

    print_result(result)
    if args.no_save:
        return

    output = args.output or (LOCAL_CAMERA_EXTRINSICS_DIR / f"{args.camera_name}.json")
    meta = {
        "method": "workspace-frame AprilTag PnP (pick_and_place.cam_align_solve)",
        "intrinsics": str(intrinsics) if intrinsics is not None else None,
        "reference_tags": list(result.used_tags),
        "rms_reproj_px": round(result.reprojection_error_px, 3),
        "nominal_delta_mm": round(result.nominal_delta.translation_m * 1000.0, 3),
        "nominal_delta_deg": round(result.nominal_delta.rotation_deg, 3),
    }
    path = save_camera_extrinsics(model, args.camera_name, path=output, meta=meta)
    print(f"Saved       : {path}")


if __name__ == "__main__":
    main()
