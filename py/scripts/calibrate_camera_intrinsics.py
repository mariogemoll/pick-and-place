#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Interactively calibrate camera intrinsics with the project's ChArUco board.

First generate and print the board at exact scale:

    cd py
    python scripts/generate_charuco_board.py

Then calibrate a camera. Keep the board still briefly at each pose; views are
captured automatically when it is stable and sufficiently different from the
previous capture.  Cover the image corners, vary distance, and include modest
30--45 degree tilts.

    python scripts/calibrate_camera_intrinsics.py \
        --camera 0 \
        --output ../config/camera_intrinsics/iphone_overview.json

The result is written to the explicit JSON path supplied with ``--output``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_charuco_board import (
    DEFAULT_MARKER_MM,
    DEFAULT_SQUARE_MM,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
    make_board,
)


@dataclass
class View:
    object_points: np.ndarray
    image_points: np.ndarray
    ids: np.ndarray
    center: np.ndarray
    extent: float


@dataclass(frozen=True)
class Coverage:
    """A compact summary of the poses represented by accepted views."""

    image_cells: frozenset[tuple[int, int]]
    distances: frozenset[str]
    has_tilted_view: bool


CELL_NAMES = (
    ("upper-left", "upper", "upper-right"),
    ("left", "center", "right"),
    ("lower-left", "lower", "lower-right"),
)


def image_cell(center: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int]:
    """Return the 3-by-3 image region containing a detected board center."""
    width, height = image_size
    column = min(2, max(0, int(3 * center[0] / width)))
    row = min(2, max(0, int(3 * center[1] / height)))
    return row, column


def distance_bucket(extent: float, image_size: tuple[int, int]) -> str:
    """Classify the board's apparent size into useful calibration distances."""
    fraction = extent / float(np.hypot(*image_size))
    if fraction < 0.25:
        return "far"
    if fraction < 0.50:
        return "medium"
    return "near"


def is_tilted(cv2_module, view: View) -> bool:
    """Detect perspective foreshortening without needing calibrated intrinsics."""
    object_xy = view.object_points.reshape(-1, 3)[:, :2]
    image_xy = view.image_points.reshape(-1, 2)
    homography, _ = cv2_module.findHomography(object_xy, image_xy, 0)
    if homography is None:
        return False
    minimum = object_xy.min(axis=0)
    maximum = object_xy.max(axis=0)
    board_corners = np.array(
        [[minimum[0], minimum[1]], [maximum[0], minimum[1]],
         [maximum[0], maximum[1]], [minimum[0], maximum[1]]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    corners = cv2_module.perspectiveTransform(board_corners, homography).reshape(-1, 2)
    top, right, bottom, left = (
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3]),
    )
    ratios = (
        top / max(bottom, 1.0),
        bottom / max(top, 1.0),
        right / max(left, 1.0),
        left / max(right, 1.0),
    )
    return max(ratios) >= 1.18


def summarize_coverage(cv2_module, views: list[View], image_size: tuple[int, int]) -> Coverage:
    """Summarize image placement, distance, and perspective across captures."""
    return Coverage(
        image_cells=frozenset(image_cell(view.center, image_size) for view in views),
        distances=frozenset(distance_bucket(view.extent, image_size) for view in views),
        has_tilted_view=any(is_tilted(cv2_module, view) for view in views),
    )


def next_pose_hint(coverage: Coverage) -> str:
    """Give one actionable suggestion for the most important missing coverage."""
    priority = ((0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1), (1, 1))
    for row, column in priority:
        if (row, column) not in coverage.image_cells:
            return f"Next: move the board toward the {CELL_NAMES[row][column]} of the image."
    for distance, instruction in (
        ("far", "move the board farther away."),
        ("medium", "use a medium distance."),
        ("near", "move the board closer to the camera."),
    ):
        if distance not in coverage.distances:
            return f"Next: {instruction}"
    if not coverage.has_tilted_view:
        return "Next: tilt the board about 30–45 degrees."
    return "Coverage looks good; add any diverse, well-detected poses."


def draw_coverage_grid(cv2_module, frame, coverage: Coverage) -> None:
    """Draw a compact image-placement coverage grid in the upper-right corner."""
    cell_size = 28
    margin = 14
    left = frame.shape[1] - 3 * cell_size - margin
    top = margin
    cv2_module.putText(
        frame,
        "image coverage",
        (left - 2, top - 4),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2_module.LINE_AA,
    )
    for row in range(3):
        for column in range(3):
            covered = (row, column) in coverage.image_cells
            color = (0, 180, 0) if covered else (0, 100, 255)
            x = left + column * cell_size
            y = top + row * cell_size
            cv2_module.rectangle(frame, (x, y), (x + cell_size, y + cell_size), color, 2)


def make_view(board, corners: np.ndarray | None, ids: np.ndarray | None) -> View | None:
    """Map a ChArUco detection into matching OpenCV calibration points."""
    if corners is None or ids is None or len(corners) < 6:
        return None
    object_points, image_points = board.matchImagePoints(corners, ids)
    if len(object_points) < 6:
        return None
    points = image_points.reshape(-1, 2).astype(np.float32)
    span = points.max(axis=0) - points.min(axis=0)
    return View(
        object_points=object_points.astype(np.float32),
        image_points=image_points.astype(np.float32),
        ids=ids.reshape(-1).astype(int),
        center=points.mean(axis=0),
        extent=float(np.linalg.norm(span)),
    )


def is_distinct(view: View, previous: View | None) -> bool:
    """Reject an automatic capture that is effectively the preceding view."""
    if previous is None:
        return True
    center_shift = float(np.linalg.norm(view.center - previous.center))
    size_change = abs(view.extent / max(previous.extent, 1.0) - 1.0)
    common = sorted(set(view.ids).intersection(previous.ids))
    if common:
        before = {tag_id: point for tag_id, point in zip(previous.ids, previous.image_points.reshape(-1, 2))}
        after = {tag_id: point for tag_id, point in zip(view.ids, view.image_points.reshape(-1, 2))}
        movement = float(np.median([np.linalg.norm(after[tag_id] - before[tag_id]) for tag_id in common]))
    else:
        movement = float("inf")
    return center_shift >= 30.0 or size_change >= 0.12 or movement >= 18.0


def orientation_is_accepted(image_size: tuple[int, int], orientation: str) -> bool:
    """Return whether a frame's dimensions match the requested orientation."""
    width, height = image_size
    return (
        orientation == "any"
        or (orientation == "landscape" and width > height)
        or (orientation == "portrait" and height > width)
    )


def calibrate(cv2_module, views: list[View], image_size: tuple[int, int], rational: bool):
    flags = cv2_module.CALIB_RATIONAL_MODEL if rational else 0
    rms, matrix, distortion, rvecs, tvecs = cv2_module.calibrateCamera(
        [view.object_points for view in views],
        [view.image_points for view in views],
        image_size,
        None,
        None,
        flags=flags,
    )
    errors = []
    for view, rvec, tvec in zip(views, rvecs, tvecs):
        projected, _ = cv2_module.projectPoints(view.object_points, rvec, tvec, matrix, distortion)
        residual = projected.reshape(-1, 2) - view.image_points.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return float(rms), matrix, distortion, errors


def write_result(path: Path, matrix, distortion, image_size: tuple[int, int], rms: float, views: int) -> None:
    width, height = image_size
    payload = {
        "model": "rational" if len(distortion.ravel()) > 5 else "standard",
        "width": width,
        "height": height,
        "camera_matrix": matrix.tolist(),
        "dist_coeffs": distortion.ravel().tolist(),
        "rms_reproj_px": rms,
        "n_views": views,
        "fovy_deg": float(np.degrees(2.0 * np.arctan((height / 2.0) / matrix[1, 1]))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def draw_status(
    cv2_module,
    frame,
    corners,
    ids,
    views: list[View],
    stable_for: float,
    result,
    orientation_warning: str | None,
    coverage: Coverage,
) -> None:
    if corners is not None and ids is not None:
        cv2_module.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0, 220, 0))
    lines = [
        f"views: {len(views)}  stable: {stable_for:.1f}s",
        f"coverage: image {len(coverage.image_cells)}/9  distance {len(coverage.distances)}/3  "
        f"tilt {'yes' if coverage.has_tilted_view else 'missing'}",
        next_pose_hint(coverage),
        "auto-capture when still; u undo, x drop worst, d reset, s save, q quit",
    ]
    if result is None:
        lines.append("collect 20-30 diverse views across the whole image, including tilted views")
    else:
        rms, _, _, errors = result
        lines.append(f"RMS: {rms:.3f}px  worst view: {max(errors):.3f}px")
    if orientation_warning is not None:
        lines.append(orientation_warning)
    for row, text in enumerate(lines):
        color = (0, 80, 255) if text == orientation_warning or text.startswith("Next:") else (255, 255, 255)
        cv2_module.putText(frame, text, (12, 28 + 25 * row), cv2_module.FONT_HERSHEY_SIMPLEX,
                           0.58, color, 2, cv2_module.LINE_AA)
    draw_coverage_grid(cv2_module, frame, coverage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="0", help="OpenCV camera index or device path")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSON path for the calibrated intrinsics",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    parser.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM)
    parser.add_argument("--stable-seconds", type=float, default=0.8)
    parser.add_argument("--max-views", type=int, default=30)
    parser.add_argument(
        "--orientation",
        choices=("landscape", "portrait", "any"),
        default="landscape",
        help="only accept captures with this frame orientation (default: landscape)",
    )
    parser.add_argument("--rational", action="store_true", help="use OpenCV's 8-coefficient rational lens model")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("camera calibration requires opencv-python") from exc
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise SystemExit("this OpenCV build has no ChArUco support; install opencv-python >= 4.7")
    try:
        board = make_board(cv2, args.squares_x, args.squares_y, args.square_mm, args.marker_mm)
    except ValueError as exc:
        parser.error(str(exc))

    from pick_and_place.cam_align_solve import open_camera, parse_index_or_path, read_camera_frame
    detector = cv2.aruco.CharucoDetector(board)
    cap = open_camera(parse_index_or_path(args.camera), args.width, args.height, 30, cv2)
    window = "ChArUco intrinsics calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    views: list[View] = []
    last_points = None
    still_since = None
    result = None
    image_size = None
    captured_image_size = None
    rejected_orientation = None
    try:
        while True:
            frame_rgb = read_camera_frame(cap, cv2)
            if frame_rgb is None:
                continue
            image_size = (frame_rgb.shape[1], frame_rgb.shape[0])
            now = time.monotonic()
            orientation_ok = orientation_is_accepted(image_size, args.orientation)
            orientation_warning = None
            frame_is_usable = orientation_ok and (
                captured_image_size is None or image_size == captured_image_size
            )
            if not orientation_ok:
                rejected_orientation = image_size
                orientation_warning = (
                    f"Ignoring {image_size[0]}x{image_size[1]} frame; "
                    f"waiting for {args.orientation} orientation"
                )
            elif not frame_is_usable:
                orientation_warning = (
                    f"Ignoring {image_size[0]}x{image_size[1]} frame; calibration uses "
                    f"{captured_image_size[0]}x{captured_image_size[1]}"
                )
                rejected_orientation = image_size

            if not frame_is_usable:
                still_since = None
                last_points = None
            else:
                if rejected_orientation is not None:
                    print(f"Accepted {image_size[0]}x{image_size[1]} {args.orientation} frames again.")
                    rejected_orientation = None
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                corners, ids, _, _ = detector.detectBoard(gray)
                view = make_view(board, corners, ids)
                points = None if view is None else view.image_points.reshape(-1, 2)
                if points is None:
                    still_since = None
                elif last_points is None or points.shape != last_points.shape or np.median(np.linalg.norm(points - last_points, axis=1)) > 1.0:
                    still_since = now
                elif still_since is None:
                    still_since = now
                last_points = points
                stable_for = 0.0 if still_since is None else now - still_since

                if view is not None and stable_for >= args.stable_seconds and is_distinct(view, views[-1] if views else None):
                    if captured_image_size is None:
                        captured_image_size = image_size
                    views.append(view)
                    still_since = None
                    if len(views) > args.max_views:
                        if result is not None:
                            views.pop(int(np.argmax(result[3])))
                        else:
                            views.pop(0)
                    if len(views) >= 6:
                        result = calibrate(cv2, views, image_size, args.rational)
                        print(f"Captured view {len(views)}; RMS {result[0]:.3f}px")
                    else:
                        print(f"Captured view {len(views)}")

            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            coverage = summarize_coverage(cv2, views, image_size)
            draw_status(
                cv2,
                display,
                corners if frame_is_usable else None,
                ids if frame_is_usable else None,
                views,
                0.0 if not frame_is_usable else stable_for,
                result,
                orientation_warning,
                coverage,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("u") and views:
                views.pop()
                result = calibrate(cv2, views, image_size, args.rational) if len(views) >= 6 else None
            if key == ord("x") and result is not None and len(views) > 6:
                views.pop(int(np.argmax(result[3])))
                result = calibrate(cv2, views, image_size, args.rational)
            if key == ord("d"):
                views.clear()
                result = None
                captured_image_size = None
            if key == ord("s"):
                if result is None:
                    print("Need at least 6 accepted views before saving.")
                elif not frame_is_usable:
                    print("Cannot save while the current frame has a rejected orientation or size.")
                else:
                    write_result(
                        args.output,
                        result[1],
                        result[2],
                        captured_image_size,
                        result[0],
                        len(views),
                    )
                    print(f"Saved {args.output} ({len(views)} views, {result[0]:.3f}px RMS)")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
