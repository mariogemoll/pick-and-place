# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evidence for why the overhead camera is not seeing the drop-zone square.

A blind plate hunt is the controller saying it has never had a sighting, which
is a perception answer, not a motion one: the useful record is the frame, the
mask the detector actually thresholded, and the measurement each surviving
contour failed on.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from pick_and_place.core.camera_projection import project_to_pixel
from pick_and_place.perception.paper_detection import (
    TargetCandidate,
    target_candidates,
    target_mask,
)
from pick_and_place.spec.drop_zone import PaperTarget

# How many of the largest contours to outline and name. Enough to cover the
# plate and whatever swallowed or split it, short of drawing every speckle.
LABELED_CANDIDATES = 6


class PlateSearch(Protocol):
    """The controller state a plate-hunt dump keys on."""

    @property
    def phase_name(self) -> str | None: ...

    @property
    def drop_target(self) -> PaperTarget | None: ...


def describe_candidate(candidate: TargetCandidate) -> str:
    """One line of a contour's measurements and the filter it failed."""
    return (
        f"area {candidate.area_fraction * 100.0:5.2f}%  "
        f"corners {candidate.corner_count:2d}  "
        f"aspect {candidate.aspect:5.2f}  "
        f"rectangularity {candidate.rectangularity:4.2f}  "
        f"-> {candidate.rejection or 'accepted'}"
    )


def _draw_candidate(bgr: NDArray, candidate: TargetCandidate) -> None:
    color = (0, 200, 0) if candidate.rejection is None else (0, 0, 255)
    box = np.round(candidate.box_px).astype(np.int32)
    cv2.polylines(bgr, [box.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    anchor = (int(box[:, 0].min()), max(12, int(box[:, 1].min()) - 6))
    cv2.putText(
        bgr,
        describe_candidate(candidate),
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def write_plate_search_debug(
    path: Path,
    frame_rgb: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    target_color: str,
    workspace_corners_world: NDArray,
) -> list[TargetCandidate]:
    """Write the frame and its drop-zone mask side by side, contours labeled.

    Returns the largest contours the mask offered, each carrying the filter it
    failed, so a caller can report them without re-running the detector.
    """
    mask = target_mask(
        frame_rgb,
        camera_matrix,
        camera_position,
        camera_rotation,
        target_color=target_color,
        workspace_corners_world=workspace_corners_world,
    )
    candidates = sorted(
        target_candidates(mask), key=lambda candidate: candidate.area_px, reverse=True
    )[:LABELED_CANDIDATES]

    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    quad_px = project_to_pixel(
        workspace_corners_world, camera_matrix, camera_position, camera_rotation
    )
    cv2.polylines(
        frame_bgr,
        [np.round(quad_px).astype(np.int32).reshape(-1, 1, 2)],
        True,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for candidate in candidates:
        _draw_candidate(frame_bgr, candidate)

    panel = np.concatenate((frame_bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), axis=1)
    cv2.putText(
        panel,
        f"no {target_color} drop zone: frame (workspace quad in yellow) | mask",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), panel):
        print(f"Warning: could not write plate debug image: {path}")
    return candidates


def plate_search_debugger(
    directory: Path,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    target_color: str,
    workspace_corners_world: NDArray,
    now: Callable[[], datetime.datetime] = datetime.datetime.now,
) -> Callable[[PlateSearch, NDArray], None]:
    """Dump the drop-zone mask once each time a blind plate hunt begins.

    The controller hunts for the plate only when it has no sighting at all, so
    the first tick of a hunt is the moment worth keeping evidence from.
    """
    hunting = False

    def observe(controller: PlateSearch, frame_rgb: NDArray) -> None:
        nonlocal hunting
        started = controller.phase_name == "find_plate" and controller.drop_target is None
        if started and not hunting:
            path = directory / f"{now().strftime('%Y%m%d_%H%M%S')}_plate_search.png"
            candidates = write_plate_search_debug(
                path,
                frame_rgb,
                camera_matrix,
                camera_position,
                camera_rotation,
                target_color=target_color,
                workspace_corners_world=workspace_corners_world,
            )
            print(f"Hunting for the plate with no sighting yet; wrote {path}")
            for candidate in candidates:
                print(f"  {describe_candidate(candidate)}")
            if not candidates:
                print("  the mask held no contours at all")
        hunting = started

    return observe
