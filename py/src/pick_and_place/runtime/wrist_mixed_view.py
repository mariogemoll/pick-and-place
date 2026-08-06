# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Show the true wrist view blended with the believed one, and the solve on top.

Two layers of the same instant: what the wrist camera sees, and what the
controller thinks it would see — the arm at the believed joints, the cube at the
pose the planner is steering to. The visual offset between them *is* the injected
miscalibration, and watching the descent pull them into register is the clearest
picture there is of the visual servo working.

The drawing is the hardware overlay's, reused rather than re-derived: the same
tag outlines and the same projected cube wireframe
(:mod:`pick_and_place.runtime.wrist_servo`), so the sim window and the rig window
show the same thing and can be compared frame to frame.
"""

from __future__ import annotations

import numpy as np

from pick_and_place.runtime.wrist_servo import draw_cube, draw_tags, project_cube

#: How much of the true view survives the blend. The believed layer is an
#: underlay: present enough to read the offset, faint enough not to hide the cube.
TRUE_LAYER_WEIGHT = 0.6

WINDOW_TITLE = "Wrist Mixed (true + believed)"


def blend_mixed(
    true_rgb: np.ndarray,
    believed_rgb: np.ndarray,
    sighting,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> np.ndarray:
    """Compose one mixed BGR frame from a tick's two renders and its solve."""
    import cv2

    bgr = cv2.cvtColor(true_rgb, cv2.COLOR_RGB2BGR)
    draw_tags(bgr, sighting.detections, cv2)
    if sighting.estimate is not None:
        # The tracker's world estimate, projected back through the believed
        # camera it was solved with — so a good solve draws the wireframe over
        # the cube in the *true* layer, however far the believed layer has drifted.
        rvec, tvec, points = project_cube(
            sighting.estimate, camera_matrix, camera_position, camera_rotation, cv2
        )
        draw_cube(bgr, rvec, tvec, points, camera_matrix, cv2)
    return cv2.addWeighted(
        bgr,
        TRUE_LAYER_WEIGHT,
        cv2.cvtColor(believed_rgb, cv2.COLOR_RGB2BGR),
        1.0 - TRUE_LAYER_WEIGHT,
        0.0,
    )


def show_mixed(mixed_bgr: np.ndarray) -> None:
    """Put one composed frame on screen. Needs no MuJoCo viewer, so plain ``python`` runs it."""
    import cv2

    cv2.imshow(WINDOW_TITLE, mixed_bgr)
    cv2.waitKey(1)


def close_mixed() -> None:
    import cv2

    cv2.destroyAllWindows()
