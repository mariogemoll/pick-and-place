# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The wrist camera's live preview window, for watching a run on the rig.

Nothing here feeds control, so it is deliberately best-effort: during the descent
the window waits for a frame the detector has actually annotated, and outside it
takes whatever the camera thread last read. A tick that has nothing new keeps the
frame it already has rather than stalling the loop for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pick_and_place.runtime.wrist_servo import show_frame


@dataclass
class WristView:
    """The preview window's state: whether it is open, and what it last showed."""

    renderer: Any = None
    show: bool = False
    last_preview_id: int = field(default=-1)


def show_preview(plant: Any, wrist: WristView, is_descent: bool) -> None:
    """Put the newest camera frame on screen, if there is a newer one."""
    if not wrist.show or plant.servo is None:
        return
    servo = plant.servo
    if is_descent:
        preview = plant.last_preview
        if preview is None or preview.frame_id == wrist.last_preview_id:
            return
        bgr, wrist.last_preview_id = preview.bgr, preview.frame_id
    else:
        snapshot = servo.reader.latest()
        if snapshot is None:
            return
        bgr = snapshot.bgr
    show_frame(
        bgr.copy(),
        renderer=wrist.renderer,
        model=plant.model,
        data=plant.data,
        undistort_map=servo.undistort_map,
        rectify=not is_descent,
    )
