# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Wrap the scene in a skybox textured with a panorama of the real surroundings.

The panorama is an equirectangular image of the room, recovered from the wrist
camera's own footage (see ``scripts/accumulate_wrist_panorama.py``). Rendering it
as a MuJoCo skybox gives every camera — the outward-pointing wrist camera in
particular — a plausible, blurry backdrop beyond the tabletop, closing the part of
the sim-to-real gap that lies outside the physical setup.

A skybox is the right primitive here: it renders unlit and infinitely far behind
all geometry, so it is immune to the scene's lights (an inward-facing textured
sphere, by contrast, is shaded by those lights and turns black on the faces that
point away from them). MuJoCo samples the skybox by direction, so the panorama is
resampled into the six cube faces MuJoCo expects, oriented to the world frame that
the panorama was built in: a world direction maps to the same pixel that imaged it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

SKYBOX_NAME = "background_panorama"

# World-frame direction of each cube face and how its (column, row) pixel axes run,
# as measured from MuJoCo's skybox sampling. Each entry maps normalized face
# coordinates (a, b) in [-1, 1] — a across columns, b down rows — to a direction.
_FACE_ORDER = ("right", "left", "up", "down", "front", "back")


def _face_direction(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    one = np.ones_like(a)
    dirs = {
        "right": (one, a, -b),
        "left": (-one, -a, -b),
        "up": (a, -b, one),
        "down": (a, b, -one),
        "front": (a, -one, -b),
        "back": (-a, one, -b),
    }
    return np.stack(dirs[name], axis=-1)


def equirect_to_skybox(equirect_rgb: np.ndarray, face_size: int) -> np.ndarray:
    """Resample an equirectangular RGB image into MuJoCo's stacked cube faces.

    Returns a ``(6 * face_size, face_size, 3)`` uint8 array with the faces in
    MuJoCo's skybox order. The equirectangular convention matches the panorama
    builder: azimuth ``atan2(y, x)`` across the width and elevation down the
    height, with the top row looking straight up.
    """
    height, width = equirect_rgb.shape[:2]
    coord = (np.arange(face_size) + 0.5) / face_size * 2.0 - 1.0
    a, b = np.meshgrid(coord, coord)

    faces = []
    for name in _FACE_ORDER:
        direction = _face_direction(name, a, b)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        lon = np.arctan2(direction[..., 1], direction[..., 0])
        lat = np.arcsin(np.clip(direction[..., 2], -1.0, 1.0))
        px = np.clip(((lon + np.pi) / (2.0 * np.pi) * width).astype(int), 0, width - 1)
        py = np.clip(((np.pi / 2.0 - lat) / np.pi * height).astype(int), 0, height - 1)
        faces.append(equirect_rgb[py, px])

    return np.concatenate(faces, axis=0).astype(np.uint8)


def add_background_panorama(
    spec: mujoco.MjSpec,
    panorama: Path | str | np.ndarray,
    *,
    face_size: int | None = None,
) -> None:
    """Add the room panorama to *spec* as a skybox.

    *panorama* is an equirectangular image path. It is resampled into cube faces
    and attached as skybox texture data (no asset files are written).
    """
    if isinstance(panorama, np.ndarray):
        if panorama.ndim != 3 or panorama.shape[2] != 3:
            raise ValueError("panorama array must have shape (height, width, 3)")
        rgb = np.asarray(panorama, dtype=np.uint8)
        face_size = face_size or 256
    else:
        bgr = cv2.imread(str(panorama))
        if bgr is None:
            raise FileNotFoundError(f"could not read panorama image: {panorama}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_size = face_size or 1024
    data = equirect_to_skybox(rgb, face_size)

    texture = spec.add_texture(
        name=SKYBOX_NAME,
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        width=face_size,
        height=6 * face_size,
        nchannel=3,
    )
    texture.data = data.tobytes()
