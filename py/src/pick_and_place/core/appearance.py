# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One draw of everything that is only pixels.

The companion to :mod:`pick_and_place.core.miscalibration`, and its opposite. A
miscalibration draw exists to make the expert do something different; this one
exists to make the same behavior look different. Change any value here and the
correct action is unchanged, which is what lets it be drawn long after an
episode has been recorded, as many times as wanted.

It is a plain record so that it can travel: the simulator applies it to a
compiled scene, the trajectory artifact stores the one an episode's pixels were
actually made with, and a variant pass draws fresh ones. None of those can
import each other, and none of them needs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Scene material families a draw tints. The last two are the episode's own
#: markers — the cube and the drop plate — which are placed after the scene is
#: painted and so are tinted separately.
MATERIAL_FAMILIES = (
    "plastic",
    "environment_plastic",
    "motor",
    "camera",
    "mdf",
    "groundplane",
    "cube",
    "target",
)

_TUPLE_FIELDS = (
    "key_light_position",
    "key_light_target",
    "overhead_camera_position_m",
    "overhead_camera_rotation_deg",
    "background_rgb",
    "table_rgb",
    "white_balance",
)


@dataclass(frozen=True)
class AppearanceDraw:
    """Lighting, materials, viewpoint, surfaces and camera response, all drawn together.

    ``seed`` keys the per-frame noise stream, so a rendered episode's grain is a
    function of the draw rather than of when it happened to be rendered.
    """

    seed: int
    light_intensity: float
    light_warm_cool: float
    key_light_position: tuple[float, float, float]
    key_light_target: tuple[float, float, float]
    key_light_bulb_radius: float
    fill_light_intensity: float
    material_factors: dict[str, tuple[float, float, float]]
    overhead_camera_position_m: tuple[float, float, float]
    overhead_camera_rotation_deg: tuple[float, float, float]
    overhead_camera_focal_scale: float
    appearance_seed: int
    appearance_mode: str
    background_rgb: tuple[float, float, float]
    table_rgb: tuple[float, float, float]
    appearance_blur_sigma: float
    appearance_blob_count: int
    exposure: float
    gamma: float
    white_balance: tuple[float, float, float]
    noise_sigma: float
    blur_sigma: float

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": int(self.seed),
            "light_intensity": float(self.light_intensity),
            "light_warm_cool": float(self.light_warm_cool),
            "key_light_bulb_radius": float(self.key_light_bulb_radius),
            "fill_light_intensity": float(self.fill_light_intensity),
            "material_factors": {
                name: [float(value) for value in factor]
                for name, factor in self.material_factors.items()
            },
            "overhead_camera_focal_scale": float(self.overhead_camera_focal_scale),
            "appearance_seed": int(self.appearance_seed),
            "appearance_mode": str(self.appearance_mode),
            "appearance_blur_sigma": float(self.appearance_blur_sigma),
            "appearance_blob_count": int(self.appearance_blob_count),
            "exposure": float(self.exposure),
            "gamma": float(self.gamma),
            "noise_sigma": float(self.noise_sigma),
            "blur_sigma": float(self.blur_sigma),
        }
        for name in _TUPLE_FIELDS:
            payload[name] = [float(value) for value in getattr(self, name)]
        return payload

    @staticmethod
    def from_json(payload: dict[str, Any]) -> AppearanceDraw:
        fields = dict(payload)
        fields["material_factors"] = {
            str(name): tuple(float(value) for value in factor)
            for name, factor in payload["material_factors"].items()
        }
        for name in _TUPLE_FIELDS:
            fields[name] = tuple(float(value) for value in payload[name])
        return AppearanceDraw(**fields)
