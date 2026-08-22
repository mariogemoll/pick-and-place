# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The envelope a randomized episode is drawn from, and the half of it that shapes behavior.

A preset describes one envelope over every axis a run may vary, and
:meth:`DomainRandomizationPreset.sample` draws one episode from it. The draw is
deliberately whole: the axes are described in one file because they belong to
one experiment, and splitting the *description* would only make an envelope
harder to read.

What the draw is used for does split, along the one line that matters — if I
change this, does the correct action change?

* **Yes**, so it has to be applied while the trajectory is generated: the wrist
  camera's mount error (:class:`WristMountRandomizer`), which the expert servos
  through; the cube's resting orientation (:func:`orient_cube`), which decides
  what there is to grasp; and the miscalibration draw, which is what the whole
  closed loop exists to correct.
* **No**, so it can be applied to an already-recorded episode as often as
  wanted: lighting, materials, the background, the overhead viewpoint and the
  camera response. Those live in :mod:`pick_and_place.variants`.
"""

from __future__ import annotations

import colorsys
import dataclasses
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.sim.background_panorama import equirect_to_skybox
from pick_and_place.sim.camera_pose_envelope import (
    CameraJitter,
    draw_camera_jitter,
    draw_overhead_camera_jitter,
    set_camera_jitter,
    snapshot_camera,
)
from pick_and_place.core.appearance import MATERIAL_FAMILIES, AppearanceDraw
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.miscalibration import MiscalibrationDraw, MiscalibrationModel

WRIST_CAMERA = "wrist_camera"

_RANGE_FIELDS = {
    "light_intensity",
    "light_warm_cool",
    "key_light_azimuth_deg",
    "key_light_elevation_deg",
    "key_light_distance_m",
    "key_light_bulb_radius_m",
    "fill_light_intensity",
    "material_brightness",
    "material_tint",
    "background_hue_deg",
    "background_value",
    "background_saturation",
    "table_hue_deg",
    "table_value",
    "table_saturation",
    "colorful_background_hue_deg",
    "colorful_background_saturation",
    "colorful_table_hue_deg",
    "colorful_table_saturation",
    "appearance_blur_sigma",
    "exposure",
    "gamma",
    "white_balance",
    "noise_sigma",
    "blur_sigma",
}
_SCALAR_FIELDS = {
    "key_light_target_jitter_m",
    "overhead_camera_position_mm",
    "overhead_camera_rotation_deg",
    "overhead_camera_frame_tag_margin_px",
    "overhead_camera_focal_pct",
    "wrist_camera_position_mm",
    "wrist_camera_rotation_deg",
    "colorful_appearance_probability",
}
_REQUIRED = {"name", "cube_orientations", "appearance_blob_count"} | _RANGE_FIELDS | _SCALAR_FIELDS


def domain_seed(root_seed: int | None, episode_index: int) -> int:
    """Derive a stable randomization seed from a run seed and episode index."""
    if root_seed is None:
        return int(np.random.default_rng().integers(2**63))
    return int(
        np.random.default_rng(np.random.SeedSequence([root_seed, episode_index, 0xD0A1])).integers(
            2**63
        )
    )


def reload_renderer_textures(renderer: mujoco.Renderer, texture_ids: tuple[int, ...]) -> None:
    """Upload changed ``model.tex_data`` into one renderer's GL context."""
    if not texture_ids:
        return
    if renderer._mjr_context is None:
        raise RuntimeError("cannot reload textures after closing the renderer")
    if renderer._gl_context:
        renderer._gl_context.make_current()
    for texture_id in texture_ids:
        mujoco.mjr_uploadTexture(renderer.model, renderer._mjr_context, texture_id)


def _range(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(x, (int, float)) for x in value)
    ):
        raise ValueError(f"{name} must be a two-number array")
    low, high = map(float, value)
    if low > high:
        raise ValueError(f"{name} must be ordered")
    return low, high


def _int_range(value: Any, name: str) -> tuple[int, int]:
    low, high = _range(value, name)
    if not low.is_integer() or not high.is_integer() or low < 1:
        raise ValueError(f"{name} must contain positive integers")
    return int(low), int(high)


@dataclass(frozen=True)
class DomainRandomizationPreset:
    name: str
    ranges: dict[str, tuple[float, float]]
    scalars: dict[str, float]
    appearance_blob_count: tuple[int, int]

    @classmethod
    def load(cls, path: Path) -> "DomainRandomizationPreset":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or set(payload) != _REQUIRED:
            unknown = set(payload) - _REQUIRED if isinstance(payload, dict) else set()
            missing = _REQUIRED - set(payload) if isinstance(payload, dict) else _REQUIRED
            raise ValueError(
                f"invalid domain-randomization preset; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if not isinstance(payload["name"], str) or not payload["name"]:
            raise ValueError("name must be a nonempty string")
        if payload["cube_orientations"] != "all_24":
            raise ValueError("cube_orientations must be 'all_24'")
        ranges = {name: _range(payload[name], name) for name in _RANGE_FIELDS}
        scalars = {name: float(payload[name]) for name in _SCALAR_FIELDS}
        if any(value < 0.0 for value in scalars.values()):
            raise ValueError("scalar preset values must be nonnegative")
        if scalars["colorful_appearance_probability"] > 1.0:
            raise ValueError("colorful_appearance_probability must be in [0, 1]")
        return cls(
            name=payload["name"],
            ranges=ranges,
            scalars=scalars,
            appearance_blob_count=_int_range(
                payload["appearance_blob_count"], "appearance_blob_count"
            ),
        )

    def sample(self, episode_seed: int) -> "DomainSample":
        rng = np.random.default_rng(episode_seed)

        def draw(name: str) -> float:
            return float(rng.uniform(*self.ranges[name]))

        # Focal length is drawn before the pose, because a narrower field of
        # view pulls the workspace-frame tags outward and so changes which
        # poses are solvable.
        overhead_position, overhead_rotation, overhead_focal_scale = draw_overhead_camera_jitter(
            rng,
            position_mm=self.scalars["overhead_camera_position_mm"],
            rotation_deg=self.scalars["overhead_camera_rotation_deg"],
            focal_pct=self.scalars["overhead_camera_focal_pct"],
            margin_px=self.scalars["overhead_camera_frame_tag_margin_px"],
        )
        wrist_position, wrist_rotation = draw_camera_jitter(
            rng, self.scalars["wrist_camera_position_mm"], self.scalars["wrist_camera_rotation_deg"]
        )

        target = rng.uniform(
            -self.scalars["key_light_target_jitter_m"],
            self.scalars["key_light_target_jitter_m"],
            size=2,
        )
        azimuth = math.radians(draw("key_light_azimuth_deg"))
        elevation = math.radians(draw("key_light_elevation_deg"))
        distance = draw("key_light_distance_m")
        key_target = np.array((target[0], target[1], 0.0))
        key_position = key_target + distance * np.array(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            )
        )

        factors = {}
        for family in MATERIAL_FAMILIES:
            brightness = draw("material_brightness")
            tint = rng.uniform(*self.ranges["material_tint"], size=3)
            factors[family] = tuple(float(x) for x in brightness * tint)

        appearance_seed = int(rng.integers(2**63))
        colorful_appearance = rng.random() < self.scalars["colorful_appearance_probability"]
        appearance_mode = "colorful" if colorful_appearance else "realistic"
        color_prefix = "colorful_" if colorful_appearance else ""
        return DomainSample(
            seed=episode_seed,
            light_intensity=draw("light_intensity"),
            light_warm_cool=draw("light_warm_cool"),
            key_light_position=tuple(float(x) for x in key_position),
            key_light_target=tuple(float(x) for x in key_target),
            key_light_bulb_radius=_draw_log_uniform(rng, self.ranges["key_light_bulb_radius_m"]),
            fill_light_intensity=draw("fill_light_intensity"),
            material_factors=factors,
            overhead_camera_position_m=overhead_position,
            overhead_camera_rotation_deg=overhead_rotation,
            overhead_camera_focal_scale=overhead_focal_scale,
            wrist_camera_position_m=wrist_position,
            wrist_camera_rotation_deg=wrist_rotation,
            cube_orientation_index=int(rng.integers(24)),
            appearance_seed=appearance_seed,
            appearance_mode=appearance_mode,
            background_rgb=_sample_color(
                rng,
                self.ranges[f"{color_prefix}background_hue_deg"],
                self.ranges[f"{color_prefix}background_saturation"],
                self.ranges["background_value"],
            ),
            table_rgb=_sample_color(
                rng,
                self.ranges[f"{color_prefix}table_hue_deg"],
                self.ranges[f"{color_prefix}table_saturation"],
                self.ranges["table_value"],
            ),
            appearance_blur_sigma=draw("appearance_blur_sigma"),
            appearance_blob_count=int(
                rng.integers(self.appearance_blob_count[0], self.appearance_blob_count[1] + 1)
            ),
            exposure=draw("exposure"),
            gamma=draw("gamma"),
            white_balance=tuple(
                float(x) for x in rng.uniform(*self.ranges["white_balance"], size=3)
            ),
            noise_sigma=draw("noise_sigma"),
            blur_sigma=draw("blur_sigma"),
            miscalibration=MiscalibrationModel().sample(rng),
        )


def _sample_color(
    rng: np.random.Generator,
    hue_deg: tuple[float, float],
    saturation: tuple[float, float],
    value: tuple[float, float],
) -> tuple[float, float, float]:
    hue = rng.uniform(*hue_deg) / 360.0
    return tuple(
        float(x) for x in colorsys.hsv_to_rgb(hue, rng.uniform(*saturation), rng.uniform(*value))
    )


def _draw_log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low <= 0.0:
        raise ValueError("log-uniform bounds must be positive")
    return float(np.exp(rng.uniform(math.log(low), math.log(high))))


@dataclass
class DomainSample:
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
    wrist_camera_position_m: tuple[float, float, float]
    wrist_camera_rotation_deg: tuple[float, float, float]
    cube_orientation_index: int
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
    miscalibration: MiscalibrationDraw

    def appearance(self) -> AppearanceDraw:
        """The half of this draw that is only pixels.

        Everything left behind — the wrist mount, the cube's resting
        orientation, the miscalibration — had to be applied while the trajectory
        was being generated, and is baked into the episode that resulted.
        """
        return AppearanceDraw(
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(AppearanceDraw)
            }
        )

    def metadata_json(self) -> str:
        payload = {name: value for name, value in self.__dict__.items() if name != "miscalibration"}
        payload["miscalibration"] = {
            "base_offsets_deg": self.miscalibration.base_offsets_deg,
            "cube_belief_error": self.miscalibration.cube_belief_error,
            "target_belief_error": self.miscalibration.target_belief_error,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


#: Fields of a serialized ``DomainSample`` that are JSON arrays and have to come
#: back as tuples, since the dataclass is compared and copied by value.
_SAMPLE_TUPLE_FIELDS = (
    "key_light_position",
    "key_light_target",
    "overhead_camera_position_m",
    "overhead_camera_rotation_deg",
    "wrist_camera_position_m",
    "wrist_camera_rotation_deg",
    "background_rgb",
    "table_rgb",
    "white_balance",
)


def domain_sample_fields() -> set[str]:
    """The names a serialized ``DomainSample`` carries, ``miscalibration`` aside."""
    return {field.name for field in dataclasses.fields(DomainSample)} - {"miscalibration"}


def domain_sample_payload(sample: DomainSample) -> dict[str, Any]:
    """A drawn sample as the block a scenario carries, ready for the inverse below.

    ``miscalibration`` is left out because a scenario serializes it separately:
    it is the half that had to be applied while the trajectory was generated,
    not while it is rendered.

    The values come out unrounded, which is what a caller holding a live draw
    wants -- the round trip through :func:`domain_sample_from_payload` returns an
    equal sample. Writing a manifest is the other case, and
    ``pap generate-scenario-manifest`` rounds on top of this so a committed file
    stays readable and diffs cleanly.
    """
    return {
        name: value
        for name, value in sample.__dict__.items()
        if name != "miscalibration"
    }


def domain_sample_from_payload(
    payload: dict[str, Any],
    miscalibration: MiscalibrationDraw,
    *,
    context: str,
) -> DomainSample:
    """Rebuild a draw from a serialized ``domain_randomization_sample`` block.

    The block is every ``DomainSample`` field but ``miscalibration``, which is
    serialized separately because it is the half that had to be applied while
    the trajectory was generated. ``enabled`` has already been consumed by the
    caller, which is the only one that can say what a disabled block means.
    """
    expected = domain_sample_fields()
    if set(payload) != expected:
        raise ValueError(
            f"{context} has invalid domain sample fields; "
            f"missing={sorted(expected - set(payload))}, unknown={sorted(set(payload) - expected)}"
        )
    payload = dict(payload)
    payload["material_factors"] = {
        name: tuple(float(value) for value in factors)
        for name, factors in payload["material_factors"].items()
    }
    for name in _SAMPLE_TUPLE_FIELDS:
        payload[name] = tuple(float(value) for value in payload[name])
    return DomainSample(**payload, miscalibration=miscalibration)


@dataclass(frozen=True)
class ProceduralAppearance:
    background_rgb: np.ndarray
    table_rgb: np.ndarray


def write_procedural_textures(
    model: mujoco.MjModel, texture_ids: tuple[int, ...], appearance: ProceduralAppearance
) -> None:
    """Write a procedural background/table appearance into ``model.tex_data``.

    Shared by every path that paints the finite-floor + skybox scene, so all of
    them draw the same texture pipeline. The caller still has to push the change
    into a live GL context, e.g. via :func:`reload_renderer_textures`.
    """
    for texture_id in texture_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_id)
        width = int(model.tex_width[texture_id])
        height = int(model.tex_height[texture_id])
        channels = int(model.tex_nchannel[texture_id])
        address = int(model.tex_adr[texture_id])
        if name == "table_texture":
            rgb = cv2.resize(appearance.table_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
            rgb = np.rot90(rgb, k=-1).copy()
        else:
            rgb = equirect_to_skybox(appearance.background_rgb, width)
        flat = rgb[..., :channels].reshape(-1)
        model.tex_data[address : address + flat.size] = flat


def generate_procedural_appearance(
    sample: AppearanceDraw,
    *,
    background_size: tuple[int, int] = (128, 256),
    table_size: tuple[int, int] = (256, 256),
) -> ProceduralAppearance:
    """Generate repeatable, low-frequency RGB textures for one episode."""
    root = np.random.SeedSequence(sample.appearance_seed)
    background_rng, table_rng = (np.random.default_rng(seed) for seed in root.spawn(2))
    return ProceduralAppearance(
        background_rgb=_blurred_texture(
            background_rng,
            background_size,
            sample.background_rgb,
            sample.appearance_blur_sigma,
            sample.appearance_blob_count,
            variation=0.12,
        ),
        table_rgb=_blurred_texture(
            table_rng,
            table_size,
            sample.table_rgb,
            sample.appearance_blur_sigma,
            sample.appearance_blob_count,
            variation=0.05,
        ),
    )


def _blurred_texture(
    rng: np.random.Generator,
    size: tuple[int, int],
    base_rgb: tuple[float, float, float],
    blur_sigma: float,
    blob_count: int,
    *,
    variation: float,
) -> np.ndarray:
    height, width = size
    image = np.broadcast_to(np.asarray(base_rgb, dtype=np.float32), (height, width, 3)).copy()
    yy, xx = np.mgrid[:height, :width]
    for _ in range(blob_count):
        center_x = rng.uniform(0.0, width)
        center_y = rng.uniform(0.0, height)
        radius_x = rng.uniform(0.08, 0.35) * width
        radius_y = rng.uniform(0.08, 0.35) * height
        field = np.exp(
            -0.5 * (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2)
        )[..., None]
        luminance_delta = rng.normal(0.0, variation)
        chroma_delta = rng.normal(0.0, variation * 0.1, size=3)
        delta = luminance_delta + chroma_delta
        image += field * delta
    pad = max(1, math.ceil(blur_sigma * 3.0))
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
    padded = cv2.GaussianBlur(padded, (0, 0), blur_sigma)
    image = padded[pad:-pad, pad:-pad]
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def orient_cube(pose: CubePose, orientation_index: int) -> CubePose:
    """Apply one of the cube's 24 axis-aligned orientations after its sampled yaw."""
    if not 0 <= orientation_index < 24:
        raise ValueError("cube orientation index must be in [0, 24)")
    symmetry = _cube_rotations()[orientation_index]
    rotation = Rotation.from_euler("z", pose.yaw) * symmetry
    with warnings.catch_warnings():
        # Exact quarter-turn cube orientations include Euler gimbal-lock cases;
        # SciPy still returns a valid, equivalent representation for them.
        warnings.simplefilter("ignore", UserWarning)
        yaw, pitch, roll = rotation.as_euler("ZYX")
    return replace(pose, roll=float(roll), pitch=float(pitch), yaw=float(yaw))


def _cube_rotations() -> tuple[Rotation, ...]:
    rotations: list[Rotation] = []
    for matrix in Rotation.create_group("O").as_matrix():
        # The proper octahedral group is exactly the 24 rotational symmetries of a cube.
        rotations.append(Rotation.from_matrix(matrix))
    return tuple(rotations)


class WristMountRandomizer:
    """The wrist camera's physical mount error, applied to a compiled scene.

    This is the one camera displacement that is *not* appearance. The expert has
    to servo through it, and the correction it makes ends up in the recorded
    trajectory — which is exactly what puts recovery behavior into the dataset,
    and exactly why it cannot be added to an episode after the fact.

    The controller is not told about it: it maps detections through
    :attr:`believed_wrist_camera_pose`, the nominal mount the calibration says is
    there, so every solve inherits the full hand-eye error the way it does on
    hardware.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._base = snapshot_camera(model, WRIST_CAMERA)

    @property
    def believed_wrist_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The mount the controller believes in: the authored, unperturbed pose."""
        return self._base.pos.copy(), self._base.quat.copy()

    def reset(self) -> None:
        """Put the camera back on its authored mount."""
        set_camera_jitter(self.model, self._base, None)

    def apply(self, sample: DomainSample) -> None:
        """Displace the camera and its housing by this episode's drawn mount error."""
        set_camera_jitter(
            self.model,
            self._base,
            CameraJitter(
                position_m=sample.wrist_camera_position_m,
                rotation_deg=sample.wrist_camera_rotation_deg,
            ),
        )
