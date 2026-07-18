# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Deterministic visual and calibration randomization for sim recording."""

from __future__ import annotations

import colorsys
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

from pick_and_place.background_panorama import equirect_to_skybox
from pick_and_place.geometry import CubePose
from pick_and_place.miscalibration import MiscalibrationDraw, MiscalibrationModel

_MATERIAL_FAMILIES = (
    "plastic",
    "environment_plastic",
    "motor",
    "camera",
    "mdf",
    "groundplane",
    "cube",
    "target",
)

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
        np.random.default_rng(
            np.random.SeedSequence([root_seed, episode_index, 0xD0A1])
        ).integers(2**63)
    )


def reload_renderer_textures(
    renderer: mujoco.Renderer, texture_ids: tuple[int, ...]
) -> None:
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

        def camera_jitter(prefix: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
            position = rng.uniform(
                -self.scalars[f"{prefix}_camera_position_mm"],
                self.scalars[f"{prefix}_camera_position_mm"],
                size=3,
            ) / 1000.0
            rotation = rng.uniform(
                -self.scalars[f"{prefix}_camera_rotation_deg"],
                self.scalars[f"{prefix}_camera_rotation_deg"],
                size=3,
            )
            return tuple(float(x) for x in position), tuple(float(x) for x in rotation)

        overhead_position, overhead_rotation = camera_jitter("overhead")
        wrist_position, wrist_rotation = camera_jitter("wrist")

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
        for family in _MATERIAL_FAMILIES:
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
            key_light_bulb_radius=_draw_log_uniform(
                rng, self.ranges["key_light_bulb_radius_m"]
            ),
            fill_light_intensity=draw("fill_light_intensity"),
            material_factors=factors,
            overhead_camera_position_m=overhead_position,
            overhead_camera_rotation_deg=overhead_rotation,
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
        float(x)
        for x in colorsys.hsv_to_rgb(
            hue, rng.uniform(*saturation), rng.uniform(*value)
        )
    )


def _draw_log_uniform(
    rng: np.random.Generator, bounds: tuple[float, float]
) -> float:
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

    def metadata_json(self) -> str:
        payload = {name: value for name, value in self.__dict__.items() if name != "miscalibration"}
        payload["miscalibration"] = {
            "base_offsets_deg": self.miscalibration.base_offsets_deg,
            "cube_belief_error": self.miscalibration.cube_belief_error,
            "target_belief_error": self.miscalibration.target_belief_error,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProceduralAppearance:
    background_rgb: np.ndarray
    table_rgb: np.ndarray


def generate_procedural_appearance(
    sample: DomainSample,
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


class DomainRandomizer:
    """Restore a compiled model's canonical values before applying each sample."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._light_pos = model.light_pos.copy()
        self._light_dir = model.light_dir.copy()
        self._light_diffuse = model.light_diffuse.copy()
        self._light_ambient = model.light_ambient.copy()
        self._light_specular = model.light_specular.copy()
        self._light_castshadow = model.light_castshadow.copy()
        self._light_bulbradius = model.light_bulbradius.copy()
        self._headlight_diffuse = np.array(model.vis.headlight.diffuse)
        self._headlight_ambient = np.array(model.vis.headlight.ambient)
        self._headlight_specular = np.array(model.vis.headlight.specular)
        self._mat_rgba = model.mat_rgba.copy()
        self._geom_rgba = model.geom_rgba.copy()
        self._cam_pos = model.cam_pos.copy()
        self._cam_quat = model.cam_quat.copy()
        self._texture_ids = tuple(
            ident
            for name in ("table_texture", "background_panorama")
            if (ident := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)) >= 0
        )
        self._frame = 0
        self._image_rng_seed = 0

    @property
    def texture_ids(self) -> tuple[int, ...]:
        return self._texture_ids

    @property
    def believed_wrist_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        return self._cam_pos[camera].copy(), self._cam_quat[camera].copy()

    def apply(self, sample: DomainSample) -> None:
        model = self.model
        model.light_pos[:] = self._light_pos
        model.light_dir[:] = self._light_dir
        model.light_diffuse[:] = self._light_diffuse
        model.light_ambient[:] = self._light_ambient
        model.light_specular[:] = self._light_specular
        model.light_castshadow[:] = self._light_castshadow
        model.light_bulbradius[:] = self._light_bulbradius
        model.vis.headlight.diffuse = self._headlight_diffuse
        model.vis.headlight.ambient = self._headlight_ambient
        model.vis.headlight.specular = self._headlight_specular
        model.mat_rgba[:] = self._mat_rgba
        model.geom_rgba[:] = self._geom_rgba
        model.cam_pos[:] = self._cam_pos
        model.cam_quat[:] = self._cam_quat

        cool = np.array((1.0 / sample.light_warm_cool, 1.0, sample.light_warm_cool))
        fill = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "scene_light")
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "warm_spotlight")
        if fill >= 0:
            model.light_diffuse[fill] *= sample.fill_light_intensity
            model.light_ambient[fill] *= sample.fill_light_intensity
            model.light_specular[fill] *= sample.fill_light_intensity
            model.light_castshadow[fill] = False
        if key >= 0:
            model.light_pos[key] = sample.key_light_position
            direction = np.asarray(sample.key_light_target) - np.asarray(sample.key_light_position)
            model.light_dir[key] = direction / np.linalg.norm(direction)
            model.light_diffuse[key] = (
                np.mean(self._light_diffuse[key]) * sample.light_intensity * cool
            )
            model.light_ambient[key] = (
                np.mean(self._light_ambient[key]) * sample.light_intensity * cool
            )
            model.light_specular[key] = (
                np.mean(self._light_specular[key]) * sample.light_intensity * cool
            )
            model.light_castshadow[key] = True
            model.light_bulbradius[key] = sample.key_light_bulb_radius
            model.light_cutoff[key] = 80.0
            model.light_exponent[key] = 2.0
        model.vis.headlight.diffuse = self._headlight_diffuse * sample.fill_light_intensity
        model.vis.headlight.ambient = self._headlight_ambient * sample.fill_light_intensity
        model.vis.headlight.specular = self._headlight_specular * sample.fill_light_intensity

        for name in _MATERIAL_FAMILIES[:-2]:
            ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if ident >= 0:
                model.mat_rgba[ident, :3] = np.clip(
                    self._mat_rgba[ident, :3] * sample.material_factors[name], 0.0, 1.0
                )

        self._apply_camera(
            "overhead_camera",
            sample.overhead_camera_position_m,
            sample.overhead_camera_rotation_deg,
        )
        self._apply_camera(
            "wrist_camera", sample.wrist_camera_position_m, sample.wrist_camera_rotation_deg
        )
        self._apply_procedural_textures(sample)
        self._sample = sample
        self._frame = 0
        self._image_rng_seed = sample.seed

    def _apply_camera(
        self,
        name: str,
        position: tuple[float, float, float],
        rotation_deg: tuple[float, float, float],
    ) -> None:
        camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera < 0:
            return
        self.model.cam_pos[camera] += position
        base = Rotation.from_quat(self.model.cam_quat[camera][[1, 2, 3, 0]])
        delta = Rotation.from_euler("xyz", rotation_deg, degrees=True)
        quat = (delta * base).as_quat()
        self.model.cam_quat[camera] = quat[[3, 0, 1, 2]]

    def _apply_procedural_textures(self, sample: DomainSample) -> None:
        appearance = generate_procedural_appearance(sample)
        for texture_id in self._texture_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_id)
            width = int(self.model.tex_width[texture_id])
            height = int(self.model.tex_height[texture_id])
            channels = int(self.model.tex_nchannel[texture_id])
            address = int(self.model.tex_adr[texture_id])
            if name == "table_texture":
                rgb = cv2.resize(appearance.table_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
                rgb = np.rot90(rgb, k=-1).copy()
            else:
                rgb = equirect_to_skybox(appearance.background_rgb, width)
            flat = rgb[..., :channels].reshape(-1)
            self.model.tex_data[address : address + flat.size] = flat

    def tint_episode_markers(self) -> None:
        sample = getattr(self, "_sample", None)
        if sample is None:
            return
        for name, family in (("pick_cube", "cube"), ("paper_target_marker_geom", "target")):
            ident = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if ident >= 0:
                self.model.geom_rgba[ident, :3] = np.clip(
                    self.model.geom_rgba[ident, :3] * sample.material_factors[family], 0.0, 1.0
                )

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(
            np.random.SeedSequence([self._image_rng_seed, self._frame, 0x1A6E])
        )
        self._frame += 1
        sample = getattr(self, "_sample", None)
        if sample is None:
            return image
        result = image.astype(np.float32) * sample.exposure
        result = np.clip(result / 255.0, 0.0, 1.0) ** (1.0 / sample.gamma)
        result *= np.asarray(sample.white_balance)
        result = np.clip(result * 255.0, 0.0, 255.0)
        if sample.blur_sigma > 0:
            result = cv2.GaussianBlur(result, (0, 0), sample.blur_sigma)
        if sample.noise_sigma > 0:
            result += rng.normal(0.0, sample.noise_sigma, result.shape)
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
