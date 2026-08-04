# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Named material system for the robot generator.

Two canonical materials cover the whole robot:
  - ``plastic``   — 3-D-printed robot parts, default grey
  - ``environment_plastic`` — 3-D-printed workspace/camera-mount parts
  - ``motor``     — STS-3215 servo bodies, default near-black

A fixed ``collision`` material (debug green, semi-transparent) is always
created for group-3 geoms and is not user-configurable.

Global colours are set via :class:`MaterialConfig`.  Per-mesh overrides map a
mesh name (without extension, e.g. ``"upper_arm_so101_v1"``) to a material
name (``"plastic"``, ``"motor"``, ``"environment_plastic"``, or a key in
``MaterialConfig.custom``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

# Printed parts are white PLA; sampled from real footage they read as a
# neutral-to-slightly-cool white rather than the flat mid-grey used before.
PLASTIC_RGBA: tuple[float, float, float, float] = (0.85, 0.86, 0.88, 1.0)
ENVIRONMENT_PLASTIC_RGBA: tuple[float, float, float, float] = (0.62, 0.62, 0.62, 1.0)
# STS-3215 servo bodies are near-black glossy plastic.
MOTOR_RGBA: tuple[float, float, float, float] = (0.11, 0.11, 0.12, 1.0)
CAMERA_RGBA: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 1.0)
MDF_RGBA: tuple[float, float, float, float] = (0.72, 0.66, 0.54, 1.0)
_COLLISION_RGBA: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 0.5)

_MOTOR_THRESHOLD = 0.3  # all RGB channels below this → classify as motor


@dataclass(frozen=True)
class Finish:
    """Specular finish of a material: highlight strength, tightness, mirror-ness."""

    specular: float = 0.0
    shininess: float = 0.0
    reflectance: float = 0.0


# Per-material surface finish. The printed PLA parts have a soft satin sheen; the
# servo and camera bodies are glossy black plastic; MDF and the workspace-frame
# plastic are near-matte. Highlights only appear when the scene lights emit
# specular (see ``scene._add_scene_lighting``).
_FINISHES: dict[str, Finish] = {
    "plastic": Finish(specular=0.2, shininess=0.35),
    "environment_plastic": Finish(specular=0.12, shininess=0.25),
    "motor": Finish(specular=0.5, shininess=0.6),
    "camera": Finish(specular=0.6, shininess=0.7),
    "mdf": Finish(specular=0.05, shininess=0.15),
    "collision": Finish(),
}


def _apply_finish(material: mujoco.MjsMaterial, name: str) -> None:
    finish = _FINISHES.get(name, Finish())
    material.specular = finish.specular
    material.shininess = finish.shininess
    material.reflectance = finish.reflectance


@dataclass
class MaterialConfig:
    plastic: tuple[float, float, float, float] = PLASTIC_RGBA
    environment_plastic: tuple[float, float, float, float] = ENVIRONMENT_PLASTIC_RGBA
    motor: tuple[float, float, float, float] = MOTOR_RGBA
    camera: tuple[float, float, float, float] = CAMERA_RGBA
    mdf: tuple[float, float, float, float] = MDF_RGBA
    custom: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    # mesh name (no extension) → material name ('plastic', 'environment_plastic',
    # 'motor', 'mdf', 'camera', or key in custom)
    mesh_overrides: dict[str, str] = field(default_factory=dict)

    def rgba_for(self, name: str) -> tuple[float, float, float, float]:
        if name == "plastic":
            return self.plastic
        if name == "environment_plastic":
            return self.environment_plastic
        if name == "motor":
            return self.motor
        if name == "camera":
            return self.camera
        if name == "mdf":
            return self.mdf
        if name == "collision":
            return _COLLISION_RGBA
        return self.custom[name]


def _classify(rgba: tuple[float, ...]) -> str:
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "motor" if r < _MOTOR_THRESHOLD and g < _MOTOR_THRESHOLD and b < _MOTOR_THRESHOLD else "plastic"


def apply_materials(spec: mujoco.MjSpec, config: MaterialConfig) -> None:
    """Replace all spec materials with the consolidated set from *config*.

    Every geom gets a named material:
    - group-2 (visual): classified as ``plastic`` or ``motor`` from current
      colour, with optional per-mesh overrides.
    - group-3 (collision): assigned the fixed ``collision`` material.

    All previous vendor material definitions are removed so the spec contains
    only the materials actually in use.
    """
    # Snapshot current material rgbas before we delete anything.
    current: dict[str, tuple[float, ...]] = {
        mat.name: tuple(mat.rgba) for mat in spec.materials
    }

    needed: set[str] = set()

    for geom in spec.geoms:
        group = int(geom.group)

        if group == 3:
            geom.material = "collision"
            needed.add("collision")
            continue

        if group != 2:
            continue

        mesh_name: str = geom.meshname or ""
        explicit_material = False
        if mesh_name in config.mesh_overrides:
            mat_name = config.mesh_overrides[mesh_name]
            explicit_material = True
        elif geom.material and geom.material in (
            "plastic",
            "environment_plastic",
            "motor",
            "mdf",
            "camera",
        ):
            # Respect explicit assignments already in the spec.
            mat_name = geom.material
            explicit_material = True
        else:
            existing_mat = geom.material or ""
            if existing_mat and existing_mat in current:
                rgba = current[existing_mat]
            else:
                rgba = tuple(geom.rgba)
            mat_name = _classify(rgba)

        needed.add(mat_name)
        geom.material = mat_name

        # MuJoCo keeps a geom-local RGBA even when a material is assigned, and
        # alpha=0 makes the geom invisible. For automatically classified geoms,
        # mirror the resolved material color locally so compiled viewers and
        # exporters see the same visible color.
        if not explicit_material:
            geom.rgba = list(config.rgba_for(mat_name))

    # Remove all old materials.
    for mat in list(spec.materials):
        spec.delete(mat)

    # Create only the materials that are actually referenced, in stable order.
    for name in ("plastic", "environment_plastic", "motor", "camera", "mdf", "collision"):
        if name in needed:
            mat = spec.add_material()
            mat.name = name
            mat.rgba = list(config.rgba_for(name))
            _apply_finish(mat, name)

    for name in config.custom:
        if name in needed:
            mat = spec.add_material()
            mat.name = name
            mat.rgba = list(config.rgba_for(name))
            _apply_finish(mat, name)
