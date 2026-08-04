# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Recolour a compiled pick-and-place scene without touching its geometry.

The scene's appearance is the one axis of a recording that can be changed after
the fact: object segmentation, physics and kinematics do not depend on it, so
the same recorded ground truth can be rendered under any of these palettes and
the results differ in pixels only. That is what
``py/scripts/sweep_scene_appearance.py`` measures over and what
``py/scripts/rerender_episodes.py`` writes out as new datasets.

Every field of :class:`SceneAppearance` may be ``None``, meaning "leave whatever
the compiled model carries". :data:`AS_RECORDED` — all fields ``None`` — is
therefore the appearance a recording already has, and re-rendering with it is
the identity that the re-renderer's verification mode checks against the
recorded video.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from pick_and_place.paper_detection import PAPER_TARGET_MARKER_NAME

FLOOR_GEOM = "floor"
FLOOR_MATERIAL = "groundplane"
CUBE_GEOM = "pick_cube"
PLATE_GEOM = f"{PAPER_TARGET_MARKER_NAME}_geom"
# The workspace frame's AprilTag calibration plates: white tagged squares at
# roughly the cube's apparent scale, and the scene's nearest visual confuser for
# it. They exist only to calibrate the overhead camera.
FRAME_TAG_GEOM_PREFIX = "workspace_frame_tag_"

#: Candidate floor colours. The floor must stay achromatic so it does not
#: compete with a coloured target; ``tan`` is the current scene and the
#: baseline every other row is measured against.
FLOOR_COLOURS: dict[str, tuple[float, float, float]] = {
    "tan": (0.82, 0.74, 0.60),
    "mid-gray": (0.30, 0.30, 0.30),
    "dark-gray": (0.15, 0.15, 0.15),
    # Dark enough to read as black, but not so close to zero reflectance that
    # the scene's lighting gradient (visible on every other floor colour)
    # disappears into a flat, unlit-looking void -- multiplied against a
    # table texture, an even darker value reads as solid black regardless.
    # Lighter than "dark-gray" in raw albedo, which looks odd on paper but not
    # in the rendered image: the two are named for how they read, not ranked.
    "black": (0.20, 0.20, 0.20),
}

#: Candidate drop-zone target colours. ``black`` is the current scene and the
#: only colour the real detector finds by darkness; ``white`` is the one
#: alternative ``detect_paper_target`` already supports. The saturated colours
#: each need a new hue branch in that detector before they can be used on
#: hardware.
TARGET_COLOURS: dict[str, tuple[float, float, float]] = {
    "black": (0.12, 0.12, 0.12),
    "white": (0.95, 0.95, 0.95),
    "yellow": (0.95, 0.80, 0.10),
    "orange": (0.95, 0.45, 0.05),
    "red": (0.85, 0.10, 0.08),
}

#: Candidate cube appearances. ``apriltag`` is the current cube and the only one
#: the descent visual servo and the real-robot cube tracker can use, so a solid
#: colour is only reachable by re-rendering episodes recorded with the tags.
#: Under the scene's warm key light the blue and green channels are heavily
#: attenuated, so those two render dark; red keeps its brightness.
CUBE_COLOURS: dict[str, tuple[float, float, float] | None] = {
    "apriltag": None,
    "red": (0.82, 0.12, 0.08),
    "orange": (0.95, 0.45, 0.05),
    "yellow": (0.95, 0.80, 0.10),
    "blue": (0.10, 0.20, 0.85),
    "green": (0.10, 0.65, 0.20),
}

#: Name used in variant labels for a field left exactly as the model compiled it.
AS_RECORDED_LABEL = "recorded"


@dataclass(frozen=True)
class SceneAppearance:
    """One combination of floor, target, cube and frame-tag appearance.

    ``None`` on any field means "as compiled", which for a recorded episode
    means "as recorded". ``cube="apriltag"`` restores the tagged material
    explicitly and is equivalent to ``cube=None`` on the standard scene.
    """

    floor: str | None = None
    target: str | None = None
    cube: str | None = None
    frame_tags: bool | None = None

    def __post_init__(self) -> None:
        for field_name, table in (
            ("floor", FLOOR_COLOURS),
            ("target", TARGET_COLOURS),
            ("cube", CUBE_COLOURS),
        ):
            value = getattr(self, field_name)
            if value is not None and value not in table:
                raise ValueError(
                    f"unknown {field_name} appearance {value!r}; "
                    f"expected one of {sorted(table)}"
                )

    @property
    def name(self) -> str:
        """Stable label, matching the scene-appearance sweep's variant names."""
        tags = (
            AS_RECORDED_LABEL
            if self.frame_tags is None
            else ("tags" if self.frame_tags else "notags")
        )
        return (
            f"floor-{self.floor or AS_RECORDED_LABEL}"
            f"_target-{self.target or AS_RECORDED_LABEL}"
            f"_cube-{self.cube or AS_RECORDED_LABEL}"
            f"_frame-{tags}"
        )

    @property
    def is_as_recorded(self) -> bool:
        """Whether applying this leaves every rendered pixel as recorded."""
        return self == AS_RECORDED

    def describe(self) -> dict[str, object]:
        """JSON-serializable record of the colours this resolves to."""
        return {
            "floor": self.floor,
            "floor_rgb": list(FLOOR_COLOURS[self.floor]) if self.floor else None,
            "target": self.target,
            "target_rgb": list(TARGET_COLOURS[self.target]) if self.target else None,
            "cube": self.cube,
            "cube_rgb": (
                list(CUBE_COLOURS[self.cube])
                if self.cube and CUBE_COLOURS[self.cube] is not None
                else None
            ),
            "frame_tags": self.frame_tags,
        }


AS_RECORDED = SceneAppearance()

#: Named appearances. ``as-recorded`` changes nothing. The rest were selected
#: by sweeping floor and drop-zone colours and measuring what survives the
#: policy's 96x96 view: recolouring the cube, or darkening the floor instead. A dark floor needs the white target — the black one is found by
#: local darkness and all but vanishes against it (plate contrast 0.24 on black,
#: against 0.64 with a white target).
#:
#: Overhead cube contrast during acquisition, medians over 3 episodes:
#: as-recorded 0.159, blue-cube 0.528, gray-floor 0.379, black-floor 0.598.
#: The floor variants keep the tagged cube, so they inherit its poor
#: separability from the white arm (~49 against the blue cube's ~196).
APPEARANCE_PRESETS: dict[str, SceneAppearance] = {
    "as-recorded": AS_RECORDED,
    "blue-cube": SceneAppearance(cube="blue"),
    "black-floor": SceneAppearance(floor="black", target="white"),
    "gray-floor": SceneAppearance(floor="dark-gray", target="white"),
}


def parse_appearance(token: str) -> tuple[str, SceneAppearance]:
    """Parse ``NAME`` from :data:`APPEARANCE_PRESETS`, or a ``key=value,...`` spec.

    Returns the appearance together with a stable label for it, which is the
    preset name for a preset and a field summary for an ad-hoc spec. Raises
    ``ValueError`` on an unknown name, field or colour.
    """
    if "=" not in token:
        if token not in APPEARANCE_PRESETS:
            raise ValueError(
                f"unknown appearance {token!r}; expected one of {sorted(APPEARANCE_PRESETS)} "
                "or a spec like 'cube=blue,floor=dark-gray'"
            )
        return token, APPEARANCE_PRESETS[token]

    fields: dict[str, str | bool] = {}
    for item in token.split(","):
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key == "frame_tags":
            fields[key] = value.lower() in ("1", "true", "on", "yes")
        elif key in ("floor", "target", "cube"):
            fields[key] = value
        else:
            raise ValueError(
                f"unknown appearance field {key!r} in {token!r}; "
                "expected floor, target, cube or frame_tags"
            )
    name = "_".join(f"{key}-{fields[key]}" for key in sorted(fields))
    return name, SceneAppearance(**fields)  # type: ignore[arg-type]


class SceneAppearanceOverride:
    """Repaint an already compiled model's floor, target, cube and frame tags.

    Every controlled colour is snapshotted at construction, so a field left
    ``None`` restores the compiled scene rather than keeping whatever the
    previous variant painted — which is what makes rendering several variants
    of the same frame independent of the order they are applied in.

    That snapshot is deliberately taken once and never retaken: refreshing it
    later would capture the *previous* variant's paint as if it were the
    recorded scene, which silently turns "as recorded" into "as last
    rendered". The one exception is the drop-zone target, which
    ``place_paper_target_marker`` legitimately repaints per episode (it always
    writes the geom's RGBA, so reading it straight after a placement is safe);
    :meth:`refresh_plate_baseline` re-reads exactly that one colour.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._floor_geom = _geom_id(model, FLOOR_GEOM)
        self._floor_material = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, FLOOR_MATERIAL
        )
        self._plate_geom = _geom_id(model, PLATE_GEOM)
        self._cube_geom = _geom_id(model, CUBE_GEOM)
        self._frame_tag_geoms = sorted(frame_tag_geom_ids(model))
        self._compiled: dict[str, object] = {
            "floor_rgba": tuple(model.geom_rgba[self._floor_geom]),
            "floor_mat_rgba": (
                tuple(model.mat_rgba[self._floor_material])
                if self._floor_material >= 0
                else None
            ),
            "cube_rgba": tuple(model.geom_rgba[self._cube_geom]),
            "cube_matid": int(model.geom_matid[self._cube_geom]),
            "frame_tag_rgba": {
                geom_id: tuple(model.geom_rgba[geom_id]) for geom_id in self._frame_tag_geoms
            },
        }
        self._plate_rgba = tuple(model.geom_rgba[self._plate_geom])

    def refresh_plate_baseline(self) -> None:
        """Adopt the drop-zone marker's current colour as its recorded one.

        Call after every ``place_paper_target_marker``: that call decides the
        episode's plate colour (black inside the drop zone, orange outside),
        which is what ``target=None`` must restore.
        """
        self._plate_rgba = tuple(self._model.geom_rgba[self._plate_geom])

    def apply(self, appearance: SceneAppearance) -> None:
        """Paint ``appearance`` over the compiled scene's colours."""
        model = self._model
        baseline = self._compiled

        if appearance.floor is None:
            model.geom_rgba[self._floor_geom] = baseline["floor_rgba"]
            if self._floor_material >= 0:
                model.mat_rgba[self._floor_material] = baseline["floor_mat_rgba"]
        else:
            floor = FLOOR_COLOURS[appearance.floor]
            # The floor renders through the ``groundplane`` material, so both the
            # material and the geom-local RGBA have to be set.
            model.geom_rgba[self._floor_geom] = (*floor, 1.0)
            if self._floor_material >= 0:
                model.mat_rgba[self._floor_material] = (*floor, 1.0)

        if appearance.target is None:
            model.geom_rgba[self._plate_geom] = self._plate_rgba
        else:
            # Overrides the usable/not-usable signalling of
            # place_paper_target_marker; recorded targets are inside the drop zone.
            model.geom_rgba[self._plate_geom] = (*TARGET_COLOURS[appearance.target], 1.0)

        cube = None if appearance.cube is None else CUBE_COLOURS[appearance.cube]
        if cube is None:
            model.geom_matid[self._cube_geom] = baseline["cube_matid"]
            model.geom_rgba[self._cube_geom] = baseline["cube_rgba"]
        else:
            # Detaching the AprilTag material is what turns the tagged cube into a
            # solid colour; geom_rgba only governs once no material is bound.
            model.geom_matid[self._cube_geom] = -1
            model.geom_rgba[self._cube_geom] = (*cube, 1.0)

        for geom_id in self._frame_tag_geoms:
            rgba = baseline["frame_tag_rgba"][geom_id]
            if appearance.frame_tags is False:
                # Peeling the calibration stickers off the frame is modelled by
                # making the plate geoms fully transparent, leaving the frame
                # surface behind.
                model.geom_rgba[geom_id] = (*rgba[:3], 0.0)
            else:
                model.geom_rgba[geom_id] = rgba


def frame_tag_geom_ids(model: mujoco.MjModel) -> set[int]:
    """Geom ids of the workspace frame's AprilTag calibration plates."""
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            FRAME_TAG_GEOM_PREFIX
        )
    }


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"scene is missing the {name!r} geom")
    return geom_id
