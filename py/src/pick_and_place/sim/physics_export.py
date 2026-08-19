# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export the policy scene as a self-contained, asset-free MJCF.

The browser steps this scene with MuJoCo's WebAssembly build and draws it with
Three.js from the web manifest, so the exported model only has to be *right
about physics*. Everything that exists solely to be looked at comes out: the
visual meshes, the textures and materials they refer to, the drop-zone marker,
the lights and the cameras.

What that leaves is a single XML string with no external references at all,
which matters because the WebAssembly bindings load a model from a string and
have no filesystem to resolve assets against.

Stripping the visual geoms is only safe once the inertia they carry has been
written down. Every body in the stock SO-101 declares an explicit ``<inertial>``,
but three of the bodies this project adds do not -- ``wrist_camera_mount``,
``overhead_camera_mount`` and ``workspace_frame_frame`` -- and MuJoCo infers
their mass from the geometry attached to them. Deleting the meshes therefore
takes two thirds of the wrist camera mount's mass off the end of the arm, which
is a change in dynamics, not in appearance.

:func:`freeze_inertials` closes that hole by compiling the intact scene, reading
back what MuJoCo computed for every body, and writing it into the spec as an
explicit inertial before anything is removed. After that, geometry carries no
mass and deleting it cannot move the arm.
"""

from __future__ import annotations

import mujoco

from pick_and_place.sim.scene import build_scene

#: Geom groups MuJoCo's own viewer treats as visual-only in this project's scenes.
VISUAL_GROUPS = frozenset({2, 4})


def freeze_inertials(spec: mujoco.MjSpec) -> None:
    """Pin every body's computed inertia into the spec as an explicit inertial.

    Compiles the spec as it stands, reads the mass, centre of mass, principal
    axes and principal moments MuJoCo derived for each body, and writes them
    back. A body that already declared an inertial is unchanged by this; a body
    that inherited one from its geoms stops depending on them.
    """
    model = spec.compile()
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        body = spec.body(name)
        # A body that spelled its inertia out as a full matrix has to give that
        # up before the diagonal form can be set; MuJoCo rejects both at once,
        # and marks the full form absent with a leading NaN.
        body.fullinertia = [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]
        body.mass = float(model.body_mass[body_id])
        body.ipos = model.body_ipos[body_id].copy()
        body.iquat = model.body_iquat[body_id].copy()
        body.inertia = model.body_inertia[body_id].copy()
        body.explicitinertial = True


def _delete_visual_meshes(spec: mujoco.MjSpec) -> None:
    """Drop mesh geoms, then the mesh assets that no longer have a referent.

    Only meshes go. A visual *primitive* may be the same geom that produces a
    contact, and deleting one by group alone would quietly remove collision
    geometry along with the decoration.
    """
    for geom in [
        geom
        for geom in spec.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group in VISUAL_GROUPS
    ]:
        spec.delete(geom)
    for mesh in list(spec.meshes):
        spec.delete(mesh)


def _delete_textures(spec: mujoco.MjSpec) -> None:
    """Drop the textures, which are the only appearance assets that read a file.

    The materials stay. They are a handful of numbers each, they are what the
    geoms and the default classes name, and removing them means chasing every
    reference through the defaults to avoid leaving a dangling material id. What
    costs bytes and needs a filesystem is the texture images, and those go.
    """
    for material in spec.materials:
        material.textures = [""] * len(material.textures)
    for texture in list(spec.textures):
        spec.delete(texture)


def _delete_scenery(spec: mujoco.MjSpec) -> None:
    """Drop lights and cameras: nothing here is ever rendered."""
    for light in list(spec.lights):
        spec.delete(light)
    for camera in list(spec.cameras):
        spec.delete(camera)


def physics_only_spec() -> mujoco.MjSpec:
    """Compose the policy scene with everything that is only pixels removed.

    The scene itself is the one :func:`pick_and_place.runtime.policy_sim.build_policy_sim_model`
    compiles -- the standard environment plus a free-floating pick cube -- minus
    the drop-zone marker, which is a non-colliding square that exists to be seen.
    The target is an ``(x, y)`` the policy is conditioned on, not a body.

    The integration timestep is deliberately left alone here and set by the
    caller, matching what the reference model does after compiling.

    **The order below is load-bearing.** Textures go first because clearing a
    material's texture reference stops taking effect once the spec has been
    compiled, and freezing the inertials compiles it. Freezing then has to
    happen while the visual meshes are still attached, because their mass is
    exactly what it is there to capture.
    """
    spec = build_scene(include_environment=True)
    spec.body("pick_cube").add_freejoint()
    _delete_textures(spec)
    freeze_inertials(spec)
    _delete_visual_meshes(spec)
    _delete_scenery(spec)
    # Nothing resolves against it any more, and an absolute build path has no
    # business in a file the browser downloads.
    spec.meshdir = ""
    return spec


def physics_only_xml() -> str:
    """Return the asset-free policy scene as an MJCF string."""
    spec = physics_only_spec()
    spec.compile()
    return spec.to_xml()
