# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The asset-free policy scene has to be the same physics as the scored one.

The browser page steps this model, so anything it quietly changes shows up as a
policy that behaves differently on the page than it does in evaluation -- with
nothing to indicate which of the two is wrong.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from pick_and_place.runtime.policy_sim import build_policy_sim_model
from pick_and_place.sim.physics_export import physics_only_spec, physics_only_xml
from pick_and_place.spec.robot import HARDWARE_SIMULATION_HZ, JOINT_NAMES

JOINT_START = (0.1, -0.6, 0.9, 0.4, 0.0, 0.6)
CUBE_START = (0.22, -0.10, 0.015, 1.0, 0.0, 0.0, 0.0)
NUDGE = (0.2, 0.3, -0.4, 0.1, 0.5, -0.5)


@pytest.fixture(scope="module")
def reference() -> mujoco.MjModel:
    model, _ = build_policy_sim_model(96, 96)
    return model


@pytest.fixture(scope="module")
def physics_only() -> mujoco.MjModel:
    return physics_only_spec().compile()


def _joint_addresses(model: mujoco.MjModel) -> dict[str, tuple[int, int]]:
    return {
        model.joint(i).name: (int(model.jnt_qposadr[i]), int(model.jnt_type[i]))
        for i in range(model.njnt)
    }


def _settle(model: mujoco.MjModel, steps: int = 1200) -> np.ndarray:
    """Drive the arm through a move and return the final joint and cube state."""
    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    mujoco.mj_resetData(model, data)
    addresses = _joint_addresses(model)
    for index, name in enumerate(JOINT_NAMES):
        data.qpos[addresses[name][0]] = JOINT_START[index]
    free = next(n for n, (_, t) in addresses.items() if t == mujoco.mjtJoint.mjJNT_FREE)
    data.qpos[addresses[free][0] : addresses[free][0] + 7] = CUBE_START
    data.ctrl[:] = JOINT_START
    mujoco.mj_forward(model, data)
    for step in range(steps):
        if step == steps // 3:
            data.ctrl[:] = np.array(JOINT_START) + np.array(NUDGE)
        mujoco.mj_step(model, data)
    return np.concatenate(
        [[data.qpos[addresses[name][0]] for name in JOINT_NAMES],
         data.qpos[addresses[free][0] : addresses[free][0] + 7]]
    )


def test_export_references_no_assets() -> None:
    """The bindings load a model from a string with no filesystem behind it."""
    xml = physics_only_xml()
    for token in ("<mesh", "<texture", "file=", ".png", ".stl", "meshdir"):
        assert token not in xml, f"exported scene still refers to {token}"


def test_inertials_survive_stripping_the_visuals(
    reference: mujoco.MjModel, physics_only: mujoco.MjModel
) -> None:
    """Mass and inertia are identical, geom by geom having gone or not.

    Three bodies -- the two camera mounts and the workspace frame -- infer their
    inertia from geometry rather than declaring it, so deleting the visual
    meshes takes real mass off the arm unless it has been frozen first. Before
    that was fixed the wrist camera mount lost two thirds of its mass here.
    """
    names = [physics_only.body(i).name for i in range(physics_only.nbody)]
    reference_names = [reference.body(i).name for i in range(reference.nbody)]
    for name in names:
        source, target = reference_names.index(name), names.index(name)
        assert reference.body_mass[source] == physics_only.body_mass[target], name
        assert np.array_equal(reference.body_inertia[source], physics_only.body_inertia[target])
        assert np.array_equal(reference.body_ipos[source], physics_only.body_ipos[target])


def test_visual_geometry_is_gone(physics_only: mujoco.MjModel) -> None:
    assert physics_only.nmesh == 0
    assert physics_only.ntex == 0


def test_stepping_matches_the_reference_scene(
    reference: mujoco.MjModel, physics_only: mujoco.MjModel
) -> None:
    """Two seconds of driven motion, with contact, landing in the same place.

    Exactly, not approximately: the two models differ only by geometry that
    carries no mass and takes part in no contact, so there is nothing left for
    them to disagree about.
    """
    assert np.array_equal(_settle(reference), _settle(physics_only))
