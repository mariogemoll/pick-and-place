# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco
import numpy as np

from pick_and_place.executor import CONTROL_HZ, HARDWARE_SIMULATION_HZ


def test_hardware_physics_substeps_advance_exactly_one_control_tick():
    assert CONTROL_HZ == 30.0
    steps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    assert steps_per_tick == 20

    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><option timestep="{1.0 / HARDWARE_SIMULATION_HZ}"/></mujoco>'
    )
    data = mujoco.MjData(model)
    times = []
    for _ in range(4):
        mujoco.mj_step(model, data, nstep=steps_per_tick)
        times.append(data.time)

    np.testing.assert_allclose(np.diff(times), 1.0 / CONTROL_HZ, atol=1e-12)
