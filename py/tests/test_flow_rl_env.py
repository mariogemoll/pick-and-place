# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

import numpy as np
import pytest

from pick_and_place.dppo_rl.observations import FlowStateObservation
from pick_and_place.spec.controller import STATE_FEATURE

OBSERVATION_DIM = 17
ACTION_DIM = 6
PREDICTION_STEPS = 4


def _export(tmp_path, endpoint_semantics="absolute joint command"):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "export.json").write_text(
        json.dumps({
            "observation_steps": 2,
            "observation_dim": OBSERVATION_DIM,
            "prediction_steps": PREDICTION_STEPS,
            "endpoint_dim": ACTION_DIM,
            "endpoint_semantics": endpoint_semantics,
            "policy_hz": 10,
        })
    )
    np.savez(
        export_dir / "normalization.npz",
        observation_min=np.zeros(OBSERVATION_DIM, dtype=np.float32),
        observation_max=np.full(OBSERVATION_DIM, 10.0, dtype=np.float32),
        endpoint_min=np.zeros(ACTION_DIM, dtype=np.float32),
        endpoint_max=np.full(ACTION_DIM, 100.0, dtype=np.float32),
    )
    return export_dir


def _simulator_step():
    observation = {STATE_FEATURE: np.arange(6, dtype=np.float32)}
    info = {
        "task_state": {
            "cube_position_m": (0.1, 0.2, 0.3),
            # A quarter turn about z, whose first two rotation columns are
            # (0, 1, 0) and (-1, 0, 0).
            "cube_orientation_wxyz": (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
            "target_xy_m": (0.4, 0.5),
        }
    }
    return observation, info


def test_the_flow_observation_is_packed_in_the_exports_declared_order(tmp_path):
    codec = FlowStateObservation(export_dir=_export(tmp_path)).build()

    observation, info = _simulator_step()
    packed = codec.observe(observation, info)["state"]

    assert set(codec.keys) == {"state"}
    assert packed.shape == (OBSERVATION_DIM,)
    # Six robot coordinates, cube xyz, the first two rotation columns, target xy,
    # each mapped from [0, 10] onto [-1, 1].
    raw = packed / 2 * 10 + 5
    assert raw[:6] == pytest.approx(np.arange(6), abs=1e-5)
    assert raw[6:9] == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)
    assert raw[9:15] == pytest.approx([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], abs=1e-5)
    assert raw[15:] == pytest.approx([0.4, 0.5], abs=1e-5)


def test_the_flow_observation_uses_belief_when_the_environment_provides_it(tmp_path):
    codec = FlowStateObservation(export_dir=_export(tmp_path)).build()
    observation, info = _simulator_step()
    info["believed_task_state"] = {
        **info["task_state"],
        "cube_position_m": (0.6, 0.7, 0.8),
        "target_xy_m": (0.9, 1.0),
    }

    packed = codec.observe(observation, info)["state"]
    raw = packed / 2 * 10 + 5

    assert raw[6:9] == pytest.approx([0.6, 0.7, 0.8], abs=1e-5)
    assert raw[15:] == pytest.approx([0.9, 1.0], abs=1e-5)


def test_an_absolute_command_does_not_depend_on_the_measured_joints(tmp_path):
    codec = FlowStateObservation(export_dir=_export(tmp_path)).build()

    action = np.full(ACTION_DIM, -1.0, dtype=np.float32)
    for measured in (np.zeros(ACTION_DIM), np.full(ACTION_DIM, 50.0)):
        assert codec.command(action, measured) == pytest.approx(np.zeros(ACTION_DIM), abs=1e-4)
    assert codec.command(
        np.ones(ACTION_DIM, dtype=np.float32), np.zeros(ACTION_DIM)
    ) == pytest.approx(np.full(ACTION_DIM, 100.0), abs=1e-4)


def test_a_delta_export_is_refused_rather_than_silently_commanded(tmp_path):
    specification = FlowStateObservation(
        export_dir=_export(tmp_path, endpoint_semantics="joint delta")
    )
    with pytest.raises(ValueError, match="absolute joint commands"):
        specification.build()
