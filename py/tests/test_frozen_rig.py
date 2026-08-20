# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A frozen rig must pin the installation and leave the session alone."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.core.physics import PhysicsDraw
from pick_and_place.sim.domain_randomization import DomainRandomizationPreset, domain_sample_fields
from pick_and_place.sim.frozen_rig import FrozenRig

PRESET = Path(__file__).resolve().parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"
VARIED = (
    "light_intensity",
    "light_warm_cool",
    "key_light_position",
    "key_light_target",
    "key_light_bulb_radius",
    "fill_light_intensity",
    "exposure",
    "gamma",
    "white_balance",
    "noise_sigma",
    "blur_sigma",
    "cube_orientation_index",
)


def _sidecar(path: Path, sample, *, varied=VARIED, physics=None) -> Path:
    payload = {name: value for name, value in sample.__dict__.items() if name != "miscalibration"}
    payload["enabled"] = True
    path.write_text(
        json.dumps(
            {
                "suite": "suite_v1-rig-scenario007",
                "source": "scenario007",
                "varied_fields": sorted(varied),
                "domain_randomization_sample": payload,
                "miscalibration_sample": {
                    "joint_offsets_deg": dict(sample.miscalibration.base_offsets_deg),
                    "pan_jitter": {"sigma_deg": 2.2, "tau_s": 10.0, "seed": 12345},
                    "cube_belief_error": list(sample.miscalibration.cube_belief_error),
                    "target_belief_error": list(sample.miscalibration.target_belief_error),
                },
                "physics_sample": dataclasses.asdict(
                    physics if physics is not None else PhysicsDraw(mass_scale=0.9)
                ),
            },
            default=list,
        )
    )
    return path


@pytest.fixture(scope="module")
def preset() -> DomainRandomizationPreset:
    return DomainRandomizationPreset.load(PRESET)


def test_session_pins_the_rig_and_keeps_the_session_varying(preset, tmp_path):
    rig_draw = preset.sample(7)
    rig = FrozenRig.load(_sidecar(tmp_path / "rig7.frozen_rig.json", rig_draw))

    varied = set(VARIED)
    for episode in (0, 1, 2):
        session = rig.session(preset.sample(1000 + episode), np.random.default_rng(episode))
        for name in domain_sample_fields():
            if name in varied:
                continue
            assert getattr(session, name) == getattr(rig_draw, name), name

    # And the varying half really does vary, or the split would be vacuous.
    sessions = [
        rig.session(preset.sample(1000 + episode), np.random.default_rng(episode))
        for episode in range(3)
    ]
    assert len({s.light_intensity for s in sessions}) == 3
    assert len({s.exposure for s in sessions}) == 3


def test_session_carries_the_rigs_miscalibration(preset, tmp_path):
    rig_draw = preset.sample(7)
    rig = FrozenRig.load(_sidecar(tmp_path / "rig7.frozen_rig.json", rig_draw))
    session = rig.session(preset.sample(4242), np.random.default_rng(0))
    assert (
        session.miscalibration.base_offsets_deg == rig_draw.miscalibration.base_offsets_deg
    )
    assert session.miscalibration.cube_belief_error == rig_draw.miscalibration.cube_belief_error


def test_pan_wander_is_redrawn_per_episode_but_keeps_the_rigs_shape(preset, tmp_path):
    """The curve is the session's; sigma and tau are the arm's."""
    rig = FrozenRig.load(_sidecar(tmp_path / "rig7.frozen_rig.json", preset.sample(7)))
    curves = []
    for episode in range(2):
        jitter = rig.session(preset.sample(episode), np.random.default_rng(episode)).miscalibration
        assert jitter.pan_jitter is not None
        curves.append([jitter.pan_jitter.value(t) for t in (0.0, 1.0, 2.0, 5.0)])
    assert curves[0] != curves[1]


def test_physics_round_trips(preset, tmp_path):
    drawn = PhysicsDraw(
        joint_gain_scale={"shoulder_pan": 1.11, "elbow_flex": 0.9},
        joint_time_constant_scale={"shoulder_pan": 1.04},
        extra_joint_friction={"wrist_roll": 0.0017},
        tracking_bias_scale=0.0128,
        mass_scale=0.958,
        friction_scale=0.916,
        damping_scale=1.119,
    )
    rig = FrozenRig.load(
        _sidecar(tmp_path / "rig7.frozen_rig.json", preset.sample(7), physics=drawn)
    )
    assert rig.physics == drawn


def test_a_disabled_rig_is_refused(preset, tmp_path):
    path = tmp_path / "nominal.frozen_rig.json"
    path.write_text(
        json.dumps(
            {
                "suite": "suite_v1-rig-nominal",
                "source": "scenario000",
                "varied_fields": [],
                "domain_randomization_sample": {"enabled": False},
                "miscalibration_sample": {
                    "joint_offsets_deg": {},
                    "pan_jitter": None,
                    "cube_belief_error": [0.0, 0.0, 0.0, 0.0],
                    "target_belief_error": [0.0, 0.0],
                },
                "physics_sample": dataclasses.asdict(PhysicsDraw()),
            }
        )
    )
    with pytest.raises(ValueError, match="domain randomization disabled"):
        FrozenRig.load(path)


def test_an_unknown_varied_field_is_refused(preset, tmp_path):
    path = _sidecar(
        tmp_path / "rig7.frozen_rig.json", preset.sample(7), varied=(*VARIED, "lens_flare")
    )
    with pytest.raises(ValueError, match="lens_flare"):
        FrozenRig.load(path)


def test_a_truncated_physics_sample_is_refused(preset, tmp_path):
    path = _sidecar(tmp_path / "rig7.frozen_rig.json", preset.sample(7))
    payload = json.loads(path.read_text())
    del payload["physics_sample"]["mass_scale"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mass_scale"):
        FrozenRig.load(path)
