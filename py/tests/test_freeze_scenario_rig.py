# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Freezing a rig must change the robot and nothing about the task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from freeze_scenario_rig import RIG_FIELDS, load_suite  # noqa: E402
import freeze_scenario_rig  # noqa: E402


def _scenario(index: int, offset: float) -> dict:
    return {
        "scenario_id": f"suite-{index:03d}",
        "source_position_m": [0.2 + index * 0.01, -0.1, 0.015],
        "target_position_m": [0.1, 0.05 + index * 0.01, 0.015],
        "workspace_region": "mid",
        "max_steps": 150,
        "control_hz": 10.0,
        "domain_randomization_sample": {"enabled": True, "light_intensity": 0.9 + offset},
        "miscalibration_sample": {"joint_offsets_deg": {"shoulder_pan": offset}},
        "physics_sample": {"mass_scale": 1.0 + offset},
    }


def _write_suite(path: Path, count: int) -> None:
    payload = {
        "schema_version": 3,
        "suite": "suite_v1",
        "scenarios": [_scenario(i, i * 0.5) for i in range(count)],
    }
    path.write_text(json.dumps(payload))


def _run(argv: list[str]) -> None:
    sys.argv = ["freeze_scenario_rig.py", *argv]
    freeze_scenario_rig.main()


def test_every_scene_gets_one_rig_and_keeps_its_geometry(tmp_path):
    source = tmp_path / "in.json"
    _write_suite(source, 4)
    out = tmp_path / "out.json"
    _run([str(source), "--from-scenario", "2", "--output", str(out)])

    header, scenarios = load_suite(out)
    assert len(scenarios) == 4
    rigs = {json.dumps({f: s[f] for f in RIG_FIELDS}, sort_keys=True) for s in scenarios}
    assert len(rigs) == 1
    assert scenarios[0]["miscalibration_sample"]["joint_offsets_deg"]["shoulder_pan"] == 1.0
    # the task is untouched
    original = json.loads(source.read_text())["scenarios"]
    for before, after in zip(original, scenarios):
        assert before["source_position_m"] == after["source_position_m"]
        assert before["target_position_m"] == after["target_position_m"]
        assert before["scenario_id"] == after["scenario_id"]
    assert header["frozen_rig"]["source"] == "scenario002"
    assert header["suite"].endswith("-rig-scenario002")


def test_the_rig_may_come_from_another_suite(tmp_path):
    target, donor = tmp_path / "in.json", tmp_path / "donor.json"
    _write_suite(target, 3)
    donor.write_text(json.dumps({
        "schema_version": 3, "suite": "canonical",
        "scenarios": [{
            **_scenario(0, 0.0),
            "domain_randomization_sample": {"enabled": False},
            "miscalibration_sample": {"joint_offsets_deg": {}},
            "physics_sample": {"mass_scale": 1.0},
        }],
    }))
    out = tmp_path / "out.json"
    _run([str(target), "--rig-from", str(donor), "--from-scenario", "0",
          "--label", "nominal", "--output", str(out)])

    header, scenarios = load_suite(out)
    assert all(s["domain_randomization_sample"] == {"enabled": False} for s in scenarios)
    assert header["suite"].endswith("-rig-nominal")


def test_a_sharded_suite_round_trips(tmp_path):
    source = tmp_path / "in.json"
    _write_suite(source, 5)
    out = tmp_path / "sharded" / "manifest.json"
    _run([str(source), "--from-scenario", "0", "--output", str(out), "--scenarios-per-file", "2"])

    header, scenarios = load_suite(out)
    assert len(header["scenario_files"]) == 3
    assert len(scenarios) == 5


def test_an_out_of_range_rig_is_refused(tmp_path):
    source = tmp_path / "in.json"
    _write_suite(source, 2)
    with pytest.raises(SystemExit, match="outside"):
        _run([str(source), "--from-scenario", "9", "--output", str(tmp_path / "o.json")])
