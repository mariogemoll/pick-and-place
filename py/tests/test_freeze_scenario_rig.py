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
    # --vary none: this covers the freeze itself, and the fixture's light would
    # otherwise legitimately differ scene to scene under the default.
    _run([str(source), "--from-scenario", "2", "--output", str(out), "--vary", "none"])

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
    assert header["suite"].endswith("-rig-scenario002")
    rig = json.loads((out.with_name(f"{out.stem}.frozen_rig.json")).read_text())
    assert rig["source"] == "scenario002"
    assert rig["miscalibration_sample"]["joint_offsets_deg"]["shoulder_pan"] == 1.0


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
    _run([str(source), "--from-scenario", "0", "--output", str(out),
          "--scenarios-per-file", "2", "--vary", "none"])

    # load_suite consumes scenario_files, so check the shards on disk.
    assert len(list(out.parent.glob("scenarios-*.json.xz"))) == 3
    _, scenarios = load_suite(out)
    assert len(scenarios) == 5
    assert [s["scenario_id"] for s in scenarios] == [f"suite-{i:03d}" for i in range(5)]


def test_an_out_of_range_rig_is_refused(tmp_path):
    source = tmp_path / "in.json"
    _write_suite(source, 2)
    with pytest.raises(SystemExit, match="outside"):
        _run([str(source), "--from-scenario", "9", "--output", str(tmp_path / "o.json")])


def test_the_light_keeps_moving_while_the_rig_holds_still(tmp_path):
    """A deployment pins the robot and the room, not the hour of the day."""
    source = tmp_path / "in.json"
    payload = {"schema_version": 3, "suite": "suite_v1", "scenarios": []}
    for index in range(4):
        scene = _scenario(index, index * 0.5)
        scene["domain_randomization_sample"] = {
            "enabled": True,
            "light_intensity": 0.5 + index,
            "exposure": 1.0 + index,
            "overhead_camera_position_m": [index * 0.01, 0.0, 0.0],
            "table_rgb": [index / 10, 0.2, 0.3],
        }
        payload["scenarios"].append(scene)
    source.write_text(json.dumps(payload))

    out = tmp_path / "out.json"
    _run([str(source), "--from-scenario", "0", "--output", str(out),
          "--vary", "lighting,camera-response"])
    _, scenarios = load_suite(out)

    assert [s["domain_randomization_sample"]["light_intensity"] for s in scenarios] == [0.5, 1.5, 2.5, 3.5]
    assert [s["domain_randomization_sample"]["exposure"] for s in scenarios] == [1.0, 2.0, 3.0, 4.0]
    assert {tuple(s["domain_randomization_sample"]["overhead_camera_position_m"]) for s in scenarios} == {(0.0, 0.0, 0.0)}
    assert {tuple(s["domain_randomization_sample"]["table_rgb"]) for s in scenarios} == {(0.0, 0.2, 0.3)}
    rig = json.loads((out.with_name(f"{out.stem}.frozen_rig.json")).read_text())
    assert "light_intensity" in rig["varied_fields"]

    frozen = tmp_path / "frozen.json"
    _run([str(source), "--from-scenario", "0", "--output", str(frozen), "--vary", "none"])
    _, whole = load_suite(frozen)
    assert {s["domain_randomization_sample"]["light_intensity"] for s in whole} == {0.5}


def test_a_misspelled_vary_field_is_refused(tmp_path):
    source = tmp_path / "in.json"
    _write_suite(source, 2)
    with pytest.raises(SystemExit, match="does not have"):
        _run([str(source), "--from-scenario", "0", "--output", str(tmp_path / "o.json"),
              "--vary", "lighting,not_a_field"])
