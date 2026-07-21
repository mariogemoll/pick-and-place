# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pick_and_place.policy_evaluation import (
    EpisodeResult,
    FailureFlags,
    ScenarioManifest,
    TaskMilestones,
    TaskOracleConfig,
    TaskState,
    TaskSuccessOracle,
    aggregate_episode_results,
    fingerprint_checkpoint,
    write_evaluation_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _state(**changes) -> TaskState:
    state = TaskState(
        cube_position_m=(0.04, 0.0, 0.015),
        cube_linear_velocity_m_s=(0.0, 0.0, 0.0),
        cube_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        target_xy_m=(0.0, 0.0),
    )
    return replace(state, **changes)


def _result(scenario_id: str, success: bool, group: str = "canonical") -> EpisodeResult:
    return EpisodeResult(
        scenario_id=scenario_id,
        group=group,
        workspace_region="near" if scenario_id.endswith("0") else "far",
        success=success,
        milestones=TaskMilestones(
            pickup_contact_attempted=True,
            cube_lifted=True,
            successful_placement=success,
        ),
        failures=FailureFlags(timeout=not success),
        final_xy_error_m=0.01 if success else 0.1,
        control_steps=100,
        simulated_time_s=2.0,
        time_to_success_s=1.5 if success else None,
    )


def test_smoke_manifest_is_frozen_and_hashable():
    manifest = ScenarioManifest.load(REPOSITORY_ROOT / "config/evaluation/smoke_v1.json")

    assert manifest.suite == "smoke_v1"
    assert len(manifest.scenarios) == 8
    assert {scenario.control_hz for scenario in manifest.scenarios} == {30.0}
    assert {scenario.max_steps / scenario.control_hz for scenario in manifest.scenarios} == {15.0}
    assert manifest.scenarios[0].source_position_m == (0.212880144, -0.160945783, 0.015)
    assert len(manifest.sha256()) == 64
    assert json.loads(manifest.canonical_json()) == json.loads(json.dumps(manifest.to_dict()))


def test_scripted_perturbation_smoke_manifest_has_frozen_joint_and_camera_offsets():
    manifest = ScenarioManifest.load(
        REPOSITORY_ROOT / "config/evaluation/scripted_perturbation_smoke_v1.json"
    )

    assert manifest.suite == "scripted_perturbation_smoke_v1"
    assert len(manifest.scenarios) == 2
    for scenario in manifest.scenarios:
        sample = scenario.domain_randomization_sample
        assert sample["enabled"] is True
        assert any(sample["overhead_camera_position_m"])
        assert any(sample["overhead_camera_rotation_deg"])
        assert any(sample["wrist_camera_position_m"])
        assert any(sample["wrist_camera_rotation_deg"])
        assert any(scenario.miscalibration_sample["joint_offsets_deg"].values())
        assert scenario.control_hz == 30.0
        assert scenario.max_steps / scenario.control_hz == 15.0


def test_initial_resting_state_is_not_a_settled_placement_milestone():
    oracle = TaskSuccessOracle()

    oracle.update(_state(cube_position_m=(0.1, 0.0, 0.015)), 0.02)

    assert not oracle.milestones.cube_settled
    assert not oracle.failure_flags(timed_out=True).off_target_placement


def test_oracle_accepts_tolerance_boundaries_after_dwell():
    oracle = TaskSuccessOracle(TaskOracleConfig(success_confirmation_s=0.1))

    oracle.update(_state(cube_position_m=(0.04, 0.0, 0.05), grasped=True), 0.01)
    assert not oracle.update(_state(), 0.05)
    assert oracle.update(_state(), 0.05)
    assert oracle.milestones.cube_settled
    assert oracle.success_time_s == pytest.approx(0.11)


@pytest.mark.parametrize(
    "state",
    [
        _state(cube_position_m=(0.040001, 0.0, 0.015)),
        _state(cube_position_m=(0.04, 0.0, 0.025001)),
        _state(cube_linear_velocity_m_s=(0.02, 0.0, 0.0)),
        _state(cube_angular_velocity_rad_s=(0.2, 0.0, 0.0)),
    ],
)
def test_oracle_rejects_states_outside_or_at_strict_speed_thresholds(state):
    oracle = TaskSuccessOracle(TaskOracleConfig(success_confirmation_s=0.01))

    assert not oracle.update(state, 0.02)


def test_collision_permanently_disqualifies_an_otherwise_valid_placement():
    oracle = TaskSuccessOracle(TaskOracleConfig(success_confirmation_s=0.04))

    oracle.update(_state(unexpected_collision=True), 0.02)
    assert not oracle.update(_state(), 0.1)
    assert oracle.failure_flags(timed_out=True).unexpected_collision


def test_oracle_tracks_lift_carry_release_and_failure_taxonomy():
    oracle = TaskSuccessOracle(TaskOracleConfig(stable_carry_confirmation_s=0.04))
    carried = _state(
        cube_position_m=(0.1, 0.0, 0.05),
        robot_cube_contact=True,
        grasped=True,
    )
    oracle.update(carried, 0.02)
    oracle.update(carried, 0.02)
    oracle.update(replace(carried, grasped=False, gripper_open=True), 0.02)

    assert oracle.milestones.pickup_contact_attempted
    assert oracle.milestones.cube_lifted
    assert oracle.milestones.stable_carry
    assert oracle.milestones.cube_released
    assert oracle.failure_flags(timed_out=True).early_release
    assert oracle.failure_flags(timed_out=True).timeout


def test_aggregation_and_artifacts_are_structured(tmp_path):
    results = [
        _result("scenario-0", True),
        replace(
            _result("scenario-1", False),
            controller_failure={"code": "planning_error", "message": "no safe plan"},
        ),
    ]

    summary = aggregate_episode_results(results)
    assert summary["success_rate"] == 0.5
    assert summary["success_rate_95ci_wilson"][0] < 0.5
    assert summary["success_rate_95ci_wilson"][1] > 0.5
    assert summary["by_workspace_region"]["near"]["success_rate"] == 1.0
    assert summary["controller_failures"] == {
        "count": 1,
        "rate": 0.5,
        "by_code": {"planning_error": 1},
    }

    output_dir = tmp_path / "evaluation"
    write_evaluation_artifacts(output_dir, {"policy_type": "no-op"}, results)
    assert json.loads((output_dir / "summary.json").read_text())["success_count"] == 1
    assert len((output_dir / "episodes.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        write_evaluation_artifacts(output_dir, {}, results)

    failure_interval = aggregate_episode_results([_result("failure", False)])[
        "success_rate_95ci_wilson"
    ]
    success_interval = aggregate_episode_results([_result("success", True)])[
        "success_rate_95ci_wilson"
    ]
    assert failure_interval[0] == 0.0
    assert success_interval[1] == 1.0


def test_checkpoint_fingerprint_tracks_local_contents(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("first")
    first = fingerprint_checkpoint(str(checkpoint))
    (checkpoint / "config.json").write_text("second")
    second = fingerprint_checkpoint(str(checkpoint))

    assert first["kind"] == "directory"
    assert first["file_count"] == 1
    assert first["sha256"] != second["sha256"]
