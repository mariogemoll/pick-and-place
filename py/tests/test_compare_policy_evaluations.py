# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Statistics and guards behind the paired evaluation comparison."""

import json
from pathlib import Path

import pytest

from pick_and_place.cli import compare_policy_evaluations as compare


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes() -> None:
    """The reason for Wilson rather than the normal approximation: 0/n and n/n."""
    low, high = compare.wilson_interval(0, 33)

    assert low == 0.0
    assert high == pytest.approx(0.1043, abs=1e-4)

    low, high = compare.wilson_interval(33, 33)

    assert low == pytest.approx(0.8957, abs=1e-4)
    assert high == 1.0


def test_wilson_interval_matches_a_known_value() -> None:
    low, high = compare.wilson_interval(39, 66)

    assert low == pytest.approx(0.4705, abs=1e-4)
    assert high == pytest.approx(0.7013, abs=1e-4)


def test_wilson_interval_of_an_empty_run_is_degenerate() -> None:
    assert compare.wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("only_first", "discordant", "expected"),
    [
        (0, 10, 0.001953),  # every disagreement one way: significant
        (5, 10, 1.0),  # evenly split: no evidence either way
        (2, 10, 0.109375),
        (0, 0, 1.0),  # the two arms never disagreed
    ],
)
def test_exact_binomial_tail_matches_scipy(
    only_first: int, discordant: int, expected: float
) -> None:
    assert compare._binomial_tail(only_first, discordant) == pytest.approx(expected, abs=1e-6)


def test_mcnemar_ignores_concordant_scenarios() -> None:
    """Scenarios both arms win, or both lose, carry no signal about which is better."""
    scenarios = [f"s{index}" for index in range(6)]
    first = dict(zip(scenarios, [True, True, True, False, False, False]))
    second = dict(zip(scenarios, [True, False, False, False, True, True]))

    result = compare.mcnemar(first, second, scenarios)

    assert result["both"] == 1
    assert result["neither"] == 1
    assert result["only_first"] == 2
    assert result["only_second"] == 2
    assert result["discordant"] == 4
    assert result["p_value_exact"] == pytest.approx(1.0)
    assert result["only_first_scenarios"] == ["s1", "s2"]
    assert result["only_second_scenarios"] == ["s4", "s5"]


def _write_run(
    directory: Path,
    *,
    seed: int = 0,
    final_xy_m: float = 0.014,
    scenario_ids: tuple[str, ...] = ("s0",),
) -> Path:
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps({
            "scenario_manifest": {"sha256": "abc", "selected_scenario_ids": list(scenario_ids)},
            "environment": {
                "scene_appearance": "blue-cube",
                "control_hz": [10.0],
                "episode_step_limits": [150],
                "image_height": 96,
                "image_width": 96,
                "oracle": {"success_xy_tolerance_m": 0.04},
            },
            "controller": {
                "executed_action_steps": 8,
                "sampler": "ddpm-100",
                "denoising_steps": 100,
                "sampling_seed": seed,
                "policy_hz": 10.0,
                "weights": "ema",
                "normalization": {"sha256": "def"},
                "integration": "euler",
                "integration_steps": 10,
                "noise_correlation": 0.0,
                "export": {
                    "manifest_sha256": "export-manifest",
                    "normalization_sha256": "export-normalization",
                },
            },
        })
    )
    (directory / "episodes.jsonl").write_text(
        "".join(
            json.dumps({
                "scenario_id": scenario_id,
                "success": True,
                "final_xy_error_m": final_xy_m,
                "min_tcp_to_cube_distance_m": 0.016,
                "milestones": {"cube_lifted": True, "pickup_contact_attempted": True},
            })
            + "\n"
            for scenario_id in scenario_ids
        )
    )
    return directory


def test_placed_metric_uses_the_six_centimetre_threshold(tmp_path: Path) -> None:
    inside = compare.EvaluationRun(_write_run(tmp_path / "inside", final_xy_m=0.059))
    outside = compare.EvaluationRun(_write_run(tmp_path / "outside", final_xy_m=0.061))

    assert inside.outcomes("placed_6cm") == {"s0": True}
    assert outside.outcomes("placed_6cm") == {"s0": False}


def test_settings_expose_a_differing_sampling_seed(tmp_path: Path) -> None:
    """The comparison refuses arms that did not face the same experiment."""
    first = compare.EvaluationRun(_write_run(tmp_path / "first", seed=0))
    second = compare.EvaluationRun(_write_run(tmp_path / "second", seed=1))

    assert first.settings()["controller/sampling_seed"] == 0
    assert second.settings()["controller/sampling_seed"] == 1
    differing = {
        key
        for key, value in first.settings().items()
        if value != second.settings()[key]
    }
    assert differing == {"controller/sampling_seed"}


def test_settings_read_every_comparable_field(tmp_path: Path) -> None:
    """A typo in a field path would silently disable that guard."""
    run = compare.EvaluationRun(_write_run(tmp_path / "run"))

    assert all(value is not None for value in run.settings().values())


def test_settings_bind_flow_image_euler_contract(tmp_path: Path) -> None:
    """Flow-image runs must agree on the sampler and export they actually use."""
    first = _write_run(tmp_path / "first")
    second = _write_run(tmp_path / "second")
    second_run = json.loads((second / "run.json").read_text())
    second_run["controller"]["integration_steps"] = 20
    (second / "run.json").write_text(json.dumps(second_run))

    first_settings = compare.EvaluationRun(first).settings()
    second_settings = compare.EvaluationRun(second).settings()

    differing = {
        key for key, value in first_settings.items() if value != second_settings[key]
    }
    assert differing == {"controller/integration_steps"}


def test_sharded_arm_reassembles_its_workers(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    _write_run(arm / "shard-000", scenario_ids=("s2", "s3"))
    _write_run(arm / "shard-001", scenario_ids=("s0", "s1"))

    run = compare.EvaluationRun(arm)

    assert len(run.shards) == 2
    # Sorted, so a shard that finishes out of order does not reorder the arm.
    assert run.scenario_ids == ["s0", "s1", "s2", "s3"]


def test_sharded_arm_rejects_a_shard_run_with_different_settings(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    _write_run(arm / "shard-000", scenario_ids=("s0",), seed=0)
    _write_run(arm / "shard-001", scenario_ids=("s1",), seed=1)

    with pytest.raises(SystemExit, match="controller/sampling_seed"):
        compare.EvaluationRun(arm)


def test_sharded_arm_rejects_overlapping_shards(tmp_path: Path) -> None:
    """An off-by-one in the offsets would double-count scenarios."""
    arm = tmp_path / "arm"
    _write_run(arm / "shard-000", scenario_ids=("s0", "s1"))
    _write_run(arm / "shard-001", scenario_ids=("s1", "s2"))

    with pytest.raises(SystemExit, match="overlap"):
        compare.EvaluationRun(arm)


def test_single_directory_arm_still_works(tmp_path: Path) -> None:
    run = compare.EvaluationRun(_write_run(tmp_path / "arm", scenario_ids=("s0", "s1")))

    assert run.shards == [tmp_path / "arm"]
    assert run.scenario_ids == ["s0", "s1"]
