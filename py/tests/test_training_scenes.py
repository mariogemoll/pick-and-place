# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import lzma
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.runtime.training_scenes import (
    NEUTRAL_ROBOT_STATE_REAL,
    TRAINING_CONTROL_HZ,
    TRAINING_MAX_STEPS,
    SceneStream,
    training_scenario,
)
from pick_and_place.policies.policy_evaluation import ScenarioManifest
from pick_and_place.planning.scenario_sampling import sample_scene

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = (
    "config/evaluation/smoke_v1.json",
    "config/evaluation/scripted_perturbation_smoke_v1.json",
    "config/evaluation/canonical_100_v1.json.xz",
    "config/evaluation/dr_100_v1.json.xz",
)


def _manifest_seeds() -> set[int]:
    seeds: set[int] = set()
    for name in MANIFESTS:
        manifest = ScenarioManifest.load(REPOSITORY_ROOT / name)
        seeds.update(scenario.seed for scenario in manifest.scenarios)
    return seeds


def test_training_seeds_never_collide_with_a_frozen_benchmark_scene():
    frozen = _manifest_seeds()
    training = {training_scenario(index).seed for index in range(2000)}
    assert frozen
    assert not (frozen & training)


def test_training_scenarios_are_reproducible_and_canonical():
    first = training_scenario(17)
    assert first == training_scenario(17)
    assert first.scenario_id == "dppo-train-000017"
    assert first.control_hz == TRAINING_CONTROL_HZ
    assert first.max_steps == TRAINING_MAX_STEPS
    assert first.domain_randomization_preset is None
    assert first.domain_randomization_sample == {"enabled": False}
    assert first.miscalibration_sample == {"joint_offsets_deg": {}}
    assert first.initial_robot_state_real == NEUTRAL_ROBOT_STATE_REAL


def test_training_scenario_matches_the_shared_scene_distribution():
    scenario = training_scenario(3)
    scene = sample_scene(np.random.default_rng(scenario.seed))
    assert scenario.source_position_m == pytest.approx(
        (scene.source.x, scene.source.y, scene.source.z)
    )
    assert scenario.target_position_m == pytest.approx(
        (scene.target.x, scene.target.y, scene.target.z)
    )
    assert scenario.target_plate_yaw_rad == pytest.approx(scene.plate_yaw_rad)


def test_worker_streams_partition_the_scene_sequence():
    streams = [SceneStream(offset=offset, stride=4) for offset in range(4)]
    drawn = [[stream.next().scenario_id for _ in range(5)] for stream in streams]
    flattened = [scenario_id for worker in drawn for scenario_id in worker]
    assert len(set(flattened)) == len(flattened)
    assert drawn[0][:2] == ["dppo-train-000000", "dppo-train-000004"]
    assert drawn[3][:2] == ["dppo-train-000003", "dppo-train-000007"]


def test_scene_stream_rejects_an_offset_outside_its_stride():
    with pytest.raises(ValueError):
        SceneStream(offset=4, stride=4)
    with pytest.raises(ValueError):
        SceneStream(offset=0, stride=0)


def test_manifests_on_disk_stay_loadable_after_the_sampler_refactor():
    # The frozen manifests are only reproducible while sample_scene keeps its
    # draw order; regenerating one is the guard, and this is the cheap sentinel
    # that the compressed suites still parse.
    payload = lzma.decompress(
        (REPOSITORY_ROOT / "config/evaluation/canonical_100_v1.json.xz").read_bytes()
    )
    assert b"canonical_100_v1" in payload
