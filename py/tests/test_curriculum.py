# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
from pathlib import Path

import pytest

from pick_and_place.rl import curriculum as cur

_SAMPLE = Path(__file__).resolve().parents[1] / "curricula" / "pick_place.yaml"


# ----------------------------------------------------------------------------
# deep_merge
# ----------------------------------------------------------------------------


def test_deep_merge_recurses_into_nested_dicts():
    base = {"ppo": {"lr": 1.0, "n_steps": 2048}, "seed": 0}
    override = {"ppo": {"lr": 2.0}}
    merged = cur.deep_merge(base, override)
    # Only the named nested key is replaced; siblings survive.
    assert merged == {"ppo": {"lr": 2.0, "n_steps": 2048}, "seed": 0}


def test_deep_merge_does_not_mutate_inputs():
    base = {"reward_weights": {"yaw": 0.0}}
    override = {"reward_weights": {"yaw": 2.0}}
    cur.deep_merge(base, override)
    assert base == {"reward_weights": {"yaw": 0.0}}
    assert override == {"reward_weights": {"yaw": 2.0}}


def test_deep_merge_replaces_when_types_differ():
    assert cur.deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}


# ----------------------------------------------------------------------------
# parse_curriculum — happy path
# ----------------------------------------------------------------------------


def _doc():
    return {
        "name": "demo",
        "defaults": {
            "n_envs": 4,
            "total_steps": 100,
            "ppo": {"learning_rate": 3.0e-4, "n_steps": 2048},
            "env_kwargs": {"reward_weights": {"reach": 1.0, "yaw": 0.0}},
        },
        "stages": [
            {"name": "ch1"},
            {
                "name": "ch2",
                "total_steps": 50,
                "env_kwargs": {"reward_weights": {"yaw": 2.0}},
            },
        ],
    }


def test_parse_resolves_defaults_into_stages():
    spec = cur.parse_curriculum(_doc())
    assert spec.name == "demo"
    ch1, ch2 = spec.stages
    assert ch1.name == "ch1"
    # Defaults flow through to every stage.
    assert ch1.n_envs == 4
    assert ch1.total_steps == 100
    assert ch1.ppo == {"learning_rate": 3.0e-4, "n_steps": 2048}
    assert ch1.env_kwargs == {"reward_weights": {"reach": 1.0, "yaw": 0.0}}


def test_parse_stage_overrides_are_deltas():
    spec = cur.parse_curriculum(_doc())
    ch2 = spec.stage("ch2")
    assert ch2.total_steps == 50  # overridden
    assert ch2.n_envs == 4  # inherited
    # The reward-weight override merges: yaw flips on, reach is retained.
    assert ch2.env_kwargs == {"reward_weights": {"reach": 1.0, "yaw": 2.0}}


def test_builtin_defaults_fill_unspecified_fields():
    spec = cur.parse_curriculum({"name": "d", "stages": [{"name": "only"}]})
    stage = spec.stages[0]
    assert stage.seed == cur.BUILTIN_DEFAULTS["seed"]
    assert stage.eval_freq == cur.BUILTIN_DEFAULTS["eval_freq"]
    assert stage.ppo == {}


# ----------------------------------------------------------------------------
# parse_curriculum — validation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc, match",
    [
        ({"stages": [{"name": "a"}]}, "name"),
        ({"name": "", "stages": [{"name": "a"}]}, "name"),
        ({"name": "x", "stages": []}, "stages"),
        ({"name": "x", "stages": "nope"}, "stages"),
        ({"name": "x", "stages": [{}]}, "name"),
        ({"name": "x", "stages": [{"name": "a"}, {"name": "a"}]}, "Duplicate"),
        ({"name": "x", "stages": [{"name": "a", "bogus": 1}]}, "unknown config key"),
    ],
)
def test_parse_rejects_malformed_specs(doc, match):
    with pytest.raises(ValueError, match=match):
        cur.parse_curriculum(doc)


# ----------------------------------------------------------------------------
# The shipped sample spec
# ----------------------------------------------------------------------------


def test_sample_curriculum_loads_and_chains():
    spec = cur.load_curriculum(_SAMPLE)
    assert spec.name == "pick_place"
    assert spec.source_path == _SAMPLE
    names = [s.name for s in spec.stages]
    assert names == [
        "ch1_reach",
        "ch2_grasp",
        "ch3_lift",
        "ch4_carry",
        "ch5_place",
        "ch6_orientation",
    ]
    ch1, ch2, ch3, ch4, ch5, ch6 = spec.stages
    # Chapter 1 has every staged weight off except reach.
    assert ch1.env_kwargs["reward_weights"]["reach"] == 1.0
    assert ch1.env_kwargs["reward_weights"]["yaw"] == 0.0
    # Each later chapter turns on one more term.
    assert ch2.env_kwargs["reward_weights"]["grasp"] == 1.0
    assert ch3.env_kwargs["reward_weights"]["lift"] == 1.0
    assert ch4.env_kwargs["reward_weights"]["carry"] == 1.0
    assert ch5.env_kwargs["reward_weights"]["place"] == 5.0
    assert ch6.env_kwargs["reward_weights"]["yaw"] == 2.0
    # The delta merge keeps every earlier chapter's weights in the final one.
    assert ch6.env_kwargs["reward_weights"]["reach"] == 1.0
    assert ch6.env_kwargs["reward_weights"]["place"] == 5.0


# ----------------------------------------------------------------------------
# Run dirs / artifact paths
# ----------------------------------------------------------------------------


def test_stage_run_dir_is_deterministic_and_named():
    d = cur.stage_run_dir("/runs", "pick_place", "ch1_position")
    assert d == Path("/runs/pick_place/ch1_position")


def test_artifact_paths_under_run_dir():
    run_dir = Path("/runs/pick_place/ch1_position")
    assert cur.final_model_path(run_dir) == run_dir / "model_final.zip"
    assert cur.final_vecnormalize_path(run_dir) == run_dir / "vec_normalize_final.pkl"


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------


def test_git_provenance_reports_repo_sha():
    prov = cur.git_provenance(cur.REPO_ROOT)
    assert prov["sha"] is not None
    assert len(prov["sha"]) == 40
    assert isinstance(prov["dirty"], bool)


def test_git_provenance_degrades_outside_a_repo(tmp_path):
    prov = cur.git_provenance(tmp_path)
    assert prov == {"sha": None, "dirty": None}


def test_build_provenance_captures_config_sha_and_seed():
    spec = cur.parse_curriculum(_doc())
    stage = spec.stage("ch2")
    prov = cur.build_provenance(
        stage,
        spec,
        git={"sha": "abc123", "dirty": False},
        versions={"stable_baselines3": "2.3.0"},
        timestamp="2026-06-16T00:00:00+00:00",
    )
    assert prov["stage"] == "ch2"
    assert prov["seed"] == stage.seed
    assert prov["git"] == {"sha": "abc123", "dirty": False}
    assert prov["versions"] == {"stable_baselines3": "2.3.0"}
    assert prov["config"]["total_steps"] == 50
    assert prov["config"]["env_kwargs"]["reward_weights"]["yaw"] == 2.0


def test_write_provenance_roundtrips(tmp_path):
    spec = cur.parse_curriculum(_doc())
    prov = cur.build_provenance(
        spec.stage("ch1"), spec, git={"sha": "x", "dirty": False}, timestamp="t"
    )
    path = cur.write_provenance(tmp_path, prov)
    assert path == tmp_path / cur.PROVENANCE_NAME
    assert json.loads(path.read_text()) == prov
