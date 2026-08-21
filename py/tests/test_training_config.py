# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The trainer's flags are data now, so the defaults are worth pinning.

A mistyped default here does not fail loudly -- it trains a different model and
reports success, which is why these read as a table of values rather than as
behavior.
"""

from __future__ import annotations

import json

import pytest

from pick_and_place.cli.training import (
    ImageTrainingRun,
    config_to_json,
    load_config,
    parse_training_config,
)

IMAGE_DEFAULTS = {
    "updates": 30_000,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "warmup_steps": 500,
    "prediction_steps": 16,
    "seed": 0,
    "validation_interval": 2_000,
    "checkpoint_interval": 5_000,
    "device": "cuda",
    "observation_steps": 2,
    "keypoints": 32,
    "pretrained_backbone": False,
    "trunk_stages": 3,
    "validation_fraction": 0.1,
    "validation_batches": 40,
    "log_interval": 100,
    "random_shift": 0,
    "random_scale_pct": 0.0,
    "photometric_augmentation": False,
    "resume": None,
    "amp": True,
}


def test_defaults_are_what_the_trainer_always_used():
    config = parse_training_config(ImageTrainingRun, ["--export", "e", "--output", "o"])
    assert {name: getattr(config, name) for name in IMAGE_DEFAULTS} == IMAGE_DEFAULTS


def test_a_flag_the_trainer_does_not_have_is_rejected():
    with pytest.raises(SystemExit):
        parse_training_config(ImageTrainingRun, ["--export", "e", "--output", "o", "--architecture", "mlp"])


def test_a_recorded_run_reads_back_as_the_run_it_recorded(tmp_path):
    config = parse_training_config(
        ImageTrainingRun,
        ["--export", "e", "--output", "o", "--batch-size", "512", "--keypoints", "8"],
    )
    written = tmp_path / "config.json"
    written.write_text(json.dumps(config_to_json(config)))
    assert load_config(ImageTrainingRun, written) == config


def test_the_command_line_beats_the_config_file(tmp_path):
    recorded = parse_training_config(
        ImageTrainingRun, ["--export", "e", "--output", "o", "--batch-size", "512", "--seed", "7"]
    )
    written = tmp_path / "config.json"
    written.write_text(json.dumps(config_to_json(recorded)))
    repeated = parse_training_config(
        ImageTrainingRun, ["--config", str(written), "--batch-size", "8", "--output", "o2"]
    )
    assert repeated.batch_size == 8, "the command line wins"
    assert repeated.seed == 7, "and everything else comes from the file"
    assert repeated.output.name == "o2"


def test_a_config_file_from_an_older_run_still_loads(tmp_path):
    """Fields added since a run was recorded fall back to today's defaults."""
    written = tmp_path / "config.json"
    written.write_text(json.dumps({"export": "e", "output": "o", "batch_size": 128,
                                   "a_flag_that_no_longer_exists": True}))
    config = load_config(ImageTrainingRun, written)
    assert config.batch_size == 128
    assert config.keypoints == 32
