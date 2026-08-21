# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The trainers' flags are data now, so the defaults are worth pinning.

A mistyped default here does not fail loudly -- it trains a different model and
reports success, which is why these read as a table of values rather than as
behavior.
"""

from __future__ import annotations

import json

import pytest

from pick_and_place.cli.training import (
    ImageTrainingRun,
    StateTrainingRun,
    config_to_json,
    load_config,
    parse_training_config,
)

STATE_DEFAULTS = {
    "updates": 20_000,
    "batch_size": 256,
    "learning_rate": 3e-3,
    "min_learning_rate": None,
    "warmup_steps": 0,
    "prediction_steps": 16,
    "seed": 0,
    "validation_interval": 1,
    "checkpoint_interval": None,
    "device": "auto",
    "architecture": "unet1d",
    "hidden_dim": 256,
    "hidden_layers": 2,
    "time_embedding_dim": 32,
    "unet_down_dims": (64, 128, 256),
    "unet_kernel_size": 5,
    "unet_groups": 8,
    "cube_symmetry_augmentation": False,
}

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


@pytest.mark.parametrize(
    "argv, config_class, expected",
    [
        (["--dataset", "d", "--output", "o"], StateTrainingRun, STATE_DEFAULTS),
        (["--export", "e", "--output", "o"], ImageTrainingRun, IMAGE_DEFAULTS),
    ],
)
def test_defaults_are_what_the_trainers_always_used(argv, config_class, expected):
    config = parse_training_config(config_class, argv)
    assert {name: getattr(config, name) for name in expected} == expected


def test_the_two_trainers_do_not_share_each_others_flags():
    """The state trainer has no --keypoints and the image trainer no --architecture."""
    with pytest.raises(SystemExit):
        parse_training_config(StateTrainingRun, ["--dataset", "d", "--output", "o", "--keypoints", "8"])
    with pytest.raises(SystemExit):
        parse_training_config(ImageTrainingRun, ["--export", "e", "--output", "o", "--architecture", "mlp"])


def test_a_recorded_run_reads_back_as_the_run_it_recorded(tmp_path):
    config = parse_training_config(
        StateTrainingRun,
        ["--dataset", "d", "--output", "o", "--batch-size", "512", "--unet-down-dims", "32", "64"],
    )
    written = tmp_path / "config.json"
    written.write_text(json.dumps(config_to_json(config)))
    assert load_config(StateTrainingRun, written) == config


def test_the_command_line_beats_the_config_file(tmp_path):
    recorded = parse_training_config(
        StateTrainingRun, ["--dataset", "d", "--output", "o", "--batch-size", "512", "--seed", "7"]
    )
    written = tmp_path / "config.json"
    written.write_text(json.dumps(config_to_json(recorded)))
    repeated = parse_training_config(
        StateTrainingRun, ["--config", str(written), "--batch-size", "8", "--output", "o2"]
    )
    assert repeated.batch_size == 8, "the command line wins"
    assert repeated.seed == 7, "and everything else comes from the file"
    assert repeated.output.name == "o2"


def test_a_config_file_from_an_older_run_still_loads(tmp_path):
    """Fields added since a run was recorded fall back to today's defaults."""
    written = tmp_path / "config.json"
    written.write_text(json.dumps({"dataset": "d", "output": "o", "batch_size": 128,
                                   "a_flag_that_no_longer_exists": True}))
    config = load_config(StateTrainingRun, written)
    assert config.batch_size == 128
    assert config.architecture == "unet1d"
