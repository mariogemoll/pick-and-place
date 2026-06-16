# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_rl_hover.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_rl_hover", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_rl_hover = _load_module()


def test_resolve_vecnormalize_path_for_periodic_checkpoint(tmp_path):
    resume = tmp_path / "rl_hover_100000_steps.zip"
    resume.touch()
    expected = tmp_path / "rl_hover_vecnormalize_100000_steps.pkl"
    expected.touch()
    assert train_rl_hover.resolve_vecnormalize_path(resume) == expected


def test_resolve_vecnormalize_path_for_best_model(tmp_path):
    resume = tmp_path / "best_model.zip"
    resume.touch()
    expected = tmp_path / "best_vecnormalize.pkl"
    expected.touch()
    assert train_rl_hover.resolve_vecnormalize_path(resume) == expected


def test_resolve_vecnormalize_path_for_final_model(tmp_path):
    resume = tmp_path / "rl_hover_final.zip"
    resume.touch()
    expected = tmp_path / "vec_normalize_final.pkl"
    expected.touch()
    assert train_rl_hover.resolve_vecnormalize_path(resume) == expected


def test_resolve_vecnormalize_path_raises_when_stats_missing(tmp_path):
    resume = tmp_path / "rl_hover_100000_steps.zip"
    resume.touch()  # model present, but no sibling .pkl was written
    with pytest.raises(FileNotFoundError, match="--resume-vecnormalize"):
        train_rl_hover.resolve_vecnormalize_path(resume)
