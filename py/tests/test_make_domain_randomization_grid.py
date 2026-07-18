# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_domain_randomization_grid.py"


def _module():
    spec = importlib.util.spec_from_file_location("make_domain_randomization_grid", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episodes(indices: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"episode_index": indices}).set_index("episode_index", drop=False)


def test_choose_episodes_keeps_explicit_shared_order():
    module = _module()
    chosen = module.choose_episodes(_episodes([0, 1, 2]), _episodes([1, 2, 3]), "2,1", 4, 0)
    assert chosen == [2, 1]


def test_choose_episodes_rejects_missing_episode():
    module = _module()
    with pytest.raises(ValueError, match="not present in both"):
        module.choose_episodes(_episodes([0]), _episodes([1]), "0", 1, 0)


def test_choose_episodes_selects_requested_grid_size_deterministically():
    module = _module()
    canonical = _episodes([0, 1, 2, 3])
    randomized = _episodes([0, 1, 2, 3])
    assert module.choose_episodes(canonical, randomized, None, 3, 42) == module.choose_episodes(
        canonical, randomized, None, 3, 42
    )
