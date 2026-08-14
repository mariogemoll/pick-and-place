# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The guard on the SmolVLA step benchmark's tower probe."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_smolvla_step.py"
HIDDEN_SIZE = 960


def load_module():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("benchmark_smolvla_step", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT.parent))
    return module


def stub_policy(embed_image):  # noqa: ANN001, ANN201
    """The three attributes `_tower_token_count` reads, and nothing else."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(resize_imgs_with_padding=(8, 8)),
        model=types.SimpleNamespace(
            vlm_with_expert=types.SimpleNamespace(
                embed_image=embed_image,
                config=types.SimpleNamespace(
                    text_config=types.SimpleNamespace(hidden_size=HIDDEN_SIZE)
                ),
            )
        ),
    )


def test_token_count_comes_from_the_tower():
    module = load_module()
    policy = stub_policy(lambda image: torch.zeros(image.shape[0], 64, HIDDEN_SIZE))
    assert module._tower_token_count(policy, torch.device("cpu")) == 64


def test_an_identity_tower_is_refused_rather_than_measured():
    # `patch_policy_for_cached_prefix` makes `embed_image` the identity, and the
    # probe would then read shape[1] off a [1, 3, H, W] image and call it 3
    # tokens. A synthetic batch built from that runs on a twentieth of the image
    # tokens training uses, and reports a step that never happens.
    from pick_and_place.policies.smolvla_prefix_cache import _identity_embed_image

    module = load_module()
    policy = stub_policy(_identity_embed_image)
    with pytest.raises(RuntimeError, match="before patching"):
        module._tower_token_count(policy, torch.device("cpu"))
