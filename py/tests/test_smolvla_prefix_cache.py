# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The storage layer under the cached SmolVLA vision tower."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pick_and_place.policies.smolvla_prefix_cache import (  # noqa: E402
    PrefixCacheSpec,
    _from_storage,
    _to_storage,
    cache_nbytes,
    embedding_key,
)

SPEC = PrefixCacheSpec(
    camera_keys=("observation.images.overhead", "observation.images.wrist"),
    num_frames=291_618,
    variants=1,
    num_tokens=64,
    width=960,
    dtype="bfloat16",
    augmented=False,
    dataset_fingerprint="1000ep-291618f-30fps",
)


def test_embedding_key_keeps_the_camera_name_but_leaves_the_image_namespace():
    # lerobot grows a batch axis on any 3D tensor under `observation.images.`,
    # so a cached token block must not live there.
    key = embedding_key("observation.images.overhead")
    assert key.endswith("overhead")
    assert not key.startswith("observation.images.")


def test_spec_round_trips_through_json():
    assert PrefixCacheSpec.from_json(SPEC.to_json()) == SPEC


def test_shape_and_size_match_the_dataset():
    assert SPEC.shape == (291_618, 1, 2, 64, 960)
    assert cache_nbytes(SPEC) == 291_618 * 2 * 64 * 960 * 2


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_storage_round_trip_is_exact(dtype):
    # bfloat16 has no numpy equivalent, so it is stored as raw uint16 and viewed
    # back. The round trip has to be bit-for-bit or the cache is not the tower's
    # output any more.
    original = torch.randn(4, 64, 960).to(getattr(torch, dtype))
    restored = _from_storage(_to_storage(original, dtype), dtype)
    assert restored.dtype == original.dtype
    assert torch.equal(restored, original)


def test_bfloat16_storage_keeps_the_exponent_range_float16_would_lose():
    large = torch.tensor([[[3.0e5, 1.0e-6]]], dtype=torch.float32)
    assert torch.isfinite(_from_storage(_to_storage(large, "bfloat16"), "bfloat16")).all()
