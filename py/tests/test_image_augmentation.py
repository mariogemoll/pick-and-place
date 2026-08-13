# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Training-time image augmentation: what is drawn per sample, camera and step."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pick_and_place.policies.image_augmentation import (  # noqa: E402
    PhotometricRanges,
    gaussian_blur,
    random_photometric,
    random_scale,
    random_shift,
)

BATCH = 4
STEPS = 2
CAMERAS = 2
SIZE = 16

# Every range degenerate: the transform must then be the identity.
IDENTITY = PhotometricRanges(
    exposure=(1.0, 1.0),
    gamma=(1.0, 1.0),
    white_balance=(1.0, 1.0),
    noise_sigma=(0.0, 0.0),
    blur_sigma=(0.0, 0.0),
)


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def images(batch: int = BATCH) -> torch.Tensor:
    """A mid-grey field with structure, safely inside [0, 1] so gain cannot clip."""
    values = torch.rand(batch, STEPS, CAMERAS * 3, SIZE, SIZE, generator=generator(7))
    return values * 0.4 + 0.3


def repeated_steps() -> torch.Tensor:
    """One frame duplicated across the observation timesteps."""
    single = images()[:, :1]
    return single.expand(BATCH, STEPS, CAMERAS * 3, SIZE, SIZE).contiguous()


def test_photometric_preserves_shape_and_stays_in_range() -> None:
    source = images()
    result = random_photometric(source, PhotometricRanges(), generator())
    assert result.shape == source.shape
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_photometric_is_deterministic_given_a_seed() -> None:
    source = images()
    first = random_photometric(source, PhotometricRanges(), generator(3))
    second = random_photometric(source, PhotometricRanges(), generator(3))
    assert torch.equal(first, second)


def test_photometric_ranges_are_shared_across_observation_steps() -> None:
    """Identical frames must stay identical: gain and focus do not move in 0.1 s."""
    ranges = PhotometricRanges(noise_sigma=(0.0, 0.0))
    result = random_photometric(repeated_steps(), ranges, generator())
    assert torch.allclose(result[:, 0], result[:, 1], atol=1e-6)


def test_photometric_noise_is_drawn_per_step() -> None:
    """Sensor noise is the exception -- sharing it would let the model cancel it."""
    ranges = PhotometricRanges(
        exposure=(1.0, 1.0),
        gamma=(1.0, 1.0),
        white_balance=(1.0, 1.0),
        noise_sigma=(4.0, 4.0),
        blur_sigma=(0.0, 0.0),
    )
    result = random_photometric(repeated_steps(), ranges, generator())
    assert not torch.allclose(result[:, 0], result[:, 1], atol=1e-4)


def test_photometric_draws_differ_between_cameras() -> None:
    """Both cameras see the same pixels here, so any difference is the draw."""
    single = images()[:, :, :3]
    both = single.repeat(1, 1, CAMERAS, 1, 1)
    ranges = PhotometricRanges(noise_sigma=(0.0, 0.0))
    result = random_photometric(both, ranges, generator())
    assert not torch.allclose(result[:, :, :3], result[:, :, 3:], atol=1e-4)


def test_degenerate_photometric_ranges_are_the_identity() -> None:
    source = images()
    result = random_photometric(source, IDENTITY, generator())
    assert torch.allclose(result, source, atol=1e-6)


@pytest.mark.parametrize(
    "field, bounds",
    [
        ("exposure", (1.2, 0.8)),
        ("exposure", (0.0, 1.0)),
        ("gamma", (-1.0, 1.0)),
        ("noise_sigma", (-1.0, 1.0)),
        ("blur_sigma", (-0.5, 0.5)),
    ],
)
def test_photometric_ranges_reject_bad_bounds(field: str, bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        PhotometricRanges(**{field: bounds})


def test_photometric_rejects_channels_that_are_not_whole_cameras() -> None:
    with pytest.raises(ValueError, match="whole number of RGB cameras"):
        random_photometric(torch.rand(1, 1, 4, SIZE, SIZE), PhotometricRanges(), generator())


def test_blur_at_zero_sigma_is_the_identity() -> None:
    source = torch.rand(3, 3, SIZE, SIZE, generator=generator())
    result = gaussian_blur(source, torch.zeros(3))
    assert torch.allclose(result, source, atol=1e-6)


def test_blur_reduces_variance_with_sigma() -> None:
    source = torch.rand(2, 3, SIZE, SIZE, generator=generator())
    mild = gaussian_blur(source, torch.full((2,), 0.5))
    strong = gaussian_blur(source, torch.full((2,), 2.0))
    assert float(strong.var()) < float(mild.var()) < float(source.var())


def test_blur_preserves_a_constant_image() -> None:
    """Normalized weights and replicate padding must not darken the edges."""
    source = torch.full((2, 3, SIZE, SIZE), 0.42)
    result = gaussian_blur(source, torch.full((2,), 1.5))
    assert torch.allclose(result, source, atol=1e-6)


def test_blur_applies_its_own_sigma_per_image() -> None:
    source = torch.rand(1, 3, SIZE, SIZE, generator=generator()).repeat(2, 1, 1, 1)
    result = gaussian_blur(source, torch.tensor([0.0, 2.0]))
    assert torch.allclose(result[0], source[0], atol=1e-6)
    assert not torch.allclose(result[1], source[1], atol=1e-3)


def test_scale_at_unit_bounds_is_the_identity() -> None:
    source = images()
    assert torch.equal(random_scale(source, (1.0, 1.0), generator()), source)


def test_scale_magnifies_a_centered_square() -> None:
    """Zooming in must grow a centered bright patch, not shrink it."""
    source = torch.zeros(1, 1, 3, SIZE, SIZE)
    source[..., 6:10, 6:10] = 1.0
    magnified = random_scale(source, (1.6, 1.6), generator())
    assert float(magnified.sum()) > float(source.sum())


def test_scale_is_shared_across_observation_steps() -> None:
    result = random_scale(repeated_steps(), (0.9, 1.1), generator())
    assert torch.allclose(result[:, 0], result[:, 1], atol=1e-6)


def test_scale_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="ordered and positive"):
        random_scale(images(), (1.1, 0.9), generator())


def test_shift_preserves_shape_and_shares_the_draw_across_steps() -> None:
    result = random_shift(repeated_steps(), 4, generator())
    assert result.shape == (BATCH, STEPS, CAMERAS * 3, SIZE, SIZE)
    assert torch.allclose(result[:, 0], result[:, 1], atol=1e-6)


def test_shift_of_zero_pixels_is_the_identity() -> None:
    source = images()
    assert torch.equal(random_shift(source, 0, generator()), source)
