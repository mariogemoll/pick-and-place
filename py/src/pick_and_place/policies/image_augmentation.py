# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Training-time image augmentation for the image-conditioned policies.

A dataset re-rendered from one recording carries exactly one camera pose, one
lighting rig, one background and one exposure. The real rig has none of those
fixed: the overhead camera measures 10-25 mm and ~2 degrees off its authored
pose, with focal length varying +/-1.45% between sessions, on top of whatever
the room lighting is doing that day. A policy trained on the fixed render has
no reason to be invariant to any of it.

Randomizing the renderer covers this properly, but only for data that is
re-recorded. These transforms cover the part that is recoverable from pixels
alone -- the camera's response and a scale change standing in for focal jitter
-- so an existing export can train a more robust policy without being rendered
again.

**What is drawn per what.** Every draw is per sample and per camera, and is
shared across the observation timesteps, with one deliberate exception. The two
timesteps are ~0.1 s apart: a camera's gain, white balance and focus do not
change in that time, so varying them across the pair would teach the policy that
a lighting change is evidence of motion. Sensor noise is the exception -- it is
genuinely independent frame to frame, and sharing its realization would let the
model difference the pair to cancel it and so learn nothing about noise.

The chain is ordered as a camera's is: blur (optics), then gain and white
balance (sensor), then gamma (transfer curve), then read noise, then clipping.
Noise lands after gamma rather than before it because the ranges below are
calibrated in grey levels on the stored 8-bit frame, which is where it was
measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

# Kernel width for the blur. The sigmas below stay under ~1.5 px, and a Gaussian
# is numerically dead beyond three sigma, so seven taps is already generous.
BLUR_KERNEL_SIZE = 7

# Below this the Gaussian weights collapse to a delta and the blur is identity.
# It exists to keep the zero-sigma draw from dividing by zero.
MIN_BLUR_SIGMA = 1e-3


@dataclass(frozen=True)
class PhotometricRanges:
    """Inclusive ``(low, high)`` bounds for one camera-response draw each.

    The defaults widen `config/domain_randomization/act_mild_v1.json`, whose
    values size the *within-session* residual. Between sessions the rig moves
    further than that -- the episode that first exposed this had a measured
    camera pose outside a jitter box built from within-session spread -- so
    these are deliberately looser than the measured variation rather than
    matched to it.

    **``exposure`` is the exception, and is asymmetric on purpose.** The tan
    table renders at a luminance of ~224 of 255, so the scene sits close to
    saturation before any gain is applied: at 1.4 the table clips to flat white
    and takes the cube's edges and the corner plates with it, which is not a
    camera this rig can be, and destroys the signal the policy needs. Dimming is
    unconstrained by that, so the range runs well below 1 and barely above it.
    The preset's symmetric 0.85-1.15 was sized by the same ceiling.

    ``noise_sigma`` is in grey levels on a 0-255 frame; everything else is a
    multiplier on a 0-1 frame. ``blur_sigma`` is in pixels.
    """

    exposure: tuple[float, float] = (0.65, 1.15)
    gamma: tuple[float, float] = (0.8, 1.25)
    white_balance: tuple[float, float] = (0.9, 1.1)
    noise_sigma: tuple[float, float] = (0.0, 6.0)
    blur_sigma: tuple[float, float] = (0.0, 1.2)

    def __post_init__(self) -> None:
        for name in ("exposure", "gamma", "white_balance", "noise_sigma", "blur_sigma"):
            low, high = getattr(self, name)
            if not low <= high:
                raise ValueError(f"{name} bounds must be ordered, got ({low}, {high})")
        if self.exposure[0] <= 0 or self.gamma[0] <= 0 or self.white_balance[0] <= 0:
            raise ValueError("exposure, gamma and white_balance bounds must be positive")
        if self.noise_sigma[0] < 0 or self.blur_sigma[0] < 0:
            raise ValueError("noise_sigma and blur_sigma bounds must be non-negative")


def _draw(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    *,
    generator: torch.Generator,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Uniform draws in ``bounds``, on the reference tensor's device and dtype."""
    low, high = bounds
    values = torch.rand(
        shape, generator=generator, device=reference.device, dtype=reference.dtype
    )
    return values * (high - low) + low


def _split_cameras(images: torch.Tensor) -> tuple[torch.Tensor, int]:
    """View ``(B, S, cameras * 3, H, W)`` as ``(B, S, cameras, 3, H, W)``."""
    if images.ndim != 5:
        raise ValueError(f"images must be (batch, steps, channels, h, w), got {tuple(images.shape)}")
    batch, steps, channels, height, width = images.shape
    if channels % 3:
        raise ValueError(f"channels must be a whole number of RGB cameras, got {channels}")
    cameras = channels // 3
    return images.reshape(batch, steps, cameras, 3, height, width), cameras


def gaussian_blur(images: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Blur each image by its own sigma, separably, replicating at the edge.

    ``images`` is ``(N, 3, H, W)`` and ``sigma`` is ``(N,)``. A sigma at or below
    `MIN_BLUR_SIGMA` leaves its image untouched.
    """
    count = len(images)
    if sigma.shape != (count,):
        raise ValueError(f"sigma must have shape {(count,)}, got {tuple(sigma.shape)}")

    offsets = torch.arange(BLUR_KERNEL_SIZE, device=images.device, dtype=images.dtype)
    offsets = offsets - (BLUR_KERNEL_SIZE - 1) / 2
    clamped = sigma.clamp_min(MIN_BLUR_SIGMA)
    weights = torch.exp(-(offsets[None, :] ** 2) / (2 * clamped[:, None] ** 2))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    # One kernel per image, applied to each of its three channels.
    weights = weights.repeat_interleave(3, dim=0)

    pad = (BLUR_KERNEL_SIZE - 1) // 2
    groups = count * 3
    flat = images.reshape(1, groups, *images.shape[-2:])
    horizontal = F.conv2d(
        F.pad(flat, (pad, pad, 0, 0), mode="replicate"),
        weights.reshape(groups, 1, 1, BLUR_KERNEL_SIZE),
        groups=groups,
    )
    blurred = F.conv2d(
        F.pad(horizontal, (0, 0, pad, pad), mode="replicate"),
        weights.reshape(groups, 1, BLUR_KERNEL_SIZE, 1),
        groups=groups,
    )
    return blurred.reshape(images.shape)


def random_photometric(
    images: torch.Tensor, ranges: PhotometricRanges, generator: torch.Generator
) -> torch.Tensor:
    """Push each camera stream through a randomly drawn camera response.

    ``images`` is ``(batch, steps, cameras * 3, height, width)`` in ``[0, 1]``,
    before any dataset normalization -- exposure and gamma are meaningless on a
    mean-subtracted frame. The result is clipped back into ``[0, 1]``, which is
    what a real sensor does to a blown highlight.
    """
    split, cameras = _split_cameras(images)
    batch, steps = split.shape[0], split.shape[1]
    per_camera = (batch, 1, cameras, 1, 1, 1)

    blur_sigma = _draw((batch, 1, cameras), ranges.blur_sigma, generator=generator, reference=split)
    folded = split.reshape(-1, 3, *split.shape[-2:])
    blurred = gaussian_blur(
        folded, blur_sigma.expand(batch, steps, cameras).reshape(-1)
    ).reshape(split.shape)

    exposure = _draw(per_camera, ranges.exposure, generator=generator, reference=split)
    white_balance = _draw(
        (batch, 1, cameras, 3, 1, 1), ranges.white_balance, generator=generator, reference=split
    )
    gamma = _draw(per_camera, ranges.gamma, generator=generator, reference=split)
    noise_sigma = _draw(per_camera, ranges.noise_sigma, generator=generator, reference=split) / 255.0

    values = (blurred * exposure * white_balance).clamp(0.0, 1.0) ** gamma
    # Drawn over the full tensor, so the two observation timesteps get different
    # noise even though they share the sigma that produced it.
    noise = torch.randn(
        values.shape, generator=generator, device=values.device, dtype=values.dtype
    )
    return (values + noise * noise_sigma).clamp(0.0, 1.0).reshape(images.shape)


def random_scale(
    images: torch.Tensor, bounds: tuple[float, float], generator: torch.Generator
) -> torch.Tensor:
    """Zoom each camera stream about its center, replicating at the edge.

    This stands in for the overhead camera's between-session focal length, which
    measures at +/-1.45% and is randomized at +/-2.5% in the renderer. A scale
    above 1 magnifies. One draw per sample and camera, shared across the
    observation timesteps so the pair cannot read as approach motion.
    """
    low, high = bounds
    if not 0 < low <= high:
        raise ValueError(f"scale bounds must be ordered and positive, got ({low}, {high})")
    if low == high == 1.0:
        return images

    split, cameras = _split_cameras(images)
    batch, steps = split.shape[0], split.shape[1]
    scale = _draw((batch, 1, cameras), (low, high), generator=generator, reference=split)
    scale = scale.expand(batch, steps, cameras).reshape(-1)

    folded = split.reshape(-1, 3, *split.shape[-2:])
    # affine_grid maps output coordinates back into the input, so magnifying by
    # ``scale`` means sampling a 1/scale slice of the normalized input square.
    theta = torch.zeros(len(folded), 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    grid = F.affine_grid(theta, list(folded.shape), align_corners=False)
    warped = F.grid_sample(
        folded, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return warped.reshape(images.shape)


def random_shift(images: torch.Tensor, pad: int, generator: torch.Generator) -> torch.Tensor:
    """Translate each camera stream by a few pixels, replicating at the edge.

    The Diffusion Policy configuration this strand inherits sets ``augment:
    true`` on its vision backbone, and image policies for manipulation rely on
    it heavily: without it the encoder can memorize absolute pixel positions
    rather than learning to locate the objects.

    One shift is drawn per sample and camera and shared across the observation
    timesteps, so the augmentation cannot manufacture apparent motion between
    the two frames the policy differences to infer velocity.

    Unlike the other transforms here this one is indifferent to normalization,
    and ``generator`` is a CPU generator: the draw is a handful of integers.
    """
    if pad < 1:
        return images
    batch, steps, channels, height, width = images.shape
    cameras = channels // 3
    folded = images.reshape(batch * steps * cameras, 3, height, width)
    padded = F.pad(folded, (pad, pad, pad, pad), mode="replicate")
    offsets = (
        torch.randint(0, 2 * pad + 1, (batch, 1, cameras, 2), generator=generator, device="cpu")
        .expand(batch, steps, cameras, 2)
        .reshape(-1, 2)
    )
    rows = torch.arange(height, device=images.device)
    columns = torch.arange(width, device=images.device)
    row_index = (offsets[:, 0:1].to(images.device) + rows[None, :]).reshape(-1, height, 1)
    column_index = (offsets[:, 1:2].to(images.device) + columns[None, :]).reshape(-1, 1, width)
    sample = torch.arange(len(folded), device=images.device).reshape(-1, 1, 1)
    cropped = padded[sample, :, row_index, column_index]
    return cropped.permute(0, 3, 1, 2).reshape(batch, steps, channels, height, width)
