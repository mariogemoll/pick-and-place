# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Gray code structured light: which projector cell lit each camera pixel.

A ChArUco board fails on this rig because the projector cannot bring the whole
workspace into focus at once -- aimed obliquely, only a stripe of the table sits
near its focal plane, and marker bits outside that stripe blur into each other.
Feature size does not rescue it: coarser boards detect *worse*, because the
limit is that a marker must be decoded from fine detail wherever it happens to
land.

Gray code has no fine detail to lose. Each pattern is a single black/white
stripe field, and a camera pixel only has to answer which of two exposures was
brighter. Blur costs resolution -- it sets how narrow a stripe can get before
the answer stops being reliable -- but it never costs *coverage*, so
correspondences come back from the blurred margins along with the sharp centre.

Two properties do the work:

**Every pattern is projected with its inverse**, and a pixel is classified by
comparing the pair rather than against any threshold. The workspace is brown
cardboard with a wooden frame, seams, and a cable or two, so albedo varies
several-fold across the field; a global threshold would read that variation as
signal. A per-pixel comparison cancels it exactly.

**The code is Gray, not binary**, so adjacent cells differ in exactly one bit.
A camera pixel straddling a stripe boundary -- which blur makes common -- then
misreads at most one bit and lands in a neighbouring cell, instead of the
plain-binary case where a boundary at the top bit flips the answer to the far
side of the field.

Cells, not pixels: the field is divided into ``stripe_px`` blocks and only the
block index is encoded. Going finer costs two more exposures per axis and buys
nothing, since a homography needs eight numbers and the blur floor is reached
long before single-pixel stripes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

#: Stripe width in projector pixels. 32 survives heavy defocus and still gives a
#: 60x34 grid on a 1080p frame, which is far more than an 8-DOF fit needs.
DEFAULT_STRIPE_PX = 32


@dataclass(frozen=True)
class GrayCodePlan:
    """How a frame is divided into coded cells, and how many exposures that costs."""

    frame_width: int
    frame_height: int
    stripe_px: int = DEFAULT_STRIPE_PX

    def __post_init__(self) -> None:
        if self.stripe_px < 1:
            raise ValueError(f"stripe_px must be positive, got {self.stripe_px}")
        if self.frame_width < 1 or self.frame_height < 1:
            raise ValueError("frame must have positive extent")

    @property
    def x_cells(self) -> int:
        """Coded cells across."""
        return -(-self.frame_width // self.stripe_px)

    @property
    def y_cells(self) -> int:
        """Coded cells down."""
        return -(-self.frame_height // self.stripe_px)

    @property
    def x_bits(self) -> int:
        """Exposures needed to index a column cell."""
        return max(1, (self.x_cells - 1).bit_length())

    @property
    def y_bits(self) -> int:
        """Exposures needed to index a row cell."""
        return max(1, (self.y_cells - 1).bit_length())

    @property
    def frame_count(self) -> int:
        """Total exposures: a white and a black reference, then each bit and its inverse."""
        return 2 + 2 * (self.x_bits + self.y_bits)


def _to_gray(values: NDArray) -> NDArray:
    """Binary to reflected Gray code."""
    return values ^ (values >> 1)


def _from_gray(gray: NDArray, bits: int) -> NDArray:
    """Reflected Gray code back to binary, by prefix-xor doubling."""
    out = gray.copy()
    shift = 1
    while shift < bits:
        out ^= out >> shift
        shift <<= 1
    return out


def _bit_plane(plan: GrayCodePlan, axis: str, bit: int) -> NDArray[np.uint8]:
    """The 0/255 stripe field for one bit of one axis, as a full frame."""
    if axis == "x":
        extent, bits = plan.frame_width, plan.x_bits
    else:
        extent, bits = plan.frame_height, plan.y_bits

    index = np.arange(extent) // plan.stripe_px
    # Most significant bit first, so truncating the sequence would still leave a
    # usable coarse answer rather than a scrambled one.
    on = (_to_gray(index) >> (bits - 1 - bit)) & 1
    line = (on * 255).astype(np.uint8)

    if axis == "x":
        return np.broadcast_to(line[None, :], (plan.frame_height, plan.frame_width))
    return np.broadcast_to(line[:, None], (plan.frame_height, plan.frame_width))


def frames(plan: GrayCodePlan) -> Iterator[NDArray[np.uint8]]:
    """Yield the exposure sequence as BGR frames, in the order ``decode`` expects.

    White reference, black reference, then for x and then y, each bit followed
    immediately by its inverse.
    """
    white = np.full((plan.frame_height, plan.frame_width), 255, dtype=np.uint8)
    black = np.zeros((plan.frame_height, plan.frame_width), dtype=np.uint8)
    yield np.dstack([white] * 3)
    yield np.dstack([black] * 3)

    for axis, bits in (("x", plan.x_bits), ("y", plan.y_bits)):
        for bit in range(bits):
            plane = _bit_plane(plan, axis, bit)
            yield np.dstack([plane] * 3)
            yield np.dstack([255 - plane] * 3)


@dataclass(frozen=True)
class Decoded:
    """Per-camera-pixel projector cell indices, and where they can be trusted."""

    cell_x: NDArray[np.int64]
    cell_y: NDArray[np.int64]
    valid: NDArray[np.bool_]

    @property
    def coverage(self) -> float:
        """Fraction of camera pixels that decoded to a projector cell."""
        return float(self.valid.mean())


def decode(
    captures: list[NDArray],
    plan: GrayCodePlan,
    *,
    min_contrast: int = 20,
    min_margin: int = 5,
) -> Decoded:
    """Decode a captured exposure sequence into projector cell indices.

    ``captures`` are grayscale camera frames in the order :func:`frames` yields
    them. ``min_contrast`` rejects pixels the projector never reaches -- outside
    the throw, or in the arm's shadow -- by how little the white and black
    references differ there. ``min_margin`` rejects a bit whose pattern and
    inverse came back too close to call.
    """
    if len(captures) != plan.frame_count:
        raise ValueError(f"expected {plan.frame_count} captures, got {len(captures)}")

    white = captures[0].astype(np.int32)
    black = captures[1].astype(np.int32)
    valid = (white - black) > min_contrast

    cells: dict[str, NDArray] = {}
    index = 2
    for axis, bits in (("x", plan.x_bits), ("y", plan.y_bits)):
        gray = np.zeros(white.shape, dtype=np.int64)
        for _ in range(bits):
            difference = captures[index].astype(np.int32) - captures[index + 1].astype(np.int32)
            index += 2
            valid &= np.abs(difference) > min_margin
            gray = (gray << 1) | (difference > 0)
        cells[axis] = _from_gray(gray, bits)

    # A Gray misread can still land outside the used range when cells are not a
    # power of two, which is most of the time.
    valid &= cells["x"] < plan.x_cells
    valid &= cells["y"] < plan.y_cells
    return Decoded(cell_x=cells["x"], cell_y=cells["y"], valid=valid)


@dataclass(frozen=True)
class Correspondences:
    """Matched projector and camera points, one per decoded projector cell."""

    projector_xy: NDArray[np.float64]
    camera_xy: NDArray[np.float64]
    pixel_count: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.projector_xy.shape[0])


def correspondences(
    decoded: Decoded, plan: GrayCodePlan, *, min_pixels: int = 20
) -> Correspondences:
    """Reduce decoded pixels to one camera point per projector cell.

    Every camera pixel that decoded to a cell is averaged into a single centroid
    for it, which is where the sub-pixel accuracy comes from: a 32 px cell covers
    a few hundred camera pixels, and averaging them beats locating any one edge.
    Cells with fewer than ``min_pixels`` behind them are dropped -- those sit on
    a shadow boundary or the edge of the throw, where the centroid is biased by
    whichever part of the cell was actually lit.
    """
    flat = (decoded.cell_y * plan.x_cells + decoded.cell_x)[decoded.valid]
    rows, columns = np.nonzero(decoded.valid)

    size = plan.x_cells * plan.y_cells
    counts = np.bincount(flat, minlength=size)
    sum_x = np.bincount(flat, weights=columns.astype(float), minlength=size)
    sum_y = np.bincount(flat, weights=rows.astype(float), minlength=size)

    keep = np.nonzero(counts >= min_pixels)[0]
    camera_xy = np.stack([sum_x[keep] / counts[keep], sum_y[keep] / counts[keep]], axis=1)

    cell_x = keep % plan.x_cells
    cell_y = keep // plan.x_cells
    projector_xy = (
        np.stack([cell_x, cell_y], axis=1).astype(float) + 0.5
    ) * plan.stripe_px

    return Correspondences(
        projector_xy=projector_xy, camera_xy=camera_xy, pixel_count=counts[keep]
    )
