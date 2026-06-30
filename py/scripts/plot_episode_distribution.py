#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Plot the spread of cube source and target poses across a recorded episode dir.

Each episode ``.npz`` stores the initial cube pose (``cube_start`` = x, y, z, yaw)
and the drop target (``cube_target`` = x, y, z, yaw). This reads every episode in
a directory and renders a 2x2 figure: the top row covers the initial cube poses
(a scatter map with yaw drawn as short heading arrows, plus a position heatmap)
and the bottom row covers the targets (scatter map and heatmap). The allowed
sampling zone for each role is shaded underneath so the spread can be read against
the workspace it was drawn from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pick_and_place.episodes import sample_cube, sample_target
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY,
    CUBE_PLACEMENT_BOUNDS,
    CUBE_PLACEMENT_OVERLAY,
    PAN_AXIS,
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
)

# Number of nominal-sampler draws used to trace the "as designed" distribution.
_NOMINAL_SAMPLES = 30000
_NOMINAL_SEED = 0


def radii_azimuths(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (radius, azimuth-deg) of each pose relative to the pan axis."""
    dx = poses[:, 0] - PAN_AXIS[0]
    dy = poses[:, 1] - PAN_AXIS[1]
    return np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))


def nominal_samples(sampler, n: int = _NOMINAL_SAMPLES, seed: int = _NOMINAL_SEED) -> np.ndarray:
    """Draw ``n`` poses straight from a production sampler, as an (n, 4) array.

    This traces the distribution the episode recorder *draws from*, before the
    collision-free trajectory search throws out the source/target pairs it cannot
    reach. Comparing it against the recorded poses exposes the reachable subset.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, 4), dtype=float)
    for i in range(n):
        pose = sampler(rng)
        out[i] = (pose.x, pose.y, pose.z, pose.yaw)
    return out


def load_poses(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (sources, targets), each an (N, 4) array of (x, y, z, yaw)."""
    files = sorted(episode_dir.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"no episode_*.npz files found in {episode_dir}")
    sources = []
    targets = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        sources.append(data["cube_start"])
        targets.append(data["cube_target"])
    return np.asarray(sources, dtype=float), np.asarray(targets, dtype=float)


def allowed_mask(
    bounds: tuple[float, float, float, float], predicate, resolution: int = 400
) -> tuple[np.ndarray, list[float]]:
    """Sample ``predicate(x, y)`` over a grid spanning ``bounds`` for shading."""
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    mask = np.array([[predicate(x, y) for x in xs] for y in ys], dtype=float)
    return mask, [x_min, x_max, y_min, y_max]


def _plot_extent(margin: float = 0.02) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = CUBE_PLACEMENT_BOUNDS
    return (x_min - margin, x_max + margin, y_min - margin, y_max + margin)


def draw_scatter(
    ax,
    poses: np.ndarray,
    predicate,
    *,
    title: str,
    point_color: str,
    show_heading: bool,
) -> None:
    extent = _plot_extent()
    mask, mask_extent = allowed_mask(CUBE_PLACEMENT_BOUNDS, predicate)
    ax.imshow(
        mask,
        origin="lower",
        extent=mask_extent,
        cmap="Greens",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    x, y, yaw = poses[:, 0], poses[:, 1], poses[:, 3]
    if show_heading:
        arrow_len = 0.018
        ax.quiver(
            x,
            y,
            np.cos(yaw) * arrow_len,
            np.sin(yaw) * arrow_len,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            color=point_color,
            alpha=0.7,
        )
    ax.scatter(x, y, s=8, c=point_color, alpha=0.6, edgecolors="none", zorder=3)
    ax.plot(PAN_AXIS[0], PAN_AXIS[1], "k+", markersize=12, markeredgewidth=2, zorder=4)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}  (n={len(poses)})")


def draw_heatmap(ax, poses: np.ndarray, *, title: str, bins: int, cmap: str) -> None:
    extent = _plot_extent()
    hist, xedges, yedges = np.histogram2d(
        poses[:, 0],
        poses[:, 1],
        bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    image = ax.imshow(
        hist.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        aspect="equal",
    )
    ax.plot(PAN_AXIS[0], PAN_AXIS[1], "w+", markersize=12, markeredgewidth=2, zorder=4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="episodes / cell")


def draw_marginal(
    ax,
    realized: np.ndarray,
    nominal: np.ndarray,
    *,
    value_range: tuple[float, float],
    bins: int,
    color: str,
    xlabel: str,
    title: str,
) -> None:
    """Compare the realized (dataset) marginal against the nominal sampler.

    The nominal sampler's density is drawn as a black step outline; the recorded
    poses are a filled histogram. Where the fill falls short of the outline is
    sampling mass the trajectory search discarded as unreachable.
    """
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    ax.hist(
        nominal,
        bins=edges,
        density=True,
        histtype="step",
        color="black",
        linewidth=1.5,
        label="nominal sampler",
    )
    ax.hist(
        realized,
        bins=edges,
        density=True,
        color=color,
        alpha=0.55,
        label="realized (dataset)",
    )
    ax.set_xlim(value_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path, help="directory of episode_*.npz files")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output image path (default: <episode_dir>/pose_distribution.png)",
    )
    parser.add_argument("--bins", type=int, default=40, help="heatmap bins per axis")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    sources, targets = load_poses(args.episode_dir)
    nominal_sources = nominal_samples(sample_cube)
    nominal_targets = nominal_samples(sample_target)

    rs, az_s = radii_azimuths(sources)
    rt, az_t = radii_azimuths(targets)
    nrs, naz_s = radii_azimuths(nominal_sources)
    nrt, naz_t = radii_azimuths(nominal_targets)

    az_range = (-115.0, 115.0)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    # Top row: initial cube poses.
    draw_scatter(
        axes[0, 0],
        sources,
        is_cube_pickup_allowed,
        title="Initial cube poses",
        point_color="tab:blue",
        show_heading=True,
    )
    draw_heatmap(axes[0, 1], sources, title="Initial cube density", bins=args.bins, cmap="viridis")
    draw_marginal(
        axes[0, 2],
        rs,
        nrs,
        value_range=(CANONICAL_PICKUP_OVERLAY.inner_radius, CANONICAL_PICKUP_OVERLAY.outer_radius),
        bins=args.bins,
        color="tab:blue",
        xlabel="radius from pan axis (m)",
        title="Initial radius: realized vs nominal",
    )
    draw_marginal(
        axes[0, 3],
        az_s,
        naz_s,
        value_range=az_range,
        bins=args.bins,
        color="tab:blue",
        xlabel="azimuth (deg)",
        title="Initial azimuth: realized vs nominal",
    )

    # Bottom row: target poses.
    draw_scatter(
        axes[1, 0],
        targets,
        is_cube_drop_allowed,
        title="Target poses",
        point_color="tab:red",
        show_heading=False,
    )
    draw_heatmap(axes[1, 1], targets, title="Target density", bins=args.bins, cmap="magma")
    draw_marginal(
        axes[1, 2],
        rt,
        nrt,
        value_range=(CUBE_PLACEMENT_OVERLAY.inner_radius, CUBE_PLACEMENT_OVERLAY.outer_radius),
        bins=args.bins,
        color="tab:red",
        xlabel="radius from pan axis (m)",
        title="Target radius: realized vs nominal",
    )
    draw_marginal(
        axes[1, 3],
        az_t,
        naz_t,
        value_range=az_range,
        bins=args.bins,
        color="tab:red",
        xlabel="azimuth (deg)",
        title="Target azimuth: realized vs nominal",
    )

    fig.suptitle(f"Episode pose distribution — {args.episode_dir}  ({len(sources)} episodes)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = args.out or (args.episode_dir / "pose_distribution.png")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")
    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()


if __name__ == "__main__":
    main()
