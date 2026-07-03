#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Plot the spread of cube source and target poses across recorded episodes.

Supports both raw ``episode_*.npz`` directories and LeRobotDataset roots whose
episode metadata contains scalar ``cube_start_*`` and ``target_*`` columns.
Passing a parent directory such as ``datasets/`` aggregates all LeRobotDataset
children under it.
The figure's top row covers initial cube poses (scatter with yaw heading arrows,
plus a position heatmap); the bottom row covers drop targets. The allowed
sampling zone for each role is shaded underneath so the spread can be read
against the workspace it was drawn from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pick_and_place.episodes import sample_cube, sample_target
from pick_and_place.geometry import CUBE_HALF_SIZE
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


def _pose4(values: np.ndarray) -> np.ndarray:
    """Normalize a saved pose vector to (x, y, z, yaw)."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] >= 4:
        return values[[0, 1, 2, 3]]
    if values.shape[0] == 3:
        return np.array([values[0], values[1], values[2], 0.0], dtype=float)
    raise ValueError(f"expected 3 or 4 pose values, got {values.shape[0]}")


def _load_npz_poses(episode_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    files = sorted(episode_dir.glob("episode_*.npz"))
    if not files:
        return None
    sources = []
    targets = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        sources.append(_pose4(data["cube_start"]))
        targets.append(_pose4(data["cube_target"]))
    return np.asarray(sources, dtype=float), np.asarray(targets, dtype=float)


def _load_lerobot_poses(dataset_root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    files = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not files:
        return None
    df = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    required = {"cube_start_x", "cube_start_y", "cube_start_yaw", "target_x", "target_y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"{dataset_root} is a LeRobotDataset but is missing pose metadata: "
            f"{', '.join(missing)}"
        )
    n = len(df)
    # The cube always rests flat, so z is the constant half-size and the target
    # marker carries no yaw; fill both to keep the (x, y, z, yaw) contract.
    z = np.full(n, CUBE_HALF_SIZE)
    sources = np.column_stack(
        [df["cube_start_x"], df["cube_start_y"], z, df["cube_start_yaw"]]
    ).astype(float)
    targets = np.column_stack(
        [df["target_x"], df["target_y"], z, np.zeros(n)]
    ).astype(float)
    finite = np.isfinite(sources).all(axis=1) & np.isfinite(targets).all(axis=1)
    if not finite.all():
        raise SystemExit(f"{dataset_root} contains {int((~finite).sum())} row(s) with missing poses")
    return sources, targets


def _lerobot_children(parent: Path) -> list[Path]:
    """Return child LeRobotDataset roots under ``parent``."""
    return sorted(path.parent.parent for path in parent.glob("*/meta/info.json"))


def load_poses(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (sources, targets), each an (N, 4) array of (x, y, z, yaw)."""
    poses = _load_npz_poses(episode_dir)
    if poses is not None:
        return poses
    poses = _load_lerobot_poses(episode_dir)
    if poses is not None:
        return poses
    raise SystemExit(f"no episode_*.npz files or LeRobot episode metadata found in {episode_dir}")


def load_many(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load and concatenate poses from paths or parent directories."""
    sources = []
    targets = []
    loaded = []
    for path in paths:
        children = _lerobot_children(path)
        candidates = children if children else [path]
        for candidate in candidates:
            src, tgt = load_poses(candidate)
            sources.append(src)
            targets.append(tgt)
            loaded.append(candidate)
    if not sources:
        raise SystemExit("no episode metadata found")
    return np.concatenate(sources), np.concatenate(targets), loaded


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
    parser.add_argument(
        "episode_dirs",
        nargs="+",
        type=Path,
        help="episode dir(s), LeRobotDataset root(s), or parent dirs such as datasets/",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output image path (default: <first input>/pose_distribution.png)",
    )
    parser.add_argument("--bins", type=int, default=40, help="heatmap bins per axis")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    sources, targets, loaded = load_many(args.episode_dirs)
    if len(loaded) > 1:
        print(f"loaded {len(loaded)} dataset(s), {len(sources)} episode(s)")
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

    label = str(args.episode_dirs[0]) if len(args.episode_dirs) == 1 else f"{len(loaded)} datasets"
    fig.suptitle(f"Episode pose distribution — {label}  ({len(sources)} episodes)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = args.out or (args.episode_dirs[0] / "pose_distribution.png")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")
    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()


if __name__ == "__main__":
    main()
