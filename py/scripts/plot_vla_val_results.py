#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Plot validation-loss results emitted by eval_vla_val_loss.py."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"/checkpoints/([^/]+)/pretrained_model$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path("results.json"),
        help="results JSON from eval_vla_val_loss.py (default: results.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <results stem>.png next to results)",
    )
    parser.add_argument("--title", default="SmolVLA Validation Loss")
    return parser.parse_args()


def checkpoint_label(row: dict) -> str:
    match = STEP_RE.search(str(row["checkpoint"]))
    return match.group(1) if match else Path(row["checkpoint"]).parent.name


def checkpoint_step(row: dict) -> int | None:
    label = checkpoint_label(row)
    return int(label) if label.isdigit() else None


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else [payload]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    for row in rows:
        if "checkpoint" not in row or "loss" not in row:
            raise ValueError("Each result row must contain checkpoint and loss")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    output = args.output or args.results.with_suffix(".png")

    numeric_rows = sorted(
        (row for row in rows if checkpoint_step(row) is not None),
        key=lambda row: checkpoint_step(row) or 0,
    )
    extra_rows = [row for row in rows if checkpoint_step(row) is None]
    if not numeric_rows:
        raise ValueError("No numeric checkpoint rows found to plot")

    steps = [checkpoint_step(row) or 0 for row in numeric_rows]
    losses = [float(row["loss"]) for row in numeric_rows]
    labels = [checkpoint_label(row) for row in numeric_rows]

    best_idx = min(range(len(losses)), key=losses.__getitem__)
    best_step = steps[best_idx]
    best_loss = losses[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, marker="o", linewidth=2.0, color="#2563eb")
    ax.scatter([best_step], [best_loss], s=90, color="#dc2626", zorder=3)
    ax.annotate(
        f"best {best_step}: {best_loss:.4f}",
        (best_step, best_loss),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=9,
        color="#7f1d1d",
    )

    ax.set_title(args.title)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("validation loss")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    frames = sorted({int(row.get("frames_scored", 0)) for row in rows if row.get("frames_scored")})
    batches = sorted({int(row.get("batches", 0)) for row in rows if row.get("batches")})
    subtitle_parts = []
    if frames:
        subtitle_parts.append(f"frames: {frames[0]}" if len(frames) == 1 else "mixed frame counts")
    if batches:
        subtitle_parts.append(f"batches: {batches[0]}" if len(batches) == 1 else "mixed batch counts")
    if extra_rows:
        extras = ", ".join(f"{checkpoint_label(row)}={float(row['loss']):.4f}" for row in extra_rows)
        subtitle_parts.append(f"extra: {extras}")
    if subtitle_parts:
        ax.text(
            0.0,
            -0.22,
            " | ".join(subtitle_parts),
            transform=ax.transAxes,
            fontsize=9,
            color="#475569",
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")
    print(f"Best numeric checkpoint: {best_step} loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
