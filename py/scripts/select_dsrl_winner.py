#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick a winner across DSRL sweeps, and say how optimistic the pick is.

Reads the JSONs `sweep_dsrl_checkpoints.py` writes -- one per run -- prints every
(run, checkpoint) cell it was given, and names the best. The point of doing it in
one place is the accounting: selecting the maximum over many cells is optimistic
by construction, and the size of that bias depends on how many cells were looked
at, which is exactly the number a reader needs and the number that is easiest to
leave out.

    python scripts/select_dsrl_winner.py sweep-a.json sweep-b.json ...

The winner it names is a *candidate*, not a result. It has to be re-measured on
a scene block that took no part in the selection before it means anything; the
DPPO strand selected over 18 cells and then validated on 512 untouched episodes,
and that is the bar here too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument(
        "--min-n",
        type=int,
        default=100,
        help="ignore cells paired on fewer than this many shared scenarios",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cells: list[tuple[str, str, dict]] = []
    for path in args.sweeps:
        payload = json.loads(path.read_text())
        run = Path(payload["config"].get("actors") or path.stem).parts[-2:]
        run_name = run[0] if run else path.stem
        for label, comparison in payload.get("comparisons", {}).items():
            if comparison["n"] >= args.min_n:
                cells.append((run_name, label, comparison))

    if not cells:
        raise SystemExit("no comparable cells found")

    cells.sort(key=lambda c: c[2]["success"] - c[2]["base_success"], reverse=True)
    width = max(len(f"{run}/{label}") for run, label, _ in cells)
    print(f"{'cell':{width}}  {'base':>6} {'steer':>6} {'delta':>7}  fixed/broke   p")
    for run, label, c in cells:
        delta = c["success"] - c["base_success"]
        print(
            f"{run + '/' + label:{width}}  {c['base_success']:6.3f} {c['success']:6.3f} "
            f"{delta:+7.3f}  {c['fixed']:4d}/{c['broke']:<4d}  {c['mcnemar_p']:.4g}"
        )

    run, label, best = cells[0]
    delta = best["success"] - best["base_success"]
    print()
    print(f"Best of {len(cells)} cells: {run} {label}")
    print(f"  {best['base_success']:.3f} -> {best['success']:.3f} ({delta:+.3f}), "
          f"fixed {best['fixed']} / broke {best['broke']}, p = {best['mcnemar_p']:.4g}")
    print()
    print(
        f"This is a maximum over {len(cells)} cells and is optimistic by "
        "construction. Re-score it on a scene block used for neither training nor\n"
        "selection before reporting it. Nothing below that bar is a result."
    )


if __name__ == "__main__":
    main()
