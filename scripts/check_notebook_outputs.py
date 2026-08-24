#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fail when a committed notebook carries stored outputs or an execution count.

A notebook's outputs are base64 images and embedded video. They are unreadable
in a diff, they make review of the actual change impossible, and one run with
videos displayed is megabytes against the repository's 40 KB per-file ceiling.
The source is the artifact here; the outputs belong to whoever runs it.

Strip them before committing:

    jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def offending_cells(notebook: dict) -> list[str]:
    """Return a description of every cell holding output state."""
    problems = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            problems.append(f"cell {index}: {len(cell['outputs'])} stored output(s)")
        if cell.get("execution_count") is not None:
            problems.append(f"cell {index}: execution_count {cell['execution_count']}")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.ipynb"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    paths = [root / name for name in listed.split("\0") if name]

    failed = False
    for path in paths:
        try:
            notebook = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            print(f"{path.relative_to(root)}: not valid JSON ({error})")
            failed = True
            continue
        for problem in offending_cells(notebook):
            print(f"{path.relative_to(root)}: {problem}")
            failed = True

    if failed:
        print("\nStrip them: jupyter nbconvert --clear-output --inplace notebooks/*.ipynb")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
