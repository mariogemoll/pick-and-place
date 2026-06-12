# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export composed SO-101 models as standalone MJCF files.

For consumers that need a file on disk (MuJoCo's ``simulate`` viewer, the
Isaac MJCF importer). The exported XML carries an absolute meshdir, so it is
machine-local; treat it as a build artifact, not something to commit.

Usage::

    python -m pick_and_place.export [-o OUTPUT]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot


def export_robot(output: Path) -> Path:
    spec = build_robot()
    # The spec was loaded relative to the stock model; rewrite meshdir so the
    # exported file resolves meshes from wherever it is saved.
    spec.meshdir = str(STOCK_ASSETS_DIR)
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None, help="output XML path")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parents[2] / "out" / "so101.xml"

    path = export_robot(output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
