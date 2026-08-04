# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Filesystem roots for machine-local data.

Datasets, checkpoints, renders, and recordings do not belong inside the source
tree: they are large, they are regenerated, and they go stale faster than the
code that produced them. ``PAP_DATA_ROOT`` names a single directory outside the
repository where all of it lives.

Roots are resolved lazily, at the moment a default is actually needed, so an
explicit command-line path keeps working whether or not the variable is set.
The idiom at a call site is::

    root = args.dataset_root or datasets_root()

rather than calling ``datasets_root()`` to build an ``argparse`` default, which
would demand the variable even when the caller supplies a path.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The checkout itself, which carries the committed configuration, the vendored
#: hardware submodule, and the printable geometry.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Environment variable naming the machine-local data directory.
ENV_VAR = "PAP_DATA_ROOT"


class DataRootNotConfigured(RuntimeError):
    """A default path was needed but :data:`ENV_VAR` is unset or empty."""

    def __init__(self) -> None:
        super().__init__(
            f"{ENV_VAR} is not set. Point it at a directory outside the repository:\n"
            f"    export {ENV_VAR}=~/pick-and-place-data\n"
            "or pass an explicit path on the command line."
        )


def data_root() -> Path:
    """Return the machine-local data directory.

    Raises :class:`DataRootNotConfigured` if the variable is unset or empty.
    The directory is not created and need not exist yet; callers that write
    into it are responsible for creating what they need.
    """
    value = os.environ.get(ENV_VAR, "").strip()
    if not value:
        raise DataRootNotConfigured
    return Path(value).expanduser()


def datasets_root() -> Path:
    """Return where recorded and simulated datasets live."""
    return data_root() / "datasets"


def outputs_root() -> Path:
    """Return where training runs, checkpoints, renders, and reports live."""
    return data_root() / "outputs"
