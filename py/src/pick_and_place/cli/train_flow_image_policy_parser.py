# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train an image-conditioned flow-matching policy on a Diffusion Policy export.

The flow objective is the state policy's, unchanged: a Gaussian conditional
optimal-transport path, a velocity target, a 16-step chunk of which 8 are
executed. Only the conditioning differs -- two camera streams and the robot's
own joints, instead of the simulator's privileged cube and target poses.

The schedule follows the state policy's selected recipe, whose 100,000-update
continuation established that this configuration overfits past roughly 30,000:
warmup, then cosine decay reaching its minimum at the final update.
"""

from __future__ import annotations

import argparse

from pick_and_place.cli.training import ImageTrainingRun, parse_training_config


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the trainer's flags through tyro, and return the config it built.

    This is the tree's one command that does not expose ``build_parser()``,
    because it does not have a parser to expose: the trainers take dataclass
    configs through ``tyro``, so a run can be written out and read back with
    ``--config``. There is no ``ArgumentParser`` object anywhere in that path.

    Rather than fake one, ``cli/commands.py`` marks this command
    ``typed_config=True`` and the dispatcher calls this instead. One command in
    fifty carrying its own idiom is worth an explicit flag in the table; hiding
    it behind a shim that pretends to be argparse would not be.

    The return type is a lie of convenience -- it is an ``ImageTrainingRun``, not
    a ``Namespace`` -- but the dispatcher only ever hands it to ``run``.
    """
    return parse_training_config(ImageTrainingRun, argv, description=__doc__)
