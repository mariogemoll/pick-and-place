# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The command table ``pap`` dispatches from: names, one-line summaries, and where to look.

**Nothing here imports a command.** The table is data, so ``pap --help`` can
list fifty commands and their summaries without importing torch, lerobot or a
MuJoCo scene -- which the dispatcher would otherwise have to do just to ask each
command what it is called. That is the whole reason this is a table rather than
a registry built by decorators: a decorator has to run, and running it means
importing the module it decorates.

The cost is that a summary lives here rather than beside its command, and can go
stale. ``tests/test_commands.py`` checks the table against the tree instead:
every command resolves, every script is either registered or deliberately not.

Each command names two things the dispatcher imports **after** parsing:

``parser``
    the module exposing ``build_parser()``, and optionally ``validate(parser,
    args)``. It defaults to the script itself, which is right whenever importing
    the script is cheap. Where it is not -- the commands that pull torch or
    lerobot at module scope -- the parser lives in ``pick_and_place.cli.<name>``
    and this field names it, so ``pap <command> --help`` costs an argparse
    import rather than a deep-learning stack.

``script``
    the file exposing ``run(args)``, relative to ``py/scripts``. Imported by
    path rather than by name, because ``scripts/`` is not a package and
    ``scripts/pick_and_place/`` would shadow the real one if it were.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One entry in the ``pap`` tree."""

    name: str
    summary: str
    script: str
    parser: str | None = None

    @property
    def parser_module(self) -> str | None:
        """The importable module holding ``build_parser``, or ``None`` for the script."""
        return self.parser


#: Every command ``pap`` offers, in the order ``pap --help`` lists them.
COMMANDS: tuple[Command, ...] = (
    Command(
        name="eval-policy-sim",
        summary="Score a controller against a frozen scenario manifest.",
        script="eval_policy_sim.py",
        parser="pick_and_place.cli.eval_policy_sim",
    ),
    Command(
        name="select-episodes",
        summary="List a dataset's episodes that pass a success filter.",
        script="select_episodes.py",
    ),
    Command(
        name="combine-datasets",
        summary="Merge several LeRobot datasets into one.",
        script="combine_datasets.py",
    ),
    Command(
        name="consolidate-datasets",
        summary="Merge run directories into one dataset per day.",
        script="consolidate_datasets.py",
    ),
    Command(
        name="split-train-val-episodes",
        summary="Split a dataset's episodes into train and validation sets.",
        script="split_train_val_episodes.py",
    ),
)

COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}
