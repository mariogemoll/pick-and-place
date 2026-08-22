# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Resolving a command name to the code behind it, lazily.

Separate from ``pap.py`` so that the dispatcher and the test that checks the
command table load a command exactly the same way.

Nothing here imports a command at module scope. That is the point: the table in
:mod:`pick_and_place.cli.commands` is data, and this is what turns one entry of
it into code, after the command has been named.

This module used to be much larger. A command was a file under ``py/scripts``,
which is not a package, so resolving one meant ``spec_from_file_location``, a
hand-registration in ``sys.modules`` -- without which a command defining a
dataclass died with ``'NoneType' object has no attribute '__dict__'`` from deep
inside the standard library -- and a ``sys.path`` insertion so that the two
commands importing a third kept working. The commands are modules now, and
``importlib.import_module`` is the whole of it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from pick_and_place.cli.commands import Command


def load_command(command: Command) -> ModuleType:
    """Import the module exposing the command's ``run(args)``."""
    return importlib.import_module(command.module)


def load_parser_owner(command: Command) -> ModuleType:
    """Import whatever exposes the command's ``build_parser`` -- and nothing more.

    For most commands that is the command's own module, which is cheap. For the
    ones that pull torch or lerobot at module scope it is a second module, so
    asking a command what its flags are costs an argparse import rather than a
    deep-learning stack.
    """
    if command.parser_module is None:
        return load_command(command)
    return importlib.import_module(command.parser_module)


def load_runner(command: Command, parser_owner: ModuleType) -> ModuleType:
    """Import what exposes the command's ``run(args)``, reusing the parser's module."""
    if command.parser_module is None:
        return parser_owner
    return load_command(command)


def parse_arguments(command: Command, parser_owner: ModuleType, arguments: list[str]):
    """Parse a command's arguments with whichever seam it exposes.

    Returns ``(args, parser)``. ``parser`` is ``None`` for the typed-config
    command, which has none -- callers use it only to run ``validate``, and that
    command has nothing to validate that ``tyro`` has not already.
    """
    if command.typed_config:
        return parser_owner.parse_arguments(arguments), None
    parser = parser_owner.build_parser()
    return parser.parse_args(arguments), parser
