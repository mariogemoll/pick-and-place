# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Resolving a command name to the code behind it, lazily.

Separate from ``scripts/pap.py`` so that the dispatcher and the test that checks
the command table load a command exactly the same way. They did not, at first,
and the difference was invisible until a command defined a dataclass: a module
executed without being registered in ``sys.modules`` leaves
``dataclasses.fields()`` unable to resolve its own class's module, and the
failure surfaces as ``'NoneType' object has no attribute '__dict__'`` from deep
inside the standard library.

Nothing here imports a command at module scope. That is the point: the table in
:mod:`pick_and_place.cli.commands` is data, and this is what turns one entry of
it into code, after the command has been named.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

from pick_and_place.cli.commands import Command
from pick_and_place.core.paths import REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "py" / "scripts"


def load_script(relative_path: str) -> ModuleType:
    """Import a file under ``py/scripts`` without treating ``scripts/`` as a package.

    ``scripts/pick_and_place/`` would shadow the installed package if it were
    importable by name, and it has no ``__init__.py`` precisely so that it is
    not. Loading by path sidesteps the question for every command at once.
    """
    # `python scripts/x.py` puts scripts/ on sys.path, and two commands rely on
    # it: render_apriltag_textures imports the tag codebook from
    # generate_apriltags, and eval_scripted_parallel imports eval_policy_sim to
    # re-run it in worker processes. A dispatcher that claims to run the same
    # command has to give it the same import path. (The first of those two would
    # be better as a fact in the package -- see CLI_SURFACE.md.)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(f"pap_command_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load command from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, not after. A module that defines a dataclass
    # needs to be findable under its own __module__ while its body runs, and a
    # command importing itself -- which eval_scripted_parallel.py's worker
    # processes do -- should get this module rather than a second copy of it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_parser_owner(command: Command) -> ModuleType:
    """Import whatever exposes the command's ``build_parser`` -- and nothing more.

    For most commands that is the script itself, which is cheap. For the ones
    that pull torch or lerobot at module scope it is a module under ``cli/``,
    so asking a command what its flags are costs an argparse import rather than
    a deep-learning stack.
    """
    if command.parser_module is None:
        return load_script(command.script)
    return importlib.import_module(command.parser_module)


def load_runner(command: Command, parser_owner: ModuleType) -> ModuleType:
    """Import what exposes the command's ``run(args)``, reusing the parser's module."""
    if command.parser_module is None:
        return parser_owner
    return load_script(command.script)
