#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One entry point for the project's commands.

    pap --help                     # every command and what it does
    pap eval-policy-sim --help     # that command's leaves
    pap eval-policy-sim scripted --manifest config/evaluation/smoke_v1.json --output out/

**Dispatch is lazy, and that is a hard requirement rather than an optimization.**
``run_policy_sim.py`` imports mujoco, torch and lerobot at module scope, and
sets ``PYTORCH_ENABLE_MPS_FALLBACK`` *before* torch is imported. A dispatcher
that imported every command module to build its command table would take seconds
to print help and would break that ordering. So the table in
``pick_and_place.cli.commands`` is data, and a command's module is imported here
only once the command has been named -- its parser first, and the code that runs
it only after the arguments parse.

``pap --help`` therefore costs an argparse import and nothing else, whatever the
commands behind it drag in.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR.parent / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

from pick_and_place.cli.commands import COMMANDS, COMMANDS_BY_NAME, Command  # noqa: E402
from pick_and_place.cli.suggest import SuggestingArgumentParser  # noqa: E402


def build_parser() -> SuggestingArgumentParser:
    """Return the top-level parser: the command names and nothing else.

    Deliberately not a subparser tree over the real parsers. Building those
    would mean importing every command, which is what lazy dispatch exists to
    avoid; the command's own flags are parsed by the command's own parser once
    the name has been resolved.
    """
    parser = SuggestingArgumentParser(
        prog="pap",
        usage="pap COMMAND [ARGS ...]",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_command_listing(),
    )
    parser.add_argument(
        "command",
        metavar="COMMAND",
        choices=[command.name for command in COMMANDS],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help=argparse.SUPPRESS,
    )
    return parser


def _command_listing() -> str:
    """Render the command table for ``pap --help``.

    Written by hand rather than as an argparse subparser listing: subparsers
    would have to exist, and they cannot exist without importing what is behind
    them.
    """
    width = max(len(command.name) for command in COMMANDS)
    lines = [f"  {command.name:<{width}}  {command.summary}" for command in COMMANDS]
    return "commands:\n" + "\n".join(lines) + "\n\nRun `pap COMMAND --help` for a command's own flags."


def _load_script(relative_path: str) -> ModuleType:
    """Import a file under ``py/scripts`` without treating ``scripts/`` as a package.

    ``scripts/pick_and_place/`` would shadow the installed package if it were
    importable by name, and it has no ``__init__.py`` precisely so that it is
    not. Loading by path sidesteps the question for every command at once.
    """
    path = SCRIPTS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(f"pap_command_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load command from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that a command importing itself -- which
    # the multiprocessing workers in eval_scripted_parallel.py do -- gets this
    # module rather than a second copy of it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dispatch(command: Command, arguments: list[str]) -> None:
    """Parse ``arguments`` with the command's parser, then run it."""
    if command.parser_module is None:
        # The parser lives with the command, so running it costs no second import.
        owner = _load_script(command.script)
        runner: ModuleType | None = owner
    else:
        owner = importlib.import_module(command.parser_module)
        runner = None

    # argparse takes a parser's default prog from ``sys.argv[0]``, and a
    # subparser takes its own from its parent's at the moment it is added. So a
    # leaf's usage line is fixed before ``build_parser`` returns, and assigning
    # ``parser.prog`` afterwards would rename the root and leave every leaf
    # still calling itself eval_policy_sim.py. Setting argv[0] first is the
    # mechanism argparse already uses, rather than a way around it.
    argv0, sys.argv[0] = sys.argv[0], f"pap {command.name}"
    try:
        parser = owner.build_parser()
    finally:
        sys.argv[0] = argv0
    args = parser.parse_args(arguments)

    validate = getattr(owner, "validate", None)
    if validate is not None:
        validate(parser, args)

    if runner is None:
        runner = _load_script(command.script)
    runner.run(args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch(COMMANDS_BY_NAME[args.command], args.arguments)


if __name__ == "__main__":
    main()
