# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What a policy's action means relative to the joints it was measured from.

A recorded control tick holds the measured joints and the command issued from
them. A dataset export can hand the policy either one:

- ``absolute`` -- the command itself, so one normalized unit spans a joint's
  whole range (~180 degrees for j1) and a small change in the normalized action
  is a large change in behavior;
- ``delta`` -- the command minus the measured joints of the same tick, so a
  normalized unit spans the range of one tick's motion instead.

Both sides of that choice have to agree or the policy is silently fed the wrong
units, so the encoding travels with the normalization bounds and is read back
from them rather than assumed. It lives here, below every branch, because the
exporter that writes it and the rollout paths that decode it must not import
each other.

A delta is defined against the measured joints of the tick it is commanded on,
which is exactly how the demonstration recorded it. Executing a chunk therefore
integrates each action onto the freshest measurement rather than onto the state
the chunk was predicted from.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np

# The key the encoding travels under, in both the export manifest and the
# normalization archive beside it.
ACTION_ENCODING_KEY = "action_encoding"


class ActionEncoding(Enum):
    """How to read an action against the state it was predicted from."""

    ABSOLUTE = "absolute"
    DELTA = "delta"


def parse_action_encoding(value: str) -> ActionEncoding:
    """Resolve a manifest string, naming the alternatives when it is not one."""
    try:
        return ActionEncoding(value)
    except ValueError:
        known = ", ".join(encoding.value for encoding in ActionEncoding)
        raise ValueError(f"unknown action encoding {value!r}; expected one of {known}") from None


def read_action_encoding(bounds: Mapping[str, Any]) -> ActionEncoding:
    """Read the encoding a ``normalization.npz`` was written with.

    Exports written before the encoding existed carry absolute joint commands,
    which is what their absence means.
    """
    if ACTION_ENCODING_KEY not in bounds:
        return ActionEncoding.ABSOLUTE
    return parse_action_encoding(str(np.asarray(bounds[ACTION_ENCODING_KEY]).item()))


def encode_actions(
    encoding: ActionEncoding, actions: np.ndarray, states: np.ndarray
) -> np.ndarray:
    """Express recorded commands in ``encoding``, against their own ticks' states.

    ``actions`` and ``states`` are aligned row for row: row ``t`` holds the
    command issued at control tick ``t`` and the joints measured there.
    """
    if actions.shape != states.shape:
        raise ValueError(
            f"actions and states must be aligned row for row, got {actions.shape} and {states.shape}"
        )
    if encoding is ActionEncoding.ABSOLUTE:
        return actions
    return actions - states


def decode_actions(
    encoding: ActionEncoding, actions: np.ndarray, state: np.ndarray
) -> np.ndarray:
    """Turn actions in ``encoding`` into joint commands, given the measured joints.

    ``state`` is the measurement the action is defined against: the joints read
    on the tick the action is commanded on. Passing a chunk with one state
    integrates every step of it onto that one measurement, which is the
    open-loop reading of a chunk and not how a chunk is executed -- execution
    decodes one action at a time, against the tick it belongs to.
    """
    if encoding is ActionEncoding.ABSOLUTE:
        return actions
    return actions + state
