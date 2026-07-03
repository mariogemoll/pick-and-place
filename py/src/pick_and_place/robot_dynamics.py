# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Helpers for applying fitted real-robot joint response in simulation."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_ROBOT_DYNAMICS_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "robot_dynamics" / "so101_follower.json"
)


def load_robot_dynamics_config(path: str | Path = DEFAULT_ROBOT_DYNAMICS_PATH) -> dict:
    """Load a fitted robot-dynamics JSON artifact."""
    return json.loads(Path(path).read_text())


def set_actuator_activation(model, data, actuator_id: int, value: float) -> None:
    """Seed a filtered actuator's activation to match its current control.

    MuJoCo ``position`` actuators with ``timeconst`` filter ``ctrl`` through an
    activation state. Initialising that state avoids a fake startup transient
    from zero whenever a scene is reset directly to a pose.
    """
    actadr = int(model.actuator_actadr[actuator_id])
    if actadr >= 0:
        data.act[actadr] = float(value)
