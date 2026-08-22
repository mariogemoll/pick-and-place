# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One rig, recorded once and applied to everything that runs against it.

``pap freeze-scenario-rig`` rewrites an evaluation suite so every
scene faces the same draw of the robot, the cameras and the physics, and leaves
a ``*.frozen_rig.json`` sidecar saying which draw that was. Scoring reads the
rewritten suite; anything else that wants the same rig -- recording
demonstrations on it, above all -- reads the sidecar.

The split the sidecar records is the point. A rig is what an installation pins:
where the cameras sit, what the room is made of, how far the joint zeros are
off, how the arm responds. What it does not pin is the session -- the light
moves with the hour, the cameras' auto exposure follows it, the sensor noise is
noise, and the cube lands however it lands. Those fields are named in
``varied_fields`` and keep drawing per episode, so a dataset recorded here is
"this robot, many sessions" rather than "this robot, one frozen photograph".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.core.miscalibration import miscalibration_from_payload
from pick_and_place.core.physics import PhysicsDraw
from pick_and_place.sim.domain_randomization import (
    DomainSample,
    domain_sample_fields,
    domain_sample_from_payload,
)


@dataclass(frozen=True)
class FrozenRig:
    """The three blocks that together are one simulated installation."""

    label: str
    source: str
    varied_fields: frozenset[str]
    domain_sample: dict[str, Any]
    miscalibration_sample: dict[str, Any]
    physics: PhysicsDraw

    @classmethod
    def load(cls, path: Path) -> "FrozenRig":
        payload = json.loads(Path(path).read_text())
        expected = {
            "suite",
            "source",
            "varied_fields",
            "domain_randomization_sample",
            "miscalibration_sample",
            "physics_sample",
        }
        if set(payload) != expected:
            raise ValueError(
                f"{path} is not a frozen-rig sidecar; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        domain_sample = dict(payload["domain_randomization_sample"])
        if not domain_sample.pop("enabled", False):
            # The nominal rig is recorded as ``enabled: false`` with no sampled
            # values, because "no randomization" is authored rather than drawn.
            # Recording against it is what the recorder already does when no
            # preset is given, so there is nothing here to apply.
            raise ValueError(
                f"{path} freezes a rig with domain randomization disabled; "
                "record without --frozen-rig to get the authored rig"
            )
        known = domain_sample_fields()
        varied = frozenset(str(name) for name in payload["varied_fields"])
        unknown = varied - known
        if unknown:
            raise ValueError(f"{path} varies fields no domain sample carries: {sorted(unknown)}")
        # Parsed now, with the sidecar's own jitter stream, so a malformed
        # sidecar fails at load rather than on the first episode of a long run.
        miscalibration_from_payload(payload["miscalibration_sample"], context=str(path))
        return cls(
            label=str(payload["suite"]),
            source=str(payload["source"]),
            varied_fields=varied,
            domain_sample=domain_sample,
            miscalibration_sample=dict(payload["miscalibration_sample"]),
            physics=PhysicsDraw.from_payload(payload["physics_sample"], context=str(path)),
        )

    def session(self, sample: DomainSample, rng: np.random.Generator) -> DomainSample:
        """This rig, wearing one session's worth of the draw in ``sample``.

        Every field but ``varied_fields`` comes from the rig -- the freeze
        script's rule run the other way round, so a dataset recorded here and a
        suite scored there are the same installation. The rig's miscalibration
        replaces the draw's, except for the shoulder-pan wander, whose
        realization is redrawn from ``rng``: ``sigma_deg`` and ``tau_s`` belong
        to the arm, but replaying the sidecar's own curve would put one
        identical wander under every episode of a dataset, which a policy can
        learn instead of learning to correct for wander.
        """
        miscalibration = miscalibration_from_payload(
            self.miscalibration_sample, context=self.label, jitter_rng=rng
        )
        payload = {
            name: (
                getattr(sample, name) if name in self.varied_fields else self.domain_sample[name]
            )
            for name in domain_sample_fields()
        }
        return domain_sample_from_payload(payload, miscalibration, context=self.label)
