# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The learning-rate schedule the flow trainers run.

Lifted out of ``flow_matching`` when the state policy was deleted: the schedule
was the only thing the image trainer took from that module, and it is not
specific to either policy.
"""

from __future__ import annotations

import math


def learning_rate_at_step(
    step: int, *, num_steps: int, peak: float, minimum: float, warmup_steps: int
) -> float:
    """Linear warmup followed by cosine decay."""
    if num_steps < 1 or not 0 <= step < num_steps:
        raise ValueError("step must be within a positive training run")
    if not 0 <= warmup_steps < num_steps:
        raise ValueError("warmup_steps must be in [0, num_steps)")
    if not 0.0 <= minimum <= peak:
        raise ValueError("minimum must be between zero and peak")
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(num_steps - warmup_steps - 1, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))
