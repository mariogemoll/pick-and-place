# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A replay buffer over cached features rather than pixels.

Storing observations would mean two 96x96 three-channel frames at two history
steps, or 110 KB per transition as ``uint8`` -- 11 GB for a hundred thousand
transitions, and that is before the ``next`` observation doubles it. Storing the
frozen diffusion policy's own conditioning vector instead costs about a kilobyte
and loses nothing, because the encoder never trains: the feature is a fixed
function of the observation, so a cached one can never go stale the way a
learned encoder's would.

That is the engineering fact that makes an off-policy method affordable on this
task at all, and it is only available because DSRL freezes the base policy.

**On ``done``.** The environment reports a step limit as terminal rather than
merely truncated, and reuses one slot for the next episode's first observation
after a reset. Both are deliberate: the step budget is part of this task, so a
timed-out episode genuinely has zero future reward, and cutting the bootstrap is
the correct estimate. The consequence here is that the ``next`` features stored
on a terminal transition belong to the following episode -- which is harmless
precisely because ``(1 - done)`` zeroes them out of the target. Nothing else in
this module may assume they are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

_FIELDS = (
    "actor_features",
    "critic_features",
    "action",
    "next_actor_features",
    "next_critic_features",
)


@dataclass(frozen=True)
class ReplaySpec:
    """Widths of everything one transition holds."""

    actor_feature_dim: int
    critic_feature_dim: int
    latent_dim: int
    capacity: int


class ReplayBuffer:
    """A fixed-capacity ring over float32 feature transitions."""

    def __init__(self, spec: ReplaySpec) -> None:
        self.spec = spec
        widths = {
            "actor_features": spec.actor_feature_dim,
            "critic_features": spec.critic_feature_dim,
            "action": spec.latent_dim,
            "next_actor_features": spec.actor_feature_dim,
            "next_critic_features": spec.critic_feature_dim,
        }
        self._data = {
            name: np.zeros((spec.capacity, widths[name]), dtype=np.float32)
            for name in _FIELDS
        }
        self._data["reward"] = np.zeros((spec.capacity, 1), dtype=np.float32)
        self._data["done"] = np.zeros((spec.capacity, 1), dtype=np.float32)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self.spec.capacity

    def add(
        self,
        *,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_actor_features: np.ndarray,
        next_critic_features: np.ndarray,
    ) -> None:
        """Insert a batch of transitions, one per parallel environment.

        Wrapping is handled by splitting the batch at the end of the ring rather
        than by inserting one row at a time.
        """
        batch = len(reward)
        incoming = {
            "actor_features": actor_features,
            "critic_features": critic_features,
            "action": action,
            "next_actor_features": next_actor_features,
            "next_critic_features": next_critic_features,
            "reward": np.asarray(reward, dtype=np.float32).reshape(batch, 1),
            "done": np.asarray(done, dtype=np.float32).reshape(batch, 1),
        }
        if batch > self.spec.capacity:
            raise ValueError(
                f"cannot insert {batch} transitions into a buffer of "
                f"{self.spec.capacity}"
            )
        start = self._cursor
        first = min(batch, self.spec.capacity - start)
        for name, values in incoming.items():
            values = np.asarray(values, dtype=np.float32)
            self._data[name][start : start + first] = values[:first]
            if first < batch:
                self._data[name][: batch - first] = values[first:]
        self._cursor = (start + batch) % self.spec.capacity
        self._size = min(self._size + batch, self.spec.capacity)

    def sample(
        self, batch_size: int, device: torch.device, generator: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Draw ``batch_size`` transitions uniformly, with replacement."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = generator.integers(0, self._size, size=batch_size)
        return {
            name: torch.from_numpy(values[indices]).to(device, non_blocking=True)
            for name, values in self._data.items()
        }

    def bytes_per_transition(self) -> int:
        """What one transition costs, for sizing a run before it starts."""
        return sum(values.shape[1] for values in self._data.values()) * 4
