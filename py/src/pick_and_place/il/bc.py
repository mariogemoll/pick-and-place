# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A tiny behavior-cloning MLP and its from-scratch training loop.

Deliberately minimal — no LeRobot, no action chunking, no sequence model. Rung 1
of the guide is "the simplest thing that mostly works, so the compounding-error
failure shows up live." That machinery arrives in rung 2.

The policy standardizes observations and actions with statistics computed from the
training set (an MLP trains poorly on raw radians-and-metres of wildly different
scale); the stats travel inside the checkpoint so inference reproduces them
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pick_and_place.il.observations import ACT_DIM, OBS_DIM


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a torch device. ``auto`` prefers CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    """Plain feed-forward net: ``OBS_DIM -> hidden... -> ACT_DIM``."""

    def __init__(self, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = OBS_DIM
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, ACT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Normalizer:
    """Per-dimension standardization: ``(x - mean) / std``."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0  # guard constant dims (e.g. a fixed gripper channel)
        return cls(mean.astype(np.float32), std.astype(np.float32))


class BCPolicy:
    """A trained MLP behind the :class:`~pick_and_place.il.policy.Policy` interface.

    Holds the network plus the observation/action normalizers, so ``act`` takes a
    raw observation and returns a raw action with no caller-side bookkeeping.
    """

    def __init__(
        self,
        model: MLP,
        obs_norm: Normalizer,
        act_norm: Normalizer,
        hidden: tuple[int, ...],
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.hidden = hidden

    def reset(self) -> None:  # noqa: D102 - stateless, nothing to clear
        pass

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        z = (np.asarray(observation, dtype=np.float32) - self.obs_norm.mean) / self.obs_norm.std
        out = self.model(torch.from_numpy(z).to(self.device).unsqueeze(0)).squeeze(0)
        out = out.cpu().numpy()
        return out * self.act_norm.std + self.act_norm.mean

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden": list(self.hidden),
                "obs_mean": self.obs_norm.mean,
                "obs_std": self.obs_norm.std,
                "act_mean": self.act_norm.mean,
                "act_std": self.act_norm.std,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "BCPolicy":
        device = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hidden = tuple(ckpt["hidden"])
        model = MLP(hidden)
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model,
            Normalizer(ckpt["obs_mean"], ckpt["obs_std"]),
            Normalizer(ckpt["act_mean"], ckpt["act_std"]),
            hidden,
            device,
        )


def train_bc(
    obs: np.ndarray,
    act: np.ndarray,
    *,
    hidden: tuple[int, ...] = (256, 256),
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: torch.device | None = None,
    seed: int = 0,
    log_every: int = 20,
) -> BCPolicy:
    """Fit an :class:`MLP` by MSE regression of ``act`` on ``obs``.

    Splits a validation slice off the front-shuffled data so the printed val loss
    is an honest held-out number rather than memorized training frames.
    """
    device = device or resolve_device()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs_norm = Normalizer.fit(obs)
    act_norm = Normalizer.fit(act)
    x = (obs - obs_norm.mean) / obs_norm.std
    y = (act - act_norm.mean) / act_norm.std

    perm = rng.permutation(len(x))
    n_val = max(1, int(len(x) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    xt = torch.from_numpy(x[train_idx]).to(device)
    yt = torch.from_numpy(y[train_idx]).to(device)
    xv = torch.from_numpy(x[val_idx]).to(device)
    yv = torch.from_numpy(y[val_idx]).to(device)

    model = MLP(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(
        f"Training BC MLP {hidden} on {len(train_idx)} frames "
        f"({len(val_idx)} val) for {epochs} epochs on {device}"
    )
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                tr = loss_fn(model(xt), yt).item()
                va = loss_fn(model(xv), yv).item()
            print(f"  epoch {epoch:4d}  train {tr:.5f}  val {va:.5f}")

    return BCPolicy(model, obs_norm, act_norm, hidden, device)
