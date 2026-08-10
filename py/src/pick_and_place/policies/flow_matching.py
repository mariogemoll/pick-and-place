# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Small conditional flow-matching model and training loop."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch


class FlowModel(torch.nn.Module):
    """Plain MLP velocity field conditioned on an observation vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        time_embedding_dim: int = 1,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.time_embedding_dim = time_embedding_dim
        layers: list[torch.nn.Module] = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def embed_time(time: torch.Tensor, dimension: int) -> torch.Tensor:
    """Map scalar flow times to fixed sinusoidal features."""
    if time.ndim != 2 or time.shape[1] != 1 or dimension < 1:
        raise ValueError("time must have shape (batch, 1) and dimension must be positive")
    if dimension == 1:
        return time
    frequencies = torch.exp(
        torch.linspace(0, -math.log(10_000), (dimension + 1) // 2, device=time.device)
    )
    angles = time * frequencies
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)[:, :dimension]


def load_examples(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a numeric conditional-flow archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {"observations", "data"}:
            raise ValueError("dataset must contain exactly observations and data")
        observations = torch.tensor(archive["observations"], dtype=torch.float32)
        data = torch.tensor(archive["data"], dtype=torch.float32)
    if observations.ndim != 2 or data.ndim != 2:
        raise ValueError("observations and data must be rank-2")
    if len(observations) != len(data) or not len(data):
        raise ValueError("observations and data must have the same nonzero length")
    return observations, data


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


def train_model(
    dataset_path: str | Path,
    validation_path: str | Path | None = None,
    *,
    num_epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    min_learning_rate: float | None = None,
    warmup_steps: int = 0,
    hidden_dim: int = 256,
    hidden_layers: int = 2,
    time_embedding_dim: int = 32,
    device: str | torch.device = "cpu",
    seed: int = 0,
    observation_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    validation_interval: int = 1,
    metrics_callback: Callable[[int, dict[str, float]], None] | None = None,
    checkpoint_callback: Callable[[int, FlowModel, list[float], list[float]], None] | None = None,
) -> tuple[FlowModel, list[float], list[float]]:
    """Fit a conditional velocity field to prepared numeric examples."""
    torch.manual_seed(seed)
    observations, data = load_examples(dataset_path)
    if validation_path is not None:
        validation_observations, validation_data = load_examples(validation_path)
        if validation_observations.shape[1] != observations.shape[1]:
            raise ValueError("training and validation observation dimensions must match")
        if validation_data.shape[1] != data.shape[1]:
            raise ValueError("training and validation data dimensions must match")
    if validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    device = torch.device(device)
    observations, data = observations.to(device), data.to(device)
    model = FlowModel(
        input_dim=data.shape[1] + time_embedding_dim + observations.shape[1],
        output_dim=data.shape[1],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        time_embedding_dim=time_embedding_dim,
    ).to(device)
    minimum = learning_rate if min_learning_rate is None else min_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_losses: list[float] = []
    validation_losses: list[float] = []
    if validation_path is not None:
        validation_observations, validation_data = (
            validation_observations.to(device),
            validation_data.to(device),
        )
        generator = torch.Generator().manual_seed(seed)
        validation_time = torch.rand(len(validation_data), 1, generator=generator).to(device)
        validation_noise = torch.randn(validation_data.shape, generator=generator).to(device)

    for epoch in range(num_epochs):
        rate = learning_rate_at_step(
            epoch, num_steps=num_epochs, peak=learning_rate, minimum=minimum, warmup_steps=warmup_steps
        )
        optimizer.param_groups[0]["lr"] = rate
        model.train()
        total_loss = 0.0
        for indices in torch.randperm(len(data), device=device).split(batch_size):
            endpoints = data[indices]
            conditions = observations[indices]
            if observation_augmentation is not None:
                conditions = observation_augmentation(conditions)
            time = torch.rand(len(endpoints), 1, device=device)
            noise = torch.randn_like(endpoints)
            values = time * endpoints + (1 - time) * noise
            prediction = model(torch.cat((values, embed_time(time, time_embedding_dim), conditions), dim=1))
            loss = torch.mean((prediction - (endpoints - noise)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        train_loss = total_loss / len(data)
        training_losses.append(train_loss)
        metrics = {"train/loss": train_loss, "learning_rate": rate}
        validate = validation_path is not None and (
            (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1
        )
        if validate:
            model.eval()
            with torch.no_grad():
                values = validation_time * validation_data + (1 - validation_time) * validation_noise
                prediction = model(
                    torch.cat(
                        (values, embed_time(validation_time, time_embedding_dim), validation_observations), dim=1
                    )
                )
                validation_loss = torch.mean((prediction - (validation_data - validation_noise)) ** 2).item()
            validation_losses.append(validation_loss)
            metrics["validation/loss"] = validation_loss
        if metrics_callback is not None:
            metrics_callback(epoch + 1, metrics)
        if checkpoint_callback is not None:
            checkpoint_callback(epoch + 1, model, training_losses, validation_losses)
    return model, training_losses, validation_losses


def generate(model: FlowModel, observations: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
    """Sample by Euler integration from Gaussian noise."""
    observation_dim = model.input_dim - model.output_dim - model.time_embedding_dim
    if observations.ndim != 2 or observations.shape[1] != observation_dim:
        raise ValueError(f"observations must have shape (batch, {observation_dim})")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    parameter = next(model.parameters())
    observations = observations.to(device=parameter.device, dtype=parameter.dtype)
    values = torch.randn(len(observations), model.output_dim, device=parameter.device)
    time = torch.zeros(len(observations), 1, device=parameter.device)
    with torch.no_grad():
        for _ in range(num_steps):
            velocity = model(torch.cat((values, embed_time(time, model.time_embedding_dim), observations), dim=1))
            values = values + velocity / num_steps
            time = time + 1 / num_steps
    return values


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> FlowModel:
    """Load a flow-matching checkpoint."""
    contents = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    model = FlowModel(**contents["model_config"]).to(device)
    model.load_state_dict(contents["model"])
    model.eval()
    return model
