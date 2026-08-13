# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Conditional flow-matching models, training, and sampling."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D


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


VelocityModel = FlowModel | FlowConditionalUnet1D


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
    num_updates: int = 20_000,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    min_learning_rate: float | None = None,
    warmup_steps: int = 0,
    hidden_dim: int = 256,
    hidden_layers: int = 2,
    time_embedding_dim: int = 32,
    architecture: str = "unet1d",
    prediction_steps: int = 16,
    unet_down_dims: tuple[int, ...] = (64, 128, 256),
    unet_kernel_size: int = 5,
    unet_groups: int = 8,
    device: str | torch.device = "cpu",
    seed: int = 0,
    observation_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    validation_interval: int = 1,
    metrics_callback: Callable[[int, dict[str, float]], None] | None = None,
    checkpoint_callback: Callable[[int, VelocityModel, list[float], list[float]], None]
    | None = None,
) -> tuple[VelocityModel, list[float], list[float]]:
    """Fit a conditional velocity field for an explicit number of optimizer updates."""
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
    if num_updates < 1 or batch_size < 1:
        raise ValueError("num_updates and batch_size must be positive")
    device = torch.device(device)
    observations, data = observations.to(device), data.to(device)
    if architecture == "mlp":
        model: VelocityModel = FlowModel(
            input_dim=data.shape[1] + time_embedding_dim + observations.shape[1],
            output_dim=data.shape[1],
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            time_embedding_dim=time_embedding_dim,
        )
    elif architecture == "unet1d":
        if prediction_steps < 1 or data.shape[1] % prediction_steps:
            raise ValueError("data dimension must be divisible by prediction_steps")
        model = FlowConditionalUnet1D(
            action_dim=data.shape[1] // prediction_steps,
            observation_dim=observations.shape[1],
            prediction_steps=prediction_steps,
            time_embedding_dim=time_embedding_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            groups=unet_groups,
        )
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    model = model.to(device)
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

    permutation = torch.randperm(len(data), device=device)
    batch_start = 0
    for update in range(num_updates):
        rate = learning_rate_at_step(
            update,
            num_steps=num_updates,
            peak=learning_rate,
            minimum=minimum,
            warmup_steps=warmup_steps,
        )
        optimizer.param_groups[0]["lr"] = rate
        model.train()
        if batch_start >= len(data):
            permutation = torch.randperm(len(data), device=device)
            batch_start = 0
        indices = permutation[batch_start : batch_start + batch_size]
        batch_start += len(indices)
        endpoints = data[indices]
        conditions = observations[indices]
        if observation_augmentation is not None:
            conditions = observation_augmentation(conditions)
        time = torch.rand(len(endpoints), 1, device=device)
        noise = torch.randn_like(endpoints)
        values = time * endpoints + (1 - time) * noise
        prediction = predict_velocity(model, values, time, conditions)
        loss = torch.mean((prediction - (endpoints - noise)) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        training_losses.append(train_loss)
        metrics = {"train/loss": train_loss, "learning_rate": rate}
        validate = validation_path is not None and (
            (update + 1) % validation_interval == 0 or update == num_updates - 1
        )
        if validate:
            model.eval()
            with torch.no_grad():
                squared_error = 0.0
                for validation_indices in torch.arange(len(validation_data)).split(batch_size):
                    endpoints = validation_data[validation_indices]
                    time = validation_time[validation_indices]
                    noise = validation_noise[validation_indices]
                    values = time * endpoints + (1 - time) * noise
                    prediction = predict_velocity(
                        model,
                        values,
                        time,
                        validation_observations[validation_indices],
                    )
                    squared_error += torch.sum((prediction - (endpoints - noise)) ** 2).item()
                validation_loss = squared_error / validation_data.numel()
            validation_losses.append(validation_loss)
            metrics["validation/loss"] = validation_loss
        if metrics_callback is not None:
            metrics_callback(update + 1, metrics)
        if checkpoint_callback is not None:
            checkpoint_callback(update + 1, model, training_losses, validation_losses)
    return model, training_losses, validation_losses


def predict_velocity(
    model: VelocityModel,
    values: torch.Tensor,
    time: torch.Tensor,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Apply either velocity architecture through its common flattened boundary."""
    if isinstance(model, FlowConditionalUnet1D):
        sequence = values.reshape(len(values), model.prediction_steps, model.action_dim)
        return model(sequence, time, observations).reshape_as(values)
    return model(
        torch.cat((values, embed_time(time, model.time_embedding_dim), observations), dim=1)
    )


def flow_sde_transition(
    values: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    *,
    step_size: float,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of one Euler-Maruyama step of the flow SDE.

    Sampling a flow policy is ordinarily deterministic, which leaves a policy
    gradient nothing to differentiate: every action is a fixed function of the
    drawn latent. Replacing the probability-flow ODE with an SDE that has the
    same marginals restores a density without changing what the model generates
    in distribution.

    For the Gaussian CondOT path this model was trained on, ``x_t = t z +
    (1 - t) e``, the score follows from the learned velocity::

        E[z | x_t] = x_t + (1 - t) v(x_t, t)
        grad log p_t(x_t) = (t v(x_t, t) - x_t) / (1 - t)

    and for any noise level ``sigma(t)`` the SDE
    ``dx = [v + sigma^2/2 * grad log p_t] dt + sigma dW`` transports the same
    marginals as ``dx = v dt``.

    The noise level here is ``sigma(t) = noise_scale * sqrt(1 - t)``, which
    cancels the score's ``1 / (1 - t)`` exactly -- so the drift correction stays
    bounded at every flow time, including the last step, rather than being
    clamped away. It also makes exploration large early and small late, leaving
    the step that emits the action the quietest one. A constant floor across
    every step is what made DPPO's default unusable on this task's absolute
    joint commands: it dumps the full noise level onto the emitted action.
    """
    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be in (0, 1]")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if values.shape != velocity.shape:
        raise ValueError("values and velocity must have the same shape")
    remaining = torch.clamp(1.0 - time, min=0.0)
    # sigma^2 / 2 * score, with the (1 - t) factors cancelled analytically.
    drift_correction = 0.5 * noise_scale**2 * (time * velocity - values)
    mean = values + (velocity + drift_correction) * step_size
    standard_deviation = noise_scale * torch.sqrt(remaining * step_size)
    return mean, standard_deviation.expand_as(mean)


def generate(
    model: VelocityModel, observations: torch.Tensor, num_steps: int = 100
) -> torch.Tensor:
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
            velocity = predict_velocity(model, values, time, observations)
            values = values + velocity / num_steps
            time = time + 1 / num_steps
    return values


def model_checkpoint_config(model: VelocityModel) -> tuple[str, dict[str, object]]:
    """Return the architecture tag and constructor arguments stored in a checkpoint."""
    if isinstance(model, FlowConditionalUnet1D):
        return "unet1d", {
            "action_dim": model.action_dim,
            "observation_dim": model.observation_dim,
            "prediction_steps": model.prediction_steps,
            "time_embedding_dim": model.time_embedding_dim,
            "down_dims": model.down_dims,
            "kernel_size": model.kernel_size,
            "groups": model.groups,
        }
    return "mlp", {
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "hidden_dim": model.hidden_dim,
        "hidden_layers": model.hidden_layers,
        "time_embedding_dim": model.time_embedding_dim,
    }


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> VelocityModel:
    """Load a flow-matching checkpoint."""
    contents = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    model_type = contents.get("model_type", "mlp")
    if model_type == "mlp":
        model: VelocityModel = FlowModel(**contents["model_config"])
    elif model_type == "unet1d":
        model = FlowConditionalUnet1D(**contents["model_config"])
    else:
        raise ValueError(f"unknown checkpoint model type: {model_type}")
    model = model.to(device)
    model.load_state_dict(contents["model"])
    model.eval()
    return model
