# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What the two flow trainers configure, as data rather than as parser calls.

The trainers share fourteen flags by name, type and meaning, and that is what
:class:`TrainingRun` collects. **They deliberately do not share the values.** A
state model trains at batch 256 and learning rate 3e-3; an image model at 64 and
1e-4, because it is a different network on different data. So each leaf inherits
the field and overrides the default, and the shared part is the declaration, not
the number.

That is the opposite of the evaluator's shared flags, where the whole point is
that two runs are only comparable if the world was described by literally the
same declaration *and* the same values. Worth keeping the distinction in view:
here sharing removes duplication, there it protects a measurement.

Configs are dataclasses so a run can be written out and read back --
``config.json`` recorded the flags for years without anything being able to
replay them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

T = TypeVar("T")


@dataclass(kw_only=True)
class TrainingRun:
    """Flags both flow trainers take. Leaves override the defaults."""

    output: Path
    """directory the checkpoints and config.json are written to"""
    updates: int = 20_000
    """optimizer steps to run"""
    batch_size: int = 256
    learning_rate: float = 3e-3
    min_learning_rate: float | None = None
    """floor for the cosine schedule; unset holds the peak rate"""
    warmup_steps: int = 0
    prediction_steps: int = 16
    """action-chunk horizon; must match the export's own"""
    seed: int = 0
    validation_interval: int = 1
    checkpoint_interval: int | None = None
    """steps between intermediate checkpoints; unset writes only the final one"""
    wandb_project: str | None = None
    """logging is opt-in, so a run without it stays offline rather than half-configured"""
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


@dataclass(kw_only=True)
class StateTrainingRun(TrainingRun):
    """The state-conditioned flow policy: cube pose and target in, actions out."""

    dataset: Path
    """flow-policy export the model trains on"""
    validation: Path | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    architecture: Literal["unet1d", "mlp"] = "unet1d"
    hidden_dim: int = 256
    hidden_layers: int = 2
    time_embedding_dim: int = 32
    unet_down_dims: tuple[int, ...] = (64, 128, 256)
    unet_kernel_size: int = 5
    unet_groups: int = 8
    cube_symmetry_augmentation: bool = False
    """reflect the cube's yaw symmetry into extra training pairs"""


@dataclass(kw_only=True)
class ImageTrainingRun(TrainingRun):
    """The image-conditioned flow policy: two camera streams in, actions out."""

    export: Path
    """Diffusion Policy image export the model trains on"""
    updates: int = 30_000
    batch_size: int = 64
    learning_rate: float = 1e-4
    min_learning_rate: float | None = 1e-6
    warmup_steps: int = 500
    validation_interval: int = 2_000
    checkpoint_interval: int | None = 5_000
    device: str = "cuda"
    """torch device string, so 'cuda:1' works on a multi-GPU box"""
    observation_steps: int = 2
    keypoints: int = 32
    pretrained_backbone: bool = False
    trunk_stages: Literal[1, 2, 3, 4] = 3
    """ResNet18 residual stages to keep; 3 stops after layer3, halving the model and
    doubling the keypoint map the spatial softmax localizes over. Pass 4 for the full
    trunk. This is the default for *new runs* only -- CameraEncoder still defaults to
    4, because checkpoints written before the flag existed carry no trunk_stages in
    their model_config and must keep loading as full trunks."""
    validation_fraction: float = 0.1
    validation_batches: int = 40
    log_interval: int = 100
    random_shift: int = 0
    """pixels of random translation augmentation per camera (0 disables)"""
    random_scale_pct: float = 0.0
    """percent of random zoom per camera, standing in for the overhead camera's
    between-session focal length (0 disables)"""
    photometric_augmentation: bool = False
    """randomize each camera's exposure, white balance, gamma, read noise and focus;
    the ranges are PhotometricRanges' defaults"""
    resume: Path | None = None
    """warm-start from a checkpoint's weights and run a fresh schedule. The optimizer
    state is deliberately not restored: this is a new cosine cycle, not a
    continuation of the old one"""
    amp: bool = True


def config_to_json(config: Any) -> dict[str, Any]:
    """A run's configuration as JSON-safe data, for `config.json` and wandb."""
    return json.loads(json.dumps(asdict(config), default=str))


def load_config(config_class: type[T], path: Path) -> T:
    """Rebuild a config from a `config.json` an earlier run wrote.

    Unknown keys are dropped rather than raising: an old run's file predates
    whatever fields have been added since, and the point of reading it back is to
    reproduce what it *did* say, with today's defaults filling the rest.
    """
    if not is_dataclass(config_class):
        raise TypeError(f"{config_class!r} is not a dataclass")
    stored = json.loads(path.read_text())
    known = {f.name: f for f in fields(config_class)}
    return config_class(**{
        name: _coerce(known[name].type, value)
        for name, value in stored.items()
        if name in known and value is not None
    })


def _coerce(annotation: Any, value: Any) -> Any:
    """Restore the few types JSON cannot carry back on its own."""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if "Path" in str(text):
        return Path(value)
    if "tuple" in str(text) and isinstance(value, list):
        return tuple(value)
    return value


def parse_training_config(
    config_class: type[T],
    argv: list[str] | None = None,
    *,
    description: str | None = None,
) -> T:
    """Parse a trainer's flags, optionally seeded from an earlier run's config.

    ``--config PATH`` reads a ``config.json`` and uses it as the defaults, so
    every flag not given on the command line comes from that run. It is handled
    here rather than declared as a field because it is not part of the
    configuration -- it says where the configuration came from.

    The precedence is the useful one for repeating an experiment with one thing
    changed: command line beats file beats the dataclass default.
    """
    import sys

    import tyro

    argv = list(sys.argv[1:] if argv is None else argv)
    default = None
    if "--config" in argv:
        index = argv.index("--config")
        if index + 1 >= len(argv):
            raise SystemExit("--config needs a path to an earlier run's config.json")
        default = load_config(config_class, Path(argv[index + 1]))
        del argv[index : index + 2]
    kwargs: dict[str, Any] = {"args": argv}
    if description is not None:
        kwargs["description"] = description
    if default is not None:
        kwargs["default"] = default
    return tyro.cli(config_class, **kwargs)
