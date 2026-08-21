# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What the flow trainer configures, as data rather than as parser calls.

A run is written out and read back: ``config.json`` recorded the flags for years
without anything being able to replay them, and ``--config`` closes that.

There used to be a second trainer here -- the state-conditioned policy -- and a
shared base for the fourteen flags the two declared in common. It was deleted
with the policy, and the base went with it: one subclass is not a hierarchy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

T = TypeVar("T")


@dataclass(kw_only=True)
class ImageTrainingRun:
    """The image-conditioned flow policy: two camera streams in, actions out."""

    export: Path
    """Diffusion Policy image export the model trains on"""
    output: Path
    """directory the checkpoints and config.json are written to"""
    updates: int = 30_000
    """optimizer steps to run"""
    batch_size: int = 64
    learning_rate: float = 1e-4
    min_learning_rate: float | None = 1e-6
    """floor for the cosine schedule; unset holds the peak rate"""
    warmup_steps: int = 500
    prediction_steps: int = 16
    """action-chunk horizon; must match the export's own"""
    seed: int = 0
    validation_interval: int = 2_000
    checkpoint_interval: int | None = 5_000
    """steps between intermediate checkpoints; unset writes only the final one"""
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
    wandb_project: str | None = None
    """logging is opt-in, so a run without it stays offline rather than half-configured"""
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


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
