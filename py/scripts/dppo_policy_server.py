#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Serve a pretrained DPPO diffusion policy over stdin/stdout.

This script runs inside the DPPO virtual environment (the one holding the
``third_party/dppo`` install with its pinned Torch stack) and speaks a small
binary protocol on its standard streams, so the main project environment never
imports DPPO's dependencies. Every message is a 4-byte big-endian length prefix
followed by an ``.npz`` payload.

On startup the server writes one handshake message describing the loaded model
(action horizon, dimensions, image size, checkpoint epoch). Afterwards each
request message with ``state`` (raw hardware-frame joint values), ``overhead``
and ``wrist`` (HxWx3 uint8 RGB) receives one reply with ``actions``: the full
denoised action horizon in raw hardware units. End-of-file on stdin shuts the
server down.

State and action normalization uses the per-dimension min-max bounds saved by
the dataset export (``normalization.npz``), applying exactly the exporter's
``[-1, 1]`` formula. Images are sent through unscaled; the ViT encoder divides
by 255 internally, matching training. The EMA weights are loaded and image-shift
augmentation is disabled, following DPPO's own evaluation configurations.

Only the standard library and NumPy are imported at module scope so the
protocol and normalization helpers stay importable (and testable) from any
environment; Torch, Hydra, and the DPPO model modules load lazily in
``load_model``.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


def write_message(stream: BinaryIO, arrays: dict[str, np.ndarray]) -> None:
    """Write one length-prefixed npz message."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def read_message(stream: BinaryIO) -> dict[str, np.ndarray] | None:
    """Read one length-prefixed npz message, or None on a clean end-of-file."""
    header = stream.read(4)
    if len(header) == 0:
        return None
    if len(header) != 4:
        raise EOFError("truncated message header")
    (length,) = struct.unpack(">I", header)
    payload = stream.read(length)
    if len(payload) != length:
        raise EOFError(f"truncated message payload: expected {length} bytes, got {len(payload)}")
    with np.load(io.BytesIO(payload)) as data:
        return {key: data[key] for key in data.files}


def normalize_state(state: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Map raw values into [-1, 1] exactly as the dataset export did."""
    return 2.0 * (state - minimum) / (maximum - minimum + 1e-6) - 1.0


def unnormalize_actions(
    actions: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Invert :func:`normalize_state` for predicted actions."""
    return (actions + 1.0) / 2.0 * (maximum - minimum + 1e-6) + minimum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="DPPO state_*.pt file")
    parser.add_argument(
        "--config", type=Path, required=True, help="the training configuration YAML"
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        required=True,
        help="normalization.npz written by the dataset export",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--seed", type=int, default=0, help="Torch sampling seed")
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=None,
        help=(
            "sample with DDIM using this many steps instead of the trained DDPM "
            "schedule; much faster, at a small fidelity cost"
        ),
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    ddim_steps: int | None = None,
) -> tuple[Any, Any, int]:
    """Build the configured diffusion model and load the checkpoint's EMA weights.

    Returns the model, the resolved configuration, and the checkpoint epoch.
    """
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from model.diffusion.diffusion import DiffusionModel

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load(config_path)
    # DPPO's own evaluation configurations disable the random-shift image
    # augmentation that is active during training.
    cfg.model.network.augment = False
    network = instantiate(cfg.model.network)
    model_kwargs = {
        key: value
        for key, value in OmegaConf.to_container(cfg.model, resolve=True).items()
        if key not in ("_target_", "network", "device")
    }
    if ddim_steps is not None:
        if not 1 <= ddim_steps <= int(cfg.denoising_steps):
            raise ValueError(
                f"--ddim-steps must be in [1, {int(cfg.denoising_steps)}], got {ddim_steps}"
            )
        model_kwargs["use_ddim"] = True
        model_kwargs["ddim_steps"] = ddim_steps

    # The upstream sampling loop passes ``deterministic`` to ``p_mean_var``, but
    # the base pretrained-model class does not accept it (only the fine-tuning
    # subclasses do). Accept and ignore it: for DDPM sampling the flag has no
    # effect on the mean/variance computation.
    class EvalDiffusionModel(DiffusionModel):
        def p_mean_var(self, x, t, cond, index=None, network_override=None, deterministic=False):
            del deterministic
            return super().p_mean_var(x, t, cond, index=index, network_override=network_override)

    model = EvalDiffusionModel(network=network, device=device, **model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    return model, cfg, int(checkpoint.get("epoch", -1))


def main() -> None:
    args = _parse_args()

    import torch

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    model, cfg, epoch = load_model(args.config, args.checkpoint, device, args.ddim_steps)
    sampler = (
        f"ddim-{args.ddim_steps}" if args.ddim_steps is not None
        else f"ddpm-{int(cfg.denoising_steps)}"
    )

    bounds = np.load(args.normalization)
    obs_min = bounds["obs_min"].astype(np.float32)
    obs_max = bounds["obs_max"].astype(np.float32)
    action_min = bounds["action_min"].astype(np.float32)
    action_max = bounds["action_max"].astype(np.float32)

    obs_dim = int(cfg.obs_dim)
    action_dim = int(cfg.action_dim)
    rgb_shape = tuple(cfg.shape_meta.obs.rgb.shape)
    if rgb_shape[0] % 3 != 0:
        raise ValueError(f"rgb channel count must be a multiple of 3, got {rgb_shape[0]}")
    num_cameras = rgb_shape[0] // 3
    if num_cameras != 2:
        raise ValueError(f"expected a two-camera model, got {num_cameras} cameras")
    image_height, image_width = int(rgb_shape[1]), int(rgb_shape[2])
    if obs_min.shape != (obs_dim,) or action_min.shape != (action_dim,):
        raise ValueError(
            "normalization bounds do not match the model: "
            f"obs {obs_min.shape} vs ({obs_dim},), action {action_min.shape} vs ({action_dim},)"
        )

    if int(cfg.cond_steps) != 1 or int(cfg.img_cond_steps) != 1:
        raise ValueError("this server only supports single-step observation conditioning")

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    write_message(
        stdout,
        {
            "horizon_steps": np.asarray(int(cfg.horizon_steps)),
            "obs_dim": np.asarray(obs_dim),
            "action_dim": np.asarray(action_dim),
            "image_height": np.asarray(image_height),
            "image_width": np.asarray(image_width),
            "denoising_steps": np.asarray(int(cfg.denoising_steps)),
            "sampler": np.asarray(sampler),
            "epoch": np.asarray(epoch),
            "device": np.asarray(device),
            "seed": np.asarray(args.seed),
            "torch_version": np.asarray(torch.__version__),
        },
    )
    print(
        f"dppo policy server: epoch {epoch} checkpoint on {device}, "
        f"horizon {int(cfg.horizon_steps)}, {image_width}x{image_height} images, "
        f"{sampler} sampling",
        file=sys.stderr,
        flush=True,
    )

    expected_image_shape = (image_height, image_width, 3)
    while True:
        request = read_message(stdin)
        if request is None:
            return
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (obs_dim,):
            raise ValueError(f"state must have shape ({obs_dim},), got {state.shape}")
        images = []
        for key in ("overhead", "wrist"):
            image = np.asarray(request[key])
            if image.shape != expected_image_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"{key} image must be uint8 with shape {expected_image_shape}, "
                    f"got {image.dtype} {image.shape}"
                )
            images.append(image.transpose(2, 0, 1))
        rgb = np.concatenate(images, axis=0)

        state_normalized = normalize_state(state, obs_min, obs_max)
        cond = {
            "state": torch.from_numpy(state_normalized).float().view(1, 1, obs_dim).to(device),
            "rgb": torch.from_numpy(rgb).float().view(1, 1, *rgb.shape).to(device),
        }
        sample = model(cond=cond, deterministic=True)
        actions_normalized = sample.trajectories.cpu().numpy().reshape(-1, action_dim)
        actions = unnormalize_actions(actions_normalized, action_min, action_max)
        write_message(stdout, {"actions": actions.astype(np.float32)})


if __name__ == "__main__":
    main()
