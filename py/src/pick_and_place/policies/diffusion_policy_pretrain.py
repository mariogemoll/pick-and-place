# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A throughput-oriented replacement for DPPO's diffusion pre-training agent.

Only DPPO's *pre-training* half is used here: it supplies the diffusion model,
loss, LR schedule and EMA that produce the Diffusion Policy this project runs.
Its reinforcement-learning half is a separate, unrelated path (``dppo_rl``).

The vendored :class:`agent.pretrain.train_diffusion_agent.TrainDiffusionAgent`
is correct but leaves an RTX 5090 mostly idle: it builds every training sample
in Python (:meth:`StitchedSequenceDataset.__getitem__` runs a ``torch.stack``
per sample), ships batches through a ``DataLoader``, runs the two camera
encoders as two separate forward passes, trains in fp32, and calls ``.item()``
on every step, which synchronises the host with the device 1,266 times an
epoch. The measured result was ~24 s/epoch on a 5090 for an 8.15M-parameter
model — the run is dominated by launch and Python overhead, not by arithmetic.

This module keeps the model, the loss, the LR schedule and the EMA semantics of
the upstream agent and replaces everything around them:

* **The whole dataset lives in VRAM** and batches are gathered on-device with
  two index tensors precomputed from ``StitchedSequenceDataset.indices``. There
  is no ``DataLoader``, no worker process and no per-sample Python.
  :func:`build_gather_indices` is derived from the upstream ``__getitem__``
  arithmetic and ``scripts/check_diffusion_policy_pretrain_fast.py`` asserts the
  batches match element for element.
* **The two camera encoders run as one batched forward.** The backbone is
  shared and every op in it is per-sample, so stacking the two images into one
  ``(2B, C, H, W)`` call is arithmetically the same work in half the launches.
* **bf16 autocast, channels-last, fused AdamW, `_foreach` EMA and
  `torch.compile`**, each independently switchable so a regression can be
  attributed.
* **No per-step host sync.** The epoch loss is accumulated in a device tensor
  and read once per epoch.

Batch size is left as the primary knob: it is the one setting here that changes
the optimisation trajectory rather than just how fast the same trajectory is
walked, so it is configured explicitly and never scaled implicitly.

Checkpoints are written in the upstream format — ``epoch``/``model``/``ema``
plus the optimizer, scheduler and RNG state needed to resume — with any
``torch.compile`` wrapper prefixes stripped, so
``scripts/diffusion_policy_server.py`` and the DPPO fine-tuning configs consume
them unchanged.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import time
from copy import deepcopy
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def _import_dppo() -> tuple[Any, Any, Any]:
    """Import the vendored DPPO pieces this agent builds on.

    Imported lazily so the module can be introspected (and its pure helpers
    tested) without ``third_party/dppo`` on the path.
    """
    from agent.pretrain.train_agent import EMA, PreTrainAgent
    from util.scheduler import CosineAnnealingWarmupRestarts

    return PreTrainAgent, EMA, CosineAnnealingWarmupRestarts


def build_gather_indices(
    indices: list[tuple[int, int]],
    cond_steps: int,
    img_cond_steps: int,
    horizon_steps: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Precompute the absolute row indices each training sample reads.

    Upstream ``StitchedSequenceDataset.__getitem__`` slices a window starting at
    ``start - num_before_start`` and then picks ``max(num_before_start - t, 0)``
    within it for ``t`` counting down to zero, so the most recent observation
    lands last. Since ``max(n - t, 0) == n - min(n, t)``, the absolute row is
    ``start - min(num_before_start, t)``: at an episode boundary the history is
    clamped to the first frame, exactly as upstream repeats it.

    Args:
        indices: ``(start, num_before_start)`` pairs, upstream's sample list.
        cond_steps: Number of proprioceptive observations per sample.
        img_cond_steps: Number of image observations per sample.
        horizon_steps: Length of the predicted action chunk.
        device: Device to place the index tensors on.

    Returns:
        ``state``/``img``/``action`` index tensors of shape ``(N, steps)``.
    """
    starts = torch.tensor([start for start, _ in indices], dtype=torch.long)
    before = torch.tensor([before for _, before in indices], dtype=torch.long)

    def history(steps: int) -> torch.Tensor:
        # t counts down so the most recent step is last, matching upstream.
        offsets = torch.arange(steps - 1, -1, -1, dtype=torch.long)
        return starts[:, None] - torch.minimum(before[:, None], offsets[None, :])

    return {
        "state": history(cond_steps).to(device),
        "img": history(img_cond_steps).to(device),
        "action": (starts[:, None] + torch.arange(horizon_steps, dtype=torch.long)).to(device),
    }


def install_compile_safe_attention() -> int:
    """Drop the deprecated ``sdp_kernel`` context manager from the ViT.

    ``MultiHeadAttention.forward`` wraps its SDPA call in
    ``torch.backends.cuda.sdp_kernel(enable_math=False)``, which is deprecated
    (it warns on every call under Torch 2.10+) and forces a graph break under
    ``torch.compile``, splitting the encoder into separate compiled regions.
    The guard only exists to assert that a fused backend was picked; for
    ``(121 tokens, head_dim 32)`` inputs Torch selects a fused kernel on its
    own, so calling SDPA directly is the same computation without the fence.

    Returns the number of patched modules (0 if the layout changed upstream).
    """
    import einops
    from model.common.vit import MultiHeadAttention

    def forward(self: Any, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(
            qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head
        ).unbind(1)
        attn_v = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, attn_mask=attn_mask
        )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)

    MultiHeadAttention.forward = forward
    return 1


class GpuBatchSampler:
    """Serves training batches by gathering directly from VRAM-resident data.

    Replaces the ``DataLoader``: one shuffled permutation per epoch and three
    index-select kernels per batch, with no host round-trip.
    """

    def __init__(
        self,
        dataset: Any,
        device: torch.device | str,
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.states = dataset.states.to(self.device)
        self.actions = dataset.actions.to(self.device)
        self.images = dataset.images.to(self.device)
        self.gather = build_gather_indices(
            dataset.indices,
            cond_steps=dataset.cond_steps,
            img_cond_steps=dataset.img_cond_steps,
            horizon_steps=dataset.horizon_steps,
            device=self.device,
        )
        self.num_samples = len(dataset.indices)

    @property
    def num_batches(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return -(-self.num_samples // self.batch_size)

    def epoch_permutation(self, generator: torch.Generator) -> torch.Tensor:
        return torch.randperm(self.num_samples, device=self.device, generator=generator)

    def batch(self, sample_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather one batch: chunked actions plus the observation condition."""
        actions = self.actions[self.gather["action"][sample_ids]]
        conditions = {
            "state": self.states[self.gather["state"][sample_ids]],
            "rgb": self.images[self.gather["img"][sample_ids]],
        }
        return actions, conditions


def batched_vision_forward(network: Any, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Encode both cameras in one backbone call instead of two.

    Upstream runs ``self.backbone`` once per camera. The backbone is shared and
    contains no cross-sample operations (LayerNorm/GroupNorm normalise within a
    sample), so folding the camera axis into the batch axis is the same
    arithmetic with half the kernel launches. The per-camera ``compress1`` and
    ``compress2`` projections stay separate, as upstream.

    Args:
        network: The ``VisionUnet1D`` providing backbone/compress modules.
        rgb: ``(B, num_img, C, H, W)`` images, already float.
        state: ``(B, cond_dim)`` flattened proprioception.

    Returns:
        The concatenated per-camera visual features, ``(B, spatial_emb * 2)``.
    """
    batch, num_img = rgb.shape[:2]
    flat = rgb.flatten(0, 1)
    if getattr(network, "prefer_channels_last", False):
        flat = flat.contiguous(memory_format=torch.channels_last)
    if network.augment:
        # RandomShiftsAug draws an independent shift per row, so applying it to
        # the folded tensor still gives each camera view its own shift.
        flat = network.aug(flat)
    feats = network.backbone(flat)
    feats = feats.view(batch, num_img, *feats.shape[1:])
    compressed = [
        network.compress1.forward(feats[:, 0], state),
        network.compress2.forward(feats[:, 1], state),
    ]
    return torch.cat(compressed, dim=-1)


def install_batched_vision_encoder(network: Any) -> bool:
    """Patch a ``VisionUnet1D`` instance to encode its cameras in one pass.

    Returns ``False`` (leaving the instance untouched) when the network is not
    the two-camera spatial-embedding configuration this optimisation covers.
    """
    from model.diffusion.unet import VisionUnet1D

    if not isinstance(network, VisionUnet1D):
        return False
    if network.num_img != 2 or not hasattr(network, "compress1"):
        return False

    original_forward = network.forward.__func__

    def forward(self: Any, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs: Any):
        batch = len(x)
        height, width = cond["rgb"].shape[-2:]
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        steps = rgb.shape[1]
        # History is concatenated along channels, per camera.
        rgb = rgb.reshape(batch, steps, self.num_img, 3, height, width)
        rgb = rgb.permute(0, 2, 1, 3, 4, 5).reshape(
            batch, self.num_img, steps * 3, height, width
        )
        state = cond["state"].view(batch, -1)
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)
        feat = batched_vision_forward(self, rgb.float(), state)
        return self.forward_from_visual_feature(x, time, feat, state)

    def forward_from_visual_feature(
        self: Any, x: torch.Tensor, time: torch.Tensor, feat: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        import einops

        x = einops.rearrange(x, "b h t -> b t h")
        cond_encoded = torch.cat([feat, state], dim=-1)
        time = time.expand(x.shape[0])
        global_feature = self.time_mlp(time)
        global_feature = torch.cat([global_feature, cond_encoded], dim=-1)

        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        return einops.rearrange(x, "b t h -> b h t")

    network.forward_from_visual_feature = forward_from_visual_feature.__get__(network)
    network.forward = forward.__get__(network)
    network.stock_forward = original_forward.__get__(network)
    return True


def _build_agent_class() -> type:
    PreTrainAgent, EMA, CosineAnnealingWarmupRestarts = _import_dppo()

    class _FastTrainDiffusionAgent(PreTrainAgent):
        """Drop-in for ``TrainDiffusionAgent`` that keeps the GPU busy."""

        def __init__(self, cfg: Any) -> None:
            # Deliberately not calling PreTrainAgent.__init__: it builds a
            # DataLoader over a 5.3 GB dataset that this agent never uses. The
            # setup below mirrors it step for step otherwise.
            self.cfg = cfg
            self.seed = cfg.get("seed", 42)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

            speed = cfg.train.get("speed", {})
            self.use_bf16 = bool(speed.get("bf16", True))
            self.use_channels_last = bool(speed.get("channels_last", True))
            self.use_compile = bool(speed.get("compile", True))
            self.compile_mode = speed.get("compile_mode", "default")
            self.use_fused_optimizer = bool(speed.get("fused_optimizer", True))
            self.use_batched_vision = bool(speed.get("batched_vision", True))
            self.use_plain_attention = bool(speed.get("plain_attention", True))
            self.drop_last = bool(speed.get("drop_last", True))
            # Benchmark aid: stop after this many seconds of wall clock.
            self.max_run_seconds = speed.get("max_run_seconds")

            if speed.get("tf32", True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

            self.device = torch.device(cfg.device)
            self.use_wandb = cfg.wandb is not None
            if self.use_wandb:
                import wandb

                wandb.init(
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    name=cfg.wandb.run,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )

            self.model = hydra.utils.instantiate(cfg.model)
            self.ema = EMA(cfg.ema)
            self.ema_model = deepcopy(self.model)

            self.n_epochs = cfg.train.n_epochs
            self.batch_size = cfg.train.batch_size
            self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
            self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
            self.val_freq = cfg.train.get("val_freq", 100)

            self.logdir = cfg.logdir
            self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.log_freq = cfg.train.get("log_freq", 1)
            self.save_model_freq = cfg.train.save_model_freq

            dataset = hydra.utils.instantiate(cfg.train_dataset)
            self.sampler = GpuBatchSampler(
                dataset,
                device=self.device,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            # The dataset object held the only reference to its host copies.
            del dataset
            log.info(
                "Fast sampler: %d samples, %d batches/epoch at batch %d",
                self.sampler.num_samples,
                self.sampler.num_batches,
                self.batch_size,
            )

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay,
                fused=self.use_fused_optimizer and self.device.type == "cuda",
            )
            self.lr_scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.learning_rate,
                min_lr=cfg.train.lr_scheduler.min_lr,
                warmup_steps=cfg.train.lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            self.reset_parameters()
            self._prepare_model()

            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
            self.epoch = 1
            self.epoch_seconds: list[float] = []

        # -- setup helpers -------------------------------------------------

        def _prepare_model(self) -> None:
            if self.use_plain_attention:
                install_compile_safe_attention()
            if self.use_batched_vision:
                applied = install_batched_vision_encoder(self.model.network)
                log.info("Batched two-camera vision encoder: %s", "on" if applied else "n/a")
            if self.use_channels_last:
                self.model.network.backbone.to(memory_format=torch.channels_last)
                self.model.network.prefer_channels_last = True
            if self.use_compile:
                # Module.compile() keeps state_dict keys unprefixed, so
                # checkpoints stay loadable by the policy server unchanged.
                self.model.network.compile(mode=self.compile_mode, dynamic=False)
                log.info("torch.compile enabled (mode=%s)", self.compile_mode)

        def _autocast(self):
            if self.use_bf16 and self.device.type == "cuda":
                return torch.autocast("cuda", dtype=torch.bfloat16)
            return contextlib.nullcontext()

        # -- EMA -----------------------------------------------------------

        def reset_parameters(self) -> None:
            self.ema_model.load_state_dict(self.model.state_dict())

        @torch.no_grad()
        def step_ema(self) -> None:
            """``ma = beta * ma + (1 - beta) * current``, fused across params.

            Same arithmetic as upstream's Python loop, issued as two `_foreach`
            calls instead of two kernel launches per parameter tensor.
            """
            if self.epoch < self.epoch_start_ema:
                self.reset_parameters()
                return
            if not hasattr(self, "_ema_pairs"):
                self._ema_pairs = (
                    [p.data for p in self.ema_model.parameters()],
                    [p.data for p in self.model.parameters()],
                )
            ema_params, model_params = self._ema_pairs
            torch._foreach_mul_(ema_params, self.ema.beta)
            torch._foreach_add_(ema_params, model_params, alpha=1.0 - self.ema.beta)

        # -- checkpointing --------------------------------------------------

        def _clean_state_dict(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
            prefix = "_orig_mod."
            return {
                key.replace(prefix, ""): value for key, value in module.state_dict().items()
            }

        def save_model(self) -> None:
            # Everything stored here must survive `torch.load(weights_only=True)`,
            # because scripts/diffusion_policy_server.py loads checkpoints that way.
            # That rules out numpy's and Python's RNG state, whose pickles pull in
            # globals the safe unpickler rejects -- storing them turns a
            # checkpoint into one the policy server cannot open. `rng` is a plain
            # tensor, matching the existing state_1500.pt exactly; the CUDA and
            # sampler generators are tensors too. Nothing in the training loop
            # draws from numpy or `random` after setup, so dropping them costs no
            # reproducibility.
            data = {
                "epoch": self.epoch,
                "model": self._clean_state_dict(self.model),
                "ema": self._clean_state_dict(self.ema_model),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "rng_generator": self.generator.get_state(),
            }
            if torch.cuda.is_available():
                data["rng_cuda"] = torch.cuda.get_rng_state_all()
            savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
            torch.save(data, savepath)
            log.info(f"Saved model to {savepath}")

        def load(self, epoch: int) -> None:
            loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
            # Safe-unpicklable by construction; see save_model.
            data = torch.load(loadpath, weights_only=True)
            self.epoch = data["epoch"]
            self.model.load_state_dict(data["model"])
            self.ema_model.load_state_dict(data["ema"])
            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
            if "lr_scheduler" in data:
                self.lr_scheduler.load_state_dict(data["lr_scheduler"])
            if "rng" in data:
                torch.set_rng_state(data["rng"])
            if "rng_generator" in data:
                self.generator.set_state(data["rng_generator"])
            if "rng_cuda" in data and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(data["rng_cuda"])

        # -- instrumentation -------------------------------------------------

        def _write_throughput(self, last_loss: float) -> None:
            """Record per-epoch timings next to the checkpoints.

            Written every epoch so a benchmark sweep or an interrupted run
            still leaves usable numbers, and so throughput can be compared
            across runs without parsing the console log.
            """
            import json

            # Epoch 1 pays for compile and cudnn autotuning; report the steady
            # state separately so the two are never conflated.
            steady = self.epoch_seconds[1:] or self.epoch_seconds
            ordered = sorted(steady)
            median = ordered[len(ordered) // 2]
            payload = {
                "batch_size": self.batch_size,
                "batches_per_epoch": self.sampler.num_batches,
                "samples_per_epoch": self.sampler.num_batches * self.batch_size,
                "epochs_recorded": len(self.epoch_seconds),
                "first_epoch_seconds": self.epoch_seconds[0],
                "median_epoch_seconds": median,
                "median_step_ms": 1000.0 * median / self.sampler.num_batches,
                "samples_per_second": self.sampler.num_batches * self.batch_size / median,
                "projected_hours_for_n_epochs": median * self.n_epochs / 3600.0,
                "n_epochs": self.n_epochs,
                "last_train_loss": last_loss,
                "epoch_seconds": self.epoch_seconds,
                "speed": {
                    "bf16": self.use_bf16,
                    "channels_last": self.use_channels_last,
                    "compile": self.use_compile,
                    "compile_mode": self.compile_mode,
                    "fused_optimizer": self.use_fused_optimizer,
                    "batched_vision": self.use_batched_vision,
                    "plain_attention": self.use_plain_attention,
                    "drop_last": self.drop_last,
                },
            }
            path = os.path.join(self.logdir, "throughput.json")
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)

        # -- training -------------------------------------------------------

        def run(self) -> None:
            batches_per_epoch = self.sampler.num_batches
            batch_size = self.batch_size
            cnt_batch = 0
            self.epoch = 1
            run_start = time.perf_counter()

            for _ in range(self.n_epochs):
                epoch_start = time.perf_counter()
                perm = self.sampler.epoch_permutation(self.generator)
                loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)

                self.model.train()
                for step in range(batches_per_epoch):
                    sample_ids = perm[step * batch_size : (step + 1) * batch_size]
                    actions, conditions = self.sampler.batch(sample_ids)
                    with self._autocast():
                        loss_train = self.model.loss(actions, conditions)
                    loss_train.backward()
                    # Accumulated on-device: reading it here would sync the host
                    # with the GPU on every step.
                    loss_sum += loss_train.detach().float()

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if cnt_batch % self.update_ema_freq == 0:
                        self.step_ema()
                    cnt_batch += 1

                self.lr_scheduler.step()
                loss_train_value = (loss_sum / batches_per_epoch).item()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                epoch_seconds = time.perf_counter() - epoch_start
                self.epoch_seconds.append(epoch_seconds)

                if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                    self.save_model()

                if self.epoch % self.log_freq == 0:
                    log.info(
                        f"{self.epoch}: train loss {loss_train_value:8.4f} "
                        f"| t:{epoch_seconds:8.4f}"
                    )
                    if self.use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "loss - train": loss_train_value,
                                "epoch seconds": epoch_seconds,
                                "lr": self.optimizer.param_groups[0]["lr"],
                            },
                            step=self.epoch,
                            commit=True,
                        )

                self._write_throughput(loss_train_value)
                self.epoch += 1
                if (
                    self.max_run_seconds is not None
                    and time.perf_counter() - run_start > self.max_run_seconds
                ):
                    log.info("Stopping early: max_run_seconds budget reached.")
                    break

    return _FastTrainDiffusionAgent


def __getattr__(name: str) -> Any:
    """Build the agent class on first access so DPPO is imported lazily."""
    if name == "FastTrainDiffusionAgent":
        cls = _build_agent_class()
        globals()["FastTrainDiffusionAgent"] = cls
        return cls
    raise AttributeError(name)
