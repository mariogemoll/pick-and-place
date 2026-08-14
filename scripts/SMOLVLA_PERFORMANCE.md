<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# What a SmolVLA training step costs

Everything measured about the speed of `vast_smolvla_train.sh`, split out of
`README.md` when it outgrew that file. `README.md` covers what the launchers do
and how to rent a host for them; this covers where the time goes.

Two rules run through all of it. **Compare arms on one host**: marketplace
machines of identical advertised specification have measured 1.68x apart, so an
absolute s/step is a fact about the host it was measured on. And **compare in
samples/s, never s/step**, whenever the batch size moves.

The short version, on an RTX 5090 at batch 64:

| | s/step | samples/s |
| --- | ---: | ---: |
| stock, eager | 0.3500 | 182.8 |
| stock, compiled | 0.2627 | 243.6 |
| cached tower output, eager | 0.0711 | 900.0 |
| **cached tower output, compiled** | **0.0266** | **2,409.6** |

Through stock `lerobot-train` on real data, eager, the same change is
**0.426 s to 0.160 s** — smaller than 4.92x because a live step carries about
0.08 s that a synthetic one does not, and which is *not* the trainer or the
dataloader. That residual is now the largest single item in a cached step.

The training loop itself is lerobot's, unmodified: `vast_smolvla_train.sh` calls
`lerobot-train`, and `train_smolvla_cached.py` still does — it swaps the dataset
and two methods on the policy and changes nothing else.

## Where a training step goes, and why quantizing the tower is not worth it

Profiled at batch 64, uncompiled, on synthetic batches:

| stage | seconds | share |
| --- | ---: | ---: |
| `embed_prefix` -- the frozen vision tower | 0.2282 | **59.2%** |
| joint VLM and expert layers | 0.0667 | 17.3% |
| backward | 0.0889 | 23.1% |
| AdamW over 100M parameters | 0.0019 | 0.5% |

The frozen tower is most of a training step, the backward is only 23% because
just 22% of the parameters train, and the optimizer is negligible -- a fused
AdamW would buy nothing.

**This replicates on a second host** (2026-08-14, a different 5090): 0.2236 s of
a 0.3500 s step, 63.9%. The shares are a property of the model, not of the
machine; the absolute seconds are not.

That invites two ideas, and **both fail, for opposite reasons**.

**Running the prefix ahead on a side stream is capped at ~1.03x.** Measured GPU
busy is **96.6%**, so there is no bubble to fill: reordering work does not
create capacity. Note `nvidia-smi`'s utilization counter said 85% and implied
15% idle -- it samples "any kernel resident" and overstates idle badly. Sum
kernel durations against an unprofiled wall clock instead, and count only
device-side entries: each `aten::` op and the kernel under it both carry device
time, so a naive sum double counts and can exceed 100%.

**Quantizing the tower buys 3.8%, and moves the embeddings 10%.** With
torchao on a 5090:

| | step | vs same-mode baseline | embedding drift |
| --- | ---: | ---: | --- |
| baseline eager | 0.3754 | -- | -- |
| baseline compiled | 0.2645 | 1.419x | -- |
| fp8, eager | 0.6064 | 0.619x | cos 0.99474, rel 0.103 |
| fp8 + compile | 0.2548 | **1.038x** | cos 0.99474, rel 0.103 |
| int8 dynamic, eager | 0.9316 | 0.404x | cos 0.99765, rel 0.069 |
| int8 weight-only, eager | 0.5258 | 0.716x | cos 0.99941, rel 0.034 |

Eager quantization is a large regression -- torchao's kernels need
`torch.compile` to fuse the quantize/dequantize, and weight-only helps
memory-bound batch-1 decoding rather than a compute-bound batch-64 ViT. Even
done right, fp8's GEMM saving is mostly cancelled by conversion overhead, and
much of the tower is attention, layernorm and elementwise work that fp8 does
not touch. Perturbing the perception front-end of a policy scoring 32-39/100 by
10% to save four minutes in five hours is not a trade worth making.

Two installation notes, since this cost longer than the measurement: the venv
is uv-managed and has no `pip`, so use `uv pip install --python <venv>/bin/python`;
and installing torchao breaks `diffusers` 0.35.2, which lerobot imports through
its groot policy, with `name 'logger' is not defined` -- diffusers only walks
its torchao branch when torchao is present. `diffusers` 0.39.0 fixes it.


## Two speed ideas that measured as nothing

Recorded so they are not retried: **casting the frozen tower (or the whole VLM)
to bf16** rather than letting autocast convert it each step is -2.3%
uncompiled and +1.6% compiled, i.e. a wash. Autocast evidently caches its
weight casts within a step, so the per-step re-cast that seemed wasteful is not
happening. Embeddings barely move (cos 0.99993).

And a **fused AdamW** cannot help: the optimizer is 0.5% of a step.


## The third idea works: cache the frozen tower's output

The two ideas above fail because they try to *reorder* or *cheapen* the tower.
Neither can win much: GPU busy is 96.6%, so there is no bubble to fill, and the
tower is mostly attention and elementwise work that quantization does not touch.

The thing neither tries is **not doing the work at all**. `freeze_vision_encoder`
and `train_expert_only` are both true, so nothing in `embed_image` — SigLIP over
1024 patches per camera, then the pixel-shuffle connector — ever moves. Its
64x960 output is a pure function of the pixels, and an 11-epoch run computes the
same block eleven times.

Compute it once:

```sh
python py/scripts/precompute_smolvla_prefix.py \
  --dataset "$artifact_root" --checkpoint "$checkpoint_dir" --output /workspace/prefix-cache
python py/scripts/train_smolvla_cached.py --prefix-cache /workspace/prefix-cache <lerobot-train args>
```

or `PREFIX_CACHE=1` on `vast_smolvla_train.sh`, which does both.

**It is exact.** `check_smolvla_prefix_cache.py` runs one batch through the stock
policy and through the cached one with the flow-matching noise and time held
fixed. Under the bf16 autocast the run uses, the two losses agree to
**0.000e+00** — the tower already emits bfloat16 there, so storing bfloat16 is
lossless. (Without that autocast the tower emits float32 and a bfloat16 cache
costs 0.2% on the loss, so the script picks its storage dtype from the precision
it is running at.)

Measured back to back on one host, RTX 5090, batch 64, synthetic batches:

| arm | s/step | samples/s | peak VRAM |
| --- | ---: | ---: | ---: |
| stock, eager | 0.3500 | 182.8 | 11,391 MiB |
| **cached, eager** | **0.0711** | **900.0** | 5,582 MiB |
| stock, compiled | 0.2627 | 243.6 | 9,116 MiB |
| **cached, compiled** | **0.0266** | **2,409.6** | 3,950 MiB |
| tower alone | 0.2243 | 570.6 images/s | 4,890 MiB |

**4.92x eager, and 9.89x compiled.** The two levers multiply rather than
overlap: `torch.compile` is worth 1.33x on the stock step here (reproducing the
1.45x that section measured on another host) and **9.9x** on the cached one,
because with the tower gone the graph is small enough for `max-autotune`'s CUDA
graphs to matter. Compiled and cached is **13.2x** the stock eager step.

The one cost is that `max-autotune` compiles for twenty to thirty minutes before
the first step on this configuration — much longer than the seven minutes that
section records — so it is only worth enabling on a run of real length. Time it
with `updt_s` past step 150, never with tqdm's rate.

The eager gain is already more than the tower's own share, and the stage split
says where the rest came from:

| stage | stock | cached |
| --- | ---: | ---: |
| tower | 0.2236 (63.9%) | 0.0000 |
| rest of forward | 0.0537 | 0.0322 |
| backward | 0.0710 | 0.0366 |
| AdamW | 0.0041 | 0.0039 |

The tower accounts for 0.2236 s of the 0.2789 s saved. The remaining 0.055 s is
the rest of the step running ~1.75x faster with the tower's activations out of
the way; peak VRAM halves alongside it, which is the likeliest cause, but that is
measured rather than explained — do not plan around it.

**The cache pays for itself after 1.2 epochs.** Building it is one pass over the
dataset, measured at **194 frames/s** over 100 episodes with 8 workers — which
extrapolates to **25 minutes** for all 291,618 frames. That is *decode* bound,
not tower bound: the tower alone sustains 285 frames/s, so more workers would
close some of the gap. It is the last time the run pays for h264 at all.

Against 182.8 samples/s stock and 900.0 cached, the break-even is about 345,000
samples — 5,400 steps at batch 64. Every run this project has done is far past
that. For a 50,000-step run at batch 64, which is 10.97 epochs:

| | training | + cache build | total |
| --- | ---: | ---: | ---: |
| stock, eager | 4.86 h | — | **4.86 h** |
| cached, eager | 0.99 h | 0.42 h | **1.41 h** |
| stock, compiled | 3.65 h | — | **3.65 h** |
| cached, compiled | 0.37 h | 0.42 h | **0.79 h** |

At $0.78/hr that is **$3.79 against $0.62**, and a working day's iteration loop
against a coffee break. The compiled rows exclude `max-autotune`'s twenty to
thirty minutes of startup, which is a fixed cost either way.

Two costs, both real:

- **72 GB of local disk**, and it must be local. 291,618 frames x 2 cameras x 64
  tokens x 960 dims x 2 bytes. Build it on the pod — do not put it in S3, where
  the egress alone would cost more than the GPU. Rent the disk at creation time;
  instance disk cannot be grown later.
- **Augmentation has to move into the cache.** lerobot's image transforms run on
  pixels a cached run never decodes, so leaving them enabled would be silently
  inert; `CachedPrefixDataset` refuses that configuration rather than let it
  happen. `--variants N` stores N independently augmented passes and each read
  draws one, at N times the disk and the build time. At the default of 1 the run
  *is* the no-augmentation arm, which `SMOLVLA.md` calls the cheapest open
  question it has.

**Do not read a frame in the parent process before forking workers.** Building
the cache first probed the tower's output shape with `dataset[0]`, and the
workers then died with `Could not push packet to decoder: Invalid data found
when processing input`. lerobot warns about this in `DatasetReader._query_videos`
— it keeps open torchcodec decoders in a module-level `VideoDecoderCache`, and a
forked worker inherits one it cannot use. The probe now runs on a blank image,
which carries the shape and decodes nothing. Any code that touches a LeRobot
dataset before handing it to a `DataLoader` is exposed to the same thing.

### End to end, through stock `lerobot-train`

The numbers above are the model alone. Run both arms as real training on the
same 100 episodes, 300 steps, `--log_freq=20`, and read lerobot's own metrics
(the first logged point is dropped — it averages in the first step, which pays
for cudnn autotuning and the decoders spinning up):

| | `updt_s` | `data_s` |
| --- | ---: | ---: |
| stock | 0.426 | 0.011 |
| **cached** | **0.160** | **0.004** |

**2.66x**, against 4.92x for the model alone, because a real step costs about
**0.08 s more than the same model work on a synthetic batch** — 0.426 against
0.350 stock, 0.160 against 0.071 cached. That constant is 18% of the stock step
and **half of the cached one**, so it is the next thing worth attacking. It was
invisible while the tower dominated.

### It is not lerobot's trainer, and that was worth checking

The obvious suspect is that lerobot's `update_policy` does more than a step
needs. It does, and it costs almost nothing. Adding its extras to a bare step one
at a time (`benchmark_trainer_overhead.py`, batch 64, eager):

| added | stock | cached |
| --- | ---: | ---: |
| bare forward/backward/AdamW | 0.3502 | 0.0715 |
| `policy.train()` every step | +0.0027 | +0.0019 |
| `clip_grad_norm_` over all parameters | +0.0016 | +0.0027 |
| `loss.item()` and `grad_norm.item()` | +0.0018 | +0.0012 |
| `unwrap_model` | +0.0002 | -0.0001 |
| the whole thing through `Accelerator` | +0.0027 | +0.0020 |
| **total** | **0.3592** | **0.0792** |

**+0.009 s, an eighth of the gap.** A faithful replica of `update_policy`,
accelerate included, is within 3% of a bare loop. So **replacing lerobot's
trainer with a hand-written one would buy nothing**, and `data_s` says it is not
the dataloader either. Whatever the remaining ~0.07 s is, it is neither.

**What it is remains unmeasured.** The one structural difference left between the
two measurements is that the benchmark reuses one batch of device tensors every
step while the real loop gets fresh ones, so allocator behaviour and host-to-
device traffic that `data_s` does not capture are the leading suspects. Anyone
picking this up should profile the live loop rather than trust that guess — it is
worth roughly 2x on the cached configuration, which is more than any remaining
model-side lever.

Two things the cache removes for free: training never decodes video, so `data_s`
collapses and the **1.68x spread between hosts** — which that section attributes
to host CPU — stops applying to the training phase. And `num_workers` stops
being a memory cliff, because a worker now reads 240 KiB from a memory map
instead of decoding two 512x512 frames.

**What is left is not worth chasing.** After the tower, the step is the VLM and
expert layers plus the backward. The frozen VLM's *prefix* path is cacheable too,
and by a less obvious argument worth recording: prefix `att_masks` are `0` for
image and language tokens and `1` for the state token, and `make_att_2d_masks`
lets a token attend only where the cumulative mask is no larger than its own — so
the image and language tokens never see `state_proj`, the one trainable thing in
the prefix, at any depth. Their representations through all 16 VLM layers are
frozen functions of the pixels as well.

It still is not worth it. Caching them means storing per-layer keys and values:
16 layers x 2 x ~177 tokens x 320 = **3.6 MB per sample** against the tower
block's 240 KiB, fifteen times the disk, to remove maybe half of the ~8% of the
old step that the prefix VLM path costs.


## LoRA buys nothing, measured

Worth stating because it is the obvious thing to reach for, and `PI05.md` makes
it look relevant. Same host, same batch, eager, only the adapters moving:

| | s/step | samples/s | trainable |
| --- | ---: | ---: | ---: |
| dense | 0.3498 | 183.0 | 99,880,992 |
| rank-16 LoRA | 0.3474 | 184.2 | 742,656 |

**0.7%, which is noise.** A 134-fold cut in trainable parameters bought nothing
measurable, and the stage split says why: the backward is 20% of a step and AdamW
is 1%. Those two are all LoRA can touch, and it does not even empty them — the
activations still have to be backpropagated through the frozen layers to reach
the adapters. The forward, where two thirds of the time is, is untouched.

So LoRA on this model is a memory and checkpoint-size decision, not a speed one.
The SmolVLA run already trains densely (`use_peft: False`, 22.2% of parameters);
leave it that way.


## torch.compile is worth 1.45x

Measured head to head on one host, same dataset, batch, seed and step count,
only `--policy.compile_model` moving:

| | `updt_s` median | min | `data_s` |
| --- | ---: | ---: | ---: |
| stock | 0.5050 | 0.4530 | 0.032 |
| `compile_model=true` | **0.3480** | 0.3350 | 0.038 |

**+45%**, or 7.01h against 4.83h over 50,000 steps. `COMPILE_MODEL=true` on
`vast_smolvla_train.sh`.

**Measure it with `updt_s` past step 150, never with tqdm's rate.** The wall
clock for 400 steps was 306 s stock against 747 s compiled, because
`max-autotune` spends about seven minutes compiling before the first step. A
tqdm running mean therefore reports compile as *slower*, which is backwards for
any run longer than about twenty minutes.

Nothing went wrong that was expected to: **zero graph breaks and zero
recompilations**. That is the dataset's doing rather than luck — it carries
exactly one task string, so `pad_language_to="longest"` gives a constant
sequence length, and a square dataset gives a constant image shape. A
multi-task or non-square dataset would have to re-establish this.

The one reservation is that enabling it also runs
`set_float32_matmul_precision("high")`, switching fp32 matmuls to TF32, so a
compiled checkpoint is not numerically identical to one trained without it.
Evidence that this does not matter much: both arms logged `loss:0.194` at step
200 and `loss:0.172` at step 400, identical at logged precision. That is 400
steps at three decimals under bf16 autocast, not a guarantee over 50,000, so
the flag stays off by default and a ladder should not change it mid-run.


## Batch size is not a throughput lever at all, above 32

Swept on one host with synthetic batches -- no dataloader, so `num_workers` and
the container memory limit cannot confound it:

| batch | uncompiled samples/s | compiled samples/s | compiled peak VRAM |
| ---: | ---: | ---: | ---: |
| 16 | 126.5 | 213.0 | 3,918 MiB |
| 32 | 167.4 | 225.8 | 6,789 MiB |
| 64 | 168.4 | **229.4** | 11,662 MiB |
| 128 | 165.7 | **230.2** | 22,262 MiB |

Throughput rises and plateaus rather than being flat: compiled, 32 to 64 is
**+1.6%** and 64 to 128 a further **+0.3%**. Only medians were recorded, no
spread, so whether 1.6% is real or noise is unmeasured -- but either way it is
~1-2% against `torch.compile`'s 36% and a 1.68x spread between hosts, so batch
size is not where speed comes from. Choose it on optimization grounds or for
VRAM headroom: batch 32 gives 98.4% of batch 64's throughput for 58% of the
memory.

Batch 16 is the one genuinely bad choice, at -7% compiled and -25% uncompiled.
Compile's gain holds across the whole range (1.68x at 16, ~1.36x at 32-128), so
it is not an artifact of one batch size.

This **supersedes an earlier measurement here that reported batch 128 as 5.9%
slower**. That arm was being OOM-killed while it was measured, so it recorded
memory pressure rather than batch size. It also supersedes the much older "+18%
at batch 112", which was measured while the run was decode bound and a larger
batch amortized decode stalls.

Compare throughput in **samples/s**, never s/step: a batch-128 step does twice
the work, so s/step makes the larger batch look worse by construction. Halving
the step count at double the batch is the same sample budget.

**Watch the container memory limit rather than VRAM.** The OOM above was not
VRAM -- 25,044 MiB of 32,607 fits -- but the container's **57 GB cgroup limit**,
against 16 workers prefetching batch-128 frames on a box whose `free` reported
440 GB and whose `nproc` reported 192, both the *host's*. `rc=137` was the only
evidence: nothing in the training log, nothing in the container's `dmesg`. Read
the real limit from `/sys/fs/cgroup/memory/memory.limit_in_bytes`.


## NVDEC decode works, and is 3x faster -- correcting an earlier claim

This document previously said lerobot 0.5.1 "cannot move this to NVDEC --
`decode_video_frames_torchcodec` takes no device". The *function* takes no
device, but torchcodec's `VideoDecoder` does, and on the pinned stack
`device="cuda"` works. Measured on this dataset's own video, 128 random-access
frames -- the pattern training uses, batch 64 across two cameras:

| | 128 random frames | throughput |
| --- | ---: | ---: |
| CPU, as lerobot decodes today | 0.106 s | 1,212 frames/s |
| **`device="cuda"` (NVDEC)** | **0.035 s** | **3,691 frames/s** |

Compiled training needs 457 frames/s, so NVDEC has ~8x headroom. Two benefits
beyond the 3x: frames arrive **already on `cuda:0`**, removing the host-to-device
copy, and the work runs on a fixed-function block rather than the SMs, so it
does not compete with training -- the decoded frames cost about 0.7 GB/s of
memory bandwidth against the card's ~1.8 TB/s.

This matters more for *variance* than for speed. On a well-provisioned host
`data_s` is already 0.01-0.06 s, so there is little to win; what NVDEC removes
is the dependence on host CPU, which is the measured cause of the 1.68x spread
between hosts. Note also that `torch.compile` raised the CPU needed per
GPU-second by ~36%, so hosts that fed the GPU adequately before are closer to
the edge now.

Not yet established: whether decoding inside a DataLoader worker works, since
forked workers cannot inherit a CUDA context -- `num_workers=0` with a prefetch
stream, or a spawn start method, are the routes around it.
