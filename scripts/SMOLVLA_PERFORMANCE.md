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

The short version, on an RTX 5090 at batch 64, synthetic batches, with the
language padded to the task string (below):

| | s/step | samples/s | peak VRAM |
| --- | ---: | ---: | ---: |
| stock, eager | 0.3420 | 187.1 | 11,446 MiB |
| stock, compiled | 0.2571 | 249.0 | 9,165 MiB |
| cached tower output | 0.1179 | 542.8 | 12,059 MiB |
| cached, frozen prefix out of the backward | 0.0779 | 821.4 | 5,517 MiB |
| cached, compiled | 0.0619 | 1,033.4 | 8,468 MiB |
| **cached, frozen prefix, compiled** | **0.0410** | **1,559.2** | 4,446 MiB |

Four changes, and they multiply rather than overlap. **Caching the frozen tower's
output is 2.90x.** **Not running 36 tokens of language padding is 1.27x** and is
already inside every number above. **Taking the frozen prefix out of the backward
is 1.51x** — and it is 1.51x again on top of `torch.compile`, which is itself
1.90x on a cached step. Together they are **8.34x** the stock eager step, and a
quarter of its memory.

The training loop itself is lerobot's, unmodified: `vast_smolvla_train.sh` calls
`lerobot-train`, and `train_smolvla_cached.py` still does — it swaps the dataset,
two methods on the policy and the tokenizer's padding, and changes nothing else.
Each of the three has a flag that turns it off for an A/B.

### These numbers replace the ones this file carried before 2026-08-14

The cached rows used to read 0.0711 and 0.0266, for a 4.92x and a 13.2x. Those
were measured on a synthetic batch carrying **3 tokens per camera instead of
64**: `_tower_token_count` probed a policy whose `embed_image` had already been
replaced by the cached path's identity, so it read `shape[1]` off a
`[1, 3, H, W]` image and got the channel count. Every cached arm ran on a
twentieth of the image tokens training uses. The tower row, the stock rows, the
LoRA comparison and the batch-size sweep are unaffected — none of them build a
cached batch — and the tower's own share of a step, the reason the cache exists,
is unchanged.

The same bug is why a live step looked **0.08 s more expensive than a synthetic
one**, which this file called the next 2x to be had. It was never there: a
synthetic batch built with the tower's real token count and the padding a real
batch carries costs **0.1199 s against a live step's 0.1236 s**, a 3% gap that is
the dataloader hand-off. The lesson is cheap to state and was expensive to find —
**a synthetic batch is a claim about the real one, so measure its shape rather
than deriving it**, and the benchmark now records `language_tokens` and
`prefix_tokens_per_camera` in every result for exactly that reason.

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

Measured back to back on one host, RTX 5090, batch 64, synthetic batches, 64
tokens per camera and 12 language tokens in every arm:

| arm | s/step | samples/s | peak VRAM |
| --- | ---: | ---: | ---: |
| stock, eager | 0.3420 | 187.1 | 11,446 MiB |
| **cached, eager** | **0.1179** | **542.8** | 12,059 MiB |
| stock, compiled | 0.2571 | 249.0 | 9,165 MiB |
| **cached, compiled** | **0.0619** | **1,033.4** | 8,468 MiB |
| tower alone | 0.2228 | 574.6 images/s | 4,887 MiB |

**2.90x eager, and 4.15x compiled**, or 5.52x for compiled-and-cached against the
stock eager step. `torch.compile` is worth 1.33x on the stock step here
(reproducing the 1.45x that section measured on another host) and **1.90x** on
the cached one, where the graph left after the tower is small enough for
`max-autotune`'s CUDA graphs to matter.

The one cost is that `max-autotune` compiles for twenty to thirty minutes before
the first step on this configuration — much longer than the seven minutes that
section records — so it is only worth enabling on a run of real length. Time it
with `updt_s` past step 150, never with tqdm's rate.

The gain is the tower's own share and nothing else, which is what the stage split
says:

| stage | stock | cached |
| --- | ---: | ---: |
| tower | 0.2226 (65.1%) | 0.0000 |
| rest of forward | 0.0495 | 0.0474 |
| backward | 0.0680 | 0.0680 |
| AdamW | 0.0030 | 0.0030 |

Every stage but the tower is the same to within a millisecond, which is the
result to expect and did not use to be there: the earlier table showed the rest
of the step getting 1.75x faster too, and speculated that halved VRAM caused it.
Both were the 3-token batch. **Peak VRAM does not fall either** — the cached arm
is slightly *higher*, because the tower's activations are replaced by a wider
batch of embeddings arriving from the dataloader.

**The cache pays for itself after 1.2 epochs.** Building it is one pass over the
dataset, measured at **194 frames/s** over 100 episodes with 8 workers — which
extrapolates to **25 minutes** for all 291,618 frames. That is *decode* bound,
not tower bound: the tower alone sustains 285 frames/s, so more workers would
close some of the gap. It is the last time the run pays for h264 at all.

Against 187.1 samples/s stock and 821.4 cached with the prefix split off, the
break-even is about 363,000 samples — 5,700 steps at batch 64, or 1.25 epochs.
Every run this project has done is far past that. For a 50,000-step run at batch
64, which is 10.97 epochs, using the end-to-end `updt_s + data_s` below rather
than the model alone:

| | training | + cache build | total |
| --- | ---: | ---: | ---: |
| stock, eager | 5.50 h | — | **5.50 h** |
| cached, repadded, split off | 1.15 h | 0.42 h | **1.57 h** |
| the same, compiled | 0.65 h | 0.42 h | **1.07 h** |

At $0.78/hr that is **$4.29 against $0.83**, and a working day's iteration loop
against a lunch break. The compiled row excludes `max-autotune`'s twenty to
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

The numbers above are the model alone. Run the arms as real training on the same
100 episodes, 300 steps, `--log_freq=20`, and read lerobot's own metrics (the
first logged point is dropped — it averages in the first step, which pays for
cudnn autotuning and the decoders spinning up):

| | `updt_s` | `data_s` |
| --- | ---: | ---: |
| stock | 0.379 | 0.017 |
| cached, language padded to 48 | 0.153 | 0.002 |
| cached, padded to the task string | 0.121 | 0.002 |
| cached, padded, frozen prefix split off | 0.081 | 0.002 |
| cached, padded, compiled | 0.065 | 0.002 |
| **cached, padded, split off, compiled** | **0.045** | 0.002 |

**8.4x**, and a live step now costs what the model costs: 0.121 against the
0.1179 the same work takes on a synthetic batch, 0.081 against 0.0779, 0.045
against 0.0410. The 0.08 s that used to sit between them was the 3-token batch
and the padding, not the loop. The stock row keeps an 11% gap because it still
decodes video and copies two 512x512 frames per sample to the device.

### The live loop was worth building anyway

`benchmark_live_step.py` runs the real dataset, preprocessor and Accelerator and
bisects a step three ways — a synthetic batch, one real batch frozen and reused,
and a fresh batch every step. Cached, at the padding a run uses now:

| arm | step | `data_s` |
| --- | ---: | ---: |
| synthetic | 0.1199 | — |
| live, 8 workers | 0.1236 | 0.0021 |
| live, 0 workers | 0.2210 | 0.1012 |

**3%** between synthetic and live with workers, and the split inside the step is
identical to four decimal places. The zero-worker row is worth keeping in view:
without workers the cached read costs 0.10 s a step, so `num_workers` still
matters after the video decoding is gone — it is just no longer a memory cliff.

### It is not lerobot's trainer either, and that was worth checking

The obvious suspect is that lerobot's `update_policy` does more than a step
needs. It does, and it costs almost nothing. Adding its extras to a bare step one
at a time (`benchmark_trainer_overhead.py`, batch 64, eager):

| added | stock | cached |
| --- | ---: | ---: |
| bare forward/backward/AdamW | 0.3431 | 0.1180 |
| `policy.train()` every step | +0.0014 | +0.0011 |
| `clip_grad_norm_` over all parameters | +0.0011 | +0.0010 |
| `loss.item()` and `grad_norm.item()` | +0.0002 | +0.0002 |
| `unwrap_model` | +0.0001 | +0.0000 |
| the whole thing through `Accelerator` | +0.0018 | +0.0017 |
| **total** | **0.3477** | **0.1220** |

**+0.004 s**, 1.3% of a stock step and 3.4% of a cached one. A faithful replica
of `update_policy`, accelerate included, is that close to a bare loop, so
**replacing lerobot's trainer with a hand-written one would buy nothing**. The
three diagnostic `.item()` syncs inside `SmolVLAPolicy.forward` are the same
story: removing them measured 0.5%, inside the noise, and is not worth carrying a
rewritten `forward` for.

**Export `ACCELERATE_MIXED_PRECISION=bf16` before running this.** Without it
`accelerator.autocast()` is a no-op, the accelerate row runs in float32, and the
ladder reports it as +0.44 s — a 2.3x regression that is the benchmark's
configuration rather than accelerate's cost.

The cache removes something else for free: training never decodes video, so the
**1.68x spread between hosts** — which that section attributes to host CPU —
stops applying to the training phase.


## The language padding is 1.27x, and it is free

`SmolVLAConfig.pad_language_to` is `"longest"`, but the tokenizer is not built
from that config: `make_pre_post_processors` loads the processor saved beside the
checkpoint, and `smolvla_base`'s pins `padding="max_length"` with
`max_length=48`. This dataset's one task string is **12 tokens**, so a step ran
36 tokens of padding through all 16 VLM layers and their backward.

They are masked out of everything — `embed_prefix` builds `pad_masks` from the
attention mask, `make_att_2d_masks` lets nothing attend to a padded position, and
`position_ids` come from a cumulative sum that padding does not advance. The
tokens cost time and change nothing:

| language tokens | cached step | samples/s |
| ---: | ---: | ---: |
| 48, as the checkpoint pads | 0.1500 | 426.8 |
| **12, the task string** | **0.1179** | **542.8** |

**1.27x**, and 0.153 to 0.121 end to end. `train_smolvla_cached.py` now does this
by default and says so at startup; `--language-padding max_length` is the A/B.

The change travels with the checkpoint: every `pretrained_model/` a run writes
records `"padding": "longest"`, so a policy is evaluated the way it was trained
rather than reverting to the base checkpoint's 48 tokens.

**The loss moves by ~1e-3 relative, and that is arithmetic order rather than a
different computation.** The VLM is loaded with `torch_dtype="bfloat16"`, so
changing the sequence length changes how things accumulate.
`check_smolvla_prefix_cache.py` sweeps padding lengths that are all
mathematically identical:

| language tokens | loss | drift from 48 |
| ---: | ---: | ---: |
| 48 | 1.27683592 | — |
| 47 | 1.27468407 | 2.152e-03 |
| 24 | 1.27416790 | 2.668e-03 |
| 12 | 1.27530575 | 1.530e-03 |

Dropping **one** padding token moves the loss further than dropping all 36 does.

Two caveats. This is worth 1.27x because *this* task string is short; a dataset
whose prompts fill the 48 tokens would gain nothing. And `"longest"` is a
constant length only because the dataset carries exactly one task string — a
multi-task dataset would give a length that varies per batch, which recompiles
under `torch.compile`.


## The prefix backward is 1.51x at batch 64, and it was running for nothing

The 1.51x is a batch-64 figure on one host. The split's worth grows with
batch size and turns negative at the small end -- 0.96x at 16 up to 1.50x at
128, remeasured on another host under the batch sweep below, where it is also
what makes batch 256 fit at all.

After the tower and the padding, a cached step is 0.1179 s: 0.0474 forward,
0.0680 backward, 0.0030 AdamW. Almost all of it is the prefix. Sweeping the
tokens per camera in the cached arm, everything else held:

| tokens per camera | prefix tokens | s/step | samples/s |
| ---: | ---: | ---: | ---: |
| 64 (what the tower emits) | 141 | 0.1177 | 544.0 |
| 48 | 109 | 0.0910 | 703.4 |
| 32 | 77 | 0.0694 | 921.8 |
| 16 | 45 | 0.0500 | 1,280.4 |
| 4 | 21 | 0.0391 | 1,637.8 |
| 1 | 15 | 0.0370 | 1,727.5 |

**The image tokens are 0.081 s of the 0.118 s step, 69% of it.** The expert and
everything else is the 0.037 s floor.

That prefix is frozen, by the argument this file has recorded for a while: prefix
`att_masks` are `0` for image and language tokens and `1` for the state token,
and `make_att_2d_masks` lets a token attend only where the cumulative mask is no
larger than its own — so the image and language tokens never see `state_proj`,
the one trainable thing in the prefix, at any depth. Their representations
through all 16 VLM layers, and the keys and values they contribute, are frozen
functions of the pixels and the task string.

So their **backward is pure waste** — and it runs anyway. Autograd cannot see the
argument: every layer concatenates prefix and suffix into one tensor, and the
suffix requires grad, so the prefix half of that tensor does too from layer 1
onward. Confirmed by measurement: freezing `state_proj` (`train_state_proj=false`,
which makes nothing in the prefix trainable) changes the step by **0.5%**,
0.1177 to 0.1171. The prefix backward survives the freeze.

So `smolvla_frozen_prefix.py` splits the stack by hand: the frozen 140 tokens run
under `no_grad`, and the state token and the action tokens run with grad against
the keys and values those produced. Same projections, same rotary positions, same
mask slices, same keys in the same order — only which tensors autograd keeps
changes. It is a fork of `SmolVLMWithExpertModel.forward`'s layer loop, about 150
lines, because the model's own prefill-then-decode path assumes the VLM stream is
finished by the time the cache is used, and here the state token still has to
travel with gradients.

| arm | s/step | samples/s | peak VRAM |
| --- | ---: | ---: | ---: |
| cached | 0.1179 | 542.8 | 12,059 MiB |
| **cached, frozen prefix split off** | **0.0779** | **821.4** | **5,517 MiB** |
| cached, compiled | 0.0619 | 1,033.4 | 8,468 MiB |
| **cached, split off, compiled** | **0.0410** | **1,559.2** | **4,446 MiB** |
| stock | 0.3420 | 187.1 | 11,446 MiB |
| stock, frozen prefix split off | 0.3026 | 211.5 | 6,257 MiB |

**1.51x cached, 1.51x again on top of `torch.compile`, and VRAM less than half** —
the activations of 140 tokens through 16 layers stop being stored. End to end
through `lerobot-train`, `updt_s` goes **0.121 to 0.081**. It is 1.13x rather
than 1.51x on the stock arm because the tower still dominates there.

### Whether it computes the same thing

A forward check would not settle this, since the point of the change is the
backward. `check_smolvla_frozen_prefix.py` compares **gradients**, parameter by
parameter, on one batch with the noise and time held fixed — and carries its own
scale for "unchanged", because bit-identical is not available: the split reduces
over shorter sequences and the weights are bfloat16. The control is the padding
above, a change the attention masks make mathematically invisible.

| | loss moves by | worst gradient moves by |
| --- | ---: | ---: |
| frozen prefix split off | 1.9e-03 | 4.4% (`state_proj.weight`) |
| control: 12 language tokens against 48 | 1.8e-03 | 128% (`action_out_proj.bias`) |

The split moves the loss by what an invisible change moves it by, and the
gradients **thirty times less**. Read the control the other way too: on a
synthetic batch the gradients are sums of near-cancelling terms, so a *relative*
gradient difference of tens of percent is what bfloat16 alone produces, and no
equivalence claim about gradients on this model is worth anything without that
control beside it.

Two things *not* to reach for. Storing those per-layer keys and values on disk
instead is 16 layers x 2 x 141 tokens x 320 = **2.9 MB per sample** against the
tower block's 240 KiB, twelve times the disk, for a saving that recomputing them
under `no_grad` gets for nothing. And `train_state_proj=false` buys the 0.5%
above, not the backward — it is a model change for no speed.

**The patch has to keep the compiled artifact.** `SmolVLAFlowMatching.__init__`
implements `compile_model` as `self.forward = torch.compile(self.forward)`, so a
patch that assigns `forward` afterwards silently throws it away. That is how this
was first measured: the compiled split arm reported 0.0780 s against the eager
0.0779 s, which is not a compile that failed to help but a compile that was never
there.


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

That is compile against the *stock* step. It is worth more as the step gets
cheaper — 1.90x on a cached one and 1.51x on a cached one with the prefix split
off — so the headline table is where to read what it buys now.

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


## Batch size is a throughput lever again, once the tower is out of the step

Measured 2026-08-14 on one RTX 5090, synthetic batches -- no dataloader, so
`num_workers` and the container memory limit cannot confound it -- in the
configuration a run uses now: cached tower output, the frozen prefix out of the
backward, language padded to the task string.

| batch | eager samples/s | eager peak VRAM | compiled samples/s | compiled peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 239.8 | 2,663 MiB | -- | -- |
| 32 | 381.5 | 3,596 MiB | 808.5 | 3,055 MiB |
| 64 | 472.0 | 5,517 MiB | 915.3 | 4,583 MiB |
| 128 | **507.0** | 9,363 MiB | 989.4 | 7,650 MiB |
| 256 | 504.1 | 17,093 MiB | **1,058.4** | 13,244 MiB |

**This supersedes the sweep this file carried before, and supersedes its
conclusion, not just its numbers.** That sweep found compiled throughput flat
from batch 32 upwards -- 225.8, 229.4, 230.2 at 32, 64, 128, a total spread of
2% -- and concluded batch size was not where speed came from. It was measured
while the frozen vision tower was 65% of a step and saturated the GPU at any
batch. With the tower cached and the prefix split off, a step is small enough
that launch and per-step overhead are visible, and the curve rises again:
**eager, 32 to 64 is +24% and 64 to 128 a further +7%**; **compiled, 64 to 128
is +8% and 128 to 256 another +7%**, flattening at 256 (batch 512 is a further
+1.2%, below).

`torch.compile` is worth 1.94-2.12x across the whole range, so the two levers
are independent: neither is an artifact of the other.

### Where the plateau sits, and it is 256

Eager it arrives at 128: batch 256 is 0.6% *slower*, and **batch 512 does not fit
at all** -- CUDA out of memory against 31.4 GiB, which the 9,363 to 17,093 MiB
step from 128 to 256 already predicted. Compiled it arrives at 256. Measured on a
second host, which is 1.57x faster than the first and so is quoted only against
itself:

| batch | compiled samples/s | compiled peak VRAM |
| ---: | ---: | ---: |
| 256 | 1,658.0 | 13,413 MiB |
| 512 | **1,677.7** | 24,858 MiB |

**Batch 512 buys 1.2% for 1.85x the memory**, so the curve that was still
climbing at 256 stops there. Compiled 512 is the largest batch that fits on a
32 GB card, and it is not worth using: 25 GB leaves nothing for a longer chunk,
a second camera, or an unfrozen encoder.

That second host is also the cleanest available illustration of the rule at the
top of this file. It ran **760.0 samples/s eager at batch 128** where the first
ran 507.0, and **1,658.0 compiled at 256** against 1,058.4 -- 1.50x and 1.57x
apart on identical arms, identical code and the same advertised GPU. The
*shape* of the curve reproduced on both; none of the absolute numbers did.

### What that is worth in wall clock, which is less than the percentages suggest

Compare a fixed *sample* budget, never a step count: 10,000 steps at batch 128
sees twice the data that 10,000 steps at batch 64 does, so a step count makes the
larger batch look worse by construction. Per **640,000 samples** -- what 10,000
steps at batch 64 is -- compiled on the first host:

| batch | s/step | per 10,000 steps | per 640,000 samples |
| ---: | ---: | ---: | ---: |
| 64 | 0.0699 | 11.7 min | **11.7 min** |
| 128 | 0.1294 | 21.6 min | **10.8 min** |
| 256 | 0.2419 | 40.3 min | **10.1 min** |

A 3,200,000-sample budget -- what 50,000 steps at batch 64 is -- therefore takes
**0.97 h compiled at batch 64, 0.90 h at 128 and 0.84 h at 256** on that host.
The whole lever is about eight minutes and six cents on a one-hour run. The old
advice "choose batch size on optimization grounds, not for speed" therefore
survives at 64 and above, even though the measurement it rested on does not.

**The small end is where this now costs real time.** Batch 32 is 81% of batch
64's compiled throughput and batch 16 is under half of batch 256's eager
throughput. The superseded sweep called 16 "the one genuinely bad choice" at -7%
compiled; it is worse than that now, and 32 has joined it. Dropping batch size
for VRAM headroom used to be nearly free and is not.

### The prefix split is what makes the large end reachable

The frozen-prefix split is not a constant factor -- it grows with batch, because
what it removes is a backward over 140 prefix tokens per sample:

| batch | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unsplit / split | 0.96 | 1.23 | 1.38 | 1.50 | OOM | OOM |

At batch 16 the split is a *loss*, which is the same launch-bound small end seen
from the other side: the split runs the prefix as a second forward, and at 16 the
extra launches cost more than the backward it removes. And **the unsplit arm at
batch 256 does not fit at all** -- CUDA out of memory against 31.4 GiB, where the
split arm peaks at 17,093 MiB. The largest batch on the curve exists because of
the split.

### The cache read does not bend the curve, which was the thing to check

A synthetic sweep holds its batch on the device; a real one reads 240 KiB of
cached prefix per sample off disk every step, which is 15 MB at batch 64 and
61 MB at 256. That was the reason to expect the synthetic curve to flatter the
large sizes. Measured through `benchmark_live_step.py`, over a 100-episode cache,
it does not:

| batch | synthetic samples/s | live samples/s | `data_s` | live peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 225.6 | 212.9 | 0.0015 | 2,661 MiB |
| 32 | 368.4 | 340.7 | 0.0018 | 3,610 MiB |
| 64 | 459.9 | 439.6 | 0.0026 | 5,528 MiB |
| 128 | 500.2 | 492.5 | 0.0035 | 9,385 MiB |
| 256 | 500.6 | 495.4 | 0.0049 | 17,155 MiB |

`data_s` grows from 1.5 ms to 4.9 ms across a 16x change in batch -- sublinear,
and 1% of a step at every size. Live tracks synthetic within 6% at 16 and within
1% at 256, so the gap *narrows* as the batch grows. The synthetic sweep can be
read as the live one.

**Watch the container memory limit rather than VRAM.** An earlier batch-128 run
was OOM-killed here, and it was not VRAM -- it was the container's **57 GB cgroup
limit**, against 16 workers prefetching frames on a box whose `free` reported
440 GB and whose `nproc` reported 192, both the *host's*. `rc=137` was the only
evidence: nothing in the training log, nothing in the container's `dmesg`. Read
the real limit from `/sys/fs/cgroup/memory/memory.limit_in_bytes`. That failure
is about the dataloader and is unchanged by anything above; a batch-256 run has
to size `num_workers` against it.

### Reproducing this

`scripts/vast_smolvla_batch_sweep.sh` on one rented host, about 50 minutes and
$0.40 including the compiled arms, which are 20-30 minutes of max-autotune each
and run last for that reason. Results are in
`$PAP_DATA_ROOT/smolvla-speed/2026-08-14-batch-sweep/` and in S3 at
`outputs/smolvla-batch-sweep/`; the 512 arms and their 256 anchor are the
`-512` siblings of both paths. **Set `RUN_NAME` when adding arms**, as those did:
the launcher keys its output prefix on it, and a second sweep under the default
name would overwrite the first, which is the collision this project has already
paid for once in its evaluation artifacts.

**None of this says to change the recipe.** Ten scored rungs exist at batch 64,
and a run at another batch size is not comparable to them. This is the speed
question; whether a different batch size trains a better policy is a separate one
that nothing here measures.


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
