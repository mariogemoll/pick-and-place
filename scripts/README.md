<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Job scripts

Long-running jobs: recording datasets, and training a policy on a rented GPU.
Everything here assumes `PAP_DATA_ROOT` points at a directory outside the
repository, and that the Python environment is the one `AGENTS.md` describes.

Anything that compiles a MuJoCo scene also needs the generated AprilTag
textures, which are not in the repository — see "The simulator needs generated
AprilTag textures" in `AGENTS.md`. `vast_pap_provision.sh` renders them, so
rented pods are covered; a fresh local clone is not.

## Does the cube's appearance decide whether the policy learns?

The working Diffusion Policy is trained on a blue cube. The physical cube
carries AprilTags, because that is how its pose is measured on hardware, so a
blue-cube policy has no path to the real robot. An earlier tagged-cube run
failed, but that comparison was confounded — different hardware, a different GL
backend, a later checkout and a separately recorded dataset — so it bounded
sample efficiency rather than showing the tagged cube cannot be learned.

`record_two_variant_dataset.sh` produces the unconfounded version of that
comparison:

```sh
export PAP_DATA_ROOT=~/pick-and-place-data
EPISODES=1000 WORKERS=12 scripts/record_two_variant_dataset.sh
```

It records once with the tagged cube, then renders both appearances from a
single replay of each recorded frame. The two variants therefore share states,
actions and phase spans bit for bit and differ only at the cube's pixels, so a
difference between two policies trained on them is attributable to the
appearance and nothing else.

The cube cannot simply be recorded blue: under `--miscalibration` the descent
visual servo detects the cube's AprilTags in the wrist image, so a solid-colour
cube fails every episode.

**Record and re-render on the same machine.** The camera calibrations are
machine-local files and the OpenGL backend decides the shading, so the
verification pass the script runs is only evidence for re-renders produced
beside it. The script refuses to render without it.

Then train each variant, changing nothing but the dataset:

```sh
ARTIFACT_NAME=two-variant-1000-as-recorded RUN_NAME=<fresh> \
  scripts/vast_diffusion_policy_train_fast.sh
ARTIFACT_NAME=two-variant-1000-blue-cube   RUN_NAME=<fresh> \
  scripts/vast_diffusion_policy_train_fast.sh
```

`vast_diffusion_policy_train_fast.sh` only downloads from S3 when
`$ARTIFACT_NAME/train.npz` is missing, so a locally recorded dataset trains
without an upload round trip — export into `/workspace/artifacts` on the pod by
setting `ARTIFACTS_DIR`. Leave `BATCH_SIZE` and `N_EPOCHS` at their defaults for
both runs, or the comparison measures the hyperparameters instead of the cube.

### Checking a pair by hand

```sh
python py/scripts/check_variant_pair.py <export-a> <export-b>
```

Exits non-zero unless the states, actions and episode lengths are identical. It
reports how much of each camera's frame the appearance touches rather than
judging it — on a 96x96 export the cube is a few tenths of a percent of the
overhead frame and roughly a tenth of the wrist frame.

### Cost

Roughly, on a 12-worker GPU box: recording 1000 episodes ~80 min, re-rendering
both variants ~45 min, one training run ~30 min. Recording is resumable — rerun
the same command and it continues at the next unused episode index.

**Delete any partial `epNNNNNN/` left by an interrupted recording** before
rerunning. A directory with a `meta/` but no `meta/episodes/chunk-*/` makes the
next run's final tally fail with `ValueError: No objects to concatenate`, after
the new episodes have already been recorded successfully. The episodes are not
lost; the tally is.

## Training on the reference dataset

`vast_diffusion_policy_train_fast.sh` trains the blue-cube policy from the
published artifact. Its header documents the hyperparameters that are
load-bearing and the ones that were tried and reverted; read it before changing
anything.

## LoRA-finetuning pi0.5

> **The first run of this scored 0/100 on `canonical_100_v1`.** It took
> lerobot's `image_transforms.enable=false` default and trained 1.10 epochs;
> both are now believed to be the cause, and the launcher enables augmentation.
> Weigh a retry against what already works: the flow policy reaches 0.71 and
> DPPO 0.746 on this task for a fraction of the time and cost.


`vast_pi05_lora_train.sh` finetunes `lerobot/pi05_base` on the recorded LeRobot
dataset, on the same rented 5090 the Diffusion Policy runs use:

```sh
RUN_NAME=<fresh> HF_TOKEN=<token> scripts/vast_pi05_lora_train.sh
```

pi0.5 is a 3.3B-parameter VLA and a full finetune is sized for an 80 GB card, so
a 5090 can only run it with adapters. That is not purely a concession to the
hardware — 1000 episodes of a single prompt is thin evidence for moving 3.3B
parameters for a task an analytic planner already solves exactly.

Five things about the configuration are load-bearing:

- **The adapter targets come from the policy.** `_get_default_peft_targets()` in
  lerobot's `modeling_pi05.py` adapts the action expert's q/v projections along
  with `state_proj`, `action_in_proj`, `action_out_proj` and the action-time
  MLPs. Those projections carry the 6-DOF joint mapping, which has no
  pretrained equivalent, so they have to be trainable. They are adapted with
  LoRA rather than fully fine-tuned — `modules_to_save` is empty in the emitted
  `adapter_config.json`, so all 1,287,168 trainable parameters are rank-16
  adapters and nothing is trained densely. Leaving `--peft.target_modules`
  unset is what selects this set; setting it replaces the whole regex.

- **A checkpoint is not self-contained.** `adapter_model.safetensors` is 5 MB
  and means nothing without its base. The emitted `adapter_config.json` records
  `base_model_name_or_path` as the *absolute path the pod used*
  (`/workspace/pi05_base_pinned`), so loading the checkpoint anywhere else
  fails until that path exists or the field is repointed. The run's
  `job-metadata/checkpoint-revision.txt` records which `lerobot/pi05_base`
  revision to materialize there.
- **`--policy.pretrained_path` loads weights only.** Feature names then come
  from the dataset, so `observation.images.overhead` and `.wrist` pass straight
  through and no `--rename_map` is involved — which is why this cannot repeat
  the SmolVLA camera-ordering bug. The cost is that every stored config value
  reverts to its class default, so `n_action_steps` and `empty_cameras` have to
  be passed explicitly.
- **The base checkpoint is pinned to a revision.** `lerobot/pi05_base` moves
  independently of the pinned `lerobot`, and its commit `7de663972b`
  (2026-06-03) added a `relative_actions_processor` step that 0.5.1's registry
  does not know, so loading HEAD fails in `make_pre_post_processors`. The
  launcher materializes `a538eb2732`, the commit before it, and passes it as a
  path — `from_pretrained` takes a revision but no config field exposes one.
  Unpinning `lerobot` is the expensive direction: 0.5.1 pins
  `transformers==5.3.0`.
- **The dataset is already in pi0.5's preferred form.** pi0.5 normalizes state
  and action with quantiles, and `meta/stats.json` carries `q01`/`q99`, so
  neither a stats recompute nor a `MEAN_STD` override is needed. The launcher
  asserts this before building the model rather than discovering it on the first
  batch.

The run opens with a five-step smoke stage into a throwaway directory. The gated
tokenizer, the camera keys, the quantile stats and the VRAM ceiling all fail in
the first few steps or not at all, so that stage turns an overnight failure into
a two-minute one.

### Which cube variant

`ARTIFACT_NAME` defaults to `two-variant-1000-as-recorded-lerobot` — the
**AprilTag** half of the two-variant pair, not the blue one the Diffusion Policy
and flow policy were trained on. That is deliberate, and it is the opposite of
the choice made for those policies:

- The physical cube carries AprilTags, so the tagged variant is the only one
  with a path to the real arm. A blue-cube policy cannot be deployed.
- The contrast argument against the tagged cube (0.159 overhead against blue's
  0.522) was measured at 96x96 for small CNN encoders. pi0.5 sees 224x224 —
  5.4x the pixels — through a pretrained SigLIP, and AprilTags are exactly the
  high-frequency texture a large pretrained encoder is good at. Whether that
  holds is one of the things this run is for.

The cost is that a tagged-cube result is **not** directly comparable to the flow
policy's 0.71, which was trained on blue: appearance is confounded with
architecture. Train `two-variant-1000-blue-cube-lerobot` for the clean
comparison — the two variants share states, actions and phase spans bit for bit
and differ only at the cube's pixels.

### Running a finished checkpoint in the sim

The adapter is 5 MB and the base is 14.5 GB, so both are needed. Materialize the
base at the revision the run recorded, then point the sim runner at the adapter:

```sh
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('lerobot/pi05_base', revision='a538eb2732', \
  local_dir='$PAP_DATA_ROOT/pi05_base_pinned')"

aws s3 sync s3://allyouneed/pick-and-place/outputs/<run>/train/checkpoints/020000 \
  "$PAP_DATA_ROOT/<run>-020000"

mjpython py/scripts/run_policy_sim.py \
  --checkpoint "$PAP_DATA_ROOT/<run>-020000/pretrained_model" \
  --base-checkpoint "$PAP_DATA_ROOT/pi05_base_pinned" \
  --recording-hw 720 960 \
  --device mps
```

Two flags are not optional here:

- `--base-checkpoint` (or `PAP_PI05_BASE`) is required off the training box:
  `adapter_config.json` records the base as the absolute path that box used.
- `--recording-hw` is required for any non-Diffusion-Policy controller, which
  otherwise reads it from the export beside its normalization file. Pass the
  resolution the checkpoint's *source dataset* was recorded at — 720x960 for the
  sim datasets, 480x640 for `real-20260701`.

Three things to expect:

- **It is slow.** pi0.5 is 3.3B parameters; on an M1 expect seconds per
  inference. `n_action_steps=10` means one inference per ten control ticks, so a
  30 Hz episode still runs far below real time. Fine for watching behavior,
  useless for timing.
- **Memory.** The checkpoint records `dtype: bfloat16` (6.6 GB of weights);
  float32 needs 13.2 GB. Both fit a 32 GB machine, neither fits 16 GB
  comfortably alongside MuJoCo.
- **`mjpython`, not `python`,** for anything opening a viewer on macOS, or pass
  `--no-viewer`.

The checkpoint's recorded image shape is the dataset's 720x960, so the sim
renders at that and pi0.5 downsamples to 224x224 internally. That is what it
trained on; do not "optimize" it by rendering at 224.

### Scoring the checkpoints

`vast_pi05_eval.sh` scores checkpoints on the frozen manifests, fetching them
from S3 if the pod does not already hold them:

```sh
RUN_NAME=<training run> STEPS="020000 014000" scripts/vast_pi05_eval.sh
```

Each checkpoint gets `smoke_v1` first — eight scenarios, a couple of minutes —
and only reaches `canonical_100_v1` if that passes. Everything that goes wrong
here goes wrong on the first scenario, so paying for a hundred of them before
finding out is pure waste.

Note that `eval_policy_sim.py` deliberately takes neither `--recording-hw` nor a
default `--n-action-steps`: it reads both from the checkpoint. The training
launcher needs the first and the sim runner needed the second, so it is easy to
carry them over by habit and get an argument error.

Scoring several steps is worth the extra minutes. Loss flattens well before the
last checkpoint, so the final one is not automatically the best, and the ladder
is already on disk.

### Sizing the run

The episodes are 9.72 s each, so 1000 of them is 2.70 hours of data — the low
end of the 1–20 hour band Physical Intelligence reports as sufficient, not above
it. The advice that these models need only tens of episodes comes from tasks
with 30–60 s episodes; fifty of these would be eight minutes. `real-20260701` is
18 minutes in total, already below the band, which is the main argument against
finetuning on it alone.

So the budget is spent on steps, not on episodes. `lerobot-train` counts steps,
which makes epoch coverage easy to misjudge:

| batch | steps | samples | epochs over `as-recorded` |
| ---: | ---: | ---: | ---: |
| 16 | 10,000 | 160,000 | 0.55 |
| 16 | 20,000 | 320,000 | 1.10 |
| 32 | 20,000 | 640,000 | 2.19 |

`STEPS` therefore defaults to 20,000 rather than the LIBERO recipe's 30,000:
that recipe runs batch 64, so cutting the step count without accounting for the
batch would have trained on barely half an epoch.

## Finetuning SmolVLA

> **Result: 32/100 on `canonical_100_v1`** (20,000 steps, batch 64, 4.39 epochs,
> ~$2.40), against π₀.₅'s 0/100 on the same dataset and harness. So a VLA does
> learn this task, and the π₀.₅ result was a misconfiguration rather than a
> verdict — at π₀.₅'s exact sample count (5,000 steps, 1.10 epochs) SmolVLA also
> scores 0/8 and cannot move the cube.
>
> **It is still not the right tool here.** DPPO scores 0.746 and the flow policy
> 0.71, both for less compute. 0.32 understates SmolVLA — tagged cube against
> their blue, 4.39 epochs, no tuning — but every one of those is fixed by
> spending *more*, and they already win for *less*, so the gap in
> cost-effectiveness only widens. **Do not run more VLA experiments in
> simulation for this task.**
>
> **Run rollouts at `--n-action-steps 20`, not 10.** A sweep on the 20,000-step
> checkpoint over the full `canonical_100_v1` measured 32/100 at horizon 10,
> **39/100 at 20**, 34/100 at 25 and 24/100 at 50 — a peak, not a trend. Across
> all four, contact stays flat at ~0.87 while `cube_lifted` tracks the score,
> so the horizon decides whether the grasp completes rather than whether the
> cube is found: too short and a replan switches modes mid-grasp, too long and
> the policy is open-loop for 1.7 s. Same effect `FLOW_POLICY_IMAGE.md` measured
> on the image flow policy, which is also flow-matching.
>
> The one live argument is sim2real: `real-20260701` is 18 minutes, far below
> the 1–20 h band, and web-scale plus SO-100/SO-101 pretraining is a plausible
> route to transferring from that little. That needs the real arm to score.

`vast_smolvla_train.sh` is the cheap retry of the question the pi0.5 run left
open, on the same rented 5090:

```sh
RUN_NAME=<fresh> scripts/vast_smolvla_train.sh
```

No `HF_TOKEN`. SmolVLA tokenizes through `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
which is public, so the gated-checkpoint dance pi0.5 needs for
`google/paligemma-3b-pt-224` is simply absent.

### Why this and not another pi0.5 run

pi0.5 scored 0/100, and its notes argue that was a misconfiguration rather than
a verdict. SmolVLA tests the same hypothesis for about a third of the price, and
its shape fits the objection better:

| | pi0.5 | SmolVLA |
| --- | ---: | ---: |
| total parameters | 4,144,691,984 | 450,046,176 |
| trainable | 1,287,168 (0.031%) | 99,880,992 (22.2%) |
| how | rank-16 LoRA, `modules_to_save` empty | dense, no adapters |
| epochs at the default budget | 1.10 | 6.58 |

The pi0.5 notes single out `state_proj`, `action_in_proj` and `action_out_proj`
— the projections carrying the 6-DOF joint mapping, the part with no pretrained
equivalent — as having been adapted at rank 16 rather than trained. SmolVLA
trains `state_proj` densely (`train_state_proj` defaults true) along with the
whole action expert, while `freeze_vision_encoder` and `train_expert_only` keep
the pretrained VLM intact. That is the recipe SmolVLA was designed around, taken
as-is.

`smolvla_base` is also pretrained largely on community LeRobot datasets recorded
on SO-100/SO-101 arms — this arm, these joint names, this action space. pi0.5's
pretraining mix is broader and further away.

### What is load-bearing in the configuration

- **`--dataset.image_transforms.enable=true`.** lerobot defaults this to false,
  the pi0.5 run took the default, and it scored 0/100. On this task the image
  flow policy measured **3/20 without random-shift augmentation against 20/20
  with it** on otherwise identical runs. This is the single knob most likely to
  decide the result.
- **`n_action_steps` must be passed.** SmolVLA defaults it to the full chunk of
  50, which is 1.67 seconds open-loop at `CONTROL_HZ` 30 — the same trap pi0.5
  has. The launcher pins 10, about a third of a second, near the flow policy's
  eight-tick execution horizon, so the closed-loop reaction rates of the
  policies being ranked are comparable.
- **No `--rename_map`.** SmolVLA's *base* config declares `camera1`/`camera2`/
  `camera3` at 256x256, which invites renaming the dataset's cameras to match.
  Do not: lerobot rebuilds `input_features` from the dataset, the names only
  determine the order image tokens are stacked in, and keeping
  `observation.images.overhead`/`.wrist` is what lets
  `resolve_checkpoint_cameras` match them **by name** rather than by position.
  Renaming is what produced the camera-ordering bug (fixed in `735a621`).
  Note the dataset lists **wrist before overhead**, so that is the stacking
  order; it is self-consistent because inference looks each frame up by key.
- **MEAN_STD, not quantiles.** SmolVLA normalizes state and action with
  mean/std, so the launcher's precondition check looks for those. pi0.5's checks
  `q01`/`q99`. Copying that check across would pass on a dataset that then fails
  on the first batch.
- **`empty_cameras` stays 0.** pi0.5 reserves three image slots and needs one
  padded; SmolVLA's finetune declares exactly the two the dataset has, so
  nothing is missing and nothing needs padding.
- **bf16 comes from the environment, not the config.** `--policy.use_amp` is
  *not* wired into the `Accelerator` lerobot builds — `lerobot_train.py` calls
  `Accelerator()` with no `mixed_precision`, so accelerate falls back to
  `ACCELERATE_MIXED_PRECISION`, default `no`. Without the launcher exporting it
  the run is fp32. Set `MIXED_PRECISION=no` to compare.

### Sizing

The same arithmetic as the pi0.5 section, at SmolVLA's batch size:

| batch | steps | samples | epochs over `as-recorded` |
| ---: | ---: | ---: | ---: |
| 64 | 10,000 | 640,000 | 2.19 |
| 64 | 30,000 | 1,920,000 | 6.58 |
| 128 | 30,000 | 3,840,000 | 13.17 |

`STEPS` defaults to 30,000 because that is also SmolVLA's own
`scheduler_decay_steps`, so the cosine decay lands at the end of the run rather
than being truncated mid-schedule. Change one and consider the other.

### The 512x512 dataset

`two-variant-1000-as-recorded-512x512-lerobot.tar.zst` is the same 1000
episodes and 291,618 frames with both camera streams re-encoded from 960x720 to
512x512. Produce one from any recorded dataset with:

```sh
python py/scripts/convert_dataset_resolution.py \
  --src "$PAP_DATA_ROOT/datasets/as-recorded" \
  --width 512 --height 512 --already-rectified --vcodec h264
```

`--already-rectified` is what makes it applicable to a sim recording: the
frames are already an ideal pinhole render, so there is no lens distortion to
undo and the script only center-crops and resizes.

The point is decode cost. A training step measured **1.12-1.45 s** against
**0.416 s** for the same forward and backward on synthetic batches, so two
thirds of it was h264 random-access decode of two 960x720 streams per sample.
512x512 is 2.6x fewer pixels per frame and takes the archive from 2.4 GB to
1.3 GB.

**It is a square crop, not a rescale, so the policy sees a narrower view.** The
saved 960x720 frame is itself the central 1440x1080 of the 1920x1080 render;
cropping it square keeps the central 1080x1080 and drops the left and right
margins. Nothing needs changing at evaluation time — `eval_policy_sim.py` reads
`image_hw` off the checkpoint and cover-crops the render to match, which lands
on that same central 1080x1080 — but a policy trained on it is **not** a
cheaper reproduction of the 32/100 run. It is a different input.

The alternative, if what you want *is* that reproduction, is **512x384**:
SmolVLA's `resize_imgs_with_padding` turns a 4:3 frame into 512x384 plus
padding anyway, so pre-resizing to it reproduces today's model input pixel for
pixel and buys the same decode saving with no change to the field of view.

Converting the full dataset takes about an hour on two cores, bound by the
h264 encode rather than the decode. That is cheap enough that renting for it is
not worth the transfer and provisioning.

It worked. On the 512x512 dataset `data_s` — lerobot's own measure of how long
a step waits for its batch — is **0.03-0.06 s of a 0.5 s step**, against roughly
0.7 s on the 960x720 dataset. Decode is no longer a constraint, and the run is
GPU-bound for the first time.

Two things follow. Effective *cores* now set the ceiling, not container memory:
the same run measured 0.68 s/step on a 21-core host and **0.50 s/step on a
32-core one**, because the remaining decode has to fit in the gaps. And the
launcher's `REQUIRE_NO_PADDING=1` will refuse a dataset that does not fill
SmolVLA's 512x512 input, which is worth setting — a pod that already has the
960x720 dataset unpacked would otherwise be used silently, since the launcher
finds any dataset under `artifacts/` rather than the one named.

### torch.compile is worth 1.45x

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

### Batch size is not a throughput lever at all, above 32

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

### NVDEC decode works, and is 3x faster -- correcting an earlier claim

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

### Two speed ideas that measured as nothing

Recorded so they are not retried: **casting the frozen tower (or the whole VLM)
to bf16** rather than letting autocast convert it each step is -2.3%
uncompiled and +1.6% compiled, i.e. a wash. Autocast evidently caches its
weight casts within a step, so the per-step re-cast that seemed wasteful is not
happening. Embeddings barely move (cos 0.99993).

And a **fused AdamW** cannot help: the optimizer is 0.5% of a step.

### The third idea works: cache the frozen tower instead of recomputing it

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
lossless. (On CPU there is no bf16 autocast, the tower emits float32, and a
bfloat16 cache costs 0.2% on the loss. The script picks the storage dtype from
the precision it is running at, so it measures the substitution rather than its
own rounding.)

Measured back to back on one host, RTX 5090, batch 64, synthetic batches:

| arm | s/step | samples/s | peak VRAM |
| --- | ---: | ---: | ---: |
| stock, eager | 0.3500 | 182.8 | 11,391 MiB |
| **cached, eager** | **0.0711** | **900.0** | 5,582 MiB |
| tower alone | 0.2243 | 570.6 images/s | 4,890 MiB |

**4.92x on the model step.** More than the tower's own share, and the stage
split says where the rest came from:

| stage | stock | cached |
| --- | ---: | ---: |
| tower | 0.2236 (63.9%) | 0.0000 |
| rest of forward | 0.0537 | 0.0322 |
| backward | 0.0710 | 0.0366 |
| AdamW | 0.0041 | 0.0039 |

The tower accounts for 0.2236 s of the 0.2789 s saved. The remaining 0.055 s is
the rest of the step running about 1.75x faster with the tower's activations out
of the way; peak VRAM halves alongside it, which is the likeliest cause, but this
is measured rather than explained — do not plan around it.

**The cache pays for itself after 0.80 epochs.** Building it is one pass through
the tower at 570.6 images/s, which for 291,618 frames across two cameras is
**17.0 minutes**. Against 182.8 samples/s stock and 900.0 cached, the break-even
is about 234,000 samples — 3,700 steps at batch 64. Every run this project has done is
far past that: a 50,000-step run is 10.97 epochs, **4.86 h stock against 1.27 h
including the cache build, or 3.8x end to end**.

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

Two things it removes for free: training never decodes video, so `data_s`
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

### LoRA is not a speed lever here

Worth stating because it is the obvious thing to reach for, and `PI05.md` makes
it look relevant. It is not, and the stage split above is why: **the backward is
20% of a step and AdamW is 1%**. LoRA can only touch those two, so even reducing
both to zero would be a fifth of the step against the tower's two thirds. It also
cannot help the forward at all, which is where the time is. The SmolVLA run
already trains densely (`use_peft: False`, 22.2% of parameters), and moving to
adapters would trade accuracy for a fraction of a fifth.

### A dud host stays a dud, so blocklist the machine

Two offers rented hours apart, 41357771 and 45944050, both reported `running`
and both refused the SSH key -- because they are the same physical machine,
visible only as the same `public_ipaddr`. The dud rule says destroy and rent
elsewhere, but "elsewhere" has to mean a different *machine*: the marketplace
will happily re-offer the broken one under a new offer id. Record the IP of a
dud and skip offers that resolve to it.

### Where a training step goes, and why quantizing the tower is not worth it

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

### Marketplace hosts vary by 1.68x on identical specs

The same compiled batch-64 configuration measured **0.386 s/step** on one host
and **0.649 s/step** on another, both advertising an RTX 5090 at reliability
0.99 or better, both on the pinned stack. Nearly a factor of two.

So an A/B is only meaningful when both arms run on the *same* host, and an
absolute s/step or projected wall clock is a fact about the host it was
measured on rather than about the configuration. Advertised `inet_down` already
could not be trusted; neither can advertised compute.

### Rent in Europe, and check the driver

Two host properties are not negotiable, and both have cost a rented hour:

- **Driver 580 or newer.** The pinned `torch==2.13.0+cu130` needs it. On an
  older driver `vast_pap_provision.sh` silently falls back to `torch 2.10.0+cu128`,
  and the pinned `torchvision==0.28.0` — built for 2.13 — then dies with
  `RuntimeError: operator torchvision::nms does not exist`. Filter offers on
  `driver_version` before renting.
- **A European host.** The `allyouneed` bucket is in **eu-north-1**. From
  Romania the 2.4 GB dataset pulls at ~100 MB/s, in 23 seconds. A California
  host advertising 1509 Mbps delivered 85 KB/s against the same bucket, which is
  eight hours for the same file.

Advertised `inet_down` is not evidence. Measure it on the rented host before
staging a workload:

```sh
aws s3 cp s3://allyouneed/pick-and-place/datasets/<artifact>.tar.zst /tmp/probe.bin
```

### Scoring the checkpoints

`vast_smolvla_eval.sh` mirrors the pi0.5 one — `smoke_v1` first, then
`canonical_100_v1`, and it syncs results to S3 before you can destroy the pod:

```sh
RUN_NAME=<training run> STEPS="030000 020000" scripts/vast_smolvla_eval.sh
```

It needs no `--base-checkpoint`: SmolVLA trains without adapters, so a
checkpoint is a complete model rather than a 5 MB diff against a 14.5 GB base.

`MANIFEST` picks the suite, and results land under
`outputs/<run>/evaluation/<manifest id>/`. The manifest is part of the path
because it is part of the result: scoring 020000 and 040000 on `heldout_256_v1`
into an unqualified path overwrote those two rungs of the `canonical_100_v1`
ladder, and their per-episode records did not survive it.

**Sync before teardown.** The pi0.5 evaluation artifacts no longer exist because
a pod was destroyed before its results reached S3.
