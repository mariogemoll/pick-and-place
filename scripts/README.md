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

## Randomized image-flow evaluation

`vast_flow_image_eval.sh` fetches the completed image-flow arms named in
`SHIFTS`, verifies each training `SHA256SUMS` manifest, scores them on the
frozen `randomized_selection_200_v1` suite, and publishes a checksummed
evaluation bundle. It fixes the operating point at Euler-10, 8 executed
actions, and flow seed 0:

```sh
RUN_PREFIX=flow-image-randomized-maxretry1-224-300k-seed0 \
  scripts/vast_flow_image_eval.sh
```

Naming exactly two arms also compares every paired scenario. A single arm is
scored on its own, to be read against a bundle another run already published —
which is the only option when an experiment varies the training data rather
than a training knob, and so has one arm per dataset:

```sh
RUN_PREFIX=flow-image-new-scripted-standardcam-dr-224-300k-seed0 \
  SHIFTS=8 ARTIFACT_NAME=new-scripted-standardcam-dr-1000-224 \
  scripts/vast_flow_image_eval.sh
```

Each arm is scored across `SHARDS` concurrent workers (default: one per
container core, less one) and merged, because scoring is MuJoCo rendering at a
few percent of the GPU and a serial arm leaves an evaluation pod idle.

Run it only after `vast_pap_provision.sh` on a verified evaluation pod. Every
training output prefix must already contain a final `SHA256SUMS` file; the
script deliberately refuses partial uploads. Results land under
`s3://allyouneed/pick-and-place/evaluations/randomized_selection_200_v1/`.

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

> The first run of this scored 0/100 on `canonical_100_v1`; the launcher now
> enables `image_transforms` by default, which is believed to have been the
> cause. Weigh a retry against what already works on this task for less
> compute.

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
**AprilTag** half of the two-variant pair, not the blue one the Diffusion
Policy and flow policy were trained on. That is deliberate: the physical cube
carries AprilTags, so only the tagged variant has a path to the real arm.
Train `two-variant-1000-blue-cube-lerobot` instead for a clean comparison
against the blue-cube policies — the two variants share states, actions and
phase spans bit for bit and differ only at the cube's pixels.

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

> The first run of this scored 32/100 on `canonical_100_v1`, so a VLA does
> learn this task — but DPPO and the flow policy both win for less compute.
> Do not run more VLA experiments in simulation for this task without a new
> reason; the live argument is sim2real, which needs the real arm to score.

`vast_smolvla_train.sh` is the cheap retry of the question the pi0.5 run left
open, on the same rented 5090:

```sh
RUN_NAME=<fresh> scripts/vast_smolvla_train.sh
```

No `HF_TOKEN`. SmolVLA tokenizes through `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
which is public, so the gated-checkpoint dance pi0.5 needs for
`google/paligemma-3b-pt-224` is simply absent.

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
than being truncated mid-schedule. Change one and consider the other. Budget
2 to 2.5 million samples for this task — count samples, not steps, since a
step count with no batch size attached says nothing.

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

The point is decode cost: two 960x720 h264 streams per sample dominate a
training step. 512x512 is 2.6x fewer pixels per frame and takes the archive
from 2.4 GB to 1.3 GB.

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

On the 512x512 dataset the run is GPU-bound rather than decode-bound, and
`REQUIRE_NO_PADDING=1` is worth setting: it refuses a dataset that does not
fill SmolVLA's 512x512 input, which catches a pod that still has the 960x720
dataset unpacked, since the launcher finds any dataset under `artifacts/`
rather than the one named.

Caching the frozen vision tower's output (`PREFIX_CACHE=1`) and compiling
(`COMPILE_MODEL=true`) are both large, independent speedups for this launcher;
set both for anything longer than a smoke run.

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
