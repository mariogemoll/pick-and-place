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

`vast_pi05_lora_train.sh` finetunes `lerobot/pi05_base` on the recorded LeRobot
dataset, on the same rented 5090 the Diffusion Policy runs use:

```sh
RUN_NAME=<fresh> HF_TOKEN=<token> scripts/vast_pi05_lora_train.sh
```

pi0.5 is a 3.3B-parameter VLA and a full finetune is sized for an 80 GB card, so
a 5090 can only run it with adapters. That is not purely a concession to the
hardware — 1000 episodes of a single prompt is thin evidence for moving 3.3B
parameters, and this task already has an overfitting result on record in
`docs/FLOW_POLICY.md`.

Three things about the configuration are load-bearing:

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
the choice `SIM2REAL.md` reaches for those policies:

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
