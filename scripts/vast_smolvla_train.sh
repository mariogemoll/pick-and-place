#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Finetune SmolVLA on a rented RTX 5090 from the recorded LeRobot dataset.
#
# This is the cheap retry of the VLA question the pi0.5 run left open. That run
# scored 0/100 on canonical_100_v1 with two suspected causes, both accidents of
# a default: no image augmentation, and 1.10 epochs. Both are fixed here, and
# SmolVLA is a better-shaped model for the question besides:
#
#   - 450M parameters against pi0.5's 4.14B, of which 99,880,992 (22.2%) train
#     against pi0.5's 1,287,168 LoRA weights (0.031%). 77x more trainable
#     parameters on a model 9x smaller, and no adapters anywhere.
#   - state_proj trains densely (train_state_proj defaults true). The pi0.5
#     notes single out the state and action projections -- "the part with no
#     pretrained equivalent" -- as having been only rank-16 adapted.
#   - smolvla_base was pretrained on community LeRobot datasets that are
#     overwhelmingly SO-100/SO-101, i.e. this arm, these joint names, this
#     6-DOF action space. pi0.5's mix is broader and further away.
#   - No gated tokenizer. SmolVLM2-500M is public, so the HF_TOKEN dance the
#     pi0.5 launcher needs for google/paligemma-3b-pt-224 is gone.
#
# Launch:
#   scp scripts/vast_smolvla_train.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'RUN_NAME=<fresh> bash /workspace/vast_smolvla_train.sh'
#
# vast_pap_provision.sh must have run first; this script does not create that
# state. Teardown is manual, matching the other launchers: the instance stays up
# after the final S3 sync so the result can be verified before the evidence
# disappears.

set -euo pipefail

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
artifact_name="${ARTIFACT_NAME:-two-variant-1000-as-recorded-lerobot}"
artifact_prefix="$bucket_root/datasets"
run_name="${RUN_NAME:?set RUN_NAME to a fresh, never-used output name}"
output_prefix="$bucket_root/outputs/$run_name"

# Steps, not epochs: lerobot-train counts steps, so epoch coverage is arithmetic
# the launcher has to do. This dataset is 291,618 frames, so batch 64 x 30,000
# steps is 1,920,000 samples = 6.58 passes. The pi0.5 run managed 1.10, against
# from-scratch image policies on this task that needed 200-300; 6.58 is still
# not many, but a pretrained model is the whole reason to expect it to be
# enough, and this is the run that tests that.
steps="${STEPS:-30000}"
# The cosine decay has to be told how long the run is, because the scheduler
# only ever rescales itself *downwards*: CosineDecayWithWarmupSchedulerConfig
# .build() rescales warmup and decay to fit when num_training_steps is *less*
# than num_decay_steps, and does nothing when it is more. SmolVLAConfig defaults
# scheduler_decay_steps to 30,000, so at STEPS=30000 the two agree by accident
# and any longer run silently spends its tail at the 2.5e-6 floor -- at 50,000
# steps that is the last 40% of the run at a fortieth of the peak rate.
#
# Tying it to the step count keeps the decay landing exactly at the end of
# whatever run is asked for. Set SCHEDULER_DECAY_STEPS to decouple them.
scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-$steps}"
# 64 fits a 32 GB card comfortably: 450M params at bf16 is under a gigabyte, the
# optimizer carries state for only the 100M trainable ones, and the sequence is
# short (two images at 64 tokens each, 48 language tokens, 50 action tokens).
# The smoke stage reports peak VRAM; if there is headroom, 128 doubles epoch
# coverage for the same step count.
batch_size="${BATCH_SIZE:-64}"
# Everything else about the optimizer is left at SmolVLA's own preset -- AdamW
# at a 1e-4 peak, 1,000 warmup steps, cosine decay to 2.5e-6 -- which
# use_policy_training_preset (default true) takes from the policy config, so
# only the decay length above needs restating.
save_freq="${SAVE_FREQ:-5000}"
seed="${SEED:-1000}"
# The dataset decodes two h264 streams per sample, so at batch 64 a step needs
# 128 random-access frame decodes, and on the 720x960 dataset that -- not the
# GPU -- was the bottleneck: measured on a 5090, a step took 1.23s while the
# same forward and backward on synthetic batches took 0.416s, so two thirds of
# every step was spent waiting for frames. lerobot 0.5.1 cannot move this to
# NVDEC -- decode_video_frames_torchcodec takes no device and its docstring
# warns that CUDA decode in a dataloader worker raises CUDA initialization
# errors. Feeding a 512x512 dataset instead is the lever that does work: 2.6x
# fewer pixels per frame should put decode under the 0.416s compute floor and
# let the prefetch hide it entirely.
#
# 16 is not a throughput choice, it is a ceiling. `nproc` reports the *host's*
# cores (256 on the box this was measured on) while the container's RAM is
# capped far lower, and 48 workers prefetching 720x960 frames gets the run
# OOM-killed. Raise NUM_WORKERS only with the container's memory limit in hand,
# not its core count.
num_workers="${NUM_WORKERS:-16}"

# Passed explicitly because SmolVLA defaults n_action_steps to the full chunk of
# 50, which is open-loop for 1.67 seconds at CONTROL_HZ 30 -- the same trap the
# pi0.5 launcher documents. Ten ticks is a third of a second, near the flow
# policy's eight-tick execution horizon, so the closed-loop reaction rates are
# comparable across the policies being ranked.
n_action_steps="${N_ACTION_STEPS:-10}"

# SmolVLA feeds every camera through resize_with_pad() to
# resize_imgs_with_padding, which is (512, 512): the frame is scaled down until
# it fits inside a 512x512 box with its aspect ratio intact, then padded on the
# left and top with zeros. A 960x720 frame therefore reaches SigLIP as 512x384
# real pixels above 128 rows of padding -- a quarter of the model's input is
# constant black, and the decoder paid for 2.6x more pixels than survive.
#
# A square dataset makes that step an identity: at 512x512 the ratio is 1, so
# nothing is rescaled and nothing is padded. Set REQUIRE_NO_PADDING=1 to assert
# that rather than trust it; the precondition below computes what the model will
# actually receive and refuses a dataset that would be padded.
require_no_padding="${REQUIRE_NO_PADDING:-0}"

# lerobot does not wire --policy.use_amp into the Accelerator it builds, so the
# run is fp32 unless this env var says otherwise: lerobot_train.py constructs
# Accelerator() with no mixed_precision argument, and accelerate then falls back
# to parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no").
#
# bf16 needs no gradient scaler, and SmolVLA already casts its transformer
# output back to float32 before the flow-matching loss, so the loss itself is
# computed in full precision either way. Set MIXED_PRECISION=no to compare.
export ACCELERATE_MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# smolvla_base moves independently of the pinned lerobot, and the pi0.5 run lost
# a day to exactly that: lerobot/pi05_base HEAD had gained a processor step that
# lerobot 0.5.1's registry did not know. This revision (2026-01-22) is verified
# to load under 0.5.1. from_pretrained accepts a revision but no config field
# exposes one, so it is materialized to a directory and passed as a path.
checkpoint_revision="${CHECKPOINT_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"

# Smoke stage: a handful of real steps into a throwaway directory before the
# paid run. It proves the camera keys match, the normalization stats satisfy the
# normalizer, and the batch fits in VRAM. Every one of those fails in the first
# few steps or not at all, so this converts an overnight failure into a
# two-minute one. It also measures s/step, which is the only honest input to the
# cost estimate.
smoke_steps="${SMOKE_STEPS:-20}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
output_root="$workspace/outputs/$run_name"
checkpoint_dir="$workspace/smolvla_base_pinned"
job_log="$workspace/training-job.log"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  cp "$job_log" "$output_root/job-metadata/training-job.log"
  synced=false
  for attempt in $(seq 1 10); do
    # --no-follow-symlinks, and warnings filtered out of the verification.
    # wandb leaves train/wandb/debug-internal.log as a dangling symlink; aws s3
    # sync then warns "File does not exist" *and* exits non-zero, while the
    # dry-run prints the same warning on stdout. Without both guards a complete
    # upload reads as a failure whose retry can never succeed, and a finished
    # run reports "Final S3 sync could not be verified".
    if aws s3 sync "$output_root" "$output_prefix" --no-follow-symlinks --only-show-errors; then
      pending=$(aws s3 sync "$output_root" "$output_prefix" --no-follow-symlinks \
        --dryrun --only-show-errors | grep -v '^warning:' || true)
      if [ -z "$pending" ]; then
        synced=true
        break
      fi
    fi
    echo "Final S3 sync attempt $attempt failed; retrying in 60 seconds."
    sleep 60
  done
  if [ "$synced" != true ]; then
    echo "Final S3 sync could not be verified; leaving instance running for recovery."
    exit 1
  fi
  echo "Final S3 sync verified; leaving instance running for independent verification."
  exit "$status"
}
trap finalize EXIT

nvidia-smi

# Same contract as the other launchers: a W&B key on the *pod* is what counts,
# and a missing one is refused up front rather than shrugged at.
if [ "${WANDB:-on}" = "off" ]; then
  echo "WANDB=off: training without W&B logging, by request."
  wandb_args=(--wandb.enable=false)
elif grep -q api.wandb.ai "${NETRC:-$HOME/.netrc}" 2>/dev/null; then
  wandb_args=(--wandb.enable=true --wandb.project=pick-and-place)
  echo "W&B credential found on this pod."
else
  echo "No api.wandb.ai entry in ${NETRC:-$HOME/.netrc} on this pod." >&2
  echo "Copy your ~/.netrc to the pod (vast_pap_provision.sh stages it), or set" >&2
  echo "WANDB=off to run without logging. Refusing to start blind." >&2
  exit 1
fi

# Refuse to overwrite or resume: checkpoints from previous runs stay put.
if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Fresh output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

# Discover before downloading. The archive's internal directory name is not the
# artifact name -- two-variant-1000-as-recorded-lerobot.tar.zst unpacks to
# `as-recorded` -- so a guard keyed on "$artifact_root/meta/info.json" never
# matches an already-unpacked dataset and re-fetches 2.4 GB on every run.
find_dataset() {
  find "$workspace/artifacts" -maxdepth 3 -path '*/meta/info.json' -print -quit 2>/dev/null
}

if [ -z "$(find_dataset)" ]; then
  archive="$artifact_prefix/$artifact_name.tar.zst"
  staging="$workspace/artifacts"
  aws s3 cp "$archive" "$staging/$artifact_name.tar.zst" --only-show-errors
  aws s3 cp "$archive.sha256" "$staging/$artifact_name.tar.zst.sha256" --only-show-errors
  # Verify before unpacking: a truncated download otherwise surfaces hours later
  # as a corrupt frame rather than as a failed transfer.
  (cd "$staging" && sha256sum -c "$artifact_name.tar.zst.sha256")
  tar -x -I zstd -f "$staging/$artifact_name.tar.zst" -C "$staging"
  rm -f "$staging/$artifact_name.tar.zst" "$staging/$artifact_name.tar.zst.sha256"
fi

found=$(find_dataset)
if [ -z "$found" ]; then
  echo "No LeRobot dataset (meta/info.json) found under $workspace/artifacts." >&2
  exit 1
fi
artifact_root=$(dirname "$(dirname "$found")")
echo "Using dataset at $artifact_root"

# SmolVLA normalizes state and action with MEAN_STD, not the quantiles pi0.5
# wants, so mean/std are what must be present. Checking here beats the
# alternative, which is a KeyError on the first batch after the model has been
# built and the dataset scanned. The camera count is checked at the same time:
# SmolVLA reserves three image slots and pads missing ones only up to
# empty_cameras, which stays 0 because this rig fills exactly two and the
# finetune's input_features are rebuilt from the dataset.
"$venv/bin/python" - "$artifact_root" "$require_no_padding" <<'PY'
import json, sys
from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

stats = json.loads((Path(sys.argv[1]) / "meta" / "stats.json").read_text())
missing = [
    key
    for key in ("observation.state", "action")
    if not {"mean", "std"} <= set(stats.get(key, {}))
]
if missing:
    raise SystemExit(f"stats.json lacks mean/std for {missing}; recompute with lerobot-edit-dataset.")

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
cameras = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
if len(cameras) != 2:
    raise SystemExit(f"expected exactly two camera streams, found {cameras}")
# Order matters and is not alphabetical: lerobot builds input_features in
# feature order, SmolVLA stacks its image tokens in that order, and inference
# has to agree. resolve_checkpoint_cameras() matches by name rather than
# position, so it does agree -- this only records what the order actually was.
print(f"Mean/std stats present. Cameras in dataset order: {cameras}")
print(f"Frames: {info['total_frames']}, episodes: {info['total_episodes']}, fps: {info['fps']}")

# Replay resize_with_pad()'s arithmetic on the recorded frame shape, so the
# geometry the model will actually see is a printed fact rather than an
# assumption. The function scales the frame by the larger of the two axis
# ratios, which fits it inside the box with its aspect ratio intact, then pads
# whatever is left over on the left and top.
box_height, box_width = SmolVLAConfig().resize_imgs_with_padding
padded = False
for camera in cameras:
    height, width = info["features"][camera]["shape"][:2]
    ratio = max(width / box_width, height / box_height)
    fitted_height, fitted_width = int(height / ratio), int(width / ratio)
    pad_rows = max(0, box_height - fitted_height)
    pad_cols = max(0, box_width - fitted_width)
    fraction = 1 - (fitted_height * fitted_width) / (box_height * box_width)
    print(
        f"{camera}: {width}x{height} -> {fitted_width}x{fitted_height} "
        f"in a {box_width}x{box_height} box, padding {pad_cols} cols and {pad_rows} rows "
        f"({fraction:.1%} of the model input)"
    )
    padded = padded or bool(pad_rows or pad_cols)

if sys.argv[2] == "1" and padded:
    raise SystemExit(
        f"REQUIRE_NO_PADDING is set, but this dataset does not fill SmolVLA's "
        f"{box_width}x{box_height} input. Convert it with convert_dataset_resolution.py "
        f"--width {box_width} --height {box_height}."
    )
PY

"$venv/bin/python" - "$checkpoint_revision" "$checkpoint_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
revision, target = sys.argv[1], sys.argv[2]
print("pinned checkpoint at", snapshot_download("lerobot/smolvla_base", revision=revision, local_dir=target))
PY
echo "$checkpoint_revision" > "$output_root/job-metadata/checkpoint-revision.txt"

git -C "$repo" rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"
cp "$0" "$output_root/job-metadata/launcher.sh"

train_args=(
  --dataset.repo_id="$artifact_name"
  --dataset.root="$artifact_root"
  # lerobot defaults image_transforms to enable=false, and the pi0.5 run took
  # that default and scored 0/100. On this task, random-shift augmentation is
  # the single most load-bearing knob for an image-conditioned policy: the image
  # flow policy measured 3/20 without it against 20/20 with it, on otherwise
  # identical runs. This is the one change most likely to decide the run.
  --dataset.image_transforms.enable=true
  --policy.type=smolvla
  --policy.pretrained_path="$checkpoint_dir"
  --policy.n_action_steps="$n_action_steps"
  --policy.scheduler_decay_steps="$scheduler_decay_steps"
  --policy.device=cuda
  --policy.push_to_hub=false
  --batch_size="$batch_size"
  --num_workers="$num_workers"
  --seed="$seed"
)

echo "=== Smoke stage: $smoke_steps steps, batch $batch_size, $num_workers workers, ${ACCELERATE_MIXED_PRECISION} ==="
rm -rf "$workspace/smolvla-smoke"
# Sample VRAM *during* the stage and keep the peak. Reading it afterwards
# reports whatever is left once the process has exited, which is 0 MiB.
( while sleep 2; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  done ) > "$workspace/vram-samples.txt" &
vram_pid=$!
smoke_start=$(date +%s)
"$venv/bin/lerobot-train" \
  "${train_args[@]}" \
  --wandb.enable=false \
  --steps="$smoke_steps" \
  --save_freq="$smoke_steps" \
  --output_dir="$workspace/smolvla-smoke" \
  2>&1 | tee "$output_root/job-metadata/smoke.log"
smoke_elapsed=$(( $(date +%s) - smoke_start ))
kill "$vram_pid" 2>/dev/null || true
wait "$vram_pid" 2>/dev/null || true
sort -n "$workspace/vram-samples.txt" | tail -1 \
  | awk -v total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" \
    '{print "peak VRAM during smoke: "$1" MiB of "total" MiB"}' \
  | tee "$output_root/job-metadata/smoke-vram.txt"

# The smoke stage pays the model build and the dataset scan once, so its wall
# clock overstates s/step badly at 20 steps. Take the rate from the training
# log's own step timings instead, and fall back to the wall clock only if the
# log format has moved.
"$venv/bin/python" - "$output_root/job-metadata/smoke.log" "$smoke_elapsed" "$steps" "$batch_size" <<'PY' \
  | tee "$output_root/job-metadata/throughput-estimate.txt"
import re, sys

log, elapsed, steps, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
text = open(log, errors="replace").read()
# lerobot drives the loop with tqdm, so the rate arrives as "N.NNs/step" in the
# progress bar and there is no per-step field to average. tqdm's figure is a
# running mean over the whole loop, so the last one is the best available and
# still carries the first step's cudnn autotuning and worker spin-up amortized
# over smoke_steps -- an overestimate that shrinks as smoke_steps grows.
rates = [float(m) for m in re.findall(r"([0-9.]+)s/step", text)]
if rates:
    per_step = rates[-1]
    source = f"tqdm running mean over {steps} steps (warmup included; an overestimate)"
else:
    per_step = elapsed / max(steps, 1)
    source = f"wall clock over {elapsed}s (includes model build; a large overestimate)"
hours = per_step * steps / 3600
print(f"per-step estimate: {per_step:.3f}s, from {source}")
print(f"projected: {steps} steps ~ {hours:.1f}h, {steps * batch:,} samples")
print(f"at $0.34/hr that is about ${hours * 0.34:.2f}")
PY
echo "=== Smoke stage passed ==="

( while sleep 900; do
    aws s3 sync "$output_root" "$output_prefix" --no-follow-symlinks --only-show-errors
  done ) &
sync_pid=$!

set +e
"$venv/bin/lerobot-train" \
  "${train_args[@]}" \
  "${wandb_args[@]}" \
  --steps="$steps" \
  --save_freq="$save_freq" \
  --output_dir="$output_root/train" \
  --job_name="$run_name" \
  "$@" \
  2>&1 | tee "$output_root/console.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

checkpoint="$output_root/train/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
if [ ! -d "$checkpoint" ]; then
  echo "Training returned success without the required step-$steps checkpoint." >&2
  echo "Present checkpoints:" >&2
  ls "$output_root/train/checkpoints" >&2 || true
  exit 1
fi
du -sh "$checkpoint" | tee "$output_root/job-metadata/checkpoint-size.txt"
echo "Training completed with the required step-$steps checkpoint."
