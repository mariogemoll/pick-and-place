#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# LoRA-finetune pi0.5 on a rented RTX 5090 from the recorded LeRobot dataset.
#
# pi0.5 is a 3.3B-parameter VLA; a full finetune is sized for an 80 GB card and
# does not fit a 5090. LoRA does, and is the better match for this task anyway:
# 1000 episodes of a single prompt is not enough signal to move 3.3B parameters
# without overfitting -- see the 100,000-update result in docs/FLOW_POLICY.md.
#
# The adapter targets come from the policy, not from here. pi0.5 ships
# _get_default_peft_targets() (modeling_pi05.py), which adapts the action
# expert's q/v projections *and* state_proj, action_in_proj, action_out_proj
# and the action-time MLPs. Those projections carry the 6-DOF joint mapping,
# the part with no pretrained equivalent, so they must be trainable.
#
# They are adapted with LoRA, not fully fine-tuned: modules_to_save is empty in
# the emitted adapter_config.json, so every trainable weight is a rank-16
# adapter and nothing is trained densely. Leave --peft.target_modules unset to
# get this set; setting it silently replaces the whole regex.
#
# Launch:
#   scp scripts/vast_pi05_lora_train.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'RUN_NAME=<fresh> bash /workspace/vast_pi05_lora_train.sh'
#
# vast_pap_provision.sh must have run first; this script does not create that
# state. Teardown is manual, matching vast_diffusion_policy_train_fast.sh: the
# instance stays up after the final S3 sync so the result can be verified before
# the evidence disappears.

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

# Steps, not epochs: lerobot-train counts steps, so the epoch coverage has to be
# worked out by hand. This dataset is 291,618 frames, so batch 16 x 20,000 steps
# is 1.1 passes over it -- 10,000 steps would not have completed even one.
#
# The instinct to cut below the LIBERO recipe's 30,000 because this is a single
# task is right about LIBERO and wrong about the arithmetic: LIBERO runs batch
# 64, four times this one. Judge the budget in samples seen, not in steps.
steps="${STEPS:-20000}"
# 16 is the safe starting point on a 32 GB card at 224x224 with three image
# slots. The smoke stage reports peak VRAM; if there is headroom, 32 doubles
# the epoch coverage for the same step count.
batch_size="${BATCH_SIZE:-16}"
learning_rate="${LEARNING_RATE:-1e-4}"
lora_rank="${LORA_RANK:-16}"
save_freq="${SAVE_FREQ:-2000}"
seed="${SEED:-1000}"

# Passed explicitly because --policy.pretrained_path loads *weights only*: every
# stored config value falls back to the class default, so n_action_steps would
# silently become 50 (the whole chunk) rather than a reactive closed-loop
# horizon. The flow policy runs act_steps 8 at 10 Hz; 10 of these 30 Hz steps is
# the same third of a second.
n_action_steps="${N_ACTION_STEPS:-10}"
# pi05_base reserves three image slots; this rig has overhead + wrist, so one
# slot is padded empty. Same reason the LIBERO recipe passes empty_cameras=1.
empty_cameras="${EMPTY_CAMERAS:-1}"

# lerobot/pi05_base moves independently of the pinned lerobot. Its commit
# 7de663972b (2026-06-03, "Add relative action processor steps") added a
# relative_actions_processor step to policy_preprocessor.json, and lerobot
# 0.5.1's registry has no such step -- loading HEAD dies in
# make_pre_post_processors with "Processor step not found in registry".
#
# a538eb2732 is the commit before it and carries the six steps 0.5.1 knows.
# from_pretrained accepts a revision but no config field exposes one, so the
# revision is materialized to a directory and passed as a path. Unpinning
# lerobot instead is not the cheap way out: 0.5.1 pins transformers==5.3.0.
checkpoint_revision="${CHECKPOINT_REVISION:-a538eb2732}"

# Smoke stage: a handful of real steps into a throwaway directory before the
# paid run. It proves the gated tokenizer resolves, the camera keys match, the
# quantile stats satisfy the normalizer and the batch fits in VRAM. Every one of
# those fails in the first few steps or not at all, so this converts an
# overnight failure into a two-minute one.
smoke_steps="${SMOKE_STEPS:-5}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
output_root="$workspace/outputs/$run_name"
checkpoint_dir="$workspace/pi05_base_pinned"
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

# Same contract as vast_diffusion_policy_train_fast.sh: a W&B key on the *pod*
# is what counts, and a missing one is refused up front rather than shrugged at.
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

# pi0.5 tokenizes through google/paligemma-3b-pt-224, which is gated. Without a
# token that has accepted its licence the run dies at model construction --
# after the dataset download, which is the expensive part. Check first.
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "No HF_TOKEN and no ~/.cache/huggingface/token on this pod." >&2
  echo "pi0.5 needs the gated google/paligemma-3b-pt-224 tokenizer. Export" >&2
  echo "HF_TOKEN before launching. Refusing to start." >&2
  exit 1
fi
"$venv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("google/paligemma-3b-pt-224", "config.json")
print("Gated PaliGemma tokenizer is reachable from this pod.")
PY

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

# pi0.5 normalizes state and action with quantiles, so q01/q99 must be present.
# Checking here beats the alternative, which is a ValueError on the first batch
# after the model has already been built and the dataset scanned.
"$venv/bin/python" - "$artifact_root" <<'PY'
import json, sys
from pathlib import Path
stats = json.loads((Path(sys.argv[1]) / "meta" / "stats.json").read_text())
missing = [
    key
    for key in ("observation.state", "action")
    if not {"q01", "q99"} <= set(stats.get(key, {}))
]
if missing:
    raise SystemExit(
        f"stats.json lacks q01/q99 for {missing}; recompute with lerobot-edit-dataset "
        "or pass MEAN_STD normalization instead."
    )
print("Quantile stats present for state and action.")
PY

"$venv/bin/python" - "$checkpoint_revision" "$checkpoint_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
revision, target = sys.argv[1], sys.argv[2]
print("pinned checkpoint at", snapshot_download("lerobot/pi05_base", revision=revision, local_dir=target))
PY
echo "$checkpoint_revision" > "$output_root/job-metadata/checkpoint-revision.txt"

git -C "$repo" rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"
cp "$0" "$output_root/job-metadata/launcher.sh"

train_args=(
  --dataset.repo_id="$artifact_name"
  --dataset.root="$artifact_root"
  --policy.type=pi05
  --policy.pretrained_path="$checkpoint_dir"
  --policy.n_action_steps="$n_action_steps"
  --policy.empty_cameras="$empty_cameras"
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.device=cuda
  --policy.push_to_hub=false
  # Leave --peft.target_modules unset: pi0.5's own defaults are the right ones.
  --peft.method_type=LORA
  --peft.r="$lora_rank"
  --batch_size="$batch_size"
  --num_workers=8
  --seed="$seed"
)

echo "=== Smoke stage: $smoke_steps steps ==="
rm -rf "$workspace/pi05-smoke"
"$venv/bin/lerobot-train" \
  "${train_args[@]}" \
  --wandb.enable=false \
  --steps="$smoke_steps" \
  --save_freq="$smoke_steps" \
  --output_dir="$workspace/pi05-smoke" \
  2>&1 | tee "$output_root/job-metadata/smoke.log"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
  | tee "$output_root/job-metadata/smoke-vram.txt"
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
du -sh "$checkpoint" | tee "$output_root/job-metadata/adapter-size.txt"
echo "Training completed with the required step-$steps checkpoint."
