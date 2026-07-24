#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Train the blue-cube Diffusion Policy on a rented RTX 5090 using the fast
# pre-training path (pick_and_place.dp_pretrain).
#
# This is the throughput-oriented replacement for the recipe that produced
# dp_blue_cube_1000 (~21 h on a 5080, ~10 h projected on a 5090). The model,
# loss, EMA and LR schedule are unchanged.
#
# Keep BATCH_SIZE at 64 and N_EPOCHS at 1500. Both defaults are load-bearing,
# and both were tried the other way and reverted on 2026-08-03:
#
#   batch 256, lr 4e-4   33/66 against 47/66 at matched epochs, p=0.016
#   500 epochs, batch 64 34/66 against 50/66 for the 1500-epoch run, p=0.002
#
# Each reached a *lower* training loss than the recipe it lost to. The only
# speedup that survived evaluation is the launch-overhead work -- CUDA graphs
# and fused AdamW, provably identical arithmetic -- which is ~4x on its own
# (30.1 to 7.34 s/epoch) and needs no hyperparameter change at all. If you
# change a hyperparameter here, score the checkpoint on scenes before believing
# the run was equivalent.
#
# The launcher installs credentials and VAST_INSTANCE_ID. Teardown is manual on
# purpose: the instance stays up after the final S3 sync so the result can be
# verified independently before the evidence disappears.

set -euo pipefail

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
artifact_name="${ARTIFACT_NAME:-blue-cube-1000-10hz-96x96}"
artifact_prefix="$bucket_root/diffusion-policy-data"
run_name="${RUN_NAME:?set RUN_NAME to a fresh, never-used output name}"
output_prefix="$bucket_root/outputs/$run_name"
batch_size="${BATCH_SIZE:-64}"
learning_rate="${LEARNING_RATE:-1e-4}"
n_epochs="${N_EPOCHS:-1500}"
# Warmup is a fixed number of epochs of data -- NOT a fraction of the run, and
# not a step count. 100 is the value that produced the known-good 50/66 policy.
#
# A previous revision scaled this with n_epochs to hold the baseline's 6.7%
# fraction, reasoning that 100 in a 500-epoch run would be "20% and eat the
# anneal". That is now the leading suspect for two failed runs: at ~500 epochs,
# the checkpoint that kept 100-epoch warmup scored 42-47/66 while both runs
# given 33 scored 31-36/66, at batch 64 and 256 alike. Goyal et al. specify
# warmup in epochs and hold it constant across a 32x batch range; that is the
# convention this follows. Override with WARMUP_STEPS only to test the
# hypothesis, and score the result on scenes before believing it.
warmup_steps="${WARMUP_STEPS:-100}"
if [ "$warmup_steps" -ge "$n_epochs" ]; then
  echo "warmup ($warmup_steps) must be shorter than the run ($n_epochs)" >&2
  exit 1
fi

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/training-job.log"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  cp "$job_log" "$output_root/job-metadata/training-job.log"
  if [ -f "$artifact_root/export.json" ]; then
    cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"
  fi
  synced=false
  for attempt in $(seq 1 10); do
    if aws s3 sync "$output_root" "$output_prefix" --only-show-errors; then
      pending=$(aws s3 sync "$output_root" "$output_prefix" --dryrun --only-show-errors)
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

# Weights & Biases is optional. Without a key on the *pod* (having one on the
# controller does nothing), wandb.init aborts the run before the first epoch --
# an hours-long job lost to a missing credential. Degrade to no logging instead,
# and say so, rather than failing at startup.
wandb_override=()
if ! grep -q api.wandb.ai "${NETRC:-$HOME/.netrc}" 2>/dev/null; then
  echo "No api.wandb.ai entry in $HOME/.netrc on this pod; training without W&B logging."
  wandb_override=(wandb=null)
fi

# Refuse to overwrite or resume: checkpoints from previous runs stay put.
if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Fresh output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

if [ ! -f "$artifact_root/train.npz" ]; then
  aws s3 sync "$artifact_prefix/$artifact_name" "$artifact_root" --only-show-errors
fi

cp "$repo/config/diffusion_policy/pretrain_so101_unet_img_fast.yaml" \
  "$output_root/job-metadata/launcher-config.yaml"
git -C "$repo" rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"

# The fast path is a reformulation of the stock one, so prove that on this
# machine before spending hours on it rather than trusting it after the fact.
"$venv/bin/python" "$repo/py/scripts/check_dp_pretrain_fast.py" \
  --dataset "$artifact_root/train.npz" --max-episodes 30 \
  | tee "$output_root/job-metadata/equivalence-check.txt"

export DPPO_DATA_DIR="$artifact_root"
export DPPO_LOG_DIR="$output_root"
export PYTHONPATH="$repo/third_party/dppo"
cd "$repo/third_party/dppo"

( while sleep 900; do
    aws s3 sync "$output_root" "$output_prefix" --only-show-errors
  done ) &
sync_pid=$!

set +e
"$venv/bin/python" script/run.py \
  --config-path "$repo/config/diffusion_policy" \
  --config-name pretrain_so101_unet_img_fast \
  train.n_epochs="$n_epochs" \
  train.batch_size="$batch_size" \
  train.learning_rate="$learning_rate" \
  train.lr_scheduler.first_cycle_steps="$n_epochs" \
  train.lr_scheduler.warmup_steps="$warmup_steps" \
  ${wandb_override[@]+"${wandb_override[@]}"} \
  "$@" \
  2>&1 | tee "$output_root/console.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

# Record the config hydra actually resolved, not the file on disk. The copy of
# the shipped YAML above still says batch_size 64 whatever was passed on the
# command line, which is exactly the kind of metadata that misleads a reader
# six months later.
run_dir=$(dirname "$(find "$output_root/pretrain" -name throughput.json -print -quit)")
if [ -n "$run_dir" ] && [ -f "$run_dir/.hydra/config.yaml" ]; then
  cp "$run_dir/.hydra/config.yaml" "$output_root/job-metadata/effective-config.yaml"
  cp "$run_dir/.hydra/overrides.yaml" "$output_root/job-metadata/effective-overrides.yaml"
fi

checkpoint=$(find "$output_root/pretrain" -type f -name "state_$n_epochs.pt" -print -quit)
if [ -z "$checkpoint" ]; then
  echo "Training returned success without the required epoch-$n_epochs checkpoint." >&2
  exit 1
fi
sha256sum "$checkpoint" | tee "$output_root/job-metadata/state_$n_epochs.pt.sha256"
echo "Training completed with the required epoch-$n_epochs checkpoint."
