#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Export a published LeRobot video dataset at the requested image resolution,
# train one or more image-conditioned flow-policy arms, and publish every
# checkpoint and run record. Run vast_pap_provision.sh first.
#
# Example:
#   SOURCE_NAME=randomized-1000-maxretry1-lerobot \
#   ARTIFACT_NAME=randomized-1000-maxretry1-224 \
#   RUN_PREFIX=flow-image-randomized-maxretry1-224-300k-seed0 \
#   RANDOM_SHIFTS='8 0' IMAGE_SIZE=224 UPDATES=300000 \
#     bash /workspace/vast_flow_image_train.sh

set -euo pipefail

: "${SOURCE_NAME:?set SOURCE_NAME to an archive under datasets/ without .tar.zst}"
: "${ARTIFACT_NAME:?set ARTIFACT_NAME to a diffusion-policy-data artifact name}"
: "${RUN_PREFIX:?set RUN_PREFIX to the checkpoint name prefix}"

image_size="${IMAGE_SIZE:-224}"
updates="${UPDATES:-300000}"
seed="${SEED:-0}"
random_shifts="${RANDOM_SHIFTS:-8}"
batch_size="${BATCH_SIZE:-64}"
learning_rate="${LEARNING_RATE:-1e-4}"
min_learning_rate="${MIN_LEARNING_RATE:-1e-6}"
warmup_steps="${WARMUP_STEPS:-2000}"
checkpoint_interval="${CHECKPOINT_INTERVAL:-20000}"
validation_interval="${VALIDATION_INTERVAL:-2000}"
trunk_stages="${TRUNK_STAGES:-3}"

bucket_root="s3://allyouneed/pick-and-place"
workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
datasets="$workspace/datasets"
artifacts="$workspace/artifacts"
outputs="$workspace/outputs"
source_s3="$bucket_root/datasets/$SOURCE_NAME.tar.zst"
artifact_s3="$bucket_root/diffusion-policy-data/$ARTIFACT_NAME.tar.zst"
artifact_root="$artifacts/$ARTIFACT_NAME"

export PATH="/root/.local/bin:$PATH"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$datasets" "$artifacts" "$outputs"

if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
fi
case "${quota:-}" in
  ''|max|-1) cores=$(nproc --all) ;;
  *)         cores=$(( quota / period )) ;;
esac
workers="${WORKERS:-$cores}"

verify_download() {
  local archive="$1" checksum="$2"
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$checksum")")
}

publish_and_verify() {
  local path="$1" destination="$2"
  local expected actual
  expected=$(sha256sum "$path" | awk '{print $1}')
  aws s3 cp "$path" "$destination" --only-show-errors
  actual=$(aws s3 cp "$destination" - --only-show-errors | sha256sum | awk '{print $1}')
  if [ "$expected" != "$actual" ]; then
    echo "checksum mismatch publishing $destination" >&2
    exit 1
  fi
}

cd "$repo"
repository_commit=$(git rev-parse HEAD)
echo "repository=$repository_commit image_size=$image_size updates=$updates seed=$seed"
echo "cores=$cores workers=$workers random_shifts=$random_shifts"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ ! -d "$artifact_root" ]; then
  if aws s3 ls "$artifact_s3" >/dev/null 2>&1; then
    echo "Downloading existing image export $artifact_s3"
    # The archive becomes visible before its publisher finishes the independent
    # read-back and uploads the checksum. Wait for that completion signal so a
    # second arm cannot consume an artifact that has not yet been verified.
    for _ in $(seq 1 60); do
      aws s3 ls "$artifact_s3.sha256" >/dev/null 2>&1 && break
      sleep 30
    done
    if ! aws s3 ls "$artifact_s3.sha256" >/dev/null 2>&1; then
      echo "verified checksum did not appear for $artifact_s3" >&2
      exit 1
    fi
    aws s3 cp "$artifact_s3" "$artifacts/$ARTIFACT_NAME.tar.zst" --only-show-errors
    aws s3 cp "$artifact_s3.sha256" "$artifacts/$ARTIFACT_NAME.tar.zst.sha256" \
      --only-show-errors
    verify_download "$artifacts/$ARTIFACT_NAME.tar.zst" \
      "$artifacts/$ARTIFACT_NAME.tar.zst.sha256"
    tar -x -I zstd -f "$artifacts/$ARTIFACT_NAME.tar.zst" -C "$artifacts"
    rm -f "$artifacts/$ARTIFACT_NAME.tar.zst" "$artifacts/$ARTIFACT_NAME.tar.zst.sha256"
  else
    echo "Creating image export $ARTIFACT_NAME from $SOURCE_NAME"
    aws s3 cp "$source_s3" "$datasets/$SOURCE_NAME.tar.zst" --only-show-errors
    aws s3 cp "$source_s3.sha256" "$datasets/$SOURCE_NAME.tar.zst.sha256" --only-show-errors
    verify_download "$datasets/$SOURCE_NAME.tar.zst" "$datasets/$SOURCE_NAME.tar.zst.sha256"
    mkdir -p "$datasets/$SOURCE_NAME"
    tar -x -I zstd -f "$datasets/$SOURCE_NAME.tar.zst" -C "$datasets/$SOURCE_NAME"
    info_path=$(find "$datasets/$SOURCE_NAME" -name info.json -path '*/meta/*' -print -quit)
    if [ -z "$info_path" ]; then
      echo "no meta/info.json found in $SOURCE_NAME" >&2
      exit 1
    fi
    source_root=$(dirname "$(dirname "$info_path")")
    "$venv/bin/python" py/scripts/export_diffusion_policy_dataset.py \
      --src "$source_root" \
      --output "$artifact_root" \
      --image-size "$image_size" \
      --policy-hz 10 \
      --workers "$workers" \
      --action-encoding absolute \
      2>&1 | tee "$workspace/image-export.log"
    cp "$workspace/image-export.log" "$artifact_root/image-export.log"
    (cd "$artifacts" && tar -c -I 'zstd -T0 -10' \
      -f "$ARTIFACT_NAME.tar.zst" "$ARTIFACT_NAME")
    (cd "$artifacts" && sha256sum "$ARTIFACT_NAME.tar.zst" \
      > "$ARTIFACT_NAME.tar.zst.sha256")
    publish_and_verify "$artifacts/$ARTIFACT_NAME.tar.zst" "$artifact_s3"
    aws s3 cp "$artifacts/$ARTIFACT_NAME.tar.zst.sha256" "$artifact_s3.sha256" \
      --only-show-errors
    rm -f "$artifacts/$ARTIFACT_NAME.tar.zst"
  fi
fi

for shift in $random_shifts; do
  run_name="$RUN_PREFIX-shift$shift"
  output_root="$outputs/$run_name"
  output_s3="$bucket_root/outputs/$run_name"
  mkdir -p "$output_root/job-metadata"
  cp "$artifact_root/export.json" "$artifact_root/normalization.npz" "$output_root/"
  cp "$0" "$output_root/job-metadata/launcher.sh"
  git rev-parse HEAD > "$output_root/job-metadata/repository-commit.txt"
  "$venv/bin/python" - "$output_root/job-metadata/config.json" <<PY
import json, sys
json.dump({
    "source_name": "$SOURCE_NAME",
    "artifact_name": "$ARTIFACT_NAME",
    "run_name": "$run_name",
    "repository_commit": "$repository_commit",
    "image_size": $image_size,
    "trunk_stages": $trunk_stages,
    "updates": $updates,
    "batch_size": $batch_size,
    "learning_rate": $learning_rate,
    "min_learning_rate": $min_learning_rate,
    "warmup_steps": $warmup_steps,
    "checkpoint_interval": $checkpoint_interval,
    "validation_interval": $validation_interval,
    "random_shift": $shift,
    "random_scale_pct": 0.0,
    "photometric_augmentation": False,
    "seed": $seed,
    "amp": True,
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY

  # Keep completed checkpoints and the live log recoverable if the pod fails.
  (
    while sleep 600; do
      aws s3 sync "$output_root" "$output_s3" --only-show-errors || true
    done
  ) &
  sync_pid=$!
  cleanup_sync() { kill "$sync_pid" 2>/dev/null || true; }
  trap cleanup_sync EXIT

  echo "Training $run_name"
  "$venv/bin/python" py/scripts/train_flow_image_policy.py \
    --export "$artifact_root" \
    --output "$output_root" \
    --updates "$updates" \
    --batch-size "$batch_size" \
    --learning-rate "$learning_rate" \
    --min-learning-rate "$min_learning_rate" \
    --warmup-steps "$warmup_steps" \
    --checkpoint-interval "$checkpoint_interval" \
    --validation-interval "$validation_interval" \
    --pretrained-backbone \
    --trunk-stages "$trunk_stages" \
    --random-shift "$shift" \
    --seed "$seed" \
    --wandb-project pick-and-place \
    --wandb-run-name "$run_name" \
    2>&1 | tee "$output_root/train.log"

  cleanup_sync
  trap - EXIT
  (
    cd "$output_root"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  )
  aws s3 sync "$output_root" "$output_s3" --only-show-errors
  publish_and_verify "$output_root/checkpoint.pt" "$output_s3/checkpoint.pt"
  publish_and_verify "$output_root/checkpoint-$(printf '%06d' "$updates").pt" \
    "$output_s3/checkpoint-$(printf '%06d' "$updates").pt"
  aws s3 cp "$output_root/SHA256SUMS" "$output_s3/SHA256SUMS" --only-show-errors
  echo "Published and verified $output_s3"
done

echo "All image-flow arms completed."
