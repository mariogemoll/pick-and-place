#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Evaluate a pair of final image-flow arms on one frozen manifest and publish
# a self-verifying result bundle. Run after vast_pap_provision.sh on a fresh
# GPU pod or on a still-live training pod.
#
#   RUN_PREFIX=flow-image-randomized-maxretry1-224-300k-seed0 \
#     scripts/vast_flow_image_paired_eval.sh
#
# The final checkpoints must already have a SHA256SUMS manifest in S3. This is
# intentional: a checkpoint that merely exists is not evidence that the pod
# finished uploading it, and a paired result is not worth publishing from a
# partial arm.

set -euo pipefail

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_prefix="${RUN_PREFIX:?set RUN_PREFIX to the common training output prefix}"
shifts="${SHIFTS:-8 0}"
manifest="${MANIFEST:-randomized_selection_200_v1/manifest.json}"
eval_name="${EVAL_NAME:-${run_prefix}-paired-$(date -u +%Y%m%d)}"
parallel_arms="${PARALLEL_ARMS:-1}"
read -r -a shift_list <<< "$shifts"
if [ "${#shift_list[@]}" -ne 2 ]; then
  echo "SHIFTS must name exactly two arms; got: $shifts" >&2
  exit 2
fi

workspace="/workspace"
repo="${REPO:-$workspace/pick-and-place}"
venv="$workspace/venvs/pick-and-place"
bucket_root="s3://allyouneed/pick-and-place"
artifact_name="${ARTIFACT_NAME:-randomized-1000-maxretry1-224}"
final_checkpoint="${FINAL_CHECKPOINT:-checkpoint-300000.pt}"
artifact_s3="$bucket_root/diffusion-policy-data/$artifact_name.tar.zst"
artifact_root="$workspace/artifacts/$artifact_name"
input_root="${INPUT_ROOT:-$workspace/evaluation-inputs}"
output_root="$workspace/evaluations/$eval_name"
output_s3="$bucket_root/evaluations/${manifest%%/manifest.json}/$eval_name"

cd "$repo"

if [ -e "$output_root" ]; then
  echo "Refusing to overwrite evaluation output: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root" "$workspace/artifacts" "$input_root"

fetch_final_arm() {
  local shift="$1" arm="$run_prefix-shift$shift"
  local source="$bucket_root/outputs/$arm"
  # Never reuse the training pod's output directory here. Its live
  # checkpoint.pt can be newer than the completed S3 manifest, and aws s3 sync
  # intentionally leaves such a local file untouched. A dedicated input root
  # makes this verification a check of the bytes that will actually be scored.
  local destination="$input_root/$arm"

  echo "Fetching and verifying $arm/$final_checkpoint"
  mkdir -p "$destination"
  aws s3 cp "$source/SHA256SUMS" "$destination/SHA256SUMS" --only-show-errors
  # Evaluation only needs the immutable final checkpoint. Downloading every
  # intermediate checkpoint turns a 31 MB input into a ~500 MB transfer and
  # delays the result without adding evidence: the training completion watcher
  # has already verified the full remote manifest before this script runs.
  aws s3 cp "$source/$final_checkpoint" "$destination/$final_checkpoint" --only-show-errors
  expected=$(awk -v file="./$final_checkpoint" '$2 == file {print $1}' "$destination/SHA256SUMS")
  actual=$(sha256sum "$destination/$final_checkpoint" | awk '{print $1}')
  [ -n "$expected" ] && [ "$actual" = "$expected" ] || {
    echo "$arm/$final_checkpoint does not match SHA256SUMS" >&2
    return 1
  }
}

if [ ! -d "$artifact_root" ]; then
  archive="$workspace/artifacts/$artifact_name.tar.zst"
  checksum="$archive.sha256"
  echo "Fetching image export $artifact_name"
  aws s3 cp "$artifact_s3" "$archive" --only-show-errors
  aws s3 cp "$artifact_s3.sha256" "$checksum" --only-show-errors
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$checksum")")
  tar -x -I zstd -f "$archive" -C "$(dirname "$artifact_root")"
  rm -f "$archive" "$checksum"
fi
[ -f "$artifact_root/export.json" ] && [ -f "$artifact_root/normalization.npz" ] || {
  echo "image export is incomplete: $artifact_root" >&2
  exit 1
}

for shift in "${shift_list[@]}"; do
  fetch_final_arm "$shift"
done

if [ ! -f "$repo/assets/apriltags/textures/tagStandard41h12_00014_60x60mm_tag40mm.png" ]; then
  "$venv/bin/python" py/scripts/render_apriltag_textures.py --all-defaults
fi

printf '%s\n' \
  "run_prefix=$run_prefix" \
  "shifts=$shifts" \
  "manifest=$manifest" \
  "flow_act_steps=8" \
  "flow_integration_steps=10" \
  "flow_seed=0" \
  "repository_commit=$(git rev-parse HEAD)" \
  > "$output_root/command-contract.txt"

score_arm() {
  local shift="$1" arm="$run_prefix-shift$shift"
  "$venv/bin/python" py/scripts/eval_policy_sim.py \
    --controller flow-image \
    --checkpoint "$input_root/$arm/$final_checkpoint" \
    --flow-export "$artifact_root" \
    --flow-act-steps 8 \
    --flow-integration-steps 10 \
    --flow-seed 0 \
    --manifest "config/evaluation/$manifest" \
    --device cuda \
    --output "$output_root/$arm" \
    2>&1 | tee "$output_root/$arm.log"
}

if [ "$parallel_arms" = 1 ]; then
  pids=()
  for shift in "${shift_list[@]}"; do
    score_arm "$shift" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  [ "$status" -eq 0 ] || exit "$status"
else
  for shift in "${shift_list[@]}"; do
    score_arm "$shift"
  done
fi

baseline="$output_root/$run_prefix-shift${shift_list[0]}"
comparison="$output_root/paired-comparison.json"
"$venv/bin/python" py/scripts/compare_policy_evaluations.py \
  "$output_root/$run_prefix-shift${shift_list[0]}" \
  "$output_root/$run_prefix-shift${shift_list[1]}" \
  --baseline "$baseline" --json "$comparison" \
  2>&1 | tee "$output_root/paired-comparison.log"

(
  cd "$output_root"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
aws s3 sync "$output_root" "$output_s3" --only-show-errors

# Verify the remote bytes from the pod before reporting the result as durable.
while read -r expected relative; do
  actual=$(aws s3 cp "$output_s3/${relative#./}" - --only-show-errors | sha256sum | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    echo "published checksum mismatch: $relative" >&2
    exit 1
  }
done < "$output_root/SHA256SUMS"

echo "Published and verified $output_s3"
