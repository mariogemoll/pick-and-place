#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Evaluate one or more final image-flow arms on a frozen manifest and publish a
# self-verifying result bundle. Run after vast_pap_provision.sh on a fresh GPU
# pod or on a still-live training pod.
#
#   RUN_PREFIX=flow-image-randomized-maxretry1-224-300k-seed0 \
#     scripts/vast_flow_image_eval.sh
#
#   RUN_PREFIX=flow-image-new-scripted-standardcam-dr-224-300k-seed0 \
#     SHIFTS=8 ARTIFACT_NAME=new-scripted-standardcam-dr-1000-224 \
#     scripts/vast_flow_image_eval.sh
#
# Naming exactly two arms adds the paired comparison; a single arm is scored on
# its own, to be read against a bundle another run already published. Both are
# the same rollout contract, so the two cases stay comparable.
#
# The final checkpoints must already have a SHA256SUMS manifest in S3. This is
# intentional: a checkpoint that merely exists is not evidence that the pod
# finished uploading it, and a published result is not worth having from a
# partial arm.

set -euo pipefail

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# One thread per shard. Torch and the BLAS libraries size their intra-op pools
# from the visible core count, which on a Vast pod is the host's, so an unpinned
# shard assumes it owns a machine it has a slice of.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_prefix="${RUN_PREFIX:?set RUN_PREFIX to the common training output prefix}"
shifts="${SHIFTS:-8 0}"
manifest="${MANIFEST:-randomized_selection_200_v1/manifest.json}"
# The flow sampler's noise draw. Holding it fixed is what makes two arms
# comparable; varying it is how the same arm is asked the same scenario more
# than once, which is the only way to tell a scene it cannot do from a scene it
# does not always do.
flow_seed="${FLOW_SEED:-0}"
read -r -a shift_list <<< "$shifts"
if [ "${#shift_list[@]}" -lt 1 ]; then
  echo "SHIFTS must name at least one arm; got: $shifts" >&2
  exit 2
fi
seed_suffix=""
[ "$flow_seed" = 0 ] || seed_suffix="-seed$flow_seed"
if [ "${#shift_list[@]}" -eq 2 ]; then
  default_eval_name="$run_prefix-paired-$(date -u +%Y%m%d)$seed_suffix"
else
  default_eval_name="$run_prefix-shift${shift_list[0]}-$(date -u +%Y%m%d)$seed_suffix"
fi
eval_name="${EVAL_NAME:-$default_eval_name}"

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

# The container's CPU budget, which on a Vast pod is not what any of the usual
# calls report: on a host renting out a slice of its cores, `nproc`,
# `nproc --all` and sched_getaffinity all answer with the host's count, and only
# the cgroup knows the allotment. Hosts run both cgroup versions.
detect_cores() {
  local quota period
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [ "$quota" != "max" ] && [ "$period" -gt 0 ]; then
      echo $(( quota / period ))
      return
    fi
  fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$quota" -gt 0 ] && [ "$period" -gt 0 ]; then
      echo $(( quota / period ))
      return
    fi
  fi
  nproc --all
}

# Scoring is MuJoCo rendering with occasional inference -- 3-8% GPU and 2.1 GB
# VRAM for one process -- so a single-process arm leaves an evaluation pod
# almost idle. Shards are what make the wall clock match what is rented: split
# the suite into contiguous windows, score them concurrently, and merge.
cores="$(detect_cores)"
total_workers="${SHARDS:-$(( cores - 1 ))}"
[ "$total_workers" -ge 1 ] || total_workers=1
if [ "$total_workers" -gt "$cores" ]; then
  echo "SHARDS=$total_workers exceeds this container's $cores cores; using $cores." >&2
  total_workers="$cores"
fi
# Arms are scored one after another, each across every worker, rather than
# concurrently across a share of them. Same total work either way, but this way
# the first arm's result exists early instead of both arriving at the end.
shards="$total_workers"

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
  "export=$artifact_name" \
  "flow_act_steps=8" \
  "flow_integration_steps=10" \
  "flow_seed=$flow_seed" \
  "shards=$shards" \
  "repository_commit=$(git rev-parse HEAD)" \
  > "$output_root/command-contract.txt"

scenario_count=$("$venv/bin/python" -c "
from pick_and_place.policies.policy_evaluation import ScenarioManifest
print(len(ScenarioManifest.load('config/evaluation/$manifest').scenarios))
")

score_arm() {
  local shift="$1" arm="$run_prefix-shift$shift"
  local shard_root="$output_root/shards/$arm"
  local pids=() index=0 status=0

  echo "=== $arm: $scenario_count scenarios across $shards shards on $cores cores ==="
  mkdir -p "$shard_root"
  while [ "$index" -lt "$shards" ]; do
    local lo hi name
    lo=$(( index * scenario_count / shards ))
    hi=$(( (index + 1) * scenario_count / shards ))
    name=$(printf 'shard-%02d' "$index")
    if [ "$hi" -gt "$lo" ]; then
      "$venv/bin/python" py/scripts/eval_policy_sim.py \
        --controller flow-image \
        --checkpoint "$input_root/$arm/$final_checkpoint" \
        --flow-export "$artifact_root" \
        --flow-act-steps 8 \
        --flow-integration-steps 10 \
        --flow-seed "$flow_seed" \
        --manifest "config/evaluation/$manifest" \
        --offset "$lo" --limit "$(( hi - lo ))" \
        --device cuda \
        --output "$shard_root/$name" \
        > "$shard_root/$name.log" 2>&1 &
      pids+=("$!")
    fi
    index=$(( index + 1 ))
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if [ "$status" -ne 0 ]; then
    echo "A shard of $arm failed; its log is under $shard_root." >&2
    return 1
  fi

  # The merge refuses overlapping windows and shards that scored a different
  # checkpoint, so a partial or stale set cannot become a headline number.
  "$venv/bin/python" py/scripts/merge_evaluation_shards.py \
    --output "$output_root/$arm" "$shard_root"/shard-*/ \
    2>&1 | tee "$output_root/$arm.log"
}

for shift in "${shift_list[@]}"; do
  arm_started=$SECONDS
  score_arm "$shift"
  echo "=== shift-$shift took $(( (SECONDS - arm_started) / 60 ))m$(( (SECONDS - arm_started) % 60 ))s ==="
done

if [ "${#shift_list[@]}" -eq 2 ]; then
  baseline="$output_root/$run_prefix-shift${shift_list[0]}"
  comparison="$output_root/paired-comparison.json"
  "$venv/bin/python" py/scripts/compare_policy_evaluations.py \
    "$output_root/$run_prefix-shift${shift_list[0]}" \
    "$output_root/$run_prefix-shift${shift_list[1]}" \
    --baseline "$baseline" --json "$comparison" \
    2>&1 | tee "$output_root/paired-comparison.log"
fi

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
