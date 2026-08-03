#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Baseline-vs-fast closed-loop parity evaluation on a rented pod.
#
#   scp scripts/vast_dp_fast_parity.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'VAST_INSTANCE_ID=... DEVICE=cuda PARALLEL=8 bash /workspace/vast_dp_fast_parity.sh'
#
# Provisioning is delegated to scripts/vast_pap_provision.sh, which the
# training and sweep jobs share. Nothing is trained; this replays a frozen
# scenario manifest against two checkpoints that came from the same dataset
# export, so the only difference between the arms is the training recipe:
# 1500 epochs at batch 64 against 500 epochs at batch 256.
#
# Scenarios are independent and carry their own seeds, so the suite shards
# cleanly across workers with --offset/--limit. `DEVICE` selects where DDPM
# sampling runs; rendering is EGL on the GPU either way, so a cuda/cpu pair of
# runs isolates the sampler rather than confounding it with the renderer.
#
# Note that cpu and cuda sampling draw different noise from the same seed, so
# the two are independent replications rather than a determinism check.

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
baseline_s3="$bucket_root/outputs/dp_blue_cube_1000/pretrain/so101_pre_diffusion_unet_img_to2_ta16_te2_td100/2026-07-31_04-01-38_42/checkpoint/state_1500.pt"
baseline_sha256="4bbcdc552942456dc76bd2911f57e808da5314505ae15e8581bd3bdcfbb57846"
fast_s3="$bucket_root/outputs/dp_blue_cube_1000_fast_b256_500e_v2_20260802/state_500_weights_only_safe.pt"
fast_sha256="b733eb3c9ca32ebc69fcf0669ad967a2464419da29ba0d20cf0ca0b577aa159e"
artifact_s3="$bucket_root/diffusion-policy-data/blue-cube-1000-10hz-96x96"
normalization_sha256="138facc5ff6611e2a82e607dc3658a2b2cd50a12e6d91d7a7051da38d16482d4"

device="${DEVICE:-cuda}"
parallel="${PARALLEL:-8}"
scenes="${SCENES:-66}"
control_scenes="${CONTROL_SCENES:-10}"
chunk="${CHUNK:-11}"
act_steps="${ACT_STEPS:-8}"
seed="${SEED:-0}"
# Suite to score. canonical_100_v1 is the comparable one; dr_100_v1 applies
# domain randomization the blue-cube checkpoints never trained on, so score it
# as a separate run rather than pooling the two.
manifest="${MANIFEST:-config/evaluation/canonical_100_v1.json.xz}"

run_name="${RUN_NAME:-dp_fast_parity_${device}_$(date +%Y%m%d_%H%M%S)}"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
artifact_root="$workspace/artifacts/blue-cube-1000-10hz-96x96"
baseline_ckpt="$workspace/artifacts/baseline_state_1500.pt"
fast_ckpt="$workspace/artifacts/fast_state_500.pt"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/dp-fast-parity.log"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  cp "$job_log" "$output_root/job-metadata/dp-fast-parity.log"
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
  echo "Final S3 sync verified: $output_prefix"
  exit "$status"
}
trap finalize EXIT

# Provisioning is shared with the training and sweep jobs; one definition.
export PATH="/root/.local/bin:$PATH"
bash "$(dirname "${BASH_SOURCE[0]}")/vast_pap_provision.sh"

venv="$workspace/venvs/pick-and-place"
cd "$repo"
git rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"
[ -f "$workspace/overlay.tar.gz" ] && \
  sha256sum "$workspace/overlay.tar.gz" | tee "$output_root/job-metadata/overlay-sha256.txt"

# Checkpoints and the normalization the policy server needs. Verified by hash:
# a truncated download would otherwise surface as a mysteriously bad policy.
[ -f "$baseline_ckpt" ] || aws s3 cp "$baseline_s3" "$baseline_ckpt" --only-show-errors
[ -f "$fast_ckpt" ] || aws s3 cp "$fast_s3" "$fast_ckpt" --only-show-errors
[ -f "$artifact_root/normalization.npz" ] || \
  aws s3 sync "$artifact_s3" "$artifact_root" --exclude '*' --include 'normalization.npz' \
    --include 'export.json' --only-show-errors
echo "$baseline_sha256  $baseline_ckpt" | sha256sum -c -
echo "$fast_sha256  $fast_ckpt" | sha256sum -c -
echo "$normalization_sha256  $artifact_root/normalization.npz" | sha256sum -c -
cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"

# One line per shard: arm, checkpoint, appearance, offset, limit. Every shard is
# an independent process with its own policy server, so the pool scales to
# whatever the box has without any shared state to contend on.
jobs_file="$workspace/parity-jobs.txt"
: > "$jobs_file"
emit_shards() {
  local arm="$1" checkpoint="$2" appearance="$3" total="$4"
  local offset=0 index=0 size
  while [ "$offset" -lt "$total" ]; do
    size=$(( total - offset ))
    [ "$size" -gt "$chunk" ] && size="$chunk"
    printf '%s\t%s\t%s\t%d\t%d\t%03d\n' \
      "$arm" "$checkpoint" "$appearance" "$offset" "$size" "$index" >> "$jobs_file"
    offset=$(( offset + size ))
    index=$(( index + 1 ))
  done
}
emit_shards baseline-1500e-blue-cube   "$baseline_ckpt" blue-cube   "$scenes"
emit_shards fast-500e-blue-cube        "$fast_ckpt"     blue-cube   "$scenes"
emit_shards baseline-1500e-as-compiled "$baseline_ckpt" as-compiled "$control_scenes"
emit_shards fast-500e-as-compiled      "$fast_ckpt"     as-compiled "$control_scenes"
echo "Sharded into $(wc -l < "$jobs_file") jobs, $parallel at a time, device $device."

cat > "$workspace/run-shard.sh" <<SHARD
#!/usr/bin/env bash
set -euo pipefail
IFS=\$'\t' read -r arm checkpoint appearance offset size index <<< "\$1"
out="$output_root/\$arm/shard-\$index"
appearance_args=()
[ "\$appearance" != "as-compiled" ] && appearance_args=(--scene-appearance "\$appearance")
started=\$(date +%s)
"$venv/bin/python" "$repo/py/scripts/eval_policy_sim.py" \\
  --controller dppo \\
  --checkpoint "\$checkpoint" \\
  --dppo-normalization "$artifact_root/normalization.npz" \\
  --dppo-python "$venv/bin/python" \\
  --dppo-act-steps "$act_steps" \\
  --dppo-seed "$seed" \\
  --device "$device" \\
  --manifest "$repo/$manifest" \\
  --offset "\$offset" --limit "\$size" \\
  "\${appearance_args[@]}" \\
  --output "\$out" > "$output_root/\$arm-shard-\$index.log" 2>&1
echo "\$arm shard \$index (\$size scenes from \$offset) took \$(( \$(date +%s) - started ))s"
SHARD
chmod +x "$workspace/run-shard.sh"

wall_start=$(date +%s)
set +e
# -I{} keeps the tab-separated record intact as one argument.
xargs -a "$jobs_file" -d '\n' -I{} -P "$parallel" "$workspace/run-shard.sh" {}
pool_status=$?
set -e
wall_seconds=$(( $(date +%s) - wall_start ))

if [ "$pool_status" -ne 0 ]; then
  echo "A shard failed; its log is in $output_root. Last lines:" >&2
  tail -n 20 "$output_root"/*-shard-*.log >&2
  exit "$pool_status"
fi

episodes=$(cat "$output_root"/*/shard-*/episodes.jsonl | wc -l)
"$venv/bin/python" - "$output_root/job-metadata/timing.json" <<PY
import json
import os
import subprocess
import sys
from pathlib import Path

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip()
episodes = $episodes
seconds = $wall_seconds
Path(sys.argv[1]).write_text(json.dumps({
    "device": "$device",
    "gpu": gpu,
    # os.cpu_count, not nproc: nproc honours OMP_NUM_THREADS, which this job
    # sets to 1, so it reported "cpu_cores": 1 on a 192-core host.
    "cpu_cores": os.cpu_count(),
    "parallel_workers": $parallel,
    "shard_size": $chunk,
    "episodes": episodes,
    "wall_seconds": seconds,
    "seconds_per_episode": seconds / episodes if episodes else None,
    "episodes_per_minute": 60.0 * episodes / seconds if seconds else None,
}, indent=2) + "\n")
print(json.dumps(json.loads(Path(sys.argv[1]).read_text()), indent=2))
PY

"$venv/bin/python" "$repo/py/scripts/compare_policy_evaluations.py" \
  "$output_root/baseline-1500e-blue-cube" \
  "$output_root/fast-500e-blue-cube" \
  --json "$output_root/comparison-blue-cube.json" \
  | tee "$output_root/comparison-blue-cube.txt"

"$venv/bin/python" "$repo/py/scripts/compare_policy_evaluations.py" \
  "$output_root/baseline-1500e-as-compiled" \
  "$output_root/fast-500e-as-compiled" \
  --json "$output_root/comparison-as-compiled.json" \
  | tee "$output_root/comparison-as-compiled.txt"

echo "Parity evaluation complete in ${wall_seconds}s over $episodes episodes on $device."
