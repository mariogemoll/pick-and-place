#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Score several epochs of the *baseline* blue-cube run on the same manifest the
# parity evaluation uses, so a fast-recipe result can be read against the
# right control.
#
#   ssh <host> 'VAST_INSTANCE_ID=... EPOCHS="300 500 1000 1500" bash /workspace/vast_dp_epoch_sweep.sh'
#
# The question this answers: when the 500-epoch batch-256 policy scores below
# the 1500-epoch batch-64 one, is that the *epochs* or the *recipe*? Only the
# baseline's own epoch-500 checkpoint separates them, and it costs a minute.
#
# Assumes scripts/vast_dp_fast_parity.sh has already provisioned this pod: same
# venv, same repo, same artifacts. Nothing here re-installs anything.

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
# Any run's checkpoint directory works, so the same script scores the baseline's
# epoch curve and a freshly trained run's single checkpoint. LOCAL_CHECKPOINT_DIR
# skips S3 entirely, for a checkpoint still sitting on the pod that produced it.
checkpoint_prefix="${CHECKPOINT_S3_PREFIX:-$bucket_root/outputs/dp_blue_cube_1000/pretrain/so101_pre_diffusion_unet_img_to2_ta16_te2_td100/2026-07-31_04-01-38_42/checkpoint}"
local_checkpoint_dir="${LOCAL_CHECKPOINT_DIR:-}"
arm_prefix="${ARM_PREFIX:-baseline}"

epochs="${EPOCHS:-300 500 1000}"
device="${DEVICE:-cpu}"
parallel="${PARALLEL:-76}"
scenes="${SCENES:-66}"
chunk="${CHUNK:-2}"
act_steps="${ACT_STEPS:-8}"
seed="${SEED:-0}"
# Suite to score. canonical_100_v1 is the comparable one; dr_100_v1 applies
# domain randomization the blue-cube checkpoints never trained on, so score it
# as a separate run rather than pooling the two.
manifest="${MANIFEST:-config/evaluation/canonical_100_v1.json.xz}"

run_name="${RUN_NAME:-dp_epoch_sweep_$(date +%Y%m%d_%H%M%S)}"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
artifact_root="$workspace/artifacts/blue-cube-1000-10hz-96x96"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/dp-epoch-sweep.log"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  cp "$job_log" "$output_root/job-metadata/dp-epoch-sweep.log"
  for attempt in $(seq 1 10); do
    aws s3 sync "$output_root" "$output_prefix" --only-show-errors && break
    echo "S3 sync attempt $attempt failed; retrying in 30 seconds."
    sleep 30
  done
  echo "Synced $output_prefix"
  exit "$status"
}
trap finalize EXIT

export PATH="/root/.local/bin:$PATH"
test -x "$venv/bin/python" || { echo "pod not provisioned; run vast_dp_fast_parity.sh first" >&2; exit 1; }

jobs_file="$workspace/sweep-jobs.txt"
: > "$jobs_file"
for epoch in $epochs; do
  if [ -n "$local_checkpoint_dir" ]; then
    checkpoint="$local_checkpoint_dir/state_$epoch.pt"
    [ -f "$checkpoint" ] || { echo "missing $checkpoint" >&2; exit 1; }
  else
    checkpoint="$workspace/artifacts/${arm_prefix}_state_$epoch.pt"
    [ -f "$checkpoint" ] || \
      aws s3 cp "$checkpoint_prefix/state_$epoch.pt" "$checkpoint" --only-show-errors
  fi
  offset=0; index=0
  while [ "$offset" -lt "$scenes" ]; do
    size=$(( scenes - offset )); [ "$size" -gt "$chunk" ] && size="$chunk"
    printf '%s\t%s\t%d\t%d\t%03d\n' "${arm_prefix}-${epoch}e-blue-cube" "$checkpoint" \
      "$offset" "$size" "$index" >> "$jobs_file"
    offset=$(( offset + size )); index=$(( index + 1 ))
  done
done
echo "Sweeping epochs [$epochs] as $(wc -l < "$jobs_file") shards, $parallel at a time."

cat > "$workspace/run-sweep-shard.sh" <<SHARD
#!/usr/bin/env bash
set -euo pipefail
IFS=\$'\t' read -r arm checkpoint offset size index <<< "\$1"
"$venv/bin/python" "$repo/py/scripts/eval_policy_sim.py" \\
  --controller dppo --checkpoint "\$checkpoint" \\
  --dppo-normalization "$artifact_root/normalization.npz" \\
  --dppo-python "$venv/bin/python" \\
  --dppo-act-steps "$act_steps" --dppo-seed "$seed" --device "$device" \\
  --manifest "$repo/$manifest" \\
  --offset "\$offset" --limit "\$size" --scene-appearance blue-cube \\
  --output "$output_root/\$arm/shard-\$index" \\
  > "$output_root/\$arm-shard-\$index.log" 2>&1
SHARD
chmod +x "$workspace/run-sweep-shard.sh"

wall_start=$(date +%s)
xargs -a "$jobs_file" -d '\n' -I{} -P "$parallel" "$workspace/run-sweep-shard.sh" {}
echo "Sweep finished in $(( $(date +%s) - wall_start ))s."

arms=()
for epoch in $epochs; do arms+=("$output_root/${arm_prefix}-${epoch}e-blue-cube"); done
"$venv/bin/python" "$repo/py/scripts/compare_policy_evaluations.py" "${arms[@]}" \
  --json "$output_root/epoch-sweep.json" | tee "$output_root/epoch-sweep.txt"
