#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Score SmolVLA checkpoints on the frozen scenario manifests, on the pod that
# trained them or a freshly provisioned one.
#
#   RUN_NAME=<training run> STEPS="030000 020000" scripts/vast_smolvla_eval.sh
#
# Every checkpoint is scored on smoke_v1 first. Eight scenarios cost a couple of
# minutes and catch the whole class of failures that otherwise surface a hundred
# scenarios in: a checkpoint that will not resolve, missing scene assets, a
# camera-key mismatch. Only then does canonical_100_v1 run.
#
# Unlike the pi0.5 equivalent this needs no --base-checkpoint. SmolVLA trains
# without adapters (use_peft stays false), so a checkpoint is a complete model
# and _peft_base_checkpoint() correctly finds no adapter_config.json to resolve.
#
# --n-action-steps is not passed: eval_policy_sim.py defaults to the
# checkpoint's own value, which the launcher pinned to 10. Pass
# EXTRA="--n-action-steps N" only to sweep it.
#
# **Sync before teardown.** The pi0.5 evaluation artifacts were lost to a pod
# destroyed before its results reached S3, which is why that run's per-episode
# records and checkpoint fingerprints no longer exist. The sync at the end of
# this script is that lesson; do not destroy an instance before it has run.

set -uo pipefail

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

run_name="${RUN_NAME:?set RUN_NAME to the training run whose checkpoints to score}"
steps="${STEPS:-030000}"
manifest="${MANIFEST:-canonical_100_v1.json.xz}"
extra="${EXTRA:-}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
bucket_root="s3://allyouneed/pick-and-place"
ckpts="$workspace/outputs/$run_name/train/checkpoints"
out="$workspace/eval/$run_name"
mkdir -p "$out"
cd "$repo"

# The textures are renders, so they are not in the repository and a fresh clone
# cannot compile a scene. Provisioning renders them; do it here too, because
# this script is the first thing that would notice they are missing.
if [ ! -f "$repo/assets/apriltags/textures/tagStandard41h12_00014_60x60mm_tag40mm.png" ]; then
  "$venv/bin/python" py/scripts/render_apriltag_textures.py --all-defaults
fi

for step in $steps; do
  if [ ! -d "$ckpts/$step" ]; then
    echo "Fetching checkpoint $step from S3."
    mkdir -p "$ckpts/$step"
    aws s3 sync "$bucket_root/outputs/$run_name/train/checkpoints/$step" "$ckpts/$step" \
      --only-show-errors || exit 1
  fi
done

score() {
  local step="$1" manifest="$2" tag="$3"
  echo "=== $tag: step $step on $manifest ==="
  rm -rf "${out:?}/$tag"
  # shellcheck disable=SC2086
  "$venv/bin/python" py/scripts/eval_policy_sim.py \
    --controller lerobot \
    --checkpoint "$ckpts/$step/pretrained_model" \
    --manifest "config/evaluation/$manifest" \
    --output "$out/$tag" \
    --device cuda $extra
  local rc=$?
  echo "rc=$rc for $tag"
  return $rc
}

status=0
for step in $steps; do
  if ! score "$step" smoke_v1.json "smoke-$step"; then
    echo "Smoke scoring failed for $step; not spending $manifest on it." >&2
    status=1
    continue
  fi
  score "$step" "$manifest" "headline-$step" || status=1
done

aws s3 sync "$out" "$bucket_root/outputs/$run_name/evaluation" --no-follow-symlinks \
  --only-show-errors
echo "Results under $out and $bucket_root/outputs/$run_name/evaluation"
exit "$status"
