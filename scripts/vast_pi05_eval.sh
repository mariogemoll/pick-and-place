#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Score pi0.5 LoRA checkpoints on the frozen scenario manifests, on the pod that
# trained them or a freshly provisioned one.
#
#   RUN_NAME=<training run> STEPS="020000 014000" scripts/vast_pi05_eval.sh
#
# Every checkpoint is scored on smoke_v1 first. Eight scenarios cost a couple of
# minutes and catch the whole class of failures that otherwise surface a hundred
# scenarios in: a base model that will not resolve, missing scene assets, a
# camera-key mismatch. Only then does canonical_100_v1 run.
#
# Two flags that the training launcher needs are *wrong* here:
#
#   --recording-hw   eval_policy_sim.py does not accept it. It resolves the
#                    resolution from the checkpoint instead, which is what the
#                    dataset recorded (720x960), so nothing needs passing.
#   --n-action-steps eval_policy_sim.py already defaults to the checkpoint's own
#                    value. Pass EXTRA="--n-action-steps N" only to sweep it.

set -uo pipefail

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

run_name="${RUN_NAME:?set RUN_NAME to the training run whose checkpoints to score}"
steps="${STEPS:-020000}"
manifest="${MANIFEST:-canonical_100_v1.json.xz}"
checkpoint_revision="${CHECKPOINT_REVISION:-a538eb2732}"
extra="${EXTRA:-}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
bucket_root="s3://allyouneed/pick-and-place"
checkpoint_dir="$workspace/pi05_base_pinned"
ckpts="$workspace/outputs/$run_name/train/checkpoints"
out="$workspace/eval/$run_name"
mkdir -p "$out"
cd "$repo"

# The textures are renders, so they are not in the repository and a fresh clone
# cannot compile a scene. Provisioning renders them; do it here too, because
# this script is the first thing that would notice they are missing.
if [ ! -f "$repo/assets/apriltags/textures/tagStandard41h12_00014_60x60mm_tag40mm.png" ]; then
  "$venv/bin/python" -m pick_and_place.cli.pap render-apriltag-textures --all-defaults
fi

if [ ! -d "$checkpoint_dir" ]; then
  "$venv/bin/python" - "$checkpoint_revision" "$checkpoint_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("pinned base at", snapshot_download("lerobot/pi05_base", revision=sys.argv[1], local_dir=sys.argv[2]))
PY
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
  "$venv/bin/python" -m pick_and_place.cli.pap eval-policy-sim lerobot \
    --checkpoint "$ckpts/$step/pretrained_model" \
    --base-checkpoint "$checkpoint_dir" \
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
