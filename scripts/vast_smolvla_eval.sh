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
# canonical_100_v1 is then sharded across SHARDS concurrent workers and merged.
#
# **Measure a rung before sizing a deadline around a ladder.** A ten-rung ladder
# at SHARDS=8 was estimated at an hour and had not finished after **four**. The
# estimate was extrapolated from a single-process GPU figure rather than timed,
# and the shards were unpinned, so they oversubscribed the cores (see the thread
# pinning below). One process scores the hundred scenarios in ~15 minutes, so a
# serial ladder is ~2.5 hours: that is the number any sharded run has to beat,
# and it is the fallback if sharding disappoints.
#
# STEPS takes a list, so the usual invocation is every checkpoint:
#
#   RUN_NAME=<run> STEPS="005000 010000 015000 020000 025000 030000 \
#     035000 040000 045000 050000" scripts/vast_smolvla_eval.sh
#
# Ranking checkpoints needs canonical_100_v1 at every rung. smoke_v1 cannot do
# it -- the previous run's ladder differed by one episode in eight between its
# 15,000- and 20,000-step checkpoints, which is noise -- so it stays the gate
# and never the number.
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

# One thread per shard. Torch and the BLAS libraries size their intra-op pools
# from the visible core count, so an unpinned shard assumes it owns the machine.
# SHARDS of them then oversubscribe the cores by SHARDS-fold and spend their
# time in the scheduler: three concurrent scoring processes on 2026-08-12 each
# took ~50 minutes for the hundred scenarios that one process alone finished in
# ~15. Every other parallel workload here -- vast_recovery_dataset.sh,
# vast_score_dppo_env.sh, vast_scene_difficulty.sh -- pins these for the same
# reason; this script forks the most processes and was the only one that did not.
#
# Note for anything that derives a worker count below: coreutils `nproc` honours
# OMP_NUM_THREADS and will report 1. Use `nproc --all`.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

# Only pretrained_model: scoring loads the model and never the optimizer state,
# which is another 0.4 GB per checkpoint and 4 GB across a ten-rung ladder.
for step in $steps; do
  if [ ! -d "$ckpts/$step/pretrained_model" ]; then
    echo "Fetching checkpoint $step from S3."
    mkdir -p "$ckpts/$step/pretrained_model"
    aws s3 sync \
      "$bucket_root/outputs/$run_name/train/checkpoints/$step/pretrained_model" \
      "$ckpts/$step/pretrained_model" --only-show-errors || exit 1
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

# Score one manifest across concurrent workers and merge the shards.
#
# This is what makes a ladder affordable. Scoring is MuJoCo rendering with
# occasional inference -- measured at 3-8% GPU and 2.1 GB VRAM for one process
# -- so it is CPU and render bound. The previous SmolVLA run could only afford
# smoke_v1's eight scenarios per checkpoint, which its own notes say is too few
# to rank them, which is why "is 20,000 steps the ceiling?" went unanswered.
#
# Two ceilings bound SHARDS, and the lower one wins:
#
#   VRAM  -- each worker loads its own model at ~2.1 GB, so a 32 GB card holds
#            about a dozen alongside the renderer.
#   Cores -- each worker is pinned to one thread above, so SHARDS beyond
#            `nproc --all` oversubscribes however much VRAM is free.
#
# Vast pods are often long on VRAM and short on vCPUs, so cores usually bind
# first. That is the opposite of the training launcher, where VRAM binds, and
# taking the training intuition on trust here is what produced the four-hour
# ladder.
cores="$(nproc --all)"
shards="${SHARDS:-8}"
if [ "$shards" -gt "$cores" ]; then
  echo "SHARDS=$shards exceeds the $cores cores on this host; using $cores." >&2
  shards="$cores"
fi

score_sharded() {
  local step="$1" manifest="$2" tag="$3"
  local total pids=() rc=0 index=0
  total=$("$venv/bin/python" -c "
import sys
from pick_and_place.policies.policy_evaluation import ScenarioManifest
print(len(ScenarioManifest.load('config/evaluation/$manifest').scenarios))
") || return 1

  echo "=== $tag: step $step on $manifest, $total scenarios across $shards shards ==="
  rm -rf "${out:?}/$tag" "${out:?}/shards/$tag"
  mkdir -p "$out/shards/$tag"

  # Contiguous windows, sized like the merge tool's own test: the first
  # remainder shards take one extra scenario so the union is exactly the suite.
  while [ "$index" -lt "$shards" ]; do
    local lo hi
    lo=$(( index * total / shards ))
    hi=$(( (index + 1) * total / shards ))
    if [ "$hi" -gt "$lo" ]; then
      # shellcheck disable=SC2086
      "$venv/bin/python" py/scripts/eval_policy_sim.py \
        --controller lerobot \
        --checkpoint "$ckpts/$step/pretrained_model" \
        --manifest "config/evaluation/$manifest" \
        --output "$out/shards/$tag/$(printf '%02d' "$index")" \
        --offset "$lo" --limit "$(( hi - lo ))" \
        --device cuda $extra \
        > "$out/shards/$tag/$(printf '%02d' "$index").log" 2>&1 &
      pids+=($!)
    fi
    index=$(( index + 1 ))
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  if [ "$rc" -ne 0 ]; then
    echo "A shard of $tag failed; its log is under $out/shards/$tag." >&2
    return 1
  fi

  # The merge refuses overlapping windows and shards that scored a different
  # checkpoint, so a partial or stale set cannot become a headline number.
  "$venv/bin/python" py/scripts/merge_evaluation_shards.py \
    --output "$out/$tag" "$out/shards/$tag"/*/
}

status=0
echo "Scoring $(set -- $steps; echo $#) rung(s) at SHARDS=$shards on $cores cores."
for step in $steps; do
  rung_started=$SECONDS
  # Serial, and first: eight scenarios cost a couple of minutes and catch the
  # failures that would otherwise appear simultaneously in every shard.
  if ! score "$step" smoke_v1.json "smoke-$step"; then
    echo "Smoke scoring failed for $step; not spending $manifest on it." >&2
    status=1
    continue
  fi
  score_sharded "$step" "$manifest" "headline-$step" || status=1
  # Report what the rung cost, so the next ladder is sized from a measurement
  # instead of an extrapolation. Anyone watching the log can multiply this by
  # the rungs left and compare it against the teardown deadline.
  echo "=== rung $step took $(( (SECONDS - rung_started) / 60 ))m$(( (SECONDS - rung_started) % 60 ))s ==="
  # Sync after every rung, not once at the end.
  #
  # On 2026-08-14 a ten-rung ladder ran unattended for four hours, was destroyed
  # by its teardown deadline before the final sync, and left *nothing* in S3 --
  # the same way the pi0.5 evaluation artifacts were lost. A ladder is ten
  # independent results, so there is no reason to hold any of them hostage to
  # the last one finishing.
  aws s3 sync "$out" "$bucket_root/outputs/$run_name/evaluation" \
    --no-follow-symlinks --only-show-errors \
    || echo "Interim sync after step $step failed; continuing." >&2
done

aws s3 sync "$out" "$bucket_root/outputs/$run_name/evaluation" --no-follow-symlinks \
  --only-show-errors
echo "Results under $out and $bucket_root/outputs/$run_name/evaluation"
exit "$status"
