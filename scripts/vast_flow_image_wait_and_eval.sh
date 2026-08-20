#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Wait for every final image-flow manifest named in SHIFTS, then run the
# evaluator on this pod. This is for reusing a completed training pod instead of
# paying for an idle third machine while its counterpart finishes.
#
# The controller must replace this pod's training-prefix completion tracker
# entry with its evaluation-prefix entry before the first training manifest
# appears. The evaluator publishes that latter prefix only after every result
# file has been hashed, so the controller's normal completion watcher can safely
# destroy the pod when this script exits successfully.

set -euo pipefail

run_prefix="${RUN_PREFIX:?set RUN_PREFIX to the common training output prefix}"
shifts="${SHIFTS:-8 0}"
poll_seconds="${POLL_SECONDS:-60}"
eval_repo="${EVAL_REPO:-/workspace/pick-and-place-eval}"
source_repo="${SOURCE_REPO:-/workspace/pick-and-place}"
eval_ref="${EVAL_REF:-origin/train-randomized-image-flow}"
bucket_root="s3://allyouneed/pick-and-place"

read -r -a shift_list <<< "$shifts"
if [ "${#shift_list[@]}" -lt 1 ]; then
  echo "SHIFTS must name at least one arm; got: $shifts" >&2
  exit 2
fi

while :; do
  ready=1
  for shift in "${shift_list[@]}"; do
    if ! aws s3 ls "$bucket_root/outputs/$run_prefix-shift$shift/SHA256SUMS" >/dev/null 2>&1; then
      ready=0
      break
    fi
  done
  if [ "$ready" -eq 1 ]; then
    break
  fi
  echo "$(date -u +%FT%TZ) waiting for every final training manifest"
  sleep "$poll_seconds"
done

if [ -e "$eval_repo" ]; then
  echo "Evaluation worktree already exists: $eval_repo" >&2
  exit 1
fi
git -C "$source_repo" fetch origin train-randomized-image-flow
git -C "$source_repo" worktree add --detach "$eval_repo" "$eval_ref"

REPO="$eval_repo" RUN_PREFIX="$run_prefix" SHIFTS="$shifts" \
  bash "$eval_repo/scripts/vast_flow_image_eval.sh"
