#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD
#
# Baseline-vs-fast parity evaluation for the Diffusion Policy throughput work.
#
# The two checkpoints come from the same dataset export (identical
# source_sha256), so the only difference is the training recipe: 1500 epochs at
# batch 64 against 500 epochs at batch 256 with CUDA graphs and fused AdamW.
# Everything about the rollout — manifest, scene selection, sampler seed,
# executed action steps, control rate — is held fixed between the arms.
#
# The `as-compiled` arms are the control for the harness itself: these
# checkpoints were trained on the blue-cube re-render, and the compiled scene
# carries the rig's AprilTag cube, so an evaluation without
# `--scene-appearance blue-cube` shows the policy an object it has never seen.

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/evaluation/dp-fast-parity}"
SCENES="${SCENES:-66}"
CONTROL_SCENES="${CONTROL_SCENES:-10}"
ACT_STEPS="${ACT_STEPS:-8}"
SEED="${SEED:-0}"
MANIFEST="${MANIFEST:-config/evaluation/canonical_100_v1.json.xz}"
NORMALIZATION="${NORMALIZATION:-output/dp_blue_cube_1000/artifact/normalization.npz}"
PYTHON="${PYTHON:-.venv/bin/python}"

BASELINE_CKPT="${BASELINE_CKPT:-output/dp_blue_cube_1000/checkpoint/state_1500.pt}"
FAST_CKPT="${FAST_CKPT:-output/dp_blue_cube_1000_fast/checkpoint/state_500.pt}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONPATH="py/src:third_party/dppo"
export PYTHONUNBUFFERED=1

run_arm() {
  local name="$1" checkpoint="$2" appearance="$3" limit="$4"
  local output="$OUTPUT_ROOT/$name"
  if [ -e "$output" ]; then
    echo "== $name: already present, skipping"
    return
  fi
  local appearance_args=()
  [ "$appearance" != "as-compiled" ] && appearance_args=(--scene-appearance "$appearance")
  echo "== $name: $limit scenes, appearance $appearance, $(date -u +%H:%M:%S)"
  "$PYTHON" py/scripts/eval_policy_sim.py \
    --controller dppo \
    --checkpoint "$checkpoint" \
    --dppo-normalization "$NORMALIZATION" \
    --dppo-python "$PYTHON" \
    --dppo-act-steps "$ACT_STEPS" \
    --dppo-seed "$SEED" \
    --device cpu \
    --manifest "$MANIFEST" \
    --limit "$limit" \
    "${appearance_args[@]}" \
    --output "$output"
}

mkdir -p "$OUTPUT_ROOT"

# The two arms the comparison is about, first: if the machine is interrupted
# these are the ones worth having.
run_arm baseline-1500e-blue-cube "$BASELINE_CKPT" blue-cube "$SCENES"
run_arm fast-500e-blue-cube      "$FAST_CKPT"     blue-cube "$SCENES"

# The harness control, on a smaller scene count because its purpose is only to
# show that the appearance mismatch — not the policy — produced the earlier 0/33.
run_arm baseline-1500e-as-compiled "$BASELINE_CKPT" as-compiled "$CONTROL_SCENES"
run_arm fast-500e-as-compiled      "$FAST_CKPT"     as-compiled "$CONTROL_SCENES"

echo "== all arms done, $(date -u +%H:%M:%S)"
