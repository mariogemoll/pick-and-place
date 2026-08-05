#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Record one sim dataset and render it under two cube appearances, producing a
# matched pair for training: the tagged cube as recorded, and the same episodes
# with the cube recoloured blue.
#
# The pair is what makes the comparison worth running. Both variants come from a
# single replay of each recorded frame, so they share states, actions and phase
# spans bit for bit and differ only at the cube's pixels. Two policies trained on
# them differ by the appearance and nothing else -- no reseeding, no second
# recording, no drift between checkouts. The earlier AprilTag-vs-blue result had
# none of that: different hardware, different GL backend, a later checkout and a
# fresh recording, which is why it bounded sample efficiency rather than
# settling whether the tagged cube can be learned at all.
#
# The cube cannot simply be *recorded* blue. Under --miscalibration the descent
# visual servo detects the cube's AprilTags in the wrist image, so a solid
# colour cube fails every episode. Record with tags, recolour on replay.
#
# Record and re-render on the SAME machine. The camera calibrations are
# machine-local files and the OpenGL backend decides the shading, so a
# verification pass is only evidence for re-renders produced beside it.
#
# Usage:
#   EPISODES=1000 WORKERS=12 scripts/record_two_variant_dataset.sh
#
# Resumable by design: rerun the same command after an interruption and the
# recorder continues at the next unused episode index.

set -euo pipefail

export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

: "${PAP_DATA_ROOT:?set PAP_DATA_ROOT to a directory outside the repository}"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
episodes="${EPISODES:-1000}"
workers="${WORKERS:-8}"
seed="${SEED:-20260805}"
name="${NAME:-two-variant-$episodes}"
python_bin="${PYTHON:-python}"

datasets="$PAP_DATA_ROOT/datasets"
staged="$datasets/${name}_episodes"
rerender="$datasets/${name}_rerender"
# vast_diffusion_policy_train_fast.sh reads $ARTIFACT_NAME from here and only
# falls back to S3 when train.npz is missing, so exporting into this layout
# makes a locally recorded dataset trainable without an upload round trip. On a
# rented pod that is ARTIFACTS_DIR=/workspace/artifacts.
artifacts="${ARTIFACTS_DIR:-$PAP_DATA_ROOT/artifacts}"

cd "$repo/py"

echo "== 1/5 record $episodes episode(s) with the tagged cube"
"$python_bin" scripts/pick_and_place/record_sim.py \
  --episodes "$episodes" \
  --workers "$workers" \
  --seed "$seed" \
  --dataset-root "$datasets/$name" \
  --repo-id "local/$name" \
  --vcodec h264

# Prove the replay reproduces the recording on THIS machine before trusting any
# re-render from it. The render pass refuses to run without a passing report.
echo "== 2/5 verify the replay against the recorded video"
"$python_bin" scripts/rerender_episodes.py \
  --episodes-root "$staged" \
  --output "$rerender" \
  --verify --max-episodes 2

echo "== 3/5 render both appearances from one replay per frame"
"$python_bin" scripts/rerender_episodes.py \
  --episodes-root "$staged" \
  --output "$rerender" \
  --variant as-recorded blue-cube \
  --workers "$workers"

for variant in as-recorded blue-cube; do
  echo "== 4/5 finalize $variant"
  "$python_bin" scripts/pick_and_place/finalize_sim_dataset.py \
    --dataset-root "$rerender/$variant" \
    --episodes "$episodes" \
    --repo-id "local/$name-$variant" \
    --write

  echo "== 5/5 export $variant for Diffusion Policy training"
  "$python_bin" scripts/export_diffusion_policy_dataset.py \
    --src "$rerender/$variant" \
    --output "$artifacts/$name-$variant"
done

echo
echo "== confirming the two exports are a matched pair"
"$python_bin" scripts/check_variant_pair.py \
  "$artifacts/$name-as-recorded" "$artifacts/$name-blue-cube"

cat <<EOF

Train each variant, one run per appearance:

  ARTIFACT_NAME=$name-as-recorded RUN_NAME=<fresh> \\
    scripts/vast_diffusion_policy_train_fast.sh
  ARTIFACT_NAME=$name-blue-cube   RUN_NAME=<fresh> \\
    scripts/vast_diffusion_policy_train_fast.sh

Keep BATCH_SIZE and N_EPOCHS at their defaults in both, or the comparison
measures the hyperparameters instead of the cube.
EOF
