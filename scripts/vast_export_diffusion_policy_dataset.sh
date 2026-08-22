#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Re-export a published LeRobot dataset as a Diffusion Policy training artifact,
# on a rented pod, and publish the result. Video decoding is the whole cost, so
# this belongs on a machine with cores rather than on a laptop.
#
#   ssh <ssh-host> 'SOURCE_NAME=two-variant-1000-blue-cube-lerobot \
#     ARTIFACT_NAME=two-variant-1000-blue-cube-delta ACTION_ENCODING=delta \
#     bash /workspace/vast_export_diffusion_policy_dataset.sh'
#
# Run vast_pap_provision.sh first: this assumes the repo and the venv.
#
# ACTION_ENCODING chooses what the policy will be trained to predict -- the
# joint command, or its offset from the joints measured on the same control
# tick. It is a property of the artifact, not of the training run, and every
# rollout path reads it back out of the normalization archive this writes.

set -euo pipefail

: "${SOURCE_NAME:?set SOURCE_NAME to a dataset archive under datasets/}"
: "${ARTIFACT_NAME:?set ARTIFACT_NAME to a fresh, never-used artifact name}"

bucket_root="s3://allyouneed/pick-and-place"
source_s3="$bucket_root/datasets/$SOURCE_NAME.tar.zst"
artifact_s3="$bucket_root/diffusion-policy-data/$ARTIFACT_NAME.tar.zst"
action_encoding="${ACTION_ENCODING:-absolute}"
image_size="${IMAGE_SIZE:-96}"
policy_hz="${POLICY_HZ:-10}"
workers="${WORKERS:-$(nproc)}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
datasets="$workspace/datasets"
artifacts="$workspace/artifacts"
artifact_root="$artifacts/$ARTIFACT_NAME"

export PATH="/root/.local/bin:$PATH"
mkdir -p "$datasets" "$artifacts"
exec > >(tee -a "$workspace/dataset-export.log") 2>&1

# Refuse to overwrite: an artifact's name is how a checkpoint is traced back to
# the bounds it was fitted against, so silently republishing one under a used
# name would make every run that cites it unreadable.
if aws s3 ls "$artifact_s3" | grep -q .; then
  echo "Artifact already published: $artifact_s3" >&2
  exit 1
fi

if [ ! -d "$datasets/$SOURCE_NAME" ]; then
  aws s3 cp "$source_s3" "$datasets/$SOURCE_NAME.tar.zst" --only-show-errors
  aws s3 cp "$source_s3.sha256" "$datasets/$SOURCE_NAME.tar.zst.sha256" --only-show-errors
  # Verify before unpacking: a truncated download otherwise surfaces as a
  # corrupt frame somewhere in the middle of an hour of decoding.
  (cd "$datasets" && sha256sum -c "$SOURCE_NAME.tar.zst.sha256")
  mkdir -p "$datasets/$SOURCE_NAME"
  tar -x -I zstd -f "$datasets/$SOURCE_NAME.tar.zst" -C "$datasets/$SOURCE_NAME"
  rm -f "$datasets/$SOURCE_NAME.tar.zst" "$datasets/$SOURCE_NAME.tar.zst.sha256"
fi

# The archives were made from different working directories over the months, so
# the LeRobot root sits at a different depth in each. Find it by what defines
# it rather than by a path convention it may not follow.
info_path=$(find "$datasets/$SOURCE_NAME" -name info.json -path '*/meta/*' -print -quit)
if [ -z "$info_path" ]; then
  echo "no meta/info.json under $datasets/$SOURCE_NAME: not a LeRobot dataset" >&2
  exit 1
fi
source_root=$(dirname "$(dirname "$info_path")")
echo "Exporting $source_root as $action_encoding actions."

cd "$repo"
git rev-parse HEAD
"$venv/bin/python" -m pick_and_place.cli.pap export-policy-dataset \
  --src "$source_root" \
  --output "$artifact_root" \
  --image-size "$image_size" \
  --policy-hz "$policy_hz" \
  --workers "$workers" \
  --action-encoding "$action_encoding"

# train.npz is deliberately ZIP_STORED, so its bytes are raw pixels and zstd has
# plenty to work with -- roughly 4.5x, which is minutes off every later launch.
# The archive holds the artifact directory, so it unpacks under its own name.
(cd "$artifacts" && tar -c -I 'zstd -T0 -10' -f "$ARTIFACT_NAME.tar.zst" "$ARTIFACT_NAME")
(cd "$artifacts" && sha256sum "$ARTIFACT_NAME.tar.zst" > "$ARTIFACT_NAME.tar.zst.sha256")
cat "$artifacts/$ARTIFACT_NAME.tar.zst.sha256"

aws s3 cp "$artifacts/$ARTIFACT_NAME.tar.zst" "$artifact_s3" --only-show-errors
aws s3 cp "$artifacts/$ARTIFACT_NAME.tar.zst.sha256" "$artifact_s3.sha256" --only-show-errors

# Read it back rather than trusting the upload: aws does not verify a multipart
# transfer, and this artifact is about to define a training run's units.
aws s3 cp "$artifact_s3" - | sha256sum | awk '{print $1}' > "$artifacts/published.sha256"
if ! grep -q "$(cat "$artifacts/published.sha256")" "$artifacts/$ARTIFACT_NAME.tar.zst.sha256"; then
  echo "Published archive does not match what was uploaded." >&2
  exit 1
fi
rm -f "$artifacts/$ARTIFACT_NAME.tar.zst"
echo "Published $artifact_s3 ($action_encoding actions)."
