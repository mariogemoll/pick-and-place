#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Run visual Diffusion Policy training on a rented RTX 5090. The launcher
# installs credentials and identifiers; this script keeps the VM alive after
# syncing so teardown can be performed only after an independent S3 check.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${VAST_INSTANCE_ID:?}"

destroy_on_exit="${DESTROY_ON_EXIT:-false}"
bucket_root="s3://allyouneed/pick-and-place"
artifact_name="blue-cube-no-dr-200-10hz-96x96"
artifact_prefix="$bucket_root/diffusion-policy-data"
run_name="dp_blue_cube_no_dr_unet_1000e_20260730"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
archive="$workspace/artifacts/$artifact_name.tar.zst"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/training-job.log"
status_file="$workspace/training-status.json"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  python3 - "$status_file" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "exit_status": int(sys.argv[2]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
  cp "$job_log" "$output_root/job-metadata/training-job.log"
  cp "$status_file" "$output_root/job-metadata/status.json"
  if [ -f "$artifact_root/export.json" ]; then
    cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"
  fi
  if [ -f "$artifact_root/source-rerender.json" ]; then
    cp "$artifact_root/source-rerender.json" "$output_root/job-metadata/source-rerender.json"
  fi

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

  echo "Final S3 sync verified; leaving instance running for independent verification."
  if [ "$destroy_on_exit" = true ]; then
    echo "DESTROY_ON_EXIT=true was ignored: teardown is intentionally manual." >&2
  fi
  exit "$status"
}
trap finalize EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl git unzip zstd
if ! command -v aws >/dev/null; then
  aws_install_dir=$(mktemp -d)
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$aws_install_dir/awscliv2.zip" \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
  unzip -tq "$aws_install_dir/awscliv2.zip" >/dev/null
  unzip -q "$aws_install_dir/awscliv2.zip" -d "$aws_install_dir"
  "$aws_install_dir/aws/install" --update
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

aws sts get-caller-identity --query Account --output text
nvidia-smi

if [ ! -d "$repo/.git" ]; then
  git clone --recurse-submodules https://github.com/mariogemoll/pick-and-place.git "$repo"
fi
cd "$repo"
git submodule update --init --recursive
git rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"

venv="$workspace/venvs/pick-and-place"
base_python="python3"
if [ -x /venv/main/bin/python ]; then
  base_python="/venv/main/bin/python"
fi
if [ ! -x "$venv/bin/python" ]; then
  uv venv --python "$base_python" "$venv"
fi
uv pip install --python "$venv/bin/python" \
  --overrides config/diffusion_policy/torch-rtx5090.txt \
  -e py -e third_party/dppo

# Refuse to overwrite or resume a run. This name is new and checkpoints from
# previous runs remain under their existing S3 prefixes.
if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Fresh output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

aws s3 cp "$artifact_prefix/$artifact_name.tar.zst" "$archive" --only-show-errors
aws s3 cp "$artifact_prefix/$artifact_name.tar.zst.sha256" "$archive.sha256" --only-show-errors
(cd "$(dirname "$archive")" && sha256sum --check "$(basename "$archive").sha256")
tar --use-compress-program=unzstd -xf "$archive" -C "$(dirname "$artifact_root")"

"$venv/bin/python" - "$artifact_root" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
export = json.loads((root / "export.json").read_text())
rerender_bytes = (root / "source-rerender.json").read_bytes()
rerender = json.loads(rerender_bytes)
if export.get("num_episodes") != 200 or export.get("num_frames") != 19306:
    raise ValueError("unexpected episode or training-frame count")
if export.get("image_size") != [96, 96] or export.get("fps") != 10:
    raise ValueError("dataset does not match the 96x96, 10 Hz training contract")
if rerender.get("variant") != "blue-cube" or len(rerender.get("episodes", [])) != 200:
    raise ValueError("dataset is not the verified 200-episode blue-cube rerender")
if rerender.get("camera_randomization") is not None:
    raise ValueError("dataset provenance enables camera randomization")
if rerender.get("background_randomization") is not None:
    raise ValueError("dataset provenance enables background randomization")
if rerender.get("verified_against") is None:
    raise ValueError("rerender lacks environment-verification evidence")
actual_rerender_hash = hashlib.sha256(rerender_bytes).hexdigest()
if export.get("source_rerender_sha256") != actual_rerender_hash:
    raise ValueError("embedded rerender manifest hash mismatch")
with zipfile.ZipFile(root / "train.npz") as archive:
    expected = {"states.npy", "actions.npy", "images.npy", "traj_lengths.npy"}
    if set(archive.namelist()) != expected:
        raise ValueError(f"unexpected train.npz members: {archive.namelist()}")
    if any(member.compress_type for member in archive.infolist()):
        raise ValueError("train.npz arrays are not directly memory-mappable")
print("Verified downloaded blue-cube artifact provenance and training contract.")
PY

cp "$repo/config/diffusion_policy/pretrain_so101_unet_img.yaml" \
  "$output_root/job-metadata/launcher-config.yaml"
cp "$archive.sha256" "$output_root/job-metadata/dataset-archive.sha256"

export DPPO_DATA_DIR="$artifact_root"
export DPPO_LOG_DIR="$output_root"
export PYTHONPATH="$repo/third_party/dppo"
cd "$repo/third_party/dppo"
( while sleep 900; do
    aws s3 sync "$output_root" "$output_prefix" --only-show-errors
  done ) &
sync_pid=$!
set +e
"$venv/bin/python" script/run.py \
  --config-path "$repo/config/diffusion_policy" \
  --config-name pretrain_so101_unet_img \
  train.n_epochs=1000 \
  train.lr_scheduler.first_cycle_steps=1000 \
  model.network.augment=false \
  2>&1 | tee "$output_root/console.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

checkpoint=$(find "$output_root/pretrain" -type f -path '*/checkpoint/state_1000.pt' -print -quit)
if [ -z "$checkpoint" ]; then
  echo "Training returned success without the required epoch-1000 checkpoint." >&2
  exit 1
fi
sha256sum "$checkpoint" | tee "$output_root/job-metadata/state_1000.pt.sha256"
echo "Training completed with the required epoch-1000 checkpoint."
