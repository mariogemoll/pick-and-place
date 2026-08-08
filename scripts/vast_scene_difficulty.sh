#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Measure per-scene difficulty of the pretrained blue-cube policy on a rented
# GPU. Provisioning is the same as scripts/vast_dppo_finetune.sh -- same repo,
# same overrides, same CUDA repair, same artifacts -- but nothing is trained:
# it replays a fixed scene set many times and syncs the per-episode record.
#
#   scp scripts/vast_scene_difficulty.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'VAST_INSTANCE_ID=... bash /workspace/vast_scene_difficulty.sh'
#
# Rollouts are CPU-bound MuJoCo physics, so N_ENVS wants roughly that many
# vCPUs. The whole job is minutes, not hours.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
# Which policy to measure. The default is the checkpoint the scene-difficulty
# and chunk-length sweeps in docs/POLICY_FAILURE_ANATOMY.md were run on; any
# other pretrain run can be measured by pointing BASE_RUN at its output prefix.
# A checkpoint and the normalization it was trained against must travel
# together: normalization is per-dataset min-max, so pairing a checkpoint with
# another dataset's statistics silently mis-scales every observation and action
# rather than failing, and the sweep still produces plausible-looking numbers.
base_run="${BASE_RUN:-dp_blue_cube_1000/pretrain/so101_pre_diffusion_unet_img_to2_ta16_te2_td100/2026-07-31_04-01-38_42}"
base_epoch="${BASE_EPOCH:-1500}"
base_policy_s3="$bucket_root/outputs/$base_run/checkpoint/state_$base_epoch.pt"
# The dataset export the policy was trained on, which is where normalization.npz
# comes from. ARTIFACT_NAME must name the export used for BASE_RUN.
artifact_name="${ARTIFACT_NAME:-blue-cube-1000-10hz-96x96}"
artifact_s3="$bucket_root/diffusion-policy-data/$artifact_name"

run_name="${RUN_NAME:-scene_difficulty_$(date +%Y%m%d_%H%M%S)}"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
base_policy="$workspace/artifacts/state_$base_epoch.pt"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/scene-difficulty.log"
status_file="$workspace/scene-difficulty-status.json"

n_envs="${N_ENVS:-64}"
scenes="${SCENES:-256}"
repeats="${REPEATS:-8}"
# The held-out stream the paired A/B used; no fine-tuning run trained on it.
scene_seed_base="${SCENE_SEED_BASE:-6000000}"
# The distribution PPO samples from during rollout collection, which is the one
# whose within-scene variance decides whether there is a learning signal.
sampling_std="${SAMPLING_STD:-0.01}"

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
  cp "$job_log" "$output_root/job-metadata/scene-difficulty.log"
  cp "$status_file" "$output_root/job-metadata/status.json"

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
  echo "Final S3 sync verified."
  exit "$status"
}
trap finalize EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl git unzip zstd libegl1 libgl1
if ! command -v aws >/dev/null; then
  aws_install_dir=$(mktemp -d)
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$aws_install_dir/awscliv2.zip" \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
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

# The RL plumbing and this sweep are not committed yet, so the launcher ships
# the working tree as a tarball and it is unpacked over the clone. Recorded in
# job-metadata so a result can be traced back to the exact files that produced it.
if [ -f "$workspace/overlay.tar.gz" ]; then
  tar -xzf "$workspace/overlay.tar.gz" -C "$repo"
  tar -tzf "$workspace/overlay.tar.gz" | tee "$output_root/job-metadata/overlay-files.txt"
  sha256sum "$workspace/overlay.tar.gz" | tee "$output_root/job-metadata/overlay-sha256.txt"
  echo "Applied working-tree overlay."
else
  echo "No overlay tarball; running the committed tree." >&2
fi

venv="$workspace/venvs/pick-and-place"
# Pin the interpreter rather than inheriting the image's, the same way
# vast_pap_provision.sh does. vastai/pytorch:latest ships CPython 3.10 at
# /venv/main/bin/python, and this package floors at 3.12 (lerobot 0.5.1), so
# inheriting it fails resolution outright -- which is exactly how this script
# failed the first time it was pointed at a current image.
#
# Not PYTHON_VERSION: that name is already exported by the image, set to its own
# 3.10, so an override called that is silently supplied by the environment.
python_version="${PAP_PYTHON_VERSION:-3.12}"
# Recreating on a version mismatch keeps the "every step is idempotent" promise
# honest: a venv left behind by an earlier run on the wrong interpreter would
# otherwise satisfy the -x test and fail the install again on every retry.
if [ -x "$venv/bin/python" ] && ! "$venv/bin/python" -c "import sys; raise SystemExit(
    0 if tuple(map(int, '$python_version'.split('.'))) <= sys.version_info[:2] else 1)"; then
  echo "Existing venv is $("$venv/bin/python" -V); rebuilding on $python_version."
  rm -rf "$venv"
fi
if [ ! -x "$venv/bin/python" ]; then
  uv venv --python "$python_version" "$venv"
fi
"$venv/bin/python" -V
# Load-bearing for resolution, not just CUDA: without the overrides DPPO's own
# pins conflict with this package and uv declares the requirements unsatisfiable.
uv pip install --python "$venv/bin/python" \
  --overrides config/diffusion_policy/torch-rtx5090.txt \
  -e py -e third_party/dppo

# CUDA forward compatibility is a data-center-GPU feature; on a GeForce card the
# image's compat libcuda fails every call with error 804 and ldconfig resolves to
# it ahead of the host driver.
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GeForce; then
  for compat in /usr/local/cuda*/compat; do
    if [ -d "$compat" ]; then
      mv "$compat" "$compat.disabled"
      echo "Disabled CUDA forward-compat libraries at $compat (unsupported on GeForce)."
    fi
  done
  ldconfig
fi

if ! "$venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Installed torch cannot see the GPU; falling back to the cu128 build."
  uv pip install --python "$venv/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
fi
"$venv/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) cannot see the GPU."
    )
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

mkdir -p "$artifact_root"
aws s3 cp "$base_policy_s3" "$base_policy" --only-show-errors
# The checkpoint checksum comes from whichever source has it. It used to be a
# hardcoded per-epoch map, which meant a checkpoint outside that map could not
# be measured at all; but the older runs the map covers never published one, so
# the map cannot simply be replaced either. Newer training jobs write
# state_<epoch>.pt.sha256 under the run root's job-metadata/, so prefer that,
# fall back to the map, and refuse to run unverified -- a truncated download
# still loads and still produces plausible-looking success rates.
declare -A known_policy_sha=(
  [dp_blue_cube_1000:1500]="4bbcdc552942456dc76bd2911f57e808da5314505ae15e8581bd3bdcfbb57846"
  [dp_blue_cube_1000:200]="a3c50124d6897aa41951178d8d54f574aac22119c09e668262d28b74c5faa153"
)
base_policy_sha256="${BASE_POLICY_SHA256:-}"
if [ -z "$base_policy_sha256" ]; then
  published="$bucket_root/outputs/${base_run%%/*}/job-metadata/state_$base_epoch.pt.sha256"
  if aws s3 cp "$published" "$base_policy.sha256.published" --only-show-errors 2>/dev/null; then
    base_policy_sha256=$(cut -d' ' -f1 "$base_policy.sha256.published")
    echo "Using published checkpoint checksum from $published."
  else
    base_policy_sha256="${known_policy_sha[${base_run%%/*}:$base_epoch]:-}"
  fi
fi
if [ -z "$base_policy_sha256" ]; then
  echo "No checksum for state_$base_epoch.pt of ${base_run%%/*}; pass BASE_POLICY_SHA256." >&2
  exit 1
fi
echo "$base_policy_sha256  $base_policy" | sha256sum --check

# Dataset exports are published only as <name>.tar.zst; the loose per-file
# directories this used to copy normalization.npz and export.json from no longer
# exist, so the download failed outright. Stream the tarball, verify it, and
# take the two files out of it.
artifact_tarball="$workspace/artifacts/$artifact_name.tar.zst"
if [ ! -f "$artifact_root/normalization.npz" ]; then
  aws s3 cp "$artifact_s3.tar.zst" "$artifact_tarball" --only-show-errors
  aws s3 cp "$artifact_s3.tar.zst.sha256" "$artifact_tarball.sha256.published" --only-show-errors
  echo "$(cut -d' ' -f1 "$artifact_tarball.sha256.published")  $artifact_tarball" | sha256sum --check
  # Only these two members. The archive also carries train.npz, 5.4 GB
  # uncompressed, which this sweep never opens: it instantiates config.model
  # alone and sets DPPO_DATA_DIR to a placeholder purely so hydra can resolve.
  tar --use-compress-program=unzstd -xf "$artifact_tarball" -C "$workspace/artifacts" \
    "$artifact_name/normalization.npz" "$artifact_name/export.json"
  rm -f "$artifact_tarball"
fi
for required in normalization.npz export.json; do
  if [ ! -f "$artifact_root/$required" ]; then
    echo "artifact $artifact_name is missing $required after extraction" >&2
    exit 1
  fi
done
cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"
sha256sum "$artifact_root/normalization.npz" \
  | tee "$output_root/job-metadata/normalization-sha256.txt"

"$venv/bin/python" py/scripts/render_apriltag_textures.py --all-defaults
# The three machine-local calibrations are absent from a fresh clone but backed
# up in the bucket, so fetch them rather than failing and making the caller do
# it by hand.
for calibration in config/camera_extrinsics/overhead_camera.json \
                   config/camera_intrinsics/overhead_camera.json \
                   config/camera_intrinsics/wrist_camera.json; do
  if [ ! -f "$repo/$calibration" ]; then
    mkdir -p "$(dirname "$repo/$calibration")"
    aws s3 cp "$bucket_root/config-backup/${calibration#config/}" \
      "$repo/$calibration" --only-show-errors
    echo "Fetched $calibration from config-backup."
  fi
done

export DPPO_DATA_DIR="$artifact_root"
export DPPO_BASE_POLICY="$base_policy"
export DPPO_LOG_DIR="$output_root"
export PYTHONPATH="$repo/third_party/dppo"

# Chunk lengths to measure, longest first. act_steps is an inference-time choice
# -- the network still predicts horizon_steps actions either way -- so the same
# weights are measured at each. 8 is the deployed setting and is re-run here as
# an in-job control rather than compared against an earlier pod's numbers:
# outcomes on this task are chaotically marginal, so a control that shares the
# machine, the driver and the torch build is worth its four minutes.
#
# At 10 Hz, act_steps 8 commits the policy to 0.8 s of open-loop motion per
# query, which spans the entire gripper close. If "contacted but never lifted"
# -- 55-90% of failures in every reach band -- is open-loop commitment rather
# than perception, it should fall as the chunk shortens.
act_steps_list="${ACT_STEPS_LIST:-8 4 2}"

for act_steps in $act_steps_list; do
  echo "=== act_steps=$act_steps, deterministic, $repeats repeats over $scenes scenes"
  "$venv/bin/python" py/scripts/scene_difficulty_sweep.py \
    --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \
    --checkpoint "$base_policy" \
    --normalization "$artifact_root/normalization.npz" \
    --scenes "$scenes" \
    --repeats "$repeats" \
    --n-envs "$n_envs" \
    --scene-seed-base "$scene_seed_base" \
    --act-steps "$act_steps" \
    --deterministic \
    --device cuda:0 \
    --output "$output_root/sweep-deterministic-act$act_steps.json"
  aws s3 sync "$output_root" "$output_prefix" --only-show-errors
done

if [ "${RUN_STOCHASTIC:-false}" = "true" ]; then
  # The exploration distribution PPO samples from, at the deployed chunk length.
  "$venv/bin/python" py/scripts/scene_difficulty_sweep.py \
    --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \
    --checkpoint "$base_policy" \
    --normalization "$artifact_root/normalization.npz" \
    --scenes "$scenes" \
    --repeats "$repeats" \
    --n-envs "$n_envs" \
    --scene-seed-base "$scene_seed_base" \
    --sampling-std "$sampling_std" \
    --device cuda:0 \
    --output "$output_root/sweep-stochastic.json"
fi

echo "Scene-difficulty sweep complete."
