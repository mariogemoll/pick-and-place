#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Generate a recovery dataset, export it, and behavior-clone on it, on one rented
# pod. Tests whether demonstrations containing a *retry* raise the imitation
# ceiling: see docs/DATA_FLYWHEEL.md, "Sim-only recovery data".
#
#   scp scripts/vast_recovery_dataset.sh overlay.tar.gz ~/.netrc <ssh-host>:/workspace/
#   scp -r ~/.aws <ssh-host>:/root/
#   ssh <ssh-host> 'EPISODES=1000 PERTURBED_FRACTION=0.25 \
#                   RUN_NAME=recovery-1000 bash /workspace/vast_recovery_dataset.sh'
#
# One pod does every stage because they need the same machine anyway: episode
# generation and re-rendering are CPU/GL, the export is CPU, training is the GPU.
# Renting once beats paying setup four times.
#
# Stages are numbered in the log and each is skipped if its output already
# exists, so a run that dies part-way resumes instead of starting over. Set
# STOP_AFTER to end early (e.g. STOP_AFTER=export to stage a dataset without
# training).

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl

episodes="${EPISODES:-1000}"
perturbed_fraction="${PERTURBED_FRACTION:-0.25}"
perturbation_magnitude="${PERTURBATION_MAGNITUDE:-0.022}"
run_name="${RUN_NAME:-recovery-$(date +%Y%m%d_%H%M%S)}"
# Generation is the parallel stage, and the binding resource is GPU memory, not
# cores. Each worker holds its own EGL context with an 8192-square shadow map
# (~268 MB) plus 1920x1080 8x-multisampled offscreen buffers, so roughly 400 MB
# of VRAM each. 96 workers on a 32 GB card needs ~38 GB and every one of them
# dies with "Offscreen framebuffer is not complete, error 0x8cdd" -- measured,
# after a 90-minute run that banked nothing. 24 is proven on a 5090; the
# documented workflow uses 8.
#
# `nproc --all`, not plain `nproc`: coreutils honours OMP_NUM_THREADS, which is
# pinned to 1 above so each worker stays single-threaded. Plain `nproc` therefore
# reports 1 and the default would silently be a one-worker run.
MAX_GL_WORKERS="${MAX_GL_WORKERS:-24}"
workers="${WORKERS:-$(( $(nproc --all) < MAX_GL_WORKERS ? $(nproc --all) : MAX_GL_WORKERS ))}"
# The recorded seed. Fixed by default so a dataset is reproducible from this
# script plus its run name; the perturbation stream is salted off it, so the
# unperturbed episodes are unaffected by the fraction.
seed="${SEED:-20260807}"
stop_after="${STOP_AFTER:-}"

bucket_root="s3://allyouneed/pick-and-place"
workspace="/workspace"
repo="$workspace/pick-and-place"
staging="$workspace/datasets/$run_name"
rerender_root="$workspace/datasets/${run_name}_rerender"
blue_cube_root="$rerender_root/blue-cube"
artifact_name="${run_name}-blue-cube"
artifact_root="$workspace/artifacts/$artifact_name"
output_root="$workspace/outputs/$run_name"
output_prefix="$bucket_root/outputs/$run_name"
job_log="$workspace/recovery-dataset.log"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata" "$workspace/datasets"
exec > >(tee -a "$job_log") 2>&1

stage() { echo; echo "=== [$(date +%H:%M:%S)] stage $*"; }
halt_if_stopping() {
  if [ "$stop_after" = "$1" ]; then
    echo "STOP_AFTER=$1 reached; stopping before the next stage."
    exit 0
  fi
}

stage 0 "provision"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl git unzip zstd libegl1 libgl1 ffmpeg
if ! command -v aws >/dev/null; then
  aws_dir=$(mktemp -d)
  curl -sS --fail --location --retry 3 --retry-all-errors \
    --output "$aws_dir/awscliv2.zip" \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
  unzip -q "$aws_dir/awscliv2.zip" -d "$aws_dir"
  "$aws_dir/aws/install" --update >/dev/null
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
aws sts get-caller-identity --query Account --output text
nvidia-smi --query-gpu=name --format=csv,noheader
echo "cores=$(nproc --all)  workers=$workers"

if [ ! -d "$repo/.git" ]; then
  git clone -q --recurse-submodules https://github.com/mariogemoll/pick-and-place.git "$repo"
fi
cd "$repo"
git submodule update --init --recursive -q
git rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"
if [ -f "$workspace/overlay.tar.gz" ]; then
  tar -xzf "$workspace/overlay.tar.gz" -C "$repo"
  tar -tzf "$workspace/overlay.tar.gz" | tee "$output_root/job-metadata/overlay-files.txt"
  sha256sum "$workspace/overlay.tar.gz" | tee "$output_root/job-metadata/overlay-sha256.txt"
fi

venv="$workspace/venvs/pick-and-place"
# Pinned, not inherited: vastai/pytorch ships CPython 3.10 and this package
# floors at 3.12 (lerobot 0.5.1), so inheriting the image's interpreter fails
# dependency resolution outright.
python_version="${PAP_PYTHON_VERSION:-3.12}"
if [ -x "$venv/bin/python" ] && ! "$venv/bin/python" -c "import sys; raise SystemExit(
    0 if tuple(map(int, '$python_version'.split('.'))) <= sys.version_info[:2] else 1)"; then
  rm -rf "$venv"
fi
[ -x "$venv/bin/python" ] || uv venv --python "$python_version" "$venv"
uv pip install -q --python "$venv/bin/python" \
  --overrides config/diffusion_policy/torch-rtx5090.txt \
  -e py -e third_party/dppo
"$venv/bin/python" -V
V="$venv/bin/python"

# CUDA forward compatibility is a data-center-GPU feature; on a GeForce card the
# image's compat libcuda fails every call with error 804.
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GeForce; then
  for compat in /usr/local/cuda*/compat; do
    [ -d "$compat" ] && mv "$compat" "$compat.disabled" && echo "disabled $compat"
  done
  ldconfig
fi

# Machine-local inputs absent from a fresh clone. The tag textures are generated;
# the camera calibrations are restored from the bucket.
"$V" py/scripts/render_apriltag_textures.py --all-defaults >/dev/null
for calibration in config/camera_extrinsics/overhead_camera.json \
                   config/camera_intrinsics/overhead_camera.json \
                   config/camera_intrinsics/wrist_camera.json; do
  if [ ! -f "$repo/$calibration" ]; then
    mkdir -p "$(dirname "$repo/$calibration")"
    aws s3 cp "$bucket_root/config-backup/${calibration#config/}" \
      "$repo/$calibration" --only-show-errors
  fi
done

stage 1 "generate $episodes episodes, perturbed fraction $perturbed_fraction"
# Counted, not merely "does the directory exist": a run whose workers died leaves
# hundreds of *partial* episode directories that carry no meta/info.json and are
# worth nothing. Skipping on the directory's presence made a failed run look
# finished. Top up until enough episodes are complete, or give up saying so.
count_complete() {
  "$V" - "${staging}_episodes" <<'PY'
import sys
from pathlib import Path

from pick_and_place.data.sim_dataset_staging import find_episode_datasets

root = Path(sys.argv[1])
print(len(find_episode_datasets(root)) if root.is_dir() else 0)
PY
}
for attempt in 1 2 3; do
  complete=$(count_complete)
  echo "complete staged episodes: $complete / $episodes (attempt $attempt)"
  [ "$complete" -ge "$episodes" ] && break
  "$V" py/scripts/pick_and_place/record_sim.py \
    --episodes "$(( episodes - complete ))" \
    --workers "$workers" \
    --seed "$seed" \
    --perturbed-fraction "$perturbed_fraction" \
    --perturbation-magnitude "$perturbation_magnitude" \
    --dataset-root "$staging" \
    --repo-id "local/$run_name" || echo "recorder returned $?; will re-count"
done
complete=$(count_complete)
echo "complete staged episodes: $complete"
if [ "$complete" -lt 1 ]; then
  echo "generation banked nothing; stopping." >&2
  exit 1
fi
halt_if_stopping generate

stage 2a "verify the re-render reproduces the recording"
# The render pass refuses to run without a passing report from the *same*
# environment, and rightly: the camera calibrations are machine-local files and
# the OpenGL backend decides the shading, so a verification made elsewhere is not
# evidence about re-renders made here. It writes verification.json into the
# output directory, which is where the render pass looks.
if [ ! -f "$rerender_root/verification.json" ]; then
  "$V" py/scripts/rerender_episodes.py \
    --episodes-root "${staging}_episodes" \
    --output "$rerender_root" \
    --verify \
    --max-episodes 2
else
  echo "verification report already present; skipping."
fi
cat "$rerender_root/verification.json" | head -40

stage 2b "re-render the blue-cube variant"
if [ ! -d "${blue_cube_root}_episodes" ]; then
  "$V" py/scripts/rerender_episodes.py \
    --episodes-root "${staging}_episodes" \
    --output "$rerender_root" \
    --variant blue-cube \
    --workers "$workers"
else
  echo "blue-cube staging already exists; skipping re-render."
fi
halt_if_stopping rerender

stage 3 "finalize the blue-cube dataset"
if [ ! -d "$blue_cube_root" ]; then
  # The merge size must be the exact number of successful staged episodes:
  # asking for more than exist fails the finalizer. Counted with the same
  # function the finalizer uses, rather than scraped out of its stdout, so the
  # two cannot disagree about what "successful" means.
  successes=$("$V" - "${blue_cube_root}_episodes" <<'PY'
import sys
from pathlib import Path

from pick_and_place.data.dataset_subset import SUCCESS_XY_TOLERANCE_M
from pick_and_place.data.sim_dataset_staging import (
    find_episode_datasets,
    successful_episode_datasets,
)

root = Path(sys.argv[1])
complete = find_episode_datasets(root)
print(len(successful_episode_datasets(complete, SUCCESS_XY_TOLERANCE_M)))
PY
)
  echo "staged blue-cube episodes: $(find "${blue_cube_root}_episodes" -maxdepth 1 -name 'ep*' -type d | wc -l), successful: $successes"
  if [ "$successes" -lt 1 ]; then
    echo "no successful episodes to merge; stopping." >&2
    exit 1
  fi
  "$V" py/scripts/pick_and_place/finalize_sim_dataset.py \
    --dataset-root "$blue_cube_root" \
    --episodes "$successes" \
    --repo-id "local/$artifact_name" \
    --write
else
  echo "finalized dataset already exists; skipping."
fi
halt_if_stopping finalize

stage 4 "export the 96x96 arrays"
if [ ! -f "$artifact_root/train.npz" ]; then
  "$V" py/scripts/export_diffusion_policy_dataset.py \
    --src "$blue_cube_root" \
    --output "$artifact_root" \
    --image-size 96 \
    --workers "$workers"
else
  echo "export already exists; skipping."
fi
ls -la "$artifact_root"
cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"

# How many of the *surviving* episodes carry a fumble. This is the number the
# result must be read against, not the requested fraction: episodes the planner
# could not recover fail their placement and the finalizer drops them, so the
# realised fraction is always below what was asked for.
#
# Read from the LeRobot dataset's episode metadata, not the DP export.json -- the
# exporter records the source fingerprint and array shapes, and carries no
# per-episode metadata at all.
"$V" - "$blue_cube_root" <<'PY' | tee "$output_root/job-metadata/perturbation-census.txt"
import glob
import sys
from collections import Counter

import pandas as pd

root = sys.argv[1]
paths = sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))
if not paths:
    raise SystemExit(f"no episode metadata under {root}/meta/episodes")
frame = pd.concat([pd.read_parquet(path) for path in paths])
kinds = Counter(frame.get("grasp_perturbation_kind", pd.Series(dtype=str)).astype(str))
total = int(len(frame))
print(f"episodes in the finalized dataset: {total}")
for kind, count in kinds.most_common():
    print(f"  {kind:24} {count:>5}  {count / total:.1%}")
if "grasp_perturbation_magnitude_m" in frame:
    perturbed = frame[frame["grasp_perturbation_kind"] != "none"]
    if len(perturbed):
        print(
            f"  magnitude: {perturbed['grasp_perturbation_magnitude_m'].mean() * 1000:.1f} mm mean"
        )
PY
halt_if_stopping export

stage 5 "upload the artifact"
tarball="$workspace/artifacts/$artifact_name.tar.zst"
if ! aws s3 ls "$bucket_root/diffusion-policy-data/$artifact_name.tar.zst" >/dev/null 2>&1; then
  tar --use-compress-program='zstd -3 -T0' \
    -cf "$tarball" -C "$workspace/artifacts" "$artifact_name"
  sha256sum "$tarball" | awk '{print $1"  '"$artifact_name"'.tar.zst"}' \
    > "$tarball.sha256"
  aws s3 cp "$tarball" "$bucket_root/diffusion-policy-data/$artifact_name.tar.zst" --only-show-errors
  aws s3 cp "$tarball.sha256" \
    "$bucket_root/diffusion-policy-data/$artifact_name.tar.zst.sha256" --only-show-errors
  echo "uploaded $artifact_name.tar.zst"
else
  echo "artifact already published; skipping upload."
fi
aws s3 sync "$output_root" "$output_prefix" --only-show-errors
halt_if_stopping upload

stage 6 "behavior-clone at the control's recipe"
# Delegated to the tested training launcher rather than invoking run.py here: it
# already downloads and checksum-verifies the artifact, applies the RTX-5090
# overrides, repairs CUDA on GeForce, refuses to overwrite an existing output
# prefix, and enforces the W&B gate. Reimplementing that would be a second,
# less-tested copy of it.
#
# The recipe is exactly what two-variant-1000-blue-cube-b256-e500-v2 used --
# batch 256, 500 epochs, lr 4e-4, warmup 100 -- so the arms differ only in the
# recovery episodes. Do not "improve" these numbers here; that would forfeit the
# pairing the whole experiment rests on.
# VAST_INSTANCE_ID is required by the training launcher (and by
# vast_scene_difficulty.sh) but never read by either -- vestigial, presumably left
# for a teardown watchdog that was never wired. It still has to be set or the
# launcher aborts on its `:?` guard before doing anything, which is exactly how
# this stage failed once: instantly, an hour into an otherwise finished pipeline.
ARTIFACT_NAME="$artifact_name" \
  RUN_NAME="${TRAIN_RUN_NAME:-$artifact_name-b256-e500}" \
  BATCH_SIZE=256 \
  LEARNING_RATE=4e-4 \
  N_EPOCHS=500 \
  WARMUP_STEPS=100 \
  NETRC="${NETRC:-$HOME/.netrc}" \
  VAST_INSTANCE_ID="${VAST_INSTANCE_ID:-unset-and-unused}" \
  bash "$repo/scripts/vast_diffusion_policy_train_fast.sh"

aws s3 sync "$output_root" "$output_prefix" --only-show-errors
echo
echo "=== [$(date +%H:%M:%S)] done. artifact=$artifact_name outputs=$output_prefix"
