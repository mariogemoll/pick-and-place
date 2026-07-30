#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Run entirely on a disposable Vast.ai instance. Credentials and VAST_INSTANCE_ID
# are installed by the launcher; this job syncs durable results before destroying
# that instance.

set -euo pipefail

# Each re-render worker owns one MuJoCo context. Keep native math and OpenCV
# libraries from multiplying those processes into large hidden thread pools.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

: "${VAST_INSTANCE_ID:?}"
: "${VAST_API_KEY_FILE:?}"
job_mode="${JOB_MODE:-all}"
destroy_on_exit="${DESTROY_ON_EXIT:-true}"

bucket_root="s3://allyouneed/pick-and-place"
source_prefix="$bucket_root/datasets/sim-200_episodes"
artifact_name="blue-cube-no-dr-200-10hz-96x96"
artifact_prefix="$bucket_root/diffusion-policy-data"
run_name="dp_blue_cube_no_dr_unet_1000e_20260730"
job_output_name="${JOB_OUTPUT_NAME:-$run_name}"
output_prefix="$bucket_root/outputs/$job_output_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
source_all="$workspace/datasets/sim-200_episodes-all"
source_selected="$workspace/datasets/sim-200_episodes-selected"
rerender_root="$workspace/datasets/blue-cube-no-dr"
finalized_root="$rerender_root/blue-cube"
artifact_root="$workspace/artifacts/$artifact_name"
output_root="$workspace/outputs/$job_output_name"
job_log="$workspace/job.log"
status_file="$workspace/job-status.json"

mkdir -p "$workspace/outputs" "$workspace/artifacts"
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
  mkdir -p "$output_root/job-metadata"
  cp "$job_log" "$output_root/job-metadata/job.log"
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

  if [ "$destroy_on_exit" != true ]; then
    echo "Final S3 sync verified; DESTROY_ON_EXIT=$destroy_on_exit, leaving instance running."
    exit "$status"
  fi

  api_key=$(<"$VAST_API_KEY_FILE")
  response=$(curl --fail --silent --show-error --request DELETE \
    "https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/" \
    --header "Authorization: Bearer $api_key")
  echo "Vast.ai destroy response: $response"
  exit "$status"
}
trap finalize EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl ffmpeg git libgl1 libglfw3 libosmesa6-dev tmux unzip zstd
if ! command -v aws >/dev/null; then
  aws_install_dir=$(mktemp -d)
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$aws_install_dir/awscliv2.zip" \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
  unzip -tq "$aws_install_dir/awscliv2.zip" >/dev/null
  unzip -q "$aws_install_dir/awscliv2.zip" -d "$aws_install_dir"
  "$aws_install_dir/aws/install" --update
  rm -rf "$aws_install_dir"
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

aws sts get-caller-identity --query Account --output text
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
fi

if [ ! -d "$repo/.git" ]; then
  git clone --recurse-submodules https://github.com/mariogemoll/pick-and-place.git "$repo"
fi
cd "$repo"
git submodule update --init --recursive
git rev-parse HEAD

venv="$workspace/venvs/pick-and-place"
base_python="python3"
if [ -x /venv/main/bin/python ]; then
  base_python="/venv/main/bin/python"
fi
if [ ! -x "$venv/bin/python" ]; then
  uv venv --python "$base_python" "$venv"
fi
if [ "$job_mode" = dataset ]; then
  uv pip install --python "$venv/bin/python" -e py
else
  uv pip install --python "$venv/bin/python" \
    --overrides config/diffusion_policy/torch-rtx5090.txt \
    -e py -e third_party/dppo
fi
uv pip install --python "$venv/bin/python" --reinstall \
  opencv-python==4.13.0.92 opencv-python-headless==4.13.0.92
"$venv/bin/python" - <<'PY'
import cv2

cv2.setNumThreads(1)
if cv2.getNumThreads() != 1:
    raise RuntimeError(f"OpenCV thread cap did not apply: {cv2.getNumThreads()}")
print(f"Verified OpenCV {cv2.__version__} with {cv2.getNumThreads()} native thread.")
PY

"$venv/bin/python" py/scripts/render_apriltag_textures.py --all-defaults
"$venv/bin/python" - "$repo/assets/apriltags/textures" <<'PY'
import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1])
expected = [
    *(f"tagStandard41h12_{tag_id:05d}_30x30mm_tag20mm.png" for tag_id in range(6)),
    *(f"tagStandard41h12_{tag_id:05d}_100x100mm_tag60mm.png" for tag_id in range(8, 12)),
    *(f"tagStandard41h12_{tag_id:05d}_60x60mm_tag40mm.png" for tag_id in range(12, 16)),
]
for name in expected:
    image = cv2.imread(str(root / name), cv2.IMREAD_COLOR)
    expected_edge = 480 if "100x100mm" in name else 432
    if image is None or image.shape != (expected_edge, expected_edge, 3):
        raise ValueError(f"invalid generated AprilTag texture: {name}")
print(f"Verified {len(expected)} generated AprilTag textures and dimensions.")
PY

mkdir -p "$source_all" "$source_selected"
selected_count=$(find "$source_selected" -mindepth 1 -maxdepth 1 -type d -name 'ep[0-9][0-9][0-9][0-9][0-9][0-9]' | wc -l)
if [ "$selected_count" -ne 200 ] || [ ! -f "$source_selected/EPISODES_IN_DATASET.txt" ]; then
  aws s3 sync "$source_prefix" "$source_all" --only-show-errors
  mapfile -t episode_ids < "$source_all/EPISODES_IN_DATASET.txt"
  if [ "${#episode_ids[@]}" -ne 200 ] || [ "$(printf '%s\n' "${episode_ids[@]}" | sort -u | wc -l)" -ne 200 ]; then
    echo "The canonical episode manifest must contain exactly 200 unique IDs." >&2
    exit 1
  fi
  for episode_id in "${episode_ids[@]}"; do
    if [[ ! "$episode_id" =~ ^ep[0-9]{6}$ ]] || [ ! -d "$source_all/$episode_id" ]; then
      echo "Invalid or missing canonical episode: $episode_id" >&2
      exit 1
    fi
    mv "$source_all/$episode_id" "$source_selected/$episode_id"
  done
  cp "$source_all/EPISODES_IN_DATASET.txt" "$source_selected/EPISODES_IN_DATASET.txt"
  if [ -f "$source_all/collection.json" ]; then
    cp "$source_all/collection.json" "$source_selected/collection.json"
  fi
fi

"$venv/bin/python" - "$source_selected" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
episodes = sorted(root.glob("ep[0-9][0-9][0-9][0-9][0-9][0-9]"))
if len(episodes) != 200:
    raise ValueError(f"expected 200 selected episodes, found {len(episodes)}")
for episode in episodes:
    metadata = list((episode / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if len(metadata) != 1:
        raise ValueError(f"{episode.name}: expected one metadata parquet, found {len(metadata)}")
    columns = set(pq.read_schema(metadata[0]).names)
    forbidden = sorted(
        column for column in columns
        if column == "domain_sample_json" or column.startswith("injected_offset_")
    )
    if forbidden:
        raise ValueError(f"{episode.name}: randomized/miscalibrated provenance: {forbidden}")
print("Verified 200 canonical source episodes with no randomization metadata.")
PY

export MUJOCO_GL=osmesa
"$venv/bin/python" py/scripts/rerender_episodes.py \
  --episodes-root "$source_selected" \
  --output "$rerender_root" \
  --verify --max-episodes 3
"$venv/bin/python" py/scripts/rerender_episodes.py \
  --episodes-root "$source_selected" \
  --output "$rerender_root" \
  --variant blue-cube --workers 20

"$venv/bin/python" - "$rerender_root/blue-cube_episodes/rerender.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if manifest.get("variant") != "blue-cube":
    raise ValueError("rerender manifest is not the blue-cube variant")
if manifest.get("camera_randomization") is not None:
    raise ValueError("camera randomization was enabled")
if manifest.get("background_randomization") is not None:
    raise ValueError("background randomization was enabled")
if len(manifest.get("episodes", [])) != 200:
    raise ValueError("rerender manifest does not contain exactly 200 episodes")
if manifest.get("verified_against") is None:
    raise ValueError("rerender was not gated by an environment verification")
print("Verified blue-cube rerender manifest: 200 episodes, no domain randomization.")
PY

"$venv/bin/python" py/scripts/pick_and_place/finalize_sim_dataset.py \
  --dataset-root "$finalized_root" --episodes 200 --keep-episodes --write
"$venv/bin/python" py/scripts/export_diffusion_policy_dataset.py \
  --src "$finalized_root" --output "$artifact_root" \
  --image-size 96 --policy-hz 10 --workers 16
cp "$rerender_root/blue-cube_episodes/rerender.json" "$artifact_root/source-rerender.json"
cp "$source_selected/EPISODES_IN_DATASET.txt" "$artifact_root/source-episodes.txt"

"$venv/bin/python" - "$artifact_root" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
rerender = json.loads((root / "source-rerender.json").read_text())
if rerender["camera_randomization"] is not None or rerender["background_randomization"] is not None:
    raise ValueError("artifact provenance contains domain randomization")
with zipfile.ZipFile(root / "train.npz") as archive:
    expected = {"states.npy", "actions.npy", "images.npy", "traj_lengths.npy"}
    if set(archive.namelist()) != expected:
        raise ValueError(f"unexpected train.npz members: {archive.namelist()}")
    if any(item.compress_type for item in archive.infolist()):
        raise ValueError("train.npz members must be ZIP_STORED")
export = json.loads((root / "export.json").read_text())
if export["num_episodes"] != 200 or export["image_size"] != [96, 96] or export["fps"] != 10:
    raise ValueError("export manifest does not match the requested training contract")
export["source_rerender_manifest"] = "source-rerender.json"
export["source_rerender_sha256"] = hashlib.sha256(
    (root / "source-rerender.json").read_bytes()
).hexdigest()
export["source_episode_manifest"] = "source-episodes.txt"
export["source_episode_manifest_sha256"] = hashlib.sha256(
    (root / "source-episodes.txt").read_bytes()
).hexdigest()
(root / "export.json").write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")
print("Verified final Diffusion Policy dataset contract and provenance.")
PY

archive="$workspace/artifacts/$artifact_name.tar.zst"
tar --use-compress-program='zstd -T0 -10' -cf "$archive" \
  -C "$(dirname "$artifact_root")" "$(basename "$artifact_root")"
(cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
aws s3 cp "$archive" "$artifact_prefix/$(basename "$archive")" --only-show-errors
aws s3 cp "$archive.sha256" "$artifact_prefix/$(basename "$archive").sha256" --only-show-errors
aws s3 sync "$artifact_root" "$artifact_prefix/$artifact_name" --only-show-errors

remote_checksum=$(aws s3 cp "$artifact_prefix/$(basename "$archive").sha256" - --only-show-errors)
local_checksum=$(<"$archive.sha256")
if [ "$remote_checksum" != "$local_checksum" ]; then
  echo "Uploaded artifact checksum sidecar differs from the local sidecar." >&2
  exit 1
fi

if [ "$job_mode" = dataset ]; then
  mkdir -p "$output_root/job-metadata"
  cp "$archive.sha256" "$output_root/job-metadata/artifact.sha256"
  echo "Dataset preparation and verified S3 upload complete."
  exit 0
fi

mkdir -p "$output_root"
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
test -f "$output_root"/pretrain/*/*/checkpoint/state_1000.pt
