#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Generate a fully randomized simulation dataset on one rented pod, publish the
# staged episodes, the 960x720 master, and a state-only flow-policy export.
#
#   scp scripts/vast_randomized_dataset.sh <ssh-host>:/workspace/
#   scp -r ~/.aws <ssh-host>:/root/
#   ssh <ssh-host> 'EPISODES=1000 RUN_NAME=randomized-1000 \
#                   bash /workspace/vast_randomized_dataset.sh'
#
# Run vast_pap_provision.sh first: this assumes the repo, the venv and the
# generated AprilTag textures.
#
# Every randomization axis is on. Two of them change what the *arm has to do*
# and so can only be applied while the episode is generated:
#
#   --domain-randomization   lighting, materials, background, camera response,
#                            the wrist camera's physical mount, and a
#                            miscalibration draw per episode
#   --overhead-perception    the cube and drop plate are localized by rendering
#                            the overhead camera and running the detector, so
#                            the planner's belief error is an outcome of a
#                            calibration that is slightly wrong rather than a
#                            number added to the truth -- and the arm blocking
#                            its own view becomes a real failure mode
#   --physics-randomization  servo gain and time constant, link mass, surface
#                            friction, joint damping, stiction, droop
#   --perturbed-fraction     a deliberate fumble on a minority of episodes, so
#                            the data contains recoveries and not only clean
#                            first attempts
#
# The appearance axis is *not* baked in. Every staged episode carries a
# trajectory artifact, so any look can be re-rendered later from what is
# published here without recording anything again. That is why stage 3 publishes
# the staging area and not only the finalized dataset.
#
# Stages are numbered and each is skipped if its output exists, so a run that
# dies part-way resumes. Set STOP_AFTER to end early.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl

episodes="${EPISODES:-1000}"
perturbed_fraction="${PERTURBED_FRACTION:-0.25}"
perturbation_magnitude="${PERTURBATION_MAGNITUDE:-0.022}"
perturbation_max_source_radius="${PERTURBATION_MAX_SOURCE_RADIUS:-0.330}"
physics_amount="${PHYSICS_RANDOMIZATION:-0.5}"
domain_preset="${DOMAIN_RANDOMIZATION:-config/domain_randomization/act_mild_v1.json}"
run_name="${RUN_NAME:-randomized-$(date +%Y%m%d_%H%M%S)}"
seed="${SEED:-20260816}"
# Plan the drop against the plate's true pose rather than the overhead estimate.
# Nothing corrects the drop -- the descent servo corrects only the pickup -- so
# that estimate's error lands directly in placement error and caps how
# accurately a cloned policy can ever place. Measured on
# randomized_selection_200_v1: 80/200 -> 121/200 settled placements for the
# scripted expert, and 23.2 mm -> 13.1 mm median error among successes.
# See pick-and-place-docs/EXPERT_DROP_TARGET.md.
ground_truth_drop_target="${GROUND_TRUTH_DROP_TARGET:-}"
record_extra_args=()
if [ -n "$ground_truth_drop_target" ]; then
  record_extra_args+=(--ground-truth-drop-target)
fi
stop_after="${STOP_AFTER:-}"
# Attempts per top-up round. Randomization costs yield -- a drawn arm misses
# placements a nominal one makes, and a scene the overhead camera cannot see is
# resampled -- so asking for exactly `episodes` would always come up short.
attempt_batch="${ATTEMPT_BATCH:-}"

# Generation is the parallel stage and the binding resource is GPU memory, not
# cores. Each worker holds three EGL contexts now, not one: the recording rig at
# 1920x1080 with an 8192-square shadow map, the wrist servo's, and the overhead
# perception renderer's, which is ~1.15 GB apiece measured on a 24 GB card. 20
# workers reached 23.0 of 24.4 GB, which is close enough to the framebuffer
# cliff to be a bad bet on an unattended run; 16 leaves real headroom.
#
# `nproc --all` reports the *host's* cores, not the container's allotment, so
# the cgroup is what bounds it. Plain `nproc` honours OMP_NUM_THREADS, pinned to
# 1 above, and would report a one-worker run.
gl_worker_vram_mb="${GL_WORKER_VRAM_MB:-1150}"
vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
vram_workers=$(( (vram_mb * 80 / 100) / gl_worker_vram_mb ))
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
fi
case "${quota:-}" in
  ''|max|-1) cores=$(nproc --all) ;;
  *)         cores=$(( quota / period )) ;;
esac
workers="${WORKERS:-$(( cores < vram_workers ? cores : vram_workers ))}"

bucket_root="s3://allyouneed/pick-and-place"
workspace="/workspace"
repo="$workspace/pick-and-place"
staging="$workspace/data/$run_name"
master_root="$staging"
attempts_root="$workspace/data/$run_name-attempts"
flow_export="$workspace/artifacts/flow-policy-state-$run_name"
output_root="$workspace/outputs/$run_name"
output_prefix="$bucket_root/outputs/$run_name"
job_log="$workspace/randomized-dataset.log"

mkdir -p "$workspace/artifacts" "$output_root/job-metadata" "$workspace/data"
exec > >(tee -a "$job_log") 2>&1

stage() { echo; echo "=== [$(date +%H:%M:%S)] stage $*"; }
halt_if_stopping() {
  if [ "$stop_after" = "$1" ]; then
    echo "STOP_AFTER=$1 reached; stopping before the next stage."
    exit 0
  fi
}

cd "$repo"
export PATH="/root/.local/bin:$PATH"
export PAP_DATA_ROOT="$workspace/data"

stage 0 "inputs"
git rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"
echo "cores=$cores vram=${vram_mb}MiB workers=$workers episodes=$episodes"
echo "ground_truth_drop_target=${ground_truth_drop_target:-off}"
# The one machine-local input a fresh clone still needs is the tag textures,
# and provisioning generates those.
#
# This deliberately does not restore the rig's camera calibration. A recording
# is built at the authored camera pose, which is the origin domain
# randomization perturbs around, so a dataset collected from a fresh clone is
# the dataset this script describes rather than one centred on whichever rig
# happened to record it. Restoring a measured pose here would move every
# rendered pixel and every localization off that origin, silently and
# unreproducibly.
V="$workspace/venvs/pick-and-place/bin/python"

# Belt and braces. `find_episode_datasets` now judges completeness by the
# episode metadata parquet rather than by info.json, so a corpse no longer
# poisons the tally -- but sweeping it also frees the index for a retry instead
# of leaving a directory that every later pass has to step around.
drop_partial_episodes() {
  local root="${staging}_episodes" removed=0
  [ -d "$root" ] || return 0
  for dir in "$root"/ep*; do
    [ -d "$dir" ] || continue
    if [ ! -d "$dir/meta/episodes" ]; then
      rm -rf "$dir"
      removed=$(( removed + 1 ))
    fi
  done
  [ "$removed" -gt 0 ] && echo "swept $removed partial episode directory(ies)"
  return 0
}

count_complete() {
  "$V" - "${staging}_episodes" <<'PY'
import sys
from pathlib import Path

from pick_and_place.data.sim_dataset_staging import find_episode_datasets

root = Path(sys.argv[1])
print(len(find_episode_datasets(root)) if root.is_dir() else 0)
PY
}

count_successful() {
  "$V" - "${staging}_episodes" <<'PY'
import sys
from pathlib import Path

from pick_and_place.data.dataset_subset import SUCCESS_XY_TOLERANCE_M
from pick_and_place.data.sim_dataset_staging import (
    find_episode_datasets,
    successful_episode_datasets,
)

root = Path(sys.argv[1])
if not root.is_dir():
    print(0)
else:
    print(len(successful_episode_datasets(find_episode_datasets(root), SUCCESS_XY_TOLERANCE_M)))
PY
}

stage 1 "generate until $episodes episodes place successfully"
# Counted, not "does the directory exist": a run whose workers died leaves
# partial directories worth nothing. And counted by *successes*, because that is
# what the finalizer can merge -- an episode that records cleanly but places
# 50 mm out is not one of the thousand.
for attempt in 1 2 3 4 5 6; do
  drop_partial_episodes
  successes=$(count_successful)
  complete=$(count_complete)
  echo "attempt $attempt: $successes successful of $complete complete (want $episodes)"
  [ "$successes" -ge "$episodes" ] && break
  # Size the next round from the yield seen so far rather than a guess, with a
  # floor so a run that has banked nothing still asks for a useful batch.
  if [ -n "$attempt_batch" ]; then
    batch="$attempt_batch"
  elif [ "$complete" -gt 0 ] && [ "$successes" -gt 0 ]; then
    batch=$(( (episodes - successes) * complete / successes + 20 ))
  else
    batch=$(( episodes - successes ))
  fi
  echo "requesting $batch more attempts"
  "$V" py/scripts/pick_and_place/record_sim.py \
    --episodes "$batch" \
    --workers "$workers" \
    --seed "$seed" \
    --domain-randomization "$domain_preset" \
    --overhead-perception \
    --physics-randomization "$physics_amount" \
    --perturbed-fraction "$perturbed_fraction" \
    --perturbation-magnitude "$perturbation_magnitude" \
    --perturbation-max-source-radius "$perturbation_max_source_radius" \
    --episode-timeout 900 \
    --dataset-root "$staging" \
    "${record_extra_args[@]}" \
    --repo-id "local/$run_name" || echo "recorder returned $?; will re-count"
done
drop_partial_episodes
successes=$(count_successful)
echo "successful staged episodes: $successes"
if [ "$successes" -lt 1 ]; then
  echo "generation banked nothing; stopping." >&2
  exit 1
fi
halt_if_stopping generate

stage 2 "finalize the 960x720 master"
merge_count=$(( successes < episodes ? successes : episodes ))
if [ ! -d "$master_root" ]; then
  # --keep-episodes: the staged episodes carry the trajectory artifacts, and
  # they are the only form a later re-render can read. Deleting them would make
  # every future appearance variant a fresh collection run.
  # --attempts-root banks the failures as a loadable dataset beside the master.
  # The success filter drops roughly two attempts in five here, and those are
  # exactly the negatives an offline critic needs -- a value function fitted only
  # to successes regresses a near-constant and cannot rank anything.
  "$V" py/scripts/pick_and_place/finalize_sim_dataset.py \
    --dataset-root "$master_root" \
    --episodes "$merge_count" \
    --repo-id "local/$run_name" \
    --attempts-root "$attempts_root" \
    --keep-episodes \
    --write
else
  echo "finalized master already exists; skipping."
fi
halt_if_stopping finalize

publish() {  # local-path s3-key
  local path="$1" key="$2"
  if aws s3 ls "$key" >/dev/null 2>&1; then
    echo "$key already published; skipping."
    return
  fi
  sha256sum "$path" | awk -v n="$(basename "$key")" '{print $1"  "n}' > "$path.sha256"
  aws s3 cp "$path" "$key" --only-show-errors
  aws s3 cp "$path.sha256" "$key.sha256" --only-show-errors
  # Verify by size rather than by reading the object back. The round-trip hash
  # is the stronger check, but it costs a full download, and S3 egress is not
  # symmetric with ingress on a rented pod: one host uploaded 2.8 GB in seven
  # minutes and then read it back at ~0.4 MB/s, which would have spent six hours
  # verifying four tarballs. What is left still catches the realistic failure --
  # `aws s3 cp` checksums every multipart chunk in transit and S3 rejects a bad
  # part, so truncation is what survives that, and a size check catches
  # truncation. The .sha256 sidecar is published either way, so any consumer can
  # still verify the bytes it actually downloads.
  # head-object, not `s3 ls`: ls matches by prefix, so it also returns the
  # .sha256 sidecar published beside the object and a naive read picks up the
  # sidecar's size instead of the tarball's.
  local local_bytes published_bytes bucket object
  local_bytes="$(stat -c %s "$path")"
  bucket="${key#s3://}"; bucket="${bucket%%/*}"
  object="${key#s3://$bucket/}"
  published_bytes="$(aws s3api head-object --bucket "$bucket" --key "$object" \
    --query ContentLength --output text 2>/dev/null)"
  if [ "$local_bytes" != "$published_bytes" ]; then
    echo "size mismatch publishing $key: local $local_bytes, published $published_bytes" >&2
    exit 1
  fi
  echo "published and verified (size $local_bytes) $key"
}

stage 3 "publish the staged episodes -- the form every other format derives from"
staged_tarball="$workspace/data/$run_name-staged.tar.zst"
[ -f "$staged_tarball" ] || tar --use-compress-program='zstd -3 -T0' \
  -cf "$staged_tarball" -C "$(dirname "${staging}_episodes")" "$(basename "${staging}_episodes")"
publish "$staged_tarball" "$bucket_root/datasets/$run_name-staged.tar.zst"
halt_if_stopping staged

stage 4 "publish the 960x720 master"
master_tarball="$workspace/data/$run_name-lerobot.tar.zst"
[ -f "$master_tarball" ] || tar --use-compress-program='zstd -3 -T0' \
  -cf "$master_tarball" -C "$(dirname "$master_root")" "$(basename "$master_root")"
publish "$master_tarball" "$bucket_root/datasets/$run_name-lerobot.tar.zst"
halt_if_stopping master

stage 4b "publish the attempt dataset -- every episode, failures included"
if [ -d "$attempts_root" ]; then
  attempts_tarball="$workspace/data/$run_name-attempts-lerobot.tar.zst"
  [ -f "$attempts_tarball" ] || tar --use-compress-program='zstd -3 -T0' \
    -cf "$attempts_tarball" -C "$(dirname "$attempts_root")" "$(basename "$attempts_root")"
  publish "$attempts_tarball" "$bucket_root/datasets/$run_name-attempts-lerobot.tar.zst"
else
  echo "no attempt dataset to publish; skipping."
fi
halt_if_stopping attempts

stage 5 "export the state-only flow-policy arrays"
if [ ! -f "$flow_export/train.npz" ]; then
  "$V" py/scripts/export_flow_policy_dataset.py \
    --src "$master_root" \
    --output "$flow_export" \
    --validation-fraction 0.1 \
    --seed 0
fi
ls -la "$flow_export"
cp "$flow_export/export.json" "$output_root/job-metadata/flow-export.json"
flow_tarball="$workspace/artifacts/$(basename "$flow_export").tar.zst"
[ -f "$flow_tarball" ] || tar --use-compress-program='zstd -3 -T0' \
  -cf "$flow_tarball" -C "$workspace/artifacts" "$(basename "$flow_export")"
publish "$flow_tarball" "$bucket_root/flow-policy-data/$(basename "$flow_export").tar.zst"

stage 6 "census: what the randomization actually produced"
"$V" - "$master_root" <<'PY' | tee "$output_root/job-metadata/randomization-census.txt"
import glob
import sys
from collections import Counter

import pandas as pd

root = sys.argv[1]
paths = sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))
if not paths:
    raise SystemExit(f"no episode metadata under {root}/meta/episodes")
frame = pd.concat([pd.read_parquet(path) for path in paths])
total = int(len(frame))
print(f"episodes in the finalized dataset: {total}")

kinds = Counter(frame.get("grasp_perturbation_kind", pd.Series(dtype=str)).astype(str))
print("\ndeliberate fumbles, and so recoveries:")
for kind, count in kinds.most_common():
    print(f"  {kind:24} {count:>5}  {count / total:.1%}")

print("\nrandomization axes, as recorded per episode:")
for column, label, scale in (
    ("physics_mass_scale", "link mass x", 1.0),
    ("physics_friction_scale", "surface friction x", 1.0),
    ("physics_damping_scale", "joint damping x", 1.0),
    ("injected_offset_shoulder_pan_deg", "pan zero (deg)", 1.0),
    ("cube_start_x", "cube start x (m)", 1.0),
):
    if column in frame:
        values = frame[column].astype(float) * scale
        print(f"  {label:24} min {values.min():+.3f}  median {values.median():+.3f}  max {values.max():+.3f}")

if "believed_cube_start_x" in frame and "cube_start_x" in frame:
    miss = ((frame["believed_cube_start_x"] - frame["cube_start_x"]) ** 2
            + (frame["believed_cube_start_y"] - frame["cube_start_y"]) ** 2) ** 0.5
    print(f"\noverhead cube localization error: median {miss.median() * 1000:.1f} mm, "
          f"p90 {miss.quantile(0.9) * 1000:.1f} mm")
PY

aws s3 sync "$output_root" "$output_prefix" --only-show-errors
echo
echo "=== [$(date +%H:%M:%S)] done"
echo "staged      $bucket_root/datasets/$run_name-staged.tar.zst"
echo "master      $bucket_root/datasets/$run_name-lerobot.tar.zst"
echo "flow export $bucket_root/flow-policy-data/$(basename "$flow_export").tar.zst"
