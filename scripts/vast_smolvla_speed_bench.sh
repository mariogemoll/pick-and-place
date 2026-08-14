#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Measure what removing SmolVLA's frozen vision tower from the training step is
# worth, on one rented GPU.
#
# The tower is 59% of a step and its weights never move, so an N-epoch run
# recomputes the same 64x960 tokens per camera N times. Caching them is not an
# overlap trick -- the earlier finding that running the prefix ahead on a side
# stream caps at 1.03x stands, because GPU busy is 96.6% and there is no bubble
# to fill. This removes the work instead.
#
# Everything runs on one host in one session, which is the only way the arms are
# comparable: marketplace hosts of identical advertised specification have
# measured 1.68x apart.
#
# Launch:
#   scp scripts/vast_smolvla_speed_bench.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'bash /workspace/vast_smolvla_speed_bench.sh'
#
# vast_pap_provision.sh must have run first, on a branch carrying
# py/scripts/precompute_smolvla_prefix.py.

set -euo pipefail

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

bucket_root="s3://allyouneed/pick-and-place"
artifact_name="${ARTIFACT_NAME:-two-variant-1000-as-recorded-512x512-lerobot}"
run_name="${RUN_NAME:-smolvla-prefix-cache-bench}"
output_prefix="$bucket_root/outputs/$run_name"

# The end-to-end arm trains on a subset, because a full cache is 72 GB and the
# question it answers -- what a step costs with the tower gone -- does not need
# the whole dataset. 100 episodes is ~29,000 frames and ~7 GB of cache.
episodes="${EPISODES:-100}"
train_steps="${TRAIN_STEPS:-260}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-8}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
results="$workspace/bench-results"
cache_dir="$workspace/prefix-cache-subset"
checkpoint_dir="$workspace/smolvla_base_pinned"
checkpoint_revision="${CHECKPOINT_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
job_log="$workspace/speed-bench.log"

mkdir -p "$results"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  cp "$job_log" "$results/speed-bench.log"
  aws s3 sync "$results" "$output_prefix" --no-follow-symlinks --only-show-errors \
    && echo "Results synced to $output_prefix" \
    || echo "Result sync failed; the pod is still up and $results still holds them."
  exit "$status"
}
trap finalize EXIT

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$results/gpu.txt"
grep -m1 'model name' /proc/cpuinfo | tee "$results/cpu.txt"
# The container's real core and memory limits, not the host's: nproc, free and
# sched_getaffinity all report the machine rather than the slice.
cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true
df -h "$workspace" | tail -1 | tee "$results/disk.txt"

find_dataset() {
  find "$workspace/artifacts" -maxdepth 3 -path '*/meta/info.json' -print -quit 2>/dev/null
}

mkdir -p "$workspace/artifacts"
if [ -z "$(find_dataset)" ]; then
  archive="$bucket_root/datasets/$artifact_name.tar.zst"
  aws s3 cp "$archive" "$workspace/artifacts/$artifact_name.tar.zst" --only-show-errors
  aws s3 cp "$archive.sha256" "$workspace/artifacts/$artifact_name.tar.zst.sha256" --only-show-errors
  (cd "$workspace/artifacts" && sha256sum -c "$artifact_name.tar.zst.sha256")
  tar -x -I zstd -f "$workspace/artifacts/$artifact_name.tar.zst" -C "$workspace/artifacts"
  rm -f "$workspace/artifacts/$artifact_name.tar.zst" "$workspace/artifacts/$artifact_name.tar.zst.sha256"
fi
dataset_root=$(dirname "$(dirname "$(find_dataset)")")
echo "Using dataset at $dataset_root"

"$venv/bin/python" - "$checkpoint_revision" "$checkpoint_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("pinned checkpoint at", snapshot_download("lerobot/smolvla_base", revision=sys.argv[1], local_dir=sys.argv[2]))
PY

echo "=== Stage 0: the cached path must reproduce the stock loss ==="
# A speedup that changes the loss is not a speedup, it is a different model. This
# runs one batch both ways with the flow-matching noise and time held fixed, so
# the only thing that can differ is where the image tokens came from.
"$venv/bin/python" "$repo/py/scripts/check_smolvla_prefix_cache.py" \
  --dataset "$dataset_root" --checkpoint "$checkpoint_dir" \
  --episodes 0 --batch-size 8 --device cuda \
  2>&1 | tee "$results/equivalence.txt"

echo "=== Stage 1: synthetic batches, eager ==="
"$venv/bin/python" "$repo/py/scripts/benchmark_smolvla_step.py" \
  --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
  --arms stock cached tower --batch-size "$batch_size" --profile \
  --output "$results/synthetic-eager.json"

echo "=== Stage 2: synthetic batches, torch.compile ==="
# max-autotune spends minutes compiling before the first step, so the warmup is
# what absorbs it and the timed steps do not. A tqdm running mean would report
# compile as slower; this does not.
"$venv/bin/python" "$repo/py/scripts/benchmark_smolvla_step.py" \
  --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
  --arms stock cached --batch-size "$batch_size" --compile \
  --output "$results/synthetic-compiled.json"

echo "=== Stage 3: LoRA against dense, eager ==="
# Asked because it is the obvious parameter-efficiency lever. It can only touch
# the backward and the optimizer, which the stage-1 profile sizes.
"$venv/bin/python" "$repo/py/scripts/benchmark_smolvla_step.py" \
  --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
  --arms stock lora --batch-size "$batch_size" \
  --output "$results/lora.json" || echo "LoRA arm failed; peft may not be installed."

episode_list=$("$venv/bin/python" -c "import sys; print('[' + ','.join(str(i) for i in range(int(sys.argv[1]))) + ']')" "$episodes")

echo "=== Stage 4: build a cache over $episodes episodes ==="
rm -rf "$cache_dir"
cache_start=$(date +%s)
"$venv/bin/python" "$repo/py/scripts/precompute_smolvla_prefix.py" \
  --dataset "$dataset_root" --checkpoint "$checkpoint_dir" --output "$cache_dir" \
  --batch-size "$batch_size" --num-workers "$num_workers" \
  --episodes $(seq 0 $((episodes - 1))) \
  2>&1 | tee "$results/precompute.log"
echo "cache build seconds: $(( $(date +%s) - cache_start ))" | tee "$results/precompute-seconds.txt"
du -sh "$cache_dir" | tee "$results/cache-size.txt"

common_args=(
  --dataset.repo_id="$artifact_name"
  --dataset.root="$dataset_root"
  --dataset.episodes="$episode_list"
  --policy.type=smolvla
  --policy.pretrained_path="$checkpoint_dir"
  --policy.n_action_steps=10
  --policy.device=cuda
  --policy.push_to_hub=false
  --batch_size="$batch_size"
  --num_workers="$num_workers"
  --seed=1000
  --steps="$train_steps"
  --save_freq="$train_steps"
  --wandb.enable=false
)

export ACCELERATE_MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

echo "=== Stage 5: end to end, stock lerobot-train ==="
rm -rf "$workspace/e2e-stock"
"$venv/bin/lerobot-train" "${common_args[@]}" --output_dir="$workspace/e2e-stock" \
  2>&1 | tee "$results/e2e-stock.log"

echo "=== Stage 6: end to end, cached prefix ==="
rm -rf "$workspace/e2e-cached"
"$venv/bin/python" "$repo/py/scripts/train_smolvla_cached.py" \
  --prefix-cache "$cache_dir" "${common_args[@]}" --output_dir="$workspace/e2e-cached" \
  2>&1 | tee "$results/e2e-cached.log"

echo "=== Summary ==="
"$venv/bin/python" - "$results" <<'PY' | tee "$results/summary.txt"
import json
import re
import statistics
import sys
from pathlib import Path

results = Path(sys.argv[1])


def step_times(log: Path) -> dict[str, float]:
    """Read lerobot's own updt_s and data_s, skipping the warmup steps."""
    text = log.read_text(errors="replace")
    updates = [float(m) for m in re.findall(r"updt_s:([0-9.]+)", text)]
    data = [float(m) for m in re.findall(r"data_s:([0-9.]+)", text)]
    return {
        "updt_s": statistics.median(updates[1:]) if len(updates) > 1 else float("nan"),
        "data_s": statistics.median(data[1:]) if len(data) > 1 else float("nan"),
        "logged_points": len(updates),
    }


for name in ("synthetic-eager", "synthetic-compiled", "lora"):
    path = results / f"{name}.json"
    if not path.is_file():
        continue
    payload = json.loads(path.read_text())
    print(f"[{name}] {payload['gpu']} batch {payload['batch_size']} compile={payload['compile']}")
    for arm, entry in payload["arms"].items():
        line = f"  {arm:6s} {entry['median_s']:.4f}s  {entry['samples_per_s']:.1f} samples/s"
        if "images_per_s" in entry:
            line += f"  {entry['images_per_s']:.0f} images/s"
        print(line)
        if "stages_s" in entry:
            total = sum(entry["stages_s"].values())
            for stage, seconds in entry["stages_s"].items():
                print(f"      {stage:10s} {seconds:.4f}s  {seconds / total:5.1%}")
    if "cached_speedup" in payload:
        print(f"  cached / stock: {payload['cached_speedup']:.3f}x")

for name in ("e2e-stock", "e2e-cached"):
    path = results / f"{name}.log"
    if path.is_file():
        print(f"[{name}] {step_times(path)}")
PY

echo "Bench complete."
