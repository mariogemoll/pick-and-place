#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Re-measure what batch size is worth to SmolVLA throughput, now that the frozen
# vision tower is out of the step.
#
# `SMOLVLA_PERFORMANCE.md` concluded that batch size is not a throughput lever
# above 32. That was measured when the tower was 65% of a step and saturated the
# GPU at any batch, and the tower is gone: it is cached, and the frozen prefix is
# out of the backward. Two signs the curve has changed shape -- the prefix split
# is 0.74x at batch 8 against 1.51x at batch 64, which is a step that is now
# launch-bound at the small end, and peak VRAM at batch 64 fell from 11-12 GB to
# ~4.4 GB compiled, so batch 256 fits where 128 used to be OOM-killed.
#
# The sweep runs eager on synthetic batches first -- minutes per arm, and no
# dataloader to confound it -- compiles only the two fastest, because
# max-autotune costs 20-30 minutes per shape, and then confirms the candidates
# through `benchmark_live_step.py`, which is the only arm that pays for the cache
# read. That read scales with batch (240 KiB per sample: 15 MB a step at 64
# against 61 MB at 256) and a synthetic sweep cannot see it.
#
# Every arm runs on one host in one session, which is the only way they are
# comparable: marketplace hosts of identical advertised specification have
# measured 1.68x apart.
#
# Launch:
#   scp scripts/vast_smolvla_batch_sweep.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'bash /workspace/vast_smolvla_batch_sweep.sh'
#
# vast_pap_provision.sh must have run first.
#
# Note that changing batch size makes new runs non-comparable to the ten rungs
# already scored, so what this measures is the speed question only; whether to
# change the recipe is a separate one.

set -euo pipefail

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

bucket_root="s3://allyouneed/pick-and-place"
artifact_name="${ARTIFACT_NAME:-two-variant-1000-as-recorded-512x512-lerobot}"
run_name="${RUN_NAME:-smolvla-batch-sweep}"
output_prefix="$bucket_root/outputs/$run_name"

stages="${STAGES:-1 2 3 4}"
has_stage() { case " $stages " in *" $1 "*) return 0;; *) return 1;; esac; }

# 16 is carried over from the superseded sweep as the one size it called
# genuinely bad, so the new curve can say whether that still holds.
batch_sizes="${BATCH_SIZES:-16 32 64 128 256}"
# Empty means "the two fastest eager arms", decided from stage 1 rather than
# assumed, because which they are is the question.
compile_batch_sizes="${COMPILE_BATCH_SIZES:-}"
live_batch_sizes="${LIVE_BATCH_SIZES:-}"
episodes="${EPISODES:-100}"
num_workers="${NUM_WORKERS:-8}"
repeats="${REPEATS:-30}"
live_steps="${LIVE_STEPS:-60}"

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
results="$workspace/batch-sweep-results"
cache_dir="$workspace/prefix-cache-subset"
checkpoint_dir="$workspace/smolvla_base_pinned"
checkpoint_revision="${CHECKPOINT_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
job_log="$workspace/batch-sweep.log"

mkdir -p "$results"
exec > >(tee -a "$job_log") 2>&1

finalize() {
  status=$?
  trap - EXIT
  set +e
  cp "$job_log" "$results/batch-sweep.log"
  aws s3 sync "$results" "$output_prefix" --no-follow-symlinks --only-show-errors \
    && echo "Results synced to $output_prefix" \
    || echo "Result sync failed; the pod is still up and $results still holds them."
  exit "$status"
}
trap finalize EXIT

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$results/gpu.txt"
grep -m1 'model name' /proc/cpuinfo | tee "$results/cpu.txt"
# The container's real core limit, not the host's: nproc, free and
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

# A batch size that does not fit is an answer, not a failure: the sweep records
# it and carries on rather than taking the remaining arms down with it.
run_arm() {
  label="$1"
  shift
  echo "--- $label"
  if "$@"; then
    return 0
  fi
  echo "ARM_FAILED $label (out of memory, or worse -- see the log above)" \
    | tee -a "$results/failed-arms.txt"
  return 0
}

if has_stage 1; then
  echo "=== Stage 1: eager, synthetic batches, the step a run takes today ==="
  # `--language-padding longest` and `--frozen-prefix` throughout: that is the
  # configuration training runs in now, and a curve measured in any other one
  # would be a curve for a configuration nothing uses.
  for batch in $batch_sizes; do
    run_arm "eager batch $batch" \
      "$venv/bin/python" "$repo/py/scripts/benchmark_smolvla_step.py" \
      --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
      --arms cached --batch-size "$batch" --repeats "$repeats" \
      --frozen-prefix --language-padding longest \
      --output "$results/eager-b$batch.json"
  done

  echo "=== Stage 1b: the same batches with the prefix split off, for the ratio ==="
  # The split is 0.74x at batch 8 and 1.51x at batch 64, so it is itself a
  # function of batch size. Measuring both arms at every size is what says
  # whether the split's gain keeps growing or has already topped out.
  for batch in $batch_sizes; do
    run_arm "eager unsplit batch $batch" \
      "$venv/bin/python" "$repo/py/scripts/benchmark_smolvla_step.py" \
      --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
      --arms cached --batch-size "$batch" --repeats "$repeats" \
      --language-padding longest \
      --output "$results/eager-unsplit-b$batch.json"
  done
fi

# The candidates for the expensive stages: whatever stage 1 found fastest in
# samples/s, which is the only comparison that survives a changing batch size.
pick_fastest() {
  count="$1"
  "$venv/bin/python" - "$results" "$count" <<'PY'
import json
import sys
from pathlib import Path

results, count = Path(sys.argv[1]), int(sys.argv[2])
scored = []
for path in results.glob("eager-b*.json"):
    payload = json.loads(path.read_text())
    arm = payload["arms"].get("cached")
    if arm:
        scored.append((arm["samples_per_s"], payload["batch_size"]))
scored.sort(reverse=True)
print(" ".join(str(batch) for _, batch in scored[:count]))
PY
}

if [ -z "$compile_batch_sizes" ]; then
  compile_batch_sizes=$(pick_fastest 2)
fi
if [ -z "$live_batch_sizes" ]; then
  live_batch_sizes="$compile_batch_sizes"
fi
echo "compile: [$compile_batch_sizes]   live: [$live_batch_sizes]" \
  | tee "$results/candidates.txt"

if has_stage 2; then
  echo "=== Stage 2: torch.compile, only the candidates ==="
  # max-autotune spends 20-30 minutes compiling per shape before the first step,
  # which is why this is two arms and not five. The warmup absorbs it, so the
  # timed steps do not.
  for batch in $compile_batch_sizes; do
    run_arm "compiled batch $batch" \
      "$venv/bin/python" "$repo/py/scripts/benchmark_smolvla_step.py" \
      --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
      --arms cached --batch-size "$batch" --repeats "$repeats" \
      --compile --frozen-prefix --language-padding longest \
      --output "$results/compiled-b$batch.json"
  done
fi

if has_stage 3 && [ ! -d "$cache_dir" ]; then
  echo "=== Stage 3: build a prefix cache over $episodes episodes, for the live arm ==="
  cache_start=$(date +%s)
  "$venv/bin/python" "$repo/py/scripts/precompute_smolvla_prefix.py" \
    --dataset "$dataset_root" --checkpoint "$checkpoint_dir" --output "$cache_dir" \
    --batch-size 64 --num-workers "$num_workers" \
    --episodes $(seq 0 $((episodes - 1))) \
    2>&1 | tee "$results/precompute.log"
  echo "cache build seconds: $(( $(date +%s) - cache_start ))" | tee "$results/precompute-seconds.txt"
  du -sh "$cache_dir" | tee "$results/cache-size.txt"
fi

if has_stage 4; then
  echo "=== Stage 4: live steps, which are the ones that pay for the cache read ==="
  # 240 KiB per sample comes off disk every step, so the read scales with batch
  # where the synthetic arm's device-resident tensors do not. `synthetic` beside
  # `live` in one process is what sizes that: the gap between them is everything
  # a real batch costs that a fabricated one does not.
  for batch in $live_batch_sizes; do
    run_arm "live batch $batch" \
      "$venv/bin/python" "$repo/py/scripts/benchmark_live_step.py" \
      --checkpoint "$checkpoint_dir" --dataset "$dataset_root" \
      --prefix-cache "$cache_dir" --arms synthetic live \
      --batch-size "$batch" --num-workers "$num_workers" \
      --episodes "$episodes" --steps "$live_steps" \
      --frozen-prefix --pad-longest \
      --output "$results/live-b$batch.json"
  done
fi

echo "=== Summary ==="
"$venv/bin/python" - "$results" <<'PY' | tee "$results/summary.txt"
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])


def load(pattern: str) -> dict[int, dict]:
    found = {}
    for path in results.glob(pattern):
        payload = json.loads(path.read_text())
        found[payload["batch_size"]] = payload
    return found


eager = load("eager-b*.json")
unsplit = load("eager-unsplit-b*.json")
compiled = load("compiled-b*.json")
live = load("live-b*.json")

print(f"gpu: {next(iter(eager.values()))['gpu'] if eager else 'unknown'}")
print()
print("synthetic, cached prefix, frozen-prefix split on, language padded to the task string")
print(f"{'batch':>6} {'s/step':>9} {'samples/s':>10} {'peak MiB':>9} "
      f"{'split x':>8} {'compiled s/s':>13} {'compiled MiB':>13}")
for batch in sorted(eager):
    arm = eager[batch]["arms"]["cached"]
    row = f"{batch:6d} {arm['median_s']:9.4f} {arm['samples_per_s']:10.1f} {arm['peak_vram_mib']:9d}"
    other = unsplit.get(batch, {}).get("arms", {}).get("cached")
    row += f" {other['median_s'] / arm['median_s']:8.2f}" if other else f" {'-':>8}"
    hot = compiled.get(batch, {}).get("arms", {}).get("cached")
    row += f" {hot['samples_per_s']:13.1f} {hot['peak_vram_mib']:13d}" if hot else f" {'-':>13} {'-':>13}"
    print(row)

if live:
    print()
    print("live, through the dataloader and the cache read")
    print(f"{'batch':>6} {'synth s/s':>10} {'live s/s':>10} {'data_s':>8} "
          f"{'updt_s':>8} {'peak MiB':>9}")
    for batch in sorted(live):
        arms = live[batch]["arms"]
        real = arms.get("live")
        synth = arms.get("synthetic")
        if not real:
            continue
        synth_rate = f"{synth['samples_per_s']:10.1f}" if synth else f"{'-':>10}"
        print(f"{batch:6d} {synth_rate} {real['samples_per_s']:10.1f} "
              f"{real['data_s']:8.4f} {real['update_s']:8.4f} {real['peak_vram_mib']:9d}")

failed = results / "failed-arms.txt"
if failed.is_file():
    print()
    print(failed.read_text().strip())
PY

echo "Batch sweep complete."
