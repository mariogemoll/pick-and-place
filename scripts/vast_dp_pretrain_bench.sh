#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Measure Diffusion Policy pre-training throughput on one GPU, one change at a
# time.
#
# DPPO_CLOSED_LOOP_STALL_HANDOFF.md listed four untested speedups and warned
# that `nvidia-smi utilization.gpu` cannot tell you which one matters. This
# sweep settles it: every variant below differs from the one above it by a
# single switch, so a number that fails to move is attributable rather than
# mysterious.
#
# The first epoch of each variant pays for torch.compile and cudnn autotuning
# and is excluded; the reported figure is the median of the rest.
#
# Usage (on the pod, after provisioning):
#   scripts/vast_dp_pretrain_bench.sh <artifact-dir> <output-dir> [epochs]

set -euo pipefail

artifact_root="${1:?usage: vast_dp_pretrain_bench.sh <artifact-dir> <output-dir> [epochs]}"
output_root="${2:?usage: vast_dp_pretrain_bench.sh <artifact-dir> <output-dir> [epochs]}"
epochs="${3:-4}"

repo="${REPO:-/workspace/pick-and-place}"
venv="${VENV:-/workspace/venvs/pick-and-place}"
python="$venv/bin/python"

export DPPO_DATA_DIR="$artifact_root"
export PYTHONPATH="$repo/third_party/dppo"
export PYTHONUNBUFFERED=1
# One MuJoCo-free process; keep BLAS from spawning a thread per core.
export OMP_NUM_THREADS=8

mkdir -p "$output_root"
results="$output_root/bench-results.json"

# name|config|extra hydra overrides
#
# The two stock rows are the "before": the first is literally the
# dp_blue_cube_1000 recipe, the second adds the one tweak the handoff doc had
# already measured (dataset in VRAM, worth ~11% on a 5080).
variants=(
  "stock-cpu-data|pretrain_so101_unet_img|train_dataset.device=cpu"
  "stock-gpu-data|pretrain_so101_unet_img|train_dataset.device=cuda:0"
  "fast-gather|pretrain_so101_unet_img_fast|train.speed.bf16=false train.speed.compile=false train.speed.channels_last=false train.speed.fused_optimizer=false train.speed.batched_vision=false train.speed.plain_attention=false"
  "fast-vision|pretrain_so101_unet_img_fast|train.speed.bf16=false train.speed.compile=false train.speed.channels_last=false train.speed.fused_optimizer=false"
  "fast-bf16|pretrain_so101_unet_img_fast|train.speed.compile=false train.speed.channels_last=false train.speed.fused_optimizer=false"
  "fast-chlast|pretrain_so101_unet_img_fast|train.speed.compile=false train.speed.fused_optimizer=false"
  "fast-fused|pretrain_so101_unet_img_fast|train.speed.compile=false"
  "fast-all-b64|pretrain_so101_unet_img_fast|"
  "fast-all-b128|pretrain_so101_unet_img_fast|train.batch_size=128"
  "fast-all-b256|pretrain_so101_unet_img_fast|train.batch_size=256"
  "fast-all-b512|pretrain_so101_unet_img_fast|train.batch_size=512"
  "fast-all-b1024|pretrain_so101_unet_img_fast|train.batch_size=1024"
)

echo "[]" > "$results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$output_root/gpu.txt"

for entry in "${variants[@]}"; do
  IFS='|' read -r name config overrides <<< "$entry"
  log="$output_root/$name.log"
  run_dir="$output_root/runs/$name"
  mkdir -p "$run_dir"
  echo "=== $name ($config) ==="

  export DPPO_LOG_DIR="$run_dir"
  set +e
  # shellcheck disable=SC2086
  ( cd "$repo/third_party/dppo" && "$python" script/run.py \
      --config-path "$repo/config/diffusion_policy" \
      --config-name "$config" \
      wandb=null \
      train.n_epochs="$epochs" \
      $overrides ) > "$log" 2>&1
  status=$?
  set -e

  "$python" - "$results" "$name" "$config" "$overrides" "$log" "$status" <<'PY'
import json
import re
import sys

results_path, name, config, overrides, log_path, status = sys.argv[1:7]
text = open(log_path, errors="replace").read()
# Both the stock agent and the fast one log "<epoch>: train loss X | t:Y".
epochs = [
    (int(epoch), float(loss), float(seconds))
    for epoch, loss, seconds in re.findall(
        r"^\[?.*?(\d+): train loss\s+([-\d.eE+]+) \| t:\s*([\d.]+)", text, re.M
    )
]
record = {
    "name": name,
    "config": config,
    "overrides": overrides.strip(),
    "exit_status": int(status),
    "epochs": len(epochs),
}
if epochs:
    # Epoch 1 carries compile and autotune cost; steady state is the rest.
    steady = sorted(seconds for _, _, seconds in epochs[1:]) or [epochs[0][2]]
    record["first_epoch_seconds"] = round(epochs[0][2], 3)
    record["median_epoch_seconds"] = round(steady[len(steady) // 2], 3)
    record["epoch_seconds"] = [round(seconds, 3) for _, _, seconds in epochs]
    record["losses"] = [round(loss, 5) for _, loss, _ in epochs]
    record["projected_hours_1500_epochs"] = round(
        steady[len(steady) // 2] * 1500 / 3600.0, 2
    )
else:
    tail = [line for line in text.strip().splitlines() if line.strip()][-3:]
    record["error_tail"] = tail

data = json.load(open(results_path))
data.append(record)
json.dump(data, open(results_path, "w"), indent=2)
print(json.dumps(record, indent=2))
PY

  if [ "$status" -ne 0 ]; then
    echo "!!! $name exited $status; continuing so one failure does not lose the sweep." >&2
  fi
  # Free VRAM between variants.
  sleep 2
done

echo
echo "=== summary ==="
"$python" - "$results" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
baseline = next(
    (r["median_epoch_seconds"] for r in data if r["name"] == "stock-cpu-data" and "median_epoch_seconds" in r),
    None,
)
header = f"{'variant':<18}{'s/epoch':>10}{'speedup':>10}{'h/1500ep':>10}  first"
print(header)
print("-" * len(header))
for record in data:
    if "median_epoch_seconds" not in record:
        print(f"{record['name']:<18}{'FAILED':>10}")
        continue
    seconds = record["median_epoch_seconds"]
    speedup = f"{baseline / seconds:.1f}x" if baseline else "-"
    print(
        f"{record['name']:<18}{seconds:>10.2f}{speedup:>10}"
        f"{record['projected_hours_1500_epochs']:>10.2f}"
        f"  {record['first_epoch_seconds']:.1f}s"
    )
PY
