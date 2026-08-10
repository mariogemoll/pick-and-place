#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# PPO fine-tuning of the state flow policy, on a rented GPU. Runs entirely on the
# pod: the launcher installs credentials and VAST_INSTANCE_ID, this script
# provisions, verifies, gates, trains, and syncs to S3. Teardown stays manual,
# after an independent S3 check.
#
#   scp scripts/vast_flow_ppo_finetune.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'VAST_INSTANCE_ID=... SEED=42 bash /workspace/vast_flow_ppo_finetune.sh'
#
# The objective is speed, not success. Under the dense reward an episode pays 1.0
# for every control tick the cube stays settled on target and runs to its budget,
# so the return is `150 - time_to_settle`. The base policy places at a median 81
# ticks with a 0.94 success rate, so the return has real range to move in while
# the success rate has almost none.
#
# Two differences from vast_dppo_finetune.sh worth knowing:
#
#   - nothing renders. A state policy's rollout is MuJoCo physics and a 4.07M
#     parameter U-Net, so this wants vCPUs far more than it wants a large GPU,
#     and it does not need the 48-core machines the visual strand did.
#   - "$@" is expanded LAST, so any hydra override passed on the command line
#     wins. In the visual launcher it lands before the two model.min_* flags,
#     which silently discards a positional override of either.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
# The behavior-cloned flow checkpoint to start from, and the export it was
# trained against. They must always move together: the export's bounds normalize
# what the policy sees and unnormalize what it commands, so pairing a checkpoint
# with another export does not fail, it feeds the policy the wrong units.
base_run_name="${BASE_RUN_NAME:-flow-policy-unet1d-rot6-cubeaug-30k-seed0}"
base_checkpoint_name="${BASE_CHECKPOINT:-checkpoint.pt}"
base_policy_s3="$bucket_root/outputs/$base_run_name/$base_checkpoint_name"
export_name="${EXPORT_NAME:-flow-policy-state-recovery-far-clean-993ep-rot6-cubeaug-val10}"
export_s3="$bucket_root/flow-policy-data/$export_name"
# aws does not verify a multipart download, so a truncated checkpoint surfaces as
# a policy that has silently forgotten the task rather than as a failed transfer.
# The default is the 30,000-update checkpoint the 188/200 selection run chose.
base_policy_sha256="${BASE_POLICY_SHA256:-9ce2818a6c23676fe4c352ddff49ad991e22847548a48d30010dd323c5601247}"

seed="${SEED:-42}"
run_name="${RUN_NAME:-flow_ppo_${base_run_name}_s${seed}_$(date +%Y%m%d)}"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
export_root="$workspace/artifacts/$export_name"
base_policy="$workspace/artifacts/${base_run_name}_${base_checkpoint_name}"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/flow-ppo-finetune.log"
status_file="$workspace/flow-ppo-finetune-status.json"

# Rollout collection is CPU-bound and the update is GPU-bound, so choose roughly
# n_envs vCPUs. The per-iteration log line reports the split.
n_envs="${N_ENVS:-32}"
n_steps="${N_STEPS:-40}"
n_train_itr="${N_TRAIN_ITR:-121}"
# Every eval iteration is a deterministic measurement over n_envs episodes and
# every checkpoint is a candidate for the paired comparison. Checkpoint cadence
# mattered as much as seed count in the visual strand's six-seed matrix: a sweep
# scoring only one iteration would have called three of six seeds failures.
val_freq="${VAL_FREQ:-10}"
save_model_freq="${SAVE_MODEL_FREQ:-10}"
# The braked configuration, which is what turned twelve consecutive collapses
# into runs that at worst end where they started. Step size and step count are
# the only working brakes here: PPO's trust region is provably disengaged on
# this action space, so the clip ratio and target KL cannot be credited.
actor_lr="${ACTOR_LR:-3e-6}"
update_epochs="${UPDATE_EPOCHS:-2}"
max_grad_norm="${MAX_GRAD_NORM:-1.0}"
target_kl="${TARGET_KL:-0.02}"
# Iterations training only the critic before the actor may move. The dense
# reward gives the critic a graded target rather than a Bernoulli, but it still
# starts from scratch, and an actor updated on advantages that are still noise
# degrades a strong policy rather than improving it.
critic_warmup="${CRITIC_WARMUP:-20}"
# Exploration, in normalized action units at the start of the chain; the SDE's
# schedule takes it to zero as the chain ends, so unlike DPPO's constant floor
# it does not land undiminished on the emitted action. Measured on the base
# policy: settled placements 0.875/0.844/0.844/0.812/0.750 at
# 0.01/0.03/0.05/0.1/0.2, against 0.94 deterministic.
sampling_noise_scale="${SAMPLING_NOISE_SCALE:-0.1}"
# The likelihood floor, a different quantity with a different job. Upstream
# clamps every log-probability to at most 2 before the importance ratio, and a
# Gaussian narrower than ~0.054 exceeds that at its peak -- whereupon old and
# new policies score identically however far the policy moved. 0.06 is the
# tightest floor clear of it.
logprob_std="${LOGPROB_STD:-0.06}"
dense_success_reward="${DENSE_SUCCESS_REWARD:-True}"
shaping_weight="${SHAPING_WEIGHT:-0.0}"
debug_action_reward="${DEBUG_ACTION_REWARD:-False}"
# The sampler the fine-tuned chain is defined over. A checkpoint must be scored
# at the values it was trained with.
flow_steps="${FLOW_STEPS:-10}"
ft_denoising_steps="${FT_DENOISING_STEPS:-5}"
preflight_episodes="${PREFLIGHT_EPISODES:-32}"
# The gate. The base policy lifts the cube in nearly every scene, so anything
# near zero means the environment is not showing it the task it was trained on.
min_lift_rate="${MIN_LIFT_RATE:-0.5}"

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
  cp "$job_log" "$output_root/job-metadata/flow-ppo-finetune.log"
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
  echo "Final S3 sync verified; leaving instance running for independent verification."
  exit "$status"
}
trap finalize EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
# libegl1 is what MuJoCo compiles its scene against even with no cameras read.
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
git checkout "${REPO_REF:-HEAD}" 2>/dev/null || true
git submodule update --init --recursive
git rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"

# The vendored image agent is the one both policy families train through,
# because it is the variant that batches an arbitrary dictionary of observation
# keys -- which is what carries the privileged critic's input. It still moves
# only "rgb" and "state" to torch, so the third key arrives as numpy and dies in
# torch.split at the end of the first iteration, after the gate has passed and
# the run looks healthy. dppo-log-ppo-diagnostics.patch must be applied after:
# both touch the same file, and it prints approx_kl, clipfrac, explained
# variance and the advantage distribution, which upstream sends only to W&B.
for patch in dppo-generic-obs-keys dppo-log-ppo-diagnostics; do
  if ! git -C third_party/dppo apply --reverse --check \
       "../../config/diffusion_policy/$patch.patch" 2>/dev/null; then
    git -C third_party/dppo apply "../../config/diffusion_policy/$patch.patch"
    echo "Applied $patch.patch to the vendored agent."
  else
    echo "$patch.patch already applied."
  fi
done

venv="$workspace/venvs/pick-and-place"
base_python="python3"
if [ -x /venv/main/bin/python ]; then
  base_python="/venv/main/bin/python"
fi
if [ ! -x "$venv/bin/python" ]; then
  uv venv --python "$base_python" "$venv"
fi
# The overrides are load-bearing for resolution, not just for CUDA: without them
# DPPO's own pins (torch 2.4, av 12.3, wandb 0.17) conflict with this package and
# uv declares the requirements unsatisfiable.
uv pip install --python "$venv/bin/python" \
  --overrides config/diffusion_policy/torch-rtx5090.txt \
  -e py -e third_party/dppo

# CUDA forward compatibility is a data-center-GPU feature. On a GeForce card the
# image's compat libcuda fails every CUDA call with error 804, and because
# ldconfig resolves to it ahead of the host driver, clearing LD_LIBRARY_PATH does
# not help. Take it out of the loader's path.
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
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) cannot see the GPU. "
        "Training would fall back to CPU or die at the first allocation."
    )
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

# Refuse to overwrite or resume a run. The fine-tuner has no resume path, so a
# name that already exists in S3 means a previous run's artifacts are there.
if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

# The export, minus the training arrays: reinforcement learning needs the
# manifest and the bounds, not the 92,092 behavior-cloning examples.
mkdir -p "$export_root"
for member in export.json normalization.npz; do
  if [ ! -f "$export_root/$member" ]; then
    aws s3 cp "$export_s3/$member" "$export_root/$member" --only-show-errors
  fi
  if [ ! -f "$export_root/$member" ]; then
    echo "$export_s3 does not hold $member" >&2
    exit 1
  fi
done

aws s3 cp "$base_policy_s3" "$base_policy" --only-show-errors
echo "$base_policy_sha256  $base_policy" | sha256sum --check
sha256sum "$export_root/normalization.npz" \
  | tee "$output_root/job-metadata/normalization.sha256"
cp "$export_root/export.json" "$output_root/job-metadata/flow-export.json"
cp config/flow_policy/ft_ppo_so101_flow.yaml "$output_root/job-metadata/launcher-config.yaml"

# Gitignored build artifacts a clean clone lacks. The scene compiles the
# AprilTag textures and the measured camera calibration whether or not any
# camera is read, so both are needed even though this policy sees neither.
"$venv/bin/python" py/scripts/render_apriltag_textures.py --all-defaults
for calibration in config/camera_extrinsics/overhead_camera.json \
                   config/camera_intrinsics/overhead_camera.json \
                   config/camera_intrinsics/wrist_camera.json; do
  if [ ! -f "$repo/$calibration" ]; then
    mkdir -p "$(dirname "$repo/$calibration")"
    aws s3 cp "$bucket_root/config-backup/${calibration#config/}" \
      "$repo/$calibration" --only-show-errors
  fi
  if [ ! -f "$repo/$calibration" ]; then
    echo "missing machine-local calibration on the pod: $calibration" >&2
    exit 1
  fi
done

export DPPO_DATA_DIR="$export_root"
export DPPO_BASE_POLICY="$base_policy"
export DPPO_LOG_DIR="$output_root"
export PYTHONPATH="$repo/third_party/dppo"

# W&B is required, and a key on the controller does nothing: the credential has
# to be on the pod. Every run in this repository's history that degraded to
# wandb=null logged its approx_kl, clipfrac and explained variance to nowhere,
# which is precisely the diagnostic eleven collapsing runs needed.
if [ "${WANDB:-on}" = "off" ]; then
  echo "WANDB=off: training without W&B logging, by request."
  wandb_override=(wandb=null)
elif grep -q api.wandb.ai "${NETRC:-$HOME/.netrc}" 2>/dev/null; then
  wandb_override=()
  echo "W&B credential found on this pod."
else
  echo "No api.wandb.ai entry in ${NETRC:-$HOME/.netrc} on this pod." >&2
  echo "Copy your ~/.netrc to the pod, or set WANDB=off to run without logging." >&2
  exit 1
fi

# Gate: does the base policy still perform the task in this environment, and
# does it still perform it under the exploration noise PPO will sample with?
# The second question is the one that decides whether there is a reward signal:
# too much noise and every episode fails, too little and the scene draw is the
# only source of variance.
"$venv/bin/python" py/scripts/check_flow_rl_env.py \
  --config config/flow_policy/ft_ppo_so101_flow.yaml \
  --checkpoint "$base_policy" \
  --export "$export_root" \
  --episodes "$preflight_episodes" \
  --n-envs "$n_envs" \
  --flow-steps "$flow_steps" \
  --device cuda:0 \
  --output "$output_root/job-metadata/preflight-deterministic.json"

"$venv/bin/python" py/scripts/check_flow_rl_env.py \
  --config config/flow_policy/ft_ppo_so101_flow.yaml \
  --checkpoint "$base_policy" \
  --export "$export_root" \
  --episodes "$preflight_episodes" \
  --n-envs "$n_envs" \
  --flow-steps "$flow_steps" \
  --stochastic --sampling-noise-scale "$sampling_noise_scale" \
  --device cuda:0 \
  --output "$output_root/job-metadata/preflight-stochastic.json"

"$venv/bin/python" - "$output_root/job-metadata" "$min_lift_rate" <<'PY'
import json
import sys
from pathlib import Path

metadata, threshold = Path(sys.argv[1]), float(sys.argv[2])
for name in ("deterministic", "stochastic"):
    summary = json.loads((metadata / f"preflight-{name}.json").read_text())["summary"]
    lifted = summary["rates"]["cube_lifted"]
    settle = summary["control_steps_to_settle"]
    print(
        f"Pre-flight ({name}): lift {lifted:.2f}, success "
        f"{summary['rates']['success']:.2f}, median settle {settle['median']} ticks "
        f"over {summary['episodes']} episodes."
    )
    if lifted < threshold:
        raise SystemExit(
            f"Pre-flight lift rate {lifted:.2f} is below {threshold:.2f} ({name}): the "
            "base policy is not performing the task in this environment. Fine-tuning "
            "would be measuring a broken observation pipeline."
        )
PY

cd "$repo/third_party/dppo"
( while sleep 900; do
    aws s3 sync "$output_root" "$output_prefix" --only-show-errors
  done ) &
sync_pid=$!
set +e
"$venv/bin/python" script/run.py \
  --config-path "$repo/config/flow_policy" \
  --config-name ft_ppo_so101_flow \
  seed="$seed" \
  env.n_envs="$n_envs" \
  train.n_steps="$n_steps" \
  train.n_train_itr="$n_train_itr" \
  train.val_freq="$val_freq" \
  train.save_model_freq="$save_model_freq" \
  train.actor_lr="$actor_lr" \
  train.actor_lr_scheduler.min_lr="$actor_lr" \
  train.n_critic_warmup_itr="$critic_warmup" \
  train.target_kl="$target_kl" \
  train.update_epochs="$update_epochs" \
  train.max_grad_norm="$max_grad_norm" \
  env.dense_success_reward="$dense_success_reward" \
  env.shaping_weight="$shaping_weight" \
  env.debug_action_reward="$debug_action_reward" \
  flow_steps="$flow_steps" \
  ft_denoising_steps="$ft_denoising_steps" \
  model.sampling_noise_scale="$sampling_noise_scale" \
  model.min_logprob_denoising_std="$logprob_std" \
  ${wandb_override[@]+"${wandb_override[@]}"} \
  "$@" \
  2>&1 | tee "$output_root/console.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

# Version sort, not lexicographic: with three-digit iteration counts a plain sort
# puts state_90.pt after state_120.pt, so the run reports and hashes the wrong
# file as its final checkpoint.
checkpoint=$(find "$output_root/finetune" -type f -name 'state_*.pt' -print | sort -V | tail -1)
if [ -z "$checkpoint" ]; then
  echo "Fine-tuning returned success without writing a checkpoint." >&2
  exit 1
fi
sha256sum "$checkpoint" | tee "$output_root/job-metadata/final-checkpoint.sha256"
echo "Flow PPO fine-tuning completed: $checkpoint"
