#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# DPPO reinforcement-learning fine-tuning of the pretrained blue-cube Diffusion
# Policy, on a rented RTX 5090. Runs entirely on the pod: the launcher installs
# credentials and VAST_INSTANCE_ID, this script provisions, verifies, gates,
# trains, and syncs to S3. Teardown stays manual, after an independent S3 check.
#
#   scp scripts/vast_dppo_finetune.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'VAST_INSTANCE_ID=... bash /workspace/vast_dppo_finetune.sh'
#
# The starting policy is chosen by run name and epoch, and the dataset export
# whose normalization bounds it was fitted against travels with it:
#
#   BASE_RUN_NAME=two-variant-1000-blue-cube-b256-e500-v2 BASE_EPOCH=500 \
#   ARTIFACT_NAME=two-variant-1000-blue-cube RUN_NAME=<fresh> \
#     bash /workspace/vast_dppo_finetune.sh
#
# Both must move together. The bounds normalize what the policy sees and
# unnormalize what it commands, so pairing a checkpoint with another run's
# artifact does not fail -- it quietly feeds the policy the wrong units.
#
# Rollout collection is CPU-bound (MuJoCo physics) and the PPO update is
# GPU-bound, so pick an instance with cores to spare: N_ENVS defaults to 32 and
# wants roughly that many vCPUs to keep the GPU fed. The per-iteration log line
# reports the rollout/update split so the next run can be sized from measurement.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
# Which pretraining run this fine-tune starts from, as an outputs/ run name. The
# default is dp_blue_cube_1000, whose epoch 1500 is the strongest checkpoint from
# the closed-loop baseline sweep.
base_run_name="${BASE_RUN_NAME:-dp_blue_cube_1000}"
# Which pretraining epoch to start RL from. Epoch 1500 of the default run is the
# strongest policy (~0.75 on held-out scenes) but leaves little headroom. DPPO's
# own paper fine-tunes deliberately undertrained checkpoints "so there is more
# room for fine-tuning improvement"; epoch 200 of that run scores ~0.45 here.
#
# Do NOT justify this by "epoch 1500's failures are scene-difficulty dominated,
# so the advantage carries no signal" -- that was measured and is false. At the
# exploration noise this launcher uses, 65% of scenes produce both successes and
# failures in 8 repeats, so the rollouts do carry same-scene/different-action
# contrast. The real gap is that a missed grasp is terminal: the policy shoves
# the cube, retreats and hovers out the clock, because no demonstration contains
# a retry. See the 2026-08-02 scene-difficulty sweep.
base_epoch="${BASE_EPOCH:-1500}"
# The pretraining launcher writes one timestamped directory under this prefix, so
# discover it rather than pinning it: a run name plus an epoch is what a human
# knows, and the timestamp is not. Set BASE_RUN_DIR to disambiguate a prefix that
# somehow holds more than one.
pretrain_prefix="$bucket_root/outputs/$base_run_name/pretrain/so101_pre_diffusion_unet_img_to2_ta16_te2_td100"
# The dataset export the checkpoint was trained on. Its normalization bounds are
# what the environment normalizes states with and unnormalizes actions by, so it
# must be the artifact that run used and not merely a similar one.
artifact_name="${ARTIFACT_NAME:-blue-cube-1000-10hz-96x96}"
artifact_s3="$bucket_root/diffusion-policy-data/$artifact_name.tar.zst"
# The appearance the base checkpoint was trained in. Getting this wrong is the
# failure the pre-flight gate exists to catch -- an AprilTag cube shown to a
# blue-cube policy scored 0/4 contact attempts -- so it travels with the
# checkpoint, not with the config file.
scene_appearance="${SCENE_APPEARANCE:-blue-cube}"
# What the base checkpoint predicts, stated rather than discovered. The bounds
# file declares it and the environment obeys the file, so a mismatch never
# raises -- the policy is simply commanded in the wrong units. Declaring it here
# turns that into a failed pre-flight.
action_encoding="${ACTION_ENCODING:-absolute}"

run_name="${RUN_NAME:-dppo_ft_${base_run_name}_${base_epoch}_$(date +%Y%m%d)}"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
base_policy="$workspace/artifacts/${base_run_name}_state_$base_epoch.pt"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/dppo-finetune.log"
status_file="$workspace/dppo-finetune-status.json"

n_envs="${N_ENVS:-32}"
n_steps="${N_STEPS:-40}"
n_train_itr="${N_TRAIN_ITR:-301}"
# Evaluation and checkpoint cadence. A first run answering "does this improve
# anything" wants both far tighter than a long production run does: every eval
# iteration is a deterministic success-rate measurement over n_envs episodes,
# and every checkpoint is a candidate for the paired before/after comparison.
val_freq="${VAL_FREQ:-10}"
save_model_freq="${SAVE_MODEL_FREQ:-10}"
actor_lr="${ACTOR_LR:-1e-5}"
# Iterations that train only the critic before the actor is allowed to move. The
# reward is sparse and the critic starts from scratch, so with too few of these
# the actor is updated on advantages that are still noise -- which degrades a
# strong behavior-cloned policy rather than improving it.
critic_warmup="${CRITIC_WARMUP:-10}"
# A guard, not a tuning knob: DPPO's default of 1 nat allows enormous drift per
# iteration, and this policy is worth protecting.
target_kl="${TARGET_KL:-0.1}"
# Gradient steps per iteration is update_epochs x minibatches. PPO's trust region
# cannot restrain this policy -- measured approx_kl runs 1e-8..3e-4 against a 0.1
# target while behaviour collapses, because a 0.1 likelihood floor treats ~9
# degrees of joint motion as one standard deviation -- so step count and step
# size are the only working brakes.
update_epochs="${UPDATE_EPOCHS:-10}"
# Exploration noise, in normalized action units. DPPO's 0.1 default is set for
# robomimic's small end-effector deltas; here an action is an absolute joint
# command spanning ~180 degrees, so 0.1 is ~9 degrees of jitter on every joint of
# every emitted action and drives the on-policy success rate to nearly zero --
# which leaves PPO with no reward signal at all.
sampling_std="${SAMPLING_STD:-0.01}"
# The *likelihood* floor is a different quantity with a different job: DPPO clips
# the std used to evaluate Gaussian log-probabilities for numerical stability,
# and the paper says to keep it at 0.1. Lowering it with the exploration floor
# makes the densities ~10x narrower, so PPO's importance ratios explode across 5
# denoising steps x 16 horizon x 6 dimensions. Keep the two decoupled.
logprob_std="${LOGPROB_STD:-0.1}"
# Image-shift augmentation during the update. In behavior cloning a shifted
# image keeps a fixed target action, so it regularizes; in PPO it breaks the
# correspondence between the sampled action and the observation it was drawn
# from, and a 4-pixel shift is comparable to the cube's own footprint in a 96x96
# overhead frame.
augment="${AUGMENT:-False}"
# Potential-based reward shaping weight (reward per metre). 0 is the sparse
# settled-placement reward. Non-zero densifies credit assignment without
# changing the optimal policy.
shaping_weight="${SHAPING_WEIGHT:-0.0}"
# Diagnostic: replace the task reward with a trivially learnable function of the
# action. Used to prove the gradient path works at all.
debug_action_reward="${DEBUG_ACTION_REWARD:-False}"
# The denoised-sample clip. At 1.0 it clips exactly at the normalized action
# bounds -- where 25% of this task's gripper commands sit, because the gripper is
# min-max normalized between fully closed and fully open and is effectively
# binary. Sampling is clipped there while the Gaussian likelihood driving the
# policy gradient is not, so that dimension's gradient is biased and the grasp
# degrades while reaching stays intact. Widening it moves the clip off the data.
denoised_clip="${DENOISED_CLIP:-1.0}"
# Which policy class to fine-tune. The arm-only variant masks the gripper out of
# the importance ratio, leaving its behavior-cloned policy untouched; see
# pick_and_place/dppo_rl/arm_only.py for why the gripper resists a Gaussian.
model_target="${MODEL_TARGET:-model.diffusion.diffusion_ppo.PPODiffusion}"
preflight_episodes="${PREFLIGHT_EPISODES:-24}"
# The pre-flight gate. The pretrained policy lifts the cube in roughly half of
# scenes; anything near zero means the environment is not showing it the task it
# was trained on, which is the failure this gate exists to catch (rendering the
# AprilTag cube to a blue-cube policy scored 0/4 contacts).
min_lift_rate="${MIN_LIFT_RATE:-0.25}"

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
  cp "$job_log" "$output_root/job-metadata/dppo-finetune.log"
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
# libegl1 is what MuJoCo renders through in each rollout worker.
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

# The vendored image agent moves only "rgb" and "state" to torch, though it
# builds obs_dims generically from shape_meta. The asymmetric critic needs a
# third key, "privileged", which the rollout collects as numpy and then hands to
# torch.split:
#
#   AttributeError: 'numpy.ndarray' object has no attribute 'split'
#
# raised at the end of the very first iteration, after the pre-flight gate has
# passed and the run looks healthy. This patch is therefore load-bearing for
# every run that uses the privileged critic, which is the default config. It was
# dropped once as vestigial because the patch file it names had never been
# committed; the file is committed now. Applied here so a fresh clone of the
# pinned submodule gets it too.
if ! git -C third_party/dppo apply --reverse --check \
     ../../config/diffusion_policy/dppo-generic-obs-keys.patch 2>/dev/null; then
  git -C third_party/dppo apply ../../config/diffusion_policy/dppo-generic-obs-keys.patch
  echo "Applied dppo-generic-obs-keys.patch to the vendored agent."
else
  echo "dppo-generic-obs-keys.patch already applied."
fi

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
# uv declares the requirements unsatisfiable. Always install with them, then
# repair the torch build below if it does not match the host driver.
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

# The torch pin above resolves to whatever CUDA build PyPI serves, which can be
# newer than the host driver supports (cu130 needs r580+; this has been r575).
# Verify before anything expensive, and fall back to the cu128 build, which runs
# on a 12.9 driver and supports the 5090's sm_120.
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

# Refuse to overwrite or resume a run. DPPO's fine-tuner has no resume path, so
# a name that already exists in S3 means a previous run's artifacts are there.
if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

mkdir -p "$artifact_root"

# The pretraining launcher writes exactly one timestamped directory per run, and
# nobody remembers a timestamp. Resolve it from the run name instead -- except
# for dp_blue_cube_1000, which predates that launcher's refusal to reuse an
# output prefix and holds five directories, four of them abandoned starts.
declare -A legacy_run_dir=(
  [dp_blue_cube_1000]="2026-07-31_04-01-38_42"
)
base_run_dir="${BASE_RUN_DIR:-${legacy_run_dir[$base_run_name]:-}}"
if [ -z "$base_run_dir" ]; then
  mapfile -t base_run_dirs < <(aws s3 ls "$pretrain_prefix/" | awk '/ PRE /{print $2}' | tr -d '/')
  if [ "${#base_run_dirs[@]}" -ne 1 ]; then
    echo "expected one run directory under $pretrain_prefix/, found ${#base_run_dirs[@]}:" >&2
    printf '  %s\n' ${base_run_dirs[@]+"${base_run_dirs[@]}"} >&2
    echo "set BASE_RUN_DIR to pick one." >&2
    exit 1
  fi
  base_run_dir="${base_run_dirs[0]}"
fi
base_policy_s3="$pretrain_prefix/$base_run_dir/checkpoint/state_$base_epoch.pt"

# The expected checkpoint hash, most authoritative source first: an explicit
# override, then the hash the pretraining launcher recorded beside its own run,
# then the two checkpoints that predate that launcher. Refuse to start without
# one: aws does not verify a multipart download, so a truncated checkpoint
# surfaces as a policy that has silently forgotten the task rather than as a
# failed transfer.
base_policy_sha256="${BASE_POLICY_SHA256:-}"
if [ -z "$base_policy_sha256" ]; then
  recorded_sha="$bucket_root/outputs/$base_run_name/job-metadata/state_$base_epoch.pt.sha256"
  base_policy_sha256=$(aws s3 cp "$recorded_sha" - 2>/dev/null | awk '{print $1}')
fi
declare -A legacy_policy_sha=(
  [dp_blue_cube_1000/1500]="4bbcdc552942456dc76bd2911f57e808da5314505ae15e8581bd3bdcfbb57846"
  [dp_blue_cube_1000/200]="a3c50124d6897aa41951178d8d54f574aac22119c09e668262d28b74c5faa153"
)
if [ -z "$base_policy_sha256" ]; then
  base_policy_sha256="${legacy_policy_sha[$base_run_name/$base_epoch]:-}"
fi
if [ -z "$base_policy_sha256" ]; then
  echo "no expected sha256 for $base_run_name epoch $base_epoch, and none recorded at" >&2
  echo "$recorded_sha -- pass BASE_POLICY_SHA256 to state what you expect." >&2
  exit 1
fi

# The artifact is published as one .tar.zst and in no other form; the loose
# per-file prefix this used to copy from is gone from the bucket. Its sha256
# covers the whole archive, so take the download, verify it, and unpack only the
# two small members the environment needs -- train.npz is gigabytes and is the
# pretrainer's business, not this run's.
if [ ! -f "$artifact_root/normalization.npz" ] || [ ! -f "$artifact_root/export.json" ]; then
  staging="$workspace/artifacts"
  aws s3 cp "$artifact_s3" "$staging/$artifact_name.tar.zst" --only-show-errors
  aws s3 cp "$artifact_s3.sha256" "$staging/$artifact_name.tar.zst.sha256" --only-show-errors
  (cd "$staging" && sha256sum -c "$artifact_name.tar.zst.sha256")
  tar -x -I zstd -f "$staging/$artifact_name.tar.zst" -C "$staging" --wildcards \
    "*/normalization.npz" "*/export.json"
  rm -f "$staging/$artifact_name.tar.zst" "$staging/$artifact_name.tar.zst.sha256"
fi
for member in normalization.npz export.json; do
  if [ ! -f "$artifact_root/$member" ]; then
    echo "$artifact_name.tar.zst does not hold $artifact_name/$member" >&2
    exit 1
  fi
done

aws s3 cp "$base_policy_s3" "$base_policy" --only-show-errors
echo "$base_policy_sha256  $base_policy" | sha256sum --check
# No separate expected hash for normalization.npz: the archive's published
# sha256, checked above, already pins its contents exactly. Record what was
# unpacked so a run can be traced back to its bounds.
sha256sum "$artifact_root/normalization.npz" \
  | tee "$output_root/job-metadata/normalization.sha256"
cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"
cp config/diffusion_policy/ft_ppo_so101_unet_img.yaml "$output_root/job-metadata/launcher-config.yaml"

# Gitignored build artifacts a clean clone lacks. The AprilTag textures are
# generated; the camera calibration is machine-local -- its 9.13 mm / 1.76 deg
# offset is outside the domain-randomization jitter, so nothing else supplies it
# and the scene would be subtly wrong without it. The bucket's config-backup is
# the copy of record, so a pod can be brought up from any machine.
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

export DPPO_DATA_DIR="$artifact_root"
export DPPO_BASE_POLICY="$base_policy"
export DPPO_LOG_DIR="$output_root"
export PYTHONPATH="$repo/third_party/dppo"

# Weights & Biases is optional, and a key on the controller does nothing: without
# one on the *pod*, wandb.init aborts the run. That would land after provisioning
# and after the pre-flight gate has already spent GPU minutes, so degrade to no
# logging and say so rather than dying an hour in. The console log and the
# per-iteration S3 sync carry the same numbers.
wandb_override=()
if ! grep -q api.wandb.ai "${NETRC:-$HOME/.netrc}" 2>/dev/null; then
  echo "No api.wandb.ai entry in $HOME/.netrc on this pod; training without W&B logging."
  wandb_override=(wandb=null)
fi

# Gate: does the pretrained policy still perform the task in this environment?
# An observation-pipeline mismatch is indistinguishable from "RL had nothing to
# learn from" once training starts, and costs a full run to discover.
"$venv/bin/python" py/scripts/check_dppo_rl_env.py \
  --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \
  --checkpoint "$base_policy" \
  --normalization "$artifact_root/normalization.npz" \
  --episodes "$preflight_episodes" \
  --n-envs "$n_envs" \
  --scene-appearance "$scene_appearance" \
  --expect-action-encoding "$action_encoding" \
  --device cuda:0 \
  --output "$output_root/job-metadata/preflight.json"

"$venv/bin/python" - "$output_root/job-metadata/preflight.json" "$min_lift_rate" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())["summary"]
threshold = float(sys.argv[2])
lifted = summary["rates"]["cube_lifted"]
contact = summary["rates"]["pickup_contact_attempted"]
print(f"Pre-flight: lift {lifted:.2f}, contact {contact:.2f} over {summary['episodes']} episodes.")
if lifted < threshold:
    raise SystemExit(
        f"Pre-flight lift rate {lifted:.2f} is below {threshold:.2f}: the base policy "
        "is not performing the task in this environment. Fine-tuning would be "
        "measuring a broken observation pipeline."
    )
PY

cd "$repo/third_party/dppo"
( while sleep 900; do
    aws s3 sync "$output_root" "$output_prefix" --only-show-errors
  done ) &
sync_pid=$!
set +e
"$venv/bin/python" script/run.py \
  --config-path "$repo/config/diffusion_policy" \
  --config-name ft_ppo_so101_unet_img \
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
  train.augment="$augment" \
  env.scene_appearance="$scene_appearance" \
  env.shaping_weight="$shaping_weight" \
  env.debug_action_reward="$debug_action_reward" \
  model.denoised_clip_value="$denoised_clip" \
  model._target_="$model_target" \
  ${wandb_override[@]+"${wandb_override[@]}"} \
  "$@" \
  model.min_sampling_denoising_std="$sampling_std" \
  model.min_logprob_denoising_std="$logprob_std" \
  2>&1 | tee "$output_root/console.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

checkpoint=$(find "$output_root/finetune" -type f -name 'state_*.pt' -print | sort | tail -1)
if [ -z "$checkpoint" ]; then
  echo "Fine-tuning returned success without writing a checkpoint." >&2
  exit 1
fi
sha256sum "$checkpoint" | tee "$output_root/job-metadata/final-checkpoint.sha256"
echo "DPPO fine-tuning completed: $checkpoint"
