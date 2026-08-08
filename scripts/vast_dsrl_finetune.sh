#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# DSRL steering of the pretrained blue-cube Diffusion Policy, on a rented RTX
# 5090. Runs entirely on the pod: the launcher installs credentials and
# VAST_INSTANCE_ID, this script provisions, verifies, gates, trains, and syncs to
# S3. Teardown stays manual, after an independent S3 check.
#
#   scp scripts/vast_dsrl_finetune.sh <ssh-host>:/workspace/
#   ssh <ssh-host> 'VAST_INSTANCE_ID=... bash /workspace/vast_dsrl_finetune.sh'
#
# Sibling of vast_dppo_finetune.sh and deliberately built from the same parts:
# the same repository, the same vendored DPPO, the same environment, the same
# pre-flight gate, the same artifact and checkpoint verification. What differs is
# what is trained. DPPO fine-tunes the diffusion policy's weights with PPO; DSRL
# freezes them and learns which noise the denoising chain should start from, so
# the run produces a small latent-noise actor rather than a new policy.
#
# Three consequences worth knowing before reading the body:
#
#   - The vendored DPPO patches are not applied. Both fix its PPO image agent --
#     one its handling of the privileged observation key, one its diagnostics --
#     and nothing here runs that agent. Only the diffusion model and its sampler
#     are used from the submodule.
#   - There is a second gate. The pre-flight gate asks whether the base policy
#     still performs the task in this environment; the steerability gate asks
#     whether its input noise changes anything, which is the precondition DSRL
#     itself rests on. A policy fitted to a deterministic analytic planner may
#     ignore its noise entirely, and if it does, no configuration of the learner
#     helps. Costs a couple of minutes; saves the run.
#   - The base checkpoint cannot be damaged. It is loaded read-only and never
#     written back, so the worst outcome of a bad run is a latent actor no better
#     than the standard normal draw it started from.
#
# Defaults steer the recovery base. That is the base DPPO worked on -- its
# demonstrations contain retries, so recovering from a missed grasp is inside the
# behavior-cloned distribution DSRL is able to steer toward -- and it makes the
# result directly comparable to the selected DPPO checkpoint's validated 0.746.
#
#   BASE_RUN_NAME=recovery-1000-blue-cube-b256-e500 BASE_EPOCH=500 \
#   ARTIFACT_NAME=recovery-1000-blue-cube RUN_NAME=<fresh> \
#     bash /workspace/vast_dsrl_finetune.sh
#
# The checkpoint and the dataset export whose normalization bounds it was fitted
# against must move together. The bounds normalize what the policy sees and
# unnormalize what it commands, so pairing a checkpoint with another run's
# artifact does not fail -- it quietly feeds the policy the wrong units.
#
# Rollout collection is CPU-bound (MuJoCo physics and rendering) and the gradient
# steps are small MLP passes, so this is more rollout-dominated than the PPO
# launcher was: pick an instance with cores to spare. N_ENVS defaults to 32 and
# wants roughly that many vCPUs.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl

: "${VAST_INSTANCE_ID:?}"

bucket_root="s3://allyouneed/pick-and-place"
# The recovery base: 0.684 on the 7M validation block, and the only base on
# which RL has been shown to work here. Its failures are recoverable, which is
# what gives an outcome-driven learner something to push against.
base_run_name="${BASE_RUN_NAME:-recovery-1000-blue-cube-b256-e500}"
base_epoch="${BASE_EPOCH:-500}"
pretrain_prefix="$bucket_root/outputs/$base_run_name/pretrain/so101_pre_diffusion_unet_img_to2_ta16_te2_td100"
artifact_name="${ARTIFACT_NAME:-recovery-1000-blue-cube}"
artifact_s3="$bucket_root/diffusion-policy-data/$artifact_name.tar.zst"
scene_appearance="${SCENE_APPEARANCE:-blue-cube}"
action_encoding="${ACTION_ENCODING:-absolute}"

run_name="${RUN_NAME:-dsrl_${base_run_name}_${base_epoch}_$(date +%Y%m%d)}"
output_prefix="$bucket_root/outputs/$run_name"
workspace="/workspace"
repo="$workspace/pick-and-place"
artifact_root="$workspace/artifacts/$artifact_name"
base_policy="$workspace/artifacts/${base_run_name}_state_$base_epoch.pt"
output_root="$workspace/outputs/$run_name"
job_log="$workspace/dsrl-finetune.log"
status_file="$workspace/dsrl-finetune-status.json"

n_envs="${N_ENVS:-32}"
seed="${SEED:-42}"
# Parallel environment steps. At 32 envs and ~19 chunk-steps an episode, 4000 is
# ~128k transitions and ~6700 episodes -- the same order of environment
# interaction one DPPO run spent, so the two cost about the same to compare.
total_iterations="${TOTAL_ITERATIONS:-4000}"
# Steps whose noise is drawn from N(0, I) rather than from the actor, so the
# buffer describes the base policy before anything is learned from it.
warmup_iterations="${WARMUP_ITERATIONS:-200}"
# Gradient steps per parallel environment step. The paper's Robomimic image
# tasks use 20. The updates are three small MLP passes at batch 256 against a
# rollout step that runs eight control ticks of physics in every worker, so this
# can be raised well before it shows in the wall clock.
gradient_steps="${GRADIENT_STEPS:-20}"
batch_size="${BATCH_SIZE:-256}"
# Largest absolute value of a latent-noise action, the paper's b_W. 1.5 across
# its Robomimic tasks. Below ~1 the actor cannot even reach the tails the base
# policy samples from; far above it, the denoiser is being asked to extrapolate
# off the noise distribution it was trained under.
action_magnitude="${ACTION_MAGNITUDE:-1.5}"
buffer_capacity="${BUFFER_CAPACITY:-400000}"
# SAC's initial entropy weight. The config's 1.0 follows the paper, whose
# action spaces are far smaller than this 96-dimensional latent: measured
# here, alpha 1 puts ~65 of entropy bonus into every bootstrapped target
# against a task reward of at most 8 per step, so the soft value converges to
# ~1000 of which the task is ~120. The auto-tuner corrects it at ~0.835 per
# 600 gradient steps, which is ~735 iterations before it stops dominating.
init_temperature="${INIT_TEMPERATURE:-}"
# Give the critic privileged simulator state. The actor never sees it, so the
# policy that deploys is unchanged; set to true to make the whole learner
# transferable to hardware at the cost of a harder value-learning problem.
observable_critic="${OBSERVABLE_CRITIC:-false}"
critic_flag=()
if [ "$observable_critic" = "true" ]; then
  critic_flag=(--observable-critic)
fi

preflight_episodes="${PREFLIGHT_EPISODES:-24}"
min_lift_rate="${MIN_LIFT_RATE:-0.25}"
# The steerability gate. Both thresholds are floors on "there is something to
# steer", not predictions of success.
#
# STEER_MIN_RATIO compares the spread of chunks produced by different noise
# draws at one state against the policy's own tick-to-tick motion. At 0 the
# denoiser ignores its noise and DSRL is exactly a no-op.
#
# STEER_MIN_CONTESTED is the fraction of scenes where repeated draws disagree
# about success. Those are the only scenes where the return depends on something
# the latent policy controls -- the same property that made the recovery base
# learnable for DPPO and the absolute base flat. Measured over STEER_REPEATS
# scorings of one scene set.
steer_repeats="${STEER_REPEATS:-4}"
steer_episodes="${STEER_EPISODES:-96}"
steer_min_ratio="${STEER_MIN_RATIO:-0.05}"
steer_min_contested="${STEER_MIN_CONTESTED:-0.10}"
skip_steerability_gate="${SKIP_STEERABILITY_GATE:-false}"

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
  cp "$job_log" "$output_root/job-metadata/dsrl-finetune.log"
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
# libopengl0 is in AGENTS.md's list and its absence shows up as a repeated
# "Failed to load library (libOpenGL.so.0)" during every rollout. Rendering
# goes through EGL either way, so this is noise rather than a fault -- but it
# is noise in the one log a failed run is read from.
apt-get install -y curl git unzip zstd libegl1 libgl1 libopengl0
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
# Which ref to run. Empty takes whatever the clone landed on, which is the
# default branch; name a branch or a commit to run work that is not on it yet.
# Detached on purpose: the pod is not a place to develop from, and a detached
# HEAD makes repository-commit.txt below unambiguous.
if [ -n "${REPO_REF:-}" ]; then
  git fetch --tags origin "$REPO_REF"
  git checkout --detach FETCH_HEAD
fi
git submodule update --init --recursive
git rev-parse HEAD | tee "$output_root/job-metadata/repository-commit.txt"

venv="$workspace/venvs/pick-and-place"
# The package needs Python >= 3.12 (lerobot 0.5.1 pins transformers 5.3.0), and
# the image's bundled interpreter is whatever the image ships -- 3.10.12 on
# vastai/pytorch:latest, which resolves to "your requirements are unsatisfiable"
# several minutes into provisioning. Test the candidates rather than assume one,
# and let uv fetch a managed 3.13 when none is new enough.
base_python="${BASE_PYTHON:-}"
if [ -z "$base_python" ]; then
  for candidate in /venv/main/bin/python python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' \
         2>/dev/null; then
      base_python="$candidate"
      break
    fi
  done
fi
if [ -z "$base_python" ]; then
  echo "No interpreter on this image is >= 3.12; installing a managed 3.13."
  uv python install 3.13
  base_python="3.13"
fi
echo "Building the venv on $base_python."
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

# Refuse to overwrite or resume a run: a name that already exists in S3 means a
# previous run's artifacts are there.
if aws s3 ls "$output_prefix/" | grep -q .; then
  echo "Output prefix already contains objects: $output_prefix" >&2
  exit 1
fi

mkdir -p "$artifact_root"

base_run_dir="${BASE_RUN_DIR:-}"
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

# aws does not verify a multipart download, so a truncated checkpoint surfaces as
# a policy that has silently forgotten the task rather than as a failed transfer.
base_policy_sha256="${BASE_POLICY_SHA256:-}"
if [ -z "$base_policy_sha256" ]; then
  recorded_sha="$bucket_root/outputs/$base_run_name/job-metadata/state_$base_epoch.pt.sha256"
  base_policy_sha256=$(aws s3 cp "$recorded_sha" - 2>/dev/null | awk '{print $1}')
fi
if [ -z "$base_policy_sha256" ]; then
  echo "no expected sha256 for $base_run_name epoch $base_epoch, and none recorded at" >&2
  echo "${recorded_sha:-<none>} -- pass BASE_POLICY_SHA256 to state what you expect." >&2
  exit 1
fi

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
sha256sum "$artifact_root/normalization.npz" \
  | tee "$output_root/job-metadata/normalization.sha256"
cp "$artifact_root/export.json" "$output_root/job-metadata/dataset-export.json"
cp config/diffusion_policy/dsrl_so101.yaml "$output_root/job-metadata/launcher-config.yaml"

# Gitignored build artifacts a clean clone lacks. The AprilTag textures are
# generated; the camera calibration is machine-local, and the scene is subtly
# wrong without it.
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

if [ "${WANDB:-on}" = "off" ]; then
  echo "WANDB=off: training without W&B logging, by request."
  wandb_flag=()
elif grep -q api.wandb.ai "${NETRC:-$HOME/.netrc}" 2>/dev/null; then
  wandb_flag=(--wandb)
  echo "W&B credential found on this pod."
else
  echo "No api.wandb.ai entry in ${NETRC:-$HOME/.netrc} on this pod." >&2
  echo "Copy your ~/.netrc to the pod (vast_pap_provision.sh stages it), or set" >&2
  echo "WANDB=off to run without logging. Refusing to start blind." >&2
  exit 1
fi

# Gate 1: does the pretrained policy still perform the task in this environment?
# An observation-pipeline mismatch is indistinguishable from "RL had nothing to
# learn from" once training starts, and costs a full run to discover.
"$venv/bin/python" py/scripts/check_dppo_rl_env.py \
  --config config/diffusion_policy/dsrl_so101.yaml \
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
        "is not performing the task in this environment. Steering would be measuring "
        "a broken observation pipeline."
    )
PY

# Gate 2: does the input noise move anything? DSRL can only re-weight modes the
# behavior-cloned policy already has, and every demonstration here came from a
# deterministic analytic planner. If the denoiser ignores its noise there is
# nothing to steer, and that is cheaper to find out here than after a run.
if [ "$skip_steerability_gate" = "true" ]; then
  echo "SKIP_STEERABILITY_GATE=true: not measuring whether the noise moves anything."
else
  "$venv/bin/python" py/scripts/measure_dsrl_steerability.py spread \
    --config config/diffusion_policy/dsrl_so101.yaml \
    --checkpoint "$base_policy" \
    --normalization "$artifact_root/normalization.npz" \
    --n-envs 8 --steps 60 --draws 16 --device cuda:0 \
    --output "$output_root/job-metadata/steerability-spread.json"

  # Repeated deterministic scorings of one scene set. The only randomness left in
  # an evaluation is the latent draw, so varying --seed varies exactly the
  # quantity DSRL controls, and disagreement between repeats is the headroom.
  for repeat in $(seq 1 "$steer_repeats"); do
    "$venv/bin/python" py/scripts/check_dppo_rl_env.py \
      --config config/diffusion_policy/dsrl_so101.yaml \
      --checkpoint "$base_policy" \
      --normalization "$artifact_root/normalization.npz" \
      --episodes "$steer_episodes" \
      --n-envs "$n_envs" \
      --scene-appearance "$scene_appearance" \
      --scene-seed-base 6000000 \
      --seed "$repeat" \
      --device cuda:0 \
      --output "$output_root/job-metadata/steerability-repeat-$repeat.json"
  done
  "$venv/bin/python" py/scripts/measure_dsrl_steerability.py outcomes \
    "$output_root"/job-metadata/steerability-repeat-*.json \
    --output "$output_root/job-metadata/steerability-outcomes.json"

  "$venv/bin/python" - \
    "$output_root/job-metadata/steerability-spread.json" \
    "$output_root/job-metadata/steerability-outcomes.json" \
    "$steer_min_ratio" "$steer_min_contested" <<'PY'
import json
import sys
from pathlib import Path

spread = json.loads(Path(sys.argv[1]).read_text())["action_spread"]
outcomes = json.loads(Path(sys.argv[2]).read_text())["outcome_spread"]
min_ratio, min_contested = float(sys.argv[3]), float(sys.argv[4])

print(
    f"Steerability: noise/step spread ratio {spread['ratio']:.3f}, "
    f"contested scenes {outcomes['contested_fraction']:.3f} "
    f"({outcomes['contested']}/{outcomes['scenarios']} over {outcomes['repeats']} draws), "
    f"mean success {outcomes['mean_success_rate']:.3f}, "
    f"per-scene oracle headroom {outcomes['oracle_headroom']:.3f}."
)
failures = []
if spread["ratio"] < min_ratio:
    failures.append(
        f"action spread ratio {spread['ratio']:.3f} is below {min_ratio:.3f}: the "
        "denoiser barely reads its input noise, so there is no latent space to search"
    )
if outcomes["contested_fraction"] < min_contested:
    failures.append(
        f"only {outcomes['contested_fraction']:.3f} of scenes are contested, below "
        f"{min_contested:.3f}: outcomes are decided by the scene rather than by the "
        "noise, so steering has nothing to select between"
    )
if failures:
    raise SystemExit(
        "Steerability gate failed.\n  - " + "\n  - ".join(failures) +
        "\nSet SKIP_STEERABILITY_GATE=true to train anyway, and record why."
    )
PY
fi

( while sleep 900; do
    aws s3 sync "$output_root" "$output_prefix" --only-show-errors
  done ) &
sync_pid=$!
set +e
"$venv/bin/python" py/scripts/train_dsrl.py \
  --config config/diffusion_policy/dsrl_so101.yaml \
  --checkpoint "$base_policy" \
  --normalization "$artifact_root/normalization.npz" \
  --output-dir "$output_root/dsrl" \
  --device cuda:0 \
  --n-envs "$n_envs" \
  --seed "$seed" \
  --total-iterations "$total_iterations" \
  --warmup-iterations "$warmup_iterations" \
  --gradient-steps "$gradient_steps" \
  --batch-size "$batch_size" \
  --action-magnitude "$action_magnitude" \
  --buffer-capacity "$buffer_capacity" \
  ${init_temperature:+--init-temperature "$init_temperature"} \
  --expect-action-encoding "$action_encoding" \
  ${critic_flag[@]+"${critic_flag[@]}"} \
  ${wandb_flag[@]+"${wandb_flag[@]}"} \
  "$@" \
  2>&1 | tee "$output_root/console.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

# Version sort, not lexicographic: with four-digit iteration counts a plain sort
# puts state_900.pt after state_4000.pt, so the run would report and hash the
# wrong file as its final checkpoint.
checkpoint=$(find "$output_root/dsrl" -type f -name 'state_*.pt' -print | sort -V | tail -1)
if [ -z "$checkpoint" ]; then
  echo "Training returned success without writing a checkpoint." >&2
  exit 1
fi
sha256sum "$checkpoint" | tee "$output_root/job-metadata/final-checkpoint.sha256"
echo "DSRL training completed: $checkpoint"
echo
echo "Score it against the base with, on this pod:"
echo "  BASE_RUN_NAME=$base_run_name BASE_EPOCH=$base_epoch ARTIFACT_NAME=$artifact_name \\"
echo "    DSRL_RUN_NAME=$run_name DSRL_ITR=<itr> SCORE_NAME=<name> \\"
echo "    bash /workspace/vast_score_dppo_env.sh"
