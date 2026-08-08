#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Score one checkpoint in the DPPO fine-tuning environment on a fixed scene set,
# deterministically. Run it once per checkpoint to build the paired comparison
# that is the only trustworthy read of a change at this effect size: DPPO's
# in-training eval draws fresh scenes every time, and 64 of them swing +/-6% on
# an unchanged policy.
#
#   ssh <ssh-host> 'BASE_RUN_NAME=two-variant-1000-blue-cube-b256-e500-v2 \
#     BASE_EPOCH=500 ARTIFACT_NAME=two-variant-1000-blue-cube \
#     ACTION_ENCODING=absolute SCORE_NAME=absolute-e500 \
#     bash /workspace/vast_score_dppo_env.sh'
#
# To score a DPPO fine-tuned checkpoint, add FT_RUN_NAME and FT_ITR: the base
# checkpoint still builds the model and pins the artifact, and the fine-tuned
# weights are loaded over it, exactly as during fine-tuning. Everything else --
# scenes, sampler, seed -- is identical, so the score pairs against the base's.
#
#   ssh <ssh-host> '... FT_RUN_NAME=dppo_ft_matched_std_20260808 FT_ITR=120 \
#     SCORE_NAME=matched-std-itr120 bash /workspace/vast_score_dppo_env.sh'
#
# Run vast_pap_provision.sh first. Everything that decides what is measured --
# scenes, sampler, torch seed, episode count -- is pinned below or passed in, so
# two invocations differing only in the checkpoint differ only in the weights.
#
# The artifact must be the one the checkpoint was trained against: its bounds
# normalize what the policy sees and unnormalize what it commands, and its
# declared action encoding decides whether an emitted action is a joint position
# or an offset from the measured joints. ACTION_ENCODING states which you expect
# and the run stops rather than measuring a policy fed the wrong units.

set -euo pipefail

: "${BASE_RUN_NAME:?set BASE_RUN_NAME to a pretraining run under outputs/}"
: "${ARTIFACT_NAME:?set ARTIFACT_NAME to the export the checkpoint was trained on}"
: "${SCORE_NAME:?set SCORE_NAME to name this measurement}"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="/root/.local/bin:$PATH"

bucket_root="s3://allyouneed/pick-and-place"
base_epoch="${BASE_EPOCH:-500}"
action_encoding="${ACTION_ENCODING:-absolute}"
scene_appearance="${SCENE_APPEARANCE:-blue-cube}"
# Held-out scenes: 6,000,000 is disjoint from the training stream (5,000,000)
# and from both frozen manifests (1701, 2701).
scene_seed_base="${SCENE_SEED_BASE:-6000000}"
episodes="${EPISODES:-256}"
n_envs="${N_ENVS:-32}"
# Executed actions per query. Empty keeps the fine-tuning config's 8.
act_steps="${ACT_STEPS:-}"
# Seeds torch's sampling. Deterministic denoising still starts from a random
# latent, so fixing this is what makes two checkpoints differ only by weights.
seed="${SEED:-0}"
# A DPPO fine-tuning run under outputs/ whose weights are scored on top of the
# base checkpoint. Empty scores the base checkpoint itself.
ft_run_name="${FT_RUN_NAME:-}"
ft_itr="${FT_ITR:-}"
if [ -n "$ft_run_name" ] && [ -z "$ft_itr" ]; then
  echo "FT_RUN_NAME is set but FT_ITR is not; set FT_ITR to the iteration to score." >&2
  exit 1
fi

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"
artifact_root="$workspace/artifacts/$ARTIFACT_NAME"
scores="$workspace/scores"
checkpoint="$workspace/artifacts/${BASE_RUN_NAME}_state_$base_epoch.pt"
pretrain_prefix="$bucket_root/outputs/$BASE_RUN_NAME/pretrain/so101_pre_diffusion_unet_img_to2_ta16_te2_td100"

mkdir -p "$artifact_root" "$scores"
cd "$repo"

# Two small members are all the environment needs; train.npz is gigabytes and is
# the pretrainer's business.
if [ ! -f "$artifact_root/normalization.npz" ]; then
  staging="$workspace/artifacts"
  aws s3 cp "$bucket_root/diffusion-policy-data/$ARTIFACT_NAME.tar.zst" \
    "$staging/$ARTIFACT_NAME.tar.zst" --only-show-errors
  aws s3 cp "$bucket_root/diffusion-policy-data/$ARTIFACT_NAME.tar.zst.sha256" \
    "$staging/$ARTIFACT_NAME.tar.zst.sha256" --only-show-errors
  (cd "$staging" && sha256sum -c "$ARTIFACT_NAME.tar.zst.sha256")
  tar -x -I zstd -f "$staging/$ARTIFACT_NAME.tar.zst" -C "$staging" --wildcards \
    "*/normalization.npz" "*/export.json"
  rm -f "$staging/$ARTIFACT_NAME.tar.zst" "$staging/$ARTIFACT_NAME.tar.zst.sha256"
fi

if [ ! -f "$checkpoint" ]; then
  base_run_dir="${BASE_RUN_DIR:-}"
  if [ -z "$base_run_dir" ]; then
    mapfile -t base_run_dirs < <(aws s3 ls "$pretrain_prefix/" | awk '/ PRE /{print $2}' | tr -d '/')
    if [ "${#base_run_dirs[@]}" -ne 1 ]; then
      echo "expected one run directory under $pretrain_prefix/, found ${#base_run_dirs[@]};" >&2
      echo "set BASE_RUN_DIR to pick one." >&2
      exit 1
    fi
    base_run_dir="${base_run_dirs[0]}"
  fi
  aws s3 cp "$pretrain_prefix/$base_run_dir/checkpoint/state_$base_epoch.pt" \
    "$checkpoint" --only-show-errors
  # aws does not verify a multipart download, so a truncated checkpoint would
  # score as a policy that has forgotten the task rather than as a failed copy.
  expected=$(aws s3 cp \
    "$bucket_root/outputs/$BASE_RUN_NAME/job-metadata/state_$base_epoch.pt.sha256" - \
    2>/dev/null | awk '{print $1}')
  if [ -z "$expected" ]; then
    echo "no recorded sha256 for $BASE_RUN_NAME epoch $base_epoch" >&2
    exit 1
  fi
  echo "$expected  $checkpoint" | sha256sum --check
fi

ft_checkpoint=""
if [ -n "$ft_run_name" ]; then
  ft_checkpoint="$workspace/artifacts/${ft_run_name}_state_$ft_itr.pt"
  if [ ! -f "$ft_checkpoint" ]; then
    # The run nests a config name and a timestamp under finetune/; search for the
    # one object named state_$FT_ITR.pt instead of hardcoding either level.
    mapfile -t ft_matches < <(aws s3 ls --recursive \
      "$bucket_root/outputs/$ft_run_name/finetune/" \
      | awk -v itr="$ft_itr" '$4 ~ ("/checkpoint/state_" itr "[.]pt$") {print $3, $4}')
    if [ "${#ft_matches[@]}" -ne 1 ]; then
      echo "expected one state_$ft_itr.pt under $bucket_root/outputs/$ft_run_name/finetune/," >&2
      echo "found ${#ft_matches[@]}." >&2
      exit 1
    fi
    ft_size="${ft_matches[0]%% *}"
    ft_key="${ft_matches[0]#* }"
    s3_bucket="${bucket_root#s3://}"
    s3_bucket="${s3_bucket%%/*}"
    aws s3 cp "s3://$s3_bucket/$ft_key" "$ft_checkpoint" --only-show-errors
    # Mid-run checkpoints have no recorded sha256, and aws does not verify a
    # multipart download; matching the listed object size at least catches
    # truncation, which would otherwise score as a policy that forgot the task.
    ft_local_size=$(stat -c %s "$ft_checkpoint")
    if [ "$ft_local_size" -ne "$ft_size" ]; then
      echo "downloaded $ft_checkpoint is $ft_local_size bytes, S3 lists $ft_size." >&2
      exit 1
    fi
  fi
fi

# Gitignored build artifacts a clean clone lacks: the AprilTag textures are
# generated, and the camera calibration is machine-local -- the scene is subtly
# wrong without it and nothing else supplies it.
"$venv/bin/python" py/scripts/render_apriltag_textures.py --all-defaults >/dev/null
for calibration in config/camera_extrinsics/overhead_camera.json \
                   config/camera_intrinsics/overhead_camera.json \
                   config/camera_intrinsics/wrist_camera.json; do
  if [ ! -f "$repo/$calibration" ]; then
    mkdir -p "$(dirname "$repo/$calibration")"
    aws s3 cp "$bucket_root/config-backup/${calibration#config/}" \
      "$repo/$calibration" --only-show-errors
  fi
done

export PYTHONPATH="$repo/third_party/dppo"
export DPPO_LOG_DIR="$workspace/unused" DPPO_DATA_DIR="$artifact_root" \
       DPPO_BASE_POLICY="$checkpoint"

"$venv/bin/python" py/scripts/check_dppo_rl_env.py \
  --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \
  --checkpoint "$checkpoint" \
  --normalization "$artifact_root/normalization.npz" \
  --expect-action-encoding "$action_encoding" \
  --scene-appearance "$scene_appearance" \
  --scene-seed-base "$scene_seed_base" \
  --episodes "$episodes" \
  --n-envs "$n_envs" \
  --seed "$seed" \
  --device cuda:0 \
  ${act_steps:+--act-steps "$act_steps"} \
  ${ft_checkpoint:+--finetuned-checkpoint "$ft_checkpoint"} \
  --output "$scores/$SCORE_NAME.json"

aws s3 cp "$scores/$SCORE_NAME.json" \
  "$bucket_root/outputs/dppo-env-scores/$SCORE_NAME.json" --only-show-errors
if [ -n "$ft_checkpoint" ]; then
  echo "Scored $ft_run_name itr $ft_itr (over $BASE_RUN_NAME epoch $base_epoch) as $SCORE_NAME."
else
  echo "Scored $BASE_RUN_NAME epoch $base_epoch as $SCORE_NAME."
fi
