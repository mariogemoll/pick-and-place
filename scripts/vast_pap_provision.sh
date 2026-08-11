#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Bring a freshly rented pod to the state every pick-and-place job script
# assumes: repo cloned with submodules, working-tree overlay applied, DPPO
# patch applied, venv built with the RTX-5090 overrides, CUDA usable.
#
#   scp scripts/vast_pap_provision.sh overlay.tar.gz ~/.netrc <ssh-host>:/workspace/
#   ssh <ssh-host> 'bash /workspace/vast_pap_provision.sh'
#
# Set PAP_BRANCH to run a pod on a pushed branch rather than main.
#
# The netrc carries the Weights & Biases credential. Copy it every time: the job
# launchers look for it on the *pod*, and having one on the controller does
# nothing. Nothing used to put it there, so every run in this repository's
# history logged to nowhere.
#
# vast_diffusion_policy_train_fast.sh starts from this state and does not create
# it. Every step is idempotent, so re-running after a partial failure is safe.

set -euo pipefail

workspace="/workspace"
repo="$workspace/pick-and-place"
venv="$workspace/venvs/pick-and-place"

export DEBIAN_FRONTEND=noninteractive
apt-get update
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
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
grep -m1 'model name' /proc/cpuinfo

if [ ! -d "$repo/.git" ]; then
  git clone --recurse-submodules https://github.com/mariogemoll/pick-and-place.git "$repo"
fi
cd "$repo"
# PAP_BRANCH runs a pod on work that is not on main yet -- an experiment branch,
# say -- without falling back to the overlay for code that is already pushed.
if [ -n "${PAP_BRANCH:-}" ]; then
  git fetch origin "$PAP_BRANCH" || { echo "BRANCH_FETCH_FAILED $PAP_BRANCH"; exit 1; }
  git checkout -B "$PAP_BRANCH" "origin/$PAP_BRANCH" || { echo "BRANCH_CHECKOUT_FAILED"; exit 1; }
fi
git submodule update --init --recursive
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD

# The clone is only as current as origin/main; anything committed locally but
# unpushed, or still uncommitted, arrives in the overlay. Without it a pod can
# silently run older code than the machine that launched it.
if [ -f "$workspace/overlay.tar.gz" ]; then
  tar -xzf "$workspace/overlay.tar.gz" -C "$repo"
  sha256sum "$workspace/overlay.tar.gz"
  echo "Applied working-tree overlay."
else
  echo "No overlay tarball; running the committed tree." >&2
fi

# Pin the interpreter rather than inheriting the image's. vastai/pytorch:latest
# currently ships Python 3.10 in /venv/main, and this package requires 3.12+
# (lerobot 0.5.1), so inheriting it fails resolution outright -- and a template
# that happens to satisfy the floor today would still let a bump change the
# interpreter a run was measured on without anyone noticing.
if [ -f "$workspace/.netrc" ] || [ -f "$workspace/netrc" ]; then
  cp "$workspace/.netrc" "$HOME/.netrc" 2>/dev/null || cp "$workspace/netrc" "$HOME/.netrc"
  chmod 600 "$HOME/.netrc"
  echo "Installed netrc; W&B logging is available."
elif [ -f "$HOME/.netrc" ]; then
  echo "netrc already present."
else
  echo "No netrc staged at $workspace/.netrc -- job launchers will refuse to start." >&2
fi

#
# Not PYTHON_VERSION: the vastai/pytorch image exports that already, set to its
# own 3.10, so an override named that is silently supplied by the environment.
python_version="${PAP_PYTHON_VERSION:-3.12}"
if [ ! -x "$venv/bin/python" ]; then
  uv python install "$python_version"
  uv venv --python "$python_version" "$venv"
fi
# Load-bearing for resolution, not just CUDA: without the overrides DPPO's own
# pins conflict with this package and uv declares the requirements unsatisfiable.
uv pip install --python "$venv/bin/python" \
  --overrides config/diffusion_policy/torch-rtx5090.txt \
  -e py -e third_party/dppo

# CUDA forward compatibility is a data-center-GPU feature; on a GeForce card the
# image's compat libcuda fails every call with error 804 and ldconfig resolves to
# it ahead of the host driver.
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

"$venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "PROVISION COMPLETE"
