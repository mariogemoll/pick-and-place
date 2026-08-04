#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Bring a freshly rented pod to the state every pick-and-place job script
# assumes: repo cloned with submodules, working-tree overlay applied, DPPO
# patch applied, venv built with the RTX-5090 overrides, CUDA usable.
#
#   scp scripts/vast_pap_provision.sh overlay.tar.gz <ssh-host>:/workspace/
#   ssh <ssh-host> 'bash /workspace/vast_pap_provision.sh'
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
git submodule update --init --recursive
git rev-parse HEAD

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

if ! git -C third_party/dppo apply --reverse --check \
     ../../config/diffusion_policy/dppo-generic-obs-keys.patch 2>/dev/null; then
  git -C third_party/dppo apply ../../config/diffusion_policy/dppo-generic-obs-keys.patch
  echo "Applied dppo-generic-obs-keys.patch to the vendored agent."
else
  echo "dppo-generic-obs-keys.patch already applied."
fi

base_python="python3"
if [ -x /venv/main/bin/python ]; then
  base_python="/venv/main/bin/python"
fi
if [ ! -x "$venv/bin/python" ]; then
  uv venv --python "$base_python" "$venv"
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
