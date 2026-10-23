#!/usr/bin/env bash
# GPU environment copied from the successful round-two execution record.
# This file belongs to the isolated theta-followup project.
module purge 2>/dev/null || true
module load python/3.12.5
source "$HOME/envs/pinn/bin/activate"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# The installed Python runtime is shared; all project files and results are local.
export THETA_FOLLOWUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$THETA_FOLLOWUP_ROOT"
