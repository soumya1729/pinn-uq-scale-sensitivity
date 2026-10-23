#!/usr/bin/env bash
# ============================================================================
# env.sh -- cluster-specific settings for padmanabha (IISER-TVM). Sourced by
# every job. Filled for: partition=gpu (gnode1:2 + gnode2:1 = 3 Tesla GPUs),
# modules python/3.12.5 + cuda/12.5, 28 cores / 128 GB per node.
# ============================================================================
module purge 2>/dev/null || true
module load python/3.12.5          # provides python3.12 + venv + pip

# --- python env with JAX(GPU)+optax+blackjax (created once, see setup below) ---
source "$HOME/envs/pinn/bin/activate"

# NOTE: we deliberately do NOT `module load cuda` — the pip wheel "jax[cuda12]"
# ships its own CUDA 12 + cuDNN libraries and only needs the GPU driver
# (always present on gnode*). If jax ever fails to see the GPU, uncomment:
# module load cuda/12.5

# --- JAX / XLA runtime knobs (safe defaults) --------------------------------
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # share gracefully on a 3-GPU queue
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_command_buffer="   # OOM workaround that worked before
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1                  # stream python logs live (no block-buffering to the .out file)

export SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUITE_DIR"

# ============================================================================
# ONE-TIME SETUP — run once on the login node (padmanabha1) BEFORE sbatch:
#   module load python/3.12.5
#   python -m venv $HOME/envs/pinn
#   source $HOME/envs/pinn/bin/activate
#   pip install -U pip
#   pip install -U "jax[cuda12]" optax numpy scipy matplotlib blackjax
#   # sanity check ON A GPU NODE (not the login node):
#   srun --partition=gpu --gres=gpu:1 --time=00:05:00 --pty \
#        bash -lc 'module load python/3.12.5; source $HOME/envs/pinn/bin/activate; \
#                  python -c "import jax; print(jax.devices())"'   # must show a GPU
# ============================================================================
