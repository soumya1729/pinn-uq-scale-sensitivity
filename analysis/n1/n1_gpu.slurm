#!/usr/bin/env bash
#SBATCH --job-name=pinn-n1-gpu
#SBATCH --account=iisertvm
#SBATCH --partition=gpu
#SBATCH --qos=gpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=analysis/n1/logs/n1-gpu-%A_%a.out

set -e
cd "$HOME/unified_suite"
source slurm/env.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -c 'import jax, numpy, scipy; print("Versions:", jax.__version__, numpy.__version__, scipy.__version__); print("Devices:", jax.devices())'

seeds=(2 0 1 3 4 5 6 7)
out=analysis/n1/outputs_cpu
mkdir -p "$out"

for ((i=SLURM_ARRAY_TASK_ID; i<${#seeds[@]}; i+=SLURM_ARRAY_TASK_COUNT)); do
    tag="burgers_mlp_seed${seeds[i]}"

    if [ -f "$out/${tag}_n1.json" ]; then
        echo "Already completed: $tag"
        continue
    fi

    echo "Starting $tag at $(date -Is)"
    python analysis/n1/n1_sensitivity.py \
        --run-dir "analysis/n5/input/$tag" \
        --out "$out"
    echo "Completed $tag at $(date -Is)"
done

echo "Assigned seeds completed for task $SLURM_ARRAY_TASK_ID."
