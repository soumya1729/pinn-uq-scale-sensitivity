#!/usr/bin/env bash
#SBATCH --job-name=pinn-round2
#SBATCH --account=iisertvm
#SBATCH --partition=gpu
#SBATCH --qos=gpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=analysis/round2/logs/round2-%A_%a.out

set -e
cd "$HOME/unified_suite"

array_job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
record="analysis/round2/cluster_record/${array_job_id}_${array_task_id}"
mkdir -p "$record" analysis/round2/outputs/production
cp "$0" "$record/executed.slurm"
cp slurm/env.sh "$record/env.sh"
cp analysis/round2/*.py analysis/round2/cohort.json "$record/"
cp analysis/n1/n1_sensitivity.py "$record/n1_sensitivity.py"
date -Is > "$record/start.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$record/job_id.txt"
printf '%s\n' "$@" > "$record/submission_arguments.txt"

source slurm/env.sh
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -c 'import jax, numpy, scipy; print("Versions:", jax.__version__, numpy.__version__, scipy.__version__); print("Devices:", jax.devices())'

tag_text=$(python - "$array_task_id" "$@" <<'PY'
import json, sys
with open("analysis/round2/cohort.json") as f:
    manifest = json.load(f)
if len(sys.argv) > 3:
    raise SystemExit("Pass at most one cohort tag for a single-checkpoint pilot.")
if len(sys.argv) == 3:
    tag = sys.argv[2]
    if tag not in {entry["tag"] for entry in manifest["cohort"]}:
        raise SystemExit(f"Pilot tag is outside the fixed cohort: {tag}")
    tags = [tag]
else:
    tags = manifest["groups"][int(sys.argv[1])]
print("\n".join(tags))
PY
)
mapfile -t tags <<< "$tag_text"
printf '%s\n' "${tags[@]}" > "$record/assigned_tags.txt"
printf 'tag\tstarted\tfinished\texit_code\tcomputation_status\n' > "$record/task_status.tsv"
: > "$record/failed_tags.tsv"
failed=0

for tag in "${tags[@]}"; do
    started=$(date -Is)
    echo "Starting or resuming $tag at $started"
    if python analysis/round2/round2_pass.py \
        --tag "$tag" \
        --inputs analysis/n5/input \
        --cohort analysis/round2/cohort.json \
        --out analysis/round2/outputs/production \
        --resume; then
        exit_code=0
        if status=$(python - "$tag" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(f"analysis/round2/outputs/production/{sys.argv[1]}_round2.json").read_text())
print(f"COMPUTED_OR_RESUMED:{summary['numerical_status']}")
PY
        ); then
            echo "$tag: $status (numerical review is separate from computation completion)."
        else
            exit_code=2
            status="UNREADABLE_COMPLETION"
        fi
    else
        exit_code=$?
        status="FAILED"
    fi
    finished=$(date -Is)
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$started" "$finished" "$exit_code" "$status" >> "$record/task_status.tsv"
    if (( exit_code != 0 )); then
        failed=1
        printf '%s\t%s\n' "$tag" "$exit_code" >> "$record/failed_tags.tsv"
        echo "Failed $tag (exit $exit_code); continuing with the remaining assigned checkpoints."
    fi
done

date -Is > "$record/end.txt"
if (( failed )); then
    echo "One or more assigned checkpoints failed; see $record/failed_tags.tsv."
    exit 1
fi
echo "Assigned computations completed or resumed for array task $array_task_id; review numerical status in each summary."
