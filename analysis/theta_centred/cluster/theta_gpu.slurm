#!/usr/bin/env bash
#SBATCH --job-name=pinn-theta
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
#SBATCH --output=logs/theta-%A_%a.out

set -e
cd "$HOME/pinn_theta_followup"
array_job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
record="cluster_records/${array_job_id}_${array_task_id}"
mkdir -p "$record" outputs/production
cp "$0" "$record/executed.slurm"
cp cluster/env.sh "$record/env.sh"
cp -r scripts "$record/scripts"
cp manifest.json "$record/manifest.json"
date -Is > "$record/start.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$record/job_id.txt"
printf '%s\n' "$@" > "$record/submission_arguments.txt"

source cluster/env.sh
archive_settings=$(python - <<'PY'
import json
from pathlib import Path
config_path = Path("cluster/archive_config.json")
if not config_path.is_file():
    raise SystemExit("Create cluster/archive_config.json after confirming storage quota and retention policy.")
config = json.loads(config_path.read_text())
if config.get("storage_confirmed") is not True:
    raise SystemExit("Storage has not been confirmed in cluster/archive_config.json.")
retain = config.get("retain")
if retain not in ("all", "burgers-mlp"):
    raise SystemExit("retain must be all or burgers-mlp.")
archive_root = config.get("archive_root")
if not isinstance(archive_root, str) or not archive_root.strip():
    raise SystemExit("archive_root must name the confirmed persistent storage directory.")
print(retain)
print(Path(archive_root).expanduser().resolve())
PY
)
mapfile -t archive_settings_array <<< "$archive_settings"
retention="${archive_settings_array[0]}"
archive_root="${archive_settings_array[1]}"
cp cluster/archive_config.json "$record/archive_config.json"
printf '%s\n' "$archive_root" > "$record/resolved_archive_root.txt"
echo "Archive retention: $retention; destination: $archive_root"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -c 'import jax, numpy, scipy; print("Versions:", jax.__version__, numpy.__version__, scipy.__version__); print("Devices:", jax.devices())'

tag_text=$(python - "$array_task_id" "$@" <<'PY'
import json, sys
with open("manifest.json") as f:
    cohort = json.load(f)
if len(sys.argv) > 3:
    raise SystemExit("Pass at most one checkpoint tag for a pilot.")
if len(sys.argv) == 3:
    tag = sys.argv[2]
    if tag not in {entry["tag"] for entry in cohort["runs"]}:
        raise SystemExit(f"Unknown pilot tag: {tag}")
    tags = [tag]
else:
    tags = cohort["groups"][int(sys.argv[1])]
print("\n".join(tags))
PY
)
mapfile -t tags <<< "$tag_text"
printf '%s\n' "${tags[@]}" > "$record/assigned_tags.txt"
printf 'tag\tstarted\tfinished\texit_code\n' > "$record/task_status.tsv"
failed=0
for tag in "${tags[@]}"; do
    started=$(date -Is)
    echo "Starting or resuming $tag at $started"
    if python scripts/theta_pass.py --tag "$tag" --resume \
        --retain "$retention" --archive-root "$archive_root"; then
        exit_code=0
    else
        exit_code=$?
        failed=1
        echo "Failed $tag (exit $exit_code); continuing with the remaining assigned checkpoints."
    fi
    printf '%s\t%s\t%s\t%s\n' "$tag" "$started" "$(date -Is)" "$exit_code" >> "$record/task_status.tsv"
done
date -Is > "$record/end.txt"
if (( failed )); then
    echo "One or more assigned computations failed; see $record/task_status.tsv."
    exit 1
fi
echo "Assigned computations completed or resumed. Inspect the numerical checks in each result before interpretation."
