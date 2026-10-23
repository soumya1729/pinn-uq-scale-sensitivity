# Isolated theta-centred study: cluster commands

Local project: `C:\dev\shortened_astra\pinn_theta_followup`.
Cluster project: `~/pinn_theta_followup`.

This study has its own inputs, scripts, selections, outputs, logs and execution records. Nothing writes to `~/unified_suite` or the first paper. The GPU environment is a project-owned copy of the previously successful settings; it activates the existing installed `~/envs/pinn` runtime. Reusing that runtime does not reuse the old analysis scripts or output directories.

The author has confirmed that `/home` is not purged and authorised all-17 retention.
The shared filesystem reports 206 TB free; `quota -s` returned no output, rather
than an explicit unlimited-quota statement. The prepared configuration uses
`~/pinn_theta_followup/retained`.

The sequence is: upload, verify inputs and configuration, run Burgers-MLP seed 2,
review it, and only then submit all 17 checkpoints. The saved-spectrum selection
is already complete locally. Preparation
and monitoring commands do not submit jobs. Each `sbatch` creates a new job; do not
repeat a submission block to check progress.

## Upload from WSL

Run from a local WSL terminal. No deletion options are used.

```bash
cd /mnt/c/dev/shortened_astra/pinn_theta_followup
ssh soumya24@padmanabha.iisertvm.ac.in \
  'mkdir -p "$HOME/pinn_theta_followup/logs" "$HOME/pinn_theta_followup/cluster_records" "$HOME/pinn_theta_followup/outputs/production"'
rsync -av --exclude='outputs/production/' --exclude='logs/' \
  --exclude='cluster_records/' --exclude='__pycache__/' --exclude='.venv/' \
  --exclude='retained/' --exclude='cluster/archive_config.json' \
  ./ soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/

# First upload only: install the author-approved all-17 archive configuration.
rsync -av cluster/archive_config.json \
  soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/cluster/
```

Keep source files fixed once the pilot has been launched. Source changes require a separately reviewed protocol and new output namespace, rather than overwriting a running or completed study.

## Cluster input check

In the cluster login shell:

```bash
cd ~/pinn_theta_followup
mkdir -p logs cluster_records outputs/production
python3 scripts/check_inputs.py
```

The manifest-only check verifies 143 project-owned checkpoint, baseline and comparison files for all 17 runs. It needs no GPU or JAX. If `python3` is unavailable on the login node, use the available Python 3 installation. Do not move this job to the CPU `medium` partition using the GPU environment: that environment previously failed there. The completed local selections in `outputs/evidence/` and the five root settings in `outputs/local/root_settings.json` are included in the upload.

## Archive configuration

The provided actual `cluster/archive_config.json` is already confirmed for this
launch. Verify it after upload using the block below; no further confirmation or
editor change is needed. The capacity commands and alternative configuration
instructions are retained for a later deliberate change of storage location.

On the cluster login node, inspect capacity and user quota:

```bash
df -h "$HOME"
quota -s
```

If quotas are managed through another site-specific command, obtain that quota
instead. Also confirm the site's retention policy: capacity does not establish
whether the location is persistent or periodically purged.

Keeping float64 factors and signed grid modes needs approximately **276 GB for all
17 checkpoints**, or **118 GB for the eight Burgers-MLP checkpoints**. These numbers
exclude existing inputs, ordinary outputs, temporary files and other storage use.
Both choices still compute the numerical comparisons for **all 17**.

For a future deliberate change, after checking quota and permanence, create/edit
the actual configuration:

```bash
cd ~/pinn_theta_followup
cp cluster/archive_config.example.json cluster/archive_config.json
nano cluster/archive_config.json
```

Set `retain` to `all` or `burgers-mlp`, choose the confirmed persistent
`archive_root`, and set `storage_confirmed` to `true`. The example's `retained`
path means `~/pinn_theta_followup/retained`. If that exceeds the home quota, name
a **new study-owned directory** on confirmed persistent project storage instead.
Keep it separate from `outputs/`, `inputs/` and the earlier `~/unified_suite`
project. Do not invent a site project-storage path or use purgeable scratch as
the only copy. Leave the confirmation false if storage is not settled.

Check the chosen configuration before creating a job:

```bash
python3 - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('cluster/archive_config.json').read_text())
assert c['storage_confirmed'] is True, 'Confirm quota and retention policy first.'
assert c['retain'] in ('all', 'burgers-mlp')
print('Retain:', c['retain'])
print('Archive directory:', Path(c['archive_root']).expanduser().resolve())
PY
```

The Slurm wrapper requires this configuration and records the executed copy and
resolved path. The main source sync excludes the actual configuration. The separate
first-upload command installs this author-approved choice; do not repeat it after
making a deliberate cluster-side storage change.

## Submit the single-checkpoint pilot once

```bash
cd ~/pinn_theta_followup
sbatch --test-only --array=0 --job-name=pinn-theta-pilot \
  --output=logs/pilot-%A_%a.out \
  cluster/theta_gpu.slurm burgers_mlp_seed2
```

Once the request is accepted:

```bash
pilot_id=$(sbatch --parsable --array=0 --job-name=pinn-theta-pilot \
  --output=logs/pilot-%A_%a.out \
  cluster/theta_gpu.slurm burgers_mlp_seed2)
printf '%s\n' "$pilot_id" | tee cluster_records/pilot_job_id.txt
squeue -j "$pilot_id" -r
```

## Monitor and review the pilot

These commands do not submit another job:

```bash
cd ~/pinn_theta_followup
pilot_id=$(cat cluster_records/pilot_job_id.txt)
squeue -j "$pilot_id" -r
tail -n 60 "logs/pilot-${pilot_id}_0.out"
sacct -j "$pilot_id" --format=JobID%24,State,Elapsed,MaxRSS,ExitCode \
  | tee "cluster_records/accounting-${pilot_id}.txt"
```

Wait until the pilot finishes. Review its prediction/loss reconstruction, baseline
covariance agreement, independent linear-algebra checks, nonlinear-versus-affine
results and archive reload checks. The archive manifest records layout, numerical
libraries and checksums. Reload validation must reproduce a saved variance field
and zero-prior shift field and apply the Householder factors to theta and saved
sample gradients. A completed computation is not by itself scientific validation.
Review every numerical warning before launching the remaining cohort. Its
accounting record provides the measured runtime and peak memory; the resource
request is not a runtime estimate.

After completion, print the pilot summary:

```bash
python3 - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('outputs/production/burgers_mlp_seed2_theta.json').read_text())
for key in ('numerical_status', 'elapsed_seconds', 'baseline_max_sigma_relative_error',
            'grid_checks', 'evidence_rebuild_checks', 'delta_theta_relative_norm',
            'normal_equation_checks', 'archive_validation_report'):
    print(key, ':', r[key])
PY
```

Its four routine output files are `burgers_mlp_seed2_theta.json`, `.csv`,
`burgers_mlp_seed2_theta_fields.npz` and `burgers_mlp_seed2_theta_shifts.npz`.
The retained factors and grid modes live under the separately configured archive
root, in a checkpoint-specific directory, and are not part of these four files.
An interior or boundary selector is a model outcome; `REVIEW_REQUIRED` is a
numerical flag that needs inspection. A small mean-check residual does not certify
weak-mode forward accuracy.

## Submit all 17 checkpoints once, after pilot review

The three manifest groups contain 6, 6 and 5 checkpoints, with two Fourier checkpoints in each group. At most two array tasks run concurrently. The completed compatible pilot is resumed in its group, so the cohort remains 17 checkpoints.

Check that the pilot is no longer running, then test the full request:

```bash
cd ~/pinn_theta_followup
pilot_id=$(cat cluster_records/pilot_job_id.txt)
squeue -j "$pilot_id" -r
sbatch --test-only cluster/theta_gpu.slurm
```

**Stop here if the pilot is still queued or running, or if its numerical results have not been reviewed.** When those checks are complete, submit once:

```bash
job_id=$(sbatch --parsable cluster/theta_gpu.slurm)
printf '%s\n' "$job_id" | tee cluster_records/full_job_id.txt
squeue -j "$job_id" -r
```

Subsequent monitoring uses the recorded ID:

```bash
cd ~/pinn_theta_followup
job_id=$(cat cluster_records/full_job_id.txt)
squeue -j "$job_id" -r
tail -n 35 logs/theta-${job_id}_*.out
sacct -j "$job_id" --format=JobID%24,State,Elapsed,MaxRSS,ExitCode \
  | tee "cluster_records/accounting-${job_id}.txt"
```

If a task fails, inspect its execution record before resubmitting. Compatible finished checkpoints may resume, but never run two attempts for the same checkpoint simultaneously. If cancellation is necessary, target the specific recorded job ID; do not cancel all of the user's jobs.

## Download results and every cluster wrapper

Run in local WSL. The entire `cluster/` directory is downloaded, so any ad hoc Slurm file added there comes back too. The wrapper's executed copy and environment snapshot are also retained in `cluster_records/`. Keep any newly created wrapper inside `cluster/`.

These commands download ordinary outputs and execution records only. **Do not
rsync the whole remote project**: that would also copy `retained/`. The large
archives stay in the configured persistent cluster location. The C: drive has
insufficient space for all of them; any later explicit archive transfer needs a
separate destination with enough capacity and must include the manifests and
checksums. No deletion options are used.

```bash
cd /mnt/c/dev/shortened_astra/pinn_theta_followup
rsync -av soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/outputs/ outputs/
rsync -av soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/logs/ logs/
rsync -av soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/cluster/ cluster/
rsync -av soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/cluster_records/ cluster_records/
rsync -av --include='*/' --include='manifest.json' --exclude='*' \
  soumya24@padmanabha.iisertvm.ac.in:~/pinn_theta_followup/retained/ \
  cluster_records/retained_manifests/
```

The last command retrieves only the small retained-archive manifests and their
directory structure; all large `.npy` arrays are excluded. Before downloading,
save final pilot/full-array accounting on the cluster:

```bash
cd ~/pinn_theta_followup
sacct -j 471403,471423 --format=JobID%24,State,Elapsed,MaxRSS,ExitCode \
  | tee cluster_records/accounting-final-471403-471423.txt
```

Retain accounting and logs for any cancelled or duplicate attempt as well as the successful runs. No manuscript is uploaded, and no results are copied into the first paper automatically.

This pass is limited to theta-centred predictions, nonlinear checks, root fields
and reusable archives. It does not select or factor a weighted residual model.
The grid-mode archive covers the original 400-by-200 grid; future new-point
experiments still require their own prediction-gradient evaluations.
