# Detailed producer commands

For the concise reviewer route, start with the top-level README and `reproduce.py`.
The commands below document the underlying producers, including optional expensive
recomputation. They are not all required for saved-data replay.

Run commands from the bundle root unless stated otherwise. Install the dependencies for the selected stage first; see `ENVIRONMENTS.md`. `regenerated/` is scratch output, separate from the frozen comparison records.

## 1. Replay the inexpensive saved-data results

The included small CSV/JSON records already support many tables and figures. Restore the production arrays to recompute the all-17 centre, evidence, oracle and third-round summaries. After obtaining the separate data asset:

```bash
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip --group "saved outputs: round2"
```

This group also contains archived toy/validation arrays. The production quartet required for each run is `<tag>_round2.json`, `.csv`, `_fields.npz`, `_spectral.npz` under `analysis/round2/outputs/production`.

Recompute the numerical review into a directory that does not already exist:

```bash
python analysis/round2/review_production.py --production analysis/round2/outputs/production --cohort analysis/round2/cohort.json --out regenerated/production_review
python analysis/round3_local/review_round3.py --out regenerated/round3
python analysis/recompute_saved_summaries.py
```

The third-round consumer reads the production arrays directly and does not use a new Jacobian or network evaluation. `recompute_saved_summaries.py` reads bundled scalar JSON inputs and writes `analysis/rms_bounds.csv`, `analysis/saved_scalar_analysis.json` and `generated/` macros.

For final typesetting, the table consumers below read the **included frozen review snapshot**. The regenerated review can be compared with that snapshot before using it as a replacement. They preserve manual formatting separately in the paper source ZIP:

```bash
python analysis/round2/generate_tables.py --production analysis/round2/outputs/production --review analysis/round2/outputs/production_review --out regenerated/generated
python analysis/round2/report_saved_summaries.py --source . --tex-out regenerated/generated
```

The last command writes its reports to `analysis/round2/outputs/reporting_closure`; it reproduces the included records there. That location also holds the frozen `source_control_record.json` needed for the retained linear control. If selecting a different `--out`, copy this control record there first.

Current figures:

```bash
python analysis/round2/plot_shortened_records.py --source-root . --paper-root regenerated/paper
python analysis/round2/plot_shortened_coverage.py --source-root . --paper-root regenerated/paper
python analysis/round2/plot_shortened_slice.py --source-root . --paper-root regenerated/paper
python analysis/round2/plot_merged_family_curves.py --source-root . --paper-root regenerated/paper
```

The first command creates the destination figure directory. The merged-family producer creates the current four-row Burgers figure and two-row Allen–Cahn figure, reducing the appendix to two family-curve pages without removing runs or diagnostics. `plot_shortened_records.py` also emits historical individual-family assets that are not used by the current manuscript. The previous `plot_combined_burgers.py` is retained for source history and is not a current figure-build step.

The slice plot reads the unchanged frozen CSV. To rebuild that CSV as well, restore the checkpoint and raw-baseline groups below, then run `python tools/export_burgers_slice.py`. Its output goes to `regenerated/burgers_slice.csv`; compare it with `analysis/burgers_slice.csv` before replacing the figure input.

The original eight-run N1 table is distinct from the all-17 family summaries:

```bash
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip --group "saved outputs: n1"
python analysis/n1/summarize_n1.py
```

## 2. Recompute from frozen checkpoints

Restore the input groups:

```bash
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip --group "17 frozen checkpoints and fields" "raw baseline covariance fields: round2"
python analysis/n5/n5_heldout.py --out regenerated/n5_all17.csv
python analysis/round2/ensemble_loo.py --out regenerated/ensemble
python analysis/round2/round2_pass.py --input-check
```

`n5_heldout.py` uses NumPy, the saved 129 features and the published random and ten-time-slice-block protocols. `summarise_n5.py` writes a note from `analysis/n5/outputs/n5_all17.csv`; its historical follow-up-status prose is not the current scientific status.

The all-17 production pass is expensive and is **not required to replay saved results**. When deliberately rerunning it, use a fresh destination:

```bash
python analysis/round2/round2_pass.py --tag burgers_fourier_seed2 --out regenerated/production
python analysis/round2/round2_pass.py --out regenerated/production --resume
```

The second command processes the fixed cohort and resumes the compatible pilot. It retains the archived raw residual weights and uses one direct-J QR/SVD factorisation per checkpoint. Historical production timing was about 1.6–3 hours per checkpoint on the recorded GPU setup; this is not a runtime prediction for another machine.

`round2_gpu.slurm` is the **actual cluster wrapper**, with site-specific account, partition, paths and environment. It covers three fixed groups with at most two concurrent tasks. Adapt those site fields and create `analysis/round2/logs` before submitting elsewhere. The preserved executed scripts and accounting are under `provenance/round2_cluster`. Do not submit the original N1 and all-17 studies together as though both were needed: the all-17 pass supersedes the early experiment for current results, while the early archives remain the source of Table 6.

## 3. Original baseline, controls and admission accounting

Restore `original baseline and reference inputs` and `retained grid-resolution check inputs`. All 55 attempted-run metadata records are already included. Run these from **`training/`** so the original relative result paths resolve:

```bash
python analysis/solve_gate_table.py --check
python analysis/ensemble_calibration.py --check
python analysis/temporal_per_seed.py --check
python analysis/ceiling_table.py --check
python analysis/offmanifold_summary.py --check
python analysis/region_resolution.py --check
```

These entry points include source/hash contracts with their matching recovered baseline producers. Consult `PAPER_OUTPUT_MAP.md` for the corresponding inputs. Projection arrays, raw sigma fields and their metadata are separate from the modern production quartets.

Original covariance producers read environment configuration through `common.get_config`; use the entire saved configuration for the selected tag, not today's defaults. The raw baseline entry point requires `RAW_METRIC=1`; the raw sigma-dump route additionally sets `SIGMA_DUMP_ONLY=1`. `m4_full_laplace.py`, `step_fullgrid_ratio.py`, `d4_manifold.py` and the loading/projector scripts are included for this deliberately expensive rebuild. Do not run them simply to reformat a table.

`legacy_core/verify_reference.py` supplies independent reference checks; `legacy_core/d4_baselines.py` supplies the retained masked-global example. Run within that folder with its matching helpers. The latter also evaluates the saved model: select Burgers–MLP seed 2 and set `MAP_PKL=../training/results/burgers_mlp_seed2/burgers_mlp_seed2_params.pkl` and a separate `OUTDIR`. Its default `legacy_core/results` folder is not an independent copy of the checkpoint collection. `legacy_core/acref.slurm` records the original Allen–Cahn reference refinement and needs site adaptation.

## 4. Training and numerical validation

The training implementation is `training/train_map.py`; it takes environment settings described in `training/common.py`. `provenance/training_configs.json` preserves every saved configuration, original environment record, admission threshold and package freeze for all 17 checkpoints. The paper uses the raw training arm (`WHITEN_TRAIN=raw`). Run new training from `training/` with a separate `OUTDIR`; preserve the frozen checkpoints.

Training a new solution does not guarantee the exact archived solution, and the recorded checkpoints do not prove training-time code identity. Earlier release sources are retained separately under `provenance/historical_release`.

Small numerical checks:

```bash
python analysis/round2/check_linalg.py
python analysis/round2/validate_round2.py --out regenerated/validation
python analysis/round2/integration_checks.py --out regenerated/validation/integration_summary.json
```

`stress_evidence_toy.py` is the additional fixed synthetic-spectrum stress validation: 21.2 decades of eigenvalues, 79 of 96 modes below the formed-Gram rounding floor, and 100 draws. It has no checkpoint dependency and is not a measured resolved PINN spectrum. It writes `analysis/round2/outputs/validation/gram_floor_stress`; the check-linalg script likewise writes its fixed preparation location. Run those two in a working copy if preserving all frozen reference outputs byte-for-byte. These tests establish the numerical claims described in their reports, not blanket accuracy for every weak mode.

Formal development, from `lean/`:

```bash
lake exe cache get
lake --wfail build
lake env lean Audit.lean
```

The pinned toolchain and manifest define Lean 4.30.0 and its matching Mathlib release.
The first command obtains the compiled dependency cache; a fresh installation may
download dependencies. The warning-failing build checks the complete focused
`PaperReliability` library, and `Audit.lean` prints the axioms of all 15 principal
declarations. Standard Lean logical axioms are permitted; `sorryAx` and
project-defined axioms are not.

The seven mathematical modules include the covariance bridge, fixed-cutoff contrast
bounds and sharpness witnesses, prior-centre prediction and coverage bounds,
positive-modal calibrated-coverage envelopes, conditional strong/weak precision
limits and their nonmonotone counterexample. `lean/THEOREM_MAP.md` and the paper's
Table 18 state the exact assumptions. Coverage uses fixed trained errors and an
actual empirical higher quantile; the root-envelope result is conditional on the
precision lying within the supplied bracket. The covariance bridge assumes an exact
orthogonal spectral representation.

These proofs establish real-model analytic results. They do not certify the numerical
QR/SVD, saved arrays, empirical 16/17 or 14/17 run counts, or computed root brackets.
The first-order strong-precision expansion and the paper's secondary mean-width and
projection remarks remain outside this focused development. No training, Jacobian
pass or additional numerical experiment is needed to build the formal project.

## Final saved-field diagnostics

After the existing round-two and round-three saved outputs are available:
`python analysis/round4_local/review_round4.py --project-root .`
This reads all seventeen field archives, checks earlier descriptive summaries,
and regenerates the regional-calibration table. It does not repeat production.

## 5. Completed theta-centred study - 16 October 2026

The isolated study is collected under `analysis/theta_centred`; it does not import
the older production scripts or write into their output folders. All 17
checkpoints have completed production and both reviews. No new cluster pass is
needed to reproduce the revised paper tables.

Restore the recorded data from the portable data asset, from the repository root:

```bash
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip --group "theta-centred: frozen inputs" "theta-centred: production fields and shifts" "theta-centred: execution logs"
```

The portable data asset restores these manifest destinations on any machine.
A download URL will be added after publication of the release. The restore tool
never launches computation; see `../data/README.md`.

Run saved-output reviews after restoration, using NumPy and the recorded core
versions in `analysis/theta_centred/requirements-cpu.txt`:

```bash
python analysis/theta_centred/scripts/check_inputs.py
python analysis/theta_centred/scripts/review_metadata.py
python analysis/theta_centred/scripts/summarize_theta.py
python analysis/theta_centred/scripts/make_paper_tables.py --out regenerated/theta_tables
```

The review commands write the study's reviewed CSV/JSON records to
`analysis/theta_centred/outputs/review`; they recompute and overwrite those small
comparison records. Run in a copy if their original file bytes must be retained.
`summarize_theta.py` reads saved fields/shifts and evaluates a few deterministic
forward predictions with NumPy; it does not build a Jacobian. The table producer
alone needs only the included reviewed CSVs and `inputs/refined_roots.csv`.

To recompute the evidence selection from original saved spectra, restore the
theta frozen-input group and run `select_theta_evidence.py`; preserve the included
selection files when reproducing the historical production signatures, because
rewriting JSON in another environment can change their byte hashes. The archive
and theta validators are small standalone tests with the study's SciPy/JAX core.

`CLUSTER_COMMANDS.md` and the executed `cluster/theta_gpu.slurm` document the
deliberate expensive route, not a necessary table-build step. They use the
original independent remote root `~/pinn_theta_followup`; adapt path/account,
GPU environment and confirmed persistent storage for a new site. Both factors
and grid modes were retained on that cluster. Those 276 GB are not in the source
or ordinary data-restore groups; the 17 downloaded archive manifests record
orientation, precision, checksums and reload checks. Existing-grid modal data do
not replace gradients at future off-grid points.
