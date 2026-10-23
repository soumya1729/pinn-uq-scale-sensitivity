# Paper output map

This map follows `paper/full_paper.tex`, the final self-contained manuscript. It retains six displayed figures and 18 tables. A label alias does not create an additional output. Producer paths are relative to the repository root. Current figure assets are in `paper/figures/`; manuscript formatting files and generated TeX inputs are relative to `paper/` unless otherwise stated.

The numerical producers and the final typesetting are separate steps. Several tables were shortened or merged by hand in LaTeX; their numerical records can be regenerated, but their current layout is preserved in the paper source snapshot. This is not a claim that one command regenerates every manuscript byte.

## Figures

| Figure | Paper label | Current file | Producer and input |
| --- | --- | --- | --- |
| 1 | `fig:burgers-slice` | `burgers_error_uncertainty_slice.pdf` | `analysis/round2/plot_shortened_slice.py`; frozen `analysis/burgers_slice.csv`, rebuildable with the new `tools/export_burgers_slice.py` replay utility. The CSV contains the 400 spatial values for Burgers–MLP seed 2 at stored time index 100, nearest `t = 0.5`. |
| 2 | `fig:n1-scales` | `round2_contrast_and_gap.pdf` | `analysis/round2/plot_shortened_records.py`; reviewed `uncertainty_contrast_curves.csv` and `regional_scale_metrics.csv`. |
| 3 | `fig:r2-evidence-energy` | `round2_evidence_energy_reference.pdf` | `analysis/round2/plot_shortened_records.py`; reviewed `evidence_slope_groups.csv`, derived from the production spectral archives. |
| 4 | `fig:r2-coverage-scale` | `round2_coverage_scale.pdf` | `analysis/round2/plot_shortened_coverage.py`; reviewed `regional_scale_metrics.csv`, production summary JSON files and pointwise `_fields.npz` files for all 17 runs. |
| 5 | `fig:r2-full-burgers`, `fig:r2-post-bm`, `fig:r2-full-bf`, `fig:r2-post` | `round2_burgers_families_merged.pdf` | `analysis/round2/plot_merged_family_curves.py`; reviewed `uncertainty_contrast_curves.csv`. Four rows show Burgers–MLP full-time, Burgers–MLP post-formation, Burgers–Fourier full-time and Burgers–Fourier post-formation records; three columns show the retained diagnostics. All eight MLP and three Fourier runs are retained. |
| 6 | `fig:r2-full-ac`, `fig:r2-full-af` | `round2_allen_cahn_families_merged.pdf` | Same merged-figure producer and reviewed records. Two rows show the three Allen–Cahn MLP and three Allen–Cahn Fourier runs, with the same three diagnostic columns. |

`analysis/round2/plot_merged_family_curves.py` produces the current compact appendix layout. The previous `plot_combined_burgers.py` was recovered from `full_submission/analysis/round2` and is preserved as a historical producer of the separate Burgers family figures. `plot_shortened_records.py` still produces Figures 2 and 3 and also emits older individual family assets; those individual assets are not used in the current appendix layout.

**Figure 1 replay.** The original standalone CSV-export script was not recovered. The frozen CSV and its source description are retained unchanged, and the newly added `tools/export_burgers_slice.py` reconstructs the slice from the archived checkpoint `analysis/n5/input/burgers_mlp_seed2/burgers_mlp_seed2_fields.npz` and the raw covariance archive `analysis/round2/baseline/burgers_mlp_seed2_fullgrid_sigma_raw.npz`. It selects the stored time nearest `t = 0.5`, preserves all 400 spatial points and archived whole-grid masks, and writes numerical fields with 17 significant digits. It does not use the historical standard-deviation field inside the checkpoint or compute new slice-specific region thresholds. Arrays follow the original time-major `200 × 400` grid. `analysis/burgers_slice_README.md` records the original construction; the utility writes a separate regenerated CSV by default.

## Tables

| Table | Paper label(s) | Numerical producer and input | Final manuscript formatting |
| --- | --- | --- | --- |
| 1 | `tab:rejected` | `training/analysis/solve_gate_table.py`; metadata for all 55 attempted configurations, including rejected runs and five linear controls. The admitted 17 checkpoints alone are insufficient. | Manually formatted in `j_setting.tex`. |
| 2 | `tab:ceiling`, `tab:absolute` | `analysis/round2/report_saved_summaries.py` gives `baseline_per_run.csv` from reviewed `regional_scale_metrics.csv`; `training/analysis/ceiling_table.py` provides the original baseline summaries and controls. | Merged baseline contrast and absolute-scale table in `j_ceiling.tex`. |
| 3 | `tab:temporal` | `training/analysis/temporal_per_seed.py`; original Burgers–MLP fields and raw full-grid sigma archives, eight seeds, split at `t = 1/π`. | Manually formatted pre/post table in `j_ceiling.tex`. |
| 4 | `tab:offmanifold` | Same-grid: `training/d4_manifold.py` and `training/analysis/offmanifold_summary.py`. Held-out: `analysis/n5/n5_heldout.py`, then `summarise_n5.py`, using the frozen parameters, feature maps and solution fields for all 17 admitted runs. | Four canonical rows and held-out columns merged in `j_offmanifold.tex`; the complete N5 records retain all runs and regions. |
| 5 | `tab:spectrum`, `tab:loading` | Original spectral/loading JSON records: `training/d_loading_perpoint.py`, `d_loading_cdf.py`, `analysis/loading_table.py` and `analysis/loading_table_tex.py`. RMS bound macros: `analysis/recompute_saved_summaries.py` with its frozen JSON input manifest. | Uses `generated/loading_table_values.tex` and `generated/rms_bounds_macros.tex`; the combined table body is in `j_mechanism.tex`. |
| 6 | `tab:n1-scales` | `analysis/n1/n1_sensitivity.py` produces the original eight Burgers–MLP N1 archives; `summarize_n1.py` supplies the operating, noise-MLE, evidence and sampled-minimum summaries. | Manually compressed in `short_seed_table.tex`. |
| 7 | `tab:r2-limits` | `analysis/round2/review_production.py` gives `spectral_limits.csv`; `generate_tables.py` produces `round2_limits_table.tex`. Inputs are the production modal energies and nearest sampled contrast-agreement records. | Generated table included in `j_round2_appendix.tex`. |
| 8 | `tab:r2-evidence-summary` | `review_evidence.py`, called by `review_production.py`, gives `evidence_best.csv`, `evidence_perturbations.csv` and `evidence_summary.json` from the saved spectral archives. | Family summary manually compressed in `j_round2_appendix.tex`; per-run records remain in Table 14. |
| 9 | `tab:r2-centres` | `analysis/theta_centred/scripts/summarize_theta.py` gives the reviewed `full_grid_errors.csv`; `make_paper_tables.py` generates the paired centre table. | Current generated `round2_centre_table.tex` includes both affine prior centres and nonlinear shifted outputs; included in `j_round2_appendix.tex`. |
| 10 | `tab:r2-relative-shift` | `report_saved_summaries.py` gives `regional_relative_shift.csv` and `round2_relative_shift_table.tex` from the reviewed centre comparison. | Baseline/MLE family ranges manually compressed in `j_round2_appendix.tex`; the extended rows remain in Table 17. |
| 11 | `tab:r2-oracle` | `review_production.py` gives `nearest_sampled_agreement.csv`; `generate_tables.py` produces `round2_oracle_table.tex`. Uses the saved pointwise fields at the selected settings and bulk higher-quantile calibration. | Generated table included in `j_round2_appendix.tex`. These are the sampled-setting oracle records; the refined-root table addresses those settings separately. |
| 12 | `tab:r3-roots` | Original `analysis/round3_local/review_round3.py` gives refined roots and envelopes; the theta pass evaluates their fields, `summarize_theta.py` reviews coverage and `make_paper_tables.py` joins direct values to saved envelopes. | Generated `round3_math_root_table.tex`; direct root coverage alongside the retained mathematical brackets. |
| 13 | `tab:r4-oracles` | `analysis/round4_local/review_round4.py` reads all 17 original pointwise archives and generates trained-centre regional calibration diagnostics. | `generated/round4_oracle_table.tex`; eight Burgers-MLP rows and all-family/full/post-formation ranges. |
| 14 | `tab:r2-evidence` | `generate_tables.py` produces `round2_evidence_table.tex` from the evidence review CSVs and saved spectra. | Table body retained in `oa_empirical.tex`, with shortened caption/labels. |
| 15 | `tab:r2-changed-bounds` | `generate_tables.py` produces `round2_changed_bounds_table.tex` from saved regional energies and best-tested evidence settings. | Table body retained in `oa_empirical.tex`. |
| 16 | `tab:r2-scale-loadings` | `report_saved_summaries.py` gives `changed_scale_loadings.csv` and `round2_scale_loadings_table.tex`, using returned spectral modes, regional energies and evidence settings. | Table body retained in `oa_empirical.tex`. |
| 17 | `tab:oa-relative-shift` | `report_saved_summaries.py` gives `regional_relative_shift.csv` and `round2_relative_shift_table.tex`; includes baseline, MLE and both evidence grids. | Table body retained in `oa_empirical.tex`. |
| 18 | `tab:lean` | The 15 principal declarations in the focused `lean/PaperReliability/` modules, imported by `lean/PaperReliability.lean` and audited by `lean/Audit.lean`. `lean/THEOREM_MAP.md` gives their analytic scope. Build with the pinned Lean/Mathlib configuration. | Manually curated statement-to-declaration map in `oa_lean.tex`; it is not generated by Python. Numerical values, run counts and root computations remain separate empirical outputs. |

## Other evidence used in the prose

- `analysis/round2/round2_pass.py` and `round2_linalg.py` are the shared numerical pass for the admitted 17-run cohort. They save summary JSON, regional CSV, pointwise fields and spectral archives. They import `analysis/n1/n1_sensitivity.py` for the model/residual construction. `cohort.json` fixes checkpoint identity, file hashes and grouping.
- `analysis/round2/review_production.py` and `review_evidence.py` consume those archives. They provide modal curves, centre comparisons, evidence mechanisms/perturbations, post-formation results, and the reviewed numerical checks.
- `analysis/round3_local/review_round3.py` provides the saved-data prior/likelihood decomposition, prediction and coverage bounds for the theta-centred prior, refined-root coverage envelopes, strong-precision approach, oracle-distribution summaries and the 13-run common-threshold subset. It does not retrain or build a new Jacobian.
- `analysis/round2/ensemble_loo.py` supplies the leave-one-member-out ensemble sensitivity. The frozen eight-member ensemble reference is `analysis/round2/reference/ensemble_calibration_burgers_mlp.json`; the original producer is `training/analysis/ensemble_calibration.py`.
- `training/m6_conformal.py` supplies the restored finite-grid split-conformal illustration. Its marginal guarantee and regional coverage are different quantities; this example is not an additional PINN fit.
- `training/analysis/region_resolution.py`, `fullgrid_metric_contract.py`, the original projector/band scripts and `legacy_core/verify_reference.py` support the retained grid, covariance-metric, spectral-projector and reference checks. Their frozen inputs are distinct from the 17 production quartets.
- `analysis/round2/validate_round2.py`, `stress_evidence_toy.py`, `check_linalg.py` and `integration_checks.py` are small numerical validation programs. They do not replace the production archives or certify weak-mode forward accuracy.

## Boundaries of the collection

1. Source collection does not imply exact historical retraining. The recovered release trainer has changes relative to the original executable; its retained `train_map.py`, `common.py`, `provenance.py` and `uq_metrics.py` are in `provenance/historical_release`. The 17 recorded training configurations are preserved in `provenance/training_configs.json`. The frozen-checkpoint path is the authoritative route for reproducing the reported analyses.
2. The plot/table consumers need the recorded data archives. Source scripts alone cannot reconstruct the exact trained checkpoints, raw sigma fields or rejected-run metadata.
3. The saved-data pipeline reproduces numerical records. The paper source snapshot preserves the final manual merges, rounding, captions and layout.
4. A new residual-informed likelihood, matched-effective-precision scope sweep and ensemble bootstrap remain outside the replication target. Theta-centred prediction and nonlinear-shift checks were completed in the later independent study below.
5. The original scope numbers and anisotropic-prior experiment are not retained paper outputs. Older scripts for those experiments are not required to reproduce the current manuscript.

## Theta-centred paper integration - 16 October 2026

The 17-checkpoint study is at `analysis/theta_centred`. Its shared producer,
`scripts/theta_pass.py`, supplies covariance, theta-centred affine prediction,
zero-prior affine prediction, actual nonlinear `u(theta + delta_L)` and direct
trained-root fields. `theta_linalg.py` and `model_residual.py` define the algebra,
model and raw training residuals; `theta_archive.py` saves reusable factors/grid
modes. `select_theta_evidence.py` selects the fixed evidence points first.

| Paper item | Authoritative new source | Generated file / scope |
|---|---|---|
| Paired centre comparison, `tab:r2-centres` | `summarize_theta.py` -> `outputs/review/full_grid_errors.csv`; `make_paper_tables.py` | `outputs/review/paper_tables/round2_centre_table.tex`. Unweighted family means of per-run full-grid RMS errors plus counts worse than the paired trained prediction; baseline and conditional MLE, both affine centres and nonlinear shifted output. |
| Refined-root table, `tab:r3-roots` | Direct trained-root fields -> `summarize_theta.py` -> `outputs/review/root_comparisons.csv`; old `inputs/refined_roots.csv` envelopes; `make_paper_tables.py` | `outputs/review/paper_tables/round3_math_root_table.tex`. Five direct sharp coverage evaluations beside their retained mathematical envelopes. The roots remain specific to the trained centre. |
| Theta affine/nonlinear regional findings and displacement | `outputs/review/paired_centres.csv`, `family_summary.csv`, `displacement_summary.csv`, `full_grid_errors.csv` | Reproduced from all 17 pointwise fields/shifts by `summarize_theta.py`; narrative rounding is retained in manuscript TeX. |
| Theta-centred evidence selections | `select_theta_evidence.py`; saved original/extended spectral grids | `outputs/evidence/theta_evidence_summary.csv` and per-run JSON/NPZ. Boundary status is finite-grid selection, not calibration. |
| Numerical/provenance statement | `review_metadata.py` and `summarize_theta.py` | `outputs/review/numerical_audit.json` and `field_reproduction.json`; source/input hashes, recorded reload checks and independently reproduced metrics/forward samples. |

The new table producer does not change the six existing scientific figure assets.
Exact posterior sampling, new likelihood weights and off-grid evaluation remain
outside this integration. Nonlinear shifted prediction is a forward diagnostic;
it is not the nonlinear posterior mean, and covariance remains linearised at the
original trained checkpoint. The focused Lean library and its assumptions are
unchanged; it does not certify the new empirical run counts or floating-point data.
