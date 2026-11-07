# One command for reviewers

Run commands from the repository root. No GPU, PINN training job, residual-Jacobian construction or covariance factorisation is launched by this entry point. The N5 readout refits use small feature-matrix SVDs.

```bash
python -m pip install -r envs/reviewer_requirements.txt
python reproduce.py --mode records --out reproduced_records
```

This inexpensive mode checks the 34 all-17 production CSV/JSON record sets, reaggregates the 55 attempted-run admission metadata, and recomputes the legacy scalar-bound arithmetic. It does **not** independently recompute coverage from pointwise fields. A successful records run is not full verification of every paper number.

After downloading and restoring the versioned data archive using `tools/restore_data.py`, run:

```bash
python reproduce.py --mode saved --out reproduced_saved
```

Choose an output directory that does not already exist. The optional `--root` argument selects a different repository checkout. All regenerated outputs go into that fresh directory; frozen reference records are retained.

## What saved mode recomputes

| Task | Inputs and producer | What is checked |
|---|---|---|
| Records | Included production JSON/CSV | Checkpoint identity, coverage bounds and the target/contrast/gap identity; saved-summary level only |
| Admission | `training/results/**/_map.json`; `training/analysis/solve_gate_table.py` | All 55 attempted identities, recorded admission decisions and family extrema/medians; matches the frozen aggregate |
| Scalar bounds | `analysis/input` and `analysis/recompute_saved_summaries.py` | Decimal/rational arithmetic, regional enclosures and outward display rounding; the loading summaries themselves remain frozen inputs |
| Data integrity | `DATA_MANIFEST.json` | Size and SHA-256 of all 331 restored destinations |
| Round-two metrics |17 pointwise/spectral quartets; `analysis/round2/review_production.py` | Regional errors, variances, contrasts, gaps, coverages, descriptive oracle widths, post-formation subsets, field/modal consistency, evidence-slope reconstruction |
| Modal/root/bounds metrics | Same arrays; `analysis/round3_local/review_round3.py` | Refined trained-centre roots and coverage envelopes, prediction bounds, parameter decomposition, strong-precision approach and common13 summaries |
| Local oracle diagnostics | Same pointwise fields; `analysis/round4_local/review_round4.py` | Sharp/bulk quantiles, signed/absolute standardised errors, local calibration and saved contrast-agreement band |
| Theta-centred study |17 theta field/shift archives and checkpoints; `analysis/theta_centred/scripts/summarize_theta.py` | All stored centre metrics, direct root coverages and their envelopes; independent NumPy nonlinear forward cross-check at nine points per gamma |
| Ensemble | Eight member prediction fields; `analysis/round2/ensemble_loo.py` | Mean, population spread and all eight leave-one-out subsets |
| Reporting summaries | Fresh round-two review records, saved spectra and `analysis/round2/report_saved_summaries.py` | Relative displacements, changed-scale loadings and nonlinear baseline ranges; the separate linear-control extract is frozen metadata |
| Feature refits | All 17 frozen checkpoints; `analysis/n5/n5_heldout.py` | Rebuilt 129 features and trained predictions, repeated regional same-grid/random/time-block fits and both recorded rank cutoffs; no PINN training |
| Headline claims | `verification/headline_claims.json`; `tools/check_headline_claims.py` | Selected registered numerical quantities, compared with current paper text, using fresh records when available and visibly labelled frozen summaries otherwise |
| Numerical tables | The freshly recomputed review outputs | Table records, including the completed theta-centred comparison and direct-root table; final manuscript layout may use manual merges/captions |
| Occurrence inventory | Active manuscript source, supplied page indexes and the headline-check report | Repeated source occurrences and producer mappings; only registered checks receive a checked status |

Each scientific task compares its regenerated CSV values with the frozen reference CSVs. Producer assertions use their original tolerances; wrapper CSV comparison uses relative tolerance 2e-9 and absolute tolerance 2e-12. The machine-readable result records compared rows/fields and maximum absolute differences. A failure or missing data returns a nonzero exit status. Missing work is never silently counted as passed.

## Outputs

- `reproduction_report.json`: task status, verification level, output comparisons, timings and scope.
- `REPORT.md`: readable task overview.
- One log and fresh result directory per task.
- Regenerated CSV/JSON/TeX and the existing consumer plots.
- `headline_claims/headline_claims.json` and `.csv`: individually registered headline quantities and their actual input/verification levels.
- Fresh generated centre/root tables compared numerically with the current manuscript source when it is present.

Refer to the numerical occurrence ledger for each individual manuscript value and its verification level. `PASS` here means the listed frozen-data calculations agree with the saved reference records. It does not prove PDE-reference accuracy, model calibration, the adequacy of the working likelihood, exact historical training provenance, or the correctness of unexecuted stages.

## Deliberately separate

The default command does not build Lean, rerun original training or covariance factorisations, reconstruct all historical loading spectra, rerun every synthetic control, or reevaluate all shifted nonlinear predictions independently at all 80,000 points. It does repeat the N5 last-layer fits from frozen checkpoints. The deliberately separate producers remain available under their documented paths; saved-array diagnostics should not be confused with new PINN training.

Historical round-three consumer notes predate the completed theta-centred study. Their old statements that direct theta/root computations were unperformed describe that historical stage; the theta task and current manuscript contain the completed computations.
