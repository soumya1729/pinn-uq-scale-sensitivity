# Reviewer guide

This is a guide to the **single, self-contained paper** in `paper/full_paper.tex`. The folders preserve the separate calculations that produced its results; they are not additional papers or required reading in sequence.

## Start with one command

From the repository root, install the review requirements and restore the accompanying saved-data release:

```bash
python -m pip install -r envs/reviewer_requirements.txt
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip
python reproduce.py --mode saved --out reproduced
```

Choose a new output directory. The command recomputes the principal regional, modal, centre, root, oracle and ensemble summaries from the frozen arrays, compares regenerated CSVs with the recorded results, and writes fresh tables, logs and a numerical occurrence index. It launches **no PINN training, residual Jacobian, covariance factorisation or cluster job**. The readout fits use a small feature-matrix SVD. The data archive restores 331 expected files from 172 distinct objects; the large retained factor archive is not needed.

Read `reproduced/REPORT.md` first. `reproduced/reproduction_report.json` gives each task's verification level, compared fields, tolerances and failures. Missing or mismatched data produces a nonzero exit status. The smaller `--mode records` route checks included summaries and arithmetic; it does not recompute coverage from pointwise fields.

## Follow the paper

Paths are relative to this repository. `docs/PAPER_OUTPUT_MAP.md` provides the more detailed producer chain for every displayed output.

| Paper part | Where to look | Relevant outputs and scope |
|---|---|---|
| Section 2: setting, regions and admission | `training/`, `provenance/training_configs.json`, `training/analysis/solve_gate_table.py` | Table 1 reaggregates metadata for all 55 attempted configurations. Recorded admission metadata is not a new solution/reference computation. |
| Section 3.1: scale, regional coverage and predictive centres | `analysis/round2/`, `analysis/theta_centred/`, `analysis/round3_local/`, `analysis/round4_local/` | Tables 2â€“3, Figures 1â€“2, all-17 scale/centre comparisons and the oracle diagnostics. Fresh replay folders are `round2/review`, `theta`, `round3`, `round4` and `reporting_closure`. |
| Section 3.2: feature expressivity | `analysis/n5/`, `training/d4_manifold.py` | Table 4. Saved mode rebuilds the features from all 17 checkpoints, checks their trained predictions, and repeats the same-grid and held-out readout fits. |
| Section 3.3: modal geometry and spectral bounds | `training/d_loading_perpoint.py`, `training/analysis/loading_table.py`, `analysis/recompute_saved_summaries.py`, `lean/` | Table 5 and the analytic results. Default replay recomputes bound arithmetic from saved loading summaries, and the later modal diagnostics from saved spectra; it does not repeat the historical loading/Jacobian calculation. |
| Appendix A: experimental/reporting detail | `training/`, `legacy_core/`, `provenance/` | Recorded configurations, controls and source history. Independent PDE references and new training are optional deeper routes, not part of saved-data replay. |
| Appendix B: spectral proofs | `paper/j_appendix.tex`, `paper/trim_aux_proofs.tex`, `lean/THEOREM_MAP.md` | Proofs and their assumptions; numerical arrays and run counts are outside Lean's scope. |
| Appendix C: scale, centre and oracle detail | `analysis/n1/`, `analysis/round2/`, `analysis/round3_local/`, `analysis/round4_local/`, `analysis/theta_centred/` | Tables 6â€“13 and Figure 3. `n1` is the original eight-seed Burgersâ€“MLP study; `round2` and `theta_centred` extend the relevant analyses to all 17 admitted checkpoints. |
| Appendix D: extended per-run results | `analysis/round2/outputs/production_review/`, `analysis/round2/outputs/reporting_closure/` | Tables 14â€“17 and Figures 4â€“6. Replay regenerates their underlying review/loading/shift records. Final table layouts are condensed in TeX; Table 16 places the existing weak-loading ranges side by side and prints the shared stiff ranges once. |
| Appendix E: numerical checks | `analysis/round2/validate_round2.py`, `check_linalg.py`, `integration_checks.py`, `stress_evidence_toy.py`; `analysis/theta_centred/` | Recorded validation and production checks. Default replay does not rerun every toy or replace the saved numerical factorisation with an independent solver. |
| Appendix F: formal development | `lean/` | Table 18 and the 15 principal declarations. Build separately as below. |

### Tables and figures at a glance

| Displayed output | Main numerical producer |
|---|---|
| Table 1 | `training/analysis/solve_gate_table.py` |
| Tables 2, 10, 16, 17 | `analysis/round2/report_saved_summaries.py`; field/spectral records from `review_production.py` |
| Table 3 | `training/analysis/temporal_per_seed.py` |
| Table 4 | `analysis/n5/n5_heldout.py`, `summarise_n5.py`; original same-grid producer `training/d4_manifold.py` |
| Table 5 | `training/analysis/loading_table.py`, `loading_table_tex.py`; `analysis/recompute_saved_summaries.py` |
| Table 6 | `analysis/n1/n1_sensitivity.py`, `summarize_n1.py` |
| Tables 7â€“8, 11, 14â€“15 | `analysis/round2/review_production.py`, `review_evidence.py`, `generate_tables.py` |
| Tables 9, 12 | `analysis/theta_centred/scripts/summarize_theta.py`, `make_paper_tables.py`; Table 12 also uses `analysis/round3_local/review_round3.py` |
| Table 13 | `analysis/round4_local/review_round4.py` |
| Table 18 | `lean/PaperReliability.lean`, `lean/Audit.lean`; curated map rather than a Python-generated numerical table |
| Figure 1 | `tools/export_burgers_slice.py`, `analysis/round2/plot_shortened_slice.py` |
| Figures 2â€“3 | `analysis/round2/plot_shortened_records.py` |
| Figure 4 | `analysis/round2/plot_shortened_coverage.py` |
| Figures 5â€“6 | `analysis/round2/plot_merged_family_curves.py` |

The current paired-centre and direct-root tables are regenerated in **`reproduced/tables/theta_current/`**. The `tables/round2/` directory also contains an older zero-mean-only centre formatter; that file is not current Table 9. Six released figure PDFs are in `paper/figures/`; their numerical inputs are reviewed by the replay, while replotting is a separate presentation step documented in `docs/RUN_ORDER.md`.

## Locate a particular number

Open `numerical_ledger/Numerical_Provenance.xlsx`, or its CSV equivalents, and filter by section, PDF page, table/figure, or printed token. The ledger links repeated occurrences to source lines, inputs and producers. The 29 October inventory contains 1934 numerical occurrences; **11 occurrences are joined to executed registered checks**, and the rest are provenance mappings, not individual scalar verifications. Counts change if the manuscript changes.

For actual executed results use `reproduced/headline_claims/headline_claims.json` and `reproduced/reproduction_report.json`. The headline checker names the specific aggregated quantities and visibly labels any frozen-summary fallback. Neither a successful replay nor a ledger mapping means that every numerical occurrence has been independently recomputed. See `docs/NUMERICAL_LEDGER.md` for extraction boundaries.

## Optional deeper checks

Saved mode already repeats the held-out last-layer fits. To run just that step from the restored frozen checkpoints, without training or a Jacobian, choose a separate output path:

```bash
python analysis/n5/n5_heldout.py --out regenerated_n5/n5_all17.csv
```

The N5 calculation uses the paper's fixed split and rank-cutoff defaults; it is not new PINN training. The complete saved-mode run automatically supplies its fresh `n5/n5_all17.csv` to the headline checker.

To build the analytic development separately, use Bash/WSL with Lean/Elan installed:

```bash
cd lean
lake exe cache get
LEAN_CACHE_PROJECT="$PWD" LEAN_COMPILER=lean bash check_cached.sh
```

The pinned Lean/Mathlib configuration defines this environment. The tested sequential helper rebuilds all project sources and runs `Audit.lean`; `lean/BUILD_REPORT.md` records the dependency-cache reuse and the ordinary Lake alternative. Table 18 and `lean/THEOREM_MAP.md` state the exact verified assumptions. Empirical root brackets, floating-point QR/SVD results, saved arrays and run counts are not formal proofs.

For deliberately expensive checkpoint-to-covariance recomputation or new training, use `docs/RUN_ORDER.md` and `docs/ENVIRONMENTS.md`. These require the appropriate JAX environment and resources. Historical Slurm files need adaptation to the reviewer's cluster. The recovered trainer and recorded settings are included, but no training-time source commit was recorded, so exact historical retraining is not claimed. The authoritative reproduction route starts from the frozen checkpoints; see `docs/PUBLIC_PROVENANCE.md`.
