# Reviewer-command execution evidence

23 October 2026. The public source ZIP was extracted into a new directory and its
331 required data destinations restored from the versioned data asset. A new
Python 3.12.14 virtual environment installed `envs/reviewer_requirements.txt` from
the package index; no author environment or `PYTHONPATH` was inherited.

- Command: `python reproduce.py --mode saved --out <fresh directory>`.
- Result: **PASS, all 14 tasks**, including **16/16 registered headline quantities**.
- Data integrity: 331 files, 2,377,096,330 bytes; all manifest SHA-256 values match.
- Compared **298,238 numeric CSV fields across 21,605 rows** with frozen records.
  All pass relative tolerance `2e-9` and absolute tolerance `2e-12`. The largest
  absolute difference is `3.248453140258789e-6`; this is within the scale-dependent
  tolerance, not a claim of bit-identical floating-point arithmetic.
- N5 rebuilt all 129 features and predictions and repeated the same-grid/random/
  time-block readout fits for all 17 checkpoints: 204 rows and 3,876 numeric fields.
  The headline checker uses these fresh fits in saved mode.
- Reporting summaries additionally regenerate regional displacements, changed-scale
  loadings and baseline ranges from fresh review records and saved spectra.
- Current Table 9's 63 and Table 12's 52 displayed numerical tokens match their
  freshly generated tables.
- Sum of task times: **283.7 seconds**, with four BLAS threads on this machine.
  This is an observed run, not a runtime guarantee for other hardware.
- Installed versions: NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2, pypdf 6.10.0,
  openpyxl 3.1.5. The complete tested Windows dependency freeze is in `envs/`.

Full evidence is in `verification/reproduction_report.json`,
`verification/historical_20261023/headline_results.json` and `verification/replay_logs/`.
The final task builds the occurrence inventory. Its PASS means that the inventory
was produced; the 1983 provenance-only occurrences are not individual scalar checks.

All six current figure producers also ran successfully in a fresh destination.
Figure 1's 400-point CSV was rebuilt from saved prediction/raw-covariance fields;
all 2000 numeric fields match exactly. See `verification/figure_replay.json`.
The manuscript compiled from the extracted source archive to 35 pages, with no
unresolved references in the final compilation log.

The current 29 October manuscript is 35 pages. Its selected-claim bindings were
refreshed and all 16 quantities checked against saved summaries in
`verification/headline_results.json`; the source/page ledger was regenerated.
These current checks are separate from the historical full replay above, whose
original timing, comparison counts and verification levels are preserved.

These are frozen-checkpoint and saved-field reproduction checks. They do not
repeat PINN training, the residual-Jacobian/covariance pass, the independent PDE
reference solve, all historical loading computations, or every shifted nonlinear
forward evaluation on all 80,000 grid points. Historical loading/control scalars
and recorded solver diagnostics retain their stated provenance level. Lean is
built separately; see `lean/BUILD_REPORT.md`.
