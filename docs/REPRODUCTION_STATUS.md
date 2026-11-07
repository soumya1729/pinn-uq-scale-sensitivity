# Release verification status

The table below records the full saved-data replay checked locally on 23 October
2026, before GitHub publication. The manuscript is one self-contained PDF; these
files are its code/data accompaniment. Later manuscript and documentation checks
are distinguished below from that historical numerical replay.

| Route | Result | Evidence |
|---|---|---|
| Clean Python installation and source-archive extraction | PASS; pinned reviewer requirements installed in a new Python 3.12.14 environment | `envs/reviewer_freeze_windows.txt` |
| Saved-data restoration | PASS; all 331 destination files match their hashes | `data/manifest.json`, `verification/reproduction_report.json` |
| One reviewer command | PASS; all 14 tasks, 298,238 numeric CSV comparisons within stated tolerances | `verification/reproduction_report.json`, `verification/replay_logs/` |
| Frozen-checkpoint feature/readout fits | PASS; all 17 checkpoints, 204 rows; prediction reconstruction and recorded rank cutoffs checked | `verification/replay_logs/n5.log` and N5 task in the report |
| Registered headline claims | PASS; all 16 quantities, including fresh held-out fits | `verification/historical_20261023/headline_results.json` |
| Current centre/root tables | PASS; generated displayed numbers agree with Tables 9 and 12 | Tables task in the report |
| Figure replay | PASS; all six current figure producers; Figure 1's 2000 numeric CSV fields match exactly | `verification/figure_replay.json` |
| Manuscript source completeness | PASS; extracted source archive compiles to 35 pages | `paper/` |
| Source-clean Lean proof rebuild | PASS; seven modules, entry library and 15-theorem axiom audit; warnings treated as errors | `lean/BUILD_REPORT.md`, `lean/verification/build-full.log` |
| Lean dependency identity | PASS; all nine pinned commits and tracked trees checked | `lean/verification/dependency-tree-audit.json` |

The full Lean recompile used the tested sequential direct-compiler helper with
an existing matching dependency cache. Complete Lake attempts were stopped for
resource/performance reasons; fresh dependency downloading was not retested.

On 7 November the current manuscript compiled to 35 pages, with all tables,
figures and numbered equations referenced and no unresolved references or
overfull boxes. The selected numerical-claim report records 16/16 saved-summary
checks (`verification/headline_results.json`). Its source bindings were checked
against the current prose without rerunning those numerical aggregations.
The source/page indexes and Excel accompaniment were refreshed. This update did
not repeat the full saved-field replay or the Lean build.

The current numerical ledger records 1944 scientific occurrences: 14 occurrence
spans link to executed registered checks and 1930 map to their producers. It is
an index, not verification of every numerical occurrence. Its statuses distinguish
fresh field/fit calculations from metadata and scalar-summary checks; see
`numerical_ledger/summary.json` and `docs/NUMERICAL_LEDGER.md`.

The authoritative reproduction starting point is the frozen checkpoints and
saved arrays. Exact historical retraining is not claimed because no training-time
source commit was recorded. The checks above do not repeat the original
residual-Jacobian pass, independent PDE reference solve, all historical loading
calculations, or every nonlinear forward evaluation on the full grid. Those
boundaries are documented rather than counted as completed work.
