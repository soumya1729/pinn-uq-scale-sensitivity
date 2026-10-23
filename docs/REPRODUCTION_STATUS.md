# Release verification status

Checked locally on 23 October 2026, before GitHub publication. The manuscript
remains one self-contained 35-page PDF; these files are its code/data accompaniment.

| Route | Result | Evidence |
|---|---|---|
| Clean Python installation and source-archive extraction | PASS; pinned reviewer requirements installed in a new Python 3.12.14 environment | `envs/reviewer_freeze_windows.txt` |
| Saved-data restoration | PASS; all 331 destination files match their hashes | `data/manifest.json`, `verification/reproduction_report.json` |
| One reviewer command | PASS; all 14 tasks, 298,238 numeric CSV comparisons within stated tolerances | `verification/reproduction_report.json`, `verification/replay_logs/` |
| Frozen-checkpoint feature/readout fits | PASS; all 17 checkpoints, 204 rows; prediction reconstruction and recorded rank cutoffs checked | `verification/replay_logs/n5.log` and N5 task in the report |
| Registered headline claims | PASS; all 16 quantities, including fresh held-out fits | `verification/headline_results.json` |
| Current centre/root tables | PASS; generated displayed numbers agree with Tables 9 and 12 | Tables task in the report |
| Figure replay | PASS; all six current figure producers; Figure 1's 2000 numeric CSV fields match exactly | `verification/figure_replay.json` |
| Manuscript source completeness | PASS; extracted source archive compiles to 35 pages | `paper/` |
| Source-clean Lean proof rebuild | PASS; seven modules, entry library and 15-theorem axiom audit; warnings treated as errors | `lean/BUILD_REPORT.md`, `lean/verification/build-full.log` |
| Lean dependency identity | PASS; all nine pinned commits and tracked trees checked | `lean/verification/dependency-tree-audit.json` |

The full Lean recompile used the tested sequential direct-compiler helper with
an existing matching dependency cache. Complete Lake attempts were stopped for
resource/performance reasons; fresh dependency downloading was not retested.

The numerical ledger is a transparent index, not an all-numeral theorem: 1996
scientific occurrences are recorded, with 13 occurrence spans linked to executed
registered checks and the remainder mapped to their producers. It distinguishes
fresh field/fit calculations from metadata and scalar-summary checks.

The authoritative reproduction starting point is the frozen checkpoints and
saved arrays. Exact historical retraining is not claimed because no training-time
source commit was recorded. The checks above do not repeat the original
residual-Jacobian pass, independent PDE reference solve, all historical loading
calculations, or every nonlinear forward evaluation on the full grid. Those
boundaries are documented rather than counted as completed work.
