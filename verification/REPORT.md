# Reproduction run

Mode: **saved**. Result: **PASS**.

| Task | Status | Verification level |
|---|---|---|
| records | PASS | CHECKED_SAVED_SUMMARY |
| admission | PASS | RECOMPUTED_METADATA |
| scalar_bounds | PASS | CHECKED_SAVED_SUMMARY |
| data_integrity | PASS | CHECKED_FILE_HASHES |
| round2 | PASS | RECOMPUTED_FIELDS_AND_MODAL |
| round3 | PASS | RECOMPUTED_FIELDS_AND_MODAL |
| round4 | PASS | RECOMPUTED_FIELDS |
| theta | PASS | RECOMPUTED_FIELDS_AND_SAMPLED_FORWARD |
| ensemble | PASS | RECOMPUTED_FIELDS |
| reporting_closure | PASS | RECOMPUTED_FIELDS_AND_MODAL_WITH_FROZEN_CONTROL |
| n5 | PASS | RECOMPUTED_CHECKPOINT_FEATURE_FITS |
| tables | PASS | GENERATED_FROM_RECOMPUTED_RESULTS |
| headline_claims | PASS | PER_CLAIM_LEVELS_REPORTED |
| ledger_inventory | PASS | PROVENANCE_INVENTORY_ONLY |

Frozen-data consistency/recomputation at the levels shown per task. Not every manuscript occurrence is independently recalculated; consult the numerical ledger for occurrence coverage.

No frozen input/reference result was deliberately overwritten.

## Outside this command

- new training
- new Jacobian or factorisation
- cluster jobs
- Lean compilation
- full-grid independent nonlinear reevaluation
- independent PDE reference solve
- full historical loading reconstruction
- all toy reruns
