# Theta-centred script inventory

All paths below are under `analysis/theta_centred/scripts/`. These 13 files are copied unchanged from the completed isolated study.

| Script | Purpose |
|---|---|
| `check_inputs.py` | Verify 143 frozen input files against the fixed study manifest (standard library). |
| `check_models.py` | Validate checkpoint model/prediction/loss reconstruction on the selected family pilots. |
| `make_paper_tables.py` | Generate paired full-grid centre and direct-root/envelope paper tables from reviewed CSVs. |
| `model_residual.py` | Frozen model, raw training residual definitions and chunked Jacobian construction. |
| `prepare_inputs.py` | Historical input collation with explicit original-project paths; not required after restoring the frozen deposit. |
| `review_metadata.py` | Audit output quartets, source/input hashes, finite diagnostics, retained manifests and downloaded execution records. |
| `select_theta_evidence.py` | Select theta-centred affine evidence settings from saved spectra on the fixed grids; includes a dense algebra check. |
| `summarize_theta.py` | Recompute CSV metrics from saved fields and check sampled nonlinear predictions with an independent NumPy forward model. |
| `theta_archive.py` | Write/reload float64 native QR/SVD factors, signed grid modes, complement fields, interpretation manifest and checksums. |
| `theta_linalg.py` | Direct-J QR/SVD, posterior shift coordinates, orthogonal transforms, positive variance sums and affine evidence. |
| `theta_pass.py` | Shared checkpoint production pass for theta/zero/trained centres, nonlinear shifted predictions and trained-root covariance fields. |
| `validate_archive.py` | Small full/rank-deficient archive round trips, dense references and corruption checks; no production pass. |
| `validate_theta.py` | Small dense and nonlinear algebra/AD checks for the theta-centred pipeline; no production pass. |

Production sources are the four modules `theta_pass.py`, `theta_linalg.py`, `model_residual.py` and `theta_archive.py`. Their hashes match all 17 output signatures and executed snapshots. The evidence selector runs first; the shared pass follows; the two reviews and table producer operate on saved outputs. Runtime-specific versions and execution sources are preserved in `cluster_records`.
