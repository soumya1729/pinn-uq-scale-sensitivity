# Downloaded metadata and provenance review

**PASS: all 17 checkpoint quartets and 17 retained-archive manifests agree.**

- 143 frozen input copies match their original manifest hashes; per-run input signatures also match local evidence/root inputs.
- Four production script hashes match all 17 summaries and all four executed script snapshots. JAX/NumPy/SciPy versions are {'jax': '0.10.1', 'numpy': '2.4.6', 'scipy': '1.17.1'}.
- All JSON metadata are finite; 95 independent normal-equation checks and all recorded reconstruction/evidence/reload checks meet the stated thresholds.
- All four Slurm tasks completed with exit 0:0. Eighteen successful checkpoint invocations comprise 17 computed checkpoints plus the reused seed-2 pilot. No failed or REVIEW_REQUIRED log entries were found.
- Recorded retained data: 275,936,352,768 bytes (275.936 decimal GB). Every downloaded manifest hash matches its summary, including the successful on-cluster reload check.

| Recorded diagnostic | Maximum | Threshold |
|---|---:|---:|
| baseline_max_sigma_relative_error | 5.41e-11 | 0.0001 |
| gradient_norm_relative_defect | 8.794e-14 | 1e-07 |
| affine_jvp_scaled_defect | 1.304e-16 | 1e-07 |
| normal_equation_backward_error | 9.117e-17 | 1e-07 |
| evidence_rebuild_regret | 0 | 0.001 |
| old_variance_relative_defect | 1.28e-07 | 0.0001 |
| old_zero_shift_scaled_defect | 5.721e-12 | 0.0001 |
| archive_reload_defect | 1.102e-14 | 1e-10 |

The large factors and grid arrays remain on cluster storage. This local audit verifies their manifest provenance, not their current physical contents or checksums. Their recorded cluster reload checks validate layout/reconstruction consistency; they are not weak-mode forward-accuracy certificates. Scientific CSV/field comparisons are reviewed separately.

Reproduce this review with `.venv/Scripts/python.exe scripts/review_metadata.py` from this study directory. No production files are modified.
