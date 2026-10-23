# Fixed formed-Gram-floor evidence stress test

Status: **REVIEW_REQUIRED** under the original numerical thresholds. This is one additional synthetic test; the earlier 16-decade toy and its results are retained.

The prescribed design has N = 96 and P = 160, a maximum eigenvalue of 1.8889569394274544e7, an eigenvalue range of 21.2 decades, and 79/96 modes (82.29%) below epsilon times the maximum eigenvalue. The true prior precision is 17.652641714178376, the true noise variance is 5.664874505422114e-10, and their product is 1e-8. Fixed random orthogonal vectors, seed 20261008, 100 independent prior/noise draws, and quarter-decade search grids are specified in protocol.json. The generating scales are interior to the search grid.

This is a synthetic stress regime motivated by Table 8's formed-Gram solver-returned range and unresolved fraction. Those numbers are not resolved measurements of the true PINN spectrum. The direct-J returned singular values also require their own uncertainty assessment.

| Check | Result |
|---|---:|
| Production and prescribed-spectrum reference choose the same grid pair | 100/100 draws |
| Interior evidence maxima | 100/100 draws |
| Prior or noise boundary maxima | 0/100 draws |
| 5th–95th percentile log10(alpha estimate / true alpha) | [-0.25, 0.25] |
| 5th–95th percentile log10(noise estimate / true noise) | [0, 0] |
| Largest evidence discrepancy at a selected maximum | 9.403925e-8 |
| Largest selected slope discrepancy against finite differences | 1.130502e-8 |
| Largest absolute evidence discrepancy over the whole grid | 0.01774877 |
| Largest scaled evidence discrepancy over the whole grid | 4.549831e-5 |

The entire-grid scaled error exceeds its prespecified 1e-7 tolerance, so the test is not marked PASS. The slope check passes its 1e-7 tolerance, and all selected grid pairs agree. No grid, seed, draw count or tolerance was adjusted after seeing the result.

The saved scaled-error ratio is a legacy numerical diagnostic: dividing by `1+abs(log evidence)` is sensitive to cancellation and an additive constant in log evidence. Its failure alone does not establish an incorrect selector or inadequate statistical model. The absolute discrepancies and selected-pair comparisons are retained alongside it; the declared threshold and status are unchanged.

Inspection of the saved surfaces locates the largest scaled discrepancy in draw 78, at alpha = 0.01765264 and v = 5.664875e-12: the reference log evidence is -1.516986 and the absolute discrepancy is 0.000114519. The largest absolute discrepancy occurs in draw 5, at alpha = 0.5582255 and v = 5.664875e-14. Across cells within one log unit of each production maximum, the largest absolute and scaled discrepancies are 9.403925e-8 and 1.270029e-10. These describe the failure's extent; they do not replace the original threshold or certify PINN stability.

The posterior model is correctly specified in this toy. Agreement with its prescribed-spectrum likelihood tests the pipeline; finite-sample estimates need not equal the generating scales. This one design neither establishes PINN likelihood adequacy nor rules out boundary estimates for other correctly specified designs.

The computation used Python 3.12.14, NumPy 2.5.3 and SciPy 1.17.1. The exact production-core and toy-script SHA256 values are recorded in protocol.json. Draw-level estimates and errors are in draws.csv; all observations, spectra, grids and both evidence surfaces are in toy.npz. No production checkpoint, TeX or cluster job was used.
