# Review of the all-17 theta-centred study

Reviewed 16 October 2026. All 17 expected checkpoints completed and were
downloaded. No manuscript was edited and no additional cluster pass was run.

## Numerical and provenance review

- All 68 routine output files and 17 retained-archive manifests are present.
- All 143 frozen input copies, per-run input signatures, production source
  signatures and executed script snapshots match. All jobs exited `0:0`.
- The 95 independent normal-equation checks have maximum normwise backward error
  9.12e-17. Baseline sigma agreement is at most 5.41e-11 relative error.
- Archive manifests match the completion-summary hashes and record successful
  reload checks; the largest recorded reload defect is 1.10e-14. The manifests
  describe 275.936 decimal GB retained on the cluster. The large arrays were not
  downloaded, so their physical contents were not freshly rehashed locally.
- `scripts/summarize_theta.py` independently reproduces the metrics in all 2,256
  CSV rows from the saved fields, using relative tolerance 1e-10 and absolute
  tolerance 1e-12. A separate NumPy forward implementation reproduces nine sampled
  nonlinear predictions at every saved gamma in every checkpoint; maximum
  absolute discrepancy is 4.72e-12.

These are consistency checks. They do not certify unresolved singular directions,
nonlinear posterior validity, or calibration.

## What the experiment changes

The prior centred on the trained parameters removes the zero-prior shrinkage
term. At fixed alpha and v it leaves the covariance unchanged. Its centre is
conditional on the fitted checkpoint, rather than independently specified.

### Baseline: the prior-centre objection is now answered directly

Both the theta-centred affine prediction and the actual network evaluated at
theta + delta_L retain rho > 1 in all 17 runs. The minimum values are 1.58175
and 1.57989 respectively. This resolves the earlier baseline bound's one
inconclusive run, Burgers-MLP seed 4; it was unresolved by the bound, not an
observed counterexample.

The theta-centred affine full-grid RMS error is 0.9587--1.0028 times the trained
error (median 0.9982). One-sigma coverage remains high: at least 0.8055 in the
sharp region and 0.9997 in the bulk. Parameter shifts are only 8.73e-6--1.54e-4
of the parameter norm. The nonlinear-minus-affine remainder is small here; the
largest regional RMS remainder/RMS sigma is below 6.4e-4.

The baseline deficit therefore does not depend on choosing the zero-mean prior
centre in this cohort. This statement is specific to the baseline scales and
specified raw homoscedastic likelihood.

### Conditional noise MLE: affine prediction often improves, calibration does not

The noise estimate stays fixed at mean(r(theta)^2) in this comparison; it is not
re-estimated after shifting the centre.

| Full-grid result across 17 runs | Trained centre | Theta-centred affine | Shifted nonlinear network |
| --- | --- | --- | --- |
| rho range | 0.0616--7.6427 | 2.1838--10.0109 | 0.00467--0.94159 |
| Runs with rho > 1 | 12 | 17 | 0 |
| Bulk one-sigma coverage range | 0.1976--0.8429 | 0.8539--1.0000 | 0.00054--0.2093 |
| Sharp one-sigma coverage range | 0.0160--0.6243 | 0.2018--1.0000 | 0.0000--0.4963 |

Affine full-grid RMS error improves in 15/17 runs, with median error ratio 0.7953
relative to the trained predictor. It would be incorrect to claim that changing
the centre cannot improve prediction. However, bulk one-sigma coverage exceeds
nominal in all 17, whereas sharp one-sigma coverage is below nominal in 12/17.
The sign of a regional RMS gap is not a coverage test.

### Nonlinear check: a small parameter displacement is insufficient

At the MLE setting, the actual nonlinear network's bulk RMS error is worse than
the trained predictor's in all 17 runs; full-grid RMS error is worse in 11/17.
Both regions under-cover at one and 1.96 sigma in all 17 when evaluated against
the frozen covariance. Its rho < 1 is driven by deterioration in the bulk and
must not be presented as repaired regional allocation.

Although ||delta_L||/||theta|| is only 0.00782--0.03867, bulk RMS
nonlinear-minus-affine remainder/RMS sigma ranges from 2.495 to 386.16. The bulk
remainder itself ranges from 0.01081 to 0.29451, so this is not solely an artefact
of division by the trained predictor's small error. The affine improvement is a
finding about the frozen affine model; it does not reliably transfer to the
nonlinear network at the displaced parameters.

This is a deterministic linearisation diagnostic. The experiment does not
compute a nonlinear posterior predictive mean, posterior samples, or a covariance
recomputed at the shifted parameters.

### Theta-centred evidence: no general calibration remedy

The extended finite grid selects the lower noise boundary in 14 runs, the upper
alpha boundary in two Allen-Cahn Fourier runs, and an interior point in the
remaining Allen-Cahn Fourier run. All 34 original/extended selections reproduce
from the new factorisation, with zero evidence-grid regret.

With each selected setting's own theta-centred affine mean, sharp 95% coverage
is below nominal in all 17 (range 0.00275--0.90175). The numerical behaviour is
family-dependent: the Allen-Cahn MLP shifted nonlinear predictions deteriorate
severely, while the two upper-alpha-boundary Allen-Cahn Fourier selections barely
move the centre but still produce poorly calibrated widths. Do not combine these
mechanisms into one universal assertion. These are best-tested settings on the
declared grids, conditional on the returned float64 factors.

### Exact trained-centre roots confirm the existing coverage envelopes

Each root below belongs to the trained centre on the full grid. Widths are then
calibrated to 68% bulk empirical coverage using the declared higher quantile.

| Burgers-MLP seed | Sharp one-sigma coverage | Sharp 1.96-sigma coverage |
| --- | --- | --- |
| 0 | 0.32750 | 0.78275 |
| 4 | 0.42700 | 0.85200 |
| 5 | 0.30325 | 0.67725 |
| 6 | 0.52325 | 0.84700 |
| 7 | 0.55725 | 0.82350 |

The directly evaluated rho values differ from one by less than 8e-14, and all
coverage values lie inside the existing saved numerical envelopes. The result
strengthens an existing conclusion: matching regional RMS contrast does not
ensure pointwise coverage. These gammas are not roots for the other centres.

## Recommended bounded paper update

The results warrant replacing the baseline-only centre comparison and its
unresolved limitations, rather than adding a new broad study:

1. Replace "16/17, one unresolved" with directly measured "17/17" for the
   theta-centred baseline comparison, retaining the mathematical bounds in their
   existing role.
2. Add a compact fixed-scale centre comparison at baseline/MLE. Acknowledge the
   affine accuracy improvement alongside its remaining coverage mismatch.
3. Include the nonlinear remainder as an empirical limitation of affine
   recentering at the MLE, explicitly keeping covariance at the original theta.
4. Put actual root coverage alongside the existing five-root envelope table.
5. Update the no-longer-untested centre/evidence/root limitations. Keep the
   homoscedastic likelihood and selected-cohort scope.

No new figures, 17-row table, cluster experiment or further sensitivity study is
required to report these results. A subsequent source edit must measure the
compiled page count; 35 pages cannot be promised before compiling.

The story is more complete: the baseline mismatch survives the natural prior
centre, affine accuracy and calibration remain different questions, and the
affine centre improvement at weaker precision need not survive nonlinear
evaluation. This does not establish failure of every Gaussian or residual-noise
model.

## Reproduction files

- `scripts/review_metadata.py`: downloaded metadata/provenance audit.
- `scripts/summarize_theta.py`: field/table reproduction, sampled NumPy network
  checks and paired/family summaries.
- `outputs/review/numerical_audit.json` and `field_reproduction.json`: checks.
- `paired_centres.csv`, `family_summary.csv`, `full_grid_errors.csv`,
  `displacement_summary.csv`, `root_comparisons.csv`: complete numerical summaries.
- `INDEPENDENT_SCIENCE_REVIEW.md` and `PAPER_INTEGRATION_NOTES.md`: independent
  science review and existing-paper source anchors.
