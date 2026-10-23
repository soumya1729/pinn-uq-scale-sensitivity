# R2-F small numerical validation

Completed locally on 7 October 2026. This is a synthetic implementation check, not an admitted-checkpoint experiment. No TeX, saved PINN results or cluster jobs were changed.

Run `python analysis/round2/validate_round2.py`. The script requires NumPy and SciPy and writes to `analysis/round2/outputs/validation/` by default. The QR/SVD and full-mode evidence formulas match the frozen N1 path; the extended production script still needs its own integration checks before execution.

## Numerical comparisons

A full-row-rank 12-by-20 design with singular values from 3 to 0.3 was checked at effective precisions 0.0001, 1 and 10000. Direct dense Cholesky calculations provide independent posterior mean, covariance and observation-covariance evidence references.

| Check | Largest observed discrepancy |
|---|---:|
| Posterior parameter mean, relative norm | 4.90e-12 |
| Posterior covariance, relative norm | 3.01e-12 |
| Log evidence, absolute | 1.42e-13 |
| Fixed-alpha evidence slope versus central log-noise finite difference | 1.91e-10 |
| Scaled normal-equation residual | 2.65e-15 |
| QR-complement shift identity, absolute norm | 1.94e-15 |
| Matched-gamma mean/shift difference | 0 |
| Matched-gamma covariance scaling, relative norm | 1.72e-16 |

These comparisons passed the declared numerical tolerances. The synthetic matrix-product residual check is not the independent JVP/VJP test planned on the original PINN residual function.

## Coverage and fitted-residual illustrations

The coverage experiment uses 1000 independent prior/noise draws, ten fixed outputs per draw, known alpha=2 and v=0.2, and the exact affine posterior output mean. Standard errors treat each draw as the independent unit; the outputs within a draw are correlated.

| Centre | One-sigma coverage | Expected probability | 95% coverage | Expected probability |
|---|---:|---:|---:|---:|
| Posterior output mean | 0.6870 (SE 0.00544) | 0.68269 | 0.9534 (SE 0.00249) | 0.95000 |
| Same mean displaced by one posterior sigma | 0.4810 (SE 0.00501) | 0.47725 | 0.8301 (SE 0.00383) | 0.82992 |

The displaced-centre case intentionally illustrates centring; it is not a code failure or a prediction of N3 on the PINNs. A separate full-column-rank N=40, P=12 least-squares illustration gives mean fitted-residual estimate 0.140411 (SE 0.001198), versus exact expectation 0.14 and generating variance 0.2. The estimator retains denominator N.

## Prespecified overparameterised evidence toy

One design, N=96 and P=160, has fixed random orthogonal singular vectors and 96 logarithmically spaced singular values from 100 to 0.000001. Independent observations are generated from the true prior/noise scales alpha=2 and v=0.0001. Seed 10072026 and 100 draws are fixed. Alpha and v grids span respectively +/-3 and +/-4 decades around their true scales, with quarter-decade spacing. No grid or draw-count extension was made.

The analytic reference uses the prescribed spectrum and observation rotation. The comparison path obtains its spectrum and rotation from direct-J QR/SVD, includes all modes and uses the same constant-including evidence formula as N1.

- All 100 selections are interior; no alpha or noise boundary was selected.
- All 100 selected grid pairs agree with the analytic reference.
- Largest absolute log-evidence discrepancy at a selected pair: 1.75e-11.
- Across all grid cells, maximum scaled discrepancy `abs(error)/(1+abs(reference))` is 3.36e-10. The maximum absolute discrepancy is 2.94e-5 in badly fitting cells with much larger likelihood magnitudes; it is reported rather than silently discarded.
- Largest selected-setting evidence-slope finite-difference discrepancy: 2.89e-9.

| log10(estimate / generating scale) | 5% | 25% | Median | 75% | 95% |
|---|---:|---:|---:|---:|---:|
| alpha | 0 | 0 | 0 | 0 | 0.25 |
| v | -0.25 | 0 | 0 | 0 | 0.25 |
| gamma=alpha*v | -0.25 | 0 | 0 | 0 | 0.25 |

The quarter-decade grid limits estimate resolution; medians equal to the true grid pair do not establish exact hyperparameter recovery. This one correctly specified weak-spectrum design does not reproduce the PINN boundary pattern. It supports numerical agreement on this design, but does not establish adequacy of the PINN working likelihood or settle evidence stability for its much larger Jacobians. The all-17 archived-spectrum perturbation and slope diagnostics remain planned.

The protocol, every draw's estimates and boundary flags, both evidence surfaces and the measured summary are retained in `weak_spectrum_protocol.json`, `weak_spectrum_draws.csv`, `weak_spectrum_toy.npz` and `validation_summary.json`. The summary includes package versions and the executed script's SHA256.
