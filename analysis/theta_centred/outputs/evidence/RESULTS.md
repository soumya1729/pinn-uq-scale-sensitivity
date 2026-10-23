# J2: checkpoint-centred affine-evidence selections

Status: completed locally for all 17 frozen checkpoints. No new Jacobian factorisation, cluster job or manuscript change was needed.

## Model and fixed protocol

For the prior `vartheta ~ N(theta, I/alpha)`, the frozen affine residual observation is `-r(theta)`. Its saved singular-basis coordinates are `-residual_modes`, and the marginal covariance eigenvalues are `v + s_i^2/alpha`. The script includes every residual mode and the Gaussian normalising constant. It uses no reference solution.

The original and extended grids are exactly the two archived grids. There was no outcome-dependent grid extension. Both are searched for each checkpoint. These are best tested points of a checkpoint-conditional affine working model, based on the returned float64 spectrum; they are not continuous optima, numerical rank certificates or calibration guarantees.

## Extended-grid selections

| Checkpoint | alpha/alpha0 | v/v_MLE | gamma = alpha v | Boundary |
|---|---:|---:|---:|---|
| burgers_mlp_seed0 | 3.16228 | 1e-08 | 6.63522e-12 | noise lower |
| burgers_mlp_seed1 | 10 | 1e-08 | 2.83005e-11 | noise lower |
| burgers_mlp_seed2 | 3.16228 | 1e-08 | 6.96207e-12 | noise lower |
| burgers_mlp_seed3 | 5.62341 | 1e-08 | 7.87201e-12 | noise lower |
| burgers_mlp_seed4 | 5.62341 | 1e-08 | 8.81043e-12 | noise lower |
| burgers_mlp_seed5 | 10 | 1e-08 | 1.89228e-11 | noise lower |
| burgers_mlp_seed6 | 17.7828 | 1e-08 | 2.21019e-11 | noise lower |
| burgers_mlp_seed7 | 5.62341 | 1e-08 | 1.03447e-11 | noise lower |
| burgers_fourier_seed0 | 177.828 | 1e-08 | 1.19295e-09 | noise lower |
| burgers_fourier_seed2 | 177.828 | 1e-08 | 1.51533e-09 | noise lower |
| burgers_fourier_seed6 | 177.828 | 1e-08 | 1.64126e-09 | noise lower |
| allen_cahn_mlp_seed0 | 0.000562341 | 1e-08 | 4.88065e-15 | noise lower |
| allen_cahn_mlp_seed1 | 1.77828e-05 | 1e-08 | 7.07039e-17 | noise lower |
| allen_cahn_mlp_seed3 | 3.16228e-05 | 1e-08 | 1.69353e-16 | noise lower |
| allen_cahn_fourier_seed0 | 1e+07 | 1 | 1428.16 | alpha upper |
| allen_cahn_fourier_seed1 | 31.6228 | 0.0177828 | 3.11441e-05 | interior |
| allen_cahn_fourier_seed3 | 1e+07 | 1 | 1532.43 | alpha upper |

Fourteen of seventeen extended-grid selections remain on the lower noise boundary: all eleven Burgers checkpoints and all three Allen–Cahn MLP checkpoints. At those settings the log-evidence derivative with respect to log(v) is negative, ranging from -2369.64 to -380.96 in the saved spectrum. The fixed extension therefore does not settle a finite best noise scale for these runs.

Allen–Cahn Fourier seeds 0 and 3 select the upper alpha boundary and the conditional residual-noise MLE. Seed 1 has an interior best grid point. In the original grid Burgers Fourier seed 2 instead selects the upper alpha boundary at the conditional noise MLE; the four-decade noise extension switches its best grid point to the lower noise boundary. This is a grid dependence to retain in any interpretation.

Each extended-grid within-one-log-unit set contains one grid cell. This reports the tested grid and is not a continuous confidence region. The selected alphas and effective precisions can differ substantially from the zero-centred selections. Covariance and prediction/coverage comparisons are pending the next stage.

## Reproducibility checks

- Modal log evidence matched direct dense covariance calculations for both prior centres, including P > N and N > P cases; maximum absolute discrepancy: 4.68958e-12.
- Recomputed original zero-centred surfaces and extended surfaces matched the archives; all 34 selected index pairs were reproduced. Maximum absolute surface discrepancy: 3.63798e-12.
- Saved residual-mode energy reproduced the conditional noise MLE for all 17 checkpoints; maximum relative discrepancy: 2.77556e-15.

## Files and next step

- `select_theta_evidence.py`: standalone NumPy calculation and dense self-test.
- Per-checkpoint JSON: both theta-centred selections, zero-centred reproductions, grid boundaries, within-one-unit ranges, local evidence derivatives and regional mean covariance.
- Per-checkpoint NPZ: the exact grids and both theta-centred evidence surfaces.
- `theta_evidence_summary.csv` / `.json`: all 17 checkpoint records.

Next: compute pointwise theta-centred affine predictions and the nonlinear network at the shifted parameters, first for the prespecified pilot and then for all 17 checkpoints. The archives do not retain the required factors or pointwise modal tangent coefficients. These evidence results alone do not determine accuracy or interval coverage.
