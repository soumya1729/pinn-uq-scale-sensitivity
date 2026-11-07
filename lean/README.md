# PaperReliability

PaperReliability formalises the finite-model analytic arguments in *Scale Sensitivity and Regional Calibration of Full-Network Gauss–Newton Uncertainty in Physics-Informed Neural Networks*. The development covers spectral covariance representation, two-subspace contrast bounds and sharpness witnesses, predictive-centre bounds, variance and empirical coverage envelopes, and the strong- and weak-precision limits.

The default Lake target imports all seven mathematical modules. There are 15 principal theorems; [THEOREM_MAP.md](THEOREM_MAP.md) records their paper anchors and exact scope. Routine matrix identities, inequalities and order-statistic steps are private dependencies.

## Build

Lean 4.30.0 and Mathlib commit `c5ea00351c28e24afc9f0f84379aa41082b1188f` are pinned by `lean-toolchain` and `lake-manifest.json`. The manifest also pins transitive dependencies.

From this directory, with Lean/Elan installed:

```bash
lake exe cache get
lake build
lake env lean Audit.lean
```

The first command obtains dependency build artifacts. The second compiles the full `PaperReliability` library. The audit prints the axioms of all 15 principal results and the signed-error convention of the centre theorems. Standard logical axioms (`propext`, `Classical.choice`, `Quot.sound`) are permitted; `sorryAx` and project-defined axioms are not.

[BUILD_REPORT.md](BUILD_REPORT.md) records the validation performed on this source package. The full command and version log is [verification/build-full.log](verification/build-full.log). A fresh dependency download is a separate reproducibility step and was not part of the cached build check.

An existing matching Mathlib cache can also be used with `check_cached.sh`. Set `LEAN_CACHE_PROJECT` to a Lean 4.30.0 project containing the pinned dependencies and `LEAN_COMPILER` to its compiler. The helper recompiles all seven modules and the aggregate library sequentially with warnings treated as errors, then runs the 15-theorem axiom audit; it does not edit the dependency project. The fresh 23 October 2026 source rebuild used this route. For a fresh installation, first run `lake exe cache get`, then `LEAN_CACHE_PROJECT="$PWD" LEAN_COMPILER=lean bash check_cached.sh` from this directory. The dependency-download step was not repeated in the pinned-cache audit.

## Model and coordinates

For finitely many modes,

`modalVariance spectrum energy gamma = sum_i energy_i / (spectrum_i + gamma)`.

In the Gaussian model, `gamma = alpha * v` and `sigma^2 = v * modalVariance`. Physical applications have nonnegative eigenvalues and energies and positive precision. Nonzero pointwise tangent energy supplies positive variance. The ratio of regional mean variances is squared RMS uncertainty contrast; its square root is RMS contrast.

| Lean quantity | Paper interpretation |
| --- | --- |
| `spectrum`, `energy` | Eigenvalues `lambda_i` and squared tangent coordinates, or their regional averages. |
| `CovarianceRepresentation.Q` | The row-oriented parameter eigenbasis. If the paper writes `F = V diag(lambda) V^T`, this `Q` is `V^T`. |
| `PredictiveCentres.singular` | Singular values `s_i`, with `lambda_i = s_i^2`; additional exact kernel coordinates have `s_i = 0`. |
| `PredictiveCentres.residual` | Projected residual coordinates. In `J^T = Q_QR R`, `R = U S V_SVD^T`, these are `V_SVD^T r` on the residual modes. `V_SVD` is a residual-space matrix, distinct from the paper's parameter eigenbasis `V`. |
| `PredictiveCentres.tangent`, `trained` | Tangent and trained-parameter coordinates in the parameter basis: `U^T Q_QR^T g` and `U^T Q_QR^T theta`, with the orthogonal complement included. |
| `precision`; the coarse bounds' parameter `alpha` | Effective precision `gamma = alpha * v`. The coarse parameter equals prior precision only when `v = 1`. |
| `PredictiveCentres.noise` | Residual noise variance `v`, retained separately from effective precision. |

Signed error is prediction minus reference, so affine recentering adds `predictiveShift`. The exact-kernel attribution includes every zero singular mode, not a numerical rank truncation. A QR complement is the whole kernel only when the residual Jacobian has full row rank. Tiny nonzero singular values are not treated as exact zeros.

## Verification scope

All statements concern exact real finite models and affine prediction. The matrix representation is conditional on the supplied orthogonal spectral basis and positive shifted eigenvalues. It does not construct an eigendecomposition or certify an eigensolver.

Theorem 8's fixed-cutoff pointwise and regional inequalities are formalised. The subsequent supremum/infimum-over-cutoffs consequence remains manuscript reasoning. Proposition 9 is supported by bounded-spectrum finite witnesses with prescribed nonnegative energies and an explicit multiplicative approach factor; the exported statements bound eigenvalues by `Lambda` rather than asserting that `Lambda` is attained. Sharpness does not concern optimisation over a fixed trained network's spectrum and tangents.

The centre theorems derive the pointwise likelihood-update bound, the minimum of the covariance-metric and modal Minkowski regional bounds, and empirical coverage brackets. They do not export the subsequent theta-centred gap-ratio inference or certify the numerical run counts.

The variance and coverage envelopes hold at every precision in a supplied bracket. For the paper's trained-centre root analysis, errors stay fixed across precision. Root existence, uniqueness, numerical bisection and membership certification are not proved. Coverage includes the actual finite bulk order statistic and sharp accepted-point counts, including ties; it is not a distributional or independence guarantee.

The strong limit requires positive bulk total energy. The weak limit requires positive bulk whole-kernel energy. The admissible counterexample has equal endpoint RMS limits but an intermediate squared contrast of `10304/503`; neither monotonicity nor bounding the curve by its endpoints follows in general.

The first-order expansion (4), secondary mean-width and coherence identities, Remarks 5–6, numerical Jacobian/factorisation outputs, floating-point rounding guarantees, empirical 16/17 and 14/14 counts, nonlinear shifted-network predictions, admission decisions and fitted last-layer coefficients are outside this verification scope.

## Higher-quantile convention

`higherQuantile` takes a zero-based integer rank `k`, with `k < n`. For the paper's finite float64 arrays and NumPy's `method="higher"`, compute that rank using the same evaluated floating-point expression:

```python
k = int(np.ceil((n - 1) * np.float64(p)))
```

For bulk calibration, `p = 0.68`. This is the ceiling of the **evaluated float64 product**, not a symbolic exact-real replacement for that product. Lean takes the resulting integer as an input; the floating-point rank calculation itself is not verified. Closed coverage inequalities preserve ties, which can put achieved bulk coverage above the nominal level.
