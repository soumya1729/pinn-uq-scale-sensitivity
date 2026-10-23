# Saved-field diagnostics for the final revision

This directory contains one local post-processing script. It reads the 17
round-2 production field archives; it does not train networks, construct
Jacobians, factorise matrices, evaluate parameter shifts, or change any
archived result.

Run from the project root, or supply its location:

```text
python analysis/round4_local/review_round4.py --project-root <paper-project-root>
```

Only NumPy and the Python standard library are required. The script uses the
full-grid 70th/95th percentiles of the reference gradient for the regions.
Burgers post-formation summaries retain these masks and restrict to
`t >= 1/pi`; they do not recompute the regional thresholds.

## Protocol

For each checkpoint and subset, the trained-centre setting is the saved gamma
minimising `abs(log(rho))`, with a tie broken at the lowest gamma. All fields
are read from `*_round2_fields.npz`. Errors are `u_pred - u_exact`.

Two reference-informed width multipliers are compared at this same setting:

- Bulk calibration uses the `higher` empirical 0.68 quantile of
  `abs(error)/sqrt(unit_variance)` in the bulk region.
- Sharp calibration uses the same quantile in the sharp region.

Coverage compares the unscaled standardised errors directly with the scalar
quantile (or 1.96 times the quantile), with inclusive inequalities. Thus 68%
coverage in the calibration region is fixed by construction up to empirical
discreteness and ties. Full-grid sharp regions have 4000 points, giving
68.025% coverage here; post-formation sharp regions have 1226 points, giving
68.108%. The informative sharp-calibrated quantities are coverage at 1.96
standard deviations and the standardised-error summaries. These are finite
grid diagnostics, not repeated-draw probability guarantees.

Separately, every distinct saved gamma with `0.9 <= rho <= 1.1` is included
in the sampled-band output, using bulk calibration. This is a complete
inventory of saved samples in that band, not a continuous-neighbourhood
envelope or a new root search.

## Outputs and checks

- `outputs/nearest_saved_oracles.csv`: both calibrations, all three regions,
  17 full-grid runs and 11 Burgers post-formation subsets (168 rows).
- `outputs/sampled_contrast_agreement_band.csv`: bulk-calibrated statistics
  for all 28 qualifying saved gamma/subset pairs (84 regional rows).
- `outputs/summary.json`: family ranges, counts and conventions.
- `outputs/run.txt`: the recorded invocation output.
- `outputs/round4_oracle_table.tex`: complete generated paper table; every value comes from the CSV rows.

The script reproduces all 84 trained-centre nearest-setting regional rows
from `round3_local/outputs/oracle_distributions.csv`, and all 84 sampled-band
regional rows from the round-2 production review. Continuous statistics
agree at relative tolerance 1e-12. A previous multiplied-width evaluation
can differ by one point exactly at the empirical quantile; the check allows
one point for coverage and records the maximum difference (one point).

## Interpretation

At the nearest saved settings, separately calibrating sharp 68% coverage
does not generally calibrate sharp 95% coverage. Full-grid Burgers--MLP
coverage at 1.96 standard deviations is 83.45--95.78%, below 95% in 7/8 runs;
Burgers--Fourier is 82.48--85.08%, below nominal in 3/3. Allen--Cahn--MLP is
92.10--95.90%, below nominal in 1/3; Allen--Cahn--Fourier is 98.33--99.95%,
above nominal in 3/3. Post-formation results are reported separately and
include overcoverage. These mixed outcomes are evidence that matching one
regional quantile is insufficient to establish a Gaussian interval shape;
they are not evidence that no choice of scale can give useful coverage.

Across all saved samples with approximately matched contrast, sharp 68%
coverage can vary widely. In Burgers--MLP seed 7, seven full-grid saved
settings with rho from 0.906 to 1.064 give sharp coverage from 44.725 to
98.600%. This includes both undercoverage and overcoverage. The five
previous refined-root coverage envelopes remain a separate result: the
band inventory neither replaces nor extends those envelopes.

Nothing here evaluates a theta-centred prior away from the baseline, a
residual-informed likelihood, or an affine-centred oracle.
