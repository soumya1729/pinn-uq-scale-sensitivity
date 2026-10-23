# N5 results — 5 October 2026

All 17 admitted frozen checkpoints were evaluated. Each region uses its original 129 last-layer features, including bias. Random halves use seed 0; alternating blocks contain 10 time slices. Both fitting directions are scored and their residual energies pooled. Primary relative SVD cutoff: 1e-10; comparison: 1e-12.

**The blocked test adds useful information.** Sharp-region held-out residuals are small in most runs, but larger than same-grid residuals. The result supports conditional held-out performance of last-layer fits. Performance of separately fitted full-network coefficients was not tested.

## All-run blocked results

Entries are residual-energy ratios, not percentages. Ratios above one are retained.

| Region | Minimum | Median | Maximum | Runs above one |
|---|---:|---:|---:|---:|
| bulk | 0.00807972 | 0.267566 | 0.899485 | 0/17 |
| transition | 0.0454837 | 6.46329 | 7869.89 | 10/17 |
| sharp | 0.000202421 | 0.00723827 | 0.142039 | 0/17 |

The largest sharp-region blocked residual ratio is 0.142039 on burgers_mlp_seed4, corresponding to 85.80% error-energy reduction. Its original sharp-region error RMS is 0.00361747 and held-out residual RMS is 0.00136335. This is the smallest sharp-region error among the eight Burgers–MLP runs. The small denominator is context, not grounds to dismiss the relative deterioration. 16/17 sharp-region ratios are below 0.04. The existing 98.8% same-grid statement must not be transferred to these held-out fits.

Because the trained prediction lies in the last-layer span and the fitting matrices have full column rank, error fitting and direct reference fitting yield the same held-out squared residuals in exact arithmetic. A pooled ratio below one therefore means that regional readouts fitted to reference values predict held-out blocks with lower mean-square error than the trained readout. This holds in the sharp region in all 17 runs. The refits use reference values and separate coefficients per region and fold; they are not a global correction based on PDE residuals alone.

Some transition-region refits have very large blocked residuals and exceed the trained readout's error in 10/17 runs. Small same-grid fractions therefore do not ensure accurate prediction across held-out time blocks. A smaller trained-error denominator provides context but does not itself explain or predict this deterioration. These results do not identify numerical solver failure or its cause.

## Canonical sharp-region rows

The same-grid column here uses the new fixed cutoff. Original Table 7 numbers are preserved in the manuscript.

| Run | Same grid | Random held out | 10-slice blocks held out |
|---|---:|---:|---:|
| burgers_mlp_seed2 | 0.000121389 | 0.000146322 | 0.000455468 |
| burgers_fourier_seed2 | 0.00149205 | 0.00162671 | 0.00669698 |
| allen_cahn_mlp_seed0 | 0.000867304 | 0.00139162 | 0.0324061 |
| allen_cahn_fourier_seed0 | 0.000743881 | 0.000841216 | 0.00987456 |

## Rank and conditioning

Training ranks range from 129 to 129. Untruncated training-matrix condition numbers range from 501.21 to 3.85146e+07. The maximum held-out ratio difference between cutoffs is 0. Both cutoffs retain all 129 directions, so identical results establish insensitivity over these two cutoffs only, not over more aggressive truncation or ridge regularisation.

## Minimal manuscript use

Table 7 includes two sharp-region held-out columns. Section 3.2 states the split, all-seed worst case and absolute RMS context, and transition-region failures. The contribution and limitations distinguish the same-grid 98.8% statement from held-out last-layer performance. Ranks, the two-cutoff comparison and per-fold results remain in the companion CSV. Last-layer functions remain contained in the full tangent function space; this does not establish the performance of a separately fitted full-network estimator.

N5 has been run and integrated into the paper. N1–N4 remain deferred. Original same-grid values and all other numerical findings are preserved.
