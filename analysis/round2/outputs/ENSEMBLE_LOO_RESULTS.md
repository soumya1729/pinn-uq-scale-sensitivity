# R2-D: ensemble member sensitivity

The original eight admitted Burgers-MLP members are used. Each omission recomputes the mean prediction, its error against the saved reference, and population spread with divisor K (ddof=0). The original reference-defined bulk and sharp masks remain fixed.

The complete ensemble reproduces the saved baseline artifact to numerical precision.

| Omitted seed | K | R_sigma RMS | R_tar | rho | Sharp cov68 | Sharp cov95 |
|---|---:|---:|---:|---:|---:|---:|
| none | 8 | 136.493 | 158.231 | 1.1593 | 0.8765 | 0.9932 |
| 0 | 7 | 136.634 | 157.797 | 1.1549 | 0.8438 | 0.9890 |
| 1 | 7 | 133.391 | 147.836 | 1.1083 | 0.8885 | 0.9720 |
| 2 | 7 | 177.141 | 191.719 | 1.0823 | 0.9075 | 0.9885 |
| 3 | 7 | 130.106 | 144.707 | 1.1122 | 0.8980 | 0.9990 |
| 4 | 7 | 135.775 | 159.699 | 1.1762 | 0.8325 | 0.9935 |
| 5 | 7 | 124.212 | 167.809 | 1.3510 | 0.4935 | 0.9862 |
| 6 | 7 | 137.295 | 156.430 | 1.1394 | 0.8502 | 0.9995 |
| 7 | 7 | 138.147 | 145.751 | 1.0550 | 0.8940 | 0.9895 |

Every omission retains rho > 1, with a range of 1.055 to 1.351. The modest regional RMS contrast gap is therefore retained by all eight subsets.

Sharp-region 68% coverage is member-sensitive: it ranges from 0.4935 to 0.9075; omitting seed 5 produces the minimum. Sharp-region 95% coverage ranges from 0.9720 to 0.9995. Bulk coverage exceeds nominal at both levels for every omission. Regional contrast proximity does not imply calibration.

Leave-one-out ranges describe sensitivity to these eight members; they are not statistical confidence intervals. Changes cannot be assigned to covariance alone because every omission also changes the predictor and its error target.
