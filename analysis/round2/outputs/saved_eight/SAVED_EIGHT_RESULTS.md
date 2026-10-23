# Saved-output round-two analyses

R2-A/B/C preliminary results on the eight completed Burgers–MLP N1 runs. All-17 sensitivity and affine-mean comparisons remain pending.

The source CSV metrics reproduce from the archived fields. Trained predictions and original reference-defined masks are unchanged. Post-formation means t >= 1/pi with the same masks intersected with this time subset.

The sampled oracle multiplies each saved unit-noise width field by the bulk 68th percentile of |e|/sigma (NumPy higher quantile). It uses the reference solution, calibrates bulk 68% coverage descriptively and does not select a practical likelihood. Width scaling leaves regional contrast and rho unchanged. 95% uses 1.96 standard deviations.

The table uses the sampled precision closest to rho=1 in log distance; it is not an exact crossing or optimum over untested settings.

| Seed | Closest sampled rho | Oracle sharp 68% | Oracle sharp 95% | Oracle bulk 95% |
|---:|---:|---:|---:|---:|
| 0 | 0.954 | 0.343 | 0.788 | 0.945 |
| 1 | 1.836 | 0.861 | 0.974 | 0.907 |
| 2 | 2.587 | 0.447 | 0.639 | 0.863 |
| 3 | 1.992 | 0.725 | 0.933 | 0.959 |
| 4 | 0.955 | 0.437 | 0.864 | 0.927 |
| 5 | 0.973 | 0.359 | 0.692 | 0.869 |
| 6 | 1.032 | 0.516 | 0.844 | 0.918 |
| 7 | 0.964 | 0.652 | 0.837 | 0.883 |

All sampled settings, width factors, regional counts, coverage and post-formation contrasts are retained in the accompanying CSVs. No assertion that endpoint limits bound the curves or that sampled oracle failures apply to every scale is made.

The contrast/gap figure separates covariance geometry from the trained error target. Connecting lines guide the eye across uneven sampled spacing; the figure does not identify the asymptotic limits.
