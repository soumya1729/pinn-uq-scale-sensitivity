# Interpretation checks from saved results

These checks read the existing prediction fields and round-two CSVs. No production pass, fitting or resampling was rerun. Original ensemble/oracle CSVs are unchanged.

## Ensemble member influence

Omitting seed 5 changes rho from 1.1593 to 1.3510 (a shift of 0.1917), and sharp 68% coverage from 0.8765 to 0.4935. The sign of the contrast deficit survives every omission, but its magnitude and coverage are sensitive to membership.

| Seed | Correlation with other seven's mean error | Uncentred cosine | Opposite signs | Error RMS |
|---:|---:|---:|---:|---:|
| 0 | -0.3010 | -0.3061 | 0.4188 | 0.01069 |
| 1 | 0.9846 | 0.9873 | 0.1963 | 0.20021 |
| 2 | 0.9791 | 0.9828 | 0.1010 | 0.28315 |
| 3 | 0.9888 | 0.9902 | 0.1278 | 0.20584 |
| 4 | 0.1363 | 0.1129 | 0.4515 | 0.00362 |
| 5 | -0.7517 | -0.7959 | 0.8468 | 0.07978 |
| 6 | 0.8569 | 0.8754 | 0.3852 | 0.02040 |
| 7 | 0.8883 | 0.9063 | 0.1770 | 0.10642 |

Seed 5's errors counteract the other members' mean error: correlation -0.7517, with opposite signs at 84.67% of the sharp grid points. Its inclusion reduces sharp error RMS from 0.11442 to 0.09238 and increases spread RMS from 0.10879 to 0.11863.

Seed 5 is not uniquely negatively aligned: seed 0 also has negative correlation, although its error magnitude is much smaller. These observations concern this fixed ensemble and mask; they do not establish that a member with opposite errors is generally necessary for calibration, or that errors have opposite signs everywhere.

## Oracle interpretation

At the sampled settings closest to rho=1, five runs are within 0.05 of one. All five have a higher average sharp-region Z and lower sharp 68% coverage after bulk calibration. This illustrates that agreement of regional RMS contrasts does not match the distribution of pointwise error-to-width ratios. A higher mean is not a statement about every point or a unique causal explanation.

| Seed | rho | Bulk mean Z | Sharp mean Z | Sharp cov68 | Near agreement |
|---:|---:|---:|---:|---:|---|
| 0 | 0.9544 | 0.8282 | 1.4033 | 0.3427 | yes |
| 1 | 1.8363 | 0.8334 | 0.4924 | 0.8605 | no |
| 2 | 2.5874 | 0.9474 | 3.2697 | 0.4467 | no |
| 3 | 1.9924 | 0.8015 | 0.6740 | 0.7250 | no |
| 4 | 0.9549 | 0.8401 | 1.2175 | 0.4370 | yes |
| 5 | 0.9733 | 0.9840 | 1.9763 | 0.3593 | yes |
| 6 | 1.0323 | 0.8556 | 1.0912 | 0.5165 | yes |
| 7 | 0.9642 | 0.9091 | 1.1044 | 0.6518 | yes |

All values are extracted from existing outputs; sampled settings are not exact roots, and the analyses retain the trained centre.
