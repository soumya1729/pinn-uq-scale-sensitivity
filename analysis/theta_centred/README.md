# Theta-centred predictive comparison

This directory contains the completed 17-checkpoint comparison used by the paper:
trained predictions, zero-mean-prior affine means, theta-centred affine means and
nonlinear predictions evaluated at the theta-centred parameter update.

For a prior centred at the trained checkpoint,

```text
delta_L = -(J^T J + gamma I)^-1 J^T r
Sigma   = v (J^T J + gamma I)^-1
gamma   = alpha v
```

Changing the prior centre at fixed alpha and v changes the affine predictive mean,
not its covariance. The nonlinear shifted network is a forward diagnostic, not
the nonlinear posterior mean; covariance stays linearised at the original
checkpoint.

## Reproduce the saved results

Start with the repository's top-level README and `reproduce.py`. The portable data
asset supplies the frozen inputs, production fields and parameter shifts. The
complete data manifest is `../../data/manifest.json`; restore with the repository
utility rather than historical machine paths.

The underlying review and table producers are:

```bash
python analysis/theta_centred/scripts/review_metadata.py
python analysis/theta_centred/scripts/summarize_theta.py
python analysis/theta_centred/scripts/make_paper_tables.py --out reproduced/theta_tables
```

Run these from the repository root. The top-level wrapper uses fresh destinations
and compares them with the frozen records. Running the first two producers
directly overwrites small records in `outputs/review`; they do not build a new
Jacobian or retrain a model.

## Source and data locations

| Location | Contents |
|---|---|
| `manifest.json` | Fixed cohort, configurations, selected settings and execution identity |
| `inputs/` | Independent checkpoint, raw baseline and preceding spectral/field inputs |
| `scripts/select_theta_evidence.py` | Affine evidence selection from saved spectra |
| `scripts/theta_pass.py` | Expensive checkpoint-based covariance and centre production |
| `scripts/theta_linalg.py`, `scripts/model_residual.py` | Direct-J linear algebra and raw model/residual definitions |
| `scripts/theta_archive.py` | Factor/grid archive layout and validation |
| `scripts/review_metadata.py`, `scripts/summarize_theta.py` | Saved-output integrity and numerical summaries |
| `scripts/make_paper_tables.py` | Centre and trained-root table producer |
| `outputs/production/` | One JSON/CSV/fields/shifts quartet for each checkpoint |
| `outputs/evidence/`, `outputs/review/` | Compact frozen selection, coverage, error and displacement summaries |
| `cluster_records/` | Executed sources, accounting and retained-archive manifests |
| `verification/` | Small implementation/consistency validation records |

Settings include the baseline, conditional residual-noise MLE, original/extended
evidence selections for both prior centres, and the five trained-centre contrast
roots. Root comparisons use the stated reference-informed regional calibration;
post-formation rows at those precisions are not independently selected
post-formation roots.

## Expensive route and retained factors

Rebuilding production from checkpoints is optional and requires JAX plus suitable
resources. It factors the residual Jacobian once per checkpoint and evaluates all
selected settings together. `CLUSTER_COMMANDS.md` and `cluster/` retain the original
site-specific execution route; adapt paths, account, partition and runtime before
using them elsewhere. They are not required for saved-array replay.

The 17 native float64 factor/grid archives total 275,936,352,768 bytes and are
separate from the routine saved-data asset. Their recorded layout, software
versions, checksums and reload results are under
`cluster_records/retained_manifests/`. These manifests do not claim a fresh local
read of the absent large arrays. New off-grid evaluation would require new
gradients.

The recorded numerical core is JAX 0.10.1, NumPy 2.4.6 and SciPy 1.17.1. Ordinary
saved-field summary/table replay uses NumPy; JAX is needed for production and
AD-based implementation validation, not for reading the completed comparisons.
