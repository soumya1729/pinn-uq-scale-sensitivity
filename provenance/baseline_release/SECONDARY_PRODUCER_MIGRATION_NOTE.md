# Secondary-producer migration note

This note separates paper-facing evidence from historical or optional diagnostics whose producer
contracts were repaired during the final metric audit.

## Reduced-model NUTS (`step3b_small_nuts.py`)

The pre-`step3b/2` artifacts averaged within-chain predictive-width ratios rather than pooling
draws before computing predictive variance. They also reused a chain's random key for its initial
perturbation, adaptation and retained sampling. Those artifacts and the former “within 0.5%”
statement are withdrawn. The current producer uses independent keys, pools draws and stamps
`schema_version = step3b/2`; no current paper claim depends on a new run.

## Inline covariance in `train_map.py`

The optional last-layer covariance formerly assembled inside the training producer did not have a
valid common metric/block interpretation across all training arms and PDEs. It was already excluded
from paper-facing covariance evidence. It is now disabled fail-closed; training, solve acceptance,
the trained-parameter artifact and deterministic fields remain available. Dedicated post-hoc
producers own every covariance result.

## Split conformal (`m6_conformal.py`)

The retained record used Python's non-standard `NaN` token for an unavailable MCE field. The
migration changes only those three unavailable values to JSON `null`, adds deterministic split
metadata and records the original file hash. No interval, coverage, error or ratio value changes.
Future output is written atomically with `allow_nan=False` under schema `split_conformal/2`.

## Full-network wrapper and shared historical core

`m4_full_laplace.py` now refuses to run unless `RAW_METRIC=1` and accurately states that its own
summary carries the secondary mean-width contrast; the primary RMS statistic comes from the
independent full-grid field producer. The hash-frozen historical `laplace_core.py` contains a
non-negative clamp after its Woodbury subtraction. It is not silently trusted by the primary
pipeline: every retained primary raw field is finite/positive and bound by the 17-field metric
contract, and the eleven current-J loading runs independently gate the negative complement and
reproduce stored sigma. Changing the frozen core would invalidate those bindings, so the historical
implementation is disclosed rather than rewritten.

## Inference-time dropout

`m8_mc_dropout.py` applies masks only at evaluation to a network trained without dropout. The paper
and output metadata now call it an inference-time perturbation probe, not a dropout-trained Bayesian
posterior.

## Legacy Git fields

Historical artifacts with `git.commit = null` and `git.dirty = false` were produced by a helper that
mistook an unavailable Git command/repository for a clean tree. That pair means **Git state unknown**,
not clean. Scientific binding is supplied by the explicit source/input/MAP hashes. New artifacts
record `git.available`, use `dirty = null` when status is unavailable and retain the error text.
