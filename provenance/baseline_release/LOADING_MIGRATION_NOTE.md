# Loading-projector migration note

The `*_loading_cdf.json` files are retained as historical diagnostic artifacts, not as the
source of the loading ratios reported in the paper. Their scalar loading ratio used an inverse
lift through the Gram eigendecomposition, omitted Fisher-null directions, and mixed a
region-mean tangent with a per-point uncertainty contrast. That scalar has therefore been
withdrawn.

There was also a small regeneration mismatch in that superseded calculation. The shipped
Burgers--Fourier artifact stores `181.941` whereas the old producer reran at `183.49`; the
Allen--Cahn--Fourier artifact stores `1130.314` whereas the old calculation reran at about
`1131`. These differences are immaterial to the current result but are disclosed because a
reader can encounter them while auditing the historical files. Repeated current runs agree
with one another on the same software stack, but the inverse lift is numerically rank-sensitive
in the unresolved tail and does not recover the archived scalars. Those archived scalars are
therefore not treated as reproducibility targets.

The retained `d_loading_cdf.py` source is frozen byte-for-byte only to reproduce and diagnose
that historical path. Its module docstring and comments describe the withdrawn pre-migration
quantity and theorem numbering; they are audit history, not documentation of the current
paper-facing loading statistic. The current producer is `d_loading_perpoint.py`.

The same byte-for-byte runtime freeze retains one stale sentence in the module header of
`uq_metrics.py`: it abbreviates the acceptance rule as a global relative-L2 gate. The function
body is the authority and is unambiguous: when region labels are supplied it gates on
`bulk_rel_l2` over `smooth | transition`, records `gate_metric = "bulk_rel_l2
(smooth+transition)"`, and stores both the global diagnostic and the literal Boolean decision.
All admitted artifacts used by the paper supplied regions, and the release checks require that
recorded bulk decision. We leave the header untouched only because changing a comment would
break the exact runtime hash of all eleven loading artifacts; it is not the paper's gate
definition, which is Equation (the `eq:gate` label) and the implementation just described.

The current `loading_perpoint/3` artifacts use the null-inclusive stiff complement

`||Pi_sl g||^2 = ||g||^2 - ||V_st^T g||^2`,

after Cholesky--QR orthogonalisation of the inverse-lifted stiff basis. This formulation divides
by no near-zero eigenvalue and includes every Fisher-null direction because `0 <= alpha`. It
gives the canonical region-mean ratios `195.4` and `1132.4` for those two Fourier
configurations, and the theorem-aligned per-point ratios `43.7` and `79.0`. The retained legacy
CDF arrays are used only to cross-check the two region-mean energy legs separately.

The exact loading runtime is frozen by `LOADING_RUNTIME_MANIFEST.json`, checked by
`analysis/check_loading_runtime.py`, and pinned at the package level by
`loading_requirements_lock.txt`. The 28 immutable data inputs needed by the eleven jobs---trained
parameters, stored sigma fields, canonical CDF legs and Fourier sidecars---are frozen separately
by `LOADING_INPUT_MANIFEST.json` and checked by `analysis/check_loading_inputs.py`. The eleven
primary JSON artifacts additionally record the
producer SHA-1, MAP/Gram/sidecar hashes, numerical gates, and schema identifier.

For a new loading rerun, use the scheduling-only `loading_rerun.slurm` companion: its default
array is 0--5, followed only after completion by `--array=6-10`. It validates the array index and
uses the same scientific producer and settings. The archived `a3b.slurm` remains byte-identical
to its runtime manifest, including its historical full-array default; do not submit that default
under the association's six-submission limit. No new loading rerun is required for the already
validated eleven primary records.
