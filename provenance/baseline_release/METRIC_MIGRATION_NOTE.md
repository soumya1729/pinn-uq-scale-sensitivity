# Residual-metric audit and migration

This note separates scientific artifacts from historical diagnostics. A filename containing
`raw` is not treated as a metric declaration; current producers stamp the metric and current
checkers verify it.

## Audit result

Every Python producer named by the release manifest was inspected. The source-level scan finds
exactly ten metric-sensitive files, all explicitly classified below:

1. `d1_geom_hessian.py` did not record the residual metric used for its historical six condition
   numbers; the current source defaulted to whitening, but the old artifact alone cannot prove
   which path produced it. Those schema-less values are withdrawn. A separate
   six-task repair computes the raw weighted-residual last-layer Gauss--Newton/Fisher condition
   from singular-value endpoints and writes schema `last_layer_ggn_spectrum/2` under
   `results/_metric_repair/`.
2. `m5_hmc_lastlayer.py` constructs an adaptive block-whitened NUTS target inline. Its archived
   Gaussian comparator is independently verified raw. The historical comparison is therefore
   cross-metric and non-diagnostic: neither its similar regional ratio nor its absolute-width
   discrepancy is used as evidence. The artifact is preserved, not rewritten.
3. `m7_bll_conformal.py` inherited the default whitened last-layer Gaussian. The paper-facing
   canonical result is regenerated raw as schema `bll_conformal/2`. The identity that a global
   conformal multiplier preserves the regional width ratio is asserted on the exact test indices.
4. `step_fullgrid_ratio.py` honored `RAW_METRIC` but did not stamp its outputs, and normal mode
   could label two raw fields as `whitened` and `raw`. Future outputs carry schemas and metrics;
   invalid mode combinations fail. The seventeen admitted frozen full-network raw fields are not
   recomputed: all reproduce their raw full-Laplace and explicit dual-metric records, differ from
   the whitened leg, and are bound with their exact reference grids and regions by SHA-256 in
   `fullgrid_sigma_metric_contract/3`; eleven also
   have schema-v3 loading records whose current raw Woodbury calculation reproduces the stored
   field to numerical precision. The three reduced-scope fields on each of eight Burgers-MLP
   seeds are regenerated separately with schema, raw metric, scope, parameter count, producer
   hash and MAP hash. No current reader infers a metric from a filename.
5. `train_map.py` deliberately permits raw, adaptive or frozen training metrics. Its historical
   inline last-layer Gaussian is disabled because its metric and boundary treatment were not
   a valid paper covariance route. The solve gate and trained-parameter output remain available;
   dedicated post-hoc producers own covariance. The `C5-OK` explanation records that distinction.
6. `d22_aniso_prior.py` is regenerated as schema `aniso_prior/5`. It fails closed unless
   `RAW_METRIC=1`, binds its complete admitted configuration, MAP, frozen raw full-grid field,
   reference-error field, full-Laplace record, and current producer/helper/common/Laplace sources, and independently rebuilds
   the isotropic field. Its sampled smooth-region claims also pass a fixed split-half stability
   gate; the sample is never labelled as an all-grid result. The original reference-error/
   mean-width tau selection is unchanged. Paired sampled RMS and matched error gaps use the
   2000 sharp/4000 bulk evaluation subset. Error-RMS sampling instability is a required disclosed
   diagnostic, not a numerical failure; uncertainty second moments have separate reconstruction
   and sensitivity checks. After the first-wave approximation failures, the covariance keeps
   every Gram mode in its isotropic baseline and updates only selected resolved blocks. The
   targeting cutoff is not a covariance cutoff. Float64 coordinates and separate current-J
   covariance checks cover isotropic, extreme, reciprocal and fitted precisions.
7. `d_loading_perpoint.py` is the raw-only schema-v3 null-inclusive loading producer. It has an
   internal raw guard and stamp, and all eleven outputs pass the independent stored-field,
   Woodbury, projector and regional-sandwich checks.
8. `m4_full_laplace.py` routes through `laplace_core.run_and_dump`, which stamps the metric
   actually selected. Current launchers are separately required to set `RAW_METRIC=1`, so a
   manual non-raw run is labelled rather than silently mistaken for raw evidence.
9. `d_loading_cdf.py` is different: it remains byte-for-byte frozen because
   `LOADING_RUNTIME_MANIFEST.json` records the exact runtime used in the schema-v3 loading audit.
   Its resolved inverse-lift scalar is withdrawn, and the exact-hash source exception exists
   only to reproduce the disclosed mismatch; it is not accepted as a current metric-stamped
   producer. Historical artifacts remain identified by their original hashes and are not
   silently migrated.
10. `d_band_projector_check.py` is the raw-only current-J validator for the theorem-band
    endpoints. It rebuilds each canonical full Jacobian, forms an orthogonal stiff projector,
    validates the primary and exact optimizing thresholds (including the all-sloppy endpoint),
    and binds every output to its MAP, legacy CDF, theorem-band summary, sidecar where applicable,
    complete configuration, and current numerical sources.

## Withdrawn inverse-lift reproducibility discrepancy

The shipped historical `loading_cdf.json` records a Burgers--Fourier mean-tangent loading ratio
of `181.941` (printed as 182), whereas rerunning that legacy inverse-lift calculation on the
frozen MAP gives `183.49`.  Allen--Cahn--Fourier similarly changes from the printed 1130 to about
1131.  These values used the numerically fragile `c^2/lambda` inverse lift and omitted the
Fisher-null complement.  They are retained only so the old bundle remains auditable.  The paper
withdraws them: its current loading table uses the null-inclusive stiff-complement projector,
whose canonical legs reproduce the frozen inputs, and no current claim relies on either legacy
endpoint.

## Enforcement

`analysis/lint_metric_producers.py` scans every Python file named by the manifest. A producer
that constructs or consumes a post-hoc covariance must use the centrally stamped
`run_and_dump`, or carry both an explicit metric stamp and either a fail-closed raw guard or a
non-empty `C5-OK` explanation. The sole exception is the exact SHA-256 of the frozen historical
CDF producer above, jointly bound by the loading-runtime manifest and this disclosure; changing
one byte removes the exception. Corruption self-tests remove each ordinary defence separately
and require the lint to fail.

The raw repair never overwrites canonical history. `analysis/metric_repair.py` accepts exactly six
schema-v2 condition records plus one schema-v2 conformal record, verifies input and code hashes,
and only then creates the JSON/TeX evidence consumed by the manuscript.
