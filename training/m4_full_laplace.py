"""Full-network post-hoc linearised Gauss--Newton Gaussian via Woodbury.

The historical ``m4``/``Laplace`` names are retained for artifact compatibility. The centre is
the trained parameter, not a point verified to maximise the later Gaussian-prior posterior.
This producer's summary record stores the secondary mean-width contrast (4.07 on the raw-metric
Burgers-MLP seed-2 configuration) and pointwise fields. The paper's primary RMS contrast 6.47 is
derived from the independent full-grid field producer ``step_fullgrid_ratio.py``; it is not an
output promised by this wrapper.
NB the old docstring quoted ~97.3x -- that was the WHITENED metric (the whitened block of
{TAG}_fullgrid_ratio.json still reads 101.34 for this config). Run with RAW_METRIC=1."""
# Docstring reference values corrected 2026-08-04: they had been carried over from the
# whitened era and no longer matched what a re-run produces, so anyone verifying against
# them would have concluded their own run was broken. See REPLICATION.md sec 0.

import os
if os.environ.get("RAW_METRIC") != "1":
    raise SystemExit("m4_full_laplace.py is paper-facing only in the raw residual metric; set RAW_METRIC=1")

import laplace_core as L
if __name__ == "__main__":
    L.run_and_dump("full", "Full Laplace", "full_laplace")
