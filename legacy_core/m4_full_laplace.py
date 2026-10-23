"""m4 -- Full-network Laplace (P=196,009) via Woodbury. THE CRUX on the Fourier MAP.
Expect (RAW metric, burgers_mlp_seed2): sigma-ratio 4.07, shock meanZ 0.83.
NB the old docstring quoted ~97.3x -- that was the WHITENED metric (the whitened block of
{TAG}_fullgrid_ratio.json still reads 101.34 for this config). Run with RAW_METRIC=1."""
# Docstring reference values corrected 2026-08-04: they had been carried over from the
# whitened era and no longer matched what a re-run produces, so anyone verifying against
# them would have concluded their own run was broken. See REPLICATION.md sec 0.

import laplace_core as L
if __name__ == "__main__":
    L.run_and_dump("full", "Full Laplace", "full_laplace")
