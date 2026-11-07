# Reproduction release v1.0.1

Versioned sources for *Scale Sensitivity and Regional Calibration of Full-Network
Gauss–Newton Uncertainty in Physics-Informed Neural Networks*, by Soumyadip Mal
and Nagaiah Chamakuri.

## Sources and data

Use **v1.0.1** for the finalized 35-page manuscript, reproduction scripts,
numerical ledger and focused Lean library. GitHub's source archives for this
release contain the code and compact reference records. `SOURCE_SHA256.json`
records the individual source-file hashes; `CITATION.cff` records this version.

The unchanged data ZIP and its checksum remain in the **v1.0.0** release:

https://github.com/soumya1729/pinn-uq-scale-sensitivity/releases/tag/v1.0.0

Download `pinn_uq_saved_data_20261022.zip` and restore it with
`python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip`.
The restoration command verifies the ZIP checksum and all restored files. The
large data asset is not duplicated in Git or in the v1.0.1 release.

## Reviewer route

Start with the repository README and `docs/REVIEWER_GUIDE.md`. The saved-data
command is `python reproduce.py --mode saved --out reproduced`. It replays
saved-field/modal analyses and repeats the all-17 frozen-checkpoint readout fits.
It does not start PINN training, a residual-Jacobian/covariance pass or a cluster
job. The report records verification levels and comparison tolerances.

`docs/REVIEWER_EXECUTION.md` preserves the historical full replay and its limits.
No new scientific computations were performed for this presentation update.
The manuscript's numerical results, figure assets and Lean proofs are unchanged.
The current occurrence inventory contains 1944 entries: 14 linked to executed
registered checks and 1930 provenance mappings. A mapping is not an independently
verified numerical assertion.

The Lean project includes the pinned toolchain, dependency manifest, theorem map,
audit and build report. Its analytic verification is separate from numerical
data replay.

## Presentation and provenance

This release replaces stale author-facing instructions, repairs text encoding,
adds checkout/environment guidance and aligns the release and citation metadata.
Two current producer headers were shortened to describe their inputs, conventions
and optional checks. Their executable calculations are unchanged. The two lines
of manuscript-source commentary updated in this pass do not affect the PDF.

`docs/SCRIPT_INVENTORY.csv` retains hashes and locations from the original source
collection. Use `SOURCE_SHA256.json` for the current released source hashes.
Executed source snapshots, loading-runtime sources and their recorded hashes
remain byte-for-byte intact. Some retain historical comments and machine paths;
`provenance/baseline_release/LOADING_MIGRATION_NOTE.md` explains superseded loading
calculations and the current numerical targets. Those limitations and corrections
are preserved rather than silently rewriting historical records.

Original code is MIT-licensed; third-party and manuscript terms are preserved.
Exact historical retraining is not claimed: the frozen checkpoints are the
authoritative starting point for reproducing the reported analyses.

## Earlier release

The v1.0.0 source tag predates the finalized manuscript. It and its saved-data
assets are retained unchanged. Use v1.0.1 sources with the v1.0.0 data asset for
the current paper.
