# Paper reproduction release v1.0.0

Sources, recorded inputs and reviewer tools for *Scale Sensitivity of Regional
Uncertainty in Physics-Informed Neural Networks: A Full-Network Gauss–Newton Study*,
by Soumyadip Mal and Nagaiah Chamakuri.

Start with `README.md` and `docs/REVIEWER_GUIDE.md`. The latter maps the paper's
sections, 18 tables and six figures to code and data.

Download `pinn_uq_saved_data_20261022.zip` from this release's Assets and restore it
with `python tools/restore_data.py --archive /path/to/the.zip`. The restore tool
checks the asset hash and every restored file. `data/ASSET_SHA256.txt` records
the expected checksum. GitHub's automatically generated source archives contain
the code; they deliberately omit this larger saved-data asset.

The reviewer command is `python reproduce.py --mode saved --out reproduced`.
It replays saved-field/modal analyses and repeats the all-17 frozen-checkpoint
readout fits. It never starts PINN training, a residual-Jacobian/covariance pass or a cluster job.
Verification levels and comparisons are recorded in its report.

The focused Lean project is in `lean/`, with its pinned toolchain, dependency
manifest, theorem map, audit and build report. It verifies the analytic statements
under their stated assumptions, separately from numerical data.

`docs/REPRODUCTION_STATUS.md` records the release checks and their limits.
Original code is MIT-licensed; third-party and manuscript terms are preserved.
