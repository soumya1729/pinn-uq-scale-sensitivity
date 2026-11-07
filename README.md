# Scale Sensitivity and Regional Calibration of PINN Uncertainty

Reproduction sources for **Scale Sensitivity and Regional Calibration of Full-Network Gauss–Newton Uncertainty in Physics-Informed Neural Networks**, by Soumyadip Mal and Nagaiah Chamakuri.

[Read the paper](paper/full_paper.pdf) · [Reviewer guide](docs/REVIEWER_GUIDE.md) · [Versioned sources](https://github.com/soumya1729/pinn-uq-scale-sensitivity/releases/tag/v1.0.1) · [Saved data](https://github.com/soumya1729/pinn-uq-scale-sensitivity/releases/tag/v1.0.0)

Start with the saved-data route below. It runs the existing numerical consumers through one entry point. It does not train networks, construct Jacobians or submit cluster jobs.

**Reviewer starting point:** [section-by-section guide](docs/REVIEWER_GUIDE.md).
It maps Sections 2–3.3, Appendices A–F, all 18 tables and all six figures to their
producers, data and formal proofs.

## Check the reported results

1. Obtain the versioned sources and create a Python 3.12 virtual environment:

   ```bash
   git clone --branch v1.0.1 --depth 1 https://github.com/soumya1729/pinn-uq-scale-sensitivity.git
   cd pinn-uq-scale-sensitivity
   python -m venv .venv
   ```

   Activate it with `source .venv/bin/activate` on Linux, macOS or WSL, or
   `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the tested
   review dependencies:

   ```bash
   python -m pip install -r envs/reviewer_requirements.txt
   ```

2. From this repository's **Releases → v1.0.0 → Assets**, download
   `pinn_uq_saved_data_20261022.zip`, then restore it:

   ```bash
   python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip
   ```

3. Recompute the saved-data summaries into a new directory:

   ```bash
   python reproduce.py --mode saved --out reproduced
   ```

This route is CPU-only. The data download is about 1.65 GB and the restored
files occupy about 2.38 GB; allow additional space for the Python environment
and regenerated outputs. A historical tested replay took about five minutes;
this is not a runtime guarantee. See [execution evidence](docs/REVIEWER_EXECUTION.md).

The command writes a task report, fresh numerical summaries and logs. `--mode records` is the smaller route using included records; it is **not** an independent field-based verification. Read `docs/REVIEWER_COMMAND.md` for the exact tasks and comparison tolerances.

Saved mode also rebuilds the 129 last-layer features from all 17 checkpoints and
repeats the held-out readout fits. These are small linear regressions, not new PINN
training. The tested complete dependency freeze is in
`envs/reviewer_freeze_windows.txt`.

The accompanying numerical ledger records repeated numerical occurrences in the active manuscript, their section/page anchors, inputs and producing scripts. Its status distinguishes **provenance mapping** from an executed numerical comparison. Configuration constants and mathematical statements are identified separately. The spreadsheet is a navigable index; the Python producers perform the scientific calculations. See `docs/NUMERICAL_LEDGER.md` for extraction coverage and limitations.

## Find one result

- `paper/`: the self-contained manuscript source and six figure assets.
- `docs/PAPER_OUTPUT_MAP.md`: all 18 tables and six figures, with their producers.
- `numerical_ledger/`: occurrence-level CSV, Excel accompaniment and table/figure map.
- `analysis/`: saved-data consumers and covariance/feature producers.
- `training/` and `legacy_core/`: recovered training and original diagnostic code.
- `lean/`: the focused formal development, pinned dependencies and theorem map.
- `data/manifest.json`: saved-data destinations and SHA-256 hashes.

The data ZIP contains 172 unique objects restored into the 331 layouts expected by the existing consumers. Duplicate scientific inputs are stored once in the ZIP. The much larger retained QR/SVD archive is unnecessary for these reported-result checks.

## Deeper reproduction

`docs/RUN_ORDER.md` documents the individual producers, including the last-layer fits and deliberate checkpoint-to-covariance recomputation. Those routes have different costs and dependencies. Historical Slurm files document the authors' cluster execution and must be adapted to another site; they are not needed for saved-data replay.

The frozen checkpoints are the authoritative starting point. Their records do not identify a training-time source commit. Recovered training code and recorded settings are preserved, but exact historical retraining is not claimed. See `docs/PUBLIC_PROVENANCE.md` and `docs/ENVIRONMENTS.md`.

To check the mathematical library separately, from Bash/WSL with Lean/Elan installed:

```bash
cd lean
lake exe cache get
LEAN_CACHE_PROJECT="$PWD" LEAN_COMPILER=lean bash check_cached.sh
```

Lean checks the stated analytic results under their assumptions. It does not certify the floating-point arrays, empirical run counts or training history.
The sequential helper recompiles every project module and runs the axiom audit.
See `lean/BUILD_REPORT.md` for the tested build and the ordinary Lake alternative.

## Licence and acknowledgement

Original code is MIT-licensed to Soumyadip Mal and Nagaiah Chamakuri. Existing third-party terms are retained in `THIRD_PARTY_NOTICES.md`; the manuscript and externally authored components retain their own terms.

The production computations used the Padmanabha high-performance computing facility at IISER Thiruvananthapuram. The authors acknowledge ANRF, Department of Science and Technology, India (ANRF/ARGM/2025/000954/MTR), for their financial support.

Repository: https://github.com/soumya1729/pinn-uq-scale-sensitivity

The finalized manuscript and reproduction sources are versioned as `v1.0.1`.
The unchanged saved-data asset and its checksum remain in `v1.0.0`; the source
version and data version are intentionally separate. See [release notes](docs/RELEASE_NOTES.md).
