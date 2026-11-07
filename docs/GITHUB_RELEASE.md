# Release layout and versioning

The finalized manuscript and reproduction sources are tagged **v1.0.1**. Use that
tag or its source archive for the version accompanying the 35-page manuscript.
`SOURCE_SHA256.json` records the individual source-file hashes.

The saved-data asset remains under **v1.0.0**:

https://github.com/soumya1729/pinn-uq-scale-sensitivity/releases/tag/v1.0.0

Download `pinn_uq_saved_data_20261022.zip` and restore it with
`python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip`.
Its expected checksum is recorded in `data/ASSET_SHA256.txt`; the restoration
utility also verifies every destination file. GitHub's automatically generated
source archives contain the code, not the larger saved-data ZIP. The v1.0.0
source tag predates the finalized manuscript; its data asset is unchanged.

The source includes the paper, scripts, compact reference records, focused Lean
library and numerical ledger. The data release supplies frozen fields and
checkpoints. The retained 276 GB factor archive and future-study files are not
required for this paper's ordinary saved-data replay.

`reproduce.py` checks the published calculations at the levels documented in its
report. The ledger maps repeated manuscript occurrences; a producer mapping alone
is not an independently checked scalar assertion. Exact historical retraining is
not claimed because the checkpoints lack a training-time source commit.

The source archive excludes private correspondence, manuscript backups,
future-study files, local virtual environments and machine-specific packaging
utilities. The Overleaf manuscript source is packaged independently.
