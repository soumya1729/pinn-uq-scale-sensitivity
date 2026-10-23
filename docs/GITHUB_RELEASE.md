# Publishing this prepared release

The source ZIP and the saved-data ZIP are separate deliverables. No upload has
been made by the preparation scripts.

1. Extract `PINN_UQ_GitHub_Source.zip` into the repository you intend to publish.
   Do not upload the large restored-data working directory as ordinary Git files.
2. Upload `pinn_uq_saved_data_20261022.zip` as a versioned release/data asset.
   Its exact checksum is recorded in `data/ASSET_SHA256.txt`. The source checkout
   contains the portable destination manifest and restoration utility.
3. Before submission, confirm that the repository link in the manuscript and the
   saved-data release assets are accessible to reviewers. Both URLs are recorded
   in the source documentation; no pending manuscript URL remains.
4. Keep the tagged source, the data asset, and `SOURCE_SHA256.json` together.
   Source and data versions must be identifiable independently of a moving branch.

The source includes the paper, scripts, compact reference records, focused Lean
library and numerical ledger. The data release supplies frozen fields and
checkpoints. The retained 276 GB factor archive and future-study files are not
required for this paper's ordinary saved-data replay.

`reproduce.py` checks the published calculations at the levels documented in its
report. The ledger maps repeated manuscript occurrences; a producer mapping alone
is not an independently checked scalar assertion. Exact historical retraining is
not claimed because the checkpoints lack a training-time source commit.

Internal cover letters, referee discussions, friend reviews, future-study plans,
local virtual environments and author-machine staging utilities are outside the
public source archive. The final Overleaf source is packaged independently.
