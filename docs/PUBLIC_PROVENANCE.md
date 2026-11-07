# Public release provenance

This release preserves the scientific sources and frozen records used for the
paper, including the completed 17-checkpoint theta-centred comparison. The
current manuscript compiles to 35 pages after prose and layout consolidation.
The 7 November documentation and copyediting updates did not change the
scientific calculations or frozen data.
The paper's numerical producers are documented in `PAPER_OUTPUT_MAP.md`.

The original release was collected on 22 October 2026 from the completed
16 October replication collection. Private correspondence, manuscript backups,
future-study files and dependency caches are excluded from the public package.
The manuscript source is also available as a separate Overleaf archive.

Scientific records are preserved byte-for-byte where they form the input or
execution hash chain. Thus some immutable JSON and Slurm records contain
historical machine paths, accounts, job identifiers and source locations. These
are provenance, not executable requirements. `data/manifest.json` is the portable
public data inventory; `tools/restore_data.py` restores arrays without accessing
the original machines. `SCRIPT_INVENTORY.csv` retains historical collection-source
paths alongside public relative destinations and hashes. Those source paths are
provenance annotations, not files a reviewer must have. Private release-staging
maps and internal review notes are kept outside the public repository.

The frozen checkpoints are the authoritative starting point for reproducing
reported numerical analyses. The surviving checkpoints do not contain a
training-time source commit/hash. Recovered training code and full recorded
settings are included, but exact historical retraining is not claimed.

Saved-record checks, saved-array recalculation, checkpoint recomputation and Lean
verification are distinct routes. A successful comparison of compact reference
records alone does not establish an independent recalculation from prediction
arrays. The mathematical Lean proofs do not certify floating-point arrays,
empirical run counts or training provenance.

The public final file manifest is generated after manuscript and numerical-ledger
integration. The original execution hashes remain in their scientific records;
packaging changes do not overwrite them.

`.gitattributes` disables automatic newline conversion so Git checkouts preserve
the exact source bytes used by scientific hash contracts. This does not change
the formulas or prevent ordinary text diffs.
