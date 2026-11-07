# Public release provenance

This release preserves the scientific sources and frozen records used for the
paper, including the completed 17-checkpoint theta-centred comparison. The
29 October final manuscript compiles to 35 pages after prose and layout consolidation;
the underlying scientific
calculations and frozen data are unchanged.
The paper's numerical producers are documented in `PAPER_OUTPUT_MAP.md`.

The public candidate was collected on 22 October 2026 from the completed
16 October replication collection. Internal manuscript reviews, friend/referee
discussions, future-study planning, dependency caches and the old manuscript ZIP
were omitted. Their private originals remain unchanged. The public manuscript
source is packaged separately from the scientific collection.

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
