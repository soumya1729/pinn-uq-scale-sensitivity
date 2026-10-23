# Build report

Status: **PASS, 23 October 2026.** All seven mathematical modules and the aggregate
library were rebuilt from the current release sources in a separate, initially
empty direct-compiler output directory. Mathematical sources and the three older
canonical Lean files were unchanged.

The rebuild used Lean 4.30.0 and the existing dependency cache. Every cached package
commit and clean tracked working/index tree was checked against the release's pinned manifest, including Mathlib commit
`c5ea00351c28e24afc9f0f84379aa41082b1188f`. This was a source-clean **project** rebuild,
not a new Mathlib dependency installation or a new download test.

| Check | Result |
|---|---|
| Pinned toolchain, all nine dependency commits and clean tracked trees | PASS |
| Seven project modules, direct compiler, warnings treated as failures | PASS |
| Aggregate library compiled from source | PASS |
| Direct `Audit.lean`, all 15 principal results | PASS; standard logical axioms only |
| Mathematical source identity and unchanged canonical files | PASS |
| Complete default Lake invocation | Not completed; resource/performance attempts were stopped |
| Fresh dependency download on a new machine | Not tested |

The initial all-target Lake invocation compiled `ModalModel` and was stopped when
it scheduled four heavy Mathlib importers concurrently in the 7.6 GiB WSL VM. A
one-module-at-a-time Lake route compiled `CovarianceRepresentation` and
`CoarseSpectralBounds`, but its setup/import overhead remained several minutes per
module. That route was also deliberately stopped. Neither interruption was a
reported proof error. The documented direct sequential compiler route then
**recompiled all seven modules and the aggregate library from source**, with
warnings treated as errors, and ran the 15-theorem axiom audit. No old project
proof objects were copied into that empty direct output directory.

The exact public helper exercised was:

```bash
LEAN_CACHE_PROJECT=/path/to/matching/lean-project LEAN_COMPILER=lean bash check_cached.sh
```

The cache project must contain the pinned dependency revisions under
`.lake/packages`; the audit used the verified existing cache. `check_cached.sh`
recompiles project sources on every invocation and runs the audit. It needs Bash
(e.g. WSL), Lean/Elan and a matching already installed Mathlib dependency cache.

The nine pinned-commit and clean tracked working/index checks are recorded in
[verification/dependency-tree-audit.json](verification/dependency-tree-audit.json).

The full commands, versions, dependency commits, interrupted Lake attempts and
successful direct build/audit are in
[verification/build-full.log](verification/build-full.log). The preceding
13 October successful log is retained as
[verification/build-20261013.log](verification/build-20261013.log).

All 15 printed axiom sets contain only Lean's standard logical axioms
(`propext`, `Classical.choice`, `Quot.sound`). There is no `sorryAx` or project-defined
axiom. The proofs establish the exact-real analytic statements in
[THEOREM_MAP.md](THEOREM_MAP.md), not numerical QR/SVD accuracy, empirical counts,
floating-point root brackets or nonlinear shifted-network predictions.

For a fresh installation, obtain the pinned dependencies/cache first, then use
the sequential helper against this project:

```bash
lake exe cache get
LEAN_CACHE_PROJECT="$PWD" LEAN_COMPILER=lean bash check_cached.sh
```

The ordinary Lake route from this directory also remains:

```bash
lake exe cache get
lake --wfail build
lake env lean Audit.lean
```

This is a supported project route, not a claim that its complete invocation or
fresh dependency-fetch step finished in the resource-limited audit reported here.
The actual fresh proof recompilation and axiom checks above used `check_cached.sh`.
