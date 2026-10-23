# Numerical occurrence ledger: scope and status

This ledger inventories **1996 scientific numerical occurrences** in **26 active TeX sources**, with repeated occurrences retained. It includes digit tokens and relevant spelled-out count patterns. It maps all **18 tables and six figures** to their documented producers and data. It does not assert that every value has been recomputed.

## How to rebuild

The reviewer entry point writes a fresh `ledger_inventory` directory automatically. To rebuild the index separately from the repository root, run `python tools/build_numerical_ledger.py --paper-root paper --out new_ledger --reproduced reproduced`. The supplied page and label indexes correspond to the released PDF; refresh them after a new LaTeX build or omit stale page coordinates. Paths in the records are relative to the public repository layout.

## What was automated

- Recursively expand active `\input`/`\include` commands starting at `full_paper.tex`; backup and inactive drafts are not scanned.
- Strip comments while preserving physical source-line coordinates.
- Suppress unprinted macro definitions and expand zero-argument numerical macros at each printed use. Each expanded value retains its call-site line and definition location.
- Inventory literal digit tokens, decimal/scientific expressions, powers of ten and count ratios, plus verbal counts near relevant scientific nouns (for example, five roots, eight seeds, two subspaces and one standard deviation). Include analytic exponents and declared settings rather than silently interpreting all numerals as empirical results.
- Associate each occurrence with its source line, current section, float/equation labels, and a candidate evidence chain. Caption aliases do not create extra tables or figures.
- Use current auxiliary-file page anchors for tables/figures and labelled equations, and unique SyncTeX source-line pages when supplied. Unresolved body-text pages remain blank; a section-start page is never represented as the actual occurrence page.

## What remains manual or separately computed

- Classification is a conservative lexical first pass. Configuration/analytic/result distinctions require semantic review, especially measured counts in setting paragraphs and nominal probabilities beside observed coverage.
- `PROVENANCE_MAPPED` means a documented producer/data chain exists. It is **not** `CHECKED`, `PASS` or agreement with raw data. Topic/file-level mappings are labelled explicitly.
- Each empirical occurrence needs a specific source selector, aggregation and rounding rule before a numerical checker can mark it checked. Unlinked quantities use occurrence-specific IDs instead of pretending equal printed values are the same measurement.
- `display_match=NOT_TESTED` and blank `recomputed_value` are intentional for provenance-only rows. With `--reproduced`, explicit registered claim spans and numerical-token roles are joined to their executed checks; no value is marked verified by matching its string to an arbitrary file.
- Analytic entries point to the manuscript statement/proof. The extractor does not infer Lean verification of each literal constant.

## Exact coverage boundaries

The inventory is a **lexical extraction with verbal-count patterns**, not a claim of exhaustive semantic extraction. Verbal numerals without a recognized nearby noun, numeric tick labels inside six figure assets, automatic equation/table/section numbering, bibliography metadata and numeric code identifiers are not scientific occurrence rows. Figure science is covered at the output/provenance level and requires its plot-data checks. Citation keys, cross-reference keys, document metadata and layout digits are separated in `document_metadata.csv`; bibliography files are not expanded.

Generated tables are included through the same active-input expansion as prose. A table interval may generate two endpoint rows; a scientific expression such as `1.94\times10^{-4}` is one row. Source context retains the surrounding interval, inequality, units and shared powers. Table-column coordinates are source-cell hints, not a reconstructed multirow semantic schema.

The public interface should distinguish a completed numerical check from this provenance map. Missing data or unresolved claims must remain visible. The source hash/aux hash permits checking that the ledger corresponds to the released manuscript, and must be refreshed after pagination or text edits.

## Current inventory totals

- Classifications: `{"numerical-result": 1576, "configuration": 212, "analytic": 208}`.
- Statuses: `{"CHECKED": 13, "PROVENANCE_MAPPED": 1983}`.
- Occurrences with resolved source-line or float/equation page anchors: 1971.
- Excluded document/reference/layout tokens: 28.
- Unmatched documented outputs: `[]`.
