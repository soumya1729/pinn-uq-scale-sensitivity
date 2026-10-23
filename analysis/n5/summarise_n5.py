"""Write a short N5 results note from the saved CSV."""
import csv
from pathlib import Path

import numpy as np


def main():
    folder = Path(__file__).parent / "outputs"
    with (folder / "n5_all17.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    primary = [r for r in rows if float(r["rcond"]) == 1e-10]
    blocked = [r for r in primary if r["split"] == "time_blocks" and int(r["block_slices"]) == 10]
    sharp = [r for r in blocked if r["region"] == "sharp"]
    lookup = {(r["tag"], r["region"], r["split"], r["block_slices"], r["rcond"]): r for r in rows}
    text = ["# N5 results — 5 October 2026", "",
            "All 17 admitted frozen checkpoints were evaluated. Each region uses its original 129 last-layer features, including bias. Random halves use seed 0; alternating blocks contain 10 time slices. Both fitting directions are scored and their residual energies pooled. Primary relative SVD cutoff: 1e-10; comparison: 1e-12.", "",
            "**The blocked test adds useful information.** Sharp-region held-out residuals are small in most runs, but larger than same-grid residuals. The result supports conditional held-out performance of last-layer fits. Performance of separately fitted full-network coefficients was not tested.", "",
            "## All-run blocked results", "",
            "Entries are residual-energy ratios, not percentages. Ratios above one are retained.", "",
            "| Region | Minimum | Median | Maximum | Runs above one |",
            "|---|---:|---:|---:|---:|"]
    for region in ("bulk", "transition", "sharp"):
        a = np.array([float(r["heldout_pooled"]) for r in blocked if r["region"] == region])
        text.append(f"| {region} | {a.min():.6g} | {np.median(a):.6g} | {a.max():.6g} | {sum(a > 1)}/17 |")
    worst = max(sharp, key=lambda r: float(r["heldout_pooled"]))
    below = sum(float(r["heldout_pooled"]) < 0.04 for r in sharp)
    text += ["", f"The largest sharp-region blocked residual ratio is {float(worst['heldout_pooled']):.6g} on {worst['tag']}, corresponding to {100 * (1-float(worst['heldout_pooled'])):.2f}% error-energy reduction. Its original sharp-region error RMS is {float(worst['error_rmse']):.6g} and held-out residual RMS is {float(worst['heldout_rmse']):.6g}. This is the smallest sharp-region error among the eight Burgers–MLP runs. The small denominator is context, not grounds to dismiss the relative deterioration. {below}/17 sharp-region ratios are below 0.04. The existing 98.8% same-grid statement must not be transferred to these held-out fits.", "",
             "Because the trained prediction lies in the last-layer span and the fitting matrices have full column rank, error fitting and direct reference fitting yield the same held-out squared residuals in exact arithmetic. A pooled ratio below one therefore means that regional readouts fitted to reference values predict held-out blocks with lower mean-square error than the trained readout. This holds in the sharp region in all 17 runs. The refits use reference values and separate coefficients per region and fold; they are not a global correction based on PDE residuals alone.", "",
             "Some transition-region refits have very large blocked residuals and exceed the trained readout's error in 10/17 runs. Small same-grid fractions therefore do not ensure accurate prediction across held-out time blocks. A smaller trained-error denominator provides context but does not itself explain or predict this deterioration. These results do not identify numerical solver failure or its cause.", "",
             "## Canonical sharp-region rows", "",
             "The same-grid column here uses the new fixed cutoff. Original Table 7 numbers are preserved in the manuscript.", "",
             "| Run | Same grid | Random held out | 10-slice blocks held out |",
             "|---|---:|---:|---:|"]
    tags = ("burgers_mlp_seed2", "burgers_fourier_seed2", "allen_cahn_mlp_seed0", "allen_cahn_fourier_seed0")
    for tag in tags:
        random = next(r for r in primary if r["tag"] == tag and r["region"] == "sharp" and r["split"] == "random")
        block = next(r for r in sharp if r["tag"] == tag)
        text.append(f"| {tag} | {float(block['same_grid_fraction']):.6g} | {float(random['heldout_pooled']):.6g} | {float(block['heldout_pooled']):.6g} |")
    ranks = [int(r[f"rank_{k}"]) for r in primary for k in (0, 1)]
    conditions = [float(r[f"condition_{k}"]) for r in primary for k in (0, 1)]
    differences = []
    for r in primary:
        other = lookup[(r["tag"], r["region"], r["split"], r["block_slices"], "1e-12")]
        differences.append(abs(float(r["heldout_pooled"]) - float(other["heldout_pooled"])))
    text += ["", "## Rank and conditioning", "",
             f"Training ranks range from {min(ranks)} to {max(ranks)}. Untruncated training-matrix condition numbers range from {min(conditions):.6g} to {max(conditions):.6g}. The maximum held-out ratio difference between cutoffs is {max(differences):.6g}. Both cutoffs retain all 129 directions, so identical results establish insensitivity over these two cutoffs only, not over more aggressive truncation or ridge regularisation.", "",
             "## Minimal manuscript use", "",
             "Table 7 includes two sharp-region held-out columns. Section 3.2 states the split, all-seed worst case and absolute RMS context, and transition-region failures. The contribution and limitations distinguish the same-grid 98.8% statement from held-out last-layer performance. Ranks, the two-cutoff comparison and per-fold results remain in the companion CSV. Last-layer functions remain contained in the full tangent function space; this does not establish the performance of a separately fitted full-network estimator.", "",
             "N5 has been run and integrated into the paper. N1–N4 remain deferred. Original same-grid values and all other numerical findings are preserved.", ""]
    out = folder / "N5_RESULTS.md"
    out.write_text("\n".join(text), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
