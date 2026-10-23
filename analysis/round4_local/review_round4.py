"""Saved-field oracle diagnostics; no network, gradients or factorisation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def rank_correlation(a, b):
    def ranks(x):
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1)/2)[inverse]
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def stats(error, sigma, multiplier, mask):
    # Compare raw standardised errors directly with the quantile threshold.
    # Equality stays inclusive without multiplying rounded pointwise widths.
    raw = error[mask] / sigma[mask]
    signed = raw / multiplier
    absolute = abs(signed)
    energy = error[mask] ** 2
    ntop = int(np.ceil(.01 * mask.sum()))
    return dict(count=int(mask.sum()), mean_Z=float(absolute.mean()),
                signed_standardized_mean=float(signed.mean()), rms_Z=rms(signed),
                median_Z=float(np.median(absolute)),
                q99_Z=float(np.quantile(absolute, .99)),
                coverage68=float(np.mean(abs(raw) <= multiplier)),
                coverage95=float(np.mean(abs(raw) <= 1.96*multiplier)),
                top1_error_energy_fraction=float(np.partition(energy, len(energy)-ntop)[-ntop:].sum()/energy.sum()),
                spearman_sigma_abs_error=rank_correlation(sigma[mask], abs(error[mask])))


def make_rows(tag, family, subset, masks, error, unit_variance, gamma, index, target, contrast, calibration):
    sigma = np.sqrt(unit_variance)
    multiplier = float(np.quantile(abs(error[masks[calibration]])/sigma[masks[calibration]], .68, method="higher"))
    common = dict(tag=tag, family=family, centre="trained", subset=subset,
                  gamma_index=int(index), gamma=float(gamma), R_tar=float(target),
                  R_sigma_rms=float(contrast), rho=float(target/contrast),
                  calibration_region=calibration, width_multiplier=multiplier,
                  target_coverage68=.68, quantile_method="higher")
    return [dict(common, region=region, **stats(error, sigma, multiplier, mask)) for region, mask in masks.items()]


def range_of(rows, key):
    return [min(row[key] for row in rows), max(row[key] for row in rows)]


TABLE_HEAD = '% Recomputed from frozen round-two fields by analysis/round4_local/review_round4.py.\n\\begin{table}[!htbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\nBurgers--MLP seed & $\\rho$ & $C^{\\mathcal B}_{\\mathcal S}(1)$ & $C^{\\mathcal B}_{\\mathcal S}(1.96)$ & $C^{\\mathcal S}_{\\mathcal S}(1.96)$ & $\\overline{|Z|}^{\\mathcal S}$ & $\\overline Z^{\\mathcal S}$\\\\\n\\midrule\n'
TABLE_MIDDLE = '\\midrule\n\\multicolumn{2}{l}{Family / subset ($n$)} & \\multicolumn{2}{c}{$C^{\\mathcal S}_{\\mathcal S}(1.96)$} & $\\overline{|Z|}^{\\mathcal S}$ & \\multicolumn{2}{c}{$\\overline Z^{\\mathcal S}$}\\\\\n\\midrule\n'
TABLE_TAIL = "\\bottomrule\n\\end{tabular}\n\\caption{Trained-centre regional calibration at each run's closest saved contrast setting.\nThe superscript $\\mathcal B$ or $\\mathcal S$ denotes the region whose higher empirical\n$0.68$ quantile supplies the width multiplier. The upper panel gives eight full-grid\nBurgers--MLP seeds; the lower panel gives observed ranges for all $17$ full-grid runs\nand the Burgers post-formation subsets. Each setting minimises $|\\log\\rho|$, with ties\nat the lowest saved precision. $\\overline{|Z|}^{\\mathcal S}$ and $\\overline Z^{\\mathcal S}$\nare absolute and signed means of sharp-calibrated standardised errors.\nSharp one-standard-deviation coverage is fixed by construction, up to discreteness\nand ties. Displayed values are rounded; reference calibration does not define a training rule.}\n\\label{tab:r4-oracles}\n\\end{table}\n"


def publication_table(nearest):
    labels = {"burgers_mlp": "Burgers--MLP", "burgers_fourier": "Burgers--Fourier",
              "allen_cahn_mlp": "Allen--Cahn--MLP", "allen_cahn_fourier": "Allen--Cahn--Fourier"}
    lines = TABLE_HEAD.splitlines()
    for seed in range(8):
        rows = [r for r in nearest if r["tag"] == f"burgers_mlp_seed{seed}"
                and r["subset"] == "full" and r["region"] == "sharp"]
        bulk = next(r for r in rows if r["calibration_region"] == "bulk")
        sharp = next(r for r in rows if r["calibration_region"] == "sharp")
        cells = [seed, sharp["rho"], bulk["coverage68"], bulk["coverage95"],
                 sharp["coverage95"], sharp["mean_Z"], sharp["signed_standardized_mean"]]
        cells = [f"{v:.5g}" for v in cells]
        cells[-1] = "$" + cells[-1] + "$"
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(TABLE_MIDDLE.splitlines())
    for subset in ("full", "post_formation"):
        for family, label in labels.items():
            rows = [r for r in nearest if r["family"] == family and r["subset"] == subset
                    and r["region"] == "sharp" and r["calibration_region"] == "sharp"]
            if not rows:
                continue
            spans = ["$[" + r",\,".join(f"{v:.5g}" for v in range_of(rows, k)) + "]$"
                     for k in ("coverage95", "mean_Z", "signed_standardized_mean")]
            subset_label = "full" if subset == "full" else "post"
            lines.append(r"\multicolumn{2}{l}{" + f"{label} / {subset_label} (${len(rows)}$)" + "} & "
                         + r"\multicolumn{2}{c}{" + spans[0] + "} & " + spans[1] + " & "
                         + r"\multicolumn{2}{c}{" + spans[2] + r"} \\")
    lines.extend(TABLE_TAIL.splitlines())
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent/"outputs")
    args = parser.parse_args()
    project = args.project_root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((project/"analysis/round2/cohort.json").read_text())["cohort"]
    reference = read_csv(project/"analysis/round3_local/outputs/oracle_distributions.csv")
    ref = {(r["tag"], r["subset"], r["region"]): r for r in reference if r["centre"] == "trained"}
    old_band = read_csv(project/"analysis/round2/outputs/production_review/sampled_oracle_coverage.csv")
    old = {(r["tag"], r["subset"], int(r["gamma_index"]), r["region"]): r for r in old_band if r["centre"] == "trained"}
    nearest, band = [], []
    checked_nearest = checked_band = 0
    maximum_previous_coverage_count_difference = 0
    for run in cohort:
        tag = run["tag"]
        family = f'{run["pde"]}_{run["architecture"]}'
        path = project/"analysis/round2/outputs/production"/f"{tag}_round2_fields.npz"
        with np.load(path) as z:
            error = z["u_pred"] - z["u_exact"]
            gradient = z["grad_abs"]
            p70, p95 = np.percentile(gradient, (70, 95))
            regions = dict(bulk=gradient < p70, transition=(gradient >= p70) & (gradient < p95), sharp=gradient >= p95)
            subsets = {"full": regions}
            if run["pde"] == "burgers":
                post = np.repeat(z["ts"], len(z["xs"])) >= 1/np.pi
                subsets["post_formation"] = {r: m & post for r, m in regions.items()}
            for subset, masks in subsets.items():
                target = rms(error[masks["sharp"]])/rms(error[masks["bulk"]])
                variance = z["unit_variance"]
                contrasts = np.sqrt(variance[:, masks["sharp"]].mean(axis=1)/variance[:, masks["bulk"]].mean(axis=1))
                rhos = target/contrasts
                # Stored gammas are increasing: first minimum breaks ties at the lowest gamma.
                index = int(np.argmin(abs(np.log(rhos))))
                for calibration in ("bulk", "sharp"):
                    rows = make_rows(tag, family, subset, masks, error, variance[index], z["gammas"][index], index, target, contrasts[index], calibration)
                    nearest.extend(rows)
                    if calibration == "bulk":
                        for row in rows:
                            previous = ref[tag, subset, row["region"]]
                            assert index == int(previous["gamma_index"])
                            for key in ("mean_Z", "rms_Z", "signed_standardized_mean", "median_Z", "q99_Z", "spearman_sigma_abs_error"):
                                np.testing.assert_allclose(row[key], float(previous[key]), rtol=1e-12, atol=1e-14)
                            for key in ("coverage68", "coverage95"):
                                difference = round(abs(row[key]-float(previous[key]))*row["count"])
                                assert difference <= 1  # One rounded equality at the quantile can differ.
                                maximum_previous_coverage_count_difference = max(maximum_previous_coverage_count_difference, difference)
                            checked_nearest += 1
                for i in np.flatnonzero((rhos >= .9) & (rhos <= 1.1)):
                    rows = make_rows(tag, family, subset, masks, error, variance[i], z["gammas"][i], i, target, contrasts[i], "bulk")
                    band.extend(rows)
                    for row in rows:
                        previous = old[tag, subset, int(i), row["region"]]
                        for key in ("rho", "mean_Z", "rms_Z"):
                            np.testing.assert_allclose(row[key], float(previous[key]), rtol=1e-12, atol=1e-14)
                        for key in ("coverage68", "coverage95"):
                            difference = round(abs(row[key]-float(previous[key]))*row["count"])
                            assert difference <= 1
                            maximum_previous_coverage_count_difference = max(maximum_previous_coverage_count_difference, difference)
                        checked_band += 1
        print(f"{tag}: saved-field summaries complete")
    write_csv(out/"nearest_saved_oracles.csv", nearest)
    write_csv(out/"sampled_contrast_agreement_band.csv", band)
    summaries = []
    for family in sorted({r["family"] for r in nearest}):
        for subset in ("full", "post_formation"):
            sharp = [r for r in nearest if r["family"] == family and r["subset"] == subset and r["calibration_region"] == "sharp" and r["region"] == "sharp"]
            if not sharp:
                continue
            summaries.append(dict(family=family, subset=subset, runs=len(sharp),
                coverage95_below_nominal_count=sum(r["coverage95"] < .95 for r in sharp),
                **{f"{key}_range": range_of(sharp, key) for key in ("rho", "coverage68", "coverage95", "mean_Z", "signed_standardized_mean", "median_Z", "q99_Z")}))
    band_summaries = []
    for tag, subset in sorted({(r["tag"], r["subset"]) for r in band}):
        rows = [r for r in band if r["tag"] == tag and r["subset"] == subset and r["region"] == "sharp"]
        band_summaries.append(dict(tag=tag, subset=subset, saved_gamma_count=len(rows),
                                  rho_range=range_of(rows, "rho"), coverage68_range=range_of(rows, "coverage68"),
                                  coverage95_range=range_of(rows, "coverage95")))
    summary = dict(cohort_runs=len(cohort), numpy_version=np.__version__,
                   checked_previous_nearest_rows=checked_nearest, checked_previous_band_rows=checked_band,
                   maximum_previous_coverage_count_difference=maximum_previous_coverage_count_difference,
                   family_sharp_calibrated_summaries=summaries, sampled_band_summaries=band_summaries,
                   conventions={"centre": "trained prediction only", "nearest": "min abs(log rho), tie lowest saved gamma",
                       "regions": "full-grid p70/p95 reference-gradient thresholds, retained for the post-formation subset",
                       "post_formation": "Burgers t >= 1/pi", "calibration": "higher empirical .68 quantile of abs(e)/unit-sigma",
                       "coverage95": "abs(e) <= 1.96 times calibrated sigma", "band": "all saved gamma samples with .9 <= rho <= 1.1; no continuous-band claim",
                       "scope": "reference-informed descriptive diagnostics; sharp68 coverage is fixed by construction, up to ties/discreteness"})
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    (out/"round4_oracle_table.tex").write_text(publication_table(nearest), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
