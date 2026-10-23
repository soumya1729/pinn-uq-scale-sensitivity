"""Plot regional coverage and mean standardised error from saved round-two fields.

No network evaluation, Jacobian factorisation, evidence search or fitted scale.
The noise panel only rescales the saved baseline covariance at fixed gamma.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--paper-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    source, paper = args.source_root, args.paper_root
    saved = read_csv(source / "analysis/round2/outputs/production_review/regional_scale_metrics.csv")
    selected = [r for r in saved if r["centre"] == "trained" and r["subset"] == "full"
                and r["region"] in ("bulk", "sharp")]
    tags = list(dict.fromkeys(r["tag"] for r in selected))
    noise = np.logspace(-10, 2, 121)
    rows = []
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8), sharex="col")
    families = ("burgers_mlp", "burgers_fourier", "allen_cahn_mlp", "allen_cahn_fourier")
    names = ("Burgers MLP", "Burgers Fourier", "Allen-Cahn MLP", "Allen-Cahn Fourier")
    colors = dict(zip(families, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    for tag in tags:
        color = colors[tag.rsplit("_seed", 1)[0]]
        stem = source / "analysis/round2/outputs/production" / (tag + "_round2")
        summary = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(str(stem) + "_fields.npz", allow_pickle=False) as data:
            gamma = data["gammas"]
            k = int(np.argmin(np.abs(np.log(gamma / summary["alpha0"]))))
            np.testing.assert_allclose(gamma[k], summary["alpha0"], rtol=1e-13)
            sigma = np.sqrt(data["unit_variance"][k])
            error = np.abs(data["u_pred"] - data["u_exact"])
            gradient = data["grad_abs"]
            masks = {"bulk": gradient < summary["region_thresholds"]["p70"],
                     "sharp": gradient >= summary["region_thresholds"]["p95"]}
            for region in ("bulk", "sharp"):
                style = "--" if region == "bulk" else "-"
                unit = [r for r in selected if r["tag"] == tag and r["region"] == region
                        and r["setting"] == "prior_sweep" and float(r["noise_variance"]) == 1]
                unit = sorted({float(r["effective_precision"]): r for r in unit}.values(),
                              key=lambda r: float(r["effective_precision"]))
                x = [float(r["effective_precision"]) for r in unit]
                for ax, metric in zip(axes[:, 0], ("coverage68", "mean_Z")):
                    ax.plot(x, [float(r[metric]) for r in unit], style, color=color,
                            linewidth=1.15, alpha=.65)
                for r in unit:
                    rows.append(dict(tag=tag, region=region, sweep="gamma_at_unit_noise",
                        effective_precision=float(r["effective_precision"]), noise_variance=1,
                        coverage68=float(r["coverage68"]), mean_Z=float(r["mean_Z"])))
                z = error[masks[region]] / sigma[masks[region]]
                ordered = np.sort(z)
                coverage = np.searchsorted(ordered, np.sqrt(noise), side="right") / len(z)
                mean_z = z.mean() / np.sqrt(noise)
                baseline = next(r for r in unit if float(r["alpha_multiplier"]) == 1)
                # Independent rescaling reproduces the saved baseline metrics at v=1.
                at_one = int(np.argmin(np.abs(np.log(noise))))
                np.testing.assert_allclose(coverage[at_one], float(baseline["coverage68"]), atol=1e-14)
                np.testing.assert_allclose(mean_z[at_one], float(baseline["mean_Z"]), rtol=2e-12)
                for ax, y in zip(axes[:, 1], (coverage, mean_z)):
                    ax.plot(noise, y, style, color=color, linewidth=1.15, alpha=.65)
                for v, c, m in zip(noise, coverage, mean_z):
                    rows.append(dict(tag=tag, region=region, sweep="noise_at_baseline_gamma",
                        effective_precision=float(gamma[k]), noise_variance=float(v),
                        coverage68=float(c), mean_Z=float(m)))
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=.16)
    for ax in axes[0]:
        ax.set_ylim(-.015, 1.03)
        ax.axhline(.682689492137086, color=".3", linestyle=":", linewidth=1)
    for ax in axes[1]:
        ax.set_yscale("log")
        ax.axhline(np.sqrt(2 / np.pi), color=".3", linestyle=":", linewidth=1)
    axes[0, 0].set_title(r"Saved precision settings: $v=1$")
    axes[0, 1].set_title(r"Width rescaling: $\gamma=\alpha_0$")
    axes[0, 0].set_ylabel("One-sigma coverage")
    axes[1, 0].set_ylabel(r"Mean standardised error $\overline{Z}$")
    axes[1, 0].set_xlabel(r"Effective precision $\gamma=\alpha v$")
    axes[1, 1].set_xlabel(r"Noise variance $v$ (with $\alpha=\alpha_0/v$)")
    family_handles = [Line2D([], [], color=colors[f], linewidth=2, label=n)
                      for f, n in zip(families, names)]
    fig.legend(handles=family_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=13.5)
    region_handles = [Line2D([], [], color=".3", linestyle=style, label=name)
                      for style, name in (("--", "Bulk"), ("-", "Sharp"), (":", "Gaussian benchmark"))]
    fig.legend(handles=region_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.02), frameon=False, fontsize=13.5)
    fig.subplots_adjust(left=.105, right=.985, top=.81, bottom=.15, hspace=.42, wspace=.20)
    figures = paper / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"round2_coverage_scale.{extension}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    output = paper / "analysis/round2/outputs/reporting_closure"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coverage_scale.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Plotted all {len(tags)} checkpoints; {len(rows)} saved/rescaled regional rows.")


if __name__ == "__main__":
    main()
