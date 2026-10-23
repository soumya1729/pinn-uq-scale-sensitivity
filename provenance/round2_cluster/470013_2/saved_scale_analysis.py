"""R2-A/B/C from completed N1 fields: contrast, sampled oracle and Burgers subsets.

No differentiation or fitting. Original trained centres and reference masks are fixed.
The oracle uses the reference error and is a capability diagnostic, not scale selection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(error, sigma, masks, width=1.0):
    rms_error = {r: np.sqrt(np.mean(error[m] ** 2)) for r, m in masks.items()}
    rms_sigma = {r: width * np.sqrt(np.mean(sigma[m] ** 2)) for r, m in masks.items()}
    target = rms_error["sharp"] / rms_error["bulk"]
    contrast = rms_sigma["sharp"] / rms_sigma["bulk"]
    rows = []
    for region, mask in masks.items():
        z = (np.abs(error[mask]) / sigma[mask]) / width
        rows.append(dict(region=region, count=int(mask.sum()), error_rms=float(rms_error[region]),
                         sigma_rms=float(rms_sigma[region]), R_tar=float(target),
                         R_sigma_rms=float(contrast), rho=float(target / contrast),
                         mean_Z=float(z.mean()), coverage68=float(np.mean(z <= 1)),
                         coverage95=float(np.mean(z <= 1.96)),
                         width_for68=float(np.quantile(z, .68, method="higher")),
                         width_for95=float(np.quantile(z / 1.96, .95, method="higher"))))
    return rows


def analyse(directory, inputs):
    observed, oracle, curves, checks = [], [], [], []
    paths = sorted(directory.glob("burgers_mlp_seed*_n1.json"))
    for path in paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        tag = path.stem.removesuffix("_n1")
        with np.load(inputs / tag / f"{tag}_fields.npz") as d:
            error = d["u_pred"].ravel() - d["u_exact"].ravel()
            gradient = d["grad_abs"].ravel()
            xs, ts = d["xs"].copy(), d["ts"].copy()
        p70, p95 = np.percentile(gradient, [70, 95])
        masks = {"bulk": gradient < p70, "transition": (gradient >= p70) & (gradient < p95),
                 "sharp": gradient >= p95}
        post = np.repeat(ts, len(xs)) >= 1 / np.pi
        with path.with_suffix(".csv").open(newline="", encoding="utf-8") as f:
            original = list(csv.DictReader(f))
        with np.load(path.with_suffix(".npz")) as data:
            np.testing.assert_array_equal(xs, data["xs"])
            np.testing.assert_array_equal(ts, data["ts"])
            for i, choice in enumerate(meta["choices"]):
                sigma = np.sqrt(data["variance"][i])
                gamma = choice["alpha"] * choice["noise_variance"]
                common = dict(tag=tag, centre="trained", **choice, effective_precision=gamma)
                for subset, select in (("full", np.ones(len(error), dtype=bool)), ("post_formation", post)):
                    selected_masks = {r: m & select for r, m in masks.items()}
                    rows = metrics(error, sigma, selected_masks)
                    observed.extend(dict(**common, subset=subset, **r) for r in rows)
                    if subset == "full":
                        for r in rows:
                            source = next(o for o in original if o["setting"] == choice["setting"]
                                          and float(o["alpha"]) == choice["alpha"]
                                          and float(o["noise_variance"]) == choice["noise_variance"]
                                          and o["region"] == r["region"])
                            for key in ("R_tar", "R_sigma_rms", "rho", "error_rms", "sigma_rms",
                                        "mean_Z", "coverage68", "coverage95"):
                                np.testing.assert_allclose(r[key], float(source[key]), rtol=2e-12, atol=1e-14)
                        if choice["setting"] == "prior_sweep":
                            curves.append(dict(**common, **next(r for r in rows if r["region"] == "sharp")))
                    if choice["setting"] == "prior_sweep":
                        width = next(r["width_for68"] for r in rows if r["region"] == "bulk")
                        scaled = metrics(error, sigma, selected_masks, width=width)
                        oracle.extend(dict(**common, subset=subset, width_multiplier=width,
                                           calibration_region="bulk", target_coverage68=.68, **r) for r in scaled)
                        bulk = next(r for r in scaled if r["region"] == "bulk")
                        assert bulk["coverage68"] >= .68
                
        checks.append(dict(tag=tag, original_metrics_reproduced=True))
        print(f"{tag}: saved metrics reproduced; contrast, oracle and post-formation summaries complete.")
    if not paths:
        raise ValueError("No completed Burgers-MLP N1 summaries found.")
    return observed, oracle, curves, checks


def plot_curves(rows, out):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    for seed, tag in enumerate(sorted({r["tag"] for r in rows})):
        points = sorted((r for r in rows if r["tag"] == tag), key=lambda r: r["effective_precision"])
        gamma = [r["effective_precision"] for r in points]
        colour = plt.cm.tab10(seed)
        for ax, key in zip(axes, ("R_sigma_rms", "rho")):
            ax.loglog(gamma, [r[key] for r in points], ".-", lw=1.2, ms=3, color=colour,
                      label=tag.removeprefix("burgers_mlp_seed"))
            operating = next(r for r in points if r["alpha_multiplier"] == 1)
            ax.plot(operating["effective_precision"], operating[key], "o", color=colour, ms=4)
            ax.set_xlabel(r"effective precision $\gamma=\alpha v$")
    axes[0].set_ylabel(r"uncertainty contrast $R_\sigma^{\mathrm{rms}}$")
    axes[1].set_ylabel(r"RMS gap $\rho=R_{\mathrm{tar}}/R_\sigma^{\mathrm{rms}}$")
    axes[1].axhline(1, ls="--", lw=.8, color=".4")
    axes[0].legend(title="seed", ncol=2, fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"uncertainty_contrast_and_gap.{ext}", dpi=220)
    plt.close(fig)


def summarise(observed, oracle, curves, out, checks):
    tags = sorted({r["tag"] for r in observed})
    summary = []
    for tag in tags:
        curve = [r for r in curves if r["tag"] == tag]
        full_oracle = [r for r in oracle if r["tag"] == tag and r["subset"] == "full" and r["region"] == "sharp"]
        # Closest sampled contrast agreement, not an interpolated or exact root.
        closest = min(full_oracle, key=lambda r: abs(np.log(r["rho"])))
        item = dict(tag=tag, uncertainty_contrast_min=min(r["R_sigma_rms"] for r in curve),
                    uncertainty_contrast_max=max(r["R_sigma_rms"] for r in curve),
                    closest_sampled_gamma=closest["effective_precision"],
                    closest_sampled_rho=closest["rho"], oracle_sharp68=closest["coverage68"],
                    oracle_sharp95=closest["coverage95"],
                    oracle_bulk95=next(r["coverage95"] for r in oracle
                                      if r["tag"] == tag and r["subset"] == "full" and r["region"] == "bulk"
                                      and r["effective_precision"] == closest["effective_precision"]))
        for subset in ("full", "post_formation"):
            for setting, selector in (("operating", lambda r: r["setting"] == "prior_sweep" and r["alpha_multiplier"] == 1),
                                      ("mle", lambda r: r["setting"] == "conditional_noise_mle")):
                row = next(r for r in observed if r["tag"] == tag and r["subset"] == subset
                           and r["region"] == "sharp" and selector(r))
                for key in ("R_tar", "R_sigma_rms", "rho", "coverage68", "coverage95"):
                    item[f"{subset}_{setting}_{key}"] = row[key]
        summary.append(item)
    write_csv(out / "saved_eight_summary.csv", summary)
    report = dict(completed_saved_runs=len(tags), production_cohort_complete=False,
                  remaining_unavailable_sensitivity_runs=9, shared_production_runs_pending=17,
                  centre="trained", nominal68=.68, interval95_multiplier=1.96,
                  oracle_quantile_method="higher", oracle_uses_reference=True,
                  post_formation_time_lower_bound=1 / np.pi,
                  masks="global original gradient percentiles; intersect subset without recomputing thresholds",
                  curve_lines="guides between sampled settings; no extrapolation or limit identification",
                  validation=checks, summary=summary)
    (out / "saved_eight_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Saved-output round-two analyses", "", "R2-A/B/C preliminary results on the eight completed Burgers–MLP N1 runs. All-17 sensitivity and affine-mean comparisons remain pending.", "",
             "The source CSV metrics reproduce from the archived fields. Trained predictions and original reference-defined masks are unchanged. Post-formation means t >= 1/pi with the same masks intersected with this time subset.", "",
             "The sampled oracle multiplies each saved unit-noise width field by the bulk 68th percentile of |e|/sigma (NumPy higher quantile). It uses the reference solution, calibrates bulk 68% coverage descriptively and does not select a practical likelihood. Width scaling leaves regional contrast and rho unchanged. 95% uses 1.96 standard deviations.", "",
             "The table uses the sampled precision closest to rho=1 in log distance; it is not an exact crossing or optimum over untested settings.", "",
             "| Seed | Closest sampled rho | Oracle sharp 68% | Oracle sharp 95% | Oracle bulk 95% |", "|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['tag'].removeprefix('burgers_mlp_seed')} | {r['closest_sampled_rho']:.3f} | {r['oracle_sharp68']:.3f} | {r['oracle_sharp95']:.3f} | {r['oracle_bulk95']:.3f} |")
    lines += ["", "All sampled settings, width factors, regional counts, coverage and post-formation contrasts are retained in the accompanying CSVs. No assertion that endpoint limits bound the curves or that sampled oracle failures apply to every scale is made.", "",
              "The contrast/gap figure separates covariance geometry from the trained error target. Connecting lines guide the eye across uneven sampled spacing; the figure does not identify the asymptotic limits."]
    (out / "SAVED_EIGHT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Summary:")
    for r in summary:
        print(f"{r['tag']}: closest rho={r['closest_sampled_rho']:.3g}, "
              f"oracle sharp68/95={r['oracle_sharp68']:.3f}/{r['oracle_sharp95']:.3f}; "
              f"MLE post/full rho={r['post_formation_mle_rho']:.3g}/{r['full_mle_rho']:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n1", type=Path, default=HERE.parent / "n1/outputs_cpu")
    parser.add_argument("--inputs", type=Path, default=HERE.parent / "n5/input")
    parser.add_argument("--out", type=Path, default=HERE / "outputs/saved_eight")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    observed, oracle, curves, checks = analyse(args.n1, args.inputs)
    write_csv(args.out / "regional_scale_metrics.csv", observed)
    write_csv(args.out / "sampled_oracle_coverage.csv", oracle)
    write_csv(args.out / "uncertainty_contrast_curves.csv", curves)
    plot_curves(curves, args.out)
    summarise(observed, oracle, curves, args.out, checks)


if __name__ == "__main__":
    main()
