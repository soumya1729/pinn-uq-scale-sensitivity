"""Report existing all-17 archives; no model evaluation or new scale search.

From the paper root:
    python analysis/round2/report_saved_summaries.py

The optional control record is a frozen extract of the existing control summary.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

FAMILIES = ("burgers_mlp", "burgers_fourier", "allen_cahn_mlp", "allen_cahn_fourier")
NAMES = dict(zip(FAMILIES, ("Burgers--MLP", "Burgers--Fourier",
                          "Allen--Cahn--MLP", "Allen--Cahn--Fourier")))
SETTINGS = ("baseline", "conditional_mle", "original_evidence", "extended_evidence")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family(tag):
    return tag.rsplit("_seed", 1)[0]


def selected_setting(row):
    setting = row["setting"]
    if setting == "prior_sweep" and float(row["alpha_multiplier"]) == 1 and float(row["noise_variance"]) == 1:
        return "baseline"
    if setting == "conditional_noise_mle":
        return "conditional_mle"
    if setting.startswith("affine_evidence_original_"):
        return "original_evidence"
    if setting.startswith("affine_evidence_extended_"):
        return "extended_evidence"
    return None


def number(value, digits=3):
    value = float(value)
    if value == 0:
        return "0"
    if .001 <= abs(value) < 1000:
        return format(value, f".{digits}g")
    mantissa, exponent = format(value, f".{digits-1}e").split("e")
    return mantissa.rstrip("0").rstrip(".") + r"\times10^{" + str(int(exponent)) + "}"


def value_range(values, digits=3):
    low, high = min(values), max(values)
    if low == high:
        return "$" + number(low, digits) + "$"
    return "$[" + number(low, digits) + r",\," + number(high, digits) + "]$"


def row(*values):
    return " & ".join(values) + r" \\"


def table(caption, label, columns, heading, rows):
    return ("% Derived from existing saved archives only.\n"
            + r"\begin{table}[htbp]" + "\n" + r"\centering\footnotesize\setlength{\tabcolsep}{4pt}" + "\n"
            + r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n"
            + r"\begin{tabular}{@{}" + columns + "@{}}\n" + r"\toprule" + "\n"
            + heading + r" \\" + "\n" + r"\midrule" + "\n" + "\n".join(rows)
            + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")


def shift_report(comparisons):
    rows = []
    for source in comparisons:
        setting = selected_setting(source)
        if source["subset"] != "full" or setting is None:
            continue
        shift, error = float(source["total_shift_rms"]), float(source["trained_error_rms"])
        rows.append(dict(tag=source["tag"], family=family(source["tag"]), setting=setting,
                         region=source["region"], effective_precision=float(source["effective_precision"]),
                         total_shift_rms=shift, trained_error_rms=error,
                         shift_to_error_rms=shift / error))
    body = []
    for name in FAMILIES:
        for region in ("bulk", "transition", "sharp"):
            values = [value_range([r["shift_to_error_rms"] for r in rows
                                   if r["family"] == name and r["region"] == region and r["setting"] == setting])
                      for setting in SETTINGS]
            body.append(row(NAMES[name] if region == "bulk" else "", region, *values))
    tex = table(
        r"Regional RMS affine posterior-mean shift divided by trained-prediction RMS error, "
        r"with both quantities evaluated on the same full-time region. Entries give across-run ranges "
        r"($8$ Burgers--MLP runs and $3$ in each other family). Baseline is $(\alpha_0,1)$; "
        r"MLE is $(\alpha_0,\widehat v)$. Original and extended evidence columns use best tested grid settings, "
        r"not certified continuous optima. A ratio above one means that the displacement exceeds the "
        r"trained error in regional RMS; it does not determine whether the displaced prediction is more accurate.",
        "tab:r2-relative-shift", "llrrrr",
        r"family & region & baseline & MLE & original evidence & extended evidence", body)
    return rows, tex


def loading_report(production, best):
    rows = []
    for tag in sorted({key[0] for key in best}):
        with np.load(production / f"{tag}_round2_spectral.npz", allow_pickle=False) as a:
            lam = a["eigenvalues"]
            labels = list(a["region_labels"])
            bi, si = labels.index("full/bulk"), labels.index("full/sharp")
            qb, qs = a["mean_mode_energy"][[bi, si]]
            db, ds = a["mean_complement_energy"][[bi, si]]
            choices = [("conditional_mle", float(a["alpha0"]) * float(a["conditional_noise_mle"]))]
            choices += [(name + "_evidence", float(best[(tag, name)]["effective_precision"]))
                        for name in ("original", "extended")]
            for setting, gamma in choices:
                weak = lam <= gamma
                bweak, sweak = float(db + qb[weak].sum()), float(ds + qs[weak].sum())
                bstiff, sstiff = float(qb[~weak].sum()), float(qs[~weak].sum())
                rows.append(dict(tag=tag, family=family(tag), setting=setting, effective_precision=gamma,
                                 weak_mode_count=int(weak.sum()),
                                 bulk_weak_energy=bweak, sharp_weak_energy=sweak,
                                 bulk_stiff_energy=bstiff, sharp_stiff_energy=sstiff,
                                 weak_sharp_bulk_ratio=sweak / bweak,
                                 stiff_sharp_bulk_ratio=sstiff / bstiff))
    body = []
    display = dict(conditional_mle="MLE", original_evidence="original evidence", extended_evidence="extended evidence")
    for name in FAMILIES:
        for setting in SETTINGS[1:]:
            group = [r for r in rows if r["family"] == name and r["setting"] == setting]
            body.append(row(NAMES[name] if setting == "conditional_mle" else "", display[setting],
                            value_range([r["weak_sharp_bulk_ratio"] for r in group]),
                            value_range([r["stiff_sharp_bulk_ratio"] for r in group])))
    tex = table(
        r"Changed-scale sharp-to-bulk mean-square tangent loading ratios on the full-time regions. "
        r"Entries give across-run ranges. At each saved setting, weak loading sums returned modes "
        r"with $\lambda_i\le\gamma$ and the QR complement; stiff loading sums the remaining modes. "
        r"Thus the partition uses $\epsilon=\gamma$, as in Table~\ref{tab:r2-changed-bounds}. "
        r"These are energy ratios, without a square root or covariance weights, and are not uncertainty contrasts. "
        r"The returned weak-mode split does not certify exact rank or weak-mode forward accuracy. "
        r"Evidence settings are best tested grid selections.",
        "tab:r2-scale-loadings", "llrr",
        r"family & setting & $Q_{\mathcal S,\mathrm{sl}}/Q_{\mathcal B,\mathrm{sl}}$ & $Q_{\mathcal S,\mathrm{st}}/Q_{\mathcal B,\mathrm{st}}$", body)
    return rows, tex


def baseline_report(metrics, control):
    chosen = [r for r in metrics if r["centre"] == "trained" and r["subset"] == "full"
              and selected_setting(r) == "baseline"]
    regions = {(r["tag"], r["region"]): r for r in chosen}
    rows = []
    for tag in sorted({r["tag"] for r in chosen}):
        s, b = regions[(tag, "sharp")], regions[(tag, "bulk")]
        rows.append(dict(tag=tag, family=family(tag), R_sigma_rms=float(s["R_sigma_rms"]),
                         R_sigma_mean=float(s["R_sigma_mean"]), R_tar=float(s["R_tar"]), rho=float(s["rho"]),
                         mean_Z_sharp=float(s["mean_Z"]), mean_Z_bulk=float(b["mean_Z"]),
                         coverage68_sharp=float(s["coverage68"]), coverage68_bulk=float(b["coverage68"])))
    control_rows = []
    if control is not None:
        record = control["record"]
        settings = [r["setting"] for r in record["historical_mean_width_comparison"]["per_setting"]]
        for i, setting in enumerate(settings):
            control_rows.append(dict(tag=setting, family="convection_diffusion_control",
                                     **{new: float(record[old]["per_run"][i])
                                        for new, old in (("R_sigma_rms", "R_rms"), ("R_sigma_mean", "R_mean_width"),
                                                         ("R_tar", "R_tar"), ("rho", "ratio_gap"))}))
    body = []
    for name in FAMILIES[1:]:
        group = [r for r in rows if r["family"] == name]
        body.append(row(NAMES[name], *[value_range([r[key] for r in group], 4)
                                      for key in ("R_sigma_rms", "R_sigma_mean", "R_tar", "rho")]))
    if control_rows:
        body.append(row("convection--diffusion control", *[value_range([r[key] for r in control_rows], 4)
                                                          for key in ("R_sigma_rms", "R_sigma_mean", "R_tar", "rho")]))
    body += [r"\midrule", row("family", r"$\meanZ_{\mathcal S}$", r"$\meanZ_{\mathcal B}$",
                              r"$\mathrm{cov}@68_{\mathcal S}$", r"$\mathrm{cov}@68_{\mathcal B}$"), r"\midrule"]
    for name in FAMILIES[1:]:
        group = [r for r in rows if r["family"] == name]
        body.append(row(NAMES[name], *[value_range([r[key] for r in group], 4)
                                      for key in ("mean_Z_sharp", "mean_Z_bulk", "coverage68_sharp", "coverage68_bulk")]))
    tex = table(
        r"Across-run ranges supplementing the baseline means and population standard deviations "
        r"in Tables~\ref{tab:ceiling}--\ref{tab:absolute} for the three-run nonlinear families. "
        r"The upper panel also gives the control's range over its three admitted pulse-width/diffusivity "
        r"settings at one seed, not three seed replicates. All use unit noise, plug-in precision and "
        r"the trained prediction. Ranges are descriptive observed minima and maxima, not confidence intervals.",
        "tab:r2-small-sample-ranges", "lrrrr",
        r"family & $\Rrms$ & $\achratio$ & $\Rtar$ & $\ratiogap$", body)
    return rows, control_rows, tex


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root, help="Paper root containing the frozen production archives.")
    parser.add_argument("--out", type=Path, default=root / "analysis/round2/outputs/reporting_closure")
    parser.add_argument("--tex-out", type=Path, default=root / "generated")
    args = parser.parse_args()
    review = args.source / "analysis/round2/outputs/production_review"
    production = args.source / "analysis/round2/outputs/production"
    args.out.mkdir(parents=True, exist_ok=True)
    args.tex_out.mkdir(parents=True, exist_ok=True)
    control_path = args.out / "source_control_record.json"
    control = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else None
    best = {(r["tag"], r["grid"]): r for r in read_csv(review / "evidence_best.csv")}
    shift, shift_tex = shift_report(read_csv(review / "n3_centre_comparison.csv"))
    loadings, loading_tex = loading_report(production, best)
    baseline, control_rows, baseline_tex = baseline_report(read_csv(review / "regional_scale_metrics.csv"), control)
    for name, rows in (("regional_relative_shift.csv", shift), ("changed_scale_loadings.csv", loadings),
                       ("baseline_per_run.csv", baseline)):
        write_csv(args.out / name, rows)
    if control_rows:
        write_csv(args.out / "control_per_setting.csv", control_rows)
    for name, text in (("round2_relative_shift_table.tex", shift_tex), ("round2_scale_loadings_table.tex", loading_tex),
                       ("round2_small_sample_ranges_table.tex", baseline_tex)):
        (args.tex_out / name).write_text(text, encoding="utf-8")
    summary = dict(source=str(args.source.resolve()), shift_rows=len(shift), loading_rows=len(loadings),
                   baseline_runs=len(baseline), control_settings=len(control_rows),
                   scope="Existing full-time archives only; no new checkpoint evaluation, Jacobian, scale search or exact-root field.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
