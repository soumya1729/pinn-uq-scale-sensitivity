"""Offline review of the fixed production archives; no PINN computation.

Reads originals and writes only to a new review directory. The sampled oracle
uses reference errors, not a practical likelihood estimator. No new gamma
fields, evidence grids, rank cutoffs or checkpoint evaluations are introduced.
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
STRINGS = {"tag", "setting", "centre", "subset", "region"}
METRICS = ("R_sigma_rms", "R_sigma_mean", "R_tar", "rho", "error_rms",
           "sigma_rms", "mean_sigma", "mean_Z", "coverage68", "coverage95")
SUFFIXES = (".json", ".csv", "_fields.npz", "_spectral.npz")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_run(directory, tag):
    """Load one four-file archive; useful independently of the cohort CLI."""
    stem = Path(directory) / f"{tag}_round2"
    summary = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if summary["tag"] != tag:
        raise ValueError(f"Archive tag mismatch: {tag}")
    arrays = []
    for suffix in ("_fields.npz", "_spectral.npz"):
        with np.load(str(stem) + suffix, allow_pickle=False) as archive:
            arrays.append({key: archive[key].copy() for key in archive.files})
    with stem.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = [{key: value if key in STRINGS else
                 int(value) if key in {"count", "gamma_index"} else float(value)
                 for key, value in row.items()} for row in csv.DictReader(handle)]
    return dict(tag=tag, summary=summary, fields=arrays[0], spectral=arrays[1], rows=rows)


def region_masks(run):
    data, summary = run["fields"], run["summary"]
    gradient = data["grad_abs"].ravel()
    p70, p95 = (summary["region_thresholds"][key] for key in ("p70", "p95"))
    masks = {"bulk": gradient < p70, "transition": (gradient >= p70) & (gradient < p95),
             "sharp": gradient >= p95}
    subsets = {"full": masks}
    if summary["config"]["pde"] == "burgers":
        post = np.repeat(data["ts"], len(data["xs"])) >= 1 / np.pi
        subsets["post_formation"] = {name: mask & post for name, mask in masks.items()}
    if any(not mask.any() for group in subsets.values() for mask in group.values()):
        raise ValueError(f"Empty archived region: {run['tag']}")
    return subsets


def metrics(error, sigma, masks, width=1.0):
    """The saved-eight higher-quantile protocol, with ordinary producer metrics."""
    rms_error = {name: np.sqrt(np.mean(error[mask] ** 2)) for name, mask in masks.items()}
    rms_sigma = {name: width * np.sqrt(np.mean(sigma[mask] ** 2)) for name, mask in masks.items()}
    target = rms_error["sharp"] / rms_error["bulk"]
    contrast = rms_sigma["sharp"] / rms_sigma["bulk"]
    mean_contrast = np.mean(sigma[masks["sharp"]]) / np.mean(sigma[masks["bulk"]])
    rows = []
    for region, mask in masks.items():
        z = (np.abs(error[mask]) / sigma[mask]) / width
        rows.append(dict(region=region, count=int(mask.sum()), error_rms=float(rms_error[region]),
            sigma_rms=float(rms_sigma[region]), mean_sigma=float(width * np.mean(sigma[mask])),
            R_tar=float(target), R_sigma_rms=float(contrast), R_sigma_mean=float(mean_contrast),
            rho=float(target / contrast), mean_Z=float(z.mean()), rms_Z=float(np.sqrt(np.mean(z ** 2))),
            coverage68=float(np.mean(z <= 1)), coverage95=float(np.mean(z <= 1.96)),
            width_for68=float(np.quantile(z, .68, method="higher")),
            width_for95=float(np.quantile(z / 1.96, .95, method="higher"))))
    return rows


def row_key(row):
    return tuple(row[key] for key in ("setting", "alpha", "noise_variance", "centre", "subset", "region"))


def validate_run(run, subsets):
    data, summary, spectral = (run[key] for key in ("fields", "summary", "spectral"))
    gammas = data["gammas"]
    np.testing.assert_array_equal(gammas, summary["gammas"])
    if np.any(gammas <= 0) or not np.all(np.diff(gammas) > 0):
        raise ValueError(f"Invalid gamma ordering: {run['tag']}")
    points = len(data["xs"]) * len(data["ts"])
    for key in ("unit_variance", "mean_shift"):
        if data[key].shape != (len(gammas), points) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} field: {run['tag']}")
    for key in ("u_pred", "u_exact", "grad_abs", "complement_shift",
                "tangent_norm_squared", "complement_energy"):
        if data[key].shape != (points,) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} field: {run['tag']}")
    if np.any(data["unit_variance"] <= 0):
        raise ValueError(f"Nonpositive archived variance: {run['tag']}")
    source = {row_key(row): row for row in run["rows"]}
    expected = len(summary["choices"]) * 2 * 3 * len(subsets)
    if len(source) != expected or len(run["rows"]) != expected:
        raise ValueError(f"Missing or duplicate scientific rows: {run['tag']}")
    maximum = 0.0
    for choice in summary["choices"]:
        k = choice["gamma_index"]
        np.testing.assert_allclose(gammas[k], choice["alpha"] * choice["noise_variance"], rtol=1e-14, atol=0)
        sigma = np.sqrt(choice["noise_variance"] * data["unit_variance"][k])
        for centre in ("trained", "affine"):
            prediction = data["u_pred"] + (data["mean_shift"][k] if centre == "affine" else 0)
            error = prediction - data["u_exact"]
            for subset, masks in subsets.items():
                for row in metrics(error, sigma, masks):
                    saved = source[row_key(dict(**choice, centre=centre, subset=subset, region=row["region"]))]
                    mask = masks[row["region"]]
                    row.update(shift_rms=float(np.sqrt(np.mean(data["mean_shift"][k, mask] ** 2))),
                               complement_shift_rms=float(np.sqrt(np.mean(data["complement_shift"][mask] ** 2))))
                    if saved["tag"] != run["tag"] or saved["count"] != row["count"] or saved["gamma_index"] != k:
                        raise ValueError(f"CSV identity/count mismatch: {run['tag']}")
                    np.testing.assert_allclose(saved["effective_precision"], gammas[k], rtol=1e-14, atol=0)
                    for key in (*METRICS, "shift_rms", "complement_shift_rms"):
                        np.testing.assert_allclose(row[key], saved[key], rtol=2e-12, atol=1e-14,
                                                   err_msg=f"{run['tag']} {centre}/{subset}/{key}")
                        maximum = max(maximum, abs(row[key] - saved[key]))
    labels = list(spectral["region_labels"])
    for subset, masks in subsets.items():
        for region, mask in masks.items():
            i = labels.index(f"{subset}/{region}")
            if int(spectral["region_counts"][i]) != int(mask.sum()):
                raise ValueError(f"Spectral region count mismatch: {run['tag']}")
            regional_variance = (spectral["mean_mode_energy"][i] / (spectral["eigenvalues"] + gammas[:, None])).sum(axis=1)
            regional_variance += spectral["mean_complement_energy"][i] / gammas
            np.testing.assert_allclose(regional_variance, data["unit_variance"][:, mask].mean(axis=1), rtol=2e-12, atol=1e-14)
    normal_values = [row["normal_residual_scaled"] for row in summary["normal_equation_checks"]]
    finite_normal_values = [value for value in normal_values if np.isfinite(value)]
    return dict(tag=run["tag"], reproduced_rows=expected, max_metric_absolute_difference=maximum,
                numerical_status=summary["numerical_status"], numerical_review_required=summary["numerical_review_required"],
                max_recorded_finite_normal_backward_error=max(finite_normal_values, default=None),
                nonfinite_normal_backward_error_count=len(normal_values) - len(finite_normal_values),
                complement_review_required=summary["complement_checks"]["review_required"])


def unit_choices(summary):
    choices = {choice["gamma_index"]: choice for choice in summary["choices"]
               if choice["setting"] == "prior_sweep" and choice["noise_variance"] == 1.0}
    if set(choices) != set(range(len(summary["gammas"]))):
        raise ValueError(f"Missing unit-noise gamma aliases: {summary['tag']}")
    return [choices[index] for index in sorted(choices)]


def sampled_summary(curves, oracle):
    nearest, brackets = [], []
    groups = sorted({(row["tag"], row["centre"], row["subset"]) for row in curves})
    for tag, centre, subset in groups:
        points = sorted((row for row in curves if (row["tag"], row["centre"], row["subset"]) == (tag, centre, subset)),
                        key=lambda row: row["effective_precision"])
        closest = min(points, key=lambda row: abs(np.log(row["rho"])))
        selected = {row["region"]: row for row in oracle if row["tag"] == tag and row["centre"] == centre
                    and row["subset"] == subset and row["gamma_index"] == closest["gamma_index"]}
        crossing_count = 0
        for i, row in enumerate(points):
            upper = row if row["rho"] == 1 else points[i + 1] if i + 1 < len(points) else None
            exact = row["rho"] == 1
            if exact or (upper is not None and (row["rho"] - 1) * (upper["rho"] - 1) < 0):
                brackets.append(dict(tag=tag, centre=centre, subset=subset,
                    crossing_kind="sampled_equality" if exact else "between_saved_samples",
                    gamma_lower=row["effective_precision"], gamma_upper=upper["effective_precision"],
                    rho_lower=row["rho"], rho_upper=upper["rho"]))
                crossing_count += 1
        nearest.append(dict(tag=tag, centre=centre, subset=subset, sampled_gamma_count=len(points),
            nearest_gamma=closest["effective_precision"], nearest_rho=closest["rho"], crossing_bracket_count=crossing_count,
            sampled_contrast_min=min(row["R_sigma_rms"] for row in points),
            sampled_contrast_max=max(row["R_sigma_rms"] for row in points),
            bulk_oracle68=selected["bulk"]["coverage68"], bulk_oracle95=selected["bulk"]["coverage95"],
            sharp_oracle68=selected["sharp"]["coverage68"], sharp_oracle95=selected["sharp"]["coverage95"],
            bulk_oracle_mean_Z=selected["bulk"]["mean_Z"], sharp_oracle_mean_Z=selected["sharp"]["mean_Z"]))
    return nearest, brackets


def analyse_run(run):
    data, summary = run["fields"], run["summary"]
    subsets = region_masks(run)
    checks = validate_run(run, subsets)
    curves, oracle, comparisons, complement = [], [], [], []
    common = dict(tag=run["tag"], pde=summary["config"]["pde"], arch=summary["config"]["arch"])
    for choice in unit_choices(summary):
        k = choice["gamma_index"]
        sigma = np.sqrt(data["unit_variance"][k])
        for centre in ("trained", "affine"):
            prediction = data["u_pred"] + (data["mean_shift"][k] if centre == "affine" else 0)
            error = prediction - data["u_exact"]
            for subset, masks in subsets.items():
                rows = metrics(error, sigma, masks)
                base = dict(**common, **choice, effective_precision=float(data["gammas"][k]), centre=centre, subset=subset)
                curves.append(dict(**base, **next(row for row in rows if row["region"] == "sharp")))
                width = next(row["width_for68"] for row in rows if row["region"] == "bulk")
                if width <= 0:
                    raise ValueError(f"Zero oracle width is undefined for standardised-error diagnostics: {run['tag']}")
                scaled = metrics(error, sigma, masks, width=width)
                oracle.extend(dict(**base, width_multiplier=width, calibration_region="bulk", target_coverage68=.68,
                                   quantile_method="higher", **row) for row in scaled)
                if next(row["coverage68"] for row in scaled if row["region"] == "bulk") < .68:
                    raise ValueError(f"Bulk oracle target not reached: {run['tag']}")
    selected = [choice for choice in summary["choices"] if choice["setting"] == "conditional_noise_mle"
                or choice["setting"].startswith("affine_evidence_")
                or (choice["setting"] == "prior_sweep" and choice["alpha_multiplier"] == 1.0)]
    source = {row_key(row): row for row in run["rows"]}
    for subset, masks in subsets.items():
        for region, mask in masks.items():
            signed = data["complement_shift"][mask]
            complement.append(dict(**common, subset=subset, region=region, count=int(mask.sum()),
                qr_complement_mean=float(signed.mean()), qr_complement_rms=float(np.sqrt(np.mean(signed ** 2))),
                qr_complement_min=float(signed.min()), qr_complement_max=float(signed.max()),
                positive_fraction=float(np.mean(signed > 0)), negative_fraction=float(np.mean(signed < 0))))
            for choice in selected:
                k = choice["gamma_index"]
                pair = {centre: source[row_key(dict(**choice, centre=centre, subset=subset, region=region))]
                        for centre in ("trained", "affine")}
                shift = data["mean_shift"][k, mask]
                remaining = shift - signed
                row = dict(**common, **choice, subset=subset, region=region, count=int(mask.sum()),
                    effective_precision=float(data["gammas"][k]), total_shift_mean=float(shift.mean()),
                    total_shift_rms=float(np.sqrt(np.mean(shift ** 2))), remaining_shift_mean=float(remaining.mean()),
                    remaining_shift_rms=float(np.sqrt(np.mean(remaining ** 2))))
                for key in ("error_rms", "R_tar", "rho", "mean_Z", "coverage68", "coverage95"):
                    row.update({f"trained_{key}": pair["trained"][key], f"affine_{key}": pair["affine"][key],
                                f"affine_minus_trained_{key}": pair["affine"][key] - pair["trained"][key]})
                comparisons.append(row)
    nearest, brackets = sampled_summary(curves, oracle)
    limits = [dict(**common, **row) for row in summary["spectral_limits"]]
    return dict(ordinary=run["rows"], curves=curves, oracle=oracle, nearest=nearest, brackets=brackets,
                n3=comparisons, complement=complement, limits=limits, checks=[checks])


def plot_curves(curves, outdir):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    groups = sorted({(row["pde"], row["arch"], row["subset"]) for row in curves})
    for pde, arch, subset in groups:
        group = [row for row in curves if (row["pde"], row["arch"], row["subset"]) == (pde, arch, subset)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        tags = sorted({row["tag"] for row in group})
        colours = plt.get_cmap("tab10")
        for i, tag in enumerate(tags):
            for centre, axis in (("trained", axes[1]), ("affine", axes[2])):
                points = sorted((row for row in group if row["tag"] == tag and row["centre"] == centre),
                                key=lambda row: row["effective_precision"])
                gamma = [row["effective_precision"] for row in points]
                axis.loglog(gamma, [row["rho"] for row in points], ".-", color=colours(i), label=tag.split("_seed")[-1])
                if centre == "trained":
                    axes[0].loglog(gamma, [row["R_sigma_rms"] for row in points], ".-", color=colours(i), label=tag.split("_seed")[-1])
                    axes[0].axhline(points[0]["R_tar"], color=colours(i), linestyle="--", linewidth=.8, alpha=.65)
        for axis in axes:
            axis.set_xlabel(r"effective precision $\gamma=\alpha v$")
            axis.grid(True, which="major", alpha=.2)
        for axis, centre in zip(axes[1:], ("trained", "affine")):
            axis.axhline(1, color="black", linestyle="--", linewidth=.8)
            axis.set_title(f"{centre.capitalize()} centre")
            axis.set_ylabel(r"$\rho=R_{\rm tar}/R_\sigma^{\rm rms}$")
        axes[0].set_title("Covariance contrast and trained targets")
        axes[0].set_ylabel(r"$R_\sigma^{\rm rms}$; dashed $R_{\rm tar}$")
        axes[0].legend(title="Seed / fixture", fontsize=7, ncol=2)
        fig.suptitle(f"{pde.replace('_', ' ').title()} / {arch.upper()} / {subset.replace('_', ' ')} ({len(tags)} archives)")
        fig.text(.5, .015, "Saved samples; connecting lines guide the eye. Limits are tabulated separately with QR-complement qualifications.\n"
                 "No exact roots, monotonicity, whole-kernel weak-limit certificate or global endpoint envelope is asserted.", ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .075, 1, .94))
        stem = Path(outdir) / f"contrast_and_gap_{pde}_{arch}_{subset}"
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=160)
        plt.close(fig)


def inventory(directory, entries):
    expected = [entry["tag"] for entry in entries]
    present = sorted(path.name.removesuffix("_round2.json") for path in Path(directory).glob("*_round2.json"))
    missing = [f"{tag}_round2{suffix}" for tag in expected for suffix in SUFFIXES
               if not (Path(directory) / f"{tag}_round2{suffix}").is_file()]
    return dict(expected_tags=expected, json_tags=present, unexpected_tags=sorted(set(present) - set(expected)),
                missing_files=missing, complete_cohort_inventory=not missing,
                execution_provenance_review="Executed scripts, environment, wrappers, logs, receipts and accounting require separate review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=HERE / "outputs/production")
    parser.add_argument("--cohort", type=Path, default=HERE / "cohort.json")
    parser.add_argument("--out", type=Path, default=HERE / "outputs/production_review")
    args = parser.parse_args()
    entries = json.loads(args.cohort.read_text(encoding="utf-8"))["cohort"]
    record = inventory(args.production, entries)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "archive_inventory.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["missing_files"]:
        raise ValueError(f"Cohort incomplete: {len(record['missing_files'])} archive files missing; see archive_inventory.json")
    runs = [read_run(args.production, entry["tag"]) for entry in entries]
    for entry, run in zip(entries, runs):
        expected = {Path(item["path"]).name: item["sha256"] for item in entry.get("files", [])}
        if expected and run["summary"]["completion_signature"]["input_hashes"] != expected:
            raise ValueError(f"Input signature differs from cohort manifest: {entry['tag']}")
    combined = {key: [] for key in ("ordinary", "curves", "oracle", "nearest", "brackets", "n3", "complement", "limits", "checks")}
    for run in runs:
        for key, rows in analyse_run(run).items():
            combined[key].extend(rows)
    filenames = dict(ordinary="regional_scale_metrics", curves="uncertainty_contrast_curves", oracle="sampled_oracle_coverage",
                     nearest="nearest_sampled_agreement", brackets="sampled_crossing_brackets", n3="n3_centre_comparison",
                     complement="qr_complement_summary", limits="spectral_limits", checks="archive_metric_checks")
    for key, filename in filenames.items():
        write_csv(args.out / f"{filename}.csv", combined[key])
    plot_curves(combined["curves"], args.out)
    try:
        from review_evidence import review_evidence
    except ModuleNotFoundError as exc:
        if exc.name != "review_evidence":
            raise
        evidence = {"status": "Evidence helper unavailable; evidence review remains pending."}
    else:
        evidence = review_evidence(runs, args.out)
    result = dict(cohort_archive_count=len(runs), complete_cohort_inventory=True, checks=combined["checks"], evidence=evidence,
        oracle=dict(target_bulk_coverage68=.68, quantile_method="higher", interval95_multiplier=1.96,
                    reference_informed=True, centres=["trained", "affine"], sampled_fields_only=True),
        nearest_sample_rule="Minimum absolute log(rho); lowest saved gamma breaks exact ties.",
        numerical_scope="Metric reproduction and recorded consistency checks do not certify weak-mode forward accuracy or evidence-model adequacy.")
    (args.out / "review_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    note = ["# Offline production archive review", "", f"Reviewed {len(runs)} archives from the supplied cohort manifest.", "",
        "The ordinary CSV preserves every scale choice, both centres and all archived subsets. The oracle uses each distinct saved unit-noise gamma once, with a separate bulk-68% higher empirical quantile for each centre and subset. It uses reference errors and is descriptive; 95% intervals use 1.96 standard deviations.", "",
        "Nearest samples minimise absolute log(rho); crossing brackets identify adjacent saved values, without interpolation or exact-root coverage. Contrast geometry is shared by the two centres, while the affine error target can change with gamma. Burgers post-formation masks intersect the unchanged global reference masks with t >= 1/pi; Allen-Cahn has no such subset.", "",
        "Strong-gamma tangent-norm ratios and QR-complement ratios retain the producer's qualifications. The QR complement need not be the entire exact kernel; small positive singular values are not declared zero. Endpoint ratios are not global bounds. Signed complement and remaining shifts may cancel, so their RMS values are not additive.", "",
        "The affine centre is the linearised predictive mean, not a nonlinear evaluation at shifted parameters. Evidence sensitivity fields use the original Jacobian. Numerical PASS labels retain their producer scope; neither they nor CSV reproduction establish weak-mode forward accuracy or likelihood adequacy. The existing stress-toy REVIEW_REQUIRED result remains separate.", "",
        "Review executed scripts, versions, actual/ad hoc wrappers, logs, receipts and accounting before scientific integration. These outputs do not mark any work package complete automatically."]
    (args.out / "PRODUCTION_REVIEW.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print(f"Reviewed {len(runs)} cohort archives; derived outputs: {args.out}")


if __name__ == "__main__":
    main()
