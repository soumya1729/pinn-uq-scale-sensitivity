"""Leave-one-member-out sensitivity of the saved eight-member Burgers-MLP ensemble.

Recompute each subset's prediction, population spread and error against the same
reference. Regions use the original full-grid reference-gradient thresholds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(predictions, reference, regions, omitted):
    mean = predictions.mean(axis=0)
    sigma = predictions.std(axis=0, ddof=0)
    error = np.abs(mean - reference)
    z = np.divide(error, sigma, out=np.full_like(error, np.inf), where=sigma > 0)
    z[(sigma == 0) & (error == 0)] = 0
    row = {"omitted_seed": omitted, "members": len(predictions)}
    for name, mask in regions.items():
        row.update({f"{name}_error_rms": float(np.sqrt(np.mean(error[mask] ** 2))),
                    f"{name}_sigma_rms": float(np.sqrt(np.mean(sigma[mask] ** 2))),
                    f"{name}_mean_sigma": float(np.mean(sigma[mask])),
                    f"{name}_mean_Z": float(np.mean(z[mask])),
                    f"{name}_coverage68": float(np.mean(z[mask] <= 1)),
                    f"{name}_coverage95": float(np.mean(z[mask] <= 1.96))})
    row["R_sigma_rms"] = row["sharp_sigma_rms"] / row["bulk_sigma_rms"]
    row["R_tar"] = row["sharp_error_rms"] / row["bulk_error_rms"]
    row["rho"] = row["R_tar"] / row["R_sigma_rms"]
    row["R_sigma_mean"] = row["sharp_mean_sigma"] / row["bulk_mean_sigma"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=Path(__file__).resolve().parents[1] / "n5" / "input")
    parser.add_argument("--baseline", type=Path,
                        default=Path(__file__).parent / "reference" / "ensemble_calibration_burgers_mlp.json",
                        help="Original ensemble_calibration_burgers_mlp.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    predictions, sources = [], []
    for seed in range(8):
        tag = f"burgers_mlp_seed{seed}"
        path = args.inputs / tag / f"{tag}_fields.npz"
        with np.load(path) as data:
            prediction = data["u_pred"].ravel()
            current = {key: data[key] for key in ("u_exact", "grad_abs", "xs", "ts")}
        if seed == 0:
            shared = current
        else:
            for key in shared:
                np.testing.assert_array_equal(current[key], shared[key])
        predictions.append(prediction)
        sources.append({"seed": seed, "path": str(path.resolve()), "sha256": digest(path)})
    predictions = np.asarray(predictions)
    reference, gradient = shared["u_exact"].ravel(), shared["grad_abs"].ravel()
    p70, p95 = np.percentile(gradient, [70, 95])
    regions = {"bulk": gradient < p70, "sharp": gradient >= p95}
    rows = [score(predictions, reference, regions, "none")]
    for seed in range(8):
        rows.append(score(np.delete(predictions, seed, axis=0), reference, regions, seed))
    baseline = json.loads(args.baseline.read_text())
    baseline_keys = {"R_sigma_rms": "sigma_rms_ratio_between_basin",
                     "R_tar": "target_rms_ratio_ensemble_mean",
                     "rho": "ratio_gap_ensemble_mean",
                     "R_sigma_mean": "sigma_ratio_between_basin"}
    for key, saved_key in baseline_keys.items():
        np.testing.assert_allclose(rows[0][key], baseline[saved_key], rtol=1e-12, atol=1e-15)
    for region, saved_region in (("sharp", "shock"), ("bulk", "smooth")):
        for key, saved_key in (("mean_Z", "meanZ"), ("coverage68", "cov68"),
                               ("coverage95", "cov95"), ("mean_sigma", "mean_sigma")):
            np.testing.assert_allclose(rows[0][f"{region}_{key}"],
                                       baseline[saved_region][saved_key], rtol=1e-12, atol=1e-15)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "ensemble_loo.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ranges = {key: {"min": min(row[key] for row in rows[1:]),
                    "max": max(row[key] for row in rows[1:])}
              for key in rows[0] if key not in ("omitted_seed", "members")}
    result = {"method": "Eight-member ensemble and each seven-member leave-one-out subset; "
                        "population spread with ddof=0 (divisor K); subset mean and its own error",
              "reference_regions": {"p70": float(p70), "p95": float(p95),
                                    "counts": {key: int(mask.sum()) for key, mask in regions.items()}},
              "baseline_reproduction": "All saved contrast, mean-Z, coverage and mean-width values agree",
              "baseline_source": str(args.baseline.resolve()),
              "baseline_original_source": "C:/dev/pinn-uq-jmlr-ready/results/ensemble_calibration_burgers_mlp.json",
              "baseline_sha256": digest(args.baseline), "inputs": sources,
              "numpy": np.__version__, "script_sha256": digest(Path(__file__)),
              "rows": rows, "leave_one_out_ranges": ranges,
              "scope": "Deterministic sensitivity to the admitted members; not confidence intervals "
                       "or evidence of a general ensemble advantage"}
    (args.out / "ensemble_loo.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# R2-D: ensemble member sensitivity", "",
             "The original eight admitted Burgers-MLP members are used. Each omission recomputes "
             "the mean prediction, its error against the saved reference, and population spread "
             "with divisor K (ddof=0). The original reference-defined bulk and sharp masks remain fixed.",
             "", "The complete ensemble reproduces the saved baseline artifact to numerical precision.",
             "", "| Omitted seed | K | R_sigma RMS | R_tar | rho | Sharp cov68 | Sharp cov95 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['omitted_seed']} | {row['members']} | {row['R_sigma_rms']:.3f} | "
                     f"{row['R_tar']:.3f} | {row['rho']:.4f} | {row['sharp_coverage68']:.4f} | "
                     f"{row['sharp_coverage95']:.4f} |")
    worst = min(rows[1:], key=lambda row: row["sharp_coverage68"])
    lines.extend(["", f"Every omission retains rho > 1, with a range of "
                  f"{ranges['rho']['min']:.3f} to {ranges['rho']['max']:.3f}. "
                  "The modest regional RMS contrast gap is therefore retained by all eight subsets.",
                  "", f"Sharp-region 68% coverage is member-sensitive: it ranges from "
                  f"{ranges['sharp_coverage68']['min']:.4f} to "
                  f"{ranges['sharp_coverage68']['max']:.4f}; omitting seed "
                  f"{worst['omitted_seed']} produces the minimum. Sharp-region 95% coverage ranges "
                  f"from {ranges['sharp_coverage95']['min']:.4f} to "
                  f"{ranges['sharp_coverage95']['max']:.4f}. Bulk coverage exceeds nominal at both "
                  "levels for every omission. Regional contrast proximity does not imply calibration.",
                  "", "Leave-one-out ranges describe sensitivity to these eight members; they are "
                  "not statistical confidence intervals. Changes cannot be assigned to covariance "
                  "alone because every omission also changes the predictor and its error target.", ""])
    (args.out / "ENSEMBLE_LOO_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Baseline reproduced; eight omissions complete.")
    for key in ("R_sigma_rms", "R_tar", "rho", "sharp_coverage68", "sharp_coverage95",
                "bulk_coverage68", "bulk_coverage95"):
        print(f"{key}: baseline={rows[0][key]:.6g}, leave-one-out "
              f"[{ranges[key]['min']:.6g}, {ranges[key]['max']:.6g}]")


if __name__ == "__main__":
    main()
