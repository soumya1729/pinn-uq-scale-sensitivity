"""Review the downloaded theta-centred experiment; no Jacobians or new fits.

Recompute CSV summaries from the saved fields, check sampled nonlinear outputs
with a NumPy forward implementation, and collect paired/family results.
"""
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review"
METRICS = ("error_rms", "sigma_rms", "R_sigma_rms", "R_tar", "rho",
           "coverage68", "coverage95", "mean_Z", "signed_mean_Z",
           "median_Z", "q99_Z", "linearisation_remainder_rms",
           "linearisation_remainder_over_sigma_rms",
           "standardised_linearisation_remainder_rms")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shifted_tree(tree, delta, cursor):
    # JAX's default dictionary flattening sorts keys; sequence order is retained.
    if isinstance(tree, dict):
        return {k: shifted_tree(tree[k], delta, cursor) for k in sorted(tree)}
    if isinstance(tree, (list, tuple)):
        return type(tree)(shifted_tree(x, delta, cursor) for x in tree)
    a = np.asarray(tree)
    start = cursor[0]
    cursor[0] += a.size
    return a + delta[start:cursor[0]].reshape(a.shape)


def numpy_forward(params, x, t, fourier):
    def mlp(layers, h):
        for layer in layers[:-1]:
            h = np.tanh(h @ np.asarray(layer["w"]) + np.asarray(layer["b"]))
        return h @ np.asarray(layers[-1]["w"]) + np.asarray(layers[-1]["b"])

    def features(z, key):
        if fourier is None:
            return z[:, None]
        phase = 2 * np.pi * z[:, None] * fourier[key].ravel()
        return np.concatenate((np.cos(phase), np.sin(phase)), axis=1)

    hx, ht = features(x, "B_X"), features(t, "B_T")
    a = [mlp(net, hx) for net in params["spatial"]]
    a += [mlp(net, ht) for net in params["temporal"]]
    return mlp(params["fusion"], np.concatenate(a, axis=1))[:, 0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    paired, roots, checks, displacement = [], [], [], []
    summaries = []
    for run in manifest["runs"]:
        tag = run["tag"]
        stem = ROOT / "outputs/production" / f"{tag}_theta"
        report = json.loads(stem.with_suffix(".json").read_text())
        with stem.with_suffix(".csv").open() as f:
            original = list(csv.DictReader(f))
        with np.load(str(stem) + "_fields.npz") as z:
            data = {k: z[k] for k in z.files}
        np.testing.assert_array_equal(data["gammas"], report["gammas"])
        gradient = data["grad_abs"]
        p70, p95 = np.percentile(gradient, (70, 95))
        masks = {"bulk": gradient < p70,
                 "sharp": gradient >= p95,
                 "transition": (gradient >= p70) & (gradient < p95)}
        times = np.repeat(data["ts"], len(data["xs"]))
        family = tag.rsplit("_seed", 1)[0]
        by_group = defaultdict(list)
        for row in original:
            by_group[(row["setting"], row["centre"], row["subset"], row["calibration"])].append(row)
        choices = {c["setting"]: c for c in report["choices"]}
        maximum_relative_defect, maximum_absolute_defect = 0., 0.
        for (setting, centre, subset, calibration), region_rows in by_group.items():
            choice = choices[setting]
            k = choice["gamma_index"]
            prediction = {"trained": data["u_pred"],
                          "theta_affine": data["u_pred"] + data["theta_shift"][k],
                          "theta_nonlinear": data["theta_nonlinear"][k],
                          "zero_affine": data["u_pred"] + data["zero_shift"][k]}[centre]
            error = prediction - data["u_exact"]
            selected = {r: mask & (times >= 1/np.pi) if subset == "post_formation" else mask
                        for r, mask in masks.items()}
            unit_sigma = np.sqrt(data["unit_variance"][k])
            multiplier = np.sqrt(choice["noise_variance"])
            if calibration == "bulk_68_oracle":
                multiplier = np.quantile(np.abs(error[selected["bulk"]])/unit_sigma[selected["bulk"]],
                                         .68, method="higher")
            sigma = unit_sigma * multiplier
            e2 = {r: np.mean(error[m]**2) for r, m in selected.items()}
            q = {r: np.mean(sigma[m]**2) for r, m in selected.items()}
            target, contrast = np.sqrt(e2["sharp"]/e2["bulk"]), np.sqrt(q["sharp"]/q["bulk"])
            all_mask = np.logical_or.reduce(list(selected.values()))
            full_error = float(np.sqrt(np.mean(error[all_mask]**2)))
            remainder = data["theta_nonlinear"][k] - data["u_pred"] - data["theta_shift"][k]
            for row in region_rows:
                region = row["region"]
                m = selected[region]
                signed = error[m]/sigma[m]
                z = np.abs(signed)
                remainder_rms = float(np.sqrt(np.mean(remainder[m]**2)))
                calculated = dict(error_rms=float(np.sqrt(e2[region])), sigma_rms=float(np.sqrt(q[region])),
                    R_sigma_rms=float(contrast), R_tar=float(target), rho=float(target/contrast),
                    coverage68=float(np.mean(z <= 1)), coverage95=float(np.mean(z <= 1.96)),
                    mean_Z=float(np.mean(z)), signed_mean_Z=float(np.mean(signed)),
                    median_Z=float(np.median(z)), q99_Z=float(np.quantile(z, .99)),
                    linearisation_remainder_rms=remainder_rms,
                    linearisation_remainder_over_sigma_rms=remainder_rms/np.sqrt(q[region]),
                    standardised_linearisation_remainder_rms=float(np.sqrt(np.mean((remainder[m]/sigma[m])**2))))
                assert int(m.sum()) == int(row["count"])
                for key, value in calculated.items():
                    expected = float(row[key])
                    np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12,
                                               err_msg=f"{tag} {setting} {centre} {subset} {region} {key}")
                    maximum_absolute_defect = max(maximum_absolute_defect, abs(value-expected))
                    maximum_relative_defect = max(maximum_relative_defect,
                        abs(value-expected)/max(abs(expected), 1e-12))
                record = dict(tag=tag, family=family, setting=setting, centre=centre,
                    subset=subset, calibration=calibration, region=region,
                    gamma=choice["gamma"], count=int(m.sum()), full_grid_error_rms=full_error,
                    delta_theta_relative_norm=report["delta_theta_relative_norm"][k], **calculated)
                if subset == "full" and calibration == "selected":
                    paired.append(record)
                if choice["selection"] == "reference_informed_trained_root":
                    roots.append(record)
            if subset == "full" and calibration == "selected":
                summaries.append(dict(tag=tag, family=family, setting=setting, centre=centre,
                    full_grid_error_rms=full_error, rho=float(target/contrast)))

        # An independent NumPy network evaluation checks stored shifted predictions.
        checkpoint = ROOT / "inputs/checkpoints" / tag
        with (checkpoint / f"{tag}_params.pkl").open("rb") as f:
            params = pickle.load(f)
        fourier_path = checkpoint / f"{tag}_fourier_B.npz"
        fourier = None
        if fourier_path.exists():
            with np.load(fourier_path) as z:
                fourier = {k: z[k] for k in ("B_X", "B_T")}
        indices = np.linspace(0, len(gradient)-1, 9, dtype=int)
        x, t = np.meshgrid(data["xs"], data["ts"])
        bx, bt = x.ravel()[indices], t.ravel()[indices]
        baseline_prediction = numpy_forward(params, bx, bt, fourier)
        np.testing.assert_allclose(baseline_prediction, data["u_pred"][indices], rtol=1e-9, atol=1e-10)
        forward_defect = 0.
        with np.load(str(stem) + "_shifts.npz") as shifts:
            for k, delta in enumerate(shifts["delta_likelihood"]):
                cursor = [0]
                updated = shifted_tree(params, delta, cursor)
                assert cursor[0] == len(delta)
                prediction = numpy_forward(updated, bx, bt, fourier)
                expected = data["theta_nonlinear"][k, indices]
                np.testing.assert_allclose(prediction, expected, rtol=1e-9, atol=1e-10)
                forward_defect = max(forward_defect, float(np.max(np.abs(prediction-expected))))
        checks.append(dict(tag=tag, csv_rows=len(original), csv_metrics_per_row=len(METRICS),
            max_csv_absolute_defect=maximum_absolute_defect,
            max_csv_relative_defect_with_1e_12_floor=maximum_relative_defect,
            numpy_forward_samples_per_gamma=len(indices), numpy_forward_max_absolute_defect=forward_defect))
        for c in report["choices"]:
            displacement.append(dict(tag=tag, family=family, setting=c["setting"], gamma=c["gamma"],
                relative_likelihood_shift=report["delta_theta_relative_norm"][c["gamma_index"]],
                relative_zero_prior_shift=report["delta_zero_relative_norm"][c["gamma_index"]],
                relative_complement_shift=report["delta_complement_relative_norm"]))
        print(f"{tag}: CSV and sampled NumPy nonlinear predictions agree.", flush=True)

    family_rows = []
    groups = defaultdict(list)
    for row in paired:
        for family in (row["family"], "cohort"):
            groups[(family, row["setting"], row["centre"], row["region"])].append(row)
    for (family, setting, centre, region), group in sorted(groups.items()):
        for metric in (*METRICS, "full_grid_error_rms", "delta_theta_relative_norm"):
            values = np.array([r[metric] for r in group])
            family_rows.append(dict(family=family, setting=setting, centre=centre, region=region,
                metric=metric, n=len(values), minimum=float(values.min()),
                median=float(np.median(values)), maximum=float(values.max())))
    write_csv(OUT / "paired_centres.csv", paired)
    write_csv(OUT / "family_summary.csv", family_rows)
    write_csv(OUT / "root_comparisons.csv", roots)
    write_csv(OUT / "displacement_summary.csv", displacement)
    write_csv(OUT / "full_grid_errors.csv", summaries)
    root_checks = []
    with (ROOT / "inputs/refined_roots.csv").open() as f:
        for bracket in csv.DictReader(f):
            actual = next(r for r in roots if r["tag"] == bracket["tag"]
                and r["centre"] == "trained" and r["subset"] == "full"
                and r["calibration"] == "bulk_68_oracle" and r["region"] == "sharp")
            assert abs(actual["rho"] - 1) < 1e-10
            for coverage in ("coverage68", "coverage95"):
                lower = float(bracket[f"sharp_{coverage}_lower"])
                upper = float(bracket[f"sharp_{coverage}_upper"])
                assert lower <= actual[coverage] <= upper
            root_checks.append(dict(tag=actual["tag"], rho=actual["rho"],
                coverage68=actual["coverage68"], coverage95=actual["coverage95"],
                inside_saved_envelopes=True))
    (OUT / "field_reproduction.json").write_text(json.dumps(dict(
        numerical_status="PASS", checkpoints=len(checks), checks=checks, root_checks=root_checks,
        scope="CSV reproduction and deterministic sampled forward checks, not posterior calibration or weak-mode accuracy."), indent=2)+"\n")
    print(f"Reviewed {len(checks)} checkpoints; outputs in {OUT}")


if __name__ == "__main__":
    main()
