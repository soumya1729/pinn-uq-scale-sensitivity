"""Round-three summaries and analytic envelopes from saved round-two archives.

No checkpoints, Jacobians, factorisations or network evaluations are used. Bounds
describe the returned floating-point affine model, not a rounding certificate for
the original Jacobian. The oracle is reference-informed, with the existing higher
bulk-68% empirical quantile. Refined roots remain in observed crossing brackets.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def masks(fields, summary):
    grad = fields["grad_abs"]
    p70, p95 = (summary["region_thresholds"][k] for k in ("p70", "p95"))
    regions = dict(bulk=grad < p70, transition=(grad >= p70) & (grad < p95), sharp=grad >= p95)
    result = {"full": regions}
    if summary["config"]["pde"] == "burgers":
        post = np.repeat(fields["ts"], len(fields["xs"])) >= 1 / np.pi
        result["post_formation"] = {r: mask & post for r, mask in regions.items()}
    return result


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def H(spectral, label, gamma):
    i = list(spectral["region_labels"]).index(label)
    return float(spectral["mean_complement_energy"][i] / gamma + np.sum(
        spectral["mean_mode_energy"][i] / (spectral["eigenvalues"] + gamma)))


def variance_envelope(q_low, q_high, low, high, gamma):
    """Concave gamma*q and 1/q yield lower and upper positive-modal bounds."""
    t = (gamma - low) / (high - low)
    lower = ((1 - t) * low * q_low + t * high * q_high) / gamma
    upper = 1 / ((1 - t) / q_low + t / q_high)
    return lower, upper


def toy_envelope_check():
    rng = np.random.default_rng(3103)
    eigenvalues = np.array([0, 1e-9, .01, 1., 100.])
    energies = rng.uniform(.01, 1, (80, len(eigenvalues)))
    q = lambda gamma: np.sum(energies / (eigenvalues + gamma), axis=1)
    low, high = .003, .3
    error = abs(rng.normal(size=80))
    bulk, sharp = np.arange(40), np.arange(40, 80)
    minimum = np.inf
    for gamma in np.geomspace(low, high, 51)[1:-1]:
        lower, upper = variance_envelope(q(low), q(high), low, high, gamma)
        exact = q(gamma)
        minimum = min(minimum, float(np.min((exact - lower) / exact)),
                      float(np.min((upper - exact) / exact)))
        assert np.all(lower <= exact * (1 + 1e-13))
        assert np.all(upper >= exact * (1 - 1e-13))
        width_low = np.quantile(error[bulk] / np.sqrt(upper[bulk]), .68, method="higher")
        width_high = np.quantile(error[bulk] / np.sqrt(lower[bulk]), .68, method="higher")
        width_exact = np.quantile(error[bulk] / np.sqrt(exact[bulk]), .68, method="higher")
        for threshold in (1, 1.96):
            coverage_low = np.mean(error[sharp] <= threshold * width_low * np.sqrt(lower[sharp]))
            coverage_high = np.mean(error[sharp] <= threshold * width_high * np.sqrt(upper[sharp]))
            coverage_exact = np.mean(error[sharp] <= threshold * width_exact * np.sqrt(exact[sharp]))
            assert coverage_low <= coverage_exact <= coverage_high
    return {"seed": 3103, "points": 80, "modes_including_null": 5,
            "interior_precisions": 49, "minimum_relative_margin": minimum,
            "bulk_calibrated_coverage_envelopes_checked": True}


def decomposition(tag, summary, fields, spectral, subsets):
    s, lam = spectral["singular"], spectral["eigenvalues"]
    theta, residual = spectral["theta_modes"], spectral["residual_modes"]
    complement_norm = summary["complement_checks"]["expected_coefficient_norm"]
    theta_norm = float(np.sqrt(spectral["parameter_count"] / summary["alpha0"]))
    all_norms = []
    for check in summary["normal_equation_checks"]:
        gamma = check["gamma"]
        likelihood = -s * residual / (lam + gamma)
        prior = -gamma * theta / (lam + gamma)
        total = float(np.sqrt(np.sum((likelihood + prior) ** 2) + complement_norm ** 2))
        np.testing.assert_allclose(total, check["delta_norm"], rtol=1e-12, atol=0)
        all_norms.append(dict(tag=tag, gamma=gamma, theta_norm=theta_norm,
            likelihood_norm=float(np.linalg.norm(likelihood)),
            prior_norm=float(np.sqrt(prior @ prior + complement_norm ** 2)),
            cross_term=float(2 * likelihood @ prior), total_norm=total,
            saved_total_norm=check["delta_norm"], total_relative_theta=total / theta_norm,
            relative_norm_reconstruction_error=abs(total / check["delta_norm"] - 1)))
    alpha = summary["alpha0"]
    k = int(np.argmin(abs(fields["gammas"] / alpha - 1)))
    np.testing.assert_allclose(fields["gammas"][k], alpha, rtol=1e-14)
    likelihood = -s * residual / (lam + alpha)
    c = float(np.sqrt(np.sum(lam * residual ** 2 / (lam + alpha))))
    mle_gamma = alpha * summary["conditional_noise_mle"]
    c_mle = float(np.sqrt(np.sum(lam * residual ** 2 / (lam + mle_gamma))
                          / summary["conditional_noise_mle"]))
    error = fields["u_pred"] - fields["u_exact"]
    sigma = np.sqrt(fields["unit_variance"][k])
    labels = list(spectral["region_labels"])
    regional = []
    for subset, group in subsets.items():
        sigma_rms, error_rms, bounds = {}, {}, {}
        for region, mask in group.items():
            i = labels.index(f"{subset}/{region}")
            sigma_rms[region], error_rms[region] = rms(sigma[mask]), rms(error[mask])
            modal_bound = float(np.sum(abs(likelihood) * np.sqrt(spectral["mean_mode_energy"][i])))
            bounds[region] = min(c * sigma_rms[region], modal_bound)
            row = dict(tag=tag, subset=subset, region=region, c_baseline=c, c_noise_mle=c_mle,
                error_rms=error_rms[region], sigma_rms=sigma_rms[region],
                covariance_metric_rms_bound=c * sigma_rms[region], modal_rms_bound=modal_bound,
                theta_shift_rms_bound=bounds[region],
                theta_error_rms_lower=max(0, error_rms[region] - bounds[region]),
                theta_error_rms_upper=error_rms[region] + bounds[region])
            z = abs(error[mask]) / sigma[mask]
            for name, threshold in (("68", 1.), ("95", 1.96)):
                row[f"coverage{name}_lower"] = float(np.mean(z <= max(0, threshold - c)))
                row[f"coverage{name}_upper"] = float(np.mean(z <= threshold + c))
            regional.append(row)
        r_sigma = sigma_rms["sharp"] / sigma_rms["bulk"]
        rho_lower = max(0, error_rms["sharp"] - bounds["sharp"]) / (
            error_rms["bulk"] + bounds["bulk"]) / r_sigma
        for row in regional:
            if row["subset"] == subset:
                row["theta_rho_lower"] = rho_lower
                row["theta_deficit_certified"] = rho_lower > 1
    group = subsets["full"]
    counts = np.array([group[r].sum() for r in group])
    indices = [labels.index(f"full/{r}") for r in group]
    full_energies = np.average(spectral["mean_mode_energy"][indices], axis=0, weights=counts)
    full_modal_bound = float(np.sum(abs(likelihood) * np.sqrt(full_energies)))
    full_bound = min(c * rms(sigma), full_modal_bound)
    trained_error, affine_error = rms(error), rms(error + fields["mean_shift"][k])
    baseline = next(row for row in all_norms if abs(row["gamma"] / alpha - 1) < 1e-14)
    baseline.update(qr_complement_relative_theta=complement_norm / theta_norm,
        likelihood_relative_theta=baseline["likelihood_norm"] / theta_norm,
        c_baseline=c, c_noise_mle=c_mle, trained_error_rms=trained_error,
        affine_error_rms=affine_error, full_shift_rms_bound=full_bound,
        prior_only_error_rms_lower=max(0, affine_error - full_bound),
        affine_worsens=affine_error > trained_error,
        prior_only_worsening_certified=affine_error - full_bound > trained_error,
        theta_rho_lower=next(row["theta_rho_lower"] for row in regional if row["subset"] == "full"))
    return all_norms, regional, baseline


def oracle_statistics(tag, fields, subsets):
    rows, curves = [], []
    for centre in ("trained", "affine"):
        for subset, group in subsets.items():
            error = fields["u_pred"] - fields["u_exact"]
            if centre == "affine":
                errors = error[None, :] + fields["mean_shift"]
            else:
                errors = np.broadcast_to(error, fields["unit_variance"].shape)
            target = np.sqrt(np.mean(errors[:, group["sharp"]] ** 2, axis=1)
                             / np.mean(errors[:, group["bulk"]] ** 2, axis=1))
            contrast = np.sqrt(np.mean(fields["unit_variance"][:, group["sharp"]], axis=1)
                               / np.mean(fields["unit_variance"][:, group["bulk"]], axis=1))
            rho = target / contrast
            k = int(np.argmin(abs(np.log(rho))))
            sigma = np.sqrt(fields["unit_variance"][k])
            error = errors[k]
            width = float(np.quantile(abs(error[group["bulk"]]) / sigma[group["bulk"]],
                                      .68, method="higher"))
            sigma = sigma * width
            for i, gamma in enumerate(fields["gammas"]):
                curves.append(dict(tag=tag, centre=centre, subset=subset, gamma=float(gamma),
                                   R_tar=float(target[i]), R_sigma=float(contrast[i]), rho=float(rho[i])))
            squared_weighted = {}
            for region, mask in group.items():
                signed = error[mask] / sigma[mask]
                absolute = abs(signed)
                energy = error[mask] ** 2
                n_top = int(np.ceil(.01 * mask.sum()))
                top_energy = np.partition(energy, len(energy) - n_top)[-n_top:].sum() / energy.sum()
                squared_weighted[region] = float(np.sum(sigma[mask] ** 2 * signed ** 2)
                                                / np.sum(sigma[mask] ** 2))
                row = dict(tag=tag, centre=centre, subset=subset, region=region, count=int(mask.sum()),
                    gamma_index=k, gamma=float(fields["gammas"][k]), nearest_rho=float(rho[k]),
                    bulk_width_multiplier=width, signed_standardized_mean=float(signed.mean()),
                    mean_Z=float(absolute.mean()), rms_Z=rms(signed), median_Z=float(np.median(absolute)),
                    coverage68=float(np.mean(absolute <= 1)), coverage95=float(np.mean(absolute <= 1.96)),
                    top1_error_energy_fraction=float(top_energy),
                    spearman_sigma_abs_error=float(spearmanr(sigma[mask], abs(error[mask])).statistic),
                    weighted_second_moment=squared_weighted[region])
                for q in (.25, .75, .9, .95, .99):
                    row[f"q{int(100*q)}_Z"] = float(np.quantile(absolute, q))
                rows.append(row)
            np.testing.assert_allclose(rho[k] ** 2, squared_weighted["sharp"] / squared_weighted["bulk"],
                                       rtol=1e-12)
    return rows, curves


def root_bounds(tag, fields, spectral, subsets, curves):
    selected = [row for row in curves if row["centre"] == "trained" and row["subset"] == "full"]
    target = selected[0]["R_tar"]
    rho = lambda gamma: target / np.sqrt(H(spectral, "full/sharp", gamma) / H(spectral, "full/bulk", gamma))
    result = []
    for index in range(len(selected) - 1):
        if (selected[index]["rho"] - 1) * (selected[index+1]["rho"] - 1) >= 0:
            continue
        low, high = selected[index]["gamma"], selected[index+1]["gamma"]
        a, b = low, high
        f_a = rho(a) - 1
        iterations = 0
        while b / a - 1 > 1e-12:
            mid = np.sqrt(a * b)
            f_mid = rho(mid) - 1
            if f_a * f_mid <= 0:
                b = mid
            else:
                a, f_a = mid, f_mid
            iterations += 1
        assert (rho(a) - 1) * (rho(b) - 1) <= 0
        q_low, q_high = fields["unit_variance"][[index, index+1]]
        lower, _ = variance_envelope(q_low, q_high, low, high, b)
        _, upper = variance_envelope(q_low, q_high, low, high, a)
        error = abs(fields["u_pred"] - fields["u_exact"])
        bulk, sharp = subsets["full"]["bulk"], subsets["full"]["sharp"]
        width_min = float(np.quantile(error[bulk] / np.sqrt(upper[bulk]), .68, method="higher"))
        width_max = float(np.quantile(error[bulk] / np.sqrt(lower[bulk]), .68, method="higher"))
        row = dict(tag=tag, saved_bracket_low=low, saved_bracket_high=high,
            root_low=a, root_high=b, root_gamma=np.sqrt(a*b), rho_low=rho(a), rho_high=rho(b),
            relative_root_interval=b/a-1, iterations=iterations,
            bulk_width_lower=width_min, bulk_width_upper=width_max)
        for name, threshold in (("68", 1.), ("95", 1.96)):
            row[f"sharp_coverage{name}_lower"] = float(np.mean(
                error[sharp] <= threshold * width_min * np.sqrt(lower[sharp])))
            row[f"sharp_coverage{name}_upper"] = float(np.mean(
                error[sharp] <= threshold * width_max * np.sqrt(upper[sharp])))
        result.append(row)
    return result


def strong_precision(tag, spectral, subsets):
    lam = spectral["eigenvalues"]
    lambda_max = float(lam.max())
    labels = list(spectral["region_labels"])
    rows = []
    for subset in subsets:
        indices = [labels.index(f"{subset}/{r}") for r in ("sharp", "bulk")]
        energies = spectral["mean_mode_energy"][indices]
        totals = energies.sum(axis=1) + spectral["mean_complement_energy"][indices]
        mu = (energies @ lam) / totals
        limit = float(np.sqrt(totals[0] / totals[1]))
        for multiple in (1, 10, 100, 1000):
            gamma = multiple * lambda_max
            exact = float(np.sqrt(H(spectral, f"{subset}/sharp", gamma)
                                  / H(spectral, f"{subset}/bulk", gamma)))
            squared_approx = totals[0] / totals[1] * (1 - (mu[0] - mu[1]) / gamma)
            rows.append(dict(tag=tag, subset=subset, lambda_max=lambda_max,
                gamma_over_lambda_max=multiple, gamma=gamma, R_sigma=exact, strong_limit=limit,
                relative_contrast_limit_error=abs(exact / limit - 1),
                mu_sharp=float(mu[0]), mu_bulk=float(mu[1]),
                R_sigma_squared_first_order=float(squared_approx),
                relative_first_order_error=abs(squared_approx / exact**2 - 1)))
    return rows


def archive_checks(tag, fields, spectral, subsets):
    maximum = 0.
    for subset, group in subsets.items():
        for region, mask in group.items():
            modal = np.array([H(spectral, f"{subset}/{region}", gamma) for gamma in fields["gammas"]])
            saved = np.mean(fields["unit_variance"][:, mask], axis=1)
            np.testing.assert_allclose(modal, saved, rtol=2e-12)
            maximum = max(maximum, float(np.max(abs(modal / saved - 1))))
    envelope_defect = 0.
    for k in range(1, len(fields["gammas"]) - 1):
        low, gamma, high = fields["gammas"][[k-1, k, k+1]]
        lower, upper = variance_envelope(fields["unit_variance"][k-1], fields["unit_variance"][k+1],
                                         low, high, gamma)
        saved = fields["unit_variance"][k]
        envelope_defect = max(envelope_defect, float(np.max(np.maximum(lower/saved-1, 0))),
                             float(np.max(np.maximum(1-upper/saved, 0))))
    assert envelope_defect < 2e-12, (tag, envelope_defect)
    return dict(tag=tag, regional_variance_max_relative_defect=maximum,
                stored_intermediate_envelope_max_relative_violation=envelope_defect,
                checked_intermediate_fields=len(fields["gammas"])-2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    canonical = Path(__file__).resolve().parent.parent / "round2"
    if not canonical.is_dir():
        canonical = Path(r"C:\dev\myshortened\pinn_uq\analysis\round2")
    parser.add_argument("--production-dir", type=Path, default=canonical / "outputs/production")
    parser.add_argument("--cohort", type=Path, default=canonical / "cohort.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tags = [r["tag"] for r in json.loads(args.cohort.read_text())["cohort"]]
    records = {name: [] for name in ("parameter_decomposition", "baseline_centres", "theta_bounds",
        "oracle_distributions", "refined_roots", "strong_precision", "common_threshold", "checks")}
    common_tags = []
    for tag in tags:
        print(f"Reading saved archive: {tag}", flush=True)
        stem = args.production_dir / f"{tag}_round2"
        summary = json.loads(stem.with_suffix(".json").read_text())
        with np.load(str(stem) + "_fields.npz") as z:
            fields = {key: z[key] for key in z.files}
        with np.load(str(stem) + "_spectral.npz") as z:
            spectral = {key: z[key] for key in z.files}
        subsets = masks(fields, summary)
        records["checks"].append(archive_checks(tag, fields, spectral, subsets))
        norms, bounds, baseline = decomposition(tag, summary, fields, spectral, subsets)
        records["parameter_decomposition"].extend(norms)
        records["theta_bounds"].extend(bounds)
        records["baseline_centres"].append(baseline)
        oracle, curves = oracle_statistics(tag, fields, subsets)
        records["oracle_distributions"].extend(oracle)
        if tag.startswith("burgers_mlp_"):
            records["refined_roots"].extend(root_bounds(tag, fields, spectral, subsets, curves))
        records["strong_precision"].extend(strong_precision(tag, spectral, subsets))
        off_sharp = ~subsets["full"]["sharp"]
        admission_error = np.linalg.norm((fields["u_pred"]-fields["u_exact"])[off_sharp]) / np.linalg.norm(fields["u_exact"][off_sharp])
        if admission_error < .02:
            common_tags.append(tag)
        selected = [choice for choice in summary["choices"] if choice["setting"] == "conditional_noise_mle"
                    or (choice["setting"] == "prior_sweep" and choice["alpha_multiplier"] == 1)]
        for choice in selected:
            k = choice["gamma_index"]
            sigma = np.sqrt(choice["noise_variance"] * fields["unit_variance"][k])
            for centre in ("trained", "affine"):
                error = fields["u_pred"] - fields["u_exact"] + (fields["mean_shift"][k] if centre == "affine" else 0)
                group = subsets["full"]
                E = {r: rms(error[mask]) for r, mask in group.items()}
                S = {r: rms(sigma[mask]) for r, mask in group.items()}
                for region, mask in group.items():
                    z = abs(error[mask])/sigma[mask]
                    records["common_threshold"].append(dict(tag=tag, admission_error=float(admission_error),
                        retained_common2=bool(admission_error < .02), setting=choice["setting"], centre=centre,
                        region=region, gamma=float(fields["gammas"][k]), R_tar=E["sharp"]/E["bulk"],
                        R_sigma=S["sharp"]/S["bulk"], rho=E["sharp"]/E["bulk"]/ (S["sharp"]/S["bulk"]),
                        error_rms=E[region], full_error_rms=rms(error),
                        coverage68=float(np.mean(z <= 1)), coverage95=float(np.mean(z <= 1.96))))
        del fields, spectral
    toy = toy_envelope_check()
    for name, rows in records.items():
        write_csv(args.out / f"{name}.csv", rows)
    base = records["baseline_centres"]
    full_bounds = [r for r in records["theta_bounds"] if r["subset"] == "full"]
    roots = records["refined_roots"]
    root_tags = {r["tag"] for r in roots}
    five_oracles = [r for r in records["oracle_distributions"] if r["tag"] in root_tags
                   and r["centre"] == "trained" and r["subset"] == "full"]
    def span(rows, key):
        return [min(r[key] for r in rows), max(r[key] for r in rows)]
    aggregate = dict(cohort_count=len(tags), common2_count=len(common_tags), common2_tags=common_tags,
        all_settings_total_relative_theta_range=span(records["parameter_decomposition"], "total_relative_theta"),
        baseline_total_relative_theta_range=span(base, "total_relative_theta"),
        baseline_complement_relative_theta_range=span(base, "qr_complement_relative_theta"),
        baseline_likelihood_norm_range=span(base, "likelihood_norm"),
        baseline_likelihood_relative_theta_range=span(base, "likelihood_relative_theta"),
        baseline_c_range=span(base, "c_baseline"), noise_mle_c_range=span(base, "c_noise_mle"),
        theta_deficit_certified_count=sum(r["theta_rho_lower"] > 1 for r in base),
        theta_deficit_unresolved=[dict(tag=r["tag"], rho_lower=r["theta_rho_lower"]) for r in base if r["theta_rho_lower"] <= 1],
        affine_worsening_count=sum(r["affine_worsens"] for r in base),
        prior_only_worsening_certified_count=sum(r["affine_worsens"] and r["prior_only_worsening_certified"] for r in base),
        max_parameter_norm_relative_defect=max(r["relative_norm_reconstruction_error"] for r in records["parameter_decomposition"]),
        normal_norm_reconstruction_count=len(records["parameter_decomposition"]),
        max_regional_variance_relative_defect=max(r["regional_variance_max_relative_defect"] for r in records["checks"]),
        max_stored_envelope_relative_violation=max(r["stored_intermediate_envelope_max_relative_violation"] for r in records["checks"]),
        root_count=len(roots), refined_roots=roots,
        toy_envelope_check=toy)
    for region in ("bulk", "sharp"):
        group = [r for r in full_bounds if r["region"] == region]
        for coverage, nominal in (("68", .6826894921370859), ("95", .95)):
            aggregate[f"theta_{region}_coverage{coverage}_min_lower"] = min(r[f"coverage{coverage}_lower"] for r in group)
            aggregate[f"theta_{region}_coverage{coverage}_certified_over_count"] = sum(r[f"coverage{coverage}_lower"] > nominal for r in group)
            aggregate[f"theta_{region}_coverage{coverage}_certified_under"] = [
                dict(tag=r["tag"], upper=r[f"coverage{coverage}_upper"]) for r in group if r[f"coverage{coverage}_upper"] < nominal]
        selected = [r for r in five_oracles if r["region"] == region]
        aggregate[f"oracle_{region}_median_range"] = span(selected, "median_Z")
        aggregate[f"oracle_{region}_q99_range"] = span(selected, "q99_Z")
        aggregate[f"oracle_{region}_top1_energy_range"] = span(selected, "top1_error_energy_fraction")
        aggregate[f"oracle_{region}_spearman_range"] = span(selected, "spearman_sigma_abs_error")
    strong = [r for r in records["strong_precision"] if r["gamma_over_lambda_max"] == 1000]
    aggregate["strong1000_max_relative_limit_error"] = max(r["relative_contrast_limit_error"] for r in strong)
    aggregate["strong1000_max_first_order_error"] = max(r["relative_first_order_error"] for r in strong)
    for cohort_name, selected_tags in (("all17", tags), ("common13", common_tags)):
        bb = [r for r in base if r["tag"] in selected_tags]
        aggregate[f"{cohort_name}_affine_worsening_count"] = sum(r["affine_worsens"] for r in bb)
        aggregate[f"{cohort_name}_theta_deficit_certified_count"] = sum(r["theta_rho_lower"] > 1 for r in bb)
        rr = [r for r in records["common_threshold"] if r["tag"] in selected_tags and r["centre"] == "trained"]
        for setting in ("prior_sweep", "conditional_noise_mle"):
            ss = [r for r in rr if r["setting"] == setting and r["region"] == "sharp"]
            aggregate[f"{cohort_name}_{setting}_rho_range"] = span(ss, "rho")
            for region in ("sharp", "bulk"):
                region_rows = [r for r in rr if r["setting"] == setting and r["region"] == region]
                for coverage, nominal in (("68", .6826894921370859), ("95", .95)):
                    aggregate[f"{cohort_name}_{setting}_{region}_coverage{coverage}_range"] = span(region_rows, f"coverage{coverage}")
                    aggregate[f"{cohort_name}_{setting}_{region}_coverage{coverage}_under_count"] = sum(r[f"coverage{coverage}"] < nominal for r in region_rows)
    protocol = dict(production_directory=str(args.production_dir.resolve()), cohort_file=str(args.cohort.resolve()),
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, oracle_quantile_method="higher", oracle_target=.68,
        root_selection="observed full-grid trained Burgers-MLP sign-changing saved intervals only",
        root_relative_interval_tolerance=1e-12, common_threshold="original nonsharp admission error < 0.02",
        coverage_thresholds=[1, 1.96], scope="returned floating-point affine model; no directed-rounding certificate",
        performed="saved-array summaries, scalar log-bisection and analytic envelopes only",
        unperformed="new pointwise fields; theta-centred MLE/evidence predictions; nonlinear shifted network; new likelihood")
    (args.out / "summary.json").write_text(json.dumps(dict(protocol=protocol, aggregate=aggregate), indent=2), encoding="utf-8")
    text = ["# Saved-archive third-round results", "", f"Reviewed all {len(tags)} admitted checkpoints; no Jacobian or network pass.", "",
        "## Centre decomposition and prediction bounds", "",
        f"- Across all saved precisions, total displacement / parameter norm: {aggregate['all_settings_total_relative_theta_range']}.",
        f"- Zero-mean baseline displacement / parameter norm: {aggregate['baseline_total_relative_theta_range']}.",
        f"- QR-complement / parameter norm: {aggregate['baseline_complement_relative_theta_range']}.",
        f"- Likelihood displacement / parameter norm: {aggregate['baseline_likelihood_relative_theta_range']}.",
        f"- Baseline covariance-metric bound c: {aggregate['baseline_c_range']}; MLE c: {aggregate['noise_mle_c_range']}.",
        f"- Theta-centred baseline deficit bounded in {aggregate['theta_deficit_certified_count']}/17; unresolved: {aggregate['theta_deficit_unresolved']}.",
        f"- Prior-only worsening bounded in {aggregate['prior_only_worsening_certified_count']}/{aggregate['affine_worsening_count']} baseline worsening cases.",
        f"- Sharp baseline 1-sigma coverage lower bound: {aggregate['theta_sharp_coverage68_min_lower']:.6f}; bulk: {aggregate['theta_bulk_coverage68_min_lower']:.6f}.",
        "", "## Numerically refined trained-centre roots", "",
        "Positive-modal envelopes include the final root interval and uncertainty in the higher bulk-68% calibration quantile.", "",
        "| Seed | Precision | Sharp 1-sigma coverage | Sharp 1.96-sigma coverage |", "|---|---:|---:|---:|"]
    for r in roots:
        text.append(f"| {r['tag'].split('seed')[-1]} | {r['root_gamma']:.8g} | {r['sharp_coverage68_lower']:.5f}--{r['sharp_coverage68_upper']:.5f} | {r['sharp_coverage95_lower']:.5f}--{r['sharp_coverage95_upper']:.5f} |")
    text.extend(["", "These bound coverage at the five observed trained-centre crossings; they neither compute exact fields nor prove uniqueness or global optimality.",
        "", "## Oracle distributions at the nearest saved settings", "",
        f"Five crossing seeds, sharp median |e|/sigma: {aggregate['oracle_sharp_median_range']}; bulk: {aggregate['oracle_bulk_median_range']}.",
        f"Sharp 99th percentile: {aggregate['oracle_sharp_q99_range']}; bulk: {aggregate['oracle_bulk_q99_range']}.",
        f"Sharp top-1% error-energy share: {aggregate['oracle_sharp_top1_energy_range']}; sharp rank association: {aggregate['oracle_sharp_spearman_range']}.",
        "These are descriptive statistics, not evidence that a particular tail or bias mechanism causes the coverage gap.",
        "rho^2 equals the ratio of regional sigma^2-weighted second moments of e/sigma; it does not determine unweighted coverage quantiles.",
        "", "## Strong precision and common threshold", "",
        f"At gamma = 1000 lambda_max, maximum contrast-limit relative error: {aggregate['strong1000_max_relative_limit_error']:.6g}; first-order squared-contrast error: {aggregate['strong1000_max_first_order_error']:.6g}.",
        f"Original nonsharp 2% gate retains {len(common_tags)}: {', '.join(common_tags)}.",
        f"Baseline affine worsening: all17 {aggregate['all17_affine_worsening_count']}/17, common13 {aggregate['common13_affine_worsening_count']}/{len(common_tags)}.",
        f"Theta-centred baseline deficit bounds: common13 {aggregate['common13_theta_deficit_certified_count']}/{len(common_tags)}.",
        "Full per-run baseline/MLE records and all aggregate coverage ranges are in common_threshold.csv and summary.json.",
        "", "## Verification and limits", "",
        f"{aggregate['normal_norm_reconstruction_count']} saved total displacement norms reproduced; max relative defect {aggregate['max_parameter_norm_relative_defect']:.3g}.",
        f"Regional field/modal variance maximum relative defect {aggregate['max_regional_variance_relative_defect']:.3g}; stored intermediate envelope maximum relative violation {aggregate['max_stored_envelope_relative_violation']:.3g}.",
        f"The positive-modal toy checks 49 interior precisions, including an exact null mode: {toy}.",
        "Theta-centred MLE/evidence fields, nonlinear shifted predictions, new likelihoods and exact pointwise root fields remain unperformed.",
        "Bounds concern the returned floating-point affine model and have ordinary tolerance checks, not interval-arithmetic certification of the original Jacobian.", ""])
    (args.out / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
