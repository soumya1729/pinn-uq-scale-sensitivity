"""Small linear-Gaussian numerical checks and one fixed weak-spectrum toy.

Run locally; this script reads no PINN checkpoint and changes no manuscript.
The QR/SVD and evidence formulas match analysis/n1/n1_sensitivity.py.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, qr, svd
from scipy.linalg.lapack import dormqr
from scipy.special import ndtr


def factor(jacobian):
    (packed, tau), triangular = qr(
        np.array(jacobian.T, order="F", copy=True), mode="raw", overwrite_a=True)
    rotation, singular, observation_rotation = svd(
        triangular, full_matrices=False, overwrite_a=True)
    return packed, tau, rotation, singular, observation_rotation


def apply_q(packed, tau, values, transpose=False):
    values = np.array(values, dtype=np.float64, order="F", copy=True)
    trans = b"T" if transpose else b"N"
    _, work, info = dormqr(b"L", trans, packed, tau, values, -1)
    if info:
        raise RuntimeError(f"QR workspace query returned {info}")
    result, _, info = dormqr(b"L", trans, packed, tau, values, int(work[0]), overwrite_c=1)
    if info:
        raise RuntimeError(f"QR transform returned {info}")
    return result


def mean_covariance(jacobian, residual, theta, factors, alpha, noise):
    packed, tau, rotation, singular, observation_rotation = factors
    n, p = jacobian.shape
    orthogonal = apply_q(packed, tau, np.eye(p))
    basis = orthogonal[:, :n] @ rotation
    complement = orthogonal[:, n:]
    gamma = alpha * noise
    theta_modes = basis.T @ theta
    delta_modes = -(singular * (observation_rotation @ residual)
                    + gamma * theta_modes) / (singular**2 + gamma)
    delta = basis @ delta_modes - complement @ (complement.T @ theta)
    covariance = (basis * (noise / (singular**2 + gamma))) @ basis.T
    covariance += complement @ complement.T / alpha
    return theta + delta, covariance, delta, complement


def evidence_grid(eigenvalues, projected_data, alpha_grid, noise_grid):
    """Same full-mode, constant-including likelihood as the frozen N1 path."""
    surface = np.empty((len(alpha_grid), len(noise_grid)))
    for i, alpha in enumerate(alpha_grid):
        diagonal = noise_grid[:, None] + eigenvalues[None, :] / alpha
        surface[i] = -0.5 * (len(eigenvalues) * np.log(2 * np.pi)
                            + np.sum(np.log(diagonal)
                                     + projected_data[None, :]**2 / diagonal, axis=1))
    return surface


def evidence_slope(eigenvalues, projected_data, alpha, noise):
    diagonal = noise + eigenvalues / alpha
    return float(-0.5 * np.sum((noise / diagonal)
                              * (1 - projected_data**2 / diagonal)))


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def small_checks(rng):
    n, p = 12, 20
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(p, n)))
    singular = np.geomspace(3.0, 0.3, n)
    jacobian = (left * singular) @ right.T
    theta = rng.normal(size=p)
    residual = rng.normal(size=n)
    y = jacobian @ theta - residual
    factors = factor(jacobian)
    z = factors[4] @ y
    alpha = 2.0
    settings = []
    for gamma in (1e-4, 1.0, 1e4):
        noise = gamma / alpha
        mean, covariance, delta, complement = mean_covariance(
            jacobian, residual, theta, factors, alpha, noise)
        precision = jacobian.T @ jacobian + gamma * np.eye(p)
        dense_factor = cho_factor(precision)
        mean_dense = cho_solve(dense_factor, jacobian.T @ y)
        covariance_dense = noise * cho_solve(dense_factor, np.eye(p))
        data_covariance = noise * np.eye(n) + jacobian @ jacobian.T / alpha
        data_factor = cho_factor(data_covariance)
        dense_log_evidence = float(-0.5 * (n * np.log(2 * np.pi)
            + 2 * np.log(np.diag(data_factor[0])).sum()
            + y @ cho_solve(data_factor, y)))
        spectral_log_evidence = float(evidence_grid(
            factors[3]**2, z, np.array([alpha]), np.array([noise]))[0, 0])
        log_step = 1e-5
        nearby = evidence_grid(factors[3]**2, z, np.array([alpha]),
                               noise * np.exp(np.array([-log_step, log_step])))
        slope_fd = float((nearby[0, 1] - nearby[0, 0]) / (2 * log_step))
        slope = evidence_slope(factors[3]**2, z, alpha, noise)
        normal = jacobian.T @ (residual + jacobian @ delta) + gamma * (theta + delta)
        denom = (np.linalg.norm(jacobian.T @ residual)
                 + np.linalg.norm(jacobian.T @ (residual + jacobian @ delta))
                 + gamma * (np.linalg.norm(theta) + np.linalg.norm(delta)))
        matched_mean, matched_covariance, matched_delta, _ = mean_covariance(
            jacobian, residual, theta, factors, alpha / 7, noise * 7)
        row = dict(gamma=gamma, alpha=alpha, noise_variance=noise,
                   mean_relative_error=relative_error(mean, mean_dense),
                   covariance_relative_error=relative_error(covariance, covariance_dense),
                   log_evidence_absolute_error=abs(spectral_log_evidence - dense_log_evidence),
                   slope_absolute_error=abs(slope - slope_fd),
                   normal_residual_absolute=float(np.linalg.norm(normal)),
                   normal_residual_scaled=float(np.linalg.norm(normal) / denom),
                   complement_shift_absolute_error=float(np.linalg.norm(
                       complement.T @ (delta + theta))),
                   matched_mean_absolute_error=float(np.linalg.norm(matched_mean - mean)),
                   matched_variance_relative_error=relative_error(matched_covariance, 7 * covariance),
                   matched_delta_absolute_error=float(np.linalg.norm(matched_delta - delta)))
        settings.append(row)
    # Tolerances here distinguish formula/implementation errors, not infer PINN accuracy.
    assert max(r["mean_relative_error"] for r in settings) < 1e-9
    assert max(r["covariance_relative_error"] for r in settings) < 1e-9
    assert max(r["log_evidence_absolute_error"] for r in settings) < 1e-9
    assert max(r["slope_absolute_error"] for r in settings) < 1e-7
    return settings, jacobian


def coverage_check(rng, jacobian):
    """Repeated true-prior/noise draws; centre comparison has fixed one-sigma shift."""
    n, p = jacobian.shape
    draws, alpha, noise = 1000, 2.0, 0.2
    output = rng.normal(size=(10, p)) / np.sqrt(p)
    precision = jacobian.T @ jacobian + alpha * noise * np.eye(p)
    dense_factor = cho_factor(precision)
    covariance = noise * cho_solve(dense_factor, np.eye(p))
    sigma = np.sqrt(np.einsum("ip,pq,iq->i", output, covariance, output))
    theta_star = rng.normal(size=(draws, p)) / np.sqrt(alpha)
    y = theta_star @ jacobian.T + np.sqrt(noise) * rng.normal(size=(draws, n))
    posterior_mean = cho_solve(dense_factor, jacobian.T @ y.T).T
    z = (theta_star - posterior_mean) @ output.T / sigma
    result = dict(draws=draws, outputs_per_draw=len(output), alpha_true=alpha,
                  noise_variance_true=noise, displacement_in_sigma=1.0,
                  centre="exact affine posterior output mean")
    for name, values, offset in (("posterior_mean", z, 0.0), ("displaced_centre", z - 1.0, 1.0)):
        result[name] = {}
        for label, critical in (("one_sigma", 1.0), ("nominal_95", 1.959963984540054)):
            per_draw = np.mean(np.abs(values) <= critical, axis=1)
            result[name][label] = dict(coverage=float(per_draw.mean()),
                draw_level_standard_error=float(per_draw.std(ddof=1) / np.sqrt(draws)),
                exact_posterior_probability=float(ndtr(critical - offset)
                                                 + ndtr(critical + offset) - 1))
    return result


def fitted_residual_mle_check(rng):
    """Separate N>P illustration: denominator N, without a df correction."""
    n, p, draws, noise = 40, 12, 1000, 0.2
    design = rng.normal(size=(n, p))
    y = rng.normal(size=(draws, p)) @ design.T + np.sqrt(noise) * rng.normal(size=(draws, n))
    fitted = np.linalg.lstsq(design, y.T, rcond=None)[0].T @ design.T
    estimates = np.mean((y - fitted)**2, axis=1)
    return dict(n=n, p=p, draws=draws, noise_variance_true=noise,
                estimator="squared fitted residual norm / N", mean_estimate=float(estimates.mean()),
                expected_estimate=(n-p)*noise/n,
                standard_error=float(estimates.std(ddof=1)/np.sqrt(draws)))


def weak_spectrum_toy(out):
    """One prespecified design and finite search grid; no adaptive enlargement."""
    seed, n, p, draws = 10072026, 96, 160, 100
    alpha_true, noise_true = 2.0, 1e-4
    rng = np.random.default_rng(seed)
    prescribed_s = np.geomspace(1e2, 1e-6, n)
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(p, n)))
    jacobian = (left * prescribed_s) @ right.T
    factors = factor(jacobian)
    alpha_grid = alpha_true * 10.0 ** np.arange(-3, 3.0001, 0.25)
    noise_grid = noise_true * 10.0 ** np.arange(-4, 4.0001, 0.25)
    protocol = dict(seed=seed, n=n, p=p, draws=draws, alpha_true=alpha_true,
                    noise_variance_true=noise_true, gamma_true=alpha_true*noise_true,
                    singular_values=prescribed_s.tolist(), alpha_grid=alpha_grid.tolist(),
                    noise_variance_grid=noise_grid.tolist(), grid_step_decades=0.25,
                    design="fixed random orthogonal singular vectors; log-spaced singular values",
                    observation="J theta_star + epsilon; independent true Gaussian prior/noise",
                    comparison="QR/SVD evidence against prescribed-spectrum analytic likelihood",
                    scope="one design, 100 independent draws, no adaptive grid or draw extension")
    (out / "weak_spectrum_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records, production_surfaces, analytic_surfaces, y_values = [], [], [], []
    for draw in range(draws):
        theta_star = rng.normal(size=p) / np.sqrt(alpha_true)
        y = jacobian @ theta_star + np.sqrt(noise_true) * rng.normal(size=n)
        production = evidence_grid(factors[3]**2, factors[4] @ y, alpha_grid, noise_grid)
        reference = evidence_grid(prescribed_s**2, left.T @ y, alpha_grid, noise_grid)
        i, j = np.unravel_index(np.argmax(production), production.shape)
        ri, rj = np.unravel_index(np.argmax(reference), reference.shape)
        log_error = float(np.max(np.abs(production - reference)))
        # All cells tested, including deliberately poorly fitting small-noise cells.
        relative_surface = float(np.max(np.abs(production - reference) / (1+np.abs(reference))))
        log_step = 1e-5
        nearby = evidence_grid(factors[3]**2, factors[4] @ y, np.array([alpha_grid[i]]),
                               noise_grid[j] * np.exp(np.array([-log_step, log_step])))
        slope_fd = float((nearby[0, 1] - nearby[0, 0]) / (2 * log_step))
        slope = evidence_slope(factors[3]**2, factors[4] @ y, alpha_grid[i], noise_grid[j])
        records.append(dict(draw=draw, alpha_hat=float(alpha_grid[i]), noise_variance_hat=float(noise_grid[j]),
            gamma_hat=float(alpha_grid[i]*noise_grid[j]), alpha_index=int(i), noise_index=int(j),
            alpha_lower_boundary=bool(i==0), alpha_upper_boundary=bool(i==len(alpha_grid)-1),
            noise_lower_boundary=bool(j==0), noise_upper_boundary=bool(j==len(noise_grid)-1),
            interior=bool(0<i<len(alpha_grid)-1 and 0<j<len(noise_grid)-1),
            reference_selected_pair_agrees=bool((i,j)==(ri,rj)),
            max_log_evidence_absolute_error=log_error, max_scaled_evidence_error=relative_surface,
            selected_log_evidence_absolute_error=float(abs(production[i,j]-reference[i,j])),
            selected_slope_absolute_error=abs(slope-slope_fd),
            log10_alpha_ratio=float(np.log10(alpha_grid[i]/alpha_true)),
            log10_noise_ratio=float(np.log10(noise_grid[j]/noise_true)),
            log10_gamma_ratio=float(np.log10(alpha_grid[i]*noise_grid[j]/(alpha_true*noise_true)))))
        production_surfaces.append(production)
        analytic_surfaces.append(reference)
        y_values.append(y)
    with (out / "weak_spectrum_draws.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(out / "weak_spectrum_toy.npz", jacobian=jacobian, prescribed_singular=prescribed_s,
        factor_singular=factors[3], observation_rotation=factors[4], prescribed_observation_rotation=left.T,
        alpha_grid=alpha_grid, noise_grid=noise_grid, y=np.array(y_values),
        production_log_evidence=np.array(production_surfaces), analytic_log_evidence=np.array(analytic_surfaces))
    summary = dict(protocol_file="weak_spectrum_protocol.json", draws=draws,
                   interpretation="Finite-sample scale estimates need not equal generating scales. "
                   "Analytic agreement tests numerics; it does not establish PINN likelihood adequacy.")
    for key in ("interior", "alpha_lower_boundary", "alpha_upper_boundary", "noise_lower_boundary",
                "noise_upper_boundary", "reference_selected_pair_agrees"):
        summary[key + "_count"] = int(sum(r[key] for r in records))
    summary["quantile_probabilities"] = [0.05, 0.25, 0.5, 0.75, 0.95]
    for key in ("log10_alpha_ratio", "log10_noise_ratio", "log10_gamma_ratio"):
        summary[key + "_quantiles"] = np.quantile([r[key] for r in records],
            summary["quantile_probabilities"]).tolist()
    for key in ("max_log_evidence_absolute_error", "max_scaled_evidence_error", "selected_log_evidence_absolute_error", "selected_slope_absolute_error"):
        summary[key] = float(max(r[key] for r in records))
    assert summary["reference_selected_pair_agrees_count"] == draws
    assert summary["max_scaled_evidence_error"] < 1e-7
    assert summary["selected_slope_absolute_error"] < 1e-7
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs" / "validation")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7102026)
    checks, jacobian = small_checks(rng)
    report = dict(status="PASS", numerical_checks=checks, coverage=coverage_check(rng, jacobian),
                  fitted_residual_mle=fitted_residual_mle_check(rng), evidence_toy=weak_spectrum_toy(args.out),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="small linear-Gaussian validation only; no PINN checkpoint or manuscript changes")
    (args.out / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
