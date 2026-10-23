"""One fixed 100-draw evidence stress test below the old formed-Gram floor.

The target spectrum is synthetic, not a resolved measurement of PINN modes.
Uses the production QR/SVD/evidence core; no checkpoint or manuscript is read.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import round2_linalg as core


def main():
    out = Path(__file__).resolve().parent / "outputs/validation/gram_floor_stress"
    out.mkdir(parents=True, exist_ok=True)
    seed, n, p, draws = 20261008, 96, 160, 100
    alpha_true, gamma_true = 17.652641714178376, 1e-8
    noise_true, maximum = gamma_true / alpha_true, 1.8889569394274544e7
    floor = np.finfo(float).eps * maximum
    eigenvalues = np.concatenate((np.geomspace(maximum, 1.1*floor, 17),
                                 np.geomspace(.9*floor, maximum*10**-21.2, 79)))
    singular = np.sqrt(eigenvalues)
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(p, n)))
    jacobian = (left * singular) @ right.T
    packed, tau, rotation, resolved_s, observation_rotation = core.factor_jacobian(
        np.array(jacobian, order="C", copy=True))
    alpha_grid = alpha_true * 10.**np.arange(-3, 3.0001, .25)
    noise_grid = noise_true * 10.**np.arange(-4, 4.0001, .25)
    protocol = dict(seed=seed, n=n, p=p, draws=draws, alpha_true=alpha_true,
        noise_variance_true=noise_true, gamma_true=gamma_true,
        eigenvalue_maximum=maximum, eigenvalue_minimum=float(eigenvalues.min()),
        eigenvalue_range_decades=21.2, old_formed_gram_floor=floor,
        eigenvalues_below_floor=int(np.sum(eigenvalues < floor)),
        fraction_below_floor=float(np.mean(eigenvalues < floor)),
        alpha_grid=alpha_grid.tolist(), noise_variance_grid=noise_grid.tolist(),
        grid_step_decades=.25, scaled_surface_error_tolerance=1e-7,
        selected_slope_fd_error_tolerance=1e-7, finite_difference_log_step=1e-5,
        design="Fixed random orthogonal vectors; 17 modes above and 79 below eps*lambda_max",
        observation="Independent true prior and observation-noise draws, y=J theta+epsilon",
        scope="One prespecified synthetic spectrum; no adaptive grid, draw count or seed changes",
        caution="Table 8's formed-Gram solver-returned range/fraction describe unresolved modes. "
                "This synthetic stress regime is not a measured resolved PINN spectrum.",
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records, production_surfaces, reference_surfaces, observations = [], [], [], []
    for draw in range(draws):
        theta = rng.normal(size=p)/np.sqrt(alpha_true)
        y = jacobian@theta + np.sqrt(noise_true)*rng.normal(size=n)
        z = observation_rotation@y
        best, actual = core.evidence_grid(resolved_s, z, alpha_grid, noise_grid)
        reference_best, reference = core.evidence_grid(singular, left.T@y, alpha_grid, noise_grid)
        i, j = best["alpha_index"], best["noise_index"]
        ri, rj = reference_best["alpha_index"], reference_best["noise_index"]
        log_step = protocol["finite_difference_log_step"]
        nearby = core.evidence_grid(resolved_s, z, [alpha_grid[i]],
                                    noise_grid[j]*np.exp([-log_step, log_step]))[1][0]
        slope_fd = (nearby[1]-nearby[0])/(2*log_step)
        slope = core.evidence_slope(resolved_s, z, alpha_grid[i], noise_grid[j])["slope"]
        records.append(dict(draw=draw, alpha_hat=best["alpha"], noise_variance_hat=best["noise_variance"],
            gamma_hat=best["effective_precision"], alpha_index=i, noise_index=j,
            reference_alpha_index=ri, reference_noise_index=rj,
            alpha_lower_boundary=i==0, alpha_upper_boundary=i==len(alpha_grid)-1,
            noise_lower_boundary=j==0, noise_upper_boundary=j==len(noise_grid)-1,
            interior=not(best["alpha_boundary"] or best["noise_boundary"]),
            reference_selected_pair_agrees=(i,j)==(ri,rj),
            max_log_evidence_absolute_error=float(np.max(np.abs(actual-reference))),
            max_scaled_evidence_error=float(np.max(np.abs(actual-reference)/(1+np.abs(reference)))),
            selected_log_evidence_absolute_error=float(abs(actual[i,j]-reference[i,j])),
            selected_slope_absolute_error=float(abs(slope-slope_fd)),
            selected_slope=float(slope), selected_slope_finite_difference=float(slope_fd),
            log10_alpha_ratio=float(np.log10(best["alpha"]/alpha_true)),
            log10_noise_ratio=float(np.log10(best["noise_variance"]/noise_true)),
            log10_gamma_ratio=float(np.log10(best["effective_precision"]/gamma_true))))
        production_surfaces.append(actual); reference_surfaces.append(reference); observations.append(y)
    with (out / "draws.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    np.savez_compressed(out / "toy.npz", jacobian=jacobian, prescribed_eigenvalues=eigenvalues,
        prescribed_singular=singular, factor_singular=resolved_s, observation_rotation=observation_rotation,
        prescribed_observation_rotation=left.T, alpha_grid=alpha_grid, noise_grid=noise_grid,
        y=np.asarray(observations), production_log_evidence=np.asarray(production_surfaces),
        analytic_log_evidence=np.asarray(reference_surfaces))
    summary = dict(draws=draws, protocol_file="protocol.json", quantile_probabilities=[.05,.25,.5,.75,.95])
    for key in ("interior", "alpha_lower_boundary", "alpha_upper_boundary", "noise_lower_boundary",
                "noise_upper_boundary", "reference_selected_pair_agrees"):
        summary[key+"_count"] = sum(row[key] for row in records)
    for key in ("log10_alpha_ratio", "log10_noise_ratio", "log10_gamma_ratio"):
        summary[key+"_quantiles"] = np.quantile([row[key] for row in records],
                                                summary["quantile_probabilities"]).tolist()
    for key in ("max_log_evidence_absolute_error", "max_scaled_evidence_error",
                "selected_log_evidence_absolute_error", "selected_slope_absolute_error"):
        summary[key] = max(row[key] for row in records)
    checks = dict(all_selected_pairs_agree=summary["reference_selected_pair_agrees_count"]==draws,
        surface_error=summary["max_scaled_evidence_error"] < protocol["scaled_surface_error_tolerance"],
        slope_error=summary["selected_slope_absolute_error"] < protocol["selected_slope_fd_error_tolerance"])
    summary.update(checks=checks, status="PASS" if all(checks.values()) else "REVIEW_REQUIRED",
        interpretation="Analytic agreement checks the implementation on this fixed synthetic design. "
        "Scale estimates need not equal generating values; this cannot establish PINN likelihood adequacy.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
