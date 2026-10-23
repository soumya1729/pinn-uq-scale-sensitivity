"""Select checkpoint-centred affine evidence from the isolated saved spectra.

For vartheta ~ N(theta, I/alpha), the affine residual model has observation
y = -r(theta) and covariance v I + J J.T / alpha. In the saved residual-space
singular basis, y has coordinates -residual_modes. These calculations use no
reference solution. They condition on the already trained checkpoint; their
evidence selections are not a calibration guarantee.

Run from any directory:
    python scripts/select_theta_evidence.py --self-test
    python scripts/select_theta_evidence.py

Both grids are copied exactly from the round-two archives. No grid is extended
adaptively after looking at the new results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def evidence_surface(singular, projected_data, alphas, noises):
    """Full log density, including all residual modes and its normalisation."""
    singular = np.asarray(singular, dtype=float)
    projected_data = np.asarray(projected_data, dtype=float)
    surface = np.empty((len(alphas), len(noises)))
    for i, alpha in enumerate(alphas):
        diagonal = noises[:, None] + singular[None, :] ** 2 / alpha
        surface[i] = -0.5 * (
            len(singular) * np.log(2 * np.pi)
            + np.sum(np.log(diagonal) + projected_data[None, :] ** 2 / diagonal, axis=1)
        )
    return surface


def selection(surface, alphas, noises):
    """Best grid point and the extent of the within-one-log-unit set."""
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    near = np.argwhere(surface >= surface[i, j] - 1.0)
    return dict(
        alpha=float(alphas[i]),
        noise_variance=float(noises[j]),
        effective_precision=float(alphas[i] * noises[j]),
        log_evidence=float(surface[i, j]),
        alpha_index=int(i),
        noise_index=int(j),
        alpha_boundary=bool(i in (0, len(alphas) - 1)),
        noise_boundary=bool(j in (0, len(noises) - 1)),
        within_one_log_unit_count=int(len(near)),
        within_one_log_unit_alpha_range=[
            float(alphas[near[:, 0]].min()), float(alphas[near[:, 0]].max())
        ],
        within_one_log_unit_noise_range=[
            float(noises[near[:, 1]].min()), float(noises[near[:, 1]].max())
        ],
    )


def dense_self_test():
    """Compare modal evidence against direct dense covariance calculations."""
    rng = np.random.default_rng(14)
    max_error = 0.0
    cases = []
    for n, p in ((7, 11), (11, 7)):
        jacobian = rng.normal(size=(n, p))
        theta = rng.normal(size=p)
        residual = rng.normal(size=n)
        basis, singular, _ = np.linalg.svd(jacobian, full_matrices=True)
        singular = np.pad(singular, (0, n - len(singular)))
        alphas = np.array([0.1, 1.0, 8.0])
        noises = np.array([0.03, 0.4, 2.0])
        for centre, data in (("theta", -residual), ("zero", jacobian @ theta - residual)):
            modal = evidence_surface(singular, basis.T @ data, alphas, noises)
            dense = np.empty_like(modal)
            for i, alpha in enumerate(alphas):
                for j, noise in enumerate(noises):
                    covariance = noise * np.eye(n) + jacobian @ jacobian.T / alpha
                    sign, logdet = np.linalg.slogdet(covariance)
                    assert sign > 0
                    dense[i, j] = -0.5 * (
                        n * np.log(2 * np.pi) + logdet
                        + data @ np.linalg.solve(covariance, data)
                    )
            error = float(np.max(np.abs(modal - dense)))
            np.testing.assert_allclose(modal, dense, rtol=1e-12, atol=1e-11)
            cases.append(dict(residual_count=n, parameter_count=p, centre=centre,
                              max_log_evidence_absolute_error=error))
            max_error = max(max_error, error)
    return dict(status="PASS", cases=cases,
                max_log_evidence_absolute_error=max_error)


def variance_summaries(archive, chosen):
    """Regional covariance at a selected setting, independent of its centre."""
    gamma = chosen["effective_precision"]
    means = chosen["noise_variance"] * (
        archive["mean_mode_energy"] @ (1.0 / (archive["singular"] ** 2 + gamma))
        + archive["mean_complement_energy"] / gamma
    )
    return {str(label): float(value) for label, value in zip(archive["region_labels"], means)}


def process_run(root, run):
    tag = run["tag"]
    spectral_path = root / "inputs" / "round2" / f"{tag}_round2_spectral.npz"
    old_json_path = root / "inputs" / "round2" / f"{tag}_round2.json"
    old = json.loads(old_json_path.read_text(encoding="utf-8"))
    output = root / "outputs" / "evidence"
    with np.load(spectral_path) as archive:
        singular = archive["singular"]
        residual_modes = archive["residual_modes"]
        alphas = archive["evidence_alpha_grid"]
        projected_zero = archive["projected_data"]
        result = dict(
            tag=tag,
            alpha0=float(archive["alpha0"]),
            conditional_noise_mle=float(archive["conditional_noise_mle"]),
            residual_count=int(archive["residual_count"]),
            parameter_count=int(archive["parameter_count"]),
            spectral_sha256=hashlib.sha256(spectral_path.read_bytes()).hexdigest(),
            prior_centre="trained theta",
            evidence_scope="conditional frozen affine model, using no reference solution",
            grid_protocol="the two archived grids, unchanged; no adaptive extension",
            old_numerical_status=old["numerical_status"],
        )
        expected_noise = float(np.dot(residual_modes, residual_modes) / len(residual_modes))
        np.testing.assert_allclose(expected_noise, result["conditional_noise_mle"],
                                   rtol=1e-11, atol=0.0)
        result["residual_energy_noise_check_relative_error"] = float(
            abs(expected_noise / result["conditional_noise_mle"] - 1.0)
        )
        surfaces = dict(evidence_alpha_grid=alphas)
        for grid in ("original", "extended"):
            noises = archive[f"{grid}_noise_grid"]
            old_surface = evidence_surface(singular, projected_zero, alphas, noises)
            archived_surface = archive[f"{grid}_log_evidence"]
            np.testing.assert_allclose(old_surface, archived_surface, rtol=1e-12, atol=1e-8)
            old_chosen = selection(old_surface, alphas, noises)
            for key in ("alpha_index", "noise_index"):
                assert old_chosen[key] == old[f"{grid}_affine_evidence"][key]
            theta_surface = evidence_surface(singular, -residual_modes, alphas, noises)
            chosen = selection(theta_surface, alphas, noises)
            prior_variance = singular ** 2 / chosen["alpha"]
            diagonal = chosen["noise_variance"] + prior_variance
            energy_factor = 1.0 - residual_modes ** 2 / diagonal
            chosen["log_evidence_log_alpha_slope"] = float(
                0.5 * np.sum(prior_variance / diagonal * energy_factor)
            )
            chosen["log_evidence_log_noise_slope"] = float(
                -0.5 * np.sum(chosen["noise_variance"] / diagonal * energy_factor)
            )
            chosen["mean_regional_variance"] = variance_summaries(archive, chosen)
            # In the centred model the affine parameter shift is the likelihood part.
            coefficients = -singular * residual_modes / (singular ** 2 + chosen["effective_precision"])
            chosen["affine_likelihood_shift_parameter_norm"] = float(np.linalg.norm(coefficients))
            result[f"theta_evidence_{grid}"] = chosen
            result[f"reproduced_zero_evidence_{grid}"] = old_chosen
            result[f"old_{grid}_surface_max_absolute_error"] = float(
                np.max(np.abs(old_surface - archived_surface))
            )
            surfaces[f"{grid}_noise_grid"] = noises
            surfaces[f"theta_{grid}_log_evidence"] = theta_surface
        result["selections"] = {
            grid: result[f"theta_evidence_{grid}"] for grid in ("original", "extended")
        }
        np.savez_compressed(output / f"{tag}_theta_evidence.npz", **surfaces)
    (output / f"{tag}_theta_evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tags", nargs="+", help="optional subset of manifest tags")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(dense_self_test(), indent=2))
        return
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    runs = manifest["runs"]
    if args.tags:
        unknown = set(args.tags) - {run["tag"] for run in runs}
        if unknown:
            raise ValueError(f"Unknown checkpoint tags: {sorted(unknown)}")
        runs = [run for run in runs if run["tag"] in args.tags]
    output = root / "outputs" / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    test = dense_self_test()
    results = []
    rows = []
    for run in runs:
        result = process_run(root, run)
        results.append(result)
        for grid in ("original", "extended"):
            chosen = result[f"theta_evidence_{grid}"]
            rows.append(dict(tag=result["tag"], grid=grid, **{
                key: chosen[key] for key in (
                    "alpha", "noise_variance", "effective_precision", "log_evidence",
                    "alpha_boundary", "noise_boundary", "within_one_log_unit_count",
                    "affine_likelihood_shift_parameter_norm"
                )
            }))
        chosen = result["theta_evidence_extended"]
        print(f"{result['tag']}: alpha={chosen['alpha']:.6g}, v={chosen['noise_variance']:.6g}, "
              f"gamma={chosen['effective_precision']:.6g}; "
              f"boundary(alpha,v)=({chosen['alpha_boundary']},{chosen['noise_boundary']})", flush=True)
    with (output / "theta_evidence_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(completed_count=len(results), dense_self_test=test,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   runs=results,
                   interpretation="Best tested conditional affine-evidence points, not a calibration certificate.")
    (output / "theta_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(results)} checkpoint selections in {output}")


if __name__ == "__main__":
    main()
