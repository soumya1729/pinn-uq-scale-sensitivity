"""N1: full-network, full-grid prior/noise sensitivity at a frozen checkpoint.

Uses the saved training points and the original training residual weights.
No training, recentering, rank truncation, or extra output-noise variance.
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax.flatten_util import ravel_pytree
from scipy.linalg import qr, svd
from scipy.linalg.lapack import dormqr

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
HERE = Path(__file__).resolve().parent


def make_model(params, config, fourier=None):
    """The saved spatial/temporal subnetworks and fusion network in common.py."""
    theta, unpack = ravel_pytree(params)

    def mlp(layers, h):
        for layer in layers[:-1]:
            h = jnp.tanh(h @ layer["w"] + layer["b"])
        return h @ layers[-1]["w"] + layers[-1]["b"]

    def embed(z, key):
        if fourier is None:
            return jnp.atleast_1d(z)
        phase = 2 * jnp.pi * z * fourier[key].ravel()
        return jnp.concatenate((jnp.cos(phase), jnp.sin(phase)))

    def u(vector, x, t):
        p = unpack(vector)
        hx, ht = embed(x, "B_X"), embed(t, "B_T")
        features = [mlp(net, hx) for net in p["spatial"]]
        features += [mlp(net, ht) for net in p["temporal"]]
        return mlp(p["fusion"], jnp.concatenate(features))[0]

    return theta, u


def residual_blocks(u, config, data):
    """Raw training residuals, including the fixed IC and boundary weights."""
    c = config
    ux, ut = jax.grad(u, 1), jax.grad(u, 2)
    uxx = jax.grad(ux, 1)
    if c["pde"] == "burgers":
        def ic(a, x):
            return c["ic_weight"] * (u(a, x, 0.0) + jnp.sin(jnp.pi * x))

        def bc1(a, t):
            return c["bc_weight"] * u(a, -1.0, t)

        def bc2(a, t):
            return c["bc_weight"] * u(a, 1.0, t)

        def pde(a, x, t):
            return ut(a, x, t) + u(a, x, t) * ux(a, x, t) - c["nu"] * uxx(a, x, t)
    elif c["pde"] == "allen_cahn":
        def ic(a, x):
            return c["ic_weight"] * (u(a, x, 0.0) - x**2 * jnp.cos(jnp.pi * x))

        def bc1(a, t):
            return c["bcu_weight"] * (u(a, -1.0, t) - u(a, 1.0, t))

        def bc2(a, t):
            return c["bcx_weight"] * (ux(a, -1.0, t) - ux(a, 1.0, t))

        def pde(a, x, t):
            value = u(a, x, t)
            return ut(a, x, t) - c["eps2"] * uxx(a, x, t) - c["reac"] * value + c["reac"] * value**3
    else:
        raise ValueError("N1 supports the paper's admitted Burgers and Allen-Cahn settings.")
    return [(ic, (data["x_u"],)), (bc1, (data["t_b"],)),
            (bc2, (data["t_b"],)), (pde, (data["x_f"], data["t_f"]))]


def build_jacobian(theta, blocks, chunk):
    n = sum(len(points[0]) for _, points in blocks)
    jacobian = np.empty((n, len(theta)), dtype=np.float64)
    residual = np.empty(n, dtype=np.float64)
    start = 0
    for function, points in blocks:
        evaluate = jax.jit(jax.vmap(jax.value_and_grad(function),
                                  in_axes=(None,) + (0,) * len(points)))
        for q in range(0, len(points[0]), chunk):
            end = min(q + chunk, len(points[0]))
            values, rows = evaluate(theta, *[jnp.asarray(p[q:end]) for p in points])
            residual[start + q:start + end] = np.asarray(values)
            jacobian[start + q:start + end] = np.asarray(rows)
        start += len(points[0])
        print(f"Residual Jacobian: {start}/{n} rows", flush=True)
    return jacobian, residual


def factor_jacobian(jacobian):
    """J.T = Q R; R = U diag(s) V.T. Store Q as Householder reflectors."""
    (packed, tau), triangular = qr(jacobian.T, mode="raw", overwrite_a=True)
    rotation, singular, observation_rotation = svd(triangular, full_matrices=False, overwrite_a=True)
    return packed, tau, rotation, singular**2, observation_rotation


def evidence_grid(eigenvalues, projected_data, alpha_grid, noise_grid):
    """Exact affine-model log evidence on a finite grid, including its constant."""
    surface = np.empty((len(alpha_grid), len(noise_grid)))
    for i, a in enumerate(alpha_grid):
        diagonal = noise_grid[:, None] + eigenvalues[None, :] / a
        surface[i] = -0.5 * (len(eigenvalues) * np.log(2 * np.pi)
                             + np.sum(np.log(diagonal) + projected_data[None, :]**2 / diagonal, axis=1))
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    near = np.argwhere(surface >= surface[i, j] - 1.0)
    best = dict(alpha=float(alpha_grid[i]), noise_variance=float(noise_grid[j]),
                log_evidence=float(surface[i, j]),
                alpha_boundary=bool(i in (0, len(alpha_grid) - 1)),
                noise_boundary=bool(j in (0, len(noise_grid) - 1)),
                within_one_log_unit_count=len(near),
                within_one_log_unit_alpha_range=[float(alpha_grid[near[:, 0]].min()), float(alpha_grid[near[:, 0]].max())],
                within_one_log_unit_noise_range=[float(noise_grid[near[:, 1]].min()), float(noise_grid[near[:, 1]].max())])
    return best, surface


def gradient_energies(gradient, packed, tau, rotation):
    """Positive mode energies and the full-network orthogonal-complement energy."""
    work_array = np.array(gradient.T, dtype=np.float64, order="F", copy=True)
    _, work, info = dormqr(b"L", b"T", packed, tau, work_array, -1)
    if info != 0:
        raise RuntimeError(f"Householder workspace query failed: {info}")
    transformed, _, info = dormqr(b"L", b"T", packed, tau, work_array,
                                 int(work[0]), overwrite_c=1)
    if info != 0:
        raise RuntimeError(f"Householder transform failed: {info}")
    n = len(tau)
    coefficients = rotation.T @ transformed[:n]
    complement = np.sum(transformed[n:]**2, axis=0)
    norm = np.sum(gradient**2, axis=1)
    rotated_norm = np.sum(transformed**2, axis=0)
    norm_error = float(np.max(np.abs(rotated_norm - norm) / norm))
    return coefficients**2, complement, norm_error


def variances_from_energies(mode_energy, complement, eigenvalues, alpha, noise_variance):
    """Covariance = (alpha I + J.T J / noise_variance)^-1; no subtraction."""
    weights = noise_variance[:, None] / (eigenvalues[None, :] + (alpha * noise_variance)[:, None])
    return weights @ mode_energy + complement[None, :] / alpha[:, None]


def metric_rows(tag, choices, variance, error, masks):
    error_rms = {name: float(np.sqrt(np.mean(error[mask]**2))) for name, mask in masks.items()}
    target = error_rms["sharp"] / error_rms["bulk"]
    rows = []
    for choice, field in zip(choices, variance):
        sigma = np.sqrt(field)
        rms = {name: float(np.sqrt(np.mean(field[mask]))) for name, mask in masks.items()}
        means = {name: float(np.mean(sigma[mask])) for name, mask in masks.items()}
        contrast = rms["sharp"] / rms["bulk"]
        for name, mask in masks.items():
            z = np.abs(error[mask]) / sigma[mask]
            rows.append(dict(tag=tag, **choice, effective_precision=choice["alpha"] * choice["noise_variance"],
                             R_sigma_rms=contrast, R_sigma_mean=means["sharp"] / means["bulk"],
                             R_tar=target, rho=target / contrast, region=name, count=int(mask.sum()),
                             error_rms=error_rms[name], sigma_rms=rms[name], mean_sigma=means[name],
                             mean_Z=float(z.mean()), coverage68=float(np.mean(z <= 1.0)),
                             coverage95=float(np.mean(z <= 1.96))))
    return rows


def check_matched_pairs(choices, variance, metrics, pairs):
    """Same effective precision: sigma scales by sqrt(v); contrast and rho agree."""
    checks = {}
    for selected, matched in pairs:
        choice = choices[selected]
        sigma = np.sqrt(variance[selected])
        expected = np.sqrt(choice["noise_variance"]) * np.sqrt(variance[matched])
        np.testing.assert_allclose(sigma, expected, rtol=1e-10, atol=0,
                                   err_msg=choice["setting"])
        actual = np.array([metrics[selected][key] for key in ("R_sigma_rms", "rho")])
        reference = np.array([metrics[matched][key] for key in ("R_sigma_rms", "rho")])
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=0,
                                   err_msg=choice["setting"])
        checks[choice["setting"]] = dict(matched_alpha=choices[matched]["alpha"],
            max_sigma_relative_difference=float(np.max(np.abs(sigma / expected - 1))),
            contrast_relative_difference=float(abs(actual[0] / reference[0] - 1)),
            rho_relative_difference=float(abs(actual[1] / reference[1] - 1)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE.parent / "n5/input/burgers_mlp_seed2")
    parser.add_argument("--baseline-sigma", type=Path, help="Saved raw full-network sigma field")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    parser.add_argument("--alpha-mults", type=float, nargs="+", default=np.logspace(-7, 2, 19).tolist())
    parser.add_argument("--evidence-alpha-mults", type=float, nargs="+", default=np.logspace(-7, 7, 57).tolist())
    parser.add_argument("--evidence-noise-mults", type=float, nargs="+", default=np.logspace(-4, 6, 41).tolist(),
                        help="Noise-variance multipliers relative to the conditional residual MLE")
    parser.add_argument("--jac-chunk", type=int, default=50)
    parser.add_argument("--grid-chunk", type=int, default=200)
    args = parser.parse_args()
    if min(args.alpha_mults + args.evidence_alpha_mults + args.evidence_noise_mults) <= 0 or min(args.jac_chunk, args.grid_chunk) <= 0:
        parser.error("Precisions and chunk sizes must be positive.")
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    tag = run_dir.name
    metadata = json.loads((run_dir / f"{tag}_map.json").read_text(encoding="utf-8-sig"))
    config = metadata["provenance"]["config"]
    with (run_dir / f"{tag}_params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    with np.load(run_dir / f"{tag}_fields.npz") as archive:
        data = {key: archive[key] for key in ("xs", "ts", "x_u", "t_b", "x_f", "t_f",
                                            "u_pred", "u_exact", "grad_abs")}
    fourier = None
    if config["arch"] == "fourier":
        with np.load(run_dir / f"{tag}_fourier_B.npz") as archive:
            fourier = {key: jnp.asarray(archive[key]) for key in ("B_X", "B_T")}
    baseline_path = args.baseline_sigma or HERE / "baseline" / f"{tag}_fullgrid_sigma_raw.npz"
    with np.load(baseline_path) as archive:
        baseline = archive["sigma"].ravel()
        np.testing.assert_array_equal(archive["xs"], data["xs"])
        np.testing.assert_array_equal(archive["ts"], data["ts"])
    xs, ts = np.meshgrid(data["xs"], data["ts"])
    x, t = xs.ravel(), ts.ravel()
    saved_prediction = data["u_pred"].ravel()
    error = saved_prediction - data["u_exact"].ravel()
    gradient = data["grad_abs"].ravel()
    p70, p95 = np.percentile(gradient, (70, 95))
    masks = {"bulk": gradient < p70, "transition": (gradient >= p70) & (gradient < p95),
             "sharp": gradient >= p95}
    theta, u = make_model(params, config, fourier)
    predict = jax.jit(jax.vmap(u, in_axes=(None, 0, 0)))
    sample = np.linspace(0, len(x) - 1, 32, dtype=int)
    np.testing.assert_allclose(np.asarray(predict(theta, x[sample], t[sample])),
                               saved_prediction[sample], rtol=1e-9, atol=1e-10)
    alpha0 = 1.0 / float(np.mean(np.asarray(theta)**2))
    jacobian, residual = build_jacobian(theta, residual_blocks(u, config, data), args.jac_chunk)
    noise_mle = float(np.mean(residual**2))
    np.testing.assert_allclose(noise_mle / 2, metadata["final_loss"], rtol=1e-6, atol=1e-12)
    if noise_mle <= 0:
        raise ValueError("The conditional noise MLE must be positive; no artificial floor is used.")
    # QR may overwrite J: form the affine observation before factorising it.
    affine_data = jacobian @ np.asarray(theta) - residual
    print(f"{tag}: alpha0={alpha0:.8g}, residual variance MLE={noise_mle:.8g}", flush=True)
    print("Factorising the full Jacobian once (QR and triangular SVD)...", flush=True)
    packed, tau, rotation, eigenvalues, observation_rotation = factor_jacobian(jacobian)
    del jacobian
    evidence_alpha = alpha0 * np.array(sorted(set(args.evidence_alpha_mults + [1.0])))
    evidence_noise = np.array(sorted(set([noise_mle * m for m in args.evidence_noise_mults] + [noise_mle, 1.0])))
    evidence, evidence_surface = evidence_grid(eigenvalues, observation_rotation @ affine_data,
                                               evidence_alpha, evidence_noise)
    del observation_rotation
    boundary = evidence["alpha_boundary"] or evidence["noise_boundary"]
    print(f"Best tested affine evidence: alpha={evidence['alpha']:.8g}, variance={evidence['noise_variance']:.8g}; "
          f"{'grid boundary (not an interior optimum)' if boundary else 'interior grid point'}", flush=True)
    matched_mle = noise_mle
    matched_evidence = evidence["alpha"] * evidence["noise_variance"] / alpha0
    choices = [dict(setting="prior_sweep", alpha_multiplier=m, alpha=alpha0 * m, noise_variance=1.0)
               for m in sorted(set(args.alpha_mults + [1.0, matched_mle, matched_evidence]))]
    choices.append(dict(setting="conditional_noise_mle", alpha_multiplier=1.0,
                        alpha=alpha0, noise_variance=noise_mle))
    choices.append(dict(setting="affine_evidence_grid_boundary" if boundary else "affine_evidence_grid_interior",
                        alpha_multiplier=evidence["alpha"] / alpha0, alpha=evidence["alpha"],
                        noise_variance=evidence["noise_variance"]))
    alpha = np.array([c["alpha"] for c in choices])
    noise_variance = np.array([c["noise_variance"] for c in choices])
    variance = np.empty((len(choices), len(x)), dtype=np.float64)
    grad_u = jax.jit(jax.vmap(jax.grad(u), in_axes=(None, 0, 0)))
    norm_error = 0.0
    for q in range(0, len(x), args.grid_chunk):
        end = min(q + args.grid_chunk, len(x))
        np.testing.assert_allclose(np.asarray(predict(theta, x[q:end], t[q:end])),
                                   saved_prediction[q:end], rtol=1e-9, atol=1e-10)
        g = np.asarray(grad_u(theta, x[q:end], t[q:end]))
        energy, complement, defect = gradient_energies(g, packed, tau, rotation)
        norm_error = max(norm_error, defect)
        variance[:, q:end] = variances_from_energies(energy, complement, eigenvalues, alpha, noise_variance)
        if q % (25 * args.grid_chunk) == 0:
            print(f"Evaluation grid: {end}/{len(x)} points", flush=True)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Non-finite or non-positive predictive variance.")
    operating = next(i for i, c in enumerate(choices) if c["setting"] == "prior_sweep" and c["alpha_multiplier"] == 1.0)
    sigma0 = np.sqrt(variance[operating])
    np.testing.assert_allclose(sigma0, baseline, rtol=1e-4, atol=1e-10)
    rows = metric_rows(tag, choices, variance, error, masks)
    matched_indices = [next(i for i, c in enumerate(choices)
                            if c["setting"] == "prior_sweep" and c["alpha_multiplier"] == m)
                       for m in (matched_mle, matched_evidence)]
    pairs = list(zip((len(choices) - 2, len(choices) - 1), matched_indices))
    matched_checks = check_matched_pairs(choices, variance, rows[::len(masks)], pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{tag}_n1"
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(stem.with_suffix(".npz"), variance=variance, alpha=alpha,
                        noise_variance=noise_variance, xs=data["xs"], ts=data["ts"],
                        evidence_alpha_grid=evidence_alpha, evidence_noise_grid=evidence_noise,
                        log_evidence=evidence_surface)
    summary = dict(tag=tag, run_dir=str(run_dir), baseline_sigma=str(baseline_path), config=config,
                   covariance="(alpha I + J.T J / noise_variance)^-1", centre="saved trained prediction",
                   residual_count=len(residual), parameter_count=len(theta), alpha0=alpha0,
                   conditional_noise_mle=noise_mle, rebuilt_loss=noise_mle / 2,
                   saved_loss=metadata["final_loss"], largest_fisher_eigenvalue=float(eigenvalues.max()),
                   smallest_jacobian_singular_value_squared=float(eigenvalues.min()),
                   max_gradient_norm_relative_error=norm_error,
                   baseline_max_sigma_relative_error=float(np.max(np.abs(sigma0 / baseline - 1))),
                   matched_pair_checks=matched_checks,
                   affine_evidence=evidence,
                   choices=choices, elapsed_seconds=time.perf_counter() - started,
                   versions=dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__))
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {stem}.csv, .json and .npz", flush=True)


if __name__ == "__main__":
    main()
