"""J1/J3/J5: exact theta-centred shifts and nonlinear checks on 17 checkpoints.

All inputs and outputs belong to this isolated study. The checkpoint, residual
weights and covariance linearisation point remain fixed. A nonlinear shifted
prediction is a centre diagnostic; its covariance is still evaluated at theta.
"""
import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from model_residual import make_model, residual_blocks, build_jacobian
from theta_linalg import (factor_jacobian, posterior_modes, reconstruct_shifts,
                          gradient_components, unit_variance, theta_log_evidence)
from theta_archive import NativeArchive, verify_archive_manifest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
ROOT = Path(__file__).resolve().parents[1]


def settings_for(tag, root, old):
    evidence_path = root/"outputs/evidence"/f"{tag}_theta_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    alpha0, noise_mle = old["alpha0"], old["conditional_noise_mle"]
    choices = [dict(setting="baseline", alpha=alpha0, noise_variance=1.,
                    selection="fixed_baseline"),
               dict(setting="conditional_noise_mle", alpha=alpha0,
                    noise_variance=noise_mle, selection="conditional_mle_at_theta")]
    for label in ("original", "extended"):
        for prior, best in (("zero", old[f"{label}_affine_evidence"]),
                            ("theta", evidence["selections"][label])):
            choices.append(dict(setting=f"{prior}_evidence_{label}",
                alpha=best["alpha"], noise_variance=best["noise_variance"],
                selection=f"best_tested_{prior}_prior_affine_evidence",
                alpha_boundary=bool(best["alpha_boundary"]),
                noise_boundary=bool(best["noise_boundary"])))
    roots_path = root/"outputs/local/root_settings.json"
    roots = json.loads(roots_path.read_text(encoding="utf-8"))
    for k, record in enumerate(roots.get(tag, [])):
        gamma = float(record["gamma"])
        choices.append(dict(setting=record.get("setting", record.get("name", f"trained_root_{k}")),
            alpha=gamma, noise_variance=1., selection="reference_informed_trained_root",
            root_lower=float(record["lower"]), root_upper=float(record["upper"]),
            root_target_subset=record.get("subset", "full")))
    for choice in choices:
        choice["gamma"] = choice["alpha"]*choice["noise_variance"]
    gammas = np.unique([c["gamma"] for c in choices])
    for choice in choices:
        choice["gamma_index"] = int(np.flatnonzero(gammas == choice["gamma"])[0])
    return choices, gammas, evidence_path, roots_path


def signature(tag, root, choices, evidence_path, roots_path):
    checkpoint = root/"inputs/checkpoints"/tag
    files = [p for p in checkpoint.iterdir() if p.is_file()]
    files += [root/"inputs/round2"/f"{tag}_round2{suffix}"
              for suffix in (".json", "_fields.npz", "_spectral.npz")]
    files += [root/"inputs/baseline"/f"{tag}_fullgrid_sigma_raw.npz",
              evidence_path, roots_path]
    sources = [Path(__file__), Path(__file__).with_name("theta_linalg.py"),
               Path(__file__).with_name("model_residual.py"),
               Path(__file__).with_name("theta_archive.py")]
    return dict(inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files},
                sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                choices=choices,
                versions=dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__))


def early_checks(theta, u, data, blocks, saved_loss, chunk, grid_chunk):
    """Validate the saved model and residual convention before building J."""
    x, t = np.meshgrid(data["xs"], data["ts"])
    take = np.linspace(0, x.size-1, min(grid_chunk, x.size), dtype=int)
    predict = jax.jit(jax.vmap(u, in_axes=(None, 0, 0)))
    rebuilt = np.asarray(predict(theta, x.ravel()[take], t.ravel()[take]))
    np.testing.assert_allclose(rebuilt, data["u_pred"][take], rtol=1e-9, atol=1e-10)
    parts = []
    for function, points in blocks:
        evaluate = jax.jit(jax.vmap(function, in_axes=(None,)+(0,)*len(points)))
        for start in range(0, len(points[0]), chunk):
            coordinates = [jnp.asarray(p[start:start+chunk]) for p in points]
            parts.append(np.asarray(evaluate(theta, *coordinates)))
    residual = np.concatenate(parts)
    loss = float(np.mean(residual**2)/2)
    np.testing.assert_allclose(loss, saved_loss, rtol=1e-6, atol=1e-12)
    print(f"Early checks passed: {len(take)} predictions; {len(residual)} residuals.", flush=True)
    return dict(prediction_samples=len(take), residual_count=len(residual),
                max_prediction_absolute_defect=float(np.max(np.abs(rebuilt-data['u_pred'][take]))),
                rebuilt_training_loss=loss)


def region_masks(gradient, times, pde):
    p70, p95 = np.percentile(gradient, (70, 95))
    masks = dict(bulk=gradient < p70, transition=(gradient >= p70)&(gradient < p95),
                 sharp=gradient >= p95)
    result = {f"full/{name}": mask for name, mask in masks.items()}
    if pde == "burgers":
        result.update({f"post_formation/{name}": mask&(times >= 1/np.pi)
                       for name, mask in masks.items()})
    return result, dict(p70=float(p70), p95=float(p95))


def evaluate_grid(theta, u, data, factors, modes, gammas, deltas, chunk, archive=None):
    """One tangent pass supplies all covariance and affine fields."""
    xx, tt = np.meshgrid(data["xs"], data["ts"])
    x, t = xx.ravel(), tt.ravel()
    predict = jax.jit(jax.vmap(u, in_axes=(None, 0, 0)))
    grad = jax.jit(jax.vmap(jax.grad(u), in_axes=(None, 0, 0)))
    push = jax.jit(lambda a, d, bx, bt: jax.jvp(
        lambda b: jax.vmap(u, in_axes=(None, 0, 0))(b, bx, bt), (a,), (d,))[1])
    fields = {name: np.empty((len(gammas), len(x))) for name in
              ("unit_variance", "theta_shift", "zero_shift", "theta_nonlinear")}
    fields["complement_shift"] = np.empty(len(x))
    maximum_norm_defect, maximum_prediction_defect, jvp_defect, jvp_scaled_defect = 0., 0., 0., 0.
    for start in range(0, len(x), chunk):
        end = min(start+chunk, len(x))
        bx, bt = x[start:end], t[start:end]
        rebuilt = np.asarray(predict(theta, bx, bt))
        np.testing.assert_allclose(rebuilt, data["u_pred"][start:end], rtol=1e-9, atol=1e-10)
        maximum_prediction_defect = max(maximum_prediction_defect,
                                       float(np.max(np.abs(rebuilt-data['u_pred'][start:end]))))
        gradients = np.asarray(grad(theta, bx, bt))
        c, cc, norm = gradient_components(factors, gradients)
        complement_energy = np.sum(cc**2, axis=0)
        rebuilt_norm = np.sum(c**2, axis=0)+complement_energy
        maximum_norm_defect = max(maximum_norm_defect, float(np.max(
            np.abs(rebuilt_norm-norm)/np.maximum(norm, np.finfo(float).tiny))))
        complement_shift = modes["complement_coefficients"] @ cc
        if archive is not None:
            samples = min(3, len(gradients)) if start == 0 else 0
            archive.write_grid_chunk(start, c, complement_energy, complement_shift,
                gradient_samples=gradients[:samples] if samples else None,
                sample_indices=np.arange(samples) if samples else None)
        fields["unit_variance"][:, start:end] = unit_variance(
            factors[3], gammas, c**2, complement_energy)
        fields["theta_shift"][:, start:end] = modes["likelihood_modes"] @ c
        fields["zero_shift"][:, start:end] = modes["zero_modes"] @ c+complement_shift
        fields["complement_shift"][start:end] = complement_shift
        for k, delta in enumerate(deltas):
            fields["theta_nonlinear"][k, start:end] = np.asarray(
                predict(theta+jnp.asarray(delta), bx, bt))
            # One fixed first batch checks the output linearisation independently.
            if start == 0:
                direct = np.asarray(push(theta, jnp.asarray(delta), bx, bt))
                difference = np.abs(direct-fields["theta_shift"][k, start:end])
                scale = np.sqrt(norm)*np.linalg.norm(delta)
                scaled = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
                jvp_defect = max(jvp_defect, float(difference.max()))
                jvp_scaled_defect = max(jvp_scaled_defect, float(scaled.max()))
        if start % (25*chunk) == 0:
            print(f"Evaluation grid: {end}/{len(x)}", flush=True)
    fields["diagnostics"] = dict(max_prediction_absolute_defect=maximum_prediction_defect,
        max_gradient_norm_relative_defect=maximum_norm_defect,
        affine_jvp_absolute_defect=jvp_defect, affine_jvp_scaled_defect=jvp_scaled_defect,
        affine_jvp_samples=min(chunk, len(x)))
    return fields


def independent_normal_checks(theta, blocks, residual, gammas, deltas, chunk, lambda_max):
    """JVP/VJP check of (J.T J + gamma I) delta_L = -J.T r."""
    operators, start, jtr = [], 0, np.zeros(len(theta))
    for function, points in blocks:
        push, pull = block_operators(function, len(points))
        operators.append((points, push, pull, start))
        for q in range(0, len(points[0]), chunk):
            end = min(q+chunk, len(points[0]))
            coordinates = [jnp.asarray(p[q:end]) for p in points]
            jtr += np.asarray(pull(theta, jnp.asarray(residual[start+q:start+end]), *coordinates))
        start += len(points[0])
    checks = []
    for gamma, delta in zip(gammas, deltas):
        pulled = np.zeros(len(theta))
        direction = jnp.asarray(delta)
        for points, push, pull, start in operators:
            for q in range(0, len(points[0]), chunk):
                end = min(q+chunk, len(points[0]))
                coordinates = [jnp.asarray(p[q:end]) for p in points]
                jd = push(theta, direction, *coordinates)
                pulled += np.asarray(pull(theta, jd, *coordinates))
        h = pulled+gamma*delta+jtr
        delta_norm = float(np.linalg.norm(delta))
        denominator = float((lambda_max+gamma)*delta_norm+np.linalg.norm(jtr))
        absolute = float(np.linalg.norm(h))
        scaled = absolute/denominator if denominator else absolute
        checks.append(dict(gamma=float(gamma), residual_norm=absolute,
            denominator=denominator, normwise_backward_error=scaled,
            delta_norm=delta_norm, rhs_norm=float(np.linalg.norm(jtr)),
            interpretation="Consistency check; not a weak-mode forward-error certificate."))
        print(f"Theta mean check: gamma={gamma:.7g}, backward error={scaled:.3g}", flush=True)
    return checks


def block_operators(function, nargs):
    """Bind each residual block before constructing its compiled operators."""
    vector = jax.vmap(function, in_axes=(None,)+(0,)*nargs)

    @jax.jit
    def push(a, d, *coords):
        return jax.jvp(lambda b: vector(b, *coords), (a,), (d,))[1]

    @jax.jit
    def pull(a, w, *coords):
        return jax.vjp(lambda b: vector(b, *coords), a)[1](w)[0]

    return push, pull


def scientific_rows(tag, data, choices, fields, masks):
    rows = []
    names = sorted({name.split('/')[0] for name in masks})
    for choice in choices:
        k = choice["gamma_index"]
        unit_sigma = np.sqrt(fields["unit_variance"][k])
        centres = {"trained": data["u_pred"],
            "theta_affine": data["u_pred"]+fields["theta_shift"][k],
            "theta_nonlinear": fields["theta_nonlinear"][k],
            "zero_affine": data["u_pred"]+fields["zero_shift"][k]}
        for centre, prediction in centres.items():
            error = prediction-data["u_exact"]
            for subset in names:
                selected = {name.split('/')[1]: mask for name, mask in masks.items()
                            if name.startswith(subset+'/')}
                scales = [("selected", float(np.sqrt(choice["noise_variance"])))]
                if choice["selection"] == "reference_informed_trained_root":
                    z_bulk = np.abs(error[selected['bulk']])/unit_sigma[selected['bulk']]
                    scales.append(("bulk_68_oracle", float(np.quantile(z_bulk, .68, method="higher"))))
                for calibration, multiplier in scales:
                    if multiplier <= 0:
                        raise ValueError("A zero oracle multiplier does not define a positive noise variance.")
                    sigma = unit_sigma*multiplier
                    error_rms = {r: float(np.sqrt(np.mean(error[m]**2))) for r, m in selected.items()}
                    sigma_rms = {r: float(np.sqrt(np.mean(sigma[m]**2))) for r, m in selected.items()}
                    contrast = sigma_rms["sharp"]/sigma_rms["bulk"]
                    target = error_rms["sharp"]/error_rms["bulk"]
                    for region, mask in selected.items():
                        signed_z = error[mask]/sigma[mask]
                        z = np.abs(signed_z)
                        remainder = fields["theta_nonlinear"][k, mask]-(
                            data["u_pred"][mask]+fields["theta_shift"][k, mask])
                        remainder_rms = float(np.sqrt(np.mean(remainder**2)))
                        trained_error_rms = float(np.sqrt(np.mean(
                            (data["u_pred"][mask]-data["u_exact"][mask])**2)))
                        rows.append(dict(tag=tag, setting=choice["setting"],
                            selection=choice["selection"], gamma=choice["gamma"],
                            alpha=choice["alpha"], noise_variance=choice["noise_variance"],
                            calibration=calibration, sigma_multiplier=multiplier,
                            evaluation_noise_variance=multiplier**2,
                            evaluation_alpha=choice["gamma"]/multiplier**2,
                            reference_informed=calibration != "selected" or
                                choice["selection"] == "reference_informed_trained_root",
                            centre=centre, subset=subset, region=region, count=int(mask.sum()),
                            root_target_subset=choice.get("root_target_subset", ""),
                            R_sigma_rms=contrast, R_tar=target, rho=target/contrast,
                            error_rms=error_rms[region], sigma_rms=sigma_rms[region],
                            mean_Z=float(z.mean()), signed_mean_Z=float(signed_z.mean()),
                            median_Z=float(np.median(z)), q99_Z=float(np.quantile(z, .99)),
                            coverage68=float(np.mean(z <= 1.)), coverage95=float(np.mean(z <= 1.96)),
                            theta_shift_rms=float(np.sqrt(np.mean(fields['theta_shift'][k, mask]**2))),
                            zero_shift_rms=float(np.sqrt(np.mean(fields['zero_shift'][k, mask]**2))),
                            linearisation_remainder_rms=remainder_rms,
                            linearisation_remainder_over_sigma_rms=remainder_rms/sigma_rms[region],
                            standardised_linearisation_remainder_rms=float(np.sqrt(np.mean((remainder/sigma[mask])**2))),
                            linearisation_remainder_over_centre_error_rms=remainder_rms
                                /error_rms[region] if error_rms[region] else None,
                            linearisation_remainder_over_trained_error_rms=remainder_rms
                                /trained_error_rms if trained_error_rms else None))
    return rows


def analyse(tag, args):
    started = time.perf_counter()
    root = args.root.resolve()
    old_path = root/"inputs/round2"/f"{tag}_round2.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    choices, gammas, evidence_path, roots_path = settings_for(tag, root, old)
    identity = signature(tag, root, choices, evidence_path, roots_path)
    identity["chunks"] = dict(jacobian=args.jac_chunk, grid=args.grid_chunk)
    archive_root = (args.archive_root or root/"retained").resolve()
    retain = args.retain == "all" or (args.retain == "burgers-mlp" and tag.startswith("burgers_mlp_"))
    if retain and (archive_root == root or any(archive_root.is_relative_to(root/name)
            for name in ("inputs", "outputs", "scripts", "cluster", "cluster_records", "logs", "verification"))):
        raise ValueError("Use a separate archive directory, outside inputs and routine outputs.")
    identity["retention"] = dict(enabled=retain,
                                  archive_root=str(archive_root) if retain else None)
    out = root/"outputs/production"
    out.mkdir(parents=True, exist_ok=True)
    stem = out/f"{tag}_theta"
    if stem.with_suffix(".json").exists() and not args.resume:
        raise ValueError(f"{tag}: a completed result exists; use --resume to reuse it without overwriting.")
    if args.resume and stem.with_suffix(".json").exists():
        done = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        companions = [stem.with_suffix(".csv"), Path(str(stem)+"_fields.npz"),
                      Path(str(stem)+"_shifts.npz")]
        if done["signature"] != identity or not all(p.is_file() for p in companions):
            raise ValueError(f"{tag}: completion is incompatible or incomplete; review before reuse.")
        if retain:
            retained = done.get("archive_validation_report") or {}
            path = Path(retained.get("manifest_path", archive_root/tag/"manifest.json"))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != retained.get("manifest_sha256"):
                raise ValueError(f"{tag}: retained archive manifest is missing or differs from completion.")
            verify_archive_manifest(path, checksums=False)
        print(f"RESUMED {tag}; numerical_status={done['numerical_status']}", flush=True)
        return done
    if retain and (archive_root/tag).exists():
        raise ValueError(f"{tag}: archive directory already exists; inspect the incomplete archive before rerunning.")
    run = root/"inputs/checkpoints"/tag
    metadata = json.loads((run/f"{tag}_map.json").read_text(encoding="utf-8-sig"))
    config = metadata["provenance"]["config"]
    with (run/f"{tag}_params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    with np.load(run/f"{tag}_fields.npz") as saved:
        data = {key: saved[key] for key in
            ("xs", "ts", "x_u", "t_b", "x_f", "t_f", "u_pred", "u_exact", "grad_abs")}
    for name in ("u_pred", "u_exact", "grad_abs"):
        data[name] = data[name].ravel()
    fourier = None
    if config["arch"] == "fourier":
        with np.load(run/f"{tag}_fourier_B.npz") as saved:
            fourier = {key: jnp.asarray(saved[key]) for key in ("B_X", "B_T")}
    theta, u = make_model(params, config, fourier)
    blocks = residual_blocks(u, config, data)
    early = early_checks(theta, u, data, blocks, metadata["final_loss"], args.jac_chunk, args.grid_chunk)
    jacobian, residual = build_jacobian(theta, blocks, args.jac_chunk)
    alpha0, noise_mle = 1/float(np.mean(np.asarray(theta)**2)), float(np.mean(residual**2))
    np.testing.assert_allclose(alpha0, old["alpha0"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(noise_mle, old["conditional_noise_mle"], rtol=1e-6, atol=1e-15)
    print(f"{tag}: P={len(theta)}, N={len(residual)}, {len(gammas)} distinct gammas.", flush=True)
    print("Direct-J QR/SVD factorisation...", flush=True)
    factors = factor_jacobian(jacobian)
    del jacobian
    singular = factors[3]
    modes = posterior_modes(np.asarray(theta), residual, factors, gammas)
    archive = None
    if retain:
        leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        parameter_layout = []
        offset = 0
        for keypath, leaf in leaves:
            size = int(np.size(leaf))
            parameter_layout.append(dict(path=jax.tree_util.keystr(keypath),
                shape=list(np.shape(leaf)), dtype=str(np.asarray(leaf).dtype),
                flat_start=offset, flat_end=offset+size))
            offset += size
        block_weights = [config["ic_weight"]]
        block_weights += ([config["bc_weight"]]*2 if config["pde"] == "burgers"
                          else [config["bcu_weight"], config["bcx_weight"]])
        block_weights += [1.]
        residual_layout, row_start = [], 0
        coordinate_names = [("x_u",), ("t_b",), ("t_b",), ("x_f", "t_f")]
        for (function, points), weight, coordinates in zip(blocks, block_weights, coordinate_names):
            row_end = row_start+len(points[0])
            residual_layout.append(dict(name=function.__name__, row_start=row_start,
                row_end=row_end, raw_working_multiplier=float(weight),
                coordinate_arrays=list(coordinates)))
            row_start = row_end
        archive = NativeArchive(archive_root, tag, factors, np.asarray(theta), residual,
            modes, len(data["u_pred"]), metadata=dict(signature=identity, config=config,
                parameter_layout=parameter_layout,
                residual_layout=residual_layout,
                parameter_flattening="jax.flatten_util.ravel_pytree; tree_flatten order, C-order per leaf",
                grid=dict(xs=data["xs"].tolist(),
                ts=data["ts"].tolist(), flattening="C-order meshgrid(xs, ts); x varies fastest")))
    delta_l = reconstruct_shifts(factors, modes["likelihood_modes"])
    delta_zero = reconstruct_shifts(factors, modes["zero_modes"], modes["complement_coefficients"])
    delta_c = reconstruct_shifts(factors, np.zeros((1, len(singular))),
                                 modes["complement_coefficients"])[0]
    evidence_checks = []
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with np.load(root/"inputs/round2"/f"{tag}_round2_spectral.npz") as archived:
        for label in ("original", "extended"):
            surface = theta_log_evidence(singular, modes["residual_modes"],
                archived["evidence_alpha_grid"], archived[f"{label}_noise_grid"])
            i, j = np.unravel_index(np.argmax(surface), surface.shape)
            best = evidence["selections"][label]
            selected_i, selected_j = int(best["alpha_index"]), int(best["noise_index"])
            regret = float(surface[i, j]-surface[selected_i, selected_j])
            evidence_checks.append(dict(grid=label, rebuilt_best_indices=[int(i), int(j)],
                saved_best_indices=[selected_i, selected_j], rebuilt_log_evidence_regret=regret,
                interpretation="Coordinates in unresolved clusters may rotate; compare grid evidence, not individual modal signs."))
    masks, thresholds = region_masks(data["grad_abs"],
                                    np.repeat(data["ts"], len(data["xs"])), config["pde"])
    fields = evaluate_grid(theta, u, data, factors, modes, gammas, delta_l, args.grid_chunk,
                           archive=archive)
    if np.any(fields["unit_variance"] <= 0) or any(not np.isfinite(fields[k]).all()
            for k in ("unit_variance", "theta_shift", "zero_shift", "theta_nonlinear")):
        raise ValueError("Nonpositive covariance or nonfinite predictive field.")
    baseline_index = next(c['gamma_index'] for c in choices if c['setting'] == 'baseline')
    with np.load(root/"inputs/baseline"/f"{tag}_fullgrid_sigma_raw.npz") as baseline:
        np.testing.assert_array_equal(baseline["xs"], data["xs"])
        np.testing.assert_array_equal(baseline["ts"], data["ts"])
        sigma0, saved_sigma = np.sqrt(fields["unit_variance"][baseline_index]), baseline["sigma"].ravel()
        np.testing.assert_allclose(sigma0, saved_sigma, rtol=1e-4, atol=1e-10)
        baseline_defect = float(np.max(np.abs(sigma0/saved_sigma-1)))
    archive_checks = []
    with np.load(root/"inputs/round2"/f"{tag}_round2_fields.npz") as archived:
        for name in ("xs", "ts", "u_pred", "u_exact", "grad_abs"):
            np.testing.assert_array_equal(np.asarray(archived[name]).ravel(), np.asarray(data[name]).ravel())
        for k, gamma in enumerate(gammas):
            matched = np.flatnonzero(np.isclose(archived["gammas"], gamma, rtol=1e-12, atol=0))
            if len(matched):
                j = int(matched[0])
                v_error = float(np.max(np.abs(fields['unit_variance'][k]/archived['unit_variance'][j]-1)))
                difference = fields['zero_shift'][k]-archived['mean_shift'][j]
                scale = np.maximum(np.sqrt(fields['unit_variance'][k]), np.abs(archived['mean_shift'][j]))
                archive_checks.append(dict(gamma=float(gamma), unit_variance_relative_defect=v_error,
                    zero_affine_shift_scaled_defect=float(np.max(np.abs(difference)/scale))))
    retained_archive = None
    if archive is not None:
        print("Reloading and checking retained factors and grid modes...", flush=True)
        retained_archive = archive.finalize(float(gammas[baseline_index]),
            fields["unit_variance"][baseline_index], fields["zero_shift"][baseline_index])
        print(f"Archive verification: {retained_archive['numerical_status']}", flush=True)
    # The independent AD check no longer needs the large factorisation in memory.
    del archive, factors
    normal = independent_normal_checks(theta, blocks, residual, gammas, delta_l,
                                       args.jac_chunk, float(singular[0]**2))
    rows = scientific_rows(tag, data, choices, fields, masks)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(str(stem)+"_fields.npz", gammas=gammas,
        **{key: value for key, value in fields.items() if key != "diagnostics"},
        xs=data["xs"], ts=data["ts"], u_pred=data["u_pred"], u_exact=data["u_exact"], grad_abs=data["grad_abs"])
    np.savez_compressed(str(stem)+"_shifts.npz", gammas=gammas, theta=np.asarray(theta),
        delta_likelihood=delta_l, delta_zero=delta_zero, delta_prior=delta_zero-delta_l,
        delta_complement=delta_c, singular=singular, residual_modes=modes['residual_modes'],
        theta_modes=modes['theta_modes'], likelihood_modes=modes['likelihood_modes'],
        prior_modes=modes['prior_modes'], complement_coefficients=modes['complement_coefficients'])
    diagnostic_values = [fields['diagnostics']['max_gradient_norm_relative_defect'],
                         fields['diagnostics']['affine_jvp_scaled_defect']]
    diagnostic_values += [r['normwise_backward_error'] for r in normal]
    numerical_review = any(not np.isfinite(v) or v > 1e-7 for v in diagnostic_values)
    numerical_review |= any(r['rebuilt_log_evidence_regret'] > 1e-3 for r in evidence_checks)
    numerical_review |= any(r['unit_variance_relative_defect'] > 1e-4 or
                            r['zero_affine_shift_scaled_defect'] > 1e-4 for r in archive_checks)
    numerical_review |= retained_archive is not None and not retained_archive["checks_passed"]
    theta_norm = float(np.linalg.norm(theta))
    summary = dict(tag=tag, config=config, choices=choices, gammas=gammas.tolist(),
        alpha0=alpha0, conditional_noise_mle=noise_mle, parameter_count=len(theta),
        residual_count=len(residual), lambda_max=float(singular[0]**2),
        theta_norm=theta_norm, delta_theta_norm=np.linalg.norm(delta_l, axis=1).tolist(),
        delta_theta_relative_norm=(np.linalg.norm(delta_l, axis=1)/theta_norm).tolist(),
        delta_zero_relative_norm=(np.linalg.norm(delta_zero, axis=1)/theta_norm).tolist(),
        delta_complement_relative_norm=float(np.linalg.norm(delta_c)/theta_norm),
        region_thresholds=thresholds, early_checks=early, grid_checks=fields['diagnostics'],
        evidence_rebuild_checks=evidence_checks, archived_field_checks=archive_checks,
        baseline_max_sigma_relative_error=baseline_defect, normal_equation_checks=normal,
        conventions=dict(prior_theta="N(theta, I/alpha)", prior_zero="N(0, I/alpha)",
            covariance="v*(J.T J+gamma I)^-1, gamma=alpha*v",
            theta_shift="-(J.T J+gamma I)^-1 J.T r", no_rank_cutoff=True,
            nonlinear_check="Evaluate u(theta+delta_L); covariance remains at theta.",
            oracle="Bulk empirical 0.68 higher quantile, reference-informed, at trained-root gammas; evaluation_alpha and evaluation_noise_variance preserve gamma.",
            numerical_scope="Backward and reconstruction consistency checks are not forward-accuracy certificates for unresolved modes.",
            shift_norm_scope="Parameter displacement is a diagnostic; nonlinear/affine disagreement tests output linearisation."),
        signature=identity, requested_retention=args.retain,
        archive_validation_report=retained_archive,
        numerical_status="REVIEW_REQUIRED" if numerical_review else "PASS",
        elapsed_seconds=time.perf_counter()-started)
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(f"Completed {tag}; {summary['elapsed_seconds']/3600:.2f} h; {summary['numerical_status']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="One cohort checkpoint; omission runs all 17 serially.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retain", choices=("all", "burgers-mlp", "none"), default="all",
        help="Retain float64 native factors and signed grid modes; default retains all checkpoints.")
    parser.add_argument("--archive-root", type=Path,
        help="Dedicated persistent archive directory; default ROOT/retained, excluded from routine downloads.")
    parser.add_argument("--jac-chunk", type=int, default=50)
    parser.add_argument("--grid-chunk", type=int, default=200)
    args = parser.parse_args()
    manifest = json.loads((args.root/"manifest.json").read_text(encoding="utf-8"))
    tags = [entry["tag"] for entry in manifest["runs"]]
    if args.tag:
        if args.tag not in tags:
            parser.error(f"Unknown cohort tag: {args.tag}")
        tags = [args.tag]
    print("Versions:", jax.__version__, np.__version__, scipy.__version__, flush=True)
    print("Devices:", jax.devices(), flush=True)
    for tag in tags:
        analyse(tag, args)


if __name__ == "__main__":
    main()
