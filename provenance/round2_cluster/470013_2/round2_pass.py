"""One shared full-network N1/N3/spectral/evidence pass on the fixed 17 checkpoints.

Frozen prediction and raw training weights; no retraining, whitening, rank cutoff,
or nonlinear network evaluation at shifted parameters. Outputs never replace N1.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "n1"))
import n1_sensitivity as n1
from round2_linalg import (factor_jacobian, transform, posterior_modes, reconstruct_shift,
                           evidence_grid, evidence_slope, evidence_stability)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def block_operators(function, nargs):
    vector = jax.vmap(function, in_axes=(None,) + (0,) * nargs)

    @jax.jit
    def push(theta, direction, *points):
        return jax.jvp(lambda a: vector(a, *points), (theta,), (direction,))[1]

    @jax.jit
    def pull(theta, weights, *points):
        return jax.vjp(lambda a: vector(a, *points), theta)[1](weights)[0]

    return push, pull


def early_checkpoint_checks(theta, u, data, blocks, saved_loss, jac_chunk, grid_chunk):
    """A small prediction batch and residual-only loss, before constructing J."""
    xx, tt = np.meshgrid(data["xs"], data["ts"])
    take = np.linspace(0, xx.size-1, min(grid_chunk, xx.size), dtype=int)
    predict = jax.jit(jax.vmap(u, in_axes=(None, 0, 0)))
    pred = np.asarray(predict(theta, xx.ravel()[take], tt.ravel()[take]))
    np.testing.assert_allclose(pred, data["u_pred"][take], rtol=1e-9, atol=1e-10)
    values = []
    for function, points in blocks:
        evaluate = jax.jit(jax.vmap(function, in_axes=(None,)+(0,)*len(points)))
        for q in range(0, len(points[0]), jac_chunk):
            end = min(q+jac_chunk, len(points[0]))
            values.append(np.asarray(evaluate(theta, *[jnp.asarray(p[q:end]) for p in points])))
    residual = np.concatenate(values)
    loss = float(np.mean(residual**2)/2)
    np.testing.assert_allclose(loss, saved_loss, rtol=1e-6, atol=1e-12)
    print(f"Early prediction/loss checks passed: {len(take)} grid samples, {len(residual)} residuals.", flush=True)
    return dict(prediction_samples=len(take), max_prediction_absolute_defect=float(np.max(np.abs(pred-data["u_pred"][take]))),
                residual_count=len(residual), residual_only_loss=loss)


def independent_complement_check(theta, blocks, delta_c, expected_coefficient_norm, lambda_max, chunk):
    """Independent null-membership test plus amplitude/projection consistency."""
    squared = 0.
    for function, points in blocks:
        push, _ = block_operators(function, len(points))
        for q in range(0, len(points[0]), chunk):
            end = min(q+chunk, len(points[0]))
            jd = np.asarray(push(theta, jnp.asarray(delta_c), *[jnp.asarray(p[q:end]) for p in points]))
            squared += float(jd@jd)
    actual_norm = float(np.linalg.norm(delta_c))
    denominator = float(np.sqrt(lambda_max)*actual_norm)
    absolute = float(np.sqrt(squared))
    norm_defect = abs(actual_norm-expected_coefficient_norm)
    projection_defect = abs(float(np.asarray(theta)@delta_c)+expected_coefficient_norm**2)
    projection_scale = float(np.linalg.norm(theta)*expected_coefficient_norm)
    return dict(delta_c_norm=actual_norm, expected_coefficient_norm=float(expected_coefficient_norm),
                norm_relative_defect=float(norm_defect/expected_coefficient_norm) if expected_coefficient_norm else float(norm_defect),
                projection_absolute_defect=projection_defect,
                projection_scaled_defect=float(projection_defect/projection_scale) if projection_scale else projection_defect,
                null_residual_absolute=absolute, null_residual_denominator=denominator,
                null_residual_scaled=absolute/denominator if denominator else None,
                interpretation="JVP checks null membership, not amplitude/completeness; norm, projection and field checks are separate consistency tests.")


def completion_signature(tag, args):
    """Version and input identity for resuming this fixed scientific protocol."""
    sources=(Path(__file__), HERE/"round2_linalg.py", Path(n1.__file__))
    inputs=[args.inputs/tag/f"{tag}_{suffix}" for suffix in ("params.pkl","map.json","fields.npz")]
    fourier=args.inputs/tag/f"{tag}_fourier_B.npz"
    if fourier.exists():
        inputs.append(fourier)
    inputs.append(args.baseline/f"{tag}_fullgrid_sigma_raw.npz")
    return dict(script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                versions=dict(jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__))


def resume_status(tag, args):
    """Return an existing compatible completion, retaining its numerical status."""
    stem=args.out/f"{tag}_round2"
    marker=stem.with_suffix(".json")
    if not marker.exists():
        return None
    record=json.loads(marker.read_text(encoding="utf-8"))
    companions=[stem.with_suffix(".csv"),Path(str(stem)+"_fields.npz"),Path(str(stem)+"_spectral.npz")]
    if record.get("tag")!=tag or not all(p.is_file() for p in companions):
        raise ValueError(f"Incomplete/mismatched completion record for {tag}; review it or use a new --out.")
    if record.get("completion_signature")!=completion_signature(tag,args):
        raise ValueError(f"Incompatible inputs/scripts/versions for {tag}; review it or use a new --out.")
    if record.get("numerical_status") not in ("PASS","REVIEW_REQUIRED"):
        raise ValueError(f"Unknown numerical status for {tag}; review it before resuming.")
    print(f"RESUMED completed computation: {tag}, numerical_status={record['numerical_status']} (status retained).",flush=True)
    return record


def independent_normal_residuals(theta, blocks, residual, gammas, deltas, chunk, lambda_max):
    """JVP/VJP of the residual functions, independent of all stored QR factors."""
    operators = []
    start = 0
    jtr = np.zeros(len(theta))
    for function, points in blocks:
        push, pull = block_operators(function, len(points))
        operators.append((points, push, pull, start))
        for q in range(0, len(points[0]), chunk):
            end = min(q + chunk, len(points[0]))
            coords = [jnp.asarray(p[q:end]) for p in points]
            jtr += np.asarray(pull(theta, jnp.asarray(residual[start+q:start+end]), *coords))
        start += len(points[0])
    rows = []
    for gamma, delta in zip(gammas, deltas):
        pulled = np.zeros(len(theta))
        direction = jnp.asarray(delta)
        for points, push, pull, start in operators:
            for q in range(0, len(points[0]), chunk):
                end = min(q + chunk, len(points[0]))
                coords = [jnp.asarray(p[q:end]) for p in points]
                jd = push(theta, direction, *coords)
                weights = jnp.asarray(residual[start+q:start+end]) + jd
                pulled += np.asarray(pull(theta, weights, *coords))
        h = pulled + gamma * (np.asarray(theta) + delta)
        delta_norm = float(np.linalg.norm(delta))
        rhs_norm = float(np.linalg.norm(jtr+gamma*np.asarray(theta)))
        denominator = (lambda_max+gamma)*delta_norm+rhs_norm
        cancellation_denominator = np.linalg.norm(jtr) + np.linalg.norm(pulled)
        cancellation_denominator += gamma * (np.linalg.norm(theta) + delta_norm)
        absolute = float(np.linalg.norm(h))
        rows.append(dict(gamma=float(gamma), normal_residual_absolute=absolute,
                         normal_residual_scaled=float(absolute/denominator) if denominator else absolute,
                         normal_residual_cancellation_scaled=float(absolute/cancellation_denominator) if cancellation_denominator else absolute,
                         normal_residual_denominator=float(denominator),
                         normal_residual_cancellation_denominator=float(cancellation_denominator),
                         stationarity_gradient_norm=rhs_norm,lambda_max=float(lambda_max),delta_norm=delta_norm))
        print(f"Independent mean check: gamma={gamma:.6g}, normwise backward error={rows[-1]['normal_residual_scaled']:.3g}", flush=True)
    return rows


def region_masks(gradient, times, pde):
    p70, p95 = np.percentile(gradient, (70, 95))
    original = {"bulk": gradient < p70, "transition": (gradient >= p70) & (gradient < p95),
                "sharp": gradient >= p95}
    masks = {f"full/{r}": m for r, m in original.items()}
    if pde == "burgers":
        masks.update({f"post_formation/{r}": m & (times >= 1/np.pi) for r, m in original.items()})
    return masks, (p70, p95)


def evaluate_grid(theta, u, data, gammas, modes, factors, masks, chunk, delta_c=None):
    packed, tau, rotation, singular, _ = factors
    n = len(singular)
    if delta_c is None:
        delta_c=reconstruct_shift(packed,tau,rotation,np.zeros(n),modes["complement_coefficients"])
    complement_norm=float(np.linalg.norm(delta_c))
    xx, tt = np.meshgrid(data["xs"], data["ts"])
    x, t = xx.ravel(), tt.ravel()
    predict = jax.jit(jax.vmap(u, in_axes=(None, 0, 0)))
    grad = jax.jit(jax.vmap(jax.grad(u), in_axes=(None, 0, 0)))
    variance, shift = np.empty((len(gammas), len(x))), np.empty((len(gammas), len(x)))
    complement_shift, tangent_norm, complement_energy = (np.empty(len(x)) for _ in range(3))
    labels = list(masks)
    energy_sum = np.zeros((len(labels), n))
    comp_sum, norm_sum = np.zeros(len(labels)), np.zeros(len(labels))
    norm_defect, prediction_defect, complement_field_abs, complement_field_scaled = 0., 0., 0., 0.
    weights = 1/(singular[None, :]**2 + gammas[:, None])
    for q in range(0, len(x), chunk):
        end = min(q+chunk, len(x))
        pred = np.asarray(predict(theta, x[q:end], t[q:end]))
        np.testing.assert_allclose(pred, data["u_pred"][q:end], rtol=1e-9, atol=1e-10)
        prediction_defect = max(prediction_defect, float(np.max(np.abs(pred-data["u_pred"][q:end]))))
        g = np.asarray(grad(theta, x[q:end], t[q:end]))
        qt = transform(packed, tau, g.T)
        c = rotation.T @ qt[:n]
        e = c**2
        d = np.sum(qt[n:]**2, axis=0)
        gn = np.sum(g**2, axis=1)
        cs = modes["complement_coefficients"] @ qt[n:]
        field_difference=np.abs(cs-g@delta_c)
        field_scale=np.sqrt(gn)*complement_norm
        complement_field_abs=max(complement_field_abs,float(field_difference.max()))
        scaled=np.divide(field_difference,field_scale,out=field_difference.copy(),where=field_scale>0)
        complement_field_scaled=max(complement_field_scaled,float(scaled.max()))
        variance[:, q:end] = weights @ e + d[None, :]/gammas[:, None]
        shift[:, q:end] = modes["delta_modes"] @ c + cs[None, :]
        complement_shift[q:end], tangent_norm[q:end], complement_energy[q:end] = cs, gn, d
        norm_defect = max(norm_defect, float(np.max(np.abs(np.sum(e, axis=0)+d-gn)/np.maximum(gn, np.finfo(float).tiny))))
        for i, label in enumerate(labels):
            select = masks[label][q:end]
            energy_sum[i] += e[:, select].sum(axis=1)
            comp_sum[i] += d[select].sum()
            norm_sum[i] += gn[select].sum()
        if q % (25*chunk) == 0:
            print(f"Evaluation grid: {end}/{len(x)}", flush=True)
    counts = np.array([masks[label].sum() for label in labels])
    return dict(unit_variance=variance, mean_shift=shift, complement_shift=complement_shift,
                tangent_norm_squared=tangent_norm, complement_energy=complement_energy,
                labels=labels, counts=counts, mean_mode_energy=energy_sum/counts[:, None],
                mean_complement_energy=comp_sum/counts, mean_tangent_norm=norm_sum/counts,
                max_norm_relative_defect=norm_defect, max_prediction_absolute_defect=prediction_defect,
                max_complement_field_absolute_defect=complement_field_abs,
                max_complement_field_scaled_defect=complement_field_scaled)


def diagnostics(singular, z, settings):
    records, terms = [], []
    for choice in settings:
        a, v = choice["alpha"], choice["noise_variance"]
        d = evidence_slope(singular, z, a, v)
        step = 1e-4
        nearby = evidence_grid(singular, z, [a], v*np.exp(np.array([-step, step])))[1]
        slope_fd = float((nearby[0, 1]-nearby[0, 0])/(2*step))
        records.append(dict(setting=choice["setting"], alpha=a, noise_variance=v,
                            effective_precision=a*v, slope=d["slope"], groups=d["groups"],
                            slope_finite_difference=slope_fd, slope_absolute_difference=abs(slope_fd-d["slope"])))
        terms.append(d["mode_slope"])
    return records, np.array(terms)


def scientific_rows(tag, data, choices, gammas, fields, masks):
    rows = []
    for choice in choices:
        k = choice["gamma_index"]
        variance = choice["noise_variance"]*fields["unit_variance"][k]
        sigma = np.sqrt(variance)
        for centre, prediction in (("trained", data["u_pred"]),
                                   ("affine", data["u_pred"]+fields["mean_shift"][k])):
            error = prediction-data["u_exact"]
            for subset in sorted({name.split('/')[0] for name in masks}):
                selected = {name.split('/')[1]: mask for name, mask in masks.items() if name.startswith(subset+'/')}
                error_rms = {r: np.sqrt(np.mean(error[m]**2)) for r, m in selected.items()}
                sigma_rms = {r: np.sqrt(np.mean(variance[m])) for r, m in selected.items()}
                contrast = sigma_rms["sharp"]/sigma_rms["bulk"]
                target = error_rms["sharp"]/error_rms["bulk"]
                for region, mask in selected.items():
                    z = np.abs(error[mask])/sigma[mask]
                    row = dict(tag=tag, **choice, effective_precision=float(gammas[k]), centre=centre,
                               subset=subset, region=region, count=int(mask.sum()),
                               R_sigma_rms=float(contrast), R_sigma_mean=float(np.mean(sigma[selected['sharp']])/np.mean(sigma[selected['bulk']])),
                               R_tar=float(target), rho=float(target/contrast),
                               error_rms=float(error_rms[region]), sigma_rms=float(sigma_rms[region]),
                               mean_sigma=float(np.mean(sigma[mask])), mean_Z=float(np.mean(z)),
                               coverage68=float(np.mean(z<=1)), coverage95=float(np.mean(z<=1.96)),
                               shift_rms=float(np.sqrt(np.mean(fields['mean_shift'][k,mask]**2))),
                               complement_shift_rms=float(np.sqrt(np.mean(fields['complement_shift'][mask]**2))))
                    rows.append(row)
    return rows


def analyse(tag, args):
    start = time.perf_counter()
    signature=completion_signature(tag,args)
    run = args.inputs/tag
    metadata = json.loads((run/f"{tag}_map.json").read_text(encoding="utf-8-sig"))
    config = metadata["provenance"]["config"]
    with (run/f"{tag}_params.pkl").open("rb") as f:
        params = pickle.load(f)
    with np.load(run/f"{tag}_fields.npz") as d:
        data = {k: d[k] for k in ("xs", "ts", "x_u", "t_b", "x_f", "t_f", "u_pred", "u_exact", "grad_abs")}
    for k in ("u_pred", "u_exact", "grad_abs"):
        data[k] = data[k].ravel()
    fourier = None
    if config["arch"] == "fourier":
        with np.load(run/f"{tag}_fourier_B.npz") as d:
            fourier = {k: jnp.asarray(d[k]) for k in ("B_X", "B_T")}
    theta, u = n1.make_model(params, config, fourier)
    blocks = n1.residual_blocks(u, config, data)
    early_checks=early_checkpoint_checks(theta,u,data,blocks,metadata["final_loss"],args.jac_chunk,args.grid_chunk)
    jacobian, residual = n1.build_jacobian(theta, blocks, args.jac_chunk)
    n, p = jacobian.shape
    alpha0 = 1/float(np.mean(np.asarray(theta)**2))
    noise_mle = float(np.mean(residual**2))
    np.testing.assert_allclose(noise_mle/2, metadata["final_loss"], rtol=1e-6, atol=1e-12)
    if noise_mle <= 0:
        raise ValueError("Zero fitted residual variance; no artificial floor is introduced.")
    affine_data = jacobian @ np.asarray(theta)-residual
    print(f"{tag}: P={p}, N={n}, alpha0={alpha0:.8g}, conditional noise={noise_mle:.8g}", flush=True)
    print("One direct-J QR/SVD factorisation...", flush=True)
    factors = factor_jacobian(jacobian)
    del jacobian
    packed, tau, rotation, singular, observation_rotation = factors
    z = observation_rotation @ affine_data
    del affine_data
    alpha_grid = alpha0*np.unique(np.r_[np.logspace(-7, 7, 57), 1.])
    original_noise_grid = np.unique(np.r_[noise_mle*np.logspace(-4, 6, 41), noise_mle, 1.])
    extended_noise_grid = np.unique(np.r_[original_noise_grid, noise_mle*np.logspace(-8, -4, 17)])
    original_evidence, original_surface = evidence_grid(singular, z, alpha_grid, original_noise_grid)
    extended_evidence, extended_surface = evidence_grid(singular, z, alpha_grid, extended_noise_grid)
    choices_selected = [dict(setting="conditional_noise_mle", alpha=alpha0, noise_variance=noise_mle)]
    for label, best in (("affine_evidence_original", original_evidence), ("affine_evidence_extended", extended_evidence)):
        boundary = best["alpha_boundary"] or best["noise_boundary"]
        choices_selected.append(dict(setting=label+("_grid_boundary" if boundary else "_grid_interior"),
                                     alpha=best["alpha"], noise_variance=best["noise_variance"]))
    prep_modes = posterior_modes(np.asarray(theta), residual, *factors, [alpha0])
    stability = evidence_stability(singular, z, prep_modes["theta_modes"], prep_modes["residual_modes"],
                                  alpha_grid, extended_noise_grid, (n,p), choices_selected[1:])
    for case in stability["perturbations"]:
        best = case["best"]
        choices_selected.append(dict(setting="evidence_sensitivity_"+case["case"],
                                     alpha=best["alpha"], noise_variance=best["noise_variance"]))
    unit_alphas = np.unique(np.r_[alpha0*np.logspace(-7,2,19), alpha0,
                                  [c["alpha"]*c["noise_variance"] for c in choices_selected]])
    choices = [dict(setting="prior_sweep", alpha_multiplier=float(a/alpha0), alpha=float(a), noise_variance=1.)
               for a in unit_alphas]
    choices += [dict(c, alpha_multiplier=c["alpha"]/alpha0) for c in choices_selected]
    # Store only distinct gamma fields; exact matched choices use the same index.
    gammas = np.unique([c["alpha"]*c["noise_variance"] for c in choices])
    for choice in choices:
        gamma = choice["alpha"]*choice["noise_variance"]
        choice["gamma_index"] = int(np.flatnonzero(gammas == gamma)[0])
    modes = posterior_modes(np.asarray(theta), residual, *factors, gammas)
    deltas = np.array([reconstruct_shift(packed,tau,rotation,d,modes["complement_coefficients"])
                       for d in modes["delta_modes"]])
    delta_c=reconstruct_shift(packed,tau,rotation,np.zeros(len(singular)),modes["complement_coefficients"])
    times = np.repeat(data["ts"], len(data["xs"]))
    masks, thresholds = region_masks(data["grad_abs"], times, config["pde"])
    fields = evaluate_grid(theta,u,data,gammas,modes,factors,masks,args.grid_chunk,delta_c)
    if not np.isfinite(fields["unit_variance"]).all() or np.any(fields["unit_variance"]<=0):
        raise ValueError("Non-finite/nonpositive predictive variance.")
    if not np.isfinite(fields["mean_shift"]).all():
        raise ValueError("Non-finite predictive mean shift.")
    baseline_path = args.baseline/f"{tag}_fullgrid_sigma_raw.npz"
    with np.load(baseline_path) as d:
        np.testing.assert_array_equal(d["xs"],data["xs"])
        np.testing.assert_array_equal(d["ts"],data["ts"])
        baseline = d["sigma"].ravel()
    operating = next(c for c in choices if c["setting"]=="prior_sweep" and c["alpha_multiplier"]==1.)
    sigma0 = np.sqrt(fields["unit_variance"][operating["gamma_index"]])
    np.testing.assert_allclose(sigma0,baseline,rtol=1e-4,atol=1e-10)
    slope_records, slope_terms = diagnostics(singular,z,choices_selected[1:])
    for choice in choices_selected:
        matched = next(c for c in choices if c["setting"]=="prior_sweep"
                       and np.isclose(c["alpha"],choice["alpha"]*choice["noise_variance"],rtol=1e-14,atol=0))
        actual = np.sqrt(choice["noise_variance"]*fields["unit_variance"][next(c for c in choices if c["setting"]==choice["setting"])["gamma_index"]])
        reference = np.sqrt(choice["noise_variance"])*np.sqrt(fields["unit_variance"][matched["gamma_index"]])
        np.testing.assert_allclose(actual,reference,rtol=1e-10,atol=0)
    rows = scientific_rows(tag,data,choices,gammas,fields,masks)
    # Matrix-free checks do not need the large factorisation retained in memory.
    del factors, packed, tau, rotation, observation_rotation
    lambda_max=float(singular.max()**2)
    normal_checks = independent_normal_residuals(theta,blocks,residual,gammas,deltas,args.jac_chunk,lambda_max)
    complement_checks=independent_complement_check(theta,blocks,delta_c,float(np.linalg.norm(modes["complement_coefficients"])),lambda_max,args.jac_chunk)
    complement_checks["field_absolute_defect"]=fields["max_complement_field_absolute_defect"]
    complement_checks["field_scaled_defect"]=fields["max_complement_field_scaled_defect"]
    comp_values=[complement_checks[k] for k in ("norm_relative_defect","projection_scaled_defect","field_scaled_defect")]
    if complement_checks["null_residual_scaled"] is not None:
        comp_values.append(complement_checks["null_residual_scaled"])
    complement_review=any(not np.isfinite(value) or value>1e-7 for value in comp_values)
    complement_checks["review_required"]=complement_review
    for check in normal_checks:
        check["threshold_exceeded"] = not np.isfinite(check["normal_residual_scaled"]) or check["normal_residual_scaled"]>1e-7
    normal_review = any(check["threshold_exceeded"] for check in normal_checks)
    numerical_review=normal_review or complement_review
    if normal_review:
        print("REVIEW REQUIRED: at least one normwise backward error exceeds 1e-7; retaining the archive and diagnostic values for assessment.", flush=True)
    if complement_review:
        print("REVIEW REQUIRED: complement consistency check exceeds its 1e-7 diagnostic threshold; retaining the archive.",flush=True)
    spectral_limits = []
    for subset in sorted({label.split('/')[0] for label in fields["labels"]}):
        b,s = [fields["labels"].index(subset+'/'+r) for r in ("bulk","sharp")]
        spectral_limits.append(dict(subset=subset,
            strong_gamma_tangent_norm_contrast=float(np.sqrt(fields["mean_tangent_norm"][s]/fields["mean_tangent_norm"][b])),
            qr_complement_contrast=float(np.sqrt(fields["mean_complement_energy"][s]/fields["mean_complement_energy"][b])),
            weak_limit_qualification="QR-complement ratio; whole-kernel weak limit requires full row rank. No rank certificate or global endpoint bound asserted."))
    args.out.mkdir(parents=True,exist_ok=True)
    stem = args.out/f"{tag}_round2"
    with stem.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(str(stem)+"_fields.npz", gammas=gammas,
        unit_variance=fields["unit_variance"], mean_shift=fields["mean_shift"],
        complement_shift=fields["complement_shift"], tangent_norm_squared=fields["tangent_norm_squared"],
        complement_energy=fields["complement_energy"], xs=data["xs"],ts=data["ts"],
        u_pred=data["u_pred"],u_exact=data["u_exact"],grad_abs=data["grad_abs"])
    np.savez_compressed(str(stem)+"_spectral.npz", singular=singular,eigenvalues=singular**2,
        projected_data=z, theta_modes=modes["theta_modes"], residual_modes=modes["residual_modes"],
        alpha0=alpha0,conditional_noise_mle=noise_mle,residual_count=n,parameter_count=p,
        region_labels=np.array(fields["labels"]),region_counts=fields["counts"],
        mean_mode_energy=fields["mean_mode_energy"],mean_complement_energy=fields["mean_complement_energy"],
        mean_tangent_norm=fields["mean_tangent_norm"],evidence_alpha_grid=alpha_grid,
        original_noise_grid=original_noise_grid,extended_noise_grid=extended_noise_grid,
        original_log_evidence=original_surface,extended_log_evidence=extended_surface,
        slope_settings=np.array([r["setting"] for r in slope_records]),mode_evidence_slope=slope_terms)
    summary=dict(tag=tag,config=config,choices=choices,gammas=gammas.tolist(),alpha0=alpha0,
        conditional_noise_mle=noise_mle,saved_loss=metadata["final_loss"],rebuilt_loss=noise_mle/2,lambda_max=lambda_max,
        original_affine_evidence=original_evidence,extended_affine_evidence=extended_evidence,
        evidence_stability=stability,evidence_slopes=slope_records,normal_equation_checks=normal_checks,
        early_checks=early_checks,complement_checks=complement_checks,completion_signature=signature,
        spectral_limits=spectral_limits,region_thresholds=dict(p70=thresholds[0],p95=thresholds[1]),
        max_prediction_absolute_defect=fields["max_prediction_absolute_defect"],
        max_gradient_norm_relative_defect=fields["max_norm_relative_defect"],
        baseline_max_sigma_relative_error=float(np.max(np.abs(sigma0/baseline-1))),
        conventions=dict(training="original raw weighted residuals",prior="absolute zero-mean isotropic Gaussian",
            covariance="v*(J.T J+gamma I)^-1",centres=["trained","affine linearised predictive mean"],
            gamma="alpha*v",field_variance="noise_variance*unit_variance[gamma_index]",
            matched_scale_note="one shared mean field and unit variance per gamma; width scales by sqrt(v)",
            matched_scale_check="identity from shared fields; not independent numerical validation",
            evidence_sensitivity_choices="alternative scales selected under perturbed evidence; fields use the original Jacobian",
            evidence_extension="one fixed four-decade downward extension in noise; no further adaptive search",
            numerical_check="small normal residual is consistency, not weak-mode forward accuracy certification",
            normal_residual_scaled="normwise backward error: ||h||/[(lambda_max+gamma)||delta||+||J.T r+gamma theta||]",
            normal_residual_cancellation_scaled="legacy cancellation-sensitive ratio; retained as a diagnostic, not a review gate",
            no_rank_cutoff=True,no_extra_predictive_noise=True),
        elapsed_seconds=time.perf_counter()-start,versions=dict(jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__),
        numerical_status="REVIEW_REQUIRED" if numerical_review else "PASS",
        normal_equation_checks_passed=not normal_review,complement_checks_passed=not complement_review,
        numerical_review_required=numerical_review,
        normal_residual_review_threshold=1e-7,
        complement_review_threshold=1e-7,jac_chunk=args.jac_chunk,grid_chunk=args.grid_chunk,
        script_hashes=signature["script_hashes"])
    stem.with_suffix(".json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(f"Completed {tag}; {len(gammas)} distinct gammas; {summary['elapsed_seconds']/3600:.2f} h. Saved {stem}.*",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="One fixed-cohort checkpoint; omission processes all 17 serially")
    parser.add_argument("--inputs",type=Path,default=HERE.parent/"n5/input")
    parser.add_argument("--baseline",type=Path,default=HERE/"baseline")
    parser.add_argument("--cohort",type=Path,default=HERE/"cohort.json")
    parser.add_argument("--out",type=Path,default=HERE/"outputs/production")
    parser.add_argument("--jac-chunk",type=int,default=50)
    parser.add_argument("--grid-chunk",type=int,default=200)
    parser.add_argument("--input-check",action="store_true",help="Verify all input files/hashes only; no differentiation")
    parser.add_argument("--resume",action="store_true",help="Reuse compatible completed archives; retain numerical review flags")
    args=parser.parse_args()
    manifest=json.loads(args.cohort.read_text(encoding="utf-8"))
    entries=manifest["cohort"]
    if args.input_check:
        for entry in entries:
            tag=entry["tag"]
            for item in entry["files"]:
                path=Path(item["path"])
                actual=args.baseline/path.name if path.parent.name=="baseline" else args.inputs/tag/path.name
                if hashlib.sha256(actual.read_bytes()).hexdigest()!=item["sha256"]:
                    raise ValueError(f"Input hash mismatch: {actual}")
        print(f"Verified {len(entries)} checkpoints and {sum(len(e['files']) for e in entries)} files. No computation started.")
        return
    tags=[entry["tag"] for entry in entries]
    if args.tag and args.tag not in tags:
        parser.error("Tag is outside the fixed 17-run admitted cohort.")
    if min(args.jac_chunk,args.grid_chunk)<=0:
        parser.error("Chunk sizes must be positive.")
    failures=[]
    for tag in [args.tag] if args.tag else tags:
        try:
            if args.resume and resume_status(tag,args) is not None:
                continue
            analyse(tag,args)
        except Exception as exc:
            print(f"FAILED {tag}: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
            failures.append(tag)
    if failures:
        raise SystemExit("Failed checkpoints: "+", ".join(failures))


if __name__=="__main__":
    main()
