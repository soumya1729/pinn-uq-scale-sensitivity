"""Checks of the shared production algebra on small independent references.

This reads no PINN checkpoint and performs no production Jacobian pass.
The checks include rank deficiency, weak precision and the actual matrix-free
normal-equation implementation in round2_pass.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

import round2_linalg as core


def relative_error(actual, expected):
    return float(np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), 1e-300))


def dense_checks():
    rng = np.random.default_rng(71020261)
    n, p = 9, 15
    gradients = rng.normal(size=(18, p))
    theta, residual = rng.normal(size=p), rng.normal(size=n)
    results = []
    for kind in ("full_row_rank", "duplicate_row", "strongly_correlated", "zero"):
        jacobian = rng.normal(size=(n, p)) / np.sqrt(p)
        if kind == "duplicate_row":
            jacobian[-1] = jacobian[0]
        elif kind == "strongly_correlated":
            jacobian = jacobian[:1] + 1e-7*jacobian
        elif kind == "zero":
            jacobian[:] = 0
        packed, tau, rotation, singular, observation = core.factor_jacobian(jacobian.copy())
        orthogonal = core.transform(packed, tau, np.eye(p), transpose=False)
        basis, complement = orthogonal[:, :n] @ rotation, orthogonal[:, n:]
        reconstruction = (observation.T * singular) @ basis.T
        np.testing.assert_allclose(reconstruction, jacobian, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(orthogonal.T@orthogonal, np.eye(p), rtol=1e-12, atol=1e-12)
        gammas = np.array([1e-2, 1., 100.])
        modes = core.posterior_modes(theta, residual, packed, tau, rotation,
                                     singular, observation, gammas)
        transformed = core.transform(packed, tau, gradients.T)
        coefficients = rotation.T @ transformed[:n]
        complement_energy = np.sum(transformed[n:]**2, axis=0)
        complement_shift = -transformed[n:].T @ modes["theta_q"][n:]
        np.testing.assert_allclose(complement_shift,
            -gradients @ complement @ (complement.T @ theta), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.sum(coefficients**2, axis=0)+complement_energy,
            np.sum(gradients**2, axis=1), rtol=1e-12, atol=1e-12)
        for k, gamma in enumerate(gammas):
            alpha, noise = 2., gamma/2
            shift = core.reconstruct_shift(packed, tau, rotation,
                modes["delta_modes"][k], modes["complement_coefficients"])
            precision = jacobian.T@jacobian + gamma*np.eye(p)
            dense_factor = cho_factor(precision)
            shift_dense = -cho_solve(dense_factor, jacobian.T@residual+gamma*theta)
            mean_dense = cho_solve(dense_factor, jacobian.T@(jacobian@theta-residual))
            covariance = (basis * (noise/(singular**2+gamma))) @ basis.T
            covariance += complement@complement.T/alpha
            covariance_dense = noise*cho_solve(dense_factor, np.eye(p))
            variance = noise*np.sum(coefficients**2/(singular**2+gamma)[:, None], axis=0)
            variance += complement_energy/alpha
            variance_dense = np.einsum("ip,pq,iq->i", gradients, covariance_dense, gradients)
            output_shift = coefficients.T@modes["delta_modes"][k]+complement_shift
            # Full-time regions and a separate post-formation subset retain the same masks.
            masks = {"bulk": np.arange(18)<10, "transition": (np.arange(18)>=10)&(np.arange(18)<15),
                     "sharp": np.arange(18)>=15}
            subset = np.arange(18)%3 != 0
            regional_errors = []
            for mask in list(masks.values()) + [mask&subset for mask in masks.values()]:
                energy = np.mean(coefficients[:, mask]**2, axis=1)
                comp = np.mean(complement_energy[mask])
                regional_variance = noise*np.sum(energy/(singular**2+gamma))+comp/alpha
                regional_errors.append(abs(regional_variance-np.mean(variance_dense[mask])))
            normal = jacobian.T@(residual+jacobian@shift)+gamma*(theta+shift)
            denominator = np.linalg.norm(jacobian.T@residual)+np.linalg.norm(jacobian.T@(residual+jacobian@shift))
            denominator += gamma*(np.linalg.norm(theta)+np.linalg.norm(shift))
            np.testing.assert_allclose(shift, shift_dense, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(theta+shift, mean_dense, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(covariance, covariance_dense, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(variance, variance_dense, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(output_shift, gradients@shift_dense, rtol=1e-9, atol=1e-9)
            # A matched pair uses the same gamma, centre and contrast, with sevenfold variance.
            matched_variance = noise*7*np.sum(coefficients**2/(singular**2+gamma)[:, None], axis=0)
            matched_variance += complement_energy/(alpha/7)
            np.testing.assert_allclose(matched_variance, 7*variance, rtol=1e-13, atol=1e-13)
            results.append(dict(case=kind, gamma=float(gamma),
                shift_relative_error=relative_error(shift, shift_dense),
                covariance_relative_error=relative_error(covariance, covariance_dense),
                output_shift_max_absolute_error=float(np.max(np.abs(output_shift-gradients@shift_dense))),
                regional_variance_max_absolute_error=float(max(regional_errors)),
                normal_residual_absolute=float(np.linalg.norm(normal)),
                normal_residual_scaled=float(np.linalg.norm(normal)/denominator)))
        affine_data = jacobian@theta-residual
        projected = observation@affine_data
        aa, vv = np.array([.1, 2., 10.]), np.array([.03, .3, 3.])
        best, surface = core.evidence_grid(singular, projected, aa, vv)
        dense = np.empty_like(surface)
        for i, alpha in enumerate(aa):
            for j, noise in enumerate(vv):
                covariance = noise*np.eye(n)+jacobian@jacobian.T/alpha
                dense_factor = cho_factor(covariance)
                dense[i,j] = -.5*(n*np.log(2*np.pi)+2*np.log(np.diag(dense_factor[0])).sum()
                                  +affine_data@cho_solve(dense_factor, affine_data))
        np.testing.assert_allclose(surface, dense, rtol=1e-10, atol=1e-10)
        slope = core.evidence_slope(singular, projected, 2., .3)
        step = 1e-5
        nearby = core.evidence_grid(singular, projected, [2.], .3*np.exp([-step,step]))[1]
        finite_difference = float((nearby[0,1]-nearby[0,0])/(2*step))
        np.testing.assert_allclose(slope["slope"], finite_difference, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(sum(row["slope"] for row in slope["groups"]), slope["slope"],
                                   rtol=1e-13, atol=1e-13)
        results.append(dict(case=kind, evidence_max_absolute_error=float(np.max(np.abs(surface-dense))),
            slope_finite_difference_absolute_error=abs(slope["slope"]-finite_difference),
            selected_pair=[best["alpha_index"], best["noise_index"]]))
    return results


def prescribed_spectrum_checks():
    """Independent singular-vector reference where dense normal equations are unsuitable."""
    rng = np.random.default_rng(71020262)
    p, n = 16, 8
    q, _ = np.linalg.qr(rng.normal(size=(p,p)))
    left, _ = np.linalg.qr(rng.normal(size=(n,n)))
    s = np.array([100., 10., 1., .01, 1e-5, 1e-7, 0., 0.])
    jacobian = (left*s)@q[:, :n].T
    theta, residual = rng.normal(size=p), rng.normal(size=n)
    gradients = rng.normal(size=(13,p))
    packed, tau, rotation, singular, observation = core.factor_jacobian(jacobian.copy())
    gammas = np.array([1e-10, 1e-6, 1., 1e4])
    modes = core.posterior_modes(theta, residual, packed, tau, rotation, singular, observation, gammas)
    gradient_coords, theta_coords, residual_coords = gradients@q, q.T@theta, left.T@residual
    transformed = core.transform(packed, tau, gradients.T)
    coefficients = rotation.T@transformed[:n]
    comp_energy = np.sum(transformed[n:]**2, axis=0)
    rows = []
    for i,gamma in enumerate(gammas):
        alpha, noise = 2., gamma/2
        variance = noise*np.sum(coefficients**2/(singular**2+gamma)[:,None], axis=0)+comp_energy/alpha
        expected = noise*np.sum(gradient_coords[:, :n]**2/(s**2+gamma), axis=1)
        expected += np.sum(gradient_coords[:, n:]**2, axis=1)/alpha
        shift_coords = np.concatenate((-(s*residual_coords+gamma*theta_coords[:n])/(s**2+gamma),
                                       -theta_coords[n:]))
        shift = core.reconstruct_shift(packed,tau,rotation,modes["delta_modes"][i],
                                       modes["complement_coefficients"])
        # The weakest mode shifts can be very large: use a prescribed spectral reference.
        np.testing.assert_allclose(variance, expected, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(shift, q@shift_coords, rtol=1e-6, atol=1e-5)
        rows.append(dict(gamma=float(gamma), variance_max_relative_error=float(np.max(np.abs(variance/expected-1))),
                         shift_relative_error=relative_error(shift,q@shift_coords)))
    return rows


def matrix_free_checks():
    """Actual production JVP/VJP path against an explicit nonlinear toy Jacobian."""
    import jax
    import jax.numpy as jnp
    import round2_pass as production
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(71020263)
    theta = jnp.asarray(rng.normal(scale=.2,size=12))
    a,b,c = (jnp.asarray(rng.normal(size=12)) for _ in range(3))
    def first(vector,x):
        return 2*jnp.tanh(a@vector+x)+.2*vector[0]*vector[1]
    def second(vector,t):
        return jnp.sin(b@vector+t)+vector[2]**2
    def third(vector,x,t):
        return jnp.tanh(c@vector+x*t)+jnp.sin(vector[3]*x)+vector[4]*t**2
    blocks = [(first,(np.array([-.8,.3]),)), (second,(np.array([.2,.7]),)),
              (third,(np.array([-.4,.1,.6]),np.array([.2,.4,.9])))]
    def vector_residual(vector):
        return jnp.concatenate([jax.vmap(fun,in_axes=(None,)+(0,)*len(points))(vector,*points)
                                for fun,points in blocks])
    residual = np.asarray(vector_residual(theta))
    jacobian = np.asarray(jax.jacrev(vector_residual)(theta))
    factors = core.factor_jacobian(jacobian.copy())
    gammas = np.array([1e-3,.2,20.])
    modes = core.posterior_modes(np.asarray(theta),residual,*factors,gammas)
    shifts = np.array([core.reconstruct_shift(factors[0],factors[1],factors[2],row,
                       modes["complement_coefficients"]) for row in modes["delta_modes"]])
    lambda_max=float(np.linalg.norm(jacobian,2)**2)
    checks = production.independent_normal_residuals(theta,blocks,residual,gammas,shifts,2,lambda_max)
    delta_c=core.reconstruct_shift(factors[0],factors[1],factors[2],np.zeros(len(residual)),
                                   modes["complement_coefficients"])
    coefficient_norm=float(np.linalg.norm(modes["complement_coefficients"]))
    complement_check=production.independent_complement_check(theta,blocks,delta_c,
                                                              coefficient_norm,lambda_max,2)
    assert max(complement_check[k] for k in
               ("norm_relative_defect","projection_scaled_defect","null_residual_scaled"))<1e-10
    doubled_complement=production.independent_complement_check(theta,blocks,2*delta_c,
                                                               coefficient_norm,lambda_max,2)
    missing_complement=production.independent_complement_check(theta,blocks,np.zeros(len(theta)),
                                                               coefficient_norm,lambda_max,2)
    # Null membership cannot detect a wrong amplitude; separate identities must do so.
    assert doubled_complement["null_residual_scaled"]<1e-10
    np.testing.assert_allclose(doubled_complement["norm_relative_defect"],1.,rtol=1e-12,atol=1e-12)
    assert doubled_complement["projection_scaled_defect"]>1e-2
    assert missing_complement["null_residual_scaled"] is None
    assert missing_complement["norm_relative_defect"]==1.
    expected = []
    for gamma,shift in zip(gammas,shifts):
        normal = jacobian.T@(residual+jacobian@shift)+gamma*(np.asarray(theta)+shift)
        expected.append(float(np.linalg.norm(normal)))
    # The callback's norm agrees up to rounding; meaningful tolerance is scaled below.
    actual_scaled = np.array([r["normal_residual_scaled"] for r in checks])
    assert np.max(actual_scaled)<1e-10, checks
    wrong = shifts.copy()
    wrong[0,0]+=.1
    failed = production.independent_normal_residuals(theta,blocks,residual,gammas,wrong,2,lambda_max)
    assert failed[0]["normal_residual_scaled"]>1e-5, failed[0]
    wrong_normal=jacobian.T@(residual+jacobian@wrong[0])+gammas[0]*(np.asarray(theta)+wrong[0])
    np.testing.assert_allclose(failed[0]["normal_residual_absolute"],np.linalg.norm(wrong_normal),
                               rtol=1e-12,atol=1e-12)
    # Exercise the actual signed-field and regional-archive path on an uneven grid chunk.
    def output(vector,x,t):
        return (jnp.sin(vector[0]*x)+jnp.cos(vector[1]*t)+jnp.tanh(vector[2]*x*t)
                +vector[3]*x+vector[4]*t+vector[5]*vector[6]+vector[7]*x*x
                +vector[8]*t*t+vector[9]*x*t+vector[10]+vector[11]*jnp.sin(x+t))
    xs,ts = np.linspace(-1,1,6),np.array([0.,.35,.8])
    xx,tt = np.meshgrid(xs,ts)
    x,t = xx.ravel(),tt.ravel()
    prediction = np.asarray(jax.vmap(output,in_axes=(None,0,0))(theta,x,t))
    gradients = np.asarray(jax.vmap(jax.grad(output),in_axes=(None,0,0))(theta,x,t))
    data = dict(xs=xs,ts=ts,u_pred=prediction,
                u_exact=prediction+.03*np.sin(x)+.01*t+.02,grad_abs=np.arange(len(x),dtype=float))
    masks,_ = production.region_masks(data["grad_abs"],t,"burgers")
    fields = production.evaluate_grid(theta,output,data,gammas,modes,factors,masks,5,delta_c)
    assert fields["max_complement_field_scaled_defect"]<1e-10
    inconsistent_fields = production.evaluate_grid(theta,output,data,gammas,modes,factors,masks,5,2*delta_c)
    assert inconsistent_fields["max_complement_field_scaled_defect"]>1e-2
    np.testing.assert_array_equal(inconsistent_fields["complement_shift"],fields["complement_shift"])
    orthogonal = core.transform(factors[0],factors[1],np.eye(len(theta)),transpose=False)
    complement = orthogonal[:,len(residual):]
    comp_gradient = gradients@complement
    np.testing.assert_allclose(fields["complement_shift"],-comp_gradient@(complement.T@theta),
                               rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(fields["complement_energy"],np.sum(comp_gradient**2,axis=1),
                               rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(fields["tangent_norm_squared"],np.sum(gradients**2,axis=1),
                               rtol=1e-11,atol=1e-11)
    dense_unit_variance = []
    for gamma in gammas:
        dense = np.linalg.solve(jacobian.T@jacobian+gamma*np.eye(len(theta)),np.eye(len(theta)))
        dense_unit_variance.append(np.einsum("ip,pq,iq->i",gradients,dense,gradients))
    np.testing.assert_allclose(fields["unit_variance"],dense_unit_variance,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(fields["mean_shift"],shifts@gradients.T,rtol=1e-11,atol=1e-11)
    coefficients = factors[2].T@core.transform(factors[0],factors[1],gradients.T)[:len(residual)]
    for k,label in enumerate(fields["labels"]):
        mask = masks[label]
        assert fields["counts"][k]==mask.sum()
        np.testing.assert_allclose(fields["mean_mode_energy"][k],np.mean(coefficients[:,mask]**2,axis=1),
                                   rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(fields["mean_complement_energy"][k],np.mean(np.sum(comp_gradient[mask]**2,axis=1)),
                                   rtol=1e-11,atol=1e-11)
    choices=[dict(setting="synthetic",alpha=float(g),noise_variance=1.,gamma_index=k)
             for k,g in enumerate(gammas)]
    metric_rows=production.scientific_rows("synthetic",data,choices,gammas,fields,masks)
    assert len(metric_rows)==len(gammas)*2*len(masks)
    for row in metric_rows:
        mask=masks[row["subset"]+"/"+row["region"]]
        pred=prediction if row["centre"]=="trained" else prediction+fields["mean_shift"][row["gamma_index"]]
        expected_rms=float(np.sqrt(np.mean((pred[mask]-data["u_exact"][mask])**2)))
        np.testing.assert_allclose(row["error_rms"],expected_rms,rtol=1e-13,atol=1e-13)
    return dict(nonlinear_residual_count=len(residual),parameter_count=len(theta),
        exact_dense_residual_norms=expected,production_checks=checks,
        deliberately_inconsistent_shift_scaled_residual=failed[0]["normal_residual_scaled"],
        complement_check=complement_check,
        doubled_complement_check=doubled_complement,
        omitted_complement_check=missing_complement,
        correct_complement_field_scaled_defect=fields["max_complement_field_scaled_defect"],
        doubled_complement_field_scaled_defect=inconsistent_fields["max_complement_field_scaled_defect"],
        production_grid_check=dict(point_count=len(x),region_labels=fields["labels"],
            point_counts=fields["counts"].tolist(),metric_row_count=len(metric_rows),
            variance_max_relative_error=float(np.max(np.abs(fields["unit_variance"]/dense_unit_variance-1)))),
        scope="small nonlinear toy; operator path does not use QR/SVD factors")


def end_to_end_check():
    """Actual analyse orchestration and four archives, with a tiny synthetic model."""
    import jax
    import jax.numpy as jnp
    import round2_pass as production
    jax.config.update("jax_enable_x64",True)
    here=Path(__file__).resolve().parent
    root=here/"outputs"/"preparation"/"end_to_end"
    tag="synthetic_fixture"
    run=root/"input"/tag
    baseline=root/"baseline"
    run.mkdir(parents=True,exist_ok=True)
    baseline.mkdir(parents=True,exist_ok=True)
    theta=jnp.array([.2,-.1,.3,.15,-.2,.4,.1,-.3,.25,.05])
    def output(vector,x,t):
        basis=jnp.array([1.,x,t,x*t,x*x,t*t,jnp.sin(x),jnp.cos(t),jnp.sin(x*t),jnp.cos(x*t)])
        return vector@basis
    def initial(vector,x):
        return output(vector,x,0.)-.1*jnp.cos(x)
    def boundary(vector,t):
        return output(vector,-1.,t)+output(vector,1.,t)-.2
    direction=jnp.linspace(-.3,.6,len(theta))
    def equation(vector,x,t):
        return jnp.sin(direction@vector+x)-jnp.cos(t)
    xs,ts=np.linspace(-1.,1.,6),np.array([0.,.35,.8])
    xx,tt=np.meshgrid(xs,ts)
    pred=np.asarray(jax.vmap(output,in_axes=(None,0,0))(theta,xx.ravel(),tt.ravel()))
    data=dict(xs=xs,ts=ts,x_u=np.array([-.8,.4]),t_b=np.array([.1,.7]),
              x_f=np.array([-.5,.5]),t_f=np.array([.25,.8]),
              u_pred=pred.reshape(xx.shape),u_exact=(pred+.03*np.sin(xx.ravel())+.01*tt.ravel()+.02).reshape(xx.shape),
              grad_abs=np.arange(xx.size,dtype=float).reshape(xx.shape))
    blocks=[(initial,(data["x_u"],)),(boundary,(data["t_b"],)),
            (equation,(data["x_f"],data["t_f"]))]
    def vector_residual(vector):
        return jnp.concatenate([jax.vmap(fun,in_axes=(None,)+(0,)*len(points))(vector,*points)
                                for fun,points in blocks])
    residual=np.asarray(vector_residual(theta))
    jacobian=np.asarray(jax.jacrev(vector_residual)(theta))
    gradients=np.asarray(jax.vmap(jax.grad(output),in_axes=(None,0,0))(theta,xx.ravel(),tt.ravel()))
    alpha0=1/float(np.mean(np.asarray(theta)**2))
    covariance=np.linalg.solve(jacobian.T@jacobian+alpha0*np.eye(len(theta)),np.eye(len(theta)))
    sigma=np.sqrt(np.einsum("ip,pq,iq->i",gradients,covariance,gradients))
    config=dict(pde="burgers",arch="mlp",synthetic_fixture=True)
    metadata=dict(provenance=dict(config=config),final_loss=float(np.mean(residual**2)/2))
    (run/f"{tag}_map.json").write_text(json.dumps(metadata),encoding="utf-8")
    with (run/f"{tag}_params.pkl").open("wb") as handle:
        pickle.dump(dict(theta=np.asarray(theta)),handle)
    np.savez_compressed(run/f"{tag}_fields.npz",**data)
    np.savez_compressed(baseline/f"{tag}_fullgrid_sigma_raw.npz",sigma=sigma.reshape(xx.shape),xs=xs,ts=ts)
    args=SimpleNamespace(inputs=root/"input",baseline=baseline,out=root/"output",jac_chunk=3,grid_chunk=5)
    original_model,original_residual_blocks=production.n1.make_model,production.n1.residual_blocks
    try:
        production.n1.make_model=lambda params,config,fourier=None:(theta,output)
        production.n1.residual_blocks=lambda u,config,data:blocks
        production.analyse(tag,args)
    finally:
        production.n1.make_model,production.n1.residual_blocks=original_model,original_residual_blocks
    summary=json.loads((args.out/f"{tag}_round2.json").read_text(encoding="utf-8"))
    names=[f"{tag}_round2{suffix}" for suffix in (".csv",".json","_fields.npz","_spectral.npz")]
    assert all((args.out/name).is_file() for name in names)
    with np.load(args.out/f"{tag}_round2_fields.npz") as fields:
        assert fields["unit_variance"].shape==(len(summary["gammas"]),xx.size)
        assert fields["mean_shift"].shape==fields["unit_variance"].shape
        assert fields["complement_shift"].shape==(xx.size,)
    with np.load(args.out/f"{tag}_round2_spectral.npz") as spectral:
        assert spectral["mean_mode_energy"].shape==(6,len(residual))
        assert len(spectral["region_labels"])==6
        assert spectral["extended_noise_grid"].min()<spectral["original_noise_grid"].min()
        assert len(spectral["projected_data"])==len(residual)
    aliases=0
    for choice in summary["choices"]:
        if choice["setting"]!="prior_sweep":
            matched=next(c for c in summary["choices"] if c["setting"]=="prior_sweep"
                         and c["alpha"]==choice["alpha"]*choice["noise_variance"])
            assert matched["gamma_index"]==choice["gamma_index"]
            aliases+=1
    assert summary["baseline_max_sigma_relative_error"]<1e-10
    assert len(summary["normal_equation_checks"])==len(summary["gammas"])
    assert summary["numerical_review_required"]==(summary["numerical_status"]=="REVIEW_REQUIRED")
    assert summary["normal_equation_checks_passed"]==(not any(c["threshold_exceeded"] for c in summary["normal_equation_checks"]))
    assert summary["complement_checks_passed"]==(not summary["complement_checks"]["review_required"])
    np.testing.assert_allclose(summary["early_checks"]["residual_only_loss"],metadata["final_loss"],rtol=1e-12,atol=1e-12)
    resume_checks=resume_checks_for_fixture(production,tag,args,summary)
    early_failures=early_failures_for_fixture(production,tag,args,metadata,data,theta,output,blocks)
    return dict(output_files=names,output_dir=str(args.out.relative_to(here)),
                parameter_count=len(theta),residual_count=len(residual),grid_point_count=xx.size,
                distinct_gammas=len(summary["gammas"]),matched_aliases_checked=aliases,
                numerical_status=summary["numerical_status"],
                early_checks=summary["early_checks"],early_failures=early_failures,resume_checks=resume_checks,
                complement_checks=summary["complement_checks"],
                baseline_max_sigma_relative_error=summary["baseline_max_sigma_relative_error"],
                scope="synthetic model hooks only; actual Jacobian, analyse, evidence, mean, fields, normal checks and file writes")


def resume_checks_for_fixture(production,tag,args,summary):
    """Completion is reusable only with compatible identity and all companions."""
    stem=args.out/f"{tag}_round2"
    marker=stem.with_suffix(".json")
    original=marker.read_bytes()
    assert production.resume_status(tag,args)["numerical_status"]==summary["numerical_status"]
    cases=[]
    try:
        review=json.loads(original)
        review["numerical_status"]="REVIEW_REQUIRED"
        review["numerical_review_required"]=True
        marker.write_text(json.dumps(review),encoding="utf-8")
        review_bytes=marker.read_bytes()
        assert production.resume_status(tag,args)["numerical_status"]=="REVIEW_REQUIRED"
        assert marker.read_bytes()==review_bytes
        cases.append("review_required_retained_without_rewrite")
        invalid=json.loads(original)
        invalid["completion_signature"]["script_hashes"]["round2_pass.py"]="different-source"
        marker.write_text(json.dumps(invalid),encoding="utf-8")
        invalid_bytes=marker.read_bytes()
        try:
            production.resume_status(tag,args)
        except ValueError as exc:
            assert "Incompatible" in str(exc)
        else:
            raise AssertionError("Incompatible source record was reused")
        assert marker.read_bytes()==invalid_bytes
        cases.append("incompatible_record_refused_without_rewrite")
    finally:
        marker.write_bytes(original)
    companion=Path(str(stem)+"_fields.npz")
    holding=companion.with_name(companion.name+".held_by_integration_check")
    companion.rename(holding)
    try:
        try:
            production.resume_status(tag,args)
        except ValueError as exc:
            assert "Incomplete" in str(exc)
        else:
            raise AssertionError("Completion with missing companion was reused")
        assert marker.read_bytes()==original
        cases.append("missing_companion_refused_without_rewrite")
    finally:
        holding.rename(companion)
    missing=SimpleNamespace(**vars(args))
    missing.out=args.out/"no_completed_marker"
    assert production.resume_status(tag,missing) is None
    cases.append("absent_marker_requires_computation")
    return dict(compatible_status=summary["numerical_status"],cases=cases)


def early_failures_for_fixture(production,tag,args,metadata,data,theta,output,blocks):
    """Prediction and loss mismatches abort analyse before allocating its Jacobian."""
    run=args.inputs/tag
    map_path=run/f"{tag}_map.json"
    fields_path=run/f"{tag}_fields.npz"
    original_map,original_fields=map_path.read_bytes(),fields_path.read_bytes()
    original_hooks=(production.n1.make_model,production.n1.residual_blocks,production.n1.build_jacobian)
    jacobian_calls=[]
    def forbidden_jacobian(*a,**k):
        jacobian_calls.append(True)
        raise RuntimeError("Jacobian must not be constructed for an early mismatch")
    cases=[]
    try:
        production.n1.make_model=lambda params,config,fourier=None:(theta,output)
        production.n1.residual_blocks=lambda u,config,data:blocks
        production.n1.build_jacobian=forbidden_jacobian
        wrong_data=dict(data)
        wrong_data["u_pred"]=data["u_pred"].copy()
        wrong_data["u_pred"].flat[0]+=.1
        np.savez_compressed(fields_path,**wrong_data)
        try:
            production.analyse(tag,args)
        except AssertionError:
            pass
        else:
            raise AssertionError("Prediction mismatch did not stop analyse")
        assert not jacobian_calls
        cases.append("prediction_mismatch_before_jacobian")
        fields_path.write_bytes(original_fields)
        wrong_metadata=dict(metadata,final_loss=metadata["final_loss"]+.1)
        map_path.write_text(json.dumps(wrong_metadata),encoding="utf-8")
        try:
            production.analyse(tag,args)
        except AssertionError:
            pass
        else:
            raise AssertionError("Loss mismatch did not stop analyse")
        assert not jacobian_calls
        cases.append("loss_mismatch_before_jacobian")
    finally:
        map_path.write_bytes(original_map)
        fields_path.write_bytes(original_fields)
        production.n1.make_model,production.n1.residual_blocks,production.n1.build_jacobian=original_hooks
    return dict(cases=cases,jacobian_calls=len(jacobian_calls))


def failure_continuation_check():
    """A failed checkpoint does not prevent later entries; overall exit remains failed."""
    import round2_pass as production
    here=Path(__file__).resolve().parent
    cohort=here/"outputs"/"preparation"/"end_to_end"/"failure_cohort.json"
    cohort.write_text(json.dumps(dict(cohort=[dict(tag=tag) for tag in ("fails","later_a","later_b")])),encoding="utf-8")
    calls=[]
    def fake_analyse(tag,args):
        calls.append(tag)
        if tag=="fails":
            raise ValueError("deliberate synthetic failure")
    original_analyse,original_argv=production.analyse,sys.argv
    try:
        production.analyse=fake_analyse
        sys.argv=["round2_pass.py","--cohort",str(cohort)]
        try:
            production.main()
        except SystemExit as exc:
            assert exc.code and "fails" in str(exc.code)
            exit_message=str(exc.code)
        else:
            raise AssertionError("Failed cohort incorrectly exited successfully")
    finally:
        production.analyse,sys.argv=original_analyse,original_argv
    assert calls==["fails","later_a","later_b"],calls
    return dict(processed_tags=calls,exit_message=exit_message,
                scope="CLI orchestration with synthetic callback; no scientific computation")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"outputs"/"validation"/"integration_summary.json")
    args=parser.parse_args()
    report=dict(status="PASS",dense_checks=dense_checks(),prescribed_spectrum=prescribed_spectrum_checks(),
                matrix_free=matrix_free_checks(),end_to_end=end_to_end_check(),
                failure_continuation=failure_continuation_check(),
                scope="synthetic integration checks only; no PINN production pass")
    report["versions"]=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    here=Path(__file__).resolve().parent
    report["source_sha256"]={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                             for name in ("integration_checks.py","round2_linalg.py","round2_pass.py")}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
