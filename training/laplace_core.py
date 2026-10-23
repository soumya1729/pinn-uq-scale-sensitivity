"""
laplace_core.py -- Gauss-Newton Laplace over an arbitrary parameter SUBSET via
the Woodbury identity. One verified engine for the whole Bayesian scope sweep:
  scope = "last_layer" (P=129) | "last_2_fusion" (16,641) | "all_fusion" (21,889)
          | "full" (196,009)
Var(u(x*)) = (1/alpha)[ ||g_*||^2 - (J~ g_*)^T M^{-1} (J~ g_*) ],  M = alpha I_N + J~ J~^T,
alpha = 1/mean(theta_subset_MAP^2).  (Woodbury variance verified to 1e-14.)
"""
import os
import numpy as np
import jax, jax.numpy as jnp
from jax import vmap, grad
from jax.flatten_util import ravel_pytree
import scipy.linalg as spla
import uq_metrics as M


def _selector(params, scope):
    if scope == "full":
        return jax.tree.map(lambda a: jnp.ones_like(a), params)
    z = jax.tree.map(lambda a: jnp.zeros_like(a), params)
    sel = {"all_fusion": [0, 1, 2], "last_2_fusion": [1, 2], "last_layer": [2]}[scope]
    fus = [{"w": jnp.zeros_like(l["w"]), "b": jnp.zeros_like(l["b"])} for l in params["fusion"]]
    for i in sel:
        fus[i] = {"w": jnp.ones_like(params["fusion"][i]["w"]),
                  "b": jnp.ones_like(params["fusion"][i]["b"])}
    z["fusion"] = fus
    return z


def run(ctx, c, params, scope, JAC_CHUNK=50, PRED_CHUNK=200):
    theta_map, unflatten = ravel_pytree(params)
    mask = np.asarray(ravel_pytree(_selector(params, scope))[0]) > 0.5
    idx = jnp.asarray(np.where(mask)[0])
    sub_map = jnp.asarray(np.asarray(theta_map)[mask])
    n_sub = int(idx.shape[0])

    def make_params(sub):
        return unflatten(theta_map.at[idx].set(sub))

    def u_sub(sub, x, t):
        return ctx.forward(make_params(sub), jnp.array([[x, t]]))[0, 0]

    d = ctx.data
    # IC residual uses ctx.u_ic so it is correct for EVERY PDE (burgers -sin(pi x),
    # allen_cahn x^2 cos(pi x), KS cos(x/16)(1+sin(x/16))). Never hardcode the IC.
    if c.pde == "burgers":
        def r_ic(s, x): return c.ic_weight * (u_sub(s, x, 0.0) - ctx.u_ic(x))
        def r_b1(s, t): return c.bc_weight * u_sub(s, -1.0, t)
        def r_b2(s, t): return c.bc_weight * u_sub(s, 1.0, t)
        def r_f(s, x, t): return ctx.residual_point(make_params(s), x, t)
        BLOCKS = [(grad(r_ic), (d.x_u,)), (grad(r_b1), (d.t_b,)),
                  (grad(r_b2), (d.t_b,)), (grad(r_f), (d.x_f, d.t_f))]
    else:
        def r_ic(s, x): return c.ic_weight * (u_sub(s, x, 0.0) - ctx.u_ic(x))
        def _ux(s, x, t): return grad(lambda xx: u_sub(s, xx, t))(x)
        def r_bu(s, t): return c.bcu_weight * (u_sub(s, -1.0, t) - u_sub(s, 1.0, t))
        def r_bx(s, t): return c.bcx_weight * (_ux(s, -1.0, t) - _ux(s, 1.0, t))
        def r_f(s, x, t): return ctx.residual_point(make_params(s), x, t)
        BLOCKS = [(grad(r_ic), (d.x_u,)), (grad(r_bu), (d.t_b,)),
                  (grad(r_bx), (d.t_b,)), (grad(r_f), (d.x_f, d.t_f))]

    # whitening sigmas from the weighted residual blocks
    r_map = np.asarray(ctx.residual_fn(params)); i = ctx.idx
    segs = ([(0, i.ic), (i.ic, i.bc1), (i.bc1, i.bc2), (i.bc2, i.f)] if c.pde == "burgers"
            else [(0, i.ic), (i.ic, i.bcu), (i.bcu, i.bcx), (i.bcx, i.f)])
    sigs = [max(float(np.mean(r_map[a:b]**2)), 1e-14) ** 0.5 for a, b in segs]
    # RAW_METRIC=1 -> no per-block whitening (J = grad r), i.e. the covariance metric
    # consistent with the raw-trained MAP. This is the global "Arm R" switch: every
    # diagnostic that consumes eng["J"] then works in the raw metric. Default (unset)
    # keeps the per-block whitening (laplace_core's original behaviour).
    if os.environ.get("RAW_METRIC", "0") == "1":
        sigs = [1.0 for _ in segs]

    Jb, sr = [], []
    for (gfn, pts), s in zip(BLOCKS, sigs):
        bg = jax.jit(vmap(gfn, in_axes=(None,) + (0,)*len(pts)))
        rows = []
        for q in range(0, pts[0].shape[0], JAC_CHUNK):
            ch = bg(sub_map, *[p[q:q+JAC_CHUNK] for p in pts]); ch.block_until_ready()
            rows.append(np.asarray(ch))
        Jblk = np.concatenate(rows, 0); Jb.append(Jblk); sr.append(np.full(Jblk.shape[0], s))
    J = np.concatenate(Jb, 0) / np.concatenate(sr)[:, None]      # (N, n_sub) whitened
    N = J.shape[0]
    alpha = 1.0 / max(float(np.mean(np.asarray(sub_map)**2)), 1e-30)
    Mmat = J @ J.T + alpha * np.eye(N); jit_ = 0.0
    for _ in range(6):
        try: L_M = np.linalg.cholesky(Mmat + jit_*np.eye(N)); break
        except np.linalg.LinAlgError: jit_ = max(jit_*10, 1e-10)

    grad_u = jax.jit(vmap(grad(u_sub, argnums=0), in_axes=(None, 0, 0)))

    def predict(xs, ts):
        XX, TT = np.meshgrid(xs, ts); Xr, Tr = XX.ravel(), TT.ravel()
        up = np.asarray(ctx.forward(params, jnp.asarray(np.stack([Xr, Tr], 1)))[:, 0])
        var = np.zeros(Xr.shape[0])
        for q in range(0, Xr.shape[0], PRED_CHUNK):
            e = min(q+PRED_CHUNK, Xr.shape[0])
            g = np.asarray(grad_u(sub_map, jnp.asarray(Xr[q:e]), jnp.asarray(Tr[q:e])))
            qq = J @ g.T
            y = spla.solve_triangular(L_M.T, spla.solve_triangular(L_M, qq, lower=True), lower=False)
            var[q:e] = np.maximum((np.sum(g**2, 1) - np.sum(qq*y, 0)) / alpha, 0.0)
        return up, np.sqrt(var)

    return dict(predict=predict, alpha=float(alpha), n_sub=n_sub,
                woodbury_jitter=float(jit_), N=int(N),
                J=J, sub_map=sub_map, grad_u=grad_u,
                make_params=make_params, loss_sub=lambda sb: ctx.loss_fn(make_params(sb)))


def run_and_dump(scope, method_label, out_suffix):
    """Full pipeline: build context, load shared MAP, run Laplace at `scope`,
    eval on grid, metrics + provenance, write JSON. Used by m1b/m2/m3/m4."""
    import json
    import common as C, provenance as prov
    c = C.get_config(); ctx = C.build_context(c)
    TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
    OUTDIR = os.environ.get("OUTDIR", os.path.join("results", TAG))
    os.makedirs(OUTDIR, exist_ok=True)
    MAP_PKL = os.environ.get("MAP_PKL", os.path.join(OUTDIR, f"{TAG}_params.pkl"))
    JAC_CHUNK = int(os.environ.get("JAC_CHUNK", 50)); PRED_CHUNK = int(os.environ.get("PRED_CHUNK", 200))
    REL = float(os.environ.get("REL_L2_GATE", 0.02))
    params = C.load_map(MAP_PKL)
    print(f"[{out_suffix}] {TAG} scope={scope} | MAP sha1 {C.sha1(MAP_PKL)[:12]}")
    t = prov.Timer()
    with t.phase("laplace_build"):
        eng = run(ctx, c, params, scope, JAC_CHUNK, PRED_CHUNK)
    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1]-xs[0], axis=1)).ravel()
    with t.phase("predict"):
        up, sg = eng["predict"](xs, ts)
    ue = U.ravel()
    regions, p70, p95 = M.assign_regions(ga)
    gate = M.solve_acceptance(up, ue, regions, REL)
    rr = M.region_metrics(up, ue, sg, regions); rat = M.calibration_ratios(rr)
    P = prov.collect(config=C.config_dict(c)); P["timings"] = t.dump()
    P["peak_gpu_mem_mb"] = prov.peak_gpu_mem_mb()
    # Stamp the metric on the record itself. Every sigma in `ratios` depends on whether
    # the Jacobian was per-block whitened, and the two differ by ~4x on the same MAP --
    # 32.18 whitened vs 7.27 raw on allen_cahn_fourier_seed0. RAW_METRIC lives only in
    # provenance.env, where a null reads the same as "not recorded by an older build",
    # so two whitened re-runs (slurm/24 and slurm/25, 2026-07-29/30) went unnoticed
    # until a witness value was compared by hand. A top-level, non-null field cannot be
    # missed and cannot be confused with an absent one.
    sigma_metric = "raw" if os.environ.get("RAW_METRIC", "0") == "1" else "whitened"
    rec = {"method": method_label, "scope": scope, "P_bayes": eng["n_sub"], "tag": TAG,
           "sigma_metric": sigma_metric,
           "map_path": MAP_PKL, "map_sha1": C.sha1(MAP_PKL), "alpha_hat": eng["alpha"],
           "woodbury_jitter": eng["woodbury_jitter"], "solve_gate": gate,
           "regions": rr, "ratios": rat, "provenance": P}
    M.save_json(rec, os.path.join(OUTDIR, f"{TAG}_{out_suffix}_metrics.json"))
    print(f"  P_bayes={eng['n_sub']} alpha={eng['alpha']:.3e}  sigma_metric={sigma_metric.upper()}")
    if rat: print(f"  sigma-ratio={rat['achieved_sigma_ratio']:.2f}x shock meanZ={rat['shock_mean_z']:.2f}")
    print(f"  timings={P['timings']}  wrote {TAG}_{out_suffix}_metrics.json")
    return rec


def tangent_fn(ctx, c, params, scope):
    """Woodbury-free handles for a parameter subset: the output tangent map
    grad_u = d u / d theta_subset, plus make_params and loss_sub. Used by d5
    (manifold reach across scope) and d9 (Laplace-validity loss walk)."""
    theta_map, unflatten = ravel_pytree(params)
    mask = np.asarray(ravel_pytree(_selector(params, scope))[0]) > 0.5
    idx = jnp.asarray(np.where(mask)[0]); sub_map = jnp.asarray(np.asarray(theta_map)[mask])
    def make_params(sub): return unflatten(theta_map.at[idx].set(sub))
    def u_sub(sub, x, t): return ctx.forward(make_params(sub), jnp.array([[x, t]]))[0, 0]
    grad_u = jax.jit(vmap(grad(u_sub, argnums=0), in_axes=(None, 0, 0)))
    return dict(sub_map=sub_map, idx=idx, make_params=make_params, grad_u=grad_u,
                u_sub=u_sub, loss_sub=lambda sb: ctx.loss_fn(make_params(sb)),
                n_sub=int(idx.shape[0]))
