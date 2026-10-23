"""
train_map.py -- the MAP producer for the unified suite.

Trains ONE network (Adam warm-up -> Levenberg-Marquardt) for the configured PDE,
computes the last-layer Laplace BLL (method #1, free here), and saves:
  {TAG}_params.pkl   - trained weights (shared MAP for all other methods)
  {TAG}_map.json     - MAP + last-layer-BLL metrics + full provenance + timings

Config via env (see common.get_config). Architecture/PDE/data all come from
common.build_context, so every downstream method reconstructs the identical
model + collocation points and loads this exact MAP.
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
import optax

import common as C
import provenance as prov
import uq_metrics as M

c = C.get_config()
ctx = C.build_context(c)
TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
OUTDIR = os.environ.get("OUTDIR", os.path.join("results", "_smoke", TAG) if c.smoke else os.path.join("results", TAG))
os.makedirs(OUTDIR, exist_ok=True)   # SMOKE never clobbers the canonical results/{TAG}/ MAP
REL_L2_GATE = float(os.environ.get("REL_L2_GATE", 0.02))

forward, extract_features = ctx.forward, ctx.extract_features
d = ctx.data
N_TOTAL = ctx.N_TOTAL

# --- training metric: raw (default) | adaptive | frozen -------------------------
# raw      : minimise the fixed-weight composite loss (the paper's existing MAP).
# adaptive : additionally divide each block by its running RMS (stop-grad) -> the
#            self-consistent balanced/whitened metric (the MAP IS the whitened-loss
#            minimum, so the whitened Laplace posterior is well-posed).
# frozen   : like adaptive, but block-RMS frozen from a supplied raw MAP
#            (FROZEN_SIG_MAP) -- the two-stage approximation.
WHITEN_TRAIN = os.environ.get("WHITEN_TRAIN", "raw").lower()
if WHITEN_TRAIN == "raw":
    residual_fn = ctx.residual_fn
    loss_fn = jax.jit(ctx.loss_fn)
elif WHITEN_TRAIN == "adaptive":
    residual_fn = lambda p: ctx.residual_fn_white(p)             # adaptive stop-grad RMS
    loss_fn = jax.jit(lambda p: ctx.loss_fn_white(p))
elif WHITEN_TRAIN == "frozen":
    _sig_src = os.environ.get("FROZEN_SIG_MAP")
    assert _sig_src, "WHITEN_TRAIN=frozen needs FROZEN_SIG_MAP=<raw {TAG}_params.pkl>"
    _sig_frozen = jax.lax.stop_gradient(ctx.block_sigma(ctx.residual_fn(C.load_map(_sig_src))))
    residual_fn = lambda p: ctx.residual_fn_white(p, _sig_frozen)
    loss_fn = jax.jit(lambda p: ctx.loss_fn_white(p, _sig_frozen))
    print(f"[{TAG}] frozen whitening sigmas from {_sig_src}")
else:
    raise SystemExit(f"WHITEN_TRAIN={WHITEN_TRAIN!r} not in raw|adaptive|frozen")
print(f"[{TAG}] WHITEN_TRAIN={WHITEN_TRAIN}")

print(f"[{TAG}] pde={c.pde} arch={c.arch} nu={c.nu:.5g} eps2={c.eps2:.3g} "
      f"fourier_scale={c.fourier_scale if c.arch=='fourier' else '-'} smoke={c.smoke}")
print("Devices:", jax.devices())

# ---- optimisers ----
opt = optax.adam(c.adam_lr)
key, sk = jax.random.split(ctx.key)
params = ctx.init_pinn(sk)
opt_state = opt.init(params)
print("Total params:", int(ravel_pytree(params)[0].shape[0]), "N_TOTAL:", N_TOTAL)


@jax.jit
def adam_step(p, st):
    l, g = jax.value_and_grad(loss_fn)(p)
    u, st = opt.update(g, st)
    return optax.apply_updates(p, u), st, l, optax.global_norm(g)   # +grad norm (log only)


def Jv(p, v):  return jax.jvp(residual_fn, (p,), (v,))[1]
def JTv(p, v): return jax.vjp(residual_fn, p)[1](v)[0]


@jax.jit
def lm_step(p, mu):
    r = residual_fn(p); JTr, unr = ravel_pytree(JTv(p, r)); b = -JTr / N_TOTAL
    def A(vf):
        v = unr(vf); jtjv, _ = ravel_pytree(JTv(p, Jv(p, v))); return jtjv/N_TOTAL + mu*vf
    df, _ = cg(A, b, maxiter=c.lm_cg); dd = unr(df)
    return dd, 0.5*jnp.dot(df, mu*df + b)


def train(t):
    # [15thJuly] instrumented: records a full per-iteration log ALONGSIDE the optimisation.
    # The optimisation path (steps, acceptance, mu/rho updates, RNG) is byte-for-byte the
    # original; everything added is read-only logging.
    global params, opt_state
    from time import perf_counter as _pc
    loss_hist = []
    la = {"iter": [], "loss": [], "grad_norm": [], "wall_s": []}
    ll = {"iter": [], "loss": [], "mu": [], "rho": [], "pred_red": [], "act_red": [],
          "n_reject": [], "step_norm": [], "accepted": [], "wall_s": []}
    with t.phase("adam"):
        for i in range(c.adam_iters):
            _t0 = _pc()
            params, opt_state, l, gn = adam_step(params, opt_state)
            lv = float(l)
            la["iter"].append(i); la["loss"].append(lv); la["grad_norm"].append(float(gn))
            la["wall_s"].append(_pc() - _t0)
            if i % max(1, c.adam_iters//10) == 0: print(f"  Adam {i:5d} | loss {lv:.3e} | gnorm {float(gn):.2e}")
            loss_hist.append(lv)
    with t.phase("lm"):
        mu = c.lm_mu_init; lc = float(loss_fn(params))
        for i in range(c.lm_iters):
            _t0 = _pc(); nrej = 0
            rho = pv = ar = sn = 0.0; mu_acc = mu; accepted = False
            while nrej < c.lm_max_rej:
                dd, pred = lm_step(params, mu)
                pn = jax.tree.map(lambda a, b: a+b, params, dd)
                ln = float(loss_fn(pn)); ar = lc-ln; pv = float(pred)
                rho = (1.0 if ar > 0 else 0.0) if abs(pv) < 1e-18 else ar/pv
                if rho > c.lm_rho_accept:
                    sn = float(optax.global_norm(dd)); mu_acc = mu; accepted = True
                    params = pn; lc = ln
                    if rho > c.lm_rho_good: mu = max(mu*c.lm_mu_dec, c.lm_mu_min)
                    if i % max(1, c.lm_iters//10) == 0: print(f"  LM {i:4d} | loss {ln:.3e} | mu {mu_acc:.1e} | rho {rho:+.2f} | rej {nrej}")
                    break
                mu *= c.lm_mu_inc; nrej += 1
            ll["iter"].append(i); ll["loss"].append(lc); ll["mu"].append(mu_acc); ll["rho"].append(rho)
            ll["pred_red"].append(pv); ll["act_red"].append(ar); ll["n_reject"].append(nrej)
            ll["step_norm"].append(sn); ll["accepted"].append(bool(accepted)); ll["wall_s"].append(_pc() - _t0)
            loss_hist.append(lc)
    print(f"  final loss {lc:.3e}")
    hp = {k: getattr(c, k) for k in ["adam_lr", "adam_iters", "lm_iters", "lm_cg", "lm_mu_init",
          "lm_mu_min", "lm_rho_accept", "lm_rho_good", "lm_mu_dec", "lm_mu_inc", "lm_max_rej"]}
    return loss_hist, {"adam": la, "lm": ll, "hyperparams": hp}


def build_laplace():
    w_last = params["fusion"][-1]["w"].reshape(-1); b_last = params["fusion"][-1]["b"][0]
    aug = lambda phi: jnp.concatenate([phi, jnp.ones((phi.shape[0], 1))], 1)

    def feat(p, x, t): return extract_features(p, jnp.array([[x, t]]))[0]
    def fdx(p, x, t):  return jax.jacfwd(feat, 1)(p, x, t)
    def fdt(p, x, t):  return jax.jacfwd(feat, 2)(p, x, t)
    def fdxx(p, x, t): return jax.jacfwd(fdx, 1)(p, x, t)
    pad = vmap(lambda x, t: (feat(params, x, t), fdx(params, x, t), fdt(params, x, t), fdxx(params, x, t)))
    phiF, dxF, dtF, dxxF = pad(d.x_f, d.t_f)
    uF = phiF @ w_last + b_last; uxF = dxF @ w_last

    # Analytic last-layer residual Jacobian, PER PDE. The pre-2026-07-17 version had only
    # burgers + an else-branch hardcoding the ALLEN-CAHN linearisation; convection_diffusion
    # (and any new PDE) silently inherited AC's coefficient, wrong diffusion, and no advection.
    # The trained MAP was never affected (training uses the autodiff residual); only this
    # inline whitened-BLL diagnostic was. CD's map.json sigma-ratios need a re-run.
    if c.pde == "burgers":
        JFw = dtF + uxF[:, None]*phiF + uF[:, None]*dxF - float(c.nu)*dxxF
        JFb = uxF[:, None]
    elif c.pde == "allen_cahn":
        coef = (-c.reac + 3.0*c.reac*uF**2)                    # d/du of (-r u + r u^3)
        JFw = dtF - float(c.eps2)*dxxF + coef[:, None]*phiF
        JFb = coef[:, None]
    elif c.pde == "fisher_kpp":
        coef = (-c.kpp_r*(1.0 - 2.0*uF))                       # d/du of (-r u (1-u))
        JFw = dtF - float(c.kpp_D)*dxxF + coef[:, None]*phiF
        JFb = coef[:, None]
    elif c.pde == "nagumo":
        # residual = u_t - D u_xx - r u(u-a)(1-u);  d(residual)/du = r(3u^2 - 2(1+a)u + a)
        coef = c.nag_r*(3.0*uF**2 - 2.0*(1.0 + c.nag_a)*uF + c.nag_a)
        JFw = dtF - float(c.nag_D)*dxxF + coef[:, None]*phiF
        JFb = coef[:, None]
    elif c.pde == "convection_diffusion":
        JFw = dtF + float(c.cd_a)*dxF - float(c.cd_nu)*dxxF    # linear PDE: no u-dependence
        JFb = jnp.zeros_like(uF)[:, None]                      # constant shift is annihilated
    else:
        raise SystemExit(f"build_laplace: no analytic last-layer Jacobian for pde={c.pde!r}; "
                         "add a branch (or use laplace_core/m1b, which autodiffs the residual).")
    J_F = jnp.concatenate([JFw, JFb], 1)

    r = residual_fn(params); blocks = []
    phi_ic = aug(extract_features(params, d.XT_IC))
    blocks.append((c.ic_weight*phi_ic, max(float(jnp.mean(r[:ctx.idx.ic]**2)), 1e-14)))
    off = ctx.idx.ic
    if c.pde == "burgers":
        b1 = aug(extract_features(params, d.XT_BC1)); b2 = aug(extract_features(params, d.XT_BC2))
        blocks.append((c.bc_weight*b1, max(float(jnp.mean(r[off:off+c.N_b]**2)), 1e-14)))
        blocks.append((c.bc_weight*b2, max(float(jnp.mean(r[off+c.N_b:off+2*c.N_b]**2)), 1e-14)))
        off += 2*c.N_b
    s2_f = max(float(jnp.mean(r[off:off+c.N_f]**2)), 1e-14)
    blocks.append((J_F, s2_f))

    alpha = 1.0/max(float(jnp.mean(w_last**2)), 1e-30)
    P = alpha*jnp.eye(c.d_aug)
    for J, s2 in blocks: P = P + (J.T@J)/s2
    L = jnp.linalg.cholesky(P + 1e-12*jnp.eye(c.d_aug))
    print(f"  alpha_hat={alpha:.3e} s2_f={s2_f:.3e}")
    return w_last, b_last, L, float(alpha)


@jax.jit
def predict(p, xt, w_last, b_last, L):
    phi = extract_features(p, xt); phia = jnp.concatenate([phi, jnp.ones((phi.shape[0], 1))], 1)
    y = jax.scipy.linalg.solve_triangular(L, phia.T, lower=True)
    return phi @ w_last + b_last, jnp.sqrt(jnp.sum(y**2, 0))


def main():
    t = prov.Timer()
    loss_hist, _tlog = train(t)
    g_norm = float(jnp.sqrt(jnp.sum(ravel_pytree(jax.grad(loss_fn)(params))[0]**2)))
    # The last-layer Laplace diagnostic is OPTIONAL; the solve gate is not.
    #
    # This used to be `w_last, b_last, L, alpha = build_laplace()` with build_laplace
    # raising SystemExit for any PDE lacking an analytic branch. On 2026-07-27 that
    # threw away five completed KS runs -- 3h40m of GPU each, Adam and LM both
    # converged (final loss 4.4e-5) -- because KS has no branch. Nothing was saved, no
    # gate was evaluated, and 22_ks_train.slurm's header claimed the opposite
    # ("train_map saves unconditionally"). The gate needs only the MAP prediction;
    # sigma is used solely by the UQ diagnostics. So a missing Laplace branch now
    # costs those diagnostics and NOTHING else.
    lap_err = None
    with t.phase("bll"):
        try:
            w_last, b_last, L, alpha = build_laplace()
        except SystemExit as e:                       # the "no analytic branch" path
            lap_err = str(e); w_last = b_last = L = None; alpha = None   # null, not NaN:
            # json.dump emits bare NaN, which is invalid strict JSON and would break
            # any non-Python consumer of the released artifacts.
            print(f"!! Laplace diagnostic UNAVAILABLE: {lap_err}")
            print("!! Continuing: MAP, fields and SOLVE GATE are still produced. "
                  "sigma-dependent metrics are omitted (recorded as null in map.json).")

    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1]-xs[0], axis=1)).ravel()
    XX, TT = np.meshgrid(xs, ts); XT = jnp.asarray(np.stack([XX.ravel(), TT.ravel()], 1))
    with t.phase("eval"):
        if L is None:
            mu = ctx.forward(params, XT)[:, 0]        # MAP mean; identical to predict's
            mu.block_until_ready(); sig = jnp.full(mu.shape, jnp.nan)
        else:
            mu, sig = predict(params, XT, w_last, b_last, L); mu.block_until_ready()
    up, sg, ue = np.asarray(mu), np.asarray(sig), U.ravel()

    regions, p70, p95 = M.assign_regions(ga)
    gate = M.solve_acceptance(up, ue, regions, REL_L2_GATE)   # MAP only -- always runs
    if L is None:
        rr = rat = cur = None
    else:
        rr = M.region_metrics(up, ue, sg, regions); rat = M.calibration_ratios(rr)
        cur = M.stiffness_curves(ga, np.abs(up-ue), sg)

    # save MAP weights (shared by all methods)
    pkl = os.path.join(OUTDIR, f"{TAG}_params.pkl"); C.save_map(params, pkl)

    # save the EXACT per-row whitening vector this arm was trained in, so the
    # covariance/probe scripts use the identical metric (WHITEN_SIG_NPY=<this file>).
    # raw -> ones (no whitening); adaptive -> RMS at this MAP (self-consistent);
    # frozen -> the sigmas frozen from the raw MAP (NOT recomputed here).
    if WHITEN_TRAIN == "raw":
        _wsig = np.ones(N_TOTAL)
    elif WHITEN_TRAIN == "frozen":
        _wsig = np.asarray(_sig_frozen)
    else:
        _wsig = np.asarray(ctx.block_sigma(ctx.residual_fn(params)))
    np.save(os.path.join(OUTDIR, f"{TAG}_whiten_sig.npy"), _wsig)

    # Persist the Fourier embedding so the network is reconstructable on ANY JAX version
    # (B is otherwise re-derived from the PRNG seed, whose threefry stream is
    # version-sensitive; common.build_context prefers this sidecar when present).
    if c.arch == "fourier":
        np.savez(os.path.join(OUTDIR, f"{TAG}_fourier_B.npz"),
                 B_X=np.asarray(ctx.B_X), B_T=np.asarray(ctx.B_T))

    P = prov.collect(config=C.config_dict(c))
    P["timings"] = t.dump(); P["peak_gpu_mem_mb"] = prov.peak_gpu_mem_mb()
    rec = {"method": "BLL last-layer (P=129)", "tag": TAG, "whiten_train": WHITEN_TRAIN,
           "map_path": pkl, "map_sha1": C.sha1(pkl),
           "final_loss": float(loss_hist[-1]), "grad_norm": g_norm, "alpha_hat": alpha,
           "p70": p70, "p95": p95, "eval_grid": [c.eval_nx, c.eval_nt],
           "solve_gate": gate, "regions": rr, "ratios": rat,
           "stiffness_curves": cur, "provenance": P,
           # Explicit, so a consumer cannot mistake "absent" for "measured and null".
           "laplace_available": lap_err is None,
           "laplace_unavailable_reason": lap_err}
    M.save_json(rec, os.path.join(OUTDIR, f"{TAG}_map.json"))
    np.savez_compressed(os.path.join(OUTDIR, f"{TAG}_fields.npz"),
        xs=xs, ts=ts, u_exact=ue, u_pred=up, sigma=sg, grad_abs=ga,
        loss_hist=np.asarray(loss_hist),
        # exact training collocation/IC/BC points, so any external tool can
        # reproduce the posterior on identical inputs (methods using common
        # regenerate these deterministically; saved here for transparency):
        x_f=np.asarray(d.x_f), t_f=np.asarray(d.t_f),
        x_u=np.asarray(d.x_u), t_b=np.asarray(d.t_b))

    # [15thJuly] full per-iteration training log for the Adam-vs-GN study
    # (Adam: loss/grad_norm/wall; LM: loss/mu/rho/pred+actual reduction/rejections/step_norm/wall).
    _tlog.update({"tag": TAG, "phase_boundary": c.adam_iters, "final_loss": float(loss_hist[-1]),
                  "whiten_train": WHITEN_TRAIN, "gpu": P.get("gpu"), "peak_gpu_mem_mb": P.get("peak_gpu_mem_mb")})
    json.dump(_tlog, open(os.path.join(OUTDIR, f"{TAG}_training_log.json"), "w"))

    print("\n================ MAP RESULT ================")
    print(f"  SOLVE GATE : {'PASS' if gate['passed'] else 'FAIL'}  "
          f"(bulk={gate.get('bulk_rel_l2',0)*100:.2f}% global={gate['global_rel_l2']*100:.2f}% thr {REL_L2_GATE*100:.0f}%)")
    if rat: print(f"  sigma-ratio={rat['achieved_sigma_ratio']:.2f}x target={rat['target_rmse_ratio']:.1f}x shock meanZ={rat['shock_mean_z']:.1f}")
    else:   print(f"  sigma-ratio : NOT COMPUTED -- {lap_err or 'Laplace unavailable'}")
    print(f"  device={P['gpu']['nvidia_smi_query']} | timings={P['timings']}")
    print(f"  MAP saved: {pkl} (sha1 {rec['map_sha1'][:12]})")
    if not gate['passed']: print("  >>> MAP REJECTED: do not run downstream methods on this MAP.")
    print("============================================")


if __name__ == "__main__":
    main()
