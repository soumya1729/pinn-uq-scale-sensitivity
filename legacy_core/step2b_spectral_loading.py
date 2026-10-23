"""
step2b_spectral_loading.py -- "is it geometric?" via the robust alpha-sweep.

The sigma-ratio NUMBER is metric/alpha-dependent. The alpha-FREE structural question is:
does the shock region's sensitivity load the SLOPPY (small-eigenvalue) directions of the GGN
Hessian more than the smooth region's? Resolving individual sloppy modes is numerically
hopeless (the Gram spectrum spans ~20 decades -- measured 19.6-21.2 across the four canonical
configs -- so most of the small eigenvalues sit below the float64 floor lambda_max*eps ~ 4e-9:
82% of the 6400 for Burgers-MLP, 66% AC-MLP, 26% AC-Fourier, 13% Burgers-Fourier. An
eigendecomposition cannot resolve them). Instead we probe the loading INDIRECTLY and robustly:

  sigma_r^2(alpha) = sum_i (g_r . v_i)^2 / (alpha + lambda_i)   [Woodbury, reliable for any alpha]

As alpha shrinks, the small-lambda (sloppy) modes are "turned on" (1/(alpha+lambda) -> 1/lambda).
If the shock loads sloppier modes than the smooth region, the shock/smooth ratio R(alpha) GROWS
as alpha decreases. So:
  R(alpha) increases toward small alpha   <=>   shock loads sloppier modes (the geometric claim).
We sweep alpha over decades around the operating prior, in BOTH metrics. PASS (geometric,
metric-robust): R rises toward small alpha in both. Computed by Cholesky (Woodbury), robust.

WHICH METRIC THE CLAIM IS SCOPED TO. The paper's empirical claim is RAW-only -- the metric
the MAP is actually trained in (paper sec 6.4: "It does rise, in the raw metric"). All live
configurations pass in raw. allen_cahn_mlp_seed0 does NOT pass in whitened (R falls 5.73 ->
10.09 as alpha rises), and its raw rise is the shallowest of the suite (slope -0.037 vs
-0.174 Burgers-MLP); the paper discloses both facts explicitly, including that its raw curve
is non-monotone. Until 2026-08-04 this script collapsed that to a bare "FAIL/MIXED", which
invited a reader of the artifact to conclude the paper's claim was contradicted by its own
data. The verdict now names the failing metric and records `raw_passes`.

NOTE ON THE SHIPPED ARTIFACTS: the 18 committed *_spectral_loading.json predate that change
and still carry the old verdict string (they are under the byte-level headline freeze and
regenerating them needs a GPU). Their NUMBERS are unaffected -- only the summary wording and
the two new fields change on a fresh run.

THE PASS TEST IS DELIBERATELY TWO-POINT (R at the smallest alpha vs the largest), not a
monotonicity test: the theory predicts the endpoints, and R(alpha) need not be monotone for
arbitrary loading profiles. Monotonicity is now recorded per metric so that caveat is
checkable rather than asserted.

K_POINTS=30 is a SAMPLED estimate of R, not the full-grid ratio (step_fullgrid_ratio.py
exists because K-sampling under-estimates heavy-tailed sigma_shock). That is acceptable here
because the question is the SIGN OF THE TREND in alpha, not the level -- but if you quote a
level from this file, quote the full-grid one instead.

Env: PDE ARCH SEED FOURIER_SCALE MAP_PKL OUTDIR  K_POINTS(30)  PT_SEED(0)  N_ALPHA(11)
"""
import os, json, time

# NOTE (2026-07-23): this script builds BOTH metric blocks itself -- Jw as "whitened",
# Jw*sig as "raw". RAW_METRIC=1 is therefore handled defensively where sig is built
# (see below), exactly as step_fullgrid_ratio.py does; either way you read the "raw"
# block. Before that fix, RAW_METRIC=1 silently corrupted both blocks.
import numpy as np
import jax.numpy as jnp
from scipy.linalg import cho_factor, cho_solve
import common as C
import laplace_core as lc


def main():
    t0 = time.time()
    c = C.get_config(); ctx = C.build_context(c)
    TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
    OUTDIR = os.environ.get("OUTDIR", os.path.join("results", TAG)); os.makedirs(OUTDIR, exist_ok=True)
    MAP_PKL = os.environ.get("MAP_PKL", os.path.join(OUTDIR, f"{TAG}_params.pkl"))
    K = int(os.environ.get("K_POINTS", 30)); PT_SEED = int(os.environ.get("PT_SEED", 0))
    NA = int(os.environ.get("N_ALPHA", 11))
    if not os.path.exists(MAP_PKL):
        print(f"[2b] {TAG}: MAP not found ({MAP_PKL}) -- skip"); return
    params = C.load_map(MAP_PKL)
    print(f"[2b] {TAG}  K={K}  (alpha-sweep, {NA} points)", flush=True)

    r_map = np.asarray(ctx.residual_fn(params)); i = ctx.idx
    segs = ([(0, i.ic), (i.ic, i.bc1), (i.bc1, i.bc2), (i.bc2, i.f)] if hasattr(i, "bc1")
            else [(0, i.ic), (i.ic, i.bcu), (i.bcu, i.bcx), (i.bcx, i.f)])
    sig = np.ones_like(r_map)
    for a, b in segs:
        sig[a:b] = max(float(np.mean(r_map[a:b] ** 2)), 1e-14) ** 0.5
    if os.environ.get("RAW_METRIC", "0") == "1":
        # laplace_core already returned a RAW J, so the "raw" branch below must not
        # scale by sig a second time; neutralise it and both branches collapse to the
        # raw field. Mirrors step_fullgrid_ratio.py. Without this line RAW_METRIC=1
        # silently produced "whitened"=raw and "raw"=doubly-scaled -- it corrupted a
        # P8 re-run on 2026-07-23 with no error and plausible-looking numbers.
        sig[:] = 1.0
        print("[step2b] RAW_METRIC=1: sig neutralised; BOTH metric blocks are raw")

    eng = lc.run(ctx, c, params, "full")
    Jw = np.asarray(eng["J"]); N = int(eng["N"]); grad_u = eng["grad_u"]
    sub_map = eng["sub_map"]; alpha0 = float(eng["alpha"])

    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    XX, TT = np.meshgrid(xs, ts); Xr, Tr = XX.ravel(), TT.ravel()
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1] - xs[0], axis=1)).ravel()
    g70, g95 = np.percentile(ga, 70), np.percentile(ga, 95)
    # FEATURE_BAND=1 defines the "shock" side as the whole feature (>p70) instead of
    # the top-5% core. Diagnostic ONLY (PROJECT_LOG P8 partition-sensitivity test):
    # the Allen-Cahn interface carries 42% of total |grad u| in the transition band
    # vs 25% for Burgers, so a top-5% cut samples a smaller share of AC's feature.
    # Headline numbers are always FEATURE_BAND=0.
    FEATURE_BAND = os.environ.get("FEATURE_BAND", "0") == "1"
    g_hi = g70 if FEATURE_BAND else g95
    print(f"[step2b] shock side = {'FEATURE (>p70)' if FEATURE_BAND else 'core (>p95)'}"
          f"  threshold={g_hi:.4f}")
    rng = np.random.default_rng(PT_SEED)
    sh = rng.choice(np.where(ga >= g_hi)[0], min(K, int((ga >= g_hi).sum())), replace=False)
    sm = rng.choice(np.where(ga < g70)[0], min(K, int((ga < g70).sum())), replace=False)
    g_shock = np.asarray(grad_u(sub_map, jnp.asarray(Xr[sh]), jnp.asarray(Tr[sh])))
    g_smooth = np.asarray(grad_u(sub_map, jnp.asarray(Xr[sm]), jnp.asarray(Tr[sm])))

    alphas = alpha0 * np.logspace(-3, 2, NA)            # decades around the operating prior
    out = {"tag": TAG, "K": K, "alpha0": alpha0, "alphas": alphas.tolist(), "metrics": {}}
    verdict_ok = True
    for mname, J in [("whitened", Jw), ("raw", Jw * sig[:, None])]:
        G = J @ J.T                                     # (N,N)
        JG_sh = (J @ g_shock.T)                          # (N,K)
        JG_sm = (J @ g_smooth.T)
        gg_sh = np.sum(g_shock ** 2, axis=1); gg_sm = np.sum(g_smooth ** 2, axis=1)

        def ratio(a):
            cM = cho_factor(G + a * np.eye(N), lower=True)
            s2_sh = np.clip((gg_sh - np.sum(JG_sh * cho_solve(cM, JG_sh), axis=0)) / a, 0, None)
            s2_sm = np.clip((gg_sm - np.sum(JG_sm * cho_solve(cM, JG_sm), axis=0)) / a, 0, None)
            return float(np.mean(np.sqrt(s2_sh)) / (np.mean(np.sqrt(s2_sm)) + 1e-300))

        Rs = [ratio(a) for a in alphas]
        rises = Rs[0] > Rs[-1]                           # R(smallest alpha) > R(largest alpha)?
        slope = float(np.polyfit(np.log(alphas), np.log(np.maximum(Rs, 1e-300)), 1)[0])  # dlogR/dlogalpha
        # Monotonicity is recorded but deliberately NOT part of the pass criterion: the theory
        # only predicts the endpoints (R need not be monotone for arbitrary loading profiles),
        # and the paper states that caveat explicitly. Recording it keeps the caveat checkable
        # instead of asserted -- the Allen-Cahn-MLP raw curve is in fact non-monotone.
        monotone = all(Rs[i] >= Rs[i + 1] for i in range(len(Rs) - 1))
        verdict_ok = verdict_ok and rises
        out["metrics"][mname] = {"R_of_alpha": Rs, "R_small_alpha": Rs[0], "R_large_alpha": Rs[-1],
                                 "dlogR_dlogalpha": slope, "shock_loads_sloppier": bool(rises),
                                 "monotone_decreasing_in_alpha": bool(monotone)}
        print(f"[2b] {mname}: R(alpha) {Rs[0]:.3g} (a={alphas[0]:.2g}) ... {Rs[-1]:.3g} (a={alphas[-1]:.2g})  "
              f"dlogR/dlogalpha={slope:+.2f}  -> shock {'SLOPPIER (R rises as alpha falls)' if rises else 'NOT sloppier'}",
              flush=True)

    # The verdict must name WHICH metric failed. The paper's empirical claim is scoped to the
    # RAW metric (the one the MAP is trained in); a bare "FAIL/MIXED" invites a reader of the
    # artifact to conclude the paper's claim is contradicted when only the whitened block
    # dissents. Allen-Cahn-MLP is exactly that case: raw passes (slope -0.037, the shallowest
    # of the suite, disclosed as such in the paper), whitened does not.
    passed = [m for m in ("raw", "whitened") if out["metrics"][m]["shock_loads_sloppier"]]
    failed = [m for m in ("raw", "whitened") if not out["metrics"][m]["shock_loads_sloppier"]]
    out["passed_metrics"], out["failed_metrics"] = passed, failed
    out["raw_passes"] = "raw" in passed
    if verdict_ok:
        verdict = ("PASS (geometric, METRIC-ROBUST): R(alpha) rises toward small alpha in BOTH "
                   "metrics -> shock loads the sloppy modes more than smooth; the structural "
                   "ceiling survives the whitening, the metric only rescales its magnitude.")
    elif "raw" in passed:
        verdict = ("PASS IN THE RAW METRIC (the metric the MAP is trained in, and the one the "
                   "paper's claim is scoped to); the %s block does not rise. Report as "
                   "raw-confirmed, metric-dependent -- NOT as a failure of the hypothesis."
                   % ", ".join(failed))
    else:
        verdict = ("FAIL: the shock does not load sloppier modes in the RAW metric -> the "
                   "geometric structure is PDE- or configuration-dependent here; report it as "
                   "such, not as universal.")
    out["verdict"] = verdict; out["wall_seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(OUTDIR, f"{TAG}_spectral_loading.json"), "w"), indent=2)
    print(f"[2b] VERDICT: {verdict}", flush=True)
    print(f"[2b] wrote {TAG}_spectral_loading.json  ({out['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
