"""
d_loading_cdf.py -- DIRECT measurement of Theorem 1's two inputs (the "hero figure").

From the N-space Gram G = J J^T (N ~ 6400) eigendecomposition, compute the per-mode
loading of the region-mean output-gradient g:

    G = W diag(lambda) W^T
    Fisher eigvec v_i (P-space) = J^T w_i / sqrt(lambda_i)
    c_i = g^T v_i = (J g)^T w_i / sqrt(lambda_i)     =>     c_i^2 = ((J g)^T w_i)^2 / lambda_i

and the cumulative loading CDF  L_r(eps) = sum_{lambda_i <= eps} c_i^2  for r in {shock, smooth}.
Null-space (unresolved) loading = ||g||^2 - sum_i c_i^2 (the equation-unconstrained directions).

This is Theorem 1's numerator (loading) and tail depth (eps) measured directly, replacing the
truncated-Lanczos apologetics with the exact all-mode decomposition. Run in the RAW metric:

    RAW_METRIC=1 REUSE_MAP=1 PDE=... ARCH=... SEED=... FOURIER_SCALE=... python d_loading_cdf.py

Env: PDE ARCH SEED FOURIER_SCALE  CHUNK(200)
"""
import os, json, time
import numpy as np, jax.numpy as jnp
import common as C, laplace_core as lc


def main():
    t0 = time.time(); c = C.get_config(); ctx = C.build_context(c)
    TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
    OUT = os.environ.get("OUTDIR", os.path.join("results", TAG)); os.makedirs(OUT, exist_ok=True)
    pk = os.environ.get("MAP_PKL", os.path.join(OUT, f"{TAG}_params.pkl"))
    if not os.path.exists(pk):
        print(f"[loadcdf] {TAG}: no MAP at {pk} -- skip"); return
    params = C.load_map(pk); CH = int(os.environ.get("CHUNK", 200))

    eng = lc.run(ctx, c, params, "full")          # RAW_METRIC=1 -> J is the raw (un-whitened) Jacobian
    J = np.asarray(eng["J"]); N = int(eng["N"]); alpha = float(eng["alpha"])
    grad_u = eng["grad_u"]; sub = eng["sub_map"]

    # Gram eigendecomposition (N x N; the nonzero Fisher spectrum lives here)
    G = J @ J.T
    lam, W = np.linalg.eigh(G)                     # ascending eigenvalues, columns = eigenvectors
    lam = np.clip(lam, 0.0, None)

    # eval grid + region split (identical to laplace_core / step_fullgrid: shock=top5% |u_x|, smooth=bottom70%)
    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    XX, TT = np.meshgrid(xs, ts); Xr, Tr = XX.ravel(), TT.ravel()
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1] - xs[0], axis=1)).ravel()
    p70, p95 = np.percentile(ga, 70), np.percentile(ga, 95)
    regions = {"shock": np.where(ga >= p95)[0], "smooth": np.where(ga < p70)[0]}

    def region_mean_grad(idx):
        acc = None; n = 0
        for q in range(0, len(idx), CH):
            e = min(q + CH, len(idx)); ii = idx[q:e]
            g = np.asarray(grad_u(sub, jnp.asarray(Xr[ii]), jnp.asarray(Tr[ii])))  # (n, P)
            acc = g.sum(0) if acc is None else acc + g.sum(0); n += len(ii)
        return acc / n

    order = np.argsort(lam)
    out = {"tag": TAG, "alpha": alpha, "N": N, "lambda_sorted": lam[order].tolist()}
    for rname, idx in regions.items():
        g = region_mean_grad(idx)                  # (P,) mean output-gradient over the region
        Jg = J @ g                                 # (N,)
        a = W.T @ Jg                               # projection onto Gram eigenvectors
        ci2 = np.where(lam > 1e-30, a * a / lam, 0.0)   # per-mode loading c_i^2
        total = float(g @ g)                       # ||g||^2 (all P directions, incl. null space)
        captured = float(ci2.sum())
        # loading below the prior precision (the "sloppy" mass Theorem 1's numerator cares about)
        sloppy = float(ci2[lam <= alpha].sum())
        out[rname] = {
            "cum_loading": np.cumsum(ci2[order]).tolist(),   # L_r(lambda) aligned with lambda_sorted
            "total_gg": total, "captured": captured, "nullspace": total - captured,
            "loading_below_alpha": sloppy, "frac_below_alpha": sloppy / max(captured, 1e-300),
        }
    # headline: sloppy-loading ratio shock/smooth (Theorem 1's numerator, measured directly)
    rr = out["shock"]["loading_below_alpha"] / max(out["smooth"]["loading_below_alpha"], 1e-300)
    out["sloppy_loading_ratio_shock_over_smooth"] = rr
    json.dump(out, open(os.path.join(OUT, f"{TAG}_loading_cdf.json"), "w"), indent=2)
    print(f"[loadcdf] {TAG}: shock sloppy-loading/smooth = {rr:.2f}  "
          f"(shock captured {out['shock']['captured']:.3g}/{out['shock']['total_gg']:.3g}, "
          f"smooth {out['smooth']['captured']:.3g}/{out['smooth']['total_gg']:.3g})  "
          f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
