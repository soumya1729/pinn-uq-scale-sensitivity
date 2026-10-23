"""Same-grid last-layer feature-span diagnostic at a fixed trained parameter.

At last-layer scope the network output lives in the span of {phi_k(x), 1}. We
report two complementary quantities:
  (a) LOCAL off-manifold fraction per region -- project the region's error onto the
      feature span RESTRICTED to that region's points; residual fraction in [0,1].
      Answers: "is the error in THIS region representable by the features here?"
  (b) GLOBAL off-manifold field magnitude (abs RMSE) per region -- the local
      magnitude of the single global-projection residual. Answers: "where does the
      residual of one same-grid global least-squares fit concentrate?"
NOTE: a per-region fraction taken from the GLOBAL projection residual is NOT a valid
fraction (the bound <=1 only holds globally) -- that was the original bug. Cheap:
features only, no AD. Env: common.get_config + MAP_PKL.

SCOPE. This is a local, same-grid diagnostic. It neither tests coefficients on held-out
points nor proves that one coefficient vector represents the full error field, and it does not
identify a posterior or a misspecification bias. Small local fractions in the released runs
therefore weigh against a large sampled local off-span component only; they do not establish
global representability or causally identify the spectral mechanism.

The theorem-aligned loading follow-up is the null-inclusive stable-complement producer
d_loading_perpoint.py and the spectral sandwich in the current paper. The deprecated
inverse-lift scalar in d_loading_cdf.py is not scientific evidence."""
import os, numpy as np, jax.numpy as jnp
import common as C, provenance as prov, uq_metrics as M


def main():
    c = C.get_config(); ctx = C.build_context(c)
    TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
    OUTDIR = os.environ.get("OUTDIR", os.path.join("results", TAG)); os.makedirs(OUTDIR, exist_ok=True)
    MAP_PKL = os.environ.get("MAP_PKL", os.path.join(OUTDIR, f"{TAG}_params.pkl"))
    params = C.load_map(MAP_PKL); t = prov.Timer()
    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1]-xs[0], axis=1)).ravel()
    XX, TT = np.meshgrid(xs, ts); XT = jnp.asarray(np.stack([XX.ravel(), TT.ravel()], 1))
    with t.phase("features"):
        Phi = np.asarray(ctx.extract_features(params, XT))               # (N,128)
        u_map = np.asarray(ctx.forward(params, XT)[:, 0])
    Phi_aug = np.concatenate([Phi, np.ones((Phi.shape[0], 1))], axis=1)   # (N,129) output span basis
    ue = U.ravel(); e = u_map - ue                                       # MAP error field

    # GLOBAL projection of e onto col(Phi_aug). Since u_map is in col(Phi_aug),
    # e_norm = P_perp e = -P_perp u_true = the structurally unrepresentable part.
    coef, *_ = np.linalg.lstsq(Phi_aug, e, rcond=None)
    e_tan = Phi_aug @ coef
    e_norm = e - e_tan
    glob_off_frac = float(np.sum(e_norm**2) / (np.sum(e**2) + 1e-30))     # bounded [0,1]

    regions, p70, p95 = M.assign_regions(ga)
    out = {}
    for nm in ("smooth", "transition", "shock"):
        m = regions == nm
        if m.sum() == 0: continue
        em = e[m]; Pm = Phi_aug[m]
        # LOCAL projection: best fit using the feature span restricted to THIS region
        cf, *_ = np.linalg.lstsq(Pm, em, rcond=None)
        e_off_loc = em - Pm @ cf
        tot = float(np.sum(em**2)) + 1e-30
        out[nm] = {
            "rmse": float(np.sqrt(np.mean(em**2))),
            "offmanifold_frac_local": float(np.sum(e_off_loc**2) / tot),        # bounded [0,1]
            "offmanifold_rmse_local": float(np.sqrt(np.mean(e_off_loc**2))),
            "offmanifold_rmse_global": float(np.sqrt(np.mean(e_norm[m]**2))),    # abs magnitude of global P_perp field
            "count": int(m.sum())}
    P = prov.collect(config=C.config_dict(c)); P["timings"] = t.dump(); P["peak_gpu_mem_mb"] = prov.peak_gpu_mem_mb()
    rec = {"method": "manifold decomposition (info-geometry)", "scope": "last_layer", "tag": TAG,
           "map_path": MAP_PKL, "map_sha1": C.sha1(MAP_PKL), "feature_span_dim": Phi_aug.shape[1],
           "global_offmanifold_frac": glob_off_frac,
           "regions": out, "provenance": P, "solve_gate": {"passed": None, "note": "diagnostic"}}
    M.save_json(rec, os.path.join(OUTDIR, f"{TAG}_manifold_metrics.json"))
    np.savez_compressed(os.path.join(OUTDIR, f"{TAG}_manifold_fields.npz"),
        xs=xs, ts=ts, e=e, e_tangent=e_tan, e_offmanifold=e_norm,
        regions=np.asarray(regions, dtype=object))
    print("[d4 manifold] per-region LOCAL off-manifold fraction (bounded [0,1]) + abs magnitudes:")
    for nm in out:
        print(f"   {nm:11s}: local_off_frac={out[nm]['offmanifold_frac_local']:.3f}  "
              f"(rmse {out[nm]['rmse']:.2e}, local_off_rmse {out[nm]['offmanifold_rmse_local']:.2e}, "
              f"global_off_rmse {out[nm]['offmanifold_rmse_global']:.2e})")
    print(f"  global off-manifold fraction = {glob_off_frac:.3f}")
    print("  read: local_off_frac ~1 at shock => this component is outside the last-layer feature span;")
    print("        last-layer covariance is blind to it; a full-network blindness claim does not follow.")
    print("        local_off_frac ~0 => sampled regional error lies in the local same-grid span;")
    print("        covariance scale/allocation and held-out or global representability remain separate questions.")


if __name__ == "__main__":
    main()
