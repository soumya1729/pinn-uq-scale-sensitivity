"""d4_baselines.py -- non-triviality + robustness companions to d4_manifold.py.

Backs three claims in the off-manifold section with a committed, reproducible script
(PROJECT_LOG open items 1-3):

(A) GENERIC-BASIS BASELINES on the real shock error. The per-region in-span fraction is
    only meaningful against what a generic 129-dim basis achieves at the same band
    geometry. We fit the SAME shock-region error with: the trained features, random
    Fourier features (129), tensor-Chebyshev (~129), and white noise (129). Finding
    (canonical Burgers-MLP seed 2): generic smooth bases already reach f_perp ~1-2e-2,
    white noise fails (~0.97), the trained features reach ~1.2e-4. The off-manifold
    hypothesis needed ~1; it is dead under ANY basis, with the trained features 100x
    lower still.

(B) ROBUSTNESS: rank-truncated projection (SVD) and shock-band percentile sensitivity
    of the trained-feature fraction.

(C) SYNTHETIC METRIC DEMO (replaces the previously uncommitted "synthetic stress test"):
    random features + error energy concentrated in the top-5% band reproduce the
    masked-global ">1 fraction" pathology, while the per-region local fraction stays
    bounded in [0,1]. Numbers are seeded and reproducible here; the paper quotes THIS
    script's output.

VERIFIED against results/burgers_mlp_seed2/..._d4_baselines.json on 2026-08-04: trained
1.214e-4, random-Fourier 1.302e-2, tensor-Chebyshev(134) 1.981e-2, white noise 0.965;
synthetic demo masked-global 4.397 (>1, the pathology) vs local 0.992 (bounded). Every
number in (A) and (C) above matches its artifact.

THE TRIO, AND WHERE THE ANSWER IS. This script is the non-triviality companion;
d4_manifold.py is the per-region local fraction; d5_manifold_scope.py extends the same
question across Bayesian scope. Together they REFUTE representability as the cause of the
sigma-ratio ceiling -- they do not explain it. The explanation is
step2b_spectral_loading.py (alpha-sweep), d_loading_cdf.py, and Theorem 1 in paper
sec 6.3, machine-verified in Lean 4. A reader who stops at d4 will correctly conclude
"it is a geometry problem" and then reinvent the alpha-sweep; the pointer is here so they
do not have to.

Env: PDE ARCH SEED FOURIER_SCALE MAP_PKL OUTDIR (suite-standard), BASE_SEED(0).
NOTE for Fourier configs: run on the cluster jax (threefry stream) or with the B sidecar
(FOURIER_B_NPZ) so the embedding matches training. MLP configs are stream-safe anywhere.
Output: results/<tag>/<tag>_d4_baselines.json
"""
import os, json
import numpy as np
import jax.numpy as jnp
import common as C
import uq_metrics as M
import provenance as prov


def frac(P, y):
    c, *_ = np.linalg.lstsq(P, y, rcond=None)
    return float(np.sum((y - P @ c) ** 2) / np.sum(y ** 2))


def main():
    c = C.get_config(); ctx = C.build_context(c)
    TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
    OUTDIR = os.environ.get("OUTDIR", os.path.join("results", TAG)); os.makedirs(OUTDIR, exist_ok=True)
    MAP_PKL = os.environ.get("MAP_PKL", os.path.join(OUTDIR, f"{TAG}_params.pkl"))
    params = C.load_map(MAP_PKL); t = prov.Timer()
    rng = np.random.default_rng(int(os.environ.get("BASE_SEED", 0)))

    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1]-xs[0], axis=1)).ravel()
    XX, TT = np.meshgrid(xs, ts); XT = jnp.asarray(np.stack([XX.ravel(), TT.ravel()], 1))
    with t.phase("features"):
        Phi, u_map = [], []
        for q in range(0, XT.shape[0], 10000):
            Phi.append(np.asarray(ctx.extract_features(params, XT[q:q+10000])))
            u_map.append(np.asarray(ctx.forward(params, XT[q:q+10000])[:, 0]))
        Phi = np.concatenate(Phi); u_map = np.concatenate(u_map)
    Phi_aug = np.concatenate([Phi, np.ones((len(u_map), 1))], 1)
    e = u_map - U.ravel()
    d_aug = Phi_aug.shape[1]
    regions, _, _ = M.assign_regions(ga)
    msk = regions == "shock"; es = e[msk]; X = np.stack([XX.ravel(), TT.ravel()], 1)[msk]

    # ---- (A) generic-basis baselines on the real shock error -------------------------
    B = rng.normal(size=(2, (d_aug - 1) // 2)) * 3.0
    RFF = np.concatenate([np.cos(2*np.pi*X@B), np.sin(2*np.pi*X@B), np.ones((len(X), 1))], 1)
    from numpy.polynomial import chebyshev as Ch
    deg = 15
    Px = np.stack([Ch.chebval(X[:, 0], [0]*i + [1]) for i in range(deg)], 1)
    Pt = np.stack([Ch.chebval(2*X[:, 1] - 1, [0]*i + [1]) for i in range(deg)], 1)
    POLY = np.stack([Px[:, i]*Pt[:, j] for i in range(deg) for j in range(deg) if i + j < 16], 1)
    WN = rng.normal(size=(len(X), d_aug))
    baselines = {
        "trained_features": frac(Phi_aug[msk], es),
        "random_fourier":   frac(RFF, es),
        f"tensor_chebyshev_{POLY.shape[1]}": frac(POLY, es),
        "white_noise":      frac(WN, es),
    }

    # ---- (B) robustness: rank truncation + band percentile ---------------------------
    Uu, s, _ = np.linalg.svd(Phi_aug[msk], full_matrices=False)
    proj = Uu.T @ es; tot = float(np.sum(es**2))
    rank_curve = {int(k): float((tot - np.sum(proj[:k]**2)) / tot) for k in (d_aug, 64, 32, 16, 8)}
    band_curve = {}
    for pct in (98, 95, 90, 85):
        m = ga >= np.percentile(ga, pct)
        band_curve[f"top{100-pct}pct"] = frac(Phi_aug[m], e[m])
    cond_phi_shock = float(s[0] / s[-1])

    # ---- (C) synthetic metric demo (masked-global >1 vs bounded local) ---------------
    Ns = 20000; F = rng.normal(size=(Ns, d_aug))
    err = rng.normal(size=Ns) * 0.01
    hot = rng.choice(Ns, Ns // 20, replace=False)          # 5% "shock" points carry the energy
    err[hot] += rng.normal(size=len(hot)) * 1.0
    lab = np.full(Ns, "smooth", dtype=object); lab[hot] = "shock"
    cg, *_ = np.linalg.lstsq(F, err, rcond=None); resid = err - F @ cg
    synth = {}
    for nm in ("smooth", "shock"):
        m = lab == nm
        synth[nm] = {"masked_global_frac": float(np.sum(resid[m]**2)/np.sum(err[m]**2)),
                     "local_frac": frac(F[m], err[m])}

    P = prov.collect(config=C.config_dict(c)); P["timings"] = t.dump()
    rec = {"method": "d4 baselines (non-triviality + robustness + synthetic demo)",
           "tag": TAG, "map_path": MAP_PKL, "map_sha1": C.sha1(MAP_PKL),
           "feature_span_dim": d_aug, "n_shock": int(msk.sum()),
           "shock_baseline_fracs": baselines,
           "rank_truncation_shock_frac": rank_curve,
           "band_percentile_shock_frac": band_curve,
           "cond_Phi_shock": cond_phi_shock,
           "synthetic_demo": synth, "provenance": P,
           "solve_gate": {"passed": None, "note": "diagnostic"}}
    M.save_json(rec, os.path.join(OUTDIR, f"{TAG}_d4_baselines.json"))
    print(f"[d4b] {TAG}  shock baselines: " + "  ".join(f"{k}={v:.2e}" for k, v in baselines.items()))
    print(f"[d4b] rank curve: {rank_curve}")
    print(f"[d4b] band curve: {band_curve}   cond(Phi_shock)={cond_phi_shock:.2e}")
    print(f"[d4b] synthetic: masked-global smooth={synth['smooth']['masked_global_frac']:.2f} (>1 = pathology), "
          f"local smooth={synth['smooth']['local_frac']:.2f}, shock local={synth['shock']['local_frac']:.2f} (bounded)")


if __name__ == "__main__":
    main()
