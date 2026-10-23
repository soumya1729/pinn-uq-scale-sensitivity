"""m6 -- Split conformal (absolute residual) on the shared trained parameter. Constant-width
interval => sigma/width-ratio = 1.0 by construction; we report per-region
coverage. In the retained Burgers-MLP seed-2 run, shock cov95 is about 0.31;
this single configuration is a diagnostic, not an architecture-wide claim. The output is strict
JSON: an unavailable maximum-calibration-error field is ``null``, never a non-standard NaN."""
import hashlib, json, os
import numpy as np, jax.numpy as jnp
import common as C, provenance as prov, uq_metrics as M

c = C.get_config(); ctx = C.build_context(c)
TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
OUTDIR = os.environ.get("OUTDIR", os.path.join("results", TAG)); os.makedirs(OUTDIR, exist_ok=True)
MAP_PKL = os.environ.get("MAP_PKL", os.path.join(OUTDIR, f"{TAG}_params.pkl"))


def main():
    params = C.load_map(MAP_PKL); t = prov.Timer()
    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    U = ctx.grid_reference(xs, ts); ga = np.abs(np.gradient(U, xs[1]-xs[0], axis=1)).ravel()
    XX, TT = np.meshgrid(xs, ts); XT = jnp.asarray(np.stack([XX.ravel(), TT.ravel()], 1))
    with t.phase("predict"):
        up = np.asarray(ctx.forward(params, XT)[:, 0])
    ue = U.ravel(); err = np.abs(up - ue)
    rng = np.random.default_rng(0); n = err.shape[0]
    cal = rng.permutation(n); half = n // 2
    calib, test = cal[:half], cal[half:]
    regions, p70, p95 = M.assign_regions(ga)
    def qlevel(a, lvl):
        m = len(a); k = int(np.ceil((m + 1) * lvl)); k = min(k, m)
        return np.sort(a)[k - 1]
    q95 = qlevel(err[calib], 0.95); q68 = qlevel(err[calib], 0.68)
    rr = {}
    for name in ("smooth", "transition", "shock"):
        mtest = (regions[test] == name)
        if mtest.sum() == 0: continue
        et = err[test][mtest]
        rr[name] = {"rmse": float(np.sqrt(np.mean(et**2))),
                    "mean_sigma": float(q95 / 1.96),   # constant width -> ratio 1.0
                    "mean_z": float(np.mean(et / max(q95/1.96, 1e-30))),
                    "coverage95": float(np.mean(et <= q95)),
                    "coverage68": float(np.mean(et <= q68)),
                    "mce": None, "count": int(mtest.sum())}
    rat = M.calibration_ratios(rr)
    P = prov.collect(config=C.config_dict(c)); P["timings"] = t.dump(); P["peak_gpu_mem_mb"] = prov.peak_gpu_mem_mb()
    split_sha256 = hashlib.sha256(np.asarray(cal, dtype=np.int64).tobytes()).hexdigest()
    rec = {"schema_version": "split_conformal/2", "method": "Split conformal", "tag": TAG,
           "map_path": MAP_PKL, "map_sha1": C.sha1(MAP_PKL),
           "split_seed": 0, "split_permutation_sha256": split_sha256,
           "n_calibration": int(len(calib)), "n_test": int(len(test)),
           "q95": float(q95), "q68": float(q68), "solve_gate": {"passed": None, "note": "post-hoc on MAP"},
           "regions": rr, "ratios": rat, "provenance": P}
    target = os.path.join(OUTDIR, f"{TAG}_conformal_metrics.json")
    tmp = f"{target}.tmp.{os.getpid()}"
    with open(tmp, "x", encoding="utf-8") as handle:
        json.dump(rec, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, target)
    print("[m6 conformal] width-ratio=1.0 (constant). per-region cov95:",
          {k: round(rr[k]["coverage95"], 3) for k in rr})


if __name__ == "__main__":
    main()

