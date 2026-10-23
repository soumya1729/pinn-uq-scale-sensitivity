"""dump_fourier_B.py -- backfill *_fourier_B.npz sidecars for existing Fourier MAPs.

Why: the random Fourier embedding B_X/B_T is re-derived from the PRNG seed at load time,
and JAX's threefry stream changed across versions (jax_threefry_partitionable). Without a
sidecar, loading a Fourier params.pkl on a different JAX silently reconstructs a DIFFERENT
network. train_map.py now writes the sidecar at training time; this script backfills it
for MAPs trained before that.

MUST run under the SAME PRNG stream that trained the MAPs (the cluster's JAX 0.10.1, or
any JAX with jax_threefry_partitionable=True -- set REPLICATE_STREAM=1 to force the flag,
verified to reproduce the cluster stream exactly).

B depends ONLY on (seed, fourier_dim, fourier_scale) -- read per-config from each
*_map.json's provenance.config, so FS-survey and CD variants get the right scale. Walks
results/ and analysis/sweep_results/. Idempotent: skips dirs that already have a sidecar.
"""
import os, sys, glob, json
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   # CRITICAL: common.py runs under x64, and
# jax.random.normal draws DIFFERENT values in x64 vs x32 (different bit consumption).
# Deriving B without x64 silently produces the wrong embedding (found the hard way).
if os.environ.get("REPLICATE_STREAM", "0") == "1":
    jax.config.update("jax_threefry_partitionable", True)
import jax.numpy as jnp

ROOTS = sys.argv[1:] or ["results", "analysis/sweep_results"]


def derive_B(seed, dim, scale):
    # EXACTLY the calls in common.build_context (same split, same shapes, same order).
    key = jax.random.PRNGKey(int(seed))
    key, kx, kt = jax.random.split(key, 3)
    B_X = jax.random.normal(kx, (1, int(dim))) * float(scale)
    B_T = jax.random.normal(kt, (1, int(dim))) * float(scale)
    return np.asarray(B_X), np.asarray(B_T)


def main():
    n_done = n_skip = 0
    for root in ROOTS:
        for mj in sorted(glob.glob(os.path.join(root, "**", "*_map.json"), recursive=True)):
            try:
                cfg = json.load(open(mj)).get("provenance", {}).get("config", {})
            except Exception:
                continue
            if cfg.get("arch") != "fourier":
                continue
            d = os.path.dirname(mj)
            tag = f"{cfg['pde']}_fourier_seed{cfg['seed']}"
            out = os.path.join(d, f"{tag}_fourier_B.npz")
            if os.path.exists(out) and os.environ.get("OVERWRITE", "0") != "1":
                n_skip += 1; continue
            if not os.path.exists(os.path.join(d, f"{tag}_params.pkl")):
                continue                      # no weights here; nothing to protect
            B_X, B_T = derive_B(cfg["seed"], cfg["fourier_dim"], cfg["fourier_scale"])
            np.savez(out, B_X=B_X, B_T=B_T)
            print(f"wrote {out}  (seed={cfg['seed']} dim={cfg['fourier_dim']} FS={cfg['fourier_scale']})")
            n_done += 1
    print(f"done: {n_done} written, {n_skip} already present")


if __name__ == "__main__":
    main()
