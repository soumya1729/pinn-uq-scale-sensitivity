"""Full-grid post-hoc Gaussian fields with an explicit metric contract.

Normal mode produces both the historical adaptive-block-whitened field and the raw field, and
therefore must start from the default whitened engine.  ``SIGMA_DUMP_ONLY=1`` is the raw-only
production path and requires ``RAW_METRIC=1``.  The old producer allowed RAW_METRIC=1 in normal
mode, silently writing the same raw field under both labels; this version fails closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve

import common as C
import laplace_core as lc


JSON_SCHEMA = "fullgrid_ratio/2"
FIELD_SCHEMA = "fullgrid_sigma/1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    # ``np.savez_compressed`` closes the archive before returning. Reopen it only to force the
    # completed bytes to stable storage before the atomic rename; a visible terminal metadata
    # record must never point at an archive still resident only in the page cache.
    with tmp.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> int:
    started = time.time()
    raw_engine = os.environ.get("RAW_METRIC") == "1"
    dump_only = os.environ.get("SIGMA_DUMP_ONLY", "0") == "1"
    if dump_only and not raw_engine:
        raise SystemExit("[fg] SIGMA_DUMP_ONLY=1 requires RAW_METRIC=1")
    if not dump_only and raw_engine:
        raise SystemExit(
            "[fg] normal dual-field mode requires the whitened engine; RAW_METRIC=1 would "
            "mislabel a raw field as whitened")

    c = C.get_config()
    tag = f"{c.pde}_{c.arch}_seed{c.seed}"
    outdir = Path(os.environ.get("OUTDIR", str(Path("results") / tag)))
    outdir.mkdir(parents=True, exist_ok=True)
    map_path = Path(os.environ.get("MAP_PKL", str(Path("results") / tag / f"{tag}_params.pkl")))
    if not map_path.is_file():
        raise SystemExit(f"[fg] {tag}: required MAP is absent: {map_path}")

    chunk = int(os.environ.get("CHUNK", 200))
    scope = os.environ.get("SCOPE", "full")
    if scope not in ("full", "all_fusion", "last_2_fusion", "last_layer"):
        raise SystemExit(f"[fg] unknown SCOPE={scope!r}")
    if chunk <= 0:
        raise SystemExit(f"[fg] CHUNK must be positive, got {chunk}")
    suffix = "" if scope == "full" else f"_{scope}"
    alpha_mult = float(os.environ.get("ALPHA_MULT", 1.0))
    if not math.isfinite(alpha_mult) or alpha_mult <= 0:
        raise SystemExit(f"[fg] ALPHA_MULT must be finite and positive, got {alpha_mult!r}")
    mult_suffix = "" if alpha_mult == 1.0 else f"_M{alpha_mult:g}"
    negative_variance_tol = float(os.environ.get("NEGATIVE_VARIANCE_REL_TOL", "1e-10"))
    if not (math.isfinite(negative_variance_tol) and
            0 < negative_variance_tol <= 1e-10):
        raise SystemExit("[fg] NEGATIVE_VARIANCE_REL_TOL must be in (0, 1e-10]")

    sidecar_path = None
    if c.arch == "fourier":
        sidecar_path = Path(os.environ.get(
            "FOURIER_B_NPZ", str(Path("results") / tag / f"{tag}_fourier_B.npz")
        )).resolve()
        if not sidecar_path.is_file():
            raise SystemExit(f"[fg] {tag}: frozen Fourier sidecar is absent: {sidecar_path}")
        # Set the checked path before constructing the context; a PRNG fallback would create a
        # different network while leaving plausible-looking uncertainty fields.
        os.environ["FOURIER_B_NPZ"] = str(sidecar_path)

    metrics_to_write = ("raw",) if dump_only else ("whitened", "raw")
    # Keep the last complete field until its atomic replacement is ready.  A failed rerun is
    # detected by the missing/stale terminal metadata and by downstream schema/hash checks; it
    # must not destroy the only retained historical field before the expensive build starts.
    terminal = (outdir / f"{tag}_fullgrid_sigma_raw{suffix}{mult_suffix}.metadata.json"
                if dump_only else
                outdir / f"{tag}_fullgrid_ratio{suffix}{mult_suffix}.json")
    terminal.unlink(missing_ok=True)

    ctx = C.build_context(c)
    params = C.load_map(str(map_path))
    eng = lc.run(ctx, c, params, scope)
    J = np.asarray(eng["J"])
    n_residual = int(eng["N"])
    alpha = float(eng["alpha"]) * alpha_mult
    if not math.isfinite(alpha) or alpha <= 0:
        raise SystemExit(f"[fg] invalid posterior precision alpha={alpha!r}")
    grad_u, sub = eng["grad_u"], eng["sub_map"]

    residual = np.asarray(ctx.residual_fn(params))
    idx = ctx.idx
    segs = ([(0, idx.ic), (idx.ic, idx.bc1), (idx.bc1, idx.bc2), (idx.bc2, idx.f)]
            if hasattr(idx, "bc1") else
            [(0, idx.ic), (idx.ic, idx.bcu), (idx.bcu, idx.bcx), (idx.bcx, idx.f)])
    adaptive_sigma = np.ones_like(residual)
    for a, b in segs:
        adaptive_sigma[a:b] = max(float(np.mean(residual[a:b] ** 2)), 1e-14) ** 0.5

    frozen_path = os.environ.get("WHITEN_SIG_NPY")
    frozen_sigma = (np.load(frozen_path) if frozen_path and Path(frozen_path).is_file()
                    else adaptive_sigma)
    if (frozen_sigma.shape != residual.shape or not np.all(np.isfinite(frozen_sigma)) or
            np.any(frozen_sigma <= 0)):
        raise SystemExit("[fg] invalid whitening vector")

    xs = np.linspace(-1, 1, c.eval_nx)
    ts = np.linspace(0, 1, c.eval_nt)
    xx, tt = np.meshgrid(xs, ts)
    xr, tr = xx.ravel(), tt.ravel()
    exact = ctx.grid_reference(xs, ts)
    grad_abs = np.abs(np.gradient(exact, xs[1] - xs[0], axis=1)).ravel()
    p70, p95 = np.percentile(grad_abs, 70), np.percentile(grad_abs, 95)
    shock = np.where(grad_abs >= p95)[0]
    smooth = np.where(grad_abs < p70)[0]

    def sigma_grid(Jm: np.ndarray) -> tuple[np.ndarray, float]:
        factor = cho_factor(Jm @ Jm.T + alpha * np.eye(n_residual), lower=True)
        variance = np.empty(len(xr))
        worst_negative = 0.0
        for q in range(0, len(xr), chunk):
            end = min(q + chunk, len(xr))
            if (q // chunk) % 25 == 0:
                print(f"[fg] sigma_grid {q}/{len(xr)}", flush=True)
            g = np.asarray(grad_u(sub, jnp.asarray(xr[q:end]), jnp.asarray(tr[q:end])))
            Jg = Jm @ g.T
            scale = np.sum(g * g, axis=1) / alpha
            raw_variance = (
                np.sum(g * g, axis=1) - np.sum(Jg * cho_solve(factor, Jg), axis=0)
            ) / alpha
            negative = np.maximum(-raw_variance, 0.0) / np.maximum(
                scale, np.finfo(np.float64).tiny)
            worst_negative = max(worst_negative, float(np.max(negative)))
            variance[q:end] = np.maximum(raw_variance, 0.0)
        if worst_negative > negative_variance_tol:
            raise SystemExit(
                f"[fg] negative Woodbury variance {worst_negative:.3e} relative exceeds "
                f"{negative_variance_tol:.3e}; refusing to hide it by clamping"
            )
        # The output-gradient contains the final bias direction, so a valid positive-definite
        # posterior has strictly positive variance at every point.  Even a tolerated roundoff
        # excursion must not be published as a plausible zero-width interval after clamping.
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise SystemExit(
                "[fg] non-finite or non-positive posterior variance remains after the "
                "Woodbury check; refusing to write a field"
            )
        return np.sqrt(variance), worst_negative

    # In raw-only mode J is already raw.  In dual mode it is adaptively whitened and is lifted
    # back exactly once before constructing the two named branches.
    Jraw = J if dump_only else J * adaptive_sigma[:, None]
    branches = ([('raw', Jraw)] if dump_only else
                [('whitened', Jraw / frozen_sigma[:, None]), ('raw', Jraw)])
    producer_path = Path(__file__).resolve()
    common_path = Path(C.__file__).resolve()
    laplace_path = Path(lc.__file__).resolve()
    code_hashes = {"producer": sha256(producer_path), "common": sha256(common_path),
                   "laplace_core": sha256(laplace_path)}
    input_hashes = {"map": sha256(map_path),
                    "fourier_sidecar": sha256(sidecar_path) if sidecar_path else None,
                    "frozen_whitening": (sha256(Path(frozen_path))
                                           if frozen_path and Path(frozen_path).is_file()
                                           else None)}
    record = {
        "schema": JSON_SCHEMA,
        "tag": tag,
        "alpha": alpha,
        "alpha_multiplier": alpha_mult,
        "scope": scope,
        "P_bayes": int(eng["n_sub"]),
        "n_shock": int(len(shock)),
        "n_smooth": int(len(smooth)),
        "configuration": C.config_dict(c),
        "sigma_metric_mode": "raw_only" if dump_only else "dual_raw_and_adaptive_whitened",
        "negative_variance_relative_tolerance": negative_variance_tol,
        "inputs_sha256": input_hashes,
        "code_sha256": code_hashes,
        "fields": {},
    }
    for metric, Jm in branches:
        sigma, worst_negative = sigma_grid(Jm)
        field_path = outdir / f"{tag}_fullgrid_sigma_{metric}{suffix}{mult_suffix}.npz"
        atomic_npz(
            field_path,
            schema=np.asarray(FIELD_SCHEMA),
            sigma_metric=np.asarray(metric),
            sigma=sigma,
            xs=xs,
            ts=ts,
            shock_idx=shock,
            smooth_idx=smooth,
            scope=np.asarray(scope),
            P_bayes=np.asarray(int(eng["n_sub"])),
            producer_sha256=np.asarray(code_hashes["producer"]),
            map_sha256=np.asarray(input_hashes["map"]),
            negative_variance_max_relative=np.asarray(worst_negative),
        )
        msh, msm = sigma[shock], sigma[smooth]
        ratio_mean = float(msh.mean() / msm.mean())
        ratio_median = float(np.median(msh) / np.median(msm))
        top = np.sort(msh)[::-1][:max(1, len(msh) // 100)]
        branch = {
            "sigma_metric": metric,
            "field_path": str(field_path),
            "field_sha256": sha256(field_path),
            "ratio_mean": ratio_mean,
            "ratio_median": ratio_median,
            "frac_mean_from_top1pct": float(top.sum() / msh.sum()),
            "shock_sigma_max_over_median": float(msh.max() / np.median(msh)),
            "negative_variance_max_relative": worst_negative,
        }
        record[metric] = branch
        record["fields"][metric] = {
            "path": branch["field_path"], "sha256": branch["field_sha256"],
            "sigma_metric": metric, "schema": FIELD_SCHEMA,
        }
        print(f"[fg] {tag} {metric}: mean-ratio={ratio_mean:.6g}", flush=True)

    if dump_only:
        metadata = outdir / f"{tag}_fullgrid_sigma_raw{suffix}{mult_suffix}.metadata.json"
        atomic_json(metadata, {
            "schema": "fullgrid_sigma_metadata/1", "tag": tag, "scope": scope,
            "P_bayes": int(eng["n_sub"]), "sigma_metric": "raw",
            "alpha": alpha, "alpha_multiplier": alpha_mult,
            "configuration": record["configuration"],
            "negative_variance_relative_tolerance": negative_variance_tol,
            "inputs_sha256": input_hashes, "code_sha256": code_hashes,
            "field": record["fields"]["raw"]})
    else:
        ratio_path = outdir / f"{tag}_fullgrid_ratio{suffix}{mult_suffix}.json"
        atomic_json(ratio_path, record)
    print(f"[fg] {tag} done ({time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
