"""Reproduce the feature-tracking and grid-resolution evidence in ``def:regions``/``app:grid``.

  (1) FEATURE TRACKING. The region partition is a space-time set: the paper endpoint
      convention thresholds each (x,t) point on its own |d_x u*| after .ravel(), so the shock
      region follows a moving feature rather than occupying a fixed band in x. This
      script measures how far it actually moves.

  (2) GRID CONVERGENCE. Whether the 400-point evaluation grid resolves the feature at
      all, and whether either the primary RMS uncertainty contrast or the secondary
      regional mean-width contrast depends on that resolution.

      WHAT IS RECOMPUTED vs WHAT IS SUBSAMPLED, and why the split is the correct one:

        RECOMPUTED at each n_x -- everything that actually depends on resolution:
          * |d_x u*| by finite differences on the coarsened u* with the coarsened dx.
            This genuinely degrades: the Burgers gradient peak falls 116.8 -> 87.4 ->
            48.8 -> 24.6 as n_x goes 400 -> 200 -> 100 -> 50.
          * the p70/p95 thresholds, over the coarsened set.
          * region membership, and the per-region means.
        Re-deriving is not cosmetic: against merely decimating the 400-grid partition,
        1.2% of points change region at n_x=100 and 5.0% at n_x=50.

        SUBSAMPLED -- sigma, via S[:, ::st], and this is EXACT rather than an
        approximation. step_fullgrid_ratio.py computes
            sigma^2(x,t) = [ ||g(x,t)||^2 - g^T J^T (J J^T + alpha I)^-1 J g ] / alpha
        where J is the TRAINING residual Jacobian (N=6400 rows) and g is the parameter
        gradient of the output at that ONE point. The Cholesky factor depends only on
        (J, alpha); each evaluation point is computed independently of every other. So
        sigma is a pointwise field over (x,t): the decimated grid carries the EXACT sigma
        at each retained location, with no interpolation and nothing to recompute.

        BE PRECISE about what that does and does not claim. The coarsened grids here are
        decimations xs[::s] of the 400-grid -- spacing s*dx, spanning [-1, 1-dx] for even
        s -- NOT linspace(-1,1,n), which is what a fresh run at a smaller EVAL_NX would
        build. Those are different point sets: only 2 of the 200 linspace(-1,1,200) points
        land on the 400-grid, and one can sit half a cell (49.7% of dx400) from the nearest.
        Evaluating THOSE would mean re-running the posterior at new locations on a GPU, not
        re-reading an array. So what this study varies is SAMPLING DENSITY, with an exact
        posterior evaluation at every sample -- the cleaner of the two tests, since it does
        not confound density with a simultaneous change of sample locations.

      In short: the resolution study varies exactly the quantities that resolution
      controls, and leaves alone the one quantity that is grid-independent by
      construction. Re-deriving sigma would be a no-op; re-deriving the partition is not.

WHY THIS FILE EXISTS. Both tables were first computed ad hoc while answering two
questions ("the shock region can move -- and the interface may drift", "what if the
interface width is less than the grid width", 2026-08-03). The numbers went into the
paper and into consistency gate 16, but nothing SHIPPED could regenerate them -- which
fails the standard every other number in the paper meets: claim -> script -> field.
A gate that re-derives a number proves it is right; it does not let a reader produce it.

    python analysis/region_resolution.py            # tables + frozen machine record
    python analysis/region_resolution.py --json     # live machine-readable rebuild
    python analysis/region_resolution.py --check    # fail if record or inputs are stale

Pure numpy, no GPU, ~10 s. Reads only shipped artifacts:
    results/{TAG}/{TAG}_fields.npz              (u_exact on the 400x200 grid)
    results/{TAG}/{TAG}_fullgrid_sigma_raw.npz  (raw-metric sigma on the same grid)
"""
import os, sys, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCHEMA = "region_resolution/3"

P_LO, P_HI = 70.0, 95.0          # the percentiles uq_metrics.assign_regions uses
TRACK = [("allen_cahn_mlp_seed0", "Allen-Cahn MLP seed 0"),
         ("burgers_mlp_seed2", "Burgers MLP seed 2")]
# stiffest sweep settings, for the worst-case width check in sec:setup:regions
STIFF = [("analysis/sweep_results/burgers/NU_0.002/mlp_seed0/burgers_mlp_seed0_fields.npz",
          "Burgers nu=0.002"),
         ("analysis/sweep_results/allen_cahn/EPS2_5e-05/mlp_seed0/allen_cahn_mlp_seed0_fields.npz",
          "Allen-Cahn eps2=5e-05")]
CANON = [("results/burgers_mlp_seed2/burgers_mlp_seed2_fields.npz", "Burgers nu=0.01/pi"),
         ("results/allen_cahn_mlp_seed0/allen_cahn_mlp_seed0_fields.npz", "Allen-Cahn eps2=1e-4")]
CONV = [("burgers_mlp_seed2", "Burgers MLP seed 2"),
        ("burgers_mlp_seed0", "Burgers MLP seed 0"),
        ("allen_cahn_mlp_seed0", "Allen-Cahn MLP seed 0"),
        ("allen_cahn_fourier_seed0", "Allen-Cahn Fourier s0")]
STEPS = (1, 2, 4, 8)


def _grid(tag):
    path = "results/%s/%s_fields.npz" % (tag, tag)
    try:
        with np.load(path) as z:
            required = {"xs", "ts", "u_exact"}
            if not required.issubset(z.files):
                raise ValueError(f"missing {sorted(required - set(z.files))}")
            xs = np.asarray(z["xs"], dtype=float)
            ts = np.asarray(z["ts"], dtype=float)
            nt, nx = len(ts), len(xs)
            U = np.asarray(z["u_exact"], dtype=float).reshape(nt, nx)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if (xs.shape != (400,) or ts.shape != (200,) or not np.all(np.isfinite(xs)) or
            not np.all(np.isfinite(ts)) or not np.all(np.isfinite(U))):
        raise ValueError(f"{path}: invalid 400x200 reference grid")
    return xs, ts, U, nt, nx


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_contract():
    path = os.path.join("calibration_figures", "fullgrid_sigma_metric_contract.json")
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    rows = record.get("files", [])
    if (record.get("schema") != "fullgrid_sigma_metric_contract/3" or
            record.get("sigma_metric") != "raw" or len(rows) != 17 or
            record.get("admitted_field_count") != 17):
        raise ValueError("complete v3 raw-field contract is required")
    bindings = {row.get("tag"): row for row in rows if isinstance(row, dict)}
    if len(bindings) != 17 or None in bindings:
        raise ValueError("raw-field contract contains duplicate or malformed tags")
    return path, bindings


def _shock_mask(U, xs):
    ga = np.abs(np.gradient(U, xs[1] - xs[0], axis=1))
    return ga, ga >= np.percentile(ga.ravel(), P_HI)


def integral_width_cells(U, xs):
    """Width W = integral|d_x u*| dx / max|d_x u*|, per time slice, in CELLS.

    WHY THIS EXISTS ALONGSIDE fwhm_cells. The walk-out FWHM below has a hard floor: it
    steps outward from the peak until the profile drops under half-max and counts the two
    crossing cells, so a perfect ONE-CELL spike returns 2 and a two-cell top-hat returns 3.
    Measured on the Burgers reference it returns 3.00 at n_x = 400, 200, 100 AND 50 -- a
    constant, while the feature is demonstrably degrading. It cannot distinguish "resolved
    with 3 cells" from "sub-cell". Until 2026-08-04 the paper's "never sub-cell" claim rested
    on exactly that number.

    The integral width has no multi-cell crossing-count floor, so it keeps shrinking as the
    grid coarsens (6.70 -> 4.47 -> 3.99 -> 3.93 on the same data). Like any discrete sum/max
    ratio it is nevertheless at least one cell. It is the primary resolution statistic;
    fwhm_cells is retained for continuity, with its larger floor documented.
    """
    ga = np.abs(np.gradient(U, xs[1] - xs[0], axis=1))
    out = []
    for row in ga:
        m = row.max()
        if m <= 0:
            continue
        out.append(float(row.sum() / m))        # dx cancels: width in cells
    return np.array(out)


def fwhm_cells(U, xs):
    """Full width at half maximum of the |d_x u*| peak, per time slice, in cells.

    FLOOR: 2 for a one-cell spike, 3 for a two-cell top-hat; over-reads a true FWHM of 5 as
    6. Uninformative once the feature approaches the grid scale -- see integral_width_cells.
    """
    ga = np.abs(np.gradient(U, xs[1] - xs[0], axis=1))
    nx = U.shape[1]; out = []
    for row in ga:
        if row.max() <= 0:
            continue
        k = int(np.argmax(row)); half = row[k] / 2.0
        l = k
        while l > 0 and row[l] > half:
            l -= 1
        r = k
        while r < nx - 1 and row[r] > half:
            r += 1
        out.append(r - l)
    return np.array(out)


def tracking(tag):
    xs, ts, U, nt, nx = _grid(tag)
    ga, m = _shock_mask(U, xs)
    empty = int(sum(1 for i in range(nt) if not m[i].any()))
    # one side only: these PDEs are symmetric, so the centroid over BOTH branches is
    # identically 0 at every t and hides the motion completely.
    rows = [i for i in range(nt) if (m[i] & (xs > 0)).any()]
    x0 = float(xs[m[rows[0]] & (xs > 0)].mean()) if rows else float("nan")
    x1 = float(xs[m[rows[-1]] & (xs > 0)].mean()) if rows else float("nan")
    w = fwhm_cells(U, xs)
    per_slice = [int(m[i].sum()) for i in range(nt)]
    return dict(tag=tag, n_t=nt, n_x=nx, dx=float(xs[1] - xs[0]),
                empty_slices=empty, first_x=x0, last_x=x1,
                drift_cells=float(abs(x1 - x0) / (xs[1] - xs[0])) if rows else float("nan"),
                fwhm_median=float(np.median(w)), fwhm_min=int(w.min()), fwhm_max=int(w.max()),
                cells_at_t=[(float(ts[i]), per_slice[i]) for i in range(nt)])


def convergence(tag, contract):
    sf = "results/%s/%s_fullgrid_sigma_raw.npz" % (tag, tag)
    if not os.path.exists(sf):
        return None
    xs, ts, U, nt, nx = _grid(tag)
    reference_path = "results/%s/%s_fields.npz" % (tag, tag)
    binding = contract.get(tag)
    if (not binding or binding.get("sigma_metric") != "raw" or
            binding.get("sha256") != sha256(sf) or
            binding.get("reference_field_sha256") != sha256(reference_path)):
        raise ValueError(f"{tag}: convergence fields are not bound by the v3 raw contract")
    try:
        with np.load(sf) as field:
            required = {"sigma", "xs", "ts", "shock_idx", "smooth_idx"}
            if not required.issubset(field.files):
                raise ValueError(f"missing {sorted(required - set(field.files))}")
            S = np.asarray(field["sigma"], dtype=float).reshape(nt, nx)
            sx, stime = np.asarray(field["xs"]), np.asarray(field["ts"])
            stored_sharp = np.asarray(field["shock_idx"], dtype=np.int64)
            stored_bulk = np.asarray(field["smooth_idx"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read {sf}: {exc}") from exc
    if (not np.array_equal(xs, sx) or not np.array_equal(ts, stime) or
            not np.all(np.isfinite(S)) or np.any(S <= 0)):
        raise ValueError(f"{tag}: invalid or misaligned raw posterior field")
    out = []
    for st in STEPS:
        # Us and X are re-processed below (gradient + percentiles recomputed); Ss is a pure
        # subsample, which is exact -- sigma is pointwise in (x,t), see the module docstring.
        X = xs[::st]; Us = U[:, ::st]; Ss = S[:, ::st]
        ga = np.abs(np.gradient(Us, X[1] - X[0], axis=1)).ravel()
        sh = ga >= np.percentile(ga, P_HI)
        sm = ga < np.percentile(ga, P_LO)
        if (np.count_nonzero(ga == np.percentile(ga, P_HI)) or
                np.count_nonzero(ga == np.percentile(ga, P_LO)) or
                int(sh.sum()) != int(0.05 * ga.size) or
                int(sm.sum()) != int(0.70 * ga.size)):
            raise ValueError(f"{tag}: coarsened percentile regions are not exact 5%/70% sets")
        if st == 1 and (not np.array_equal(np.flatnonzero(sh), stored_sharp) or
                        not np.array_equal(np.flatnonzero(sm), stored_bulk)):
            raise ValueError(f"{tag}: full-grid regions disagree with the raw field")
        sig_sh, sig_sm = Ss.ravel()[sh], Ss.ravel()[sm]
        integral_by_time = integral_width_cells(Us, X)
        if integral_by_time.size != Us.shape[0]:
            raise ValueError(f"{tag}: a zero-gradient time slice makes the width audit undefined")
        sharpest_time = int(np.argmin(integral_by_time))
        out.append({
            "n_x": int(len(X)),
            "integral_width_min_cells": float(integral_by_time[sharpest_time]),
            "sharpest_slice_selected_cells": int(sh.reshape(Us.shape)[sharpest_time].sum()),
            "mean_width_contrast": float(sig_sh.mean() / sig_sm.mean()),
            "rms_uncertainty_contrast": float(
                np.sqrt(np.mean(sig_sh ** 2) / np.mean(sig_sm ** 2))),
        })
    return out


def width_stats(path):
    """Width statistics: integral sum/max width AND the walk-out FWHM. The minimum is the binding statistic:
    the grid must resolve the sharpest slice, not the typical one. For Allen-Cahn the two
    differ by ~5x because the interface is broad until it sharpens."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"required resolution-study field is missing: {path}")
    try:
        with np.load(path) as z:
            nt, nx = z["ts"].shape[0], z["xs"].shape[0]
            U = np.asarray(z["u_exact"], dtype=float).reshape(nt, nx)
            X = np.asarray(z["xs"], dtype=float)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read resolution-study field {path}: {exc}") from exc
    w = fwhm_cells(U, X)
    iw = integral_width_cells(U, X)
    return dict(median=float(np.median(w)), min=int(w.min()), max=int(w.max()),
                p5=float(np.percentile(w, 5)), n_slices=int(len(w)),
                integral_width_median=float(np.median(iw)),
                integral_width_min=float(iw.min()))


def build():
    contract_path, contract = _raw_contract()
    rep = {"schema": SCHEMA, "tracking": {}, "convergence": {}, "widths": {},
           "input_sha256": {},
           "code_sha256": {"producer": sha256(__file__)}}
    # Store portable release-relative paths even when the producer runs on Windows.
    rep["input_sha256"][contract_path.replace(os.sep, "/")] = sha256(contract_path)
    for path, lab in CANON + STIFF:
        st = width_stats(path)
        rep["widths"][lab] = st
        rep["input_sha256"][path] = sha256(path)

    for tag, lab in TRACK:
        rep["tracking"][tag] = tracking(tag)
        path = "results/%s/%s_fields.npz" % (tag, tag)
        rep["input_sha256"][path] = sha256(path)
    for tag, lab in CONV:
        c = convergence(tag, contract)
        if c is None:
            raise FileNotFoundError(f"{tag}: required raw full-grid sigma field is missing")
        rep["convergence"][tag] = c
        for suffix in ("fields.npz", "fullgrid_sigma_raw.npz"):
            path = "results/%s/%s_%s" % (tag, tag, suffix)
            rep["input_sha256"][path] = sha256(path)

    if min(row["integral_width_min"] for row in rep["widths"].values()) <= 1.0:
        raise RuntimeError("a claimed production/stiff-sweep feature is not wider than one cell")
    finest_tol = 0.011
    for tag, rows in rep["convergence"].items():
        if [row["n_x"] for row in rows[:2]] != [400, 200]:
            raise RuntimeError(f"{tag}: convergence record does not begin with n_x=400,200")
        for key in ("mean_width_contrast", "rms_uncertainty_contrast"):
            relative = abs(rows[1][key] - rows[0][key]) / rows[0][key]
            if relative > finest_tol:
                raise RuntimeError(f"{tag}: {key} moves {relative:.3%} between the two "
                                   f"finest grids, above {finest_tol:.1%}")
        if any(float(row["integral_width_min_cells"]) <= 1.0 for row in rows):
            raise RuntimeError(f"{tag}: a tested evaluation-grid feature is not wider than one cell")
    burgers_coarsest = rep["convergence"]["burgers_mlp_seed2"][-1]
    if not (burgers_coarsest["n_x"] == 50 and
            burgers_coarsest["sharpest_slice_selected_cells"] <
            burgers_coarsest["integral_width_min_cells"]):
        raise RuntimeError("coarsest-grid Burgers partition is not narrower than the "
                           "sharpest-slice integral-width estimate")
    rep["contracts"] = {
        "minimum_integral_width_cells_gt": 1.0,
        "all_resolution_integral_width_cells_gt": 1.0,
        "finest_grid_relative_change_max": finest_tol,
        "finest_grid_pair": [400, 200],
        "coarsest_burgers_selected_band_narrower_than_integral_width": True,
    }
    return rep


def main():
    os.chdir(ROOT)
    as_json = "--json" in sys.argv
    check = "--check" in sys.argv
    rep = build()
    target = os.path.join("calibration_figures", "region_resolution.json")
    rendered = json.dumps(rep, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if as_json:
        print(rendered, end="")
        return 0
    if check:
        if not os.path.isfile(target) or open(target, encoding="utf-8").read() != rendered:
            raise SystemExit(f"REGION RESOLUTION CHECK FAILED: {target} is absent or stale")
        print("REGION RESOLUTION CHECK: PASS")
        return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target + ".partial", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered); handle.flush(); os.fsync(handle.fileno())
    os.replace(target + ".partial", target)

    print("=" * 74)
    print("1. FEATURE TRACKING -- the partition is a space-time set")
    print("=" * 74)
    for tag, lab in TRACK:
        r = rep["tracking"][tag]
        print("\n%s   grid %dx%d, dx=%.5f" % (lab, r["n_t"], r["n_x"], r["dx"]))
        print("   gradient-peak FWHM: median %.0f cells (min %d, max %d)"
              % (r["fwhm_median"], r["fwhm_min"], r["fwhm_max"]))
        print("   time slices with NO shock points: %d of %d" % (r["empty_slices"], r["n_t"]))
        if not np.isnan(r["drift_cells"]):
            print("   right-branch interface: %+.4f -> %+.4f  = %.1f grid cells"
                  % (r["first_x"], r["last_x"], r["drift_cells"]))
        c = dict(r["cells_at_t"])
        for t in (0.13, 0.50):
            k = min(c, key=lambda x: abs(x - t))
            print("   shock cells at t~%.2f: %d" % (t, c[k]))
    print("\n   (both PDEs are symmetric, so the centroid over BOTH branches is 0.000")
    print("    at every t -- the motion is only visible on one side.)")

    print()
    print("=" * 74)
    print("1b. GRADIENT-PEAK WIDTH, canonical and stiffest -- min is the binding statistic")
    print("=" * 74)
    print("  %-26s%12s%9s%11s%8s" % ("configuration", "intW median", "intW min",
                                         "FWHM med", "FWHM min"))
    for lab, st in rep["widths"].items():
        print("  %-26s%12.2f%9.2f%11.1f%8d"
              % (lab, st["integral_width_median"], st["integral_width_min"],
                 st["median"], st["min"]))
    print("\n  intW = discrete integral width: no multi-cell crossing-count floor, but >=1 cell.")
    print("  FWHM = walk-out, floors at 2-3 cells and reads 3.00 at every n_x here --")
    print("  retained for continuity only.")
    print("  NOTE the percentile partition cannot detect under-resolution: the shock region")
    print("  is the top 5%% of points BY CONSTRUCTION, so it is non-empty at every n_x (500")
    print("  points even at n_x=50). Its non-emptiness is no evidence at all; the widths and")
    print("  the convergence table below are.")
    print("\n  The grid must resolve the SHARPEST slice, not the typical one. Allen-Cahn's")
    print("  median sits far above its minimum because the interface is broad until it")
    print("  sharpens; the minimum is the reported discrete width diagnostic, not a continuum proof.")

    print()
    print("=" * 74)
    print("2. GRID CONVERGENCE -- partition re-derived at each resolution")
    print("=" * 74)
    print("  Secondary regional mean-width contrast:")
    print("  %-26s%9s%9s%9s%9s" % ("configuration", "n_x=400", "200", "100", "50"))
    for tag, lab in CONV:
        if tag not in rep["convergence"]:
            continue
        v = [r["mean_width_contrast"] for r in rep["convergence"][tag]]
        print("  %-26s" % lab + "".join("%9.3f" % r for r in v))
    print("\n  Primary RMS uncertainty contrast:")
    print("  %-26s%9s%9s%9s%9s" % ("configuration", "n_x=400", "200", "100", "50"))
    for tag, lab in CONV:
        if tag not in rep["convergence"]:
            continue
        v = [r["rms_uncertainty_contrast"] for r in rep["convergence"][tag]]
        print("  %-26s" % lab + "".join("%9.3f" % r for r in v))
    print()
    for tag, lab in CONV:
        if tag not in rep["convergence"]:
            continue
        mw = [r["mean_width_contrast"] for r in rep["convergence"][tag]]
        rr = [r["rms_uncertainty_contrast"] for r in rep["convergence"][tag]]
        print("  %-26s 400->200 mean %5.2f%%, RMS %5.2f%%; 200->100 mean %5.2f%%, RMS %5.2f%%"
              % (lab, 100 * abs(mw[1] - mw[0]) / mw[0],
                 100 * abs(rr[1] - rr[0]) / rr[0],
                 100 * abs(mw[2] - mw[1]) / mw[1],
                 100 * abs(rr[2] - rr[1]) / rr[1]))
    print("\n  Halving from n_x=400 to 200 must move either contrast by <=1.1% for the")
    print("  plateau claim in sec:setup:regions to hold; the consistency gate enforces both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
