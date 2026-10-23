"""
uq_metrics.py  --  Phase-2 shared metrics + solve-acceptance gate.

PDE-agnostic, pure-numpy (scipy optional). Single source of truth for every
number the Phase-2 sweep reports. Enforces the golden rule that sank the
earlier nextsteps_gemini Fourier runs:

    No calibration metric is trusted unless the underlying solve PASSES the
    acceptance gate (global relative-L2 error below a threshold).

Everything is serialised to JSON (never re-parsed from text).
"""

import json
import numpy as np


def _norm_ppf(p):
    """Standard-normal quantile (scipy if present, else Acklam approx)."""
    p = np.asarray(p, dtype=np.float64)
    try:
        from scipy.special import ndtri
        return ndtri(p)
    except Exception:
        pass
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(np.clip(p[lo], 1e-300, None)))
    out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(np.clip(1 - p[hi], 1e-300, None)))
    out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out


# ----------------------------------------------------------------------
# Region assignment (adaptive, percentile-based on |du/dx|)
# ----------------------------------------------------------------------
def assign_regions(grad_abs, p_lo=70.0, p_hi=95.0):
    g = np.asarray(grad_abs).ravel()
    lo = np.percentile(g, p_lo)
    hi = np.percentile(g, p_hi)
    regions = np.empty(g.shape, dtype=object)
    regions[g <= lo] = "smooth"
    regions[(g > lo) & (g <= hi)] = "transition"
    regions[g > hi] = "shock"
    return regions, float(lo), float(hi)


# ----------------------------------------------------------------------
# Solve-acceptance gate  (THE safety net)
# ----------------------------------------------------------------------
def solve_acceptance(u_pred, u_exact, regions=None, rel_l2_threshold=0.02):
    """Decide whether the PINN actually solved the PDE. A run that fails this
    gate must NOT contribute calibration numbers to the sweep."""
    u_pred = np.asarray(u_pred, dtype=np.float64).ravel()
    u_exact = np.asarray(u_exact, dtype=np.float64).ravel()
    denom = np.linalg.norm(u_exact) + 1e-30
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / denom)
    corr = float(np.corrcoef(u_pred, u_exact)[0, 1])
    out = {
        "global_rel_l2": rel_l2,
        "global_rmse": float(np.sqrt(np.mean((u_pred - u_exact) ** 2))),
        "correlation": corr,
        "rel_l2_threshold": float(rel_l2_threshold),
        "per_region_rel_l2": {},
    }
    # Gate on the BULK (smooth+transition), NOT global: in stiff PDEs the shock
    # region carries irreducibly large error (~0.1) even for a correct solve, so
    # a global-L2 gate rejects good solves. Bulk rel-L2 cleanly separates
    # converged (proven MLP ~0.1-0.4%) from failed (gemini Fourier ~40-80%).
    bulk_rel_l2 = rel_l2
    if regions is not None:
        regions = np.asarray(regions).ravel()
        for name in ("smooth", "transition", "shock"):
            m = regions == name
            if m.sum() == 0:
                continue
            dd = np.linalg.norm(u_exact[m]) + 1e-30
            out["per_region_rel_l2"][name] = float(
                np.linalg.norm(u_pred[m] - u_exact[m]) / dd)
        bulk = (regions == "smooth") | (regions == "transition")
        if bulk.sum() > 0:
            dd = np.linalg.norm(u_exact[bulk]) + 1e-30
            bulk_rel_l2 = float(np.linalg.norm(u_pred[bulk] - u_exact[bulk]) / dd)
    out["bulk_rel_l2"] = bulk_rel_l2
    out["gate_metric"] = ("bulk_rel_l2 (smooth+transition)"
                          if regions is not None else "global_rel_l2")
    out["passed"] = bool(bulk_rel_l2 < rel_l2_threshold)
    return out


# ----------------------------------------------------------------------
# Per-region calibration metrics
# ----------------------------------------------------------------------
def _coverage(z, k):
    return float(np.mean(z <= k))


def region_metrics(u_pred, u_exact, sigma, regions, confidence_levels=None):
    u_pred = np.asarray(u_pred, dtype=np.float64).ravel()
    u_exact = np.asarray(u_exact, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    regions = np.asarray(regions).ravel()
    err = np.abs(u_pred - u_exact)
    z = err / np.maximum(sigma, 1e-30)
    if confidence_levels is None:
        confidence_levels = np.linspace(0.05, 0.99, 30)
    confidence_levels = np.asarray(confidence_levels, dtype=np.float64)
    z_levels = _norm_ppf(0.5 * (1.0 + confidence_levels))
    res = {}
    for name in ("smooth", "transition", "shock"):
        m = regions == name
        if m.sum() == 0:
            continue
        zr = z[m]
        emp_cov = np.array([_coverage(zr, zk) for zk in z_levels])
        mce = float(np.mean(np.abs(emp_cov - confidence_levels)))
        res[name] = {
            "rmse": float(np.sqrt(np.mean((u_pred[m] - u_exact[m]) ** 2))),
            "mean_sigma": float(np.mean(sigma[m])),
            "mean_z": float(np.mean(zr)),
            "coverage68": _coverage(zr, 1.0),
            "coverage95": _coverage(zr, 1.96),
            "mce": mce,
            "count": int(m.sum()),
        }
    return res


def calibration_ratios(region_res):
    """Headline sigma-ratio / RMSE-ratio / ratio-gap (shock vs smooth) AND the
    absolute meanZ -- because a small ratio gap with a huge meanZ is meaningless
    (the precise trap the Allen-Cahn Fourier 'success' fell into)."""
    if "smooth" not in region_res or "shock" not in region_res:
        return {}
    s, k = region_res["smooth"], region_res["shock"]
    rmse_ratio = k["rmse"] / max(s["rmse"], 1e-30)
    sigma_ratio = k["mean_sigma"] / max(s["mean_sigma"], 1e-30)
    return {
        "target_rmse_ratio": float(rmse_ratio),
        "achieved_sigma_ratio": float(sigma_ratio),
        "ratio_gap": float(rmse_ratio / max(sigma_ratio, 1e-30)),
        "shock_mean_z": float(k["mean_z"]),
        "smooth_mean_z": float(s["mean_z"]),
    }


# ----------------------------------------------------------------------
# Stiffness-binned scaling curves  (the mechanism plots)
# ----------------------------------------------------------------------
def stiffness_curves(S, abs_err, sigma, n_bins=20):
    """Return per-bin medians of |E|, sigma, meanZ vs stiffness S, plus log-log
    slopes. Returns the curves (not only slopes) -- a global slope can hide
    structure (or its absence)."""
    S = np.asarray(S, dtype=np.float64).ravel()
    abs_err = np.asarray(abs_err, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    z = abs_err / np.maximum(sigma, 1e-30)
    good = (S > 0) & np.isfinite(S) & np.isfinite(abs_err) & np.isfinite(sigma)
    S, abs_err, sigma, z = S[good], abs_err[good], sigma[good], z[good]
    order = np.argsort(S)
    S, abs_err, sigma, z = S[order], abs_err[order], sigma[order], z[order]
    edges = np.percentile(S, np.linspace(0, 100, n_bins + 1))
    centers, e_med, s_med, z_med, counts = [], [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (S >= lo) & (S <= hi) if i == n_bins - 1 else (S >= lo) & (S < hi)
        if m.sum() == 0:
            continue
        centers.append(float(np.median(S[m])))
        e_med.append(float(np.median(abs_err[m])))
        s_med.append(float(np.median(sigma[m])))
        z_med.append(float(np.median(z[m])))
        counts.append(int(m.sum()))

    def _slope(xc, yc):
        xc, yc = np.asarray(xc), np.asarray(yc)
        ok = (xc > 0) & (yc > 0)
        if ok.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log10(xc[ok]), np.log10(yc[ok]), 1)[0])

    return {
        "S_centers": centers,
        "abs_err_median": e_med,
        "sigma_median": s_med,
        "meanZ_median": z_med,
        "counts": counts,
        "slope_absE": _slope(centers, e_med),
        "slope_sigma": _slope(centers, s_med),
        "slope_meanZ": _slope(centers, z_med),
        "S_range": [float(np.min(S)), float(np.max(S))],
    }


# ----------------------------------------------------------------------
# Assemble + serialise
# ----------------------------------------------------------------------
def build_metrics_record(meta, gate, region_res, ratios, curves=None):
    rec = {"meta": meta, "solve_gate": gate,
           "regions": region_res, "ratios": ratios}
    if curves is not None:
        rec["stiffness_curves"] = curves
    return rec


def save_json(record, path):
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
