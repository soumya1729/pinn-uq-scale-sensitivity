"""Stable complementary accounting for disclosed legacy loading CDFs.

``total_gg - captured`` is a signed closure residual of the old inverse lift, not
an independently measured null-space energy.  It must not be clamped before
combining it with the cumulative loading: that loses the cancellation of the
unstable low-mode contribution.  Physical-energy gates apply to the completed
complement and variance instead.  Current-J projector/variance measurements are
still required; agreement between CDF consumers is not independent validation.
"""
from __future__ import annotations

import math
import numpy as np


ACCOUNTING_CONTRACT = "legacy_cdf_stiff_complement/1"
DEFAULT_REL_TOL = 5e-7


def load_region(record, eigenvalues, label="CDF", relative_tolerance=DEFAULT_REL_TOL):
    """Validate the historical record and retain its signed closure discrepancy."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    cumulative = np.asarray(record.get("cum_loading", []), dtype=np.float64)
    total = float(record.get("total_gg", math.nan))
    captured = float(record.get("captured", math.nan))
    recorded_null = float(record.get("nullspace", math.nan))
    if (not math.isfinite(relative_tolerance) or not 0 < relative_tolerance <= DEFAULT_REL_TOL or
            lam.ndim != 1 or not len(lam) or not np.all(np.isfinite(lam)) or
            np.any(lam < 0) or np.any(np.diff(lam) < 0) or
            cumulative.shape != lam.shape or not np.all(np.isfinite(cumulative)) or
            not all(math.isfinite(value) for value in (total, captured, recorded_null)) or
            total <= 0):
        raise SystemExit(f"{label}: malformed legacy CDF energy/spectrum")
    tolerance = relative_tolerance * total
    closure = total - captured
    if (np.any(np.diff(cumulative) < -tolerance) or cumulative[0] < -tolerance or
            closure < -tolerance or recorded_null < -tolerance or
            abs(float(cumulative[-1]) - captured) > tolerance or
            abs(closure - recorded_null) > tolerance):
        raise SystemExit(f"{label}: inconsistent legacy CDF closure/loading")
    return {
        "cum": cumulative, "tot": total, "captured": captured,
        "closure_residual": closure, "recorded_nullspace": recorded_null,
        "relative_tolerance": float(relative_tolerance), "label": label,
    }


def split_parts(region, k):
    """Return completed null-inclusive sloppy/stiff energies after k Gram modes.

    Only the final physical pair may receive bounded roundoff clipping; its sum
    remains the measured total.  In particular, the all-sloppy state is exact.
    """
    cumulative = region["cum"]
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= len(cumulative):
        raise SystemExit(f"{region['label']}: invalid spectral split index {k!r}")
    total = region["tot"]
    if k == len(cumulative):
        return total, 0.0
    below = float(cumulative[k - 1]) if k else 0.0
    stiff = region["captured"] - below
    sloppy = total - stiff
    tolerance = region["relative_tolerance"] * total
    if not (math.isfinite(sloppy) and math.isfinite(stiff)) or min(sloppy, stiff) < -tolerance:
        raise SystemExit(
            f"{region['label']}: invalid completed complement at k={k}: "
            f"sloppy={sloppy}, stiff={stiff}"
        )
    sloppy = min(max(sloppy, 0.0), total)
    return sloppy, total - sloppy


def variance(region, eigenvalues, alpha, k=None):
    """Evaluate full (or sloppy-block) variance without a clamped null addend.

    The full expression is (total - sum(c_i^2 lambda_i/(alpha+lambda_i)))/alpha.
    It damps the unreliable tiny-eigenvalue lift rather than weighting it by
    another inverse eigenvalue.  It is a CDF-derived comparator, not current-J
    evidence.  With k supplied, use the completed sloppy energy for that block.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    if (lam.shape != region["cum"].shape or not np.all(np.isfinite(lam)) or
            np.any(lam < 0) or not math.isfinite(alpha) or alpha <= 0):
        raise SystemExit(f"{region['label']}: invalid variance spectrum/alpha")
    count = len(lam) if k is None else k
    sloppy, _ = split_parts(region, count)
    c2 = np.diff(np.concatenate([[0.0], region["cum"]]))
    correction = float(np.sum(c2[:count] * lam[:count] / (alpha + lam[:count])))
    value = (sloppy - correction) / alpha
    upper = sloppy / alpha
    tolerance = region["relative_tolerance"] * region["tot"] / alpha
    if not math.isfinite(value) or value < -tolerance or value > upper + tolerance:
        raise SystemExit(f"{region['label']}: non-physical completed CDF variance {value}")
    return min(max(value, 0.0), upper)


def closure_diagnostic(region):
    """This diagnostic is expressly not a physical null-space estimate."""
    return {
        "total_gg": region["tot"], "captured": region["captured"],
        "signed_closure_residual": region["closure_residual"],
        "signed_closure_relative_to_total": region["closure_residual"] / region["tot"],
        "recorded_legacy_nullspace": region["recorded_nullspace"],
    }
