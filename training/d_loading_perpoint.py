"""Per-point sloppy/stiff loading, averaged over each region (A3(b); unblocks C5 and C6).

WHAT THIS MEASURES, AND WHY IT IS NOT `d_loading_cdf.py`. That script takes the region-MEAN
output gradient g_r = <g(x)>_r and reports the loading of that single vector below alpha. The
quantity Theorem `thm:ceiling` needs when the contrast is aggregated per point is

    Q_{r,sl} = < ||Pi_sl g(x)||^2 >_r ,     Q_{r,st} = < ||Pi_st g(x)||^2 >_r

-- the region AVERAGE of each point's loading. These are different objects, and an EXACT
identity relates them: with kappa_r = M_r / Q_r in [0,1],

    M_S / M_B = (kappa_S / kappa_B) * (Q_S / Q_B),      Q_r - M_r = <||Pi_sl(g - <g>_r)||^2>_r,

so the gap is precisely the within-region dispersion of the projected tangent. (An earlier
version of this note claimed no identity relates them, which is wrong and contradicted the
coherence factor this same file computes.) A9 flags this as the
place a headline can move against us: the 182--1130 asymmetry is a region-mean-tangent number,
and this script is what says whether the per-point version agrees.

Pi_sl projects onto {lambda_i <= alpha}, which INCLUDES the Fisher-null directions. It is
computed as the complement of the stiff projector, ||Pi_sl g||^2 = ||g||^2 - ||V_st^T g||^2 --
never as a sum of c^2/lambda over sloppy modes. See the long note at the projector for why:
that sum both divides by the worst-resolved eigenvalues and silently drops the null space,
and the second defect alone moves Burgers-Fourier by 7.4%.

C6's numerics, all reported so the loadings are not merely produced but checked:
  * raw inverse-lift loss of orthogonality, followed by Cholesky--QR orthogonalisation of the
    SAME stiff subspace and a gate on the corrected projector, max |V_st^T V_st - I|
  * eigenpair residual over the stiff modes
  * the eigenvalue separation either side of alpha
  * a "resolved-sloppy" variant excluding lambda <= lam_max*FLOOR_REL, reported as a distinct
    labelled quantity rather than as an error bar on Pi_sl
  * end-to-end: <sigma^2>_r rebuilt from the loadings against the stored per-point sigma field

Env: PDE ARCH SEED FOURIER_SCALE [SCOPE=full] [EPS_MULT=1.0] [CHUNK=200] [FLOOR_REL=1e-12] [OUTDIR]
Out: results/{TAG}/{TAG}_loading_perpoint{_SCOPE}.json
"""
import os, json, time, math, hashlib
import numpy as np, jax.numpy as jnp
from scipy.linalg import cho_factor, cho_solve, solve_triangular
import common as C, laplace_core as lc


SCHEMA = "loading_perpoint/3"      # v3 separates eigensolver accuracy from cutoff stability
CANONICAL_TAGS = {
    "burgers_mlp_seed2", "burgers_fourier_seed2",
    "allen_cahn_mlp_seed0", "allen_cahn_fourier_seed0",
}


def main():
    t0 = time.time()
    c = C.get_config()
    TAG = f"{c.pde}_{c.arch}_seed{c.seed}"
    OUT = os.environ.get("OUTDIR", os.path.join("results", TAG)); os.makedirs(OUT, exist_ok=True)
    SCOPE = os.environ.get("SCOPE", "full")
    EPS_MULT = float(os.environ.get("EPS_MULT", 1.0))
    CH = int(os.environ.get("CHUNK", 200))
    WANT_RES = os.environ.get("WANT_RESOLVED", "0") == "1"
    FLOOR_REL = float(os.environ.get("FLOOR_REL", 1e-12))
    _sc = "" if SCOPE == "full" else f"_{SCOPE}"
    _eps_slug = format(EPS_MULT, ".12g").replace("-", "m").replace("+", "p").replace(".", "p")
    _eps_suffix = "" if EPS_MULT == 1.0 else f"_epsmult{_eps_slug}"
    _production = SCOPE == "full" and EPS_MULT == 1.0

    # ---- FAIL CLEAN, AND FAIL FIRST -----------------------------------------------------------
    # Everything that can fail without touching a GPU happens before anything expensive, and the
    # stale target is removed before ANY of it. Previously `build_context` and the MAP check ran
    # first, so a context failure left the previous JSON in place, indistinguishable to every
    # downstream reader from a fresh one. The resolved-sloppy variant writes to its own filename
    # so that enabling it cannot overwrite the primary artifact.
    _floor_slug = format(FLOOR_REL, ".3e").replace("-", "m").replace("+", "p").replace(".", "p")
    _resolved_suffix = f"_resolved_floor{_floor_slug}" if WANT_RES else ""
    _out = os.path.join(OUT, f"{TAG}_loading_perpoint{_sc}{_eps_suffix}{_resolved_suffix}.json")
    if os.path.exists(_out):
        os.remove(_out)
        print(f"[lpp] removed stale {os.path.basename(_out)} before recomputing", flush=True)

    if os.environ.get("RAW_METRIC", "0") != "1":
        raise SystemExit("[lpp] refusing to run outside the raw metric: set RAW_METRIC=1.")
    pk = os.environ.get("MAP_PKL", os.path.join(OUT, f"{TAG}_params.pkl"))
    if not os.path.exists(pk):
        raise SystemExit(f"[lpp] {TAG}: no MAP at {pk}. Exiting nonzero -- a silent skip inside "
                         f"an array task is indistinguishable from a success.")

    # The stored sigma field is REQUIRED in production: it is the only genuinely independent
    # artifact here, and both the regional sandwich and the J cross-check are built on it.
    sig_p = os.path.join(OUT, f"{TAG}_fullgrid_sigma_raw.npz")
    if _production and not os.path.exists(sig_p):
        raise SystemExit(f"[lpp] {TAG}: {os.path.basename(sig_p)} absent. Fatal in production -- "
                         f"without it there is no independent sigma to check against.")
    # The loading-CDF cross-check is CANONICAL-ONLY. The four configurations quoted by the paper
    # must have it; other seeds do not need it. This is tag-based rather than existence-based so
    # deleting a canonical CDF cannot silently disable its validation.
    cdf_p = os.path.join(OUT, f"{TAG}_loading_cdf.json")
    _want_cdf = _production and TAG in CANONICAL_TAGS
    if _want_cdf and not os.path.exists(cdf_p):
        raise SystemExit(f"[lpp] {TAG}: canonical run requires {os.path.basename(cdf_p)}; "
                         f"refusing to remove its mean-tangent validation.")
    print(f"[lpp] {TAG}: sigma-field check {'ON' if _production else 'off'}, "
          f"loading_cdf cross-check {'ON' if _want_cdf else 'not required'}", flush=True)

    # Fourier runs must use the FROZEN sidecar. A PRNG fallback silently produces a different
    # feature matrix from the one the network was trained with, and the resulting numbers look
    # entirely plausible.
    fourier_b_path = None
    if c.arch == "fourier":
        _bp = os.path.abspath(os.path.join(OUT, f"{TAG}_fourier_B.npz"))
        if _production and not os.path.exists(_bp):
            raise SystemExit(f"[lpp] {TAG}: no frozen Fourier sidecar at {_bp}; refusing to fall "
                             f"back to the PRNG.")
        if _production:
            inherited = os.environ.get("FOURIER_B_NPZ")
            if inherited:
                inherited_abs = (inherited if inherited == "derive" else
                                 os.path.abspath(os.path.expanduser(inherited)))
                if inherited_abs != _bp:
                    raise SystemExit(f"[lpp] {TAG}: FOURIER_B_NPZ={inherited!r} would override "
                                     f"the frozen sidecar {_bp}.")
            os.environ["FOURIER_B_NPZ"] = _bp
            fourier_b_path = _bp

    ctx = C.build_context(c)
    params = C.load_map(pk)
    eng = lc.run(ctx, c, params, SCOPE)
    J = np.asarray(eng["J"]); N = int(eng["N"]); alpha = float(eng["alpha"])
    grad_u = eng["grad_u"]; sub = eng["sub_map"]; P = int(eng["n_sub"])

    # Build and validate the evaluation grid before the expensive spectrum/sweep. The released
    # sigma artifact carries the grid and region indices; matching only its length is insufficient
    # because a reordered 80k-vector is still 80k long.
    xs = np.linspace(-1, 1, c.eval_nx); ts = np.linspace(0, 1, c.eval_nt)
    XX, TT = np.meshgrid(xs, ts); Xr, Tr = XX.ravel(), TT.ravel()
    Uref = ctx.grid_reference(xs, ts)
    ga = np.abs(np.gradient(Uref, xs[1] - xs[0], axis=1)).ravel()
    p70, p95 = np.percentile(ga, 70), np.percentile(ga, 95)
    regions = {"sharp": np.where(ga >= p95)[0], "bulk": np.where(ga < p70)[0]}
    n = len(Xr)

    s2_stored = None
    if _production:
        with np.load(sig_p) as _z:
            required = {"sigma", "xs", "ts", "shock_idx", "smooth_idx"}
            missing = sorted(required.difference(_z.files))
            if missing:
                raise SystemExit(f"[lpp] {TAG}: stored sigma artifact is missing keys {missing}.")
            sigma_stored = np.asarray(_z["sigma"]).ravel()
            xs_stored = np.asarray(_z["xs"]).ravel()
            ts_stored = np.asarray(_z["ts"]).ravel()
            sharp_stored = np.asarray(_z["shock_idx"], dtype=int).ravel()
            bulk_stored = np.asarray(_z["smooth_idx"], dtype=int).ravel()
        if sigma_stored.shape[0] != n:
            raise SystemExit(f"[lpp] {TAG}: stored sigma field has {sigma_stored.shape[0]} points, "
                             f"the grid has {n}.")
        grid_atol = 10 * np.finfo(np.float64).eps
        if (xs_stored.shape != xs.shape or ts_stored.shape != ts.shape or
                not np.allclose(xs_stored, xs, rtol=0.0, atol=grid_atol) or
                not np.allclose(ts_stored, ts, rtol=0.0, atol=grid_atol)):
            raise SystemExit(f"[lpp] {TAG}: stored sigma grid does not match the current grid.")
        if (not np.array_equal(sharp_stored, regions["sharp"]) or
                not np.array_equal(bulk_stored, regions["bulk"])):
            raise SystemExit(f"[lpp] {TAG}: stored sigma region indices do not match the current "
                             f"sharp/bulk definitions.")
        if not np.isfinite(sigma_stored).all() or np.any(sigma_stored < 0):
            raise SystemExit(f"[lpp] {TAG}: stored sigma contains non-finite or negative values.")
        s2_stored = sigma_stored ** 2

    G = J @ J.T
    if _production and G.dtype != np.float64:
        raise SystemExit(f"[lpp] {TAG}: production eigensystem must be float64, got {G.dtype}.")
    lam_raw, W = np.linalg.eigh(G)
    spectral_scale = float(max(np.abs(lam_raw).max(), 1e-300))
    neg_eig_abs = float(max(-lam_raw.min(), 0.0))
    lam = np.clip(lam_raw, 0.0, None)
    Lam = float(lam.max())

    eps = alpha * EPS_MULT
    sl = lam <= eps
    st = ~sl
    k_st = int(st.sum())
    if not sl.any():
        raise SystemExit(f"[lpp] {TAG}: no sloppy modes at eps = {eps:.6g}; cannot certify the "
                         f"cutoff from both sides.")
    if k_st == 0:
        raise SystemExit(f"[lpp] {TAG}: no stiff modes at eps = {eps:.6g}; the complement would "
                         f"be the whole tangent and the split is meaningless.")

    # The inverse lift V = J^T W Lambda^{-1/2} has the right stiff SUBSPACE, but it is not a QR
    # factorisation. Its forward loss of orthogonality is amplified by ||G||/lambda_min(stiff),
    # so demanding O(k eps_mach) here (as schema-v3 initially did) is mathematically the wrong
    # gate. The six first-wave failures, 1.1e-10--1.7e-9, were this expected lift error rather
    # than evidence of a bad eigensystem.
    #
    # Correct the basis before using it as a projector. If S=B^T B=L L^T, then
    #
    #       Q = B L^{-T},                 Q^T Q = L^{-1} S L^{-T} = I,
    #
    # and Q spans exactly the same stiff subspace. This is one Cholesky--QR step; S is already
    # within sqrt(eps_mach) of I, so it is the well-conditioned regime for that algorithm. The
    # raw defect is retained as a diagnostic and gated independently so the correction cannot
    # hide a structurally wrong inverse lift. Q is applied IMPLICITLY: for C=gB, gQ is obtained by
    # solving L (gQ)^T=C^T. Materialising Q would add another multi-GB P-by-k array and P*k^2
    # work; the triangular solve adds only chunk*k^2 work and represents the identical projector.
    EPSM = float(np.finfo(G.dtype).eps)
    B_st = (J.T @ W[:, st]) / np.sqrt(np.maximum(lam[st], 1e-300))[None, :]
    _raw_gram = B_st.T @ B_st
    _raw_gram = 0.5 * (_raw_gram + _raw_gram.T)
    _raw_orth_err = _raw_gram - np.eye(k_st)
    orth_raw = float(np.abs(_raw_orth_err).max())
    orth_inf_raw = float(np.linalg.norm(_raw_orth_err, ord=np.inf))
    raw_basis_condition_upper_bound = math.sqrt((1 + orth_inf_raw) / (1 - orth_inf_raw)) \
        if orth_inf_raw < 1 else math.inf
    RAW_ORTH_CAP = math.sqrt(EPSM)
    RAW_ORTH_TOL = float(os.environ.get("RAW_ORTH_TOL", RAW_ORTH_CAP))
    if (not math.isfinite(RAW_ORTH_TOL) or RAW_ORTH_TOL <= 0 or
            RAW_ORTH_TOL > RAW_ORTH_CAP):
        raise SystemExit(f"[lpp] {TAG}: RAW_ORTH_TOL must be finite, positive, and no larger "
                         f"than sqrt(eps_mach)={RAW_ORTH_CAP:.3e}; got {RAW_ORTH_TOL!r}.")
    if orth_inf_raw > RAW_ORTH_TOL:
        raise SystemExit(
            f"[lpp] {TAG}: raw stiff inverse lift is too far from orthonormal for a "
            f"roundoff-only Cholesky--QR correction: ||V^T V-I||_inf={orth_inf_raw:.3e} > "
            f"the small-correction tolerance {RAW_ORTH_TOL:.3e}."
        )
    try:
        _lift_chol = np.linalg.cholesky(_raw_gram)
    except np.linalg.LinAlgError as exc:
        raise SystemExit(
            f"[lpp] {TAG}: raw stiff inverse-lift Gram is not positive definite; "
            "cannot construct an orthogonal projector."
        ) from exc

    # Validate the implicit corrected basis entirely in k-space. This is Q^T Q without forming Q.
    _qgram_left = solve_triangular(_lift_chol, _raw_gram, lower=True,
                                   overwrite_b=False, check_finite=False)
    _qgram = solve_triangular(_lift_chol, _qgram_left.T, lower=True,
                              overwrite_b=True, check_finite=False).T
    _orth_err = _qgram - np.eye(k_st)
    orth = float(np.abs(_orth_err).max())
    orth_inf = float(np.linalg.norm(_orth_err, ord=np.inf))

    def stiff_coordinates(g):
        """Return gQ for the implicit Cholesky--QR basis Q=B_st L^{-T}."""
        c_raw = np.asarray(g) @ B_st
        c_t = solve_triangular(_lift_chol, c_raw.T, lower=True,
                               overwrite_b=True, check_finite=False)
        return c_t.T
    # Per-mode residuals are used for a scale-free eigensolver backward-error test. Cutoff
    # classification is a separate question and is checked on both bracketing modes below.
    _R = G @ W[:, st] - W[:, st] * lam[st][None, :]
    _rn = np.linalg.norm(_R, axis=0)
    # TWO RESIDUALS, AND ONLY THE BACKWARD ERROR IS AN EIGENSOLVER CORRECTNESS GATE.
    #
    # `backward` is the standard normwise backward error
    # ||G w - lam w|| / ((||G|| + |lam|)||w||). LAPACK accuracy scales with N*eps_mach,
    # so the gate is expressed as a multiple of that dimension-aware scale.
    #
    # `resid` divides instead by (|lam| + alpha), which is what the LOADING is sensitive to,
    # since sigma^2 weights each mode by 1/(alpha + lam). It is reported, not gated. Gating it
    # was a mistake: with ||G|| ~ 1.9e7 the roundoff scale eps_mach*||G|| is ~4e-9, while near
    # alpha the denominator is only ~2*alpha. The resulting ~1e-10 diagnostic is compatible with
    # a backward-stable decomposition and must not itself be used as the correctness threshold.
    backward_per_mode = _rn / (spectral_scale + np.abs(lam[st]))
    backward = float(backward_per_mode.max())
    per_mode = _rn / (np.abs(lam[st]) + alpha)
    resid = float(per_mode.max())
    _near = np.argsort(lam[st])[:min(16, k_st)]
    resid_near = float(per_mode[_near].max())
    _lo_i = int(np.flatnonzero(sl)[-1])
    _hi_i = int(np.flatnonzero(st)[0])
    _boundary_i = np.array([_lo_i, _hi_i])
    _boundary_R = (G @ W[:, _boundary_i]
                   - W[:, _boundary_i] * lam[_boundary_i][None, :])
    _boundary_rn = np.linalg.norm(_boundary_R, axis=0)
    margin_below = float(eps - lam[_lo_i])
    margin_above = float(lam[_hi_i] - eps)
    roundoff_abs = float(N * EPSM * spectral_scale)
    boundary_scales = np.maximum(_boundary_rn, roundoff_abs)
    boundary_safety = np.array([margin_below, margin_above]) / boundary_scales
    gap = {"lam_just_below_eps": float(lam[_lo_i]),
           "lam_just_above_eps": float(lam[_hi_i]),
           "margin_below": margin_below, "margin_above": margin_above,
           "relative_margin_below": margin_below / eps,
           "relative_margin_above": margin_above / eps,
           "boundary_residual_below": float(_boundary_rn[0]),
           "boundary_residual_above": float(_boundary_rn[1]),
           "roundoff_scale_N_eps_normG": roundoff_abs,
           "classification_safety_below": float(boundary_safety[0]),
           "classification_safety_above": float(boundary_safety[1])}

    BACKWARD_FACTOR = float(os.environ.get("BACKWARD_FACTOR", 100.0))
    BACKWARD_TOL = BACKWARD_FACTOR * N * EPSM
    ORTH_FACTOR = float(os.environ.get("ORTH_FACTOR", 100.0))
    NEG_TOL = float(os.environ.get("NEG_TOL", 1e-10))
    CUTOFF_SAFETY = float(os.environ.get("CUTOFF_SAFETY", 10.0))
    PSD_FACTOR = float(os.environ.get("PSD_FACTOR", 10.0))
    for _name, _value in (("BACKWARD_FACTOR", BACKWARD_FACTOR),
                          ("ORTH_FACTOR", ORTH_FACTOR), ("NEG_TOL", NEG_TOL),
                          ("CUTOFF_SAFETY", CUTOFF_SAFETY), ("PSD_FACTOR", PSD_FACTOR)):
        if not math.isfinite(_value) or _value <= 0:
            raise SystemExit(f"[lpp] {TAG}: {_name} must be finite and positive; got {_value!r}.")
    # After Cholesky--QR this is a genuine projection-accuracy gate. For the symmetric
    # E=Q^TQ-I, ||E||_2 <= ||E||_inf, so holding the latter below NEG_TOL also bounds the relative
    # coefficient-energy distortion. Half of NEG_TOL is reserved for the basis so subsequent
    # contraction/summation roundoff cannot consume the whole negative-complement allowance.
    # max(P,k)*eps is reported as the Gram-accumulation scale; the smaller of it and the
    # application cap is the default gate. An override may tighten but may not weaken the cap.
    ORTH_ROUNDOFF_SCALE = ORTH_FACTOR * max(P, k_st) * EPSM
    ORTH_APPLICATION_CAP = 0.5 * NEG_TOL
    ORTH_TOL = float(os.environ.get("ORTH_TOL",
                                    min(ORTH_ROUNDOFF_SCALE, ORTH_APPLICATION_CAP)))
    if not math.isfinite(ORTH_TOL) or ORTH_TOL <= 0:
        raise SystemExit(f"[lpp] {TAG}: ORTH_TOL must be finite and positive; got "
                         f"{ORTH_TOL!r}.")
    if ORTH_TOL > ORTH_APPLICATION_CAP:
        raise SystemExit(f"[lpp] {TAG}: ORTH_TOL={ORTH_TOL:.3e} exceeds the projector "
                         f"application cap {ORTH_APPLICATION_CAP:.3e} (= 0.5*NEG_TOL); "
                         "refusing to let the projector gate be weaker than the complement gate.")
    # G=J J^T accumulates dot products over J.shape[1], so its PSD roundoff scale depends on that
    # contraction dimension rather than only on the N x N eigensystem size.
    PSD_TOL_ABS = PSD_FACTOR * max(N, J.shape[1]) * EPSM * spectral_scale
    for cond, msg in (
            (orth_inf > ORTH_TOL,
             f"Cholesky-QR stiff projector not orthonormal: ||Q^T Q-I||_inf={orth_inf:.3e} > "
             f"{ORTH_TOL:.3e}"),
            (backward > BACKWARD_TOL,
             f"eigenpair backward error {backward:.3e} > {BACKWARD_TOL:.3e} "
             f"({BACKWARD_FACTOR:.0f} N eps_mach)"),
            (neg_eig_abs > PSD_TOL_ABS,
             f"Gram matrix has a negative eigenvalue {-neg_eig_abs:.3e}, beyond the PSD "
             f"backward-error allowance {PSD_TOL_ABS:.3e}"),
            (float(boundary_safety.min()) < CUTOFF_SAFETY,
             f"cutoff classification safety {boundary_safety.min():.1f} < "
             f"{CUTOFF_SAFETY:.1f}; margins/residual scales are {gap}")):
        if cond:
            raise SystemExit(f"[lpp] {TAG}: {msg}")
    print(f"[lpp] {TAG}: eigenpair backward error {backward:.2e} "
          f"({backward/(N*EPSM):.2f} N*eps); cutoff safety "
          f"{boundary_safety.min():.1f}x; (|lam|+alpha)-normalised residual {resid:.2e}, "
          f"{resid_near:.2e} near alpha", flush=True)
    print(f"[lpp] {TAG}: stiff inverse-lift orthogonality "
          f"{orth_inf_raw:.2e} raw -> {orth_inf:.2e} after Cholesky-QR "
          f"(raw cond upper bound {raw_basis_condition_upper_bound:.9f})", flush=True)

    floor = Lam * FLOOR_REL
    B_res = None
    resolved_coordinates = None
    resolved_basis_numerics = None
    if WANT_RES:
        _res_mask = sl & (lam > floor)
        if not _res_mask.any():
            raise SystemExit(f"[lpp] {TAG}: resolved-sloppy sensitivity has no modes above "
                             f"floor={floor:.6g} and below eps={eps:.6g}.")
        B_res = ((J.T @ W[:, _res_mask])
                 / np.sqrt(np.maximum(lam[_res_mask], 1e-300))[None, :])
        _res_k = int(_res_mask.sum())
        _res_gram = B_res.T @ B_res
        _res_gram = 0.5 * (_res_gram + _res_gram.T)
        _res_err_raw = _res_gram - np.eye(_res_k)
        _res_raw_max = float(np.abs(_res_err_raw).max())
        _res_raw_inf = float(np.linalg.norm(_res_err_raw, ord=np.inf))
        if _res_raw_inf > RAW_ORTH_TOL:
            raise SystemExit(f"[lpp] {TAG}: resolved-sloppy raw inverse lift is too far from "
                             f"orthonormal for roundoff-only correction: "
                             f"||B^T B-I||_inf={_res_raw_inf:.3e} > {RAW_ORTH_TOL:.3e}.")
        try:
            _res_chol = np.linalg.cholesky(_res_gram)
        except np.linalg.LinAlgError as exc:
            raise SystemExit(f"[lpp] {TAG}: resolved-sloppy inverse-lift Gram is not positive "
                             "definite; sensitivity projector is invalid.") from exc
        _res_qleft = solve_triangular(_res_chol, _res_gram, lower=True,
                                      overwrite_b=False, check_finite=False)
        _res_qgram = solve_triangular(_res_chol, _res_qleft.T, lower=True,
                                      overwrite_b=True, check_finite=False).T
        _res_err = _res_qgram - np.eye(_res_k)
        _res_orth_max = float(np.abs(_res_err).max())
        _res_orth_inf = float(np.linalg.norm(_res_err, ord=np.inf))
        if _res_orth_inf > ORTH_TOL:
            raise SystemExit(f"[lpp] {TAG}: implicit Cholesky-QR resolved-sloppy projector is "
                             f"not orthonormal: ||Q^T Q-I||_inf={_res_orth_inf:.3e} > "
                             f"{ORTH_TOL:.3e}.")

        def resolved_coordinates(g):
            c_raw = np.asarray(g) @ B_res
            c_t = solve_triangular(_res_chol, c_raw.T, lower=True,
                                   overwrite_b=True, check_finite=False)
            return c_t.T

        _res_cond_bound = math.sqrt((1 + _res_raw_inf) / (1 - _res_raw_inf))
        resolved_basis_numerics = {
            "n_modes": _res_k,
            "raw_orthonormality_max_dev": _res_raw_max,
            "raw_orthonormality_inf_norm": _res_raw_inf,
            "condition_number_upper_bound": _res_cond_bound,
            "corrected_orthonormality_max_dev": _res_orth_max,
            "corrected_orthonormality_inf_norm": _res_orth_inf,
            "basis_correction": "implicit_cholesky_qr",
        }

    # ---- the sweep: STIFF CONTRACTION ONLY ----------------------------------------------------
    # No Woodbury here. sigma^2 over the whole grid would cost J @ g.T plus a dense solve at every
    # point -- the same N x P x m product the old all-modes version paid, which is why the earlier
    # "18x cheaper" claim was doubly wrong. The regional sandwich uses the STORED sigma field, and
    # the current J is checked against that field on a small stratified sample below.
    q_sl = np.empty(n); q_st = np.empty(n); q_tot = np.empty(n)
    q_res = np.empty(n) if resolved_coordinates is not None else None
    worst_neg = 0.0
    for a in range(0, n, CH):
        b = min(a + CH, n)
        if (a // CH) % 25 == 0:
            print(f"[lpp]   sweep {a}/{n} pts", flush=True)
        g = np.asarray(grad_u(sub, jnp.asarray(Xr[a:b]), jnp.asarray(Tr[a:b])))
        cst = stiff_coordinates(g)
        q_tot[a:b] = (g * g).sum(1)
        q_st[a:b] = (cst * cst).sum(1)
        comp = q_tot[a:b] - q_st[a:b]
        # POINTWISE relative violation, against each point's own ||g||^2 -- dividing by the
        # chunk maximum, as this used to, lets a small-norm point violate badly and go unnoticed.
        rel = np.where(q_tot[a:b] > 0, -comp / np.maximum(q_tot[a:b], 1e-300), 0.0)
        worst_neg = max(worst_neg, float(rel.max()))
        q_sl[a:b] = np.maximum(comp, 0.0)
        if resolved_coordinates is not None:
            q_res[a:b] = (resolved_coordinates(g) ** 2).sum(1)

    if worst_neg > NEG_TOL:
        raise SystemExit(f"[lpp] {TAG}: ||g||^2 - ||Pi_st g||^2 went negative by {worst_neg:.3e} "
                         f"of the offending point's own ||g||^2 (> {NEG_TOL:.0e}). The stiff "
                         f"projector is not a projector.")

    # DETERMINISTIC STRATIFIED SAMPLE: does the CURRENT J reproduce the stored sigma? This is the
    # only place Woodbury is evaluated, and it is what makes the sandwich below a check against a
    # released artifact rather than against a quantity from this same run.
    #     sigma^2 = ( ||g||^2 - (Jg)^T (G + alpha I)^{-1} (Jg) ) / alpha
    sample_diag = None
    if _production:
        NS = int(os.environ.get("WOODBURY_SAMPLE", 256))
        idx_s = np.concatenate([r[np.linspace(0, len(r) - 1, max(1, NS // 3)).astype(int)]
                                for r in (regions["sharp"], regions["bulk"],
                                          np.setdiff1d(np.arange(n),
                                                       np.union1d(regions["sharp"],
                                                                  regions["bulk"])))])
        chol = cho_factor(G + alpha * np.eye(N), lower=True)
        gs = np.asarray(grad_u(sub, jnp.asarray(Xr[idx_s]), jnp.asarray(Tr[idx_s])))
        Jg = J @ gs.T
        quad = np.sum(Jg * cho_solve(chol, Jg), 0)
        s2_now = ((gs * gs).sum(1) - quad) / alpha
        neg_s2 = float((np.maximum(-s2_now, 0.0)
                        / np.maximum(s2_stored[idx_s], 1e-300)).max())
        if neg_s2 > NEG_TOL:
            raise SystemExit(f"[lpp] {TAG}: Woodbury sigma^2 negative by {neg_s2:.3e} relative "
                             f"to that point's stored sigma^2 -- clamping it would hide an "
                             f"indefinite (G + alpha I).")
        s2_now = np.maximum(s2_now, 0.0)
        rel = np.abs(s2_now - s2_stored[idx_s]) / np.maximum(s2_stored[idx_s], 1e-300)
        sample_diag = {"n_sampled": int(len(idx_s)), "max_rel_err": float(rel.max()),
                       "median_rel_err": float(np.median(rel)),
                       "woodbury_min_sigma2_rel": neg_s2}
        SAMP_TOL = float(os.environ.get("SAMPLE_TOL", 1e-3))
        if rel.max() > SAMP_TOL:
            raise SystemExit(f"[lpp] {TAG}: the current Jacobian does not reproduce the stored "
                             f"sigma field -- max relative error {rel.max():.3e} over "
                             f"{len(idx_s)} stratified points (> {SAMP_TOL:.0e}). The loadings "
                             f"would describe a different posterior from the released one.")
        print(f"[lpp] {TAG}: current J reproduces the stored sigma on {len(idx_s)} points "
              f"(max rel {rel.max():.2e})", flush=True)

    # ---- regions ------------------------------------------------------------------------------
    out = {"schema": SCHEMA, "tag": TAG, "scope": SCOPE, "sigma_metric": "raw",
           "P_bayes": P, "alpha": alpha,
           "eps": eps, "eps_mult": EPS_MULT, "lam_max": Lam,
           "n_gram_modes": int(N), "n_gram_modes_sloppy": int(sl.sum()), "n_stiff_modes": k_st,
           "want_resolved": WANT_RES, "floor_rel": FLOOR_REL,
           "config": {k: getattr(c, k) for k in ("pde", "arch", "seed", "fourier_scale",
                                                 "eval_nx", "eval_nt")},
           "dtypes": {"J": str(J.dtype), "G": str(G.dtype)},
           "hashes": {"producer": C.sha1(os.path.abspath(__file__)),
                       "map_pkl": C.sha1(pk),
                       "sigma_raw": C.sha1(sig_p) if _production else None,
                       "loading_cdf": C.sha1(cdf_p) if _want_cdf else None,
                       "fourier_B": (C.sha1(fourier_b_path) if fourier_b_path else None),
                       "fourier_B_path": fourier_b_path,
                       "gram": hashlib.sha1(np.ascontiguousarray(G).view(np.uint8)).hexdigest()},
           "validation_tolerances": {
               "backward_factor_N_eps": BACKWARD_FACTOR,
               "backward_tolerance": BACKWARD_TOL,
                "raw_inverse_lift_orthogonality_inf_tolerance": RAW_ORTH_TOL,
                "orthogonality_dimension_roundoff_scale": ORTH_ROUNDOFF_SCALE,
                "orthogonality_application_cap": ORTH_APPLICATION_CAP,
                "orthogonality_inf_tolerance": ORTH_TOL,
               "cutoff_min_safety": CUTOFF_SAFETY,
               "psd_factor_dimension_eps_normG": PSD_FACTOR,
               "psd_tolerance_abs": PSD_TOL_ABS,
               "negative_complement_rel": NEG_TOL,
               "woodbury_sample_rel": float(os.environ.get("SAMPLE_TOL", 1e-3)),
               "cdf_leg_rel": float(os.environ.get("CDF_TOL", 1e-3))},
             "numerics": {"stiff_inverse_lift_orthonormality_max_dev_raw": orth_raw,
                          "stiff_inverse_lift_orthonormality_inf_norm_raw": orth_inf_raw,
                          "stiff_inverse_lift_condition_number_upper_bound":
                              raw_basis_condition_upper_bound,
                          "stiff_basis_correction": "implicit_cholesky_qr",
                          "stiff_projector_orthonormality_max_dev": orth,
                          "stiff_projector_orthonormality_inf_norm": orth_inf,
                         "eigenpair_backward_error": backward,
                         "eigenpair_backward_error_N_eps_mach": backward / (N * EPSM),
                         "eigenpair_max_per_mode_residual": resid,
                         "eigenpair_residual_near_alpha": resid_near,
                         "gram_min_eigenvalue_raw": float(lam_raw.min()),
                         "gram_negative_eigenvalue_abs": neg_eig_abs,
                         "gram_negative_eigenvalue_rel_lam_max": neg_eig_abs / spectral_scale,
                         "gram_negative_eigenvalue_rel_alpha": neg_eig_abs / alpha,
                         "worst_negative_complement_rel": worst_neg,
                        "separation_around_alpha": gap,
                        "woodbury_sample": sample_diag}}

    for r, idx in regions.items():
        acc = np.zeros(P)
        for a in range(0, len(idx), CH):
            sel = idx[a:a + CH]
            acc += np.asarray(grad_u(sub, jnp.asarray(Xr[sel]), jnp.asarray(Tr[sel]))).sum(0)
        gbar = acc / max(len(idx), 1)
        st_b = float((stiff_coordinates(gbar[None, :])[0] ** 2).sum())
        m_raw = float(gbar @ gbar) - st_b
        m_rel = -m_raw / max(float(gbar @ gbar), 1e-300) if m_raw < 0 else 0.0
        if m_rel > NEG_TOL:
            raise SystemExit(f"[lpp] {TAG}: mean-tangent complement negative by {m_rel:.3e} "
                             f"relative in region {r}.")
        Q_sl = float(q_sl[idx].mean())
        Q_st = float(q_st[idx].mean())
        M_sl = max(m_raw, 0.0)
        JENSEN_TOL = float(os.environ.get("JENSEN_TOL", 1e-8))
        if Q_sl <= 0 or Q_st < 0 or M_sl < 0 or st_b < 0:
            raise SystemExit(f"[lpp] {TAG}: non-positive loading needed by a ratio in region {r}: "
                             f"Q_sl={Q_sl}, Q_st={Q_st}, M_sl={M_sl}, M_st={st_b}.")
        if M_sl > Q_sl * (1 + JENSEN_TOL) or st_b > Q_st * (1 + JENSEN_TOL):
            raise SystemExit(f"[lpp] {TAG}: Jensen/coherence bound failed in region {r}: "
                             f"M_sl/Q_sl={M_sl/Q_sl:.6g}, M_st/Q_st={st_b/max(Q_st,1e-300):.6g}.")
        out[r] = {"n": int(len(idx)), "Q_sl": Q_sl, "Q_st": Q_st,
                  "Q_tot": float(q_tot[idx].mean()),
                  "M_sl": M_sl, "M_st": st_b,
                  "M_sl_negative_rel": m_rel,
                  "kappa": float(M_sl / Q_sl),
                  "sigma2_stored_mean": (float(s2_stored[idx].mean())
                                          if s2_stored is not None else None)}
        if resolved_coordinates is not None:
            out[r]["Q_sl_resolved"] = float(q_res[idx].mean())
            out[r]["M_sl_resolved"] = float((resolved_coordinates(gbar[None, :])[0] ** 2).sum())

    def ratio(k):
        den = out["bulk"][k]
        if den <= 0:
            raise SystemExit(f"[lpp] {TAG}: bulk {k}={den} makes the regional ratio undefined.")
        return out["sharp"][k] / den
    out["ratio_perpoint_sl"] = ratio("Q_sl")
    out["ratio_mean_tangent_sl"] = ratio("M_sl")
    out["ratio_perpoint_sl_resolved"] = (ratio("Q_sl_resolved")
                                          if resolved_coordinates is not None else None)
    out["ratio_mean_tangent_sl_resolved"] = (ratio("M_sl_resolved")
                                               if resolved_coordinates is not None else None)
    out["sensitivity"] = ({"computed": True,
                           "basis_numerics": resolved_basis_numerics,
                           "resolved_vs_Pi_sl_perpoint":
                               float(abs(out["ratio_perpoint_sl_resolved"]
                                         - out["ratio_perpoint_sl"])
                                     / max(out["ratio_perpoint_sl"], 1e-300))}
                           if resolved_coordinates is not None else
                          {"computed": False,
                           "note": "run separately with WANT_RESOLVED=1; writes its own file"})

    # ---- the sandwich, against the STORED sigma ------------------------------------------------
    if s2_stored is not None:
        band = {}
        for r, idx in regions.items():
            lo = out[r]["Q_sl"] / (alpha + eps) + out[r]["Q_st"] / (alpha + Lam)
            hi = out[r]["Q_sl"] / alpha + out[r]["Q_st"] / (alpha + eps)
            mid = out[r]["sigma2_stored_mean"]
            band[r] = {"lower": lo, "sigma2_stored": mid, "upper": hi,
                       "inside": bool(lo <= mid * (1 + 1e-9) and mid <= hi * (1 + 1e-9))}
        out["sandwich"] = band
        if not all(v["inside"] for v in band.values()):
            raise SystemExit(f"[lpp] {TAG}: the stored <sigma^2> falls outside the loading "
                             f"sandwich {band}. The split does not bound the released posterior.")
        print(f"[lpp] {TAG}: sandwich holds against the stored sigma in both regions", flush=True)

    # ---- the loading-CDF cross-check, canonical configurations only ---------------------------
    if _want_cdf:
        with open(cdf_p) as _f:
            _c = json.load(_f)
        if _c.get("tag") != TAG or int(_c.get("N", -1)) != N:
            raise SystemExit(f"[lpp] {TAG}: CDF identity mismatch: tag={_c.get('tag')!r}, "
                             f"N={_c.get('N')!r}; expected {TAG!r}, {N}.")
        if not math.isclose(float(_c.get("alpha", math.nan)), alpha, rel_tol=1e-12, abs_tol=0.0):
            raise SystemExit(f"[lpp] {TAG}: CDF alpha {_c.get('alpha')!r} != current {alpha:.17g}.")
        comp = {}
        for _r, _k in (("sharp", "shock"), ("bulk", "smooth")):
            _v = _c[_k]
            comp[_r] = _v["total_gg"] - (_v["captured"] - _v["loading_below_alpha"])
        CDF_TOL = float(os.environ.get("CDF_TOL", 1e-3))
        # sharp and bulk compared SEPARATELY: a ratio can match while both legs are wrong.
        legs = {}
        for _r in ("sharp", "bulk"):
            got, want = out[_r]["M_sl"], comp[_r]
            legs[_r] = {"recomputed": got, "cdf_complement": want,
                        "rel_err": abs(got - want) / max(abs(want), 1e-300)}
        out["validation"] = {"legs": legs,
                             "ratio_recomputed": out["ratio_mean_tangent_sl"],
                             "ratio_cdf_complement": comp["sharp"] / max(comp["bulk"], 1e-300),
                             "cdf_ratio_as_shipped":
                                 _c["sloppy_loading_ratio_shock_over_smooth"],
                             "note": "target is the theorem-aligned complement, not the shipped "
                                     "ratio; legs checked individually"}
        bad = [f"{_r} {v['rel_err']*100:.1f}%" for _r, v in legs.items()
               if v["rel_err"] > CDF_TOL]
        if bad:
            raise SystemExit(f"[lpp] {TAG}: mean-tangent legs disagree with the shipped totals: "
                             f"{', '.join(bad)} (> {CDF_TOL*100:.0f}%).")
        print(f"[lpp] {TAG}: mean-tangent legs agree with the shipped totals "
              f"(sharp {legs['sharp']['rel_err']*100:.2f}%, "
              f"bulk {legs['bulk']['rel_err']*100:.2f}%)", flush=True)

    # ---- write, atomically, only now -----------------------------------------------------------
    def _finite(o):
        if isinstance(o, dict):
            return all(_finite(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return all(_finite(v) for v in o)
        return not isinstance(o, float) or math.isfinite(o)
    if not _finite(out):
        raise SystemExit(f"[lpp] {TAG}: a non-finite value reached the output.")
    _tmp = _out + ".partial"
    with open(_tmp, "w") as _f:
        json.dump(out, _f, indent=2, allow_nan=False)
    os.replace(_tmp, _out)
    if not os.path.getsize(_out):
        raise SystemExit(f"[lpp] {TAG}: {_out} is empty after the rename.")
    print(f"[lpp] {TAG} scope={SCOPE}: per-point Pi_sl ratio = {out['ratio_perpoint_sl']:.1f}  "
          f"vs mean-tangent {out['ratio_mean_tangent_sl']:.1f}  "
          f"(kappa_S {out['sharp']['kappa']:.3f}, kappa_B {out['bulk']['kappa']:.4f}; "
          f"orth {orth:.2e}, resid {resid:.2e})  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
