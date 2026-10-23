"""theorem_band_check.py -- evaluate the two-sided spectral sandwich on measured spectra.

For each canonical config, read {TAG}_loading_cdf.json (full N-space Gram eigenvalues
lambda_sorted, per-region cumulative loadings cum_loading, null-space component, alpha)
and evaluate BOTH sides of the two-sided spectral bound at every threshold
eps > 0:

  LB(eps)  = (a/(a+eps)) * L_s / ( L_b + (a/(a+eps)) * S_b )              [lower, sigma^2]
  UB(eps)  = ( L_s/a + S_s/(a+eps) ) / ( L_b/(a+eps) + S_b/(a+lam_max) )  [upper, sigma^2]

with L_r(eps) = sum_{lambda_i <= eps} c_{r,i}^2 + nullspace_r (the null directions are
sloppy for every eps > 0) and S_r(eps) = ||g_r||^2 - L_r(eps). For exact spectral energies,
every eps gives a valid
  bracket, so the intersection over eps is the tightest envelope supplied by this family of
  blockwise bounds for the measured geometry. It is not the tightest statement obtainable from
  the complete modewise profile, which gives the exact ratio. Reported in sigma-ratio units
  (square roots).

What the envelope brackets, precisely: the ratio sigma(g_bar_shock)/sigma(g_bar_smooth)
of the REGION-MEAN tangents (the loadings in loading_cdf.json are projections of g_bar_r).
The paper's secondary mean-width ratio averages PER-POINT sigma instead; it is displayed here
only as a non-certified comparison, together with the sub-full scopes (BLL / last-2 /
all-fusion), which are different post-hoc Gaussian constructions entirely.
The exact all-mode region-mean ratio (no inequality) is also reported.
For the disclosed legacy CDFs, compute L_r as total minus the captured stiff tail.
The signed total-minus-captured residual is not an independently measured null energy;
clamping it separately would bias both the bands and the all-mode variance.

Rounding: the *_safe fields floor the lower edge and ceil the upper edge, so quoting
them never claims a tighter bound than was computed. Quote the safe fields in the paper.

Output: results/_theorem_band/theorem_band_check.json  + console table.
With PLOT=1 (default if matplotlib is importable), also regenerates the paper figure
submissions/jmlr_ceiling/figures/theorem_bounds.png (and calibration_figures/theorem_bounds.png)
from the SAME computation, so figure legend == table == JSON, one source of truth.
Pass --check for a strictly read-only comparison against the stored JSON, a hash contract for
both figure copies, and the independent current-J projector summary.
Pure numpy (+ matplotlib for the figure); no JAX, no GPU.
"""
import os, json, math, sys, hashlib
import numpy as np
import legacy_loading_contract as legacy

CONFIGS = [
    ("burgers_mlp_seed2",        "Burgers - MLP"),
    ("burgers_fourier_seed2",    "Burgers - Fourier"),
    ("allen_cahn_mlp_seed0",     "Allen-Cahn - MLP"),
    ("allen_cahn_fourier_seed0", "Allen-Cahn - Fourier"),
]
SCOPE_SUFFIXES = ["bll_lastlayer", "bll_last2fusion", "bll_allfusion"]
SCOPE_LABELS   = ["BLL", "last-2", "all-fusion", "full"]
LEGACY_CDF_REL_TOL = float(os.environ.get("LEGACY_CDF_REL_TOL", "5e-7"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(path + ".partial", path)


def load_ratio(path, key=("ratios", "achieved_sigma_ratio")):
    try:
        d = json.load(open(path))
        for k in key: d = d[k]
        return float(d)
    except Exception:
        return None


def floor2(x): return math.floor(x * 100.0) / 100.0     # safe lower edge
def ceil0(x):  return float(math.ceil(x))               # safe upper edge


def split_parts(region, k, tag):
    """Return null-inclusive sloppy/stiff energies after the first k Gram modes.

    The complement is allowed only roundoff-scale negativity.  Anything larger is a structural
    failure and is never silently clamped into a plausible loading.
    """
    return legacy.split_parts(region, k)


def envelope_at(eps, k, reg, alpha, lmax, tag):
    Ls, Ss = split_parts(reg["shock"], k, tag)
    Lb, Sb = split_parts(reg["smooth"], k, tag)
    pref = alpha / (alpha + eps)
    den_lo = Lb + pref * Sb
    den_hi = Lb / (alpha + eps) + Sb / (alpha + lmax)
    if den_lo <= 0 or den_hi <= 0:
        raise SystemExit(f"{tag}: non-positive envelope denominator at eps={eps}, k={k}")
    lb = pref * Ls / den_lo
    ub = (Ls / alpha + Ss / (alpha + eps)) / den_hi
    if lb < -1e-14 or ub < -1e-14 or lb > ub * (1 + 1e-10):
        raise SystemExit(f"{tag}: invalid envelope [{lb}, {ub}] at eps={eps}, k={k}")
    return max(lb, 0.0), max(ub, 0.0)


def exact_endpoint_states(lam):
    """All interval endpoints needed for an exact piecewise-rational optimisation.

    On each interval between distinct eigenvalues the allocation is fixed and each envelope has
    derivative of fixed sign, so an extremum occurs at an endpoint.  At an eigenvalue we evaluate
    both the left limit (before adding the tied eigenspace) and the closed-threshold value (after).
    """
    states = [(0.0, int(np.searchsorted(lam, 0.0, side="right")), "zero-plus")]
    for value in np.unique(lam[lam > 0]):
        states.append((float(value), int(np.searchsorted(lam, value, side="left")), "left-limit"))
        states.append((float(value), int(np.searchsorted(lam, value, side="right")), "at"))
    return states


def build_figure_contract(theorem_target, figure_targets):
    return {
        "schema": "theorem_band_figure_contract/1",
        "theorem_band_sha256": sha256(theorem_target),
        "producer_sha256": sha256(__file__),
        "legacy_loading_contract_sha256": sha256(legacy.__file__),
        "figures_sha256": {path: sha256(path) for path in figure_targets},
    }


def validate_independent_projector(theorem_target):
    path = "results/_band_projector/band_projector_check.json"
    try:
        record = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"CHECK FAIL -- cannot read independent projector summary {path}: {exc}")
    if (record.get("schema") != "band_projector_check/2" or
            record.get("cdf_accounting_contract") != legacy.ACCOUNTING_CONTRACT or
            record.get("theorem_band_sha256") != sha256(theorem_target) or
            set(record.get("configurations", {})) != {tag for tag, _ in CONFIGS} or
            not all(item.get("validation_passed") is True
                    for item in record.get("configurations", {}).values())):
        raise SystemExit("CHECK FAIL -- independent current-J projector summary is absent, "
                         "stale, incomplete, or did not pass")


def main():
    unknown = [arg for arg in sys.argv[1:] if arg != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown)}")
    check_only = "--check" in sys.argv[1:]
    out, curves = {}, {}
    for tag, label in CONFIGS:
        j = json.load(open(f"results/{tag}/{tag}_loading_cdf.json"))
        a = float(j["alpha"]); lam = np.asarray(j["lambda_sorted"], float); lmax = float(lam[-1])
        if (lam.ndim != 1 or len(lam) != int(j.get("N", -1)) or
                not np.all(np.isfinite(lam)) or np.any(lam < 0) or np.any(np.diff(lam) < 0) or
                not (math.isfinite(a) and a > 0 and lmax > 0)):
            raise SystemExit(f"{tag}: malformed spectrum/alpha in loading CDF")
        reg = {}
        for r in ("shock", "smooth"):
            reg[r] = legacy.load_region(j[r], lam, f"{tag}/{r}", LEGACY_CDF_REL_TOL)
        # Algebraically exact all-mode expression over the recorded CDF; only the separate
        # current-J Woodbury comparison is independent validation of this measurement.
        def sig2(r):
            return legacy.variance(reg[r], lam, a)
        exact_regmean = float(np.sqrt(sig2("shock") / sig2("smooth")))

        # Exact optimisation: both sides of every spectral breakpoint.  The plotting grid below
        # is deliberately separate and cannot change the reported bracket.
        states = exact_endpoint_states(lam)
        exact_lb = np.empty(len(states)); exact_ub = np.empty(len(states))
        for i, (eps, k, _) in enumerate(states):
            exact_lb[i], exact_ub[i] = envelope_at(eps, k, reg, a, lmax, tag)
        ilo, ihi = int(np.argmax(exact_lb)), int(np.argmin(exact_ub))
        band = (float(np.sqrt(exact_lb[ilo])), float(np.sqrt(exact_ub[ihi])))

        ka = int(np.searchsorted(lam, a, side="right"))
        la, ua = envelope_at(a, ka, reg, a, lmax, tag)
        at_alpha = (float(np.sqrt(la)), float(np.sqrt(ua)))

        # Dense grid only for drawing the curves.
        pos = lam[lam > 0]
        eps_grid = np.unique(np.concatenate([
            pos, np.logspace(np.log10(pos.min()), np.log10(lmax), 600), [a]]))
        lb_arr = np.empty_like(eps_grid); ub_arr = np.empty_like(eps_grid)
        for i, eps in enumerate(eps_grid):
            k = np.searchsorted(lam, eps, side="right")
            lb_arr[i], ub_arr[i] = envelope_at(float(eps), int(k), reg, a, lmax, tag)

        # Secondary mean-width ratios: scope ladder (canonical seed) + full-grid construction.
        measured = {}
        for s in SCOPE_SUFFIXES:
            measured[s] = load_ratio(f"results/{tag}/{tag}_{s}_metrics.json")
        measured["full_laplace_fullgrid"] = load_ratio(
            f"results/{tag}/{tag}_fullgrid_ratio.json", key=("raw", "ratio_mean"))
        missing_measured = [name for name, value in measured.items()
                            if value is None or not math.isfinite(value) or value <= 0]
        if missing_measured:
            raise SystemExit(
                f"{tag}: missing/non-positive measured scope contrast(s): "
                + ", ".join(missing_measured)
            )
        vals = list(measured.values())
        inside_exact = bool(band[0] <= exact_regmean <= band[1])

        out[tag] = {"alpha": a, "lambda_max": lmax,
                     "cdf_accounting_contract": legacy.ACCOUNTING_CONTRACT,
                     "cdf_closure_diagnostics": {
                         r: legacy.closure_diagnostic(reg[r]) for r in reg
                     },
                    # NOT pre-rounded. gate_numbers.py formats this field to 2dp, so a
                    # round(...,3) here rounds twice: Burgers-MLP seed 2 is 7.564979, which
                    # became 7.565 and then 7.57, and the paper carried 7.57 in five places.
                    # The correct 2dp value is 7.56. Emit full precision and round once.
                     "exact_regmean_ratio": exact_regmean,
                     "band_at_eps_alpha_full": [at_alpha[0], at_alpha[1]],
                     "band_at_eps_alpha": [round(at_alpha[0], 3), round(at_alpha[1], 1)],
                     "band_at_eps_alpha_safe": [floor2(at_alpha[0]), ceil0(at_alpha[1])],
                     "band_tightest_full": [band[0], band[1]],
                     "band_tightest": [round(band[0], 3), round(band[1], 1)],
                     "band_tightest_safe": [floor2(band[0]), ceil0(band[1])],
                     "optimization": {
                         "algorithm": "exact spectral interval endpoints; both left-limit and closed-threshold states",
                         "lower_eps": states[ilo][0],
                         "lower_side": states[ilo][2],
                         "upper_eps": states[ihi][0],
                         "upper_side": states[ihi][2],
                     },
                    "measured": {k: (round(v, 3) if v is not None else None)
                                 for k, v in measured.items()},
                    "exact_regmean_inside_band": inside_exact}
        curves[tag] = dict(label=label, eps=eps_grid, lb=np.sqrt(lb_arr), ub=np.sqrt(ub_arr),
                           alpha=a, band=band, exact=exact_regmean,
                           measured=[measured[s] for s in SCOPE_SUFFIXES] +
                                    [measured["full_laplace_fullgrid"]])
        print(f"{tag:26s} band(eps=a)=[{at_alpha[0]:.3f},{at_alpha[1]:.1f}]  "
              f"tightest=[{band[0]:.3f},{band[1]:.1f}]  safe=[{floor2(band[0])},{ceil0(band[1]):.0f}]  "
              f"regmean_exact={exact_regmean:.2f}(in={inside_exact})  "
              f"non-certified mean-width comparisons="
              f"{[round(v,2) if v else None for v in curves[tag]['measured']]}")

    target = "results/_theorem_band/theorem_band_check.json"
    figure_targets = (
        "submissions/jmlr_ceiling/figures/theorem_bounds.png",
        "calibration_figures/theorem_bounds.png",
    )
    figure_contract_target = (
        "results/_theorem_band/theorem_band_figure_contract.json"
    )
    if check_only:
        try:
            with open(target, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"CHECK FAIL -- cannot read {target}: {exc}")
        if stored != out:
            raise SystemExit(
                "CHECK FAIL -- stored theorem-band JSON differs from the live computation; "
                "run without --check to regenerate"
            )
        missing = [path for path in figure_targets if not os.path.isfile(path)]
        if missing:
            raise SystemExit("CHECK FAIL -- missing figure output(s): " + ", ".join(missing))
        try:
            contract = json.load(open(figure_contract_target, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"CHECK FAIL -- cannot read {figure_contract_target}: {exc}")
        expected_contract = build_figure_contract(target, figure_targets)
        if contract != expected_contract:
            raise SystemExit("CHECK FAIL -- theorem-band figure hash contract is stale")
        validate_independent_projector(target)
        print("CHECK OK -- theorem-band JSON, both figure hashes, and independent current-J "
              "projectors agree")
        return

    atomic_json(target, out)
    print("wrote results/_theorem_band/theorem_band_check.json")

    if os.environ.get("PLOT", "1") == "1":
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib unavailable ({e}); using Pillow fallback")
            from render_theorem_bounds_pil import render
            render(curves, CONFIGS, figure_targets)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
            for ax, (tag, _) in zip(axes.ravel(), CONFIGS):
                c = curves[tag]
                ax.plot(c["eps"], c["ub"], color="tab:red",  lw=1.8, label=r"$\sqrt{U(\varepsilon)}$ (upper)")
                ax.plot(c["eps"], c["lb"], color="tab:blue", lw=1.8, label=r"$\sqrt{L(\varepsilon)}$ (lower)")
                ax.axhspan(c["band"][0], c["band"][1], color="grey", alpha=0.25,
                           label=f"provable band [{floor2(c['band'][0])}, {ceil0(c['band'][1]):.0f}]")
                ax.axvline(c["alpha"], color="k", ls=":", lw=1.2, label=r"$\varepsilon=\alpha$")
                ax.axhline(c["exact"], color="tab:green", lw=2.0,
                           label=f"region-mean exact = {c['exact']:.1f} (verified)")
                for v, sl, ls in zip(c["measured"], SCOPE_LABELS, ["--", "--", "--", "-."]):
                    if v is not None:
                        ax.axhline(v, color="tab:purple", ls=ls, lw=1.0, alpha=0.8)
                ax.axhline(c["measured"][-1], color="tab:purple", ls="-.", lw=1.4,
                           label=f"measured scopes {['%.2f' % v for v in c['measured'] if v is not None]}")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(c["label"], fontweight="bold")
                ax.set_xlabel(r"threshold $\varepsilon$"); ax.set_ylabel(r"$\sigma$-ratio")
                ax.legend(fontsize=7.5, loc="upper left")
            fig.suptitle("Spectral sandwich on the measured spectrum "
                         "(canonical seeds; grey = tightest provable band)",
                         fontweight="bold", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            for dest in figure_targets:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                fig.savefig(dest, dpi=180)
                print(f"wrote {dest}")
            plt.close(fig)
        atomic_json(figure_contract_target, build_figure_contract(target, figure_targets))
        print(f"wrote {figure_contract_target}")


if __name__ == "__main__":
    main()
