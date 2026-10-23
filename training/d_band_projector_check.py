"""Independent current-J validation of the region-mean spectral brackets.

The legacy loading CDF remains useful for eigenvalues and cumulative upper-tail energies, but
agreement between two consumers of that file is not an independent measurement.  This producer
rebuilds the raw-metric Jacobian for one canonical configuration, orthonormalises each stiff
inverse-lift basis by Cholesky-QR, and checks both regional legs at:

  * the declared primary split epsilon = alpha;
  * the exact lower-envelope optimiser;
  * the exact upper-envelope optimiser; and
  * for Burgers-MLP seed 2, the information-bound optimiser.

It also compares the direct Woodbury variance of each region-mean tangent with a stable
CDF-derived complementary expression.  Signed legacy closure residuals are diagnostics,
never independently clamped null-space energies.  Every check is fail-closed and the output
is written atomically only at the end.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

import common as C
import laplace_core as L
import provenance as prov
import legacy_loading_contract as legacy


SCHEMA = "band_projector_check/2"
LEGACY_CDF_REL_TOL = float(os.environ.get("LEGACY_CDF_REL_TOL", "5e-7"))
CANONICAL = {
    "burgers_mlp_seed2",
    "burgers_fourier_seed2",
    "allen_cahn_mlp_seed0",
    "allen_cahn_fourier_seed0",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def exact_information_threshold(cdf: dict) -> tuple[float, str]:
    lam = np.asarray(cdf["lambda_sorted"], dtype=np.float64)
    alpha = float(cdf["alpha"])
    region = legacy.load_region(cdf["shock"], lam, "information/sharp", LEGACY_CDF_REL_TOL)
    total = region["tot"]
    best = (math.inf, None, None)
    states = [(0.0, int(np.searchsorted(lam, 0.0, side="right")), "zero-plus")]
    states.extend((float(eps), int(np.searchsorted(lam, eps, side="right")), "at")
                  for eps in np.unique(lam[lam > 0]))
    for eps, k, side in states:
        sloppy, _ = legacy.split_parts(region, k)
        share = sloppy / total
        if share <= 0:
            continue
        bound = 0.5 * math.log1p(float(eps) / alpha) - 0.5 * math.log(share)
        if bound < best[0]:
            best = (bound, eps, side)
    require(best[1] is not None, "information threshold could not be determined")
    return float(best[1]), str(best[2])


def main() -> int:
    source_paths = {
        "producer": Path(__file__).resolve(), "common": Path(C.__file__).resolve(),
        "laplace_core": Path(L.__file__).resolve(),
        "legacy_loading_contract": Path(legacy.__file__).resolve(),
    }
    source_hashes = {key: sha256(path) for key, path in source_paths.items()}
    require(os.environ.get("RAW_METRIC") == "1",
            "[band-projector] RAW_METRIC=1 is mandatory")
    config = C.get_config()
    tag = f"{config.pde}_{config.arch}_seed{config.seed}"
    require(tag in CANONICAL, f"[band-projector] non-canonical tag {tag}")

    outdir = Path(os.environ.get(
        "OUTDIR", str(Path("results") / "_band_projector" / tag)
    ))
    target = outdir / f"{tag}_band_projector.json"
    target.unlink(missing_ok=True)

    result_dir = Path("results") / tag
    map_path = Path(os.environ.get("MAP_PKL", str(result_dir / f"{tag}_params.pkl")))
    cdf_path = result_dir / f"{tag}_loading_cdf.json"
    band_path = Path(os.environ.get(
        "THEOREM_BAND", str(Path("results") / "_theorem_band" / "theorem_band_check.json")
    ))
    for path, label in ((map_path, "MAP"), (cdf_path, "loading CDF"),
                        (band_path, "theorem-band summary")):
        require(path.is_file(), f"[band-projector] {tag}: required {label} is absent: {path}")

    sidecar = None
    if config.arch == "fourier":
        sidecar = result_dir / f"{tag}_fourier_B.npz"
        require(sidecar.is_file(),
                f"[band-projector] {tag}: frozen Fourier sidecar is absent: {sidecar}")
        os.environ["FOURIER_B_NPZ"] = str(sidecar)

    input_paths = {"map": map_path, "loading_cdf": cdf_path,
                   "theorem_band": band_path, "fourier_sidecar": sidecar}
    input_hashes = {key: sha256(path) if path is not None else None
                    for key, path in input_paths.items()}

    cdf = json.loads(cdf_path.read_text(encoding="utf-8"))
    cdf_lam_input = np.asarray(cdf.get("lambda_sorted", []), dtype=np.float64)
    require(cdf_lam_input.ndim == 1 and len(cdf_lam_input) == int(cdf.get("N", -1)) and
            np.all(np.isfinite(cdf_lam_input)) and np.all(cdf_lam_input >= 0) and
            np.all(np.diff(cdf_lam_input) >= 0),
            f"[band-projector] {tag}: malformed loading-CDF spectrum")
    cdf_regions = {}
    for region_name in ("shock", "smooth"):
        cdf_regions[region_name] = legacy.load_region(
            cdf.get(region_name, {}), cdf_lam_input,
            f"[band-projector] {tag}/{region_name}", LEGACY_CDF_REL_TOL,
        )
    band_all = json.loads(band_path.read_text(encoding="utf-8"))
    band = band_all.get(tag)
    require(isinstance(band, dict) and isinstance(band.get("optimization"), dict),
            f"[band-projector] {tag}: theorem-band summary lacks exact optimisers")
    require(band.get("cdf_accounting_contract") == legacy.ACCOUNTING_CONTRACT,
            f"[band-projector] {tag}: regenerate the theorem-band summary with signed-closure accounting")

    thresholds = {
        "alpha": {"epsilon": float(cdf["alpha"]), "side": "at"},
        "lower_optimum": {
            "epsilon": float(band["optimization"]["lower_eps"]),
            "side": str(band["optimization"]["lower_side"]),
        },
        "upper_optimum": {
            "epsilon": float(band["optimization"]["upper_eps"]),
            "side": str(band["optimization"]["upper_side"]),
        },
    }
    if tag == "burgers_mlp_seed2":
        info_epsilon, info_side = exact_information_threshold(cdf)
        thresholds["information_optimum"] = {
            "epsilon": info_epsilon, "side": info_side
        }

    ctx = C.build_context(config)
    params = C.load_map(str(map_path))
    timer = prov.Timer()
    with timer.phase("raw_full_jacobian"):
        engine = L.run(ctx, config, params, "full")
    J = np.asarray(engine["J"], dtype=np.float64)
    N = int(engine["N"])
    P = int(engine["n_sub"])
    alpha = float(engine["alpha"])
    require(J.dtype == np.float64 and J.shape == (N, P),
            f"[band-projector] {tag}: invalid Jacobian {J.shape} {J.dtype}")
    require(math.isclose(alpha, float(cdf["alpha"]), rel_tol=1e-12, abs_tol=0.0),
            f"[band-projector] {tag}: alpha differs from the CDF")

    with timer.phase("gram_eigendecomposition"):
        G = J @ J.T
        lam_raw, W = np.linalg.eigh(G)
    spectral_scale = max(float(np.max(np.abs(lam_raw))), np.finfo(np.float64).tiny)
    epsmach = np.finfo(np.float64).eps
    psd_tol = 10.0 * max(N, P) * epsmach * spectral_scale
    require(float(lam_raw.min()) >= -psd_tol,
            f"[band-projector] {tag}: Gram negative eigenvalue {lam_raw.min():.3e} "
            f"exceeds PSD allowance {psd_tol:.3e}")
    lam = np.maximum(lam_raw, 0.0)
    cdf_lam = np.asarray(cdf["lambda_sorted"], dtype=np.float64)
    require(cdf_lam.shape == lam.shape,
            f"[band-projector] {tag}: CDF/current spectrum length mismatch")
    spectrum_error = float(np.max(np.abs(lam - cdf_lam)) / spectral_scale)
    spectrum_limit = float(os.environ.get("BAND_SPECTRUM_TOL", "1e-10"))
    require(spectrum_error <= spectrum_limit,
            f"[band-projector] {tag}: current spectrum differs from the CDF by "
            f"{spectrum_error:.3e} normwise")

    xs = np.linspace(-1, 1, config.eval_nx)
    ts = np.linspace(0, 1, config.eval_nt)
    XX, TT = np.meshgrid(xs, ts)
    Xr, Tr = XX.ravel(), TT.ravel()
    reference = ctx.grid_reference(xs, ts)
    grad_abs = np.abs(np.gradient(reference, xs[1] - xs[0], axis=1)).ravel()
    p70, p95 = np.percentile(grad_abs, (70, 95))
    regions = {
        "sharp": np.flatnonzero(grad_abs >= p95),
        "bulk": np.flatnonzero(grad_abs < p70),
    }
    chunk = int(os.environ.get("CHUNK", "200"))
    grad_u, sub = engine["grad_u"], engine["sub_map"]

    def mean_tangent(indices: np.ndarray) -> np.ndarray:
        acc = np.zeros(P, dtype=np.float64)
        for start in range(0, len(indices), chunk):
            idx = indices[start:start + chunk]
            gradients = np.asarray(
                grad_u(sub, jnp.asarray(Xr[idx]), jnp.asarray(Tr[idx])),
                dtype=np.float64,
            )
            acc += gradients.sum(axis=0)
        return acc / len(indices)

    means = {name: mean_tangent(idx) for name, idx in regions.items()}
    cdf_region = {"sharp": "shock", "bulk": "smooth"}
    raw_orth_limit = math.sqrt(epsmach)
    orth_limit = float(os.environ.get("BAND_ORTH_TOL", "1e-8"))
    negative_limit = float(os.environ.get("BAND_NEGATIVE_TOL", "1e-8"))
    leg_limit = float(os.environ.get("BAND_LEG_TOL", "1e-6"))
    partition_gap_min = float(os.environ.get("BAND_PARTITION_GAP_SAFETY", "50"))
    threshold_records = {}

    for name, specification in thresholds.items():
        threshold = float(specification["epsilon"])
        side = specification["side"]
        require(side in {"at", "left-limit", "zero-plus"},
                f"[band-projector] {tag}: unknown threshold side {side!r}")
        search_side = "left" if side == "left-limit" else "right"
        k_sloppy = int(np.searchsorted(cdf_lam, threshold, side=search_side))
        sloppy = np.zeros(N, dtype=bool)
        sloppy[:k_sloppy] = True
        stiff = ~sloppy
        k_stiff = int(stiff.sum())
        backward_limit = 64.0 * N * epsmach
        if k_stiff:
            require(np.all(lam[stiff] > 0),
                    f"[band-projector] {tag}: {name} asks to inverse-lift a zero mode")
            B = (J.T @ W[:, stiff]) / np.sqrt(lam[stiff])[None, :]
            raw_gram = B.T @ B
            raw_gram = 0.5 * (raw_gram + raw_gram.T)
            raw_inf = float(np.linalg.norm(raw_gram - np.eye(k_stiff), ord=np.inf))
            require(raw_inf <= raw_orth_limit,
                    f"[band-projector] {tag}: {name} inverse lift too far from orthonormal "
                    f"for Cholesky-QR ({raw_inf:.3e} > {raw_orth_limit:.3e})")
            try:
                chol = np.linalg.cholesky(raw_gram)
            except np.linalg.LinAlgError as exc:
                raise SystemExit(
                    f"[band-projector] {tag}: {name} inverse-lift Gram is not positive definite"
                ) from exc
            qleft = solve_triangular(chol, raw_gram, lower=True, check_finite=False)
            qgram = solve_triangular(
                chol, qleft.T, lower=True, check_finite=False
            ).T
            orth_inf = float(np.linalg.norm(qgram - np.eye(k_stiff), ord=np.inf))
            require(orth_inf <= orth_limit,
                    f"[band-projector] {tag}: {name} corrected projector is not orthonormal "
                    f"({orth_inf:.3e} > {orth_limit:.3e})")
            used_W = W[:, stiff]
            residual = np.linalg.norm(G @ used_W - used_W * lam[stiff], axis=0)
            backward = float(np.max(residual / (spectral_scale + lam[stiff])))
            require(backward <= backward_limit,
                    f"[band-projector] {tag}: {name} eigenpair backward error "
                    f"{backward:.3e} > {backward_limit:.3e}")
        else:
            # epsilon at/above lambda_max: all Gram modes (and the implicit Fisher null)
            # are sloppy.  The stiff projector is the empty orthogonal projector exactly.
            B = np.empty((P, 0), dtype=np.float64)
            chol = np.empty((0, 0), dtype=np.float64)
            raw_inf = orth_inf = backward = 0.0

        # These exact envelope endpoints are themselves eigenvalues.  Therefore
        # distance from the numerical threshold is *supposed* to be zero and is
        # not a meaningful stability test.  The stable object is the ordinal
        # partition inherited from the CDF: all members of an exactly tied CDF
        # block move together through searchsorted.  Gate the separation of
        # that partition from its neighbouring block in the freshly rebuilt
        # spectrum, measured against the eigensolver's backward-error scale.
        partition_trivial = k_sloppy == N
        if 0 < k_sloppy < N:
            boundary = np.array([k_sloppy - 1, k_sloppy], dtype=int)
            partition_gap = float(lam[k_sloppy] - lam[k_sloppy - 1])
        elif k_sloppy == 0:
            # Separate the implicit Fisher null at zero from the first positive Gram mode.
            boundary = np.array([0], dtype=int)
            partition_gap = float(lam[0])
        else:
            boundary = np.empty(0, dtype=int)
            partition_gap = None
        if not partition_trivial:
            boundary_residual = np.linalg.norm(
                G @ W[:, boundary] - W[:, boundary] * lam[boundary], axis=0
            )
            boundary_scale = np.maximum(
                boundary_residual, 8.0 * N * epsmach * spectral_scale
            )
            partition_scale = float(np.max(boundary_scale))
            partition_safety = float(partition_gap / partition_scale)
            require(partition_safety >= partition_gap_min,
                    f"[band-projector] {tag}: {name} ordinal spectral partition is "
                    f"not numerically separated ({partition_safety:.1f}x < "
                    f"{partition_gap_min:.1f}x)")
        else:
            partition_safety = None

        def stiff_energy(vector: np.ndarray) -> float:
            if not k_stiff:
                return 0.0
            raw = vector @ B
            coords = solve_triangular(chol, raw, lower=True, check_finite=False)
            return float(coords @ coords)

        legs = {}
        for region, vector in means.items():
            total = float(vector @ vector)
            stiff_value = stiff_energy(vector)
            sloppy_raw = total - stiff_value
            negative = max(-sloppy_raw, 0.0) / max(total, np.finfo(np.float64).tiny)
            require(negative <= negative_limit,
                    f"[band-projector] {tag}: {name}/{region} negative complement "
                    f"{negative:.3e} > {negative_limit:.3e}")
            sloppy_value = max(sloppy_raw, 0.0)

            old = cdf_regions[cdf_region[region]]
            k = int(np.searchsorted(cdf_lam, threshold, side=search_side))
            # The signed legacy closure discrepancy cancels in total minus the stiff tail.
            # Clipping that discrepancy first changes the target (notably for Allen--Cahn).
            expected, _ = legacy.split_parts(old, k)
            rel = abs(sloppy_value - expected) / max(abs(expected), np.finfo(np.float64).tiny)
            require(rel <= leg_limit,
                    f"[band-projector] {tag}: {name}/{region} current-J sloppy leg "
                    f"disagrees with stable CDF complement by {rel:.3e}")
            total_rel = abs(total - float(old["tot"])) / max(
                abs(float(old["tot"])), np.finfo(np.float64).tiny
            )
            require(total_rel <= leg_limit,
                    f"[band-projector] {tag}: {name}/{region} tangent norm drift {total_rel:.3e}")
            legs[region] = {
                "total": total,
                "sloppy": sloppy_value,
                "stiff": stiff_value,
                "negative_complement_relative": negative,
                "cdf_complement": expected,
                "cdf_leg_relative_error": rel,
                "cdf_total_relative_error": total_rel,
            }

        threshold_records[name] = {
            "epsilon": threshold,
            "side": side,
            "n_sloppy": int(sloppy.sum()),
            "n_stiff": k_stiff,
            "raw_inverse_lift_orthogonality_inf": raw_inf,
            "corrected_projector_orthogonality_inf": orth_inf,
            "eigenpair_backward_error": backward,
            "partition_gap": partition_gap,
            "partition_gap_safety": partition_safety,
            "partition_is_trivial_all_sloppy": partition_trivial,
            "legs": legs,
        }

    factor = cho_factor(G + alpha * np.eye(N), lower=True, check_finite=False)
    direct_sigma2 = {}
    cdf_sigma2 = {}
    variance_relative_error = {}
    for region, vector in means.items():
        Jg = J @ vector
        raw = (float(vector @ vector) - float(Jg @ cho_solve(factor, Jg))) / alpha
        negative = max(-raw, 0.0) / max(float(vector @ vector) / alpha,
                                        np.finfo(np.float64).tiny)
        require(negative <= negative_limit,
                f"[band-projector] {tag}: {region} direct Woodbury variance is negative")
        direct_sigma2[region] = max(raw, 0.0)

        old = cdf_regions[cdf_region[region]]
        cdf_sigma2[region] = legacy.variance(old, cdf_lam, alpha)
        rel = abs(direct_sigma2[region] - cdf_sigma2[region]) / max(
            abs(direct_sigma2[region]), np.finfo(np.float64).tiny
        )
        variance_relative_error[region] = rel
        variance_limit = float(os.environ.get("BAND_VARIANCE_TOL", "5e-6"))
        require(rel <= variance_limit,
                f"[band-projector] {tag}: {region} direct Woodbury and CDF variance "
                f"differ by {rel:.3e}")

    exact_ratio = math.sqrt(direct_sigma2["sharp"] / direct_sigma2["bulk"])
    summary_ratio = float(band["exact_regmean_ratio"])
    ratio_rel = abs(exact_ratio - summary_ratio) / summary_ratio
    ratio_limit = float(os.environ.get("BAND_RATIO_TOL", "5e-6"))
    require(ratio_rel <= ratio_limit,
            f"[band-projector] {tag}: direct ratio and theorem-band ratio differ by "
            f"{ratio_rel:.3e}")

    provenance = prov.collect(config=C.config_dict(config))
    provenance["timings"] = timer.dump()
    provenance["peak_gpu_mem_mb"] = prov.peak_gpu_mem_mb()
    record = {
        "schema": SCHEMA,
        "cdf_accounting_contract": legacy.ACCOUNTING_CONTRACT,
        "cdf_closure_diagnostics": {
            region: legacy.closure_diagnostic(cdf_regions[cdf_region[region]])
            for region in means
        },
        "tag": tag,
        "configuration": C.config_dict(config),
        "sigma_metric": "raw",
        "scope": "full",
        "alpha": alpha,
        "lambda_max": float(lam.max()),
        "spectrum_max_abs_over_lambda_max": spectrum_error,
        "P_bayes": P,
        "N_residual_rows": N,
        "thresholds": threshold_records,
        "direct_sigma2": direct_sigma2,
        "cdf_sigma2": cdf_sigma2,
        "direct_vs_cdf_variance_relative_error": variance_relative_error,
        "exact_regmean_ratio": exact_ratio,
        "theorem_band_ratio_relative_error": ratio_rel,
        "validation_passed": True,
        "tolerances": {
            "psd_absolute": psd_tol,
            "raw_inverse_lift_orthogonality_inf": raw_orth_limit,
            "corrected_projector_orthogonality_inf": orth_limit,
            "negative_complement_relative": negative_limit,
            "cdf_leg_relative": leg_limit,
            "spectrum_max_abs_over_lambda_max": spectrum_limit,
            "partition_gap_safety_min": partition_gap_min,
            "eigenpair_backward_error": backward_limit,
            "legacy_loading_cdf_relative": LEGACY_CDF_REL_TOL,
            "direct_vs_cdf_variance_relative": variance_limit,
            "theorem_band_ratio_relative": ratio_limit,
        },
        "inputs_sha256": input_hashes,
        "code_sha256": source_hashes,
        "provenance": provenance,
    }
    require(source_hashes == {key: sha256(path) for key, path in source_paths.items()},
            f"[band-projector] {tag}: source changed during this run; no artifact written")
    require(input_hashes == {key: sha256(path) if path is not None else None
                            for key, path in input_paths.items()},
            f"[band-projector] {tag}: input changed during this run; no artifact written")
    atomic_json(target, record)
    print(
        f"BAND_PROJECTOR_DONE {tag}: ratio={exact_ratio:.9g}, "
        f"thresholds={list(thresholds)}, wrote {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
