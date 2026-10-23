#!/usr/bin/env python3
"""Validate four current-J band-projector records and build their release summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legacy_loading_contract as legacy


SCHEMA = "band_projector_check/2"
TAGS = (
    "burgers_mlp_seed2",
    "burgers_fourier_seed2",
    "allen_cahn_mlp_seed0",
    "allen_cahn_fourier_seed0",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def validate(root: Path, artifact_root: Path) -> dict:
    producer_hash = sha256(root / "d_band_projector_check.py")
    common_hash = sha256(root / "common.py")
    laplace_hash = sha256(root / "laplace_core.py")
    helper_hash = sha256(root / "legacy_loading_contract.py")
    require(helper_hash == sha256(Path(legacy.__file__).resolve()),
            "the validation root uses a different legacy-loading helper than this reader loaded")
    band_path = root / "results" / "_theorem_band" / "theorem_band_check.json"
    band = read_json(band_path)
    configurations = {}
    artifact_hashes = {}
    for tag in TAGS:
        path = artifact_root / tag / f"{tag}_band_projector.json"
        record = read_json(path)
        require(record.get("schema") == SCHEMA, f"{tag}: wrong schema")
        require(record.get("cdf_accounting_contract") == legacy.ACCOUNTING_CONTRACT,
                f"{tag}: missing signed-closure complementary accounting contract")
        require(record.get("tag") == tag and record.get("sigma_metric") == "raw",
                f"{tag}: tag/metric mismatch")
        require(record.get("scope") == "full" and
                int(record.get("N_residual_rows", -1)) == 6400 and
                int(record.get("P_bayes", -1)) == (275369 if "fourier" in tag else 196009),
                f"{tag}: scope or Jacobian dimensions do not match the declared experiment")
        require(record.get("validation_passed") is True, f"{tag}: validation did not pass")
        require(float(record["spectrum_max_abs_over_lambda_max"]) <=
                float(record["tolerances"]["spectrum_max_abs_over_lambda_max"]),
                f"{tag}: current spectrum/CDF gate not satisfied")
        require(record.get("code_sha256", {}).get("producer") == producer_hash,
                f"{tag}: producer hash differs from the released source")
        require(record.get("code_sha256", {}).get("common") == common_hash and
                record.get("code_sha256", {}).get("laplace_core") == laplace_hash and
                record.get("code_sha256", {}).get("legacy_loading_contract") == helper_hash,
                f"{tag}: common/laplace_core/legacy-loading helper hash differs from the released source")
        pde = "allen_cahn" if tag.startswith("allen_cahn") else "burgers"
        arch = "fourier" if "fourier" in tag else "mlp"
        seed = int(tag.rsplit("seed", 1)[1])
        config = record.get("configuration", {})
        expected_config = {
            "pde": pde, "arch": arch, "seed": seed,
            "fourier_scale": 1.0 if pde == "allen_cahn" else 2.0,
            "fourier_dim": 16, "Mx": 20, "Mt": 20, "hidden": 64,
            "fusion": 128, "d": 128, "d_aug": 129,
            "N_f": 5000, "N_u": 1000, "N_b": 200,
            "ic_weight": 10.0, "bc_weight": 10.0,
            "bcu_weight": 10.0, "bcx_weight": 1.0,
            "eval_nx": 400, "eval_nt": 200, "smoke": 0,
        }
        expected_config["nu" if pde == "burgers" else "eps2"] = (
            0.01 / math.pi if pde == "burgers" else 1e-4
        )
        if pde == "allen_cahn":
            expected_config["reac"] = 5.0
        require(all(config.get(key) == value for key, value in expected_config.items()),
                f"{tag}: configuration is not the admitted canonical run")
        provenance_raw = record.get("provenance", {}).get("env", {}).get("RAW_METRIC")
        require(provenance_raw == "1", f"{tag}: provenance does not record RAW_METRIC=1")
        expected_inputs = {
            "map": root / "results" / tag / f"{tag}_params.pkl",
            "loading_cdf": root / "results" / tag / f"{tag}_loading_cdf.json",
            "theorem_band": band_path,
        }
        if "fourier" in tag:
            expected_inputs["fourier_sidecar"] = (
                root / "results" / tag / f"{tag}_fourier_B.npz"
            )
        for key, input_path in expected_inputs.items():
            require(input_path.is_file(), f"{tag}: missing bound input {input_path}")
            require(record["inputs_sha256"].get(key) == sha256(input_path),
                    f"{tag}: {key} hash drift")
        if "fourier" not in tag:
            require(record["inputs_sha256"].get("fourier_sidecar") is None,
                    f"{tag}: unexpected Fourier sidecar")
        cdf = read_json(expected_inputs["loading_cdf"])
        cdf_lam = np.asarray(cdf.get("lambda_sorted", []), dtype=np.float64)
        cdf_regions = {
            name: legacy.load_region(cdf[old_name], cdf_lam, f"{tag}/{name}",
                                    float(record["tolerances"]["legacy_loading_cdf_relative"]))
            for name, old_name in (("sharp", "shock"), ("bulk", "smooth"))
        }
        require(record.get("cdf_closure_diagnostics") == {
            name: legacy.closure_diagnostic(region) for name, region in cdf_regions.items()
        }, f"{tag}: legacy signed closure diagnostics do not reconstruct from the bound CDF")

        summary = band.get(tag)
        require(isinstance(summary, dict), f"{tag}: missing theorem-band summary")
        require(summary.get("cdf_accounting_contract") == legacy.ACCOUNTING_CONTRACT,
                f"{tag}: theorem-band summary predates signed-closure accounting")
        alpha = float(record.get("alpha", math.nan))
        lambda_max = float(record.get("lambda_max", math.nan))
        require(math.isfinite(alpha) and alpha > 0 and math.isfinite(lambda_max) and
                lambda_max > 0 and
                math.isclose(alpha, float(summary.get("alpha", math.nan)),
                             rel_tol=1e-12, abs_tol=0.0) and
                math.isclose(lambda_max, float(summary.get("lambda_max", math.nan)),
                             rel_tol=1e-10, abs_tol=0.0),
                f"{tag}: alpha/lambda_max disagrees with the theorem-band summary")
        require(abs(float(record["exact_regmean_ratio"]) -
                    float(summary["exact_regmean_ratio"])) <=
                1e-7 * float(summary["exact_regmean_ratio"]),
                f"{tag}: exact ratio disagrees with theorem-band summary")
        for name, field in (("lower_optimum", "lower"), ("upper_optimum", "upper")):
            measured = record["thresholds"].get(name)
            require(measured is not None, f"{tag}: missing {name}")
            require(math.isclose(float(measured["epsilon"]),
                                 float(summary["optimization"][field + "_eps"]),
                                 rel_tol=0.0, abs_tol=0.0),
                    f"{tag}: {name} epsilon drift")
            require(measured["side"] == summary["optimization"][field + "_side"],
                    f"{tag}: {name} side drift")
        expected_thresholds = {"alpha", "lower_optimum", "upper_optimum"}
        if tag == "burgers_mlp_seed2":
            expected_thresholds.add("information_optimum")
        require(set(record.get("thresholds", {})) == expected_thresholds,
                f"{tag}: threshold set is incomplete or contains an undeclared result")
        alpha_threshold = record["thresholds"]["alpha"]
        require(alpha_threshold.get("side") == "at" and
                math.isclose(float(alpha_threshold.get("epsilon", math.nan)), alpha,
                             rel_tol=1e-12, abs_tol=0.0),
                f"{tag}: alpha threshold is not the current prior precision")
        tolerances = record.get("tolerances", {})
        require(0 < float(tolerances.get("legacy_loading_cdf_relative", math.inf)) <= 5e-7,
                f"{tag}: legacy CDF roundoff allowance is missing or too broad")
        # Environment variables may tighten these gates for diagnostics, but a rerun may not
        # silently loosen the release standard and then certify itself against the looser
        # number stored in its own JSON.
        upper_caps = {
            "raw_inverse_lift_orthogonality_inf": 2e-8,
            "corrected_projector_orthogonality_inf": 1e-8,
            "negative_complement_relative": 1e-8,
            "cdf_leg_relative": 1e-6,
            "spectrum_max_abs_over_lambda_max": 1e-10,
            "eigenpair_backward_error": 1e-10,
            "direct_vs_cdf_variance_relative": 5e-6,
            "theorem_band_ratio_relative": 5e-6,
        }
        for tolerance_name, cap in upper_caps.items():
            require(0 < float(tolerances.get(tolerance_name, math.inf)) <= cap,
                    f"{tag}: {tolerance_name} is missing or broader than {cap:g}")
        require(float(tolerances.get("partition_gap_safety_min", -math.inf)) >= 50.0,
                f"{tag}: ordinal partition-gap requirement is weaker than 50x")
        for region in ("sharp", "bulk"):
            direct = float(record.get("direct_sigma2", {}).get(region, math.nan))
            cdf = float(record.get("cdf_sigma2", {}).get(region, math.nan))
            expected_cdf = legacy.variance(cdf_regions[region], cdf_lam, alpha)
            require(math.isclose(cdf, expected_cdf, rel_tol=1e-12, abs_tol=0.0),
                    f"{tag}/{region}: CDF variance does not reconstruct from the bound CDF")
            relative = float(record.get("direct_vs_cdf_variance_relative_error", {}).get(
                region, math.nan))
            expected_relative = abs(direct - cdf) / max(
                abs(direct), float.fromhex("0x1.0p-1022"))
            require(math.isfinite(direct) and direct > 0 and math.isfinite(cdf) and cdf > 0 and
                    math.isclose(relative, expected_relative, rel_tol=1e-12, abs_tol=1e-15) and
                    relative <= float(tolerances["direct_vs_cdf_variance_relative"]),
                    f"{tag}/{region}: direct-vs-CDF variance gate not satisfied")
        reconstructed_ratio = math.sqrt(
            float(record["direct_sigma2"]["sharp"]) /
            float(record["direct_sigma2"]["bulk"]))
        require(math.isclose(float(record["exact_regmean_ratio"]), reconstructed_ratio,
                             rel_tol=1e-12, abs_tol=1e-15),
                f"{tag}: exact region-mean ratio is not reconstructed from its two legs")
        expected_ratio_relative = abs(reconstructed_ratio -
                                      float(summary["exact_regmean_ratio"])) / abs(
                                          float(summary["exact_regmean_ratio"]))
        require(float(record["theorem_band_ratio_relative_error"]) <=
                float(tolerances["theorem_band_ratio_relative"]) and
                math.isclose(float(record["theorem_band_ratio_relative_error"]),
                             expected_ratio_relative, rel_tol=1e-12, abs_tol=1e-15),
                f"{tag}: theorem-band ratio gate not satisfied")
        for name, threshold in record["thresholds"].items():
            require(threshold.get("side") in {"at", "left-limit", "zero-plus"} and
                    math.isfinite(float(threshold.get("epsilon", math.nan))) and
                    float(threshold["epsilon"]) >= 0 and
                    int(threshold.get("n_sloppy", -1)) >= 0 and
                    int(threshold.get("n_stiff", -1)) >= 0 and
                    int(threshold["n_sloppy"]) + int(threshold["n_stiff"]) == 6400,
                    f"{tag}/{name}: threshold definition/counts are malformed")
            search_side = "left" if threshold["side"] == "left-limit" else "right"
            cdf_k = int(np.searchsorted(cdf_lam, float(threshold["epsilon"]), side=search_side))
            require(cdf_k == int(threshold["n_sloppy"]),
                    f"{tag}/{name}: recorded split count does not match the bound CDF")
            require(float(threshold["raw_inverse_lift_orthogonality_inf"]) <=
                    float(tolerances["raw_inverse_lift_orthogonality_inf"]),
                    f"{tag}/{name}: raw inverse-lift gate not satisfied")
            require(float(threshold["corrected_projector_orthogonality_inf"]) <=
                    float(tolerances["corrected_projector_orthogonality_inf"]),
                    f"{tag}/{name}: orthogonality gate not satisfied")
            require(float(threshold["eigenpair_backward_error"]) <=
                    float(tolerances["eigenpair_backward_error"]),
                    f"{tag}/{name}: eigenpair backward-error gate not satisfied")
            if threshold.get("partition_is_trivial_all_sloppy"):
                require(threshold.get("partition_gap") is None and
                        threshold.get("partition_gap_safety") is None,
                        f"{tag}/{name}: trivial all-sloppy partition is misrecorded")
            else:
                require(float(threshold["partition_gap_safety"]) >=
                        float(tolerances["partition_gap_safety_min"]),
                        f"{tag}/{name}: ordinal partition-gap gate not satisfied")
            for region in ("sharp", "bulk"):
                leg = threshold["legs"][region]
                total = float(leg.get("total", math.nan))
                sloppy = float(leg.get("sloppy", math.nan))
                stiff = float(leg.get("stiff", math.nan))
                complement = float(leg.get("cdf_complement", math.nan))
                expected_complement, _ = legacy.split_parts(cdf_regions[region], cdf_k)
                require(math.isclose(complement, expected_complement, rel_tol=1e-12, abs_tol=0.0),
                        f"{tag}/{name}/{region}: CDF complement does not reconstruct from the bound CDF")
                require(all(math.isfinite(value) for value in
                            (total, sloppy, stiff, complement)) and
                        total > 0 and sloppy >= 0 and stiff >= 0 and complement >= 0 and
                        abs(total - sloppy - stiff) <=
                        float(tolerances["negative_complement_relative"]) * total,
                        f"{tag}/{name}/{region}: projector energies do not add up")
                expected_leg_error = abs(sloppy - complement) / max(
                    abs(complement), float.fromhex("0x1.0p-1022"))
                require(float(leg["negative_complement_relative"]) <=
                        float(tolerances["negative_complement_relative"]),
                        f"{tag}/{name}/{region}: negative-complement gate not satisfied")
                require(math.isclose(float(leg["cdf_leg_relative_error"]),
                                     expected_leg_error, rel_tol=1e-12, abs_tol=1e-15) and
                        float(leg["cdf_leg_relative_error"]) <=
                        float(tolerances["cdf_leg_relative"]),
                        f"{tag}/{name}/{region}: leg gate not satisfied")
                require(float(leg["cdf_total_relative_error"]) <=
                        float(tolerances["cdf_leg_relative"]) and
                        math.isclose(float(leg["cdf_total_relative_error"]),
                                     abs(total - cdf_regions[region]["tot"]) / cdf_regions[region]["tot"],
                                     rel_tol=1e-12, abs_tol=1e-15),
                        f"{tag}/{name}/{region}: tangent-total gate not satisfied")
        configurations[tag] = record
        artifact_hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256(path)

    return {
        "schema": SCHEMA,
        "cdf_accounting_contract": legacy.ACCOUNTING_CONTRACT,
        "sigma_metric": "raw",
        "configurations": configurations,
        "artifact_sha256": artifact_hashes,
        "code_sha256": {"producer": producer_hash, "common": common_hash,
                        "laplace_core": laplace_hash, "legacy_loading_contract": helper_hash},
        "theorem_band_sha256": sha256(band_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_root = (
        args.artifacts_root or root / "results" / "_band_projector"
    ).resolve()
    summary = validate(root, artifact_root)
    target = root / "results" / "_band_projector" / "band_projector_check.json"
    canonical = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        require(target.is_file(), f"missing {target}")
        require(target.read_text(encoding="utf-8") == canonical,
                f"{target} is stale")
        print("BAND PROJECTOR CHECK: PASS")
    else:
        atomic_text(target, canonical)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
