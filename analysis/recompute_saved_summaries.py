"""Reproduce scalar paper checks from saved JSON only (Python standard library).

Run `python analysis/recompute_saved_summaries.py` from any directory.
The optional --import-from argument refreshes the small, byte-exact input copies.
No trained model, Jacobian, eigensolver, accelerator or experiment is evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL = ["burgers_mlp_seed2", "burgers_fourier_seed2",
             "allen_cahn_mlp_seed0", "allen_cahn_fourier_seed0"]
TAGS = [f"burgers_mlp_seed{i}" for i in range(8)] + CANONICAL[1:]
SLOPE_TAGS = [CANONICAL[0], CANONICAL[3], CANONICAL[2]]
INPUTS = [f"{t}/{t}_loading_perpoint.json" for t in TAGS]
INPUTS += [f"{t}/{t}_spectral_loading.json" for t in SLOPE_TAGS]
INPUTS += [f"_alpha/burgers_mlp_seed0_M{m}.json" for m in ["0.1", "1", "10"]]
INPUTS += ["burgers_mlp_seed2/burgers_mlp_seed2_loading_cdf.json",
           "_theorem_band/theorem_band_check.json"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((HERE / "input" / relative).read_text(encoding="utf-8"),
                      parse_float=Decimal)


def dec(value):
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(value)


def exact(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def root(value):
    return dec(value).sqrt()


def outward(value, places, lower):
    d = root(value).quantize(Decimal(1).scaleb(-places),
                            rounding=ROUND_FLOOR if lower else ROUND_CEILING)
    # Exact rational checks certify display direction relative to the recorded inputs.
    if lower:
        assert Fraction(d) ** 2 <= value
    else:
        assert Fraction(d) ** 2 >= value
    return format(d, "f")


def close(a, b, tolerance=Decimal("1e-12")):
    assert abs(dec(a) - dec(b)) <= tolerance * max(abs(dec(a)), abs(dec(b)), Decimal(1))


def analyse_rms(tag):
    relative = f"{tag}/{tag}_loading_perpoint.json"
    d = read(relative)
    assert d["tag"] == tag and d["scope"] == "full" and d["sigma_metric"] == "raw"
    a, e, lm = (Fraction(d[k]) for k in ["alpha", "eps", "lam_max"])
    assert a > 0 and e == a and lm >= e
    parts = {}
    for region in ["sharp", "bulk"]:
        z = d[region]
        qs, qt, actual = (Fraction(z[k]) for k in ["Q_sl", "Q_st", "sigma2_stored_mean"])
        assert qs >= 0 and qt >= 0
        lo, hi = qs / (a + e) + qt / (a + lm), qs / a + qt / (a + e)
        assert 0 < lo <= actual <= hi
        close(lo, d["sandwich"][region]["lower"])
        close(hi, d["sandwich"][region]["upper"])
        parts[region] = {"lower": lo, "upper": hi, "actual": actual, "sloppy": qs}
    sharp, bulk = parts["sharp"], parts["bulk"]
    lower = sharp["lower"] / bulk["upper"]
    legacy_lower = (sharp["sloppy"] / (a + e)) / bulk["upper"]
    upper = sharp["upper"] / bulk["lower"]
    observed = sharp["actual"] / bulk["actual"]
    assert 0 <= legacy_lower <= lower <= observed <= upper
    return {"tag": tag, "source": relative, "source_sha256": digest(HERE / "input" / relative),
            "seed": d["config"]["seed"], "alpha": str(d["alpha"]), "epsilon": str(d["eps"]),
            "lambda_max": str(d["lam_max"]),
            "lower_sd": str(root(lower)), "upper_sd": str(root(upper)),
            "legacy_sloppy_only_lower_sd": str(root(legacy_lower)),
            "observed_rms_contrast": str(root(observed)),
            "lower_display_outward": outward(lower, 4, True),
            "upper_display_outward": outward(upper, 2, False),
            "variance_ratio_rationals": {"lower": exact(lower), "upper": exact(upper),
                                         "legacy_lower": exact(legacy_lower), "observed": exact(observed)},
            "regional_enclosures_pass": True,
            "stored_sandwich_already_two_term": True}


def ols(xs, ys):
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar)**2 for x in xs)


def analyse_slopes():
    records = []
    for tag in SLOPE_TAGS:
        d = read(f"{tag}/{tag}_spectral_loading.json")
        aa = list(map(dec, d["alphas"]))
        rr = list(map(dec, d["metrics"]["raw"]["R_of_alpha"]))
        slope = ols([x.ln() for x in aa], [x.ln() for x in rr])
        close(slope, d["metrics"]["raw"]["dlogR_dlogalpha"])
        assert len(aa) == len(rr) == 11 and d["K"] == 30
        records.append({"tag": tag, "sampled_points_per_region": d["K"],
                        "alpha_count": len(aa), "alpha_min_over_alpha0": str(aa[0]/d["alpha0"]),
                        "alpha_max_over_alpha0": str(aa[-1]/d["alpha0"]),
                        "recorded_ols_loglog_slope": str(d["metrics"]["raw"]["dlogR_dlogalpha"]),
                        "recomputed_ols_loglog_slope": str(slope),
                        "statistic": "mean pointwise standard deviation contrast on sampled points",
                        "is_local_derivative": False})
    full = [read(f"_alpha/burgers_mlp_seed0_M{m}.json") for m in ["0.1", "1", "10"]]
    secant = (full[-1]["raw"]["ratio_mean"] / full[0]["raw"]["ratio_mean"]).ln() / (full[-1]["alpha"] / full[0]["alpha"]).ln()
    return {"fits": records,
            "full_grid_seed0_comparator": {"tag": "burgers_mlp_seed0",
                "mean_width_contrasts_ascending_alpha": [str(z["raw"]["ratio_mean"]) for z in full],
                "endpoint_secant_from_rounded_saved_values": str(secant),
                "note": "Different seed, full-grid evaluation, 100-fold range and endpoint secant; not a replacement for the sampled 11-point OLS fits."}}


def analyse_mean_tangent():
    d = read("burgers_mlp_seed2/burgers_mlp_seed2_loading_cdf.json")
    a = d["alpha"]
    lam = d["lambda_sorted"]
    k = sum(x <= a for x in lam)
    parts = {}
    for region in ["shock", "smooth"]:
        z = d[region]
        cum = z["cum_loading"]
        c2 = [cum[0]] + [v - u for u, v in zip(cum, cum[1:])]
        stiff_energy = z["captured"] - cum[k - 1]
        sloppy_energy = z["total_gg"] - stiff_energy
        stiff_var = sum(w / (a + l) for w, l in zip(c2[k:], lam[k:]))
        sloppy_var = (sloppy_energy - sum(w * l / (a + l) for w, l in zip(c2[:k], lam[:k]))) / a
        assert min(stiff_energy, sloppy_energy, stiff_var, sloppy_var) > 0
        parts[region] = {"stiff_energy": stiff_energy, "sloppy_energy": sloppy_energy,
                         "stiff_variance": stiff_var, "sloppy_variance": sloppy_var,
                         "sloppy_variance_fraction": sloppy_var/(stiff_var+sloppy_var)}
    s, b = parts["shock"], parts["smooth"]
    stiff_ratio = s["stiff_variance"] / b["stiff_variance"]
    energy_ratio = s["stiff_energy"] / b["stiff_energy"]
    combined = ((s["stiff_variance"]+s["sloppy_variance"])/(b["stiff_variance"]+b["sloppy_variance"])).sqrt()
    expected = read("_theorem_band/theorem_band_check.json")["burgers_mlp_seed2"]["exact_regmean_ratio"]
    close(combined, expected, Decimal("1e-8"))
    return {"tag": d["tag"], "cutoff": "epsilon = alpha", "alpha": str(a),
            "aggregation": "region-mean tangent; not primary regional RMS",
            "parts": {r: {k: str(v) for k, v in z.items()} for r, z in parts.items()},
            "stiff_variance_ratio": str(stiff_ratio), "stiff_sd_ratio": str(stiff_ratio.sqrt()),
            "stiff_energy_ratio": str(energy_ratio), "stiff_inverse_precision_weighting_ratio": str(stiff_ratio/energy_ratio),
            "combined_mean_tangent_sd_ratio": str(combined),
            "note": "Uses signed closure via total minus captured stiff tail, never clamps the legacy nullspace addend separately. This decomposition at alpha does not explain the separately cutoff-optimised upper bound 78."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-from", type=Path)
    args = parser.parse_args()
    input_dir = HERE / "input"
    if args.import_from:
        manifest = []
        for relative in INPUTS:
            source, target = args.import_from / relative, input_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest.append({"relative_path": relative, "source_path": str(source.resolve()),
                             "sha256": digest(source), "bytes": source.stat().st_size})
        source = args.import_from.parent / "step2b_spectral_loading.py"
        if not source.exists():
            source = Path(r"C:\dev\pinn-uq\pinn-uq-core\step2b_spectral_loading.py")
        target = input_dir / "producers" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"relative_path": "producers/" + source.name, "source_path": str(source.resolve()),
                         "sha256": digest(source), "bytes": source.stat().st_size})
        (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        assert digest(input_dir / item["relative_path"]) == item["sha256"], item["relative_path"]
    with localcontext() as context:
        context.prec = 80
        rows = [analyse_rms(t) for t in TAGS]
        seed_rows = rows[:8]
        stats = {}
        for key in ["lower_sd", "upper_sd", "observed_rms_contrast"]:
            values = [Decimal(r[key]) for r in seed_rows]
            mean = sum(values)/len(values)
            stats[key] = {"mean": str(mean), "population_sd": str((sum((v-mean)**2 for v in values)/len(values)).sqrt()),
                          "minimum": str(min(values)), "maximum": str(max(values))}
        report = {"schema": "saved_scalar_analysis/1", "decimal_precision": 80,
                  "scope": "Saved scalar summaries only; no new model, Jacobian, covariance or suite evaluation.",
                  "rounding_note": "Display bounds are outward rounded and their squares checked exactly against rational expressions of the recorded decimal inputs. This does not certify errors in the original floating-point inputs.",
                  "finding": "All 11 stored per-region sandwich lower bounds already include the stiff term; Table 8 therefore already used the stronger lower bound numerically.",
                  "canonical_tags": CANONICAL, "rows": rows,
                  "burgers_mlp_eight_seed_summary": stats,
                  "mean_tangent_decomposition": analyse_mean_tangent(), "slope_provenance": analyse_slopes()}
    (HERE / "saved_scalar_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = ["tag", "seed", "alpha", "epsilon", "lambda_max", "legacy_sloppy_only_lower_sd", "lower_sd", "upper_sd", "observed_rms_contrast", "lower_display_outward", "upper_display_outward", "source_sha256"]
    with (HERE / "rms_bounds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    codes = ["BurgersMLP", "BurgersFourier", "AllenCahnMLP", "AllenCahnFourier"]
    tex = ["% Generated by analysis/recompute_saved_summaries.py; outward rounding of recorded inputs."]
    by_tag = {r["tag"]: r for r in rows}
    for tag, code in zip(CANONICAL, codes):
        r = by_tag[tag]
        tex += [r"\newcommand{\RMSBound"+code+r"Lower}{"+r["lower_display_outward"]+"}",
                r"\newcommand{\RMSBound"+code+r"Upper}{"+r["upper_display_outward"]+"}"]
    (HERE.parent / "generated" / "rms_bounds_macros.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    print("PASS: 11 saved regional enclosures; stronger/legacy ratio ordering; exact display direction; 3 OLS fits; reference mean-tangent consistency.")
    for tag in CANONICAL:
        r = by_tag[tag]
        print(f"{tag}: [{r['lower_display_outward']}, {r['upper_display_outward']}], observed {float(r['observed_rms_contrast']):.8f}")
    print("Mean-tangent stiff SD contrast:", report["mean_tangent_decomposition"]["stiff_sd_ratio"])


if __name__ == "__main__":
    main()
