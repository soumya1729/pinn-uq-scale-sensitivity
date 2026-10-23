"""Producer for the loading table (C5) and its numerics (C6).

WHAT IT SETTLES. The historical Table `tab:loading` reported only
`||Pi_sl g_S||^2 / ||Pi_sl g_B||^2` -- the loading of each region's MEAN tangent. The reported
primary contrast is a per-point aggregate, and the loading governing THAT is
`<||Pi_sl g(x)||^2>_S / <||Pi_sl g(x)||^2>_B`, a different number. This prints both, plus the
factor relating them, which is the coherence quotient of `eq:rmratio` applied to loadings
rather than to sigma. The current table prints both:

    (mean-tangent ratio) = (kappa_S / kappa_B) * (per-point ratio),
    kappa_r = ||Pi_sl <g>_r||^2 / <||Pi_sl g(x)||^2>_r  in [0,1].

kappa_r is how much region r's tangents cancel on averaging. The identity is exact by
construction, so it is asserted rather than reported as evidence.

WHY THE PER-POINT COLUMN IS NOT MERELY A REFINEMENT. The old provisional per-point values used a
projector that dropped Fisher-null directions and are intentionally not repeated here. Schema-v3
artifacts recompute the ordering and seed spread using the theorem-aligned stiff complement.

C6's retained numerics travel with every row and are GATED in the producer, not merely logged:
stiff-projector orthonormality, eigensolver backward error, two-sided cutoff stability, a
non-negative complement, a loading sandwich against the released sigma field, and a sampled
Woodbury reproduction check. The floor-dependent resolved-sloppy and perturbation studies were
explicitly withdrawn; neither is a release claim.

    python analysis/loading_table.py           # table + calibration_figures/loading_table.json
    python analysis/loading_table.py --check    # re-derive and hold to the frozen values

INPUT   results/{TAG}/{TAG}_loading_perpoint.json   (producer: d_loading_perpoint.py, cluster)
OUTPUT  calibration_figures/loading_table.json
"""
import os, sys, json, glob, hashlib, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "calibration_figures")

# The four canonical configurations, in the order Table `tab:loading` prints them.
CANON = [("Burgers, MLP (seed 2)", "burgers_mlp_seed2"),
         ("Burgers, Fourier (seed 2)", "burgers_fourier_seed2"),
         ("Allen--Cahn, MLP (seed 0)", "allen_cahn_mlp_seed0"),
         ("Allen--Cahn, Fourier (seed 0)", "allen_cahn_fourier_seed0")]
CANON_TAGS = {tag for _, tag in CANON}
SEED_FAMILY = "burgers_mlp"          # the family with an across-seed spread

# These targets replace the historical 506/182/269/1130 values computed with the WRONG
# projector -- they sum c^2/lam over sloppy modes and drop the Fisher-null directions, which
# belong to Pi_sl since lam = 0 <= alpha. The theorem-aligned targets below are reconstructed
# from each shipped loading_cdf.json's own totals as ||g||^2 - (captured - loading_below_alpha),
# and differ from the withdrawn values by +1.8%, +7.4%, -0.1% and +0.2% respectively. The
# current manuscript uses generated macros from schema-v3 records, not that withdrawn table.
PAPER_MEAN_TANGENT = {"burgers_mlp_seed2": 514.9, "burgers_fourier_seed2": 195.4,
                      "allen_cahn_mlp_seed0": 268.6, "allen_cahn_fourier_seed0": 1132.4}
TOL = 0.03      # coarse legacy-leg consistency; not bitwise reproduction or a projector proof
SCHEMA = "loading_perpoint/3"   # v3 separates eigensolver accuracy from cutoff stability
REQUIRED_SEEDS = set(range(8))  # the across-seed row is over EXACTLY these; a subset is not it


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_artifact(d, expected_tag):
    if d.get("schema") != SCHEMA:
        raise SystemExit(f"{expected_tag}: artifact schema {d.get('schema')!r}, this reader wants "
                         f"{SCHEMA!r}. Older versions do not satisfy the current numerical "
                         f"validation contract. Some historical outputs additionally use the "
                         f"inverse lift and omit Fisher-null directions. Regenerate or use "
                         f"the validated schema-v3 records; do not mix versions.")
    if d.get("tag") != expected_tag:
        raise SystemExit(f"{expected_tag}: artifact declares tag {d.get('tag')!r}.")
    if d.get("scope") != "full" or d.get("eps_mult") != 1.0:
        raise SystemExit(f"{expected_tag}: primary table requires full scope and eps_mult=1; "
                         f"got scope={d.get('scope')!r}, eps_mult={d.get('eps_mult')!r}.")
    if d.get("sigma_metric") != "raw" or d.get("want_resolved") is not False:
        raise SystemExit(f"{expected_tag}: primary table requires raw metric and the unsuffixed "
                         f"Pi_sl artifact.")
    pde = "allen_cahn" if expected_tag.startswith("allen_cahn") else "burgers"
    arch = "fourier" if "fourier" in expected_tag else "mlp"
    seed = int(expected_tag.rsplit("seed", 1)[1])
    expected_config = {"pde": pde, "arch": arch, "seed": seed,
                       "fourier_scale": 1.0 if pde == "allen_cahn" else 2.0,
                       "eval_nx": 400, "eval_nt": 200}
    if any(d.get("config", {}).get(key) != value for key, value in expected_config.items()):
        raise SystemExit(f"{expected_tag}: recorded configuration is not the declared run.")
    expected_p = 275369 if arch == "fourier" else 196009
    if (int(d.get("P_bayes", -1)) != expected_p or
            int(d.get("n_gram_modes", -1)) != 6400 or
            int(d.get("n_gram_modes_sloppy", -1)) + int(d.get("n_stiff_modes", -1)) != 6400):
        raise SystemExit(f"{expected_tag}: scope dimensions or spectral partition are wrong.")
    alpha, eps, lmax = (float(d.get(key, math.nan)) for key in ("alpha", "eps", "lam_max"))
    if not (math.isfinite(alpha) and alpha > 0 and eps == alpha and lmax >= eps):
        raise SystemExit(f"{expected_tag}: invalid alpha/epsilon/lambda_max contract.")
    if not all(k in d.get("hashes", {}) for k in ("producer", "map_pkl", "sigma_raw", "gram")):
        raise SystemExit(f"{expected_tag}: required provenance hashes are absent.")
    bound_inputs = {
        "producer": os.path.join(ROOT, "d_loading_perpoint.py"),
        "map_pkl": os.path.join(ROOT, "results", expected_tag, f"{expected_tag}_params.pkl"),
        "sigma_raw": os.path.join(
            ROOT, "results", expected_tag, f"{expected_tag}_fullgrid_sigma_raw.npz"),
    }
    if expected_tag in CANON_TAGS:
        bound_inputs["loading_cdf"] = os.path.join(
            ROOT, "results", expected_tag, f"{expected_tag}_loading_cdf.json")
    if arch == "fourier":
        bound_inputs["fourier_B"] = os.path.join(
            ROOT, "results", expected_tag, f"{expected_tag}_fourier_B.npz")
    for hash_name, path in bound_inputs.items():
        if not os.path.isfile(path) or d["hashes"].get(hash_name) != _sha1(path):
            raise SystemExit(f"{expected_tag}: {hash_name} hash is absent or stale ({path}).")
    if not d.get("sandwich") or not d.get("numerics", {}).get("woodbury_sample"):
        raise SystemExit(f"{expected_tag}: stored-sigma sandwich or sampled Woodbury validation "
                         f"is absent.")
    if expected_tag in CANON_TAGS:
        if not d.get("validation", {}).get("legs") or not d["hashes"].get("loading_cdf"):
            raise SystemExit(f"{expected_tag}: canonical CDF validation is absent.")

    tolerances = d.get("validation_tolerances", {})
    caps = {
        "backward_factor_N_eps": 100.0,
        "backward_tolerance": 1.5e-10,
        "raw_inverse_lift_orthogonality_inf_tolerance": math.sqrt(np.finfo(np.float64).eps),
        "orthogonality_inf_tolerance": 5e-11,
        "negative_complement_rel": 1e-10,
        "woodbury_sample_rel": 1e-3,
        "cdf_leg_rel": 1e-3,
        "psd_factor_dimension_eps_normG": 10.0,
    }
    for name, cap in caps.items():
        value = float(tolerances.get(name, math.inf))
        if not (0 < value <= cap * (1 + 1e-12)):
            raise SystemExit(f"{expected_tag}: {name} is absent or broader than {cap:g}.")
    if float(tolerances.get("cutoff_min_safety", -math.inf)) < 10.0:
        raise SystemExit(f"{expected_tag}: cutoff-safety requirement is weaker than 10x.")

    numerics = d["numerics"]
    if (float(numerics["stiff_inverse_lift_orthonormality_inf_norm_raw"]) >
            float(tolerances["raw_inverse_lift_orthogonality_inf_tolerance"]) or
            float(numerics["stiff_projector_orthonormality_inf_norm"]) >
            float(tolerances["orthogonality_inf_tolerance"]) or
            float(numerics["eigenpair_backward_error"]) >
            float(tolerances["backward_tolerance"]) or
            float(numerics["gram_negative_eigenvalue_abs"]) >
            float(tolerances["psd_tolerance_abs"]) or
            float(numerics["worst_negative_complement_rel"]) >
            float(tolerances["negative_complement_rel"])):
        raise SystemExit(f"{expected_tag}: a recorded projector/eigensystem gate did not pass.")
    separation = numerics["separation_around_alpha"]
    if min(float(separation["classification_safety_below"]),
           float(separation["classification_safety_above"])) < float(
               tolerances["cutoff_min_safety"]):
        raise SystemExit(f"{expected_tag}: alpha partition is not numerically separated.")
    sample = numerics["woodbury_sample"]
    if (int(sample.get("n_sampled", 0)) < 250 or
            float(sample.get("max_rel_err", math.inf)) >
            float(tolerances["woodbury_sample_rel"]) or
            float(sample.get("woodbury_min_sigma2_rel", math.inf)) >
            float(tolerances["negative_complement_rel"])):
        raise SystemExit(f"{expected_tag}: sampled Woodbury reconstruction did not pass.")
    for region in ("sharp", "bulk"):
        row = d[region]
        expected_n = 4_000 if region == "sharp" else 56_000
        if not (int(row.get("n", -1)) == expected_n and
                float(row["Q_sl"]) > 0 and float(row["Q_st"]) >= 0 and
                0 <= float(row["M_sl"]) <= float(row["Q_sl"]) * (1 + 1e-8) and
                0 <= float(row["M_st"]) <= float(row["Q_st"]) * (1 + 1e-8) and
                0 <= float(row["kappa"]) <= 1 + 1e-8 and
                np.isclose(float(row["Q_tot"]),
                           float(row["Q_sl"]) + float(row["Q_st"]),
                           rtol=1e-12, atol=1e-15) and
                np.isclose(float(row["kappa"]),
                           float(row["M_sl"]) / float(row["Q_sl"]),
                           rtol=1e-12, atol=1e-15)):
            raise SystemExit(f"{expected_tag}: {region} loading/coherence constraints failed.")
        band = d["sandwich"][region]
        if not (float(band["lower"]) <= float(band["sigma2_stored"]) * (1 + 1e-9) and
                float(band["sigma2_stored"]) <= float(band["upper"]) * (1 + 1e-9)):
            raise SystemExit(f"{expected_tag}: {region} stored sigma is outside its sandwich.")
    sharp, bulk = d["sharp"], d["bulk"]
    if not (float(bulk["M_sl"]) > 0 and
            np.isclose(float(d["ratio_perpoint_sl"]),
                       float(sharp["Q_sl"]) / float(bulk["Q_sl"]),
                       rtol=1e-12, atol=1e-15) and
            np.isclose(float(d["ratio_mean_tangent_sl"]),
                       float(sharp["M_sl"]) / float(bulk["M_sl"]),
                       rtol=1e-12, atol=1e-15) and
            np.isclose(float(d["ratio_mean_tangent_sl"]) /
                       float(d["ratio_perpoint_sl"]),
                       float(sharp["kappa"]) / float(bulk["kappa"]),
                       rtol=1e-12, atol=1e-15)):
        raise SystemExit(f"{expected_tag}: reported loading ratios violate their definitions.")
    if expected_tag in CANON_TAGS:
        if any(float(leg["rel_err"]) > float(tolerances["cdf_leg_rel"])
               for leg in d["validation"]["legs"].values()):
            raise SystemExit(f"{expected_tag}: canonical mean-tangent CDF leg check failed.")
    return d


def _load(tag):
    p = os.path.join(ROOT, "results", tag, f"{tag}_loading_perpoint.json")
    if not os.path.exists(p):
        return None
    with open(p) as _f:
        return _validate_artifact(json.load(_f), tag)


def _row(d):
    s, b = d["sharp"], d["bulk"]
    kS, kB = s["kappa"], b["kappa"]
    legs = d.get("validation", {}).get("legs", {})
    leg_errors = [v["rel_err"] for v in legs.values() if "rel_err" in v]
    cutoff = d["numerics"]["separation_around_alpha"]
    return {"tag": d["tag"], "per_point": d["ratio_perpoint_sl"],
            "mean_tangent": d["ratio_mean_tangent_sl"],
            "per_point_resolved": d.get("ratio_perpoint_sl_resolved"),
            "quotient": d["ratio_mean_tangent_sl"] / d["ratio_perpoint_sl"],
            "kappa_S": kS, "kappa_B": kB,
            "Q_sl_S": s["Q_sl"], "Q_st_S": s["Q_st"],
            "Q_sl_B": b["Q_sl"], "Q_st_B": b["Q_st"],
            "M_sl_S": s["M_sl"], "M_sl_B": b["M_sl"],
            "n_gram_modes_sloppy": d["n_gram_modes_sloppy"], "n_stiff_modes": d["n_stiff_modes"],
            "orthonormality": d["numerics"]["stiff_projector_orthonormality_inf_norm"],
            "eigenpair_backward_error": d["numerics"]["eigenpair_backward_error"],
            "cutoff_min_safety": min(cutoff["classification_safety_below"],
                                        cutoff["classification_safety_above"]),
            "resolved_vs_Pi_sl": (d["sensitivity"].get("resolved_vs_Pi_sl_perpoint")
                                  if d["sensitivity"].get("computed") else None),
            "sandwich_holds": all(v["inside"] for v in d["sandwich"].values()),
            "validation_rel_err": max(leg_errors) if leg_errors else None}


def main():
    unknown = [arg for arg in sys.argv[1:] if arg != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown)}")
    check = "--check" in sys.argv
    rows, missing, fails = {}, [], []

    for label, tag in CANON:
        d = _load(tag)
        if d is None:
            missing.append(f"{label}: no loading_perpoint.json")
            continue
        rows[label] = _row(d)

    print(f"{'configuration':<32}{'per-point':>11}{'mean-tan':>10}{'quotient':>10}"
          f"{'kappa_S':>9}{'kappa_B':>9}")
    for label, _ in CANON:
        if label not in rows:
            print(f"{label:<32}   (absent)")
            continue
        r = rows[label]
        print(f"{label:<32}{r['per_point']:>11.1f}{r['mean_tangent']:>10.1f}"
              f"{r['quotient']:>10.2f}{r['kappa_S']:>9.3f}{r['kappa_B']:>9.4f}")

    # the identity, asserted -- it is algebra, so a violation means a corrupt artifact
    for label, r in rows.items():
        lhs = r["mean_tangent"] / r["per_point"]
        rhs = r["kappa_S"] / r["kappa_B"]
        if abs(lhs - rhs) > 1e-6 * max(1.0, abs(rhs)):
            fails.append(f"{label}: coherence identity violated, {lhs:.6g} vs {rhs:.6g}")

    # across-seed spread on the one family that has one
    seeds, have = [], set()
    for p in sorted(glob.glob(os.path.join(ROOT, "results", f"{SEED_FAMILY}_seed*",
                                           f"{SEED_FAMILY}_seed*_loading_perpoint.json"))):
        expected_tag = os.path.basename(os.path.dirname(p))
        with open(p) as _f:
            d = _validate_artifact(json.load(_f), expected_tag)
        seeds.append(_row(d)); have.add(int(d["config"]["seed"]))
    if have != REQUIRED_SEEDS or len(seeds) != len(REQUIRED_SEEDS):
        fails.append(f"{SEED_FAMILY}: across-seed row needs seeds {sorted(REQUIRED_SEEDS)}, "
                     f"have {sorted(have)} across {len(seeds)} artifacts. A partial or duplicate "
                     f"set is a different statistic, and the "
                     f"paper's 17% seed-stability claim is exactly what quoting one became.")
    spread = {}
    if seeds:
        agg = lambda k: (lambda a: {"mean": float(a.mean()), "sd": float(a.std(ddof=0)),
                                    "min": float(a.min()), "max": float(a.max()),
                                    "max_over_min": float(a.max() / a.min()),
                                    "per_seed": [float(x) for x in a]})(
            np.array([r[k] for r in seeds]))
        spread = {"n": len(seeds), "per_point": agg("per_point"),
                  "mean_tangent": agg("mean_tangent"), "quotient": agg("quotient")}
        print(f"\n{SEED_FAMILY}, n={spread['n']}, mean +- population sd:")
        for k in ("per_point", "mean_tangent", "quotient"):
            v = spread[k]
            print(f"  {k:<13}{v['mean']:8.1f} +- {v['sd']:5.1f}   "
                  f"[{v['min']:.1f}, {v['max']:.1f}]   max/min {v['max_over_min']:.2f}x")

    print("\nC6 numerics, worst over the configurations present:")
    if rows:
        print(f"  orthonormality       {max(r['orthonormality'] for r in rows.values()):.2e}")
        print(f"  backward error       {max(r['eigenpair_backward_error'] for r in rows.values()):.2e}")
        print(f"  cutoff safety        {min(r['cutoff_min_safety'] for r in rows.values()):.1f}x")
        print(f"  sandwich holds everywhere  {all(r['sandwich_holds'] for r in rows.values())}")

    if check:
        for label, tag in CANON:
            if label not in rows:
                fails.append(f"{label}: absent, cannot check against the printed table")
                continue
            got = rows[label]["mean_tangent"]
            want = PAPER_MEAN_TANGENT[tag]
            if abs(got - want) > TOL * want:
                fails.append(f"{label}: mean-tangent Pi_sl recomputes to {got:.1f}, the "
                             f"theorem-aligned target from the shipped totals is {want:.1f} "
                             f"({abs(got-want)/want*100:.1f}%)")

    for m in missing:
        fails.append(m)
    if fails:
        print("\nFAIL")
        for m in fails:
            print("  " + m)
        sys.exit(1)

    # Producer mode writes only once every validation has passed, and atomically. Check mode is
    # deliberately read-only so a verification command cannot mutate the release it inspects.
    if not check:
        os.makedirs(OUT, exist_ok=True)
        _p = os.path.join(OUT, "loading_table.json")
        with open(_p + ".partial", "w") as _f:
            json.dump({"schema": SCHEMA, "canonical": rows, "seed_spread": spread}, _f, indent=2,
                      allow_nan=False)
            _f.write("\n"); _f.flush(); os.fsync(_f.fileno())
        os.replace(_p + ".partial", _p)
    print("\nOK" + (" (matches the theorem-aligned targets and the generated tab:loading macros)"
                if check else ""))


if __name__ == "__main__":
    main()
