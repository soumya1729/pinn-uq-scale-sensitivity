"""Check the registered numerical claims in the abstract/contribution bullets.

This is deliberately a bounded claim audit, not a claim to verify every number
in the paper. It aggregates records; --reproduced prefers the fresh outputs of
reproduce.py. Frozen fallback records are explicitly labelled in the report.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_per_tag(rows, required=None):
    tags = [row["tag"] for row in rows]
    if len(tags) != len(set(tags)):
        raise ValueError("A metric expected exactly one selected row per checkpoint")
    if required is not None and set(tags) != set(required):
        raise ValueError(f"Selected checkpoint set differs: {set(tags) ^ set(required)}")
    return rows


class Records:
    def __init__(self, root, reproduced):
        self.root, self.reproduced = root, reproduced
        self.used = {}

    def get(self, key, frozen, fresh=None):
        candidate = self.reproduced / fresh if self.reproduced is not None and fresh else None
        if candidate is not None and candidate.is_file():
            path, level = candidate, "AGGREGATED_FRESH_RECOMPUTED_RECORDS"
        else:
            path, level = self.root / frozen, "CHECKED_SAVED_SUMMARY"
        if not path.is_file():
            raise FileNotFoundError(path)
        self.used[key] = {"path": str(path), "sha256": sha256(path), "verification_level": level}
        return read_csv(path) if path.suffix == ".csv" else read_json(path)


def calculate(records):
    cohort = records.get("cohort", "analysis/round2/cohort.json")["cohort"]
    tags = [r["tag"] for r in cohort]
    if len(tags) != len(set(tags)):
        raise ValueError("Cohort tags are not unique")
    bm = {r["tag"] for r in cohort if r["pde"] == "burgers" and r["architecture"] == "mlp"}
    roots = records.get("roots", "analysis/round3_local/outputs/refined_roots.csv", "round3/refined_roots.csv")
    root_protocol = records.get("root_protocol", "analysis/round3_local/outputs/summary.json", "round3/summary.json")["protocol"]
    root_tags = {r["tag"] for r in roots}
    one_per_tag(roots)
    gammas = [float(r["root_gamma"]) for r in roots]
    if any(g <= 0 for g in gammas):
        raise ValueError("Root precisions must be positive")

    scales = records.get("scale_metrics", "analysis/round2/outputs/production_review/regional_scale_metrics.csv", "round2/review/regional_scale_metrics.csv")
    sweep = [r for r in scales if r["tag"] in bm and r["centre"] == "trained" and r["subset"] == "full" and r["region"] == "sharp" and r["setting"] == "prior_sweep"]
    per_tag = {tag: [float(r["rho"]) for r in sweep if r["tag"] == tag] for tag in sorted(bm)}
    if any(not values for values in per_tag.values()):
        raise ValueError("Missing Burgers-MLP sampled curve")
    noncross = [tag for tag, values in per_tag.items() if min(values) > 1]
    mle = [r for r in scales if r["tag"] in bm and r["centre"] == "trained" and r["subset"] == "full" and r["setting"] == "conditional_noise_mle" and r["region"] in {"bulk", "sharp"}]
    lookup = {(r["tag"], r["region"]): r for r in mle}
    if len(mle) != 2 * len(bm) or len(lookup) != len(mle):
        raise ValueError("Expected one MLE row per Burgers-MLP checkpoint and region")
    under68 = [tag for tag in sorted(bm) if all(float(lookup[tag, reg]["coverage68"]) < .68 for reg in ("bulk", "sharp"))]
    under95 = [tag for tag in sorted(bm) if all(float(lookup[tag, reg]["coverage95"]) < .95 for reg in ("bulk", "sharp"))]

    paired = records.get("theta_metrics", "analysis/theta_centred/outputs/review/paired_centres.csv", "theta/paired_centres.csv")
    baseline = one_per_tag([r for r in paired if r["setting"] == "baseline" and r["centre"] == "theta_affine" and r["subset"] == "full" and r["calibration"] == "selected" and r["region"] == "sharp"], tags)
    baseline_deficit = [r["tag"] for r in baseline if float(r["rho"]) > 1]

    direct = records.get("direct_roots", "analysis/theta_centred/outputs/review/root_comparisons.csv", "theta/root_comparisons.csv")
    select = lambda reg: one_per_tag([r for r in direct if r["setting"] == "trained_centre_refined_root" and r["centre"] == "trained" and r["subset"] == "full" and r["calibration"] == "bulk_68_oracle" and r["region"] == reg], root_tags)
    sharp, bulk = select("sharp"), select("bulk")
    sharp_cov = [float(r["coverage68"]) for r in sharp]
    bulk_cov = [float(r["coverage68"]) for r in bulk]

    n5 = records.get("n5", "analysis/n5/outputs/n5_all17.csv", "n5/n5_all17.csv")
    blocked = one_per_tag([r for r in n5 if r["region"] == "sharp" and r["split"] == "time_blocks" and int(r["block_slices"]) == 10 and float(r["rcond"]) == 1e-10], tags)
    heldout = [r["tag"] for r in blocked if float(r["heldout_pooled"]) < .04]

    values = {
        "H01_admitted_cohort": len(tags),
        "H02_architectures": len({r["architecture"] for r in cohort}),
        "H03_burgers_mlp_seeds": len(bm),
        "H04_trained_roots": len(roots),
        "H05_root_precision_decades": math.log10(max(gammas) / min(gammas)),
        "H06_noncrossing_seeds": len(noncross),
        "H07_mle_both_regions_under68": len(under68),
        "H08_mle_both_regions_under95": len(under95),
        "H09_theta_baseline_deficits": len(baseline_deficit),
        "H10_bulk_oracle_target": float(root_protocol["oracle_target"]),
        "H11_root_sharp_coverage_min": min(sharp_cov),
        "H12_root_sharp_coverage_max": max(sharp_cov),
        "H14_heldout_below_threshold": len(heldout),
        "H15_heldout_cohort": len(blocked),
        "H16_bulk_root_coverage_min": min(bulk_cov),
        "H17_root_rho_max_defect": max(abs(float(r["rho"]) - 1) for r in sharp),
    }
    details = {"cohort_tags": tags, "burgers_mlp_tags": sorted(bm), "root_tags": sorted(root_tags), "root_precisions": gammas, "noncrossing_tags": noncross, "minimum_sampled_rho": {tag: min(v) for tag, v in per_tag.items()}, "mle_under68_tags": under68, "mle_under95_tags": under95, "theta_baseline_deficit_tags": baseline_deficit, "root_sharp_coverages": {r["tag"]: float(r["coverage68"]) for r in sharp}, "heldout_below_threshold_tags": heldout, "heldout_fractions": {r["tag"]: float(r["heldout_pooled"]) for r in blocked}}
    return values, details


def test(value, expected):
    op = expected["operator"]
    target = expected["value"]
    if op == "eq":
        return value == target
    if op == "gt":
        return value > target
    if op == "ge":
        return value >= target
    if op == "lt":
        return value < target
    if op == "format":
        return format(value * expected.get("scale", 1), expected["format"]) == target
    raise ValueError(f"Unknown registry operator {op}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--paper", type=Path, help="Manuscript sources, default <root>/paper")
    parser.add_argument("--registry", type=Path, help="Explicit claim registry, default <root>/verification/headline_claims.json")
    parser.add_argument("--reproduced", type=Path, help="Fresh saved-data reproduction output tree")
    parser.add_argument("--out", type=Path, default=Path("reproduced_headlines"))
    args = parser.parse_args()
    args.root = args.root.resolve()
    paper = (args.paper or args.root / "paper").resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    registry = read_json(args.registry or args.root / "verification/headline_claims.json")
    records = Records(args.root, args.reproduced.resolve() if args.reproduced else None)
    results = []
    try:
        values, details = calculate(records)
        for claim in registry["quantities"]:
            value = values[claim["id"]]
            paper_checks = []
            for occurrence in claim.get("occurrences", []):
                path = paper / occurrence["source"]
                text = path.read_text(encoding="utf-8-sig")
                paper_checks.append({"source": occurrence["source"], "location": occurrence["location"], "pattern": occurrence["pattern"], "matches_current_paper": bool(re.search(occurrence["pattern"], text, flags=re.MULTILINE))})
            inputs = [records.used[k] for k in claim["records"]]
            numeric = test(value, claim["expected"])
            source_matches = all(p["matches_current_paper"] for p in paper_checks)
            fresh = [p["verification_level"] == "AGGREGATED_FRESH_RECOMPUTED_RECORDS" for p in inputs]
            level = "AGGREGATED_FRESH_RECOMPUTED_RECORDS" if all(fresh) else ("FRESH_RECOMPUTED_RECORDS_WITH_FROZEN_METADATA" if any(fresh) else "CHECKED_SAVED_SUMMARY")
            results.append({"id": claim["id"], "description": claim["description"], "computed_value": value, "expected": claim["expected"], "formula": claim["formula"], "status": "PASS" if numeric and source_matches else "FAIL", "numeric_check": numeric, "paper_check": source_matches, "verification_level": level, "records": inputs, "occurrences": paper_checks})
        status = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
        report = {"schema": "headline_claims/1", "status": status, "scope": registry["scope"], "quantities": results, "details": details, "source_records": records.used}
    except Exception as exc:
        status = "FAIL"
        report = {"schema": "headline_claims/1", "status": status, "error": f"{type(exc).__name__}: {exc}", "quantities": results, "source_records": records.used}
    (args.out / "headline_claims.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["id", "description", "computed_value", "expected", "status", "verification_level", "formula", "record_paths", "paper_locations"]
    with (args.out / "headline_claims.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fields if k in r}
            row.update(expected=json.dumps(r["expected"]), record_paths="; ".join(p["path"] for p in r["records"]), paper_locations="; ".join(p["source"] + ": " + p["location"] for p in r["occurrences"]))
            writer.writerow(row)
    print(f"Headline claim audit: {status}; {sum(r['status'] == 'PASS' for r in results)}/{len(registry['quantities'])} quantities pass.")
    if "error" in report:
        print(report["error"])
    for r in results:
        if r["status"] != "PASS":
            print(r["id"], "value=", r["computed_value"], "numeric=", r["numeric_check"], "paper=", r["paper_check"])
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
