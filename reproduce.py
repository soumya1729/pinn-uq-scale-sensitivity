"""Reviewer entry point for frozen-data reproduction, without training or Jacobians.

The calculation modules retain their original formulas. This wrapper gives them
fresh destinations and separates saved-summary checks from field recomputation.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

LEVELS = {
    "records": "CHECKED_SAVED_SUMMARY",
    "admission": "RECOMPUTED_METADATA",
    "scalar_bounds": "CHECKED_SAVED_SUMMARY",
    "data_integrity": "CHECKED_FILE_HASHES",
    "round2": "RECOMPUTED_FIELDS_AND_MODAL",
    "round3": "RECOMPUTED_FIELDS_AND_MODAL",
    "round4": "RECOMPUTED_FIELDS",
    "theta": "RECOMPUTED_FIELDS_AND_SAMPLED_FORWARD",
    "ensemble": "RECOMPUTED_FIELDS",
    "reporting_closure": "RECOMPUTED_FIELDS_AND_MODAL_WITH_FROZEN_CONTROL",
    "n5": "RECOMPUTED_CHECKPOINT_FEATURE_FITS",
    "tables": "GENERATED_FROM_RECOMPUTED_RESULTS",
    "headline_claims": "PER_CLAIM_LEVELS_REPORTED",
    "ledger_inventory": "PROVENANCE_INVENTORY_ONLY",
}
BASE_TASKS = ["records", "admission", "scalar_bounds"]
SAVED_TASKS = ["data_integrity", "round2", "round3", "round4", "theta", "ensemble", "reporting_closure", "n5", "tables"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required data/source files:\n" + "\n".join(missing[:30])
                                + (f"\n... {len(missing)} missing in total" if len(missing) > 30 else ""))


def load_module(path):
    path = Path(path)
    require([path])
    name = "reproduce_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def arguments(path, *args):
    old = sys.argv
    sys.argv = [str(path), *map(str, args)]
    try:
        yield
    finally:
        sys.argv = old


def invoke(path, *args):
    module = load_module(path)
    with arguments(path, *args):
        result = module.main()
    if isinstance(result, int) and result:
        raise RuntimeError(f"Producer returned failure code {result}: {path}")
    return module


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def compare_csv(actual, reference, *, ignored=(), rtol=2e-9, atol=2e-12):
    """Compare numeric/scientific fields, preserving producer row order."""
    require([actual, reference])
    a, b = rows(actual), rows(reference)
    if len(a) != len(b):
        raise AssertionError(f"Row-count mismatch: {actual.name}: {len(a)} != {len(b)}")
    compared = 0
    largest = 0.0
    for index, (left, right) in enumerate(zip(a, b)):
        if set(left) != set(right):
            raise AssertionError(f"Columns differ: {actual.name}")
        for key in left:
            if key in ignored:
                continue
            x, y = left[key], right[key]
            try:
                fx, fy = float(x), float(y)
            except (TypeError, ValueError):
                if x != y:
                    raise AssertionError(f"Record differs: {actual.name} row {index}, {key}: {x!r} != {y!r}")
                continue
            if not math.isfinite(fx) or not math.isfinite(fy):
                if fx != fy:
                    raise AssertionError(f"Nonfinite mismatch: {actual.name} row {index}, {key}")
                continue
            if not math.isclose(fx, fy, rel_tol=rtol, abs_tol=atol):
                raise AssertionError(f"Numeric mismatch: {actual.name} row {index}, {key}: {fx:.17g} != {fy:.17g}")
            compared += 1
            largest = max(largest, abs(fx-fy))
    return dict(file=actual.name, rows=len(a), numeric_fields_compared=compared,
                max_absolute_difference=largest, relative_tolerance=rtol, absolute_tolerance=atol)


def cohort(root):
    entries = read_json(root / "analysis/round2/cohort.json")["cohort"]
    tags = [entry["tag"] for entry in entries]
    if len(tags) != len(set(tags)) or len(tags) != 17:
        raise AssertionError("Expected the declared 17 unique checkpoints.")
    return tags


def compare_table_numbers(actual, reference):
    """Compare displayed numeric tokens in a generated/current tabular body."""
    require([actual, reference])
    def values(path):
        text = Path(path).read_text(encoding="utf-8-sig")
        text = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.S).group()
        return [float(token) for token in re.findall(r"(?<![A-Za-z])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)]
    a, b = values(actual), values(reference)
    if len(a) != len(b) or any(not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-14) for x, y in zip(a, b)):
        raise AssertionError(f"Displayed numeric table tokens differ: {actual.name}")
    return dict(generated=str(actual), paper=str(reference), displayed_numeric_tokens=len(a), match=True,
                scope="Current displayed table values, not independent validation of every printed prose occurrence.")


def execute(task, root, output):
    output.mkdir(parents=True, exist_ok=True)
    tags = cohort(root)
    if task == "records":
        checked, identities = 0, 0
        for subdir, suffix in (("analysis/round2/outputs/production", "round2"),
                               ("analysis/theta_centred/outputs/production", "theta")):
            for tag in tags:
                stem = root / subdir / f"{tag}_{suffix}"
                require([stem.with_suffix(".json"), stem.with_suffix(".csv")])
                summary = read_json(stem.with_suffix(".json"))
                if summary["tag"] != tag:
                    raise AssertionError(f"Wrong checkpoint identity: {stem}")
                for row in rows(stem.with_suffix(".csv")):
                    if all(key in row for key in ("R_tar", "R_sigma_rms", "rho")):
                        target, contrast, gap = [float(row[k]) for k in ("R_tar", "R_sigma_rms", "rho")]
                        if not math.isclose(target/contrast, gap, rel_tol=1e-11, abs_tol=1e-13):
                            raise AssertionError(f"Contrast identity failed: {stem}")
                        identities += 1
                    for key in ("coverage68", "coverage95"):
                        if key in row and not 0 <= float(row[key]) <= 1:
                            raise AssertionError(f"Invalid stored coverage: {stem}: {key}")
                    checked += 1
        return dict(checkpoints=17, production_record_sets=34, csv_rows=checked,
                    contrast_identities=identities,
                    scope="Checks stored records/identities; does not recompute errors or coverage from fields.")

    if task == "admission":
        path = root / "training/analysis/solve_gate_table.py"
        module = load_module(path)
        module.OUT = str(output)
        with arguments(path):
            result = module.main()
        if result:
            raise RuntimeError("Admission producer failed; see log.")
        actual = read_json(output / "solve_gate_table.json")
        reference = read_json(root / "training/calibration_figures/solve_gate_table.json")
        if actual != reference:
            raise AssertionError("Admission aggregation differs from the frozen reference record.")
        return dict(families=len(actual["families"]),
                    attempted=sum(v["attempted"] for v in actual["families"].values()),
                    reference_record_matches=True,
                    scope="Reaggregates recorded per-run admission metadata; does not rerun rejected solutions.")

    if task == "scalar_bounds":
        work = output / "work"
        (work / "analysis").mkdir(parents=True)
        (work / "generated").mkdir()
        shutil.copy2(root / "analysis/recompute_saved_summaries.py", work / "analysis")
        shutil.copytree(root / "analysis/input", work / "analysis/input")
        invoke(work / "analysis/recompute_saved_summaries.py")
        report = read_json(work / "analysis/saved_scalar_analysis.json")
        return dict(regional_enclosures=len(report["rows"]), decimal_precision=report["decimal_precision"],
                    scope="80-digit arithmetic from saved scalar summaries with internal enclosure and rounding assertions; not new loading/eigensolver calculations or a complete printed-value check.")

    if task == "data_integrity":
        inventory = read_json(root / "DATA_MANIFEST.json")
        require([root / record["path"] for record in inventory])
        total = 0
        for record in inventory:
            path = root / record["path"]
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise AssertionError(f"Data hash or size mismatch: {record['path']}")
            total += record["bytes"]
        return dict(files=len(inventory), bytes=total, all_sha256_match=True,
                    scope="Data identity, not numerical or scientific correctness.")

    production = root / "analysis/round2/outputs/production"
    if task == "round2":
        require([production / f"{tag}_round2{suffix}" for tag in tags
                 for suffix in (".json", ".csv", "_fields.npz", "_spectral.npz")])
        # Existing producer deliberately requires a not-yet-existing destination.
        review = output / "review"
        invoke(root / "analysis/round2/review_production.py", "--production", production,
               "--cohort", root / "analysis/round2/cohort.json", "--out", review)
        reference = root / "analysis/round2/outputs/production_review"
        names = ["regional_scale_metrics.csv", "uncertainty_contrast_curves.csv",
                 "sampled_oracle_coverage.csv", "nearest_sampled_agreement.csv",
                 "sampled_crossing_brackets.csv", "n3_centre_comparison.csv",
                 "qr_complement_summary.csv", "spectral_limits.csv", "evidence_best.csv",
                 "evidence_slope_groups.csv"]
        comparisons = [compare_csv(review / name, reference / name) for name in names]
        return dict(checkpoints=17, comparisons=comparisons,
                    scope="Recomputed field metrics/modal consistency and evidence-slope arithmetic. Perturbed optimum selections remain recorded producer diagnostics.")

    if task == "round3":
        invoke(root / "analysis/round3_local/review_round3.py", "--production-dir", production,
               "--cohort", root / "analysis/round2/cohort.json", "--out", output)
        names = ["parameter_decomposition.csv", "baseline_centres.csv", "theta_bounds.csv",
                 "oracle_distributions.csv", "refined_roots.csv", "strong_precision.csv", "common_threshold.csv"]
        comparisons = [compare_csv(output / name, root / "analysis/round3_local/outputs" / name) for name in names]
        return dict(comparisons=comparisons,
                    scope="Recomputed modal roots, bounds, strong-limit approach and field diagnostics. Historical consumer prose predates the completed direct theta study.")

    if task == "round4":
        invoke(root / "analysis/round4_local/review_round4.py", "--project-root", root, "--out", output)
        comparisons = [compare_csv(output / name, root / "analysis/round4_local/outputs" / name)
                       for name in ("nearest_saved_oracles.csv", "sampled_contrast_agreement_band.csv")]
        return dict(comparisons=comparisons,
                    scope="Recomputed regional oracle quantiles, distributions and saved rho-band metrics from fields.")

    if task == "theta":
        study = root / "analysis/theta_centred"
        require([study / "outputs/production" / f"{tag}_theta{suffix}" for tag in tags
                 for suffix in (".json", ".csv", "_fields.npz", "_shifts.npz")])
        path = study / "scripts/summarize_theta.py"
        module = load_module(path)
        module.ROOT, module.OUT = study, output
        with arguments(path):
            module.main()
        names = ["paired_centres.csv", "family_summary.csv", "root_comparisons.csv",
                 "displacement_summary.csv", "full_grid_errors.csv"]
        comparisons = [compare_csv(output / name, study / "outputs/review" / name) for name in names]
        report = read_json(output / "field_reproduction.json")
        return dict(checkpoints=report["checkpoints"], comparisons=comparisons,
                    scope="Recomputed all saved field metrics and direct root coverage; independent NumPy nonlinear forward checks at nine points per gamma, not a new full-grid forward/Jacobian pass.")

    if task == "ensemble":
        invoke(root / "analysis/round2/ensemble_loo.py", "--inputs", root / "analysis/n5/input",
               "--baseline", root / "analysis/round2/reference/ensemble_calibration_burgers_mlp.json", "--out", output)
        comparison = compare_csv(output / "ensemble_loo.csv", root / "analysis/round2/outputs/ensemble_loo.csv")
        return dict(comparison=comparison,
                    scope="Reconstructed ensemble mean, population spread and all eight leave-one-out subsets from member prediction fields.")

    if task == "reporting_closure":
        module = load_module(root / "analysis/round2/report_saved_summaries.py")
        review = output.parent / "round2/review"
        reference = root / "analysis/round2/outputs/reporting_closure"
        best = {(r["tag"], r["grid"]): r for r in rows(review / "evidence_best.csv")}
        shifts, shift_tex = module.shift_report(rows(review / "n3_centre_comparison.csv"))
        loadings, loading_tex = module.loading_report(production, best)
        baseline, control, baseline_tex = module.baseline_report(
            rows(review / "regional_scale_metrics.csv"),
            read_json(reference / "source_control_record.json"))
        comparisons = []
        for name, values in (("regional_relative_shift.csv", shifts),
                             ("changed_scale_loadings.csv", loadings),
                             ("baseline_per_run.csv", baseline),
                             ("control_per_setting.csv", control)):
            module.write_csv(output / name, values)
            comparisons.append(compare_csv(output / name, reference / name))
        for name, tex in (("round2_relative_shift_table.tex", shift_tex),
                          ("round2_scale_loadings_table.tex", loading_tex),
                          ("round2_small_sample_ranges_table.tex", baseline_tex)):
            (output / name).write_text(tex, encoding="utf-8")
        return dict(comparisons=comparisons,
                    scope="Regional displacement ratios, scale-dependent loadings and nonlinear baseline ranges regenerated from fresh review records/saved spectra; the separate linear-control extract remains frozen metadata.")

    if task == "n5":
        invoke(root / "analysis/n5/n5_heldout.py", "--results", root / "analysis/n5/input",
               "--out", output / "n5_all17.csv")
        comparison = compare_csv(output / "n5_all17.csv", root / "analysis/n5/outputs/n5_all17.csv")
        return dict(comparison=comparison, checkpoints=17,
                    scope="Rebuilt the trained 129 last-layer features and predictions, and refitted all regional same-grid/random/time-block regressions at the published split seed and rank cutoffs. No PINN training or Jacobian.")

    if task == "tables":
        review = output.parent / "round2/review"
        invoke(root / "analysis/round2/generate_tables.py", "--production", production,
               "--review", review, "--out", output / "round2")
        # Current paired-centre/root tables must use the completed theta study,
        # not the older zero-mean-only centre formatter above.
        virtual = output / "theta_work"
        (virtual / "outputs/review").mkdir(parents=True)
        (virtual / "inputs").mkdir()
        for name in ("full_grid_errors.csv", "root_comparisons.csv"):
            shutil.copy2(output.parent / "theta" / name, virtual / "outputs/review" / name)
        shutil.copy2(root / "analysis/theta_centred/inputs/refined_roots.csv", virtual / "inputs/refined_roots.csv")
        path = root / "analysis/theta_centred/scripts/make_paper_tables.py"
        module = load_module(path)
        module.ROOT = virtual
        with arguments(path, "--out", output / "theta_current"):
            module.main()
        paper_comparisons = []
        for name in ("round2_centre_table.tex", "round3_math_root_table.tex"):
            candidates = [root / "paper/generated" / name, root / "paper" / name]
            reference = next((p for p in candidates if p.is_file()), None)
            if reference is not None:
                paper_comparisons.append(compare_table_numbers(output / "theta_current" / name, reference))
        return dict(generated_files=[str(p.relative_to(output)) for p in output.rglob("*.tex")],
                    current_manuscript_table_comparisons=paper_comparisons,
                    manuscript_comparison_status="PASS" if len(paper_comparisons) == 2 else "NOT_AVAILABLE",
                    scope="Regenerated numerical table records. Current manuscript retains deliberate manual merges, display rounding and captions; byte-identical TeX is not asserted.")
    if task == "ledger_inventory":
        invoke(root / 'tools/build_numerical_ledger.py', '--paper-root', root / 'paper',
               '--output-map', root / 'docs/PAPER_OUTPUT_MAP.md', '--out', output,
               '--reproduced', output.parent)
        return dict(inventory=read_json(output / 'summary.json'),
                    scope='Occurrence mapping with explicit registered claim joins; not independent verification of every numeral.')
    if task == "headline_claims":
        invoke(root / "tools/check_headline_claims.py", "--root", root,
               "--reproduced", output.parent, "--out", output)
        report = read_json(output / "headline_claims.json")
        return dict(report=report,
                    scope="Explicit abstract/contribution claims at the individual verification levels in the headline report; saved mode uses the freshly refitted N5 records, while records mode labels its saved-summary fallback.")
    raise ValueError(task)


def worker(task, root, output):
    started = time.monotonic()
    record = dict(task=task, verification_level=LEVELS[task])
    try:
        record["details"] = execute(task, root, output)
        record["status"] = "PASS"
    except FileNotFoundError as exc:
        record.update(status="MISSING_DATA", error=str(exc))
        traceback.print_exc()
    except Exception as exc:
        record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    record["elapsed_seconds"] = time.monotonic()-started
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "task_result.json", record)
    return 0 if record["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Repository root (defaults to this script's directory).")
    parser.add_argument("--out", type=Path, default=Path("reproduced"), help="Fresh output directory; must not already exist.")
    parser.add_argument("--mode", choices=("records", "saved"), default="saved",
                        help="records checks saved summaries; saved additionally recomputes fields/modal metrics after restoring data.")
    parser.add_argument("--worker", choices=list(LEVELS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    if args.worker:
        return worker(args.worker, root, output)
    require([root / "analysis/round2/cohort.json"])
    for protected in (root / "analysis", root / "training", root / "paper_snapshot", root / "provenance"):
        if output == protected or protected in output.parents:
            parser.error("Use a fresh output directory outside the frozen scientific/input folders.")
    if output == root or output in root.parents:
        parser.error("Output cannot be the repository root or one of its ancestors.")
    if output.exists():
        parser.error(f"Output already exists; choose a fresh directory: {output}")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    task_names = BASE_TASKS + (SAVED_TASKS if args.mode == "saved" else [])
    if (root / "tools/check_headline_claims.py").is_file():
        task_names.append("headline_claims")
    if (root / 'tools/build_numerical_ledger.py').is_file():
        task_names.append('ledger_inventory')
    tasks = []
    for name in task_names:
        destination = output / name
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                   "--out", str(destination), "--worker", name]
        print(f"{name}: running ({LEVELS[name]})", flush=True)
        completed = subprocess.run(command, cwd=root, env=environment, text=True, encoding="utf-8", errors="replace",
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
        result_path = destination / "task_result.json"
        record = read_json(result_path) if result_path.is_file() else dict(task=name, status="FAIL", verification_level=LEVELS[name], error="Worker terminated before writing a result; see log.")
        record["exit_code"] = completed.returncode
        tasks.append(record)
        print(f"{name}: {record['status']}", flush=True)
        if name == "data_integrity" and record["status"] != "PASS":
            for pending in SAVED_TASKS[1:]:
                tasks.append(dict(task=pending, status="NOT_RUN", verification_level=LEVELS[pending],
                                  reason="Restore/verify the missing or mismatched data first."))
            break
    passed = all(task["status"] == "PASS" for task in tasks)
    report = dict(schema="paper_reproduction/1", mode=args.mode, status="PASS" if passed else "INCOMPLETE_OR_FAILED",
                  root=str(root), output=str(output), python=sys.version, tasks=tasks,
                  package_versions={name: importlib.metadata.version(name) for name in
                                    ("numpy", "scipy", "matplotlib", "pypdf", "openpyxl")
                                    if importlib.util.find_spec(name) is not None},
                  scope="Frozen-data consistency/recomputation at the levels shown per task. Not every manuscript occurrence is independently recalculated; consult the numerical ledger for occurrence coverage.",
                  not_performed=["new PINN training", "new residual Jacobian or covariance factorisation", "cluster jobs", "Lean compilation",
                                 "full-grid independent nonlinear reevaluation", "independent PDE reference solve",
                                 "full historical loading reconstruction", "all toy reruns"])
    if args.mode == "records":
        report["not_performed"].append("N5 checkpoint feature fits")
    save_json(output / "reproduction_report.json", report)
    note = ["# Reproduction run", "", f"Mode: **{args.mode}**. Result: **{report['status']}**.", "",
            "| Task | Status | Verification level |", "|---|---|---|"]
    note.extend(f"| {task['task']} | {task['status']} | {task['verification_level']} |" for task in tasks)
    note += ["", report["scope"], "", "No frozen input/reference result was deliberately overwritten.", "",
             "## Outside this command", "", *[f"- {item}" for item in report["not_performed"]], ""]
    (output / "REPORT.md").write_text("\n".join(note), encoding="utf-8")
    print(f"Report: {output / 'reproduction_report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
