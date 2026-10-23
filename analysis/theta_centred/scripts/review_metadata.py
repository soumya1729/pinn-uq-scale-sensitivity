"""Review downloaded theta-study completion, diagnostics and provenance.

This checks local outputs and downloaded archive manifests. The large retained
factor/grid arrays remain on the cluster, so their checksums are not re-read here.
No checkpoint differentiation, cluster access or production output changes occur.
"""
import csv
from collections import Counter
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/review"
PRODUCTION_SOURCES = ("theta_pass.py", "theta_linalg.py", "model_residual.py", "theta_archive.py")


@lru_cache(maxsize=None)
def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_finite(value, label):
    if isinstance(value, float):
        assert math.isfinite(value), f"Nonfinite metadata: {label}"
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{label}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{label}/{index}")


def main():
    manifest = read(ROOT/"manifest.json")
    tags = [record["tag"] for record in manifest["runs"]]
    assert len(tags) == len(set(tags)) == 17
    assert sorted(tags) == sorted(tag for group in manifest["groups"] for tag in group)
    for record in manifest["files"]:
        path = ROOT/record["path"]
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        assert digest(path) == record["sha256"], f"Changed frozen input: {path}"

    source_hashes = {name: digest(ROOT/"scripts"/name) for name in PRODUCTION_SOURCES}
    records, normal_count, archive_bytes = [], 0, 0
    maxima = Counter()
    version_sets = set()
    for tag in tags:
        stem = ROOT/"outputs/production"/f"{tag}_theta"
        for suffix in (".json", ".csv", "_fields.npz", "_shifts.npz"):
            assert Path(str(stem)+suffix).is_file(), f"Missing output: {stem}{suffix}"
        result = read(stem.with_suffix(".json"))
        assert result["tag"] == tag and result["numerical_status"] == "PASS"
        assert_finite(result, tag)
        assert result["signature"]["choices"] == result["choices"]
        assert result["signature"]["sources"] == source_hashes
        for name, expected in result["signature"]["inputs"].items():
            assert digest(ROOT/name) == expected, f"Changed run input: {tag}/{name}"
        version_sets.add(tuple(sorted(result["signature"]["versions"].items())))
        thresholds = {"baseline_max_sigma_relative_error": 1e-4,
            "gradient_norm_relative_defect": 1e-7, "affine_jvp_scaled_defect": 1e-7,
            "normal_equation_backward_error": 1e-7,
            "evidence_rebuild_regret": 1e-3, "old_variance_relative_defect": 1e-4,
            "old_zero_shift_scaled_defect": 1e-4, "archive_reload_defect": 1e-10}
        values = dict(baseline_max_sigma_relative_error=result["baseline_max_sigma_relative_error"],
            gradient_norm_relative_defect=result["grid_checks"]["max_gradient_norm_relative_defect"],
            affine_jvp_scaled_defect=result["grid_checks"]["affine_jvp_scaled_defect"],
            normal_equation_backward_error=max(r["normwise_backward_error"]
                for r in result["normal_equation_checks"]),
            evidence_rebuild_regret=max(r["rebuilt_log_evidence_regret"]
                for r in result["evidence_rebuild_checks"]),
            old_variance_relative_defect=max(r["unit_variance_relative_defect"]
                for r in result["archived_field_checks"]),
            old_zero_shift_scaled_defect=max(r["zero_affine_shift_scaled_defect"]
                for r in result["archived_field_checks"]))
        assert len(result["normal_equation_checks"]) == len(result["gammas"])
        normal_count += len(result["normal_equation_checks"])
        retained = result["archive_validation_report"]
        assert retained["checks_passed"] and retained["numerical_status"] == "PASS"
        path = ROOT/"cluster_records/retained_manifests"/tag/"manifest.json"
        assert digest(path) == retained["manifest_sha256"], f"Changed archive manifest: {tag}"
        archived = read(path)
        assert archived["tag"] == tag and archived["numerical_status"] == "PASS"
        assert archived["metadata"]["signature"] == result["signature"]
        assert archived["reload_checks"] == retained["reload_checks"]
        assert archived["parameter_count"] == result["parameter_count"]
        assert archived["residual_count"] == result["residual_count"] == 6400
        assert archived["grid_point_count"] == 80000
        assert len(archived["arrays"]) == 15
        assert archived["arrays"]["packed_qr.npy"]["order"] == "F"
        assert archived["arrays"]["grid_modal_tangents.npy"]["shape"] == [6400, 80000]
        assert archived["arrays"]["grid_modal_tangents.npy"]["order"] == "F"
        assert sum(r["file_bytes"] for r in archived["arrays"].values()) == retained["retained_bytes"]
        values["archive_reload_defect"] = max(value for name, value in archived["reload_checks"].items()
            if name.endswith("defect"))
        assert archived["reload_checks"]["tolerance"] == thresholds["archive_reload_defect"]
        for name, value in values.items():
            assert 0 <= value <= thresholds[name], f"Threshold failed: {tag}/{name}={value}"
            maxima[name] = max(maxima[name], value)
        archive_bytes += retained["retained_bytes"]
        records.append(dict(tag=tag, numerical_status="PASS", choices=len(result["choices"]),
            distinct_gammas=len(result["gammas"]), elapsed_hours=result["elapsed_seconds"]/3600,
            parameter_count=result["parameter_count"], archive_manifest_sha256=digest(path),
            retained_bytes=retained["retained_bytes"], diagnostic_maxima=values))

    assert len(version_sets) == 1
    task_records = []
    for name in ("471403_0", "471423_0", "471423_1", "471423_2"):
        folder = ROOT/"cluster_records"/name
        for source, expected in source_hashes.items():
            assert digest(folder/"scripts"/source) == expected, f"Executed source mismatch: {name}/{source}"
        assert digest(folder/"manifest.json") == digest(ROOT/"manifest.json")
        with (folder/"task_status.tsv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert [row["tag"] for row in rows] == (folder/"assigned_tags.txt").read_text().splitlines()
        assert all(row["exit_code"] == "0" for row in rows)
        for row in rows:
            row["record"] = name
            row["wall_seconds"] = (datetime.fromisoformat(row["finished"])
                -datetime.fromisoformat(row["started"])).total_seconds()
        task_records.extend(rows)
    counts = Counter(row["tag"] for row in task_records)
    assert set(counts) == set(tags)
    assert counts["burgers_mlp_seed2"] == 2 and all(counts[tag] == 1 for tag in tags if tag != "burgers_mlp_seed2")
    logs = list((ROOT/"logs").glob("*.out"))
    log_text = "\n".join(path.read_text(encoding="utf-8") for path in logs)
    completed = re.findall(r"^Completed (\S+);", log_text, re.M)
    assert sorted(completed) == sorted(tags)
    assert re.findall(r"^RESUMED (\S+);", log_text, re.M) == ["burgers_mlp_seed2"]
    assert not re.search(r"Traceback|REVIEW_REQUIRED|^Failed ", log_text, re.M)

    accounting_path = ROOT/"cluster_records/accounting-final-471403-471423.txt"
    accounting = []
    for line in accounting_path.read_text().splitlines():
        match = re.match(r"\s*(\d+_\d+(?:\.batch)?)\s+(\w+)\s+(\S+)\s+(?:(\d+)K\s+)?(\d+:\d+)\s*$", line)
        if match:
            job, state, elapsed, rss, exit_code = match.groups()
            assert state == "COMPLETED" and exit_code == "0:0"
            accounting.append(dict(job=job, state=state, elapsed=elapsed, max_rss_KiB=int(rss) if rss else None,
                exit_code=exit_code))
    assert {row["job"] for row in accounting} == {name+suffix for name in
        ("471403_0", "471423_0", "471423_1", "471423_2") for suffix in ("", ".batch")}
    report = dict(numerical_status="PASS", checkpoint_count=len(tags), output_quartets=17,
        frozen_input_files_checked=len(manifest["files"]), normal_equation_checks=normal_count,
        diagnostic_maxima=dict(maxima), thresholds=thresholds, production_source_hashes=source_hashes,
        versions=dict(next(iter(version_sets))), retained_bytes_recorded=archive_bytes,
        retained_decimal_GB=archive_bytes/1e9, task_records=task_records, accounting=accounting,
        pilot_reused=True, checkpoint_records=records,
        limitation="The 17 downloaded manifests match the summaries and record successful on-cluster reload checks. Large factor/grid arrays were not downloaded; their data/checksums have not been freshly verified locally. Consistency/backward-error checks do not certify unresolved weak-mode forward accuracy.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"numerical_audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    maximum_lines = "\n".join(f"| {name} | {value:.4g} | {thresholds[name]:.4g} |" for name, value in maxima.items())
    note = f"""# Downloaded metadata and provenance review

**PASS: all 17 checkpoint quartets and 17 retained-archive manifests agree.**

- {len(manifest['files'])} frozen input copies match their original manifest hashes; per-run input signatures also match local evidence/root inputs.
- Four production script hashes match all 17 summaries and all four executed script snapshots. JAX/NumPy/SciPy versions are {report['versions']}.
- All JSON metadata are finite; {normal_count} independent normal-equation checks and all recorded reconstruction/evidence/reload checks meet the stated thresholds.
- All four Slurm tasks completed with exit 0:0. Eighteen successful checkpoint invocations comprise 17 computed checkpoints plus the reused seed-2 pilot. No failed or REVIEW_REQUIRED log entries were found.
- Recorded retained data: {archive_bytes:,} bytes ({archive_bytes/1e9:.3f} decimal GB). Every downloaded manifest hash matches its summary, including the successful on-cluster reload check.

| Recorded diagnostic | Maximum | Threshold |
|---|---:|---:|
{maximum_lines}

The large factors and grid arrays remain on cluster storage. This local audit verifies their manifest provenance, not their current physical contents or checksums. Their recorded cluster reload checks validate layout/reconstruction consistency; they are not weak-mode forward-accuracy certificates. Scientific CSV/field comparisons are reviewed separately.

Reproduce this review with `.venv/Scripts/python.exe scripts/review_metadata.py` from this study directory. No production files are modified.
"""
    (OUT/"METADATA_REVIEW.md").write_text(note, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("numerical_status", "checkpoint_count",
        "frozen_input_files_checked", "normal_equation_checks", "diagnostic_maxima",
        "retained_decimal_GB", "pilot_reused")}, indent=2))


if __name__ == "__main__":
    main()
