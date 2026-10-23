#!/usr/bin/env python3
"""Bind every admitted frozen full-grid posterior-scale field to the raw metric.

The historical NPZ and full-Laplace JSON files predate explicit ``sigma_metric``
metadata, so this checker does not infer the metric from a filename.  For every
admitted run it recomputes the regional mean-width contrast from the frozen field,
requires agreement with both the full-Laplace record and the independently stored
``raw`` leg of the dual-metric producer, and requires disagreement with that
producer's ``whitened`` leg.  It also binds the exact reference/error field,
requires identical evaluation coordinates, reconstructs Definition 4's regions
from ``grad_abs``, and reproduces the target ratio and absolute calibration
columns.  Eleven fields have the stronger, current-J Woodbury reproduction in
``loading_perpoint/3`` as additional evidence.

The large historical arrays are never rewritten. ``--check`` is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = "fullgrid_sigma_metric_contract/3"
TAGS = (
    *(f"burgers_mlp_seed{i}" for i in range(8)),
    "burgers_fourier_seed0", "burgers_fourier_seed2", "burgers_fourier_seed6",
    "allen_cahn_mlp_seed0", "allen_cahn_mlp_seed1", "allen_cahn_mlp_seed3",
    "allen_cahn_fourier_seed0", "allen_cahn_fourier_seed1",
    "allen_cahn_fourier_seed3",
)


def digest(path: Path, name: str) -> str:
    h = hashlib.new(name)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def build(root: Path) -> dict:
    files = []
    n_woodbury = 0
    for tag in TAGS:
        run_dir = root / "results" / tag
        field = run_dir / f"{tag}_fullgrid_sigma_raw.npz"
        ratio_path = run_dir / f"{tag}_fullgrid_ratio.json"
        metric_path = run_dir / f"{tag}_full_laplace_metrics.json"
        reference = run_dir / f"{tag}_fields.npz"
        loading = run_dir / f"{tag}_loading_perpoint.json"
        require(field.is_file(), f"missing frozen field: {field}")
        require(reference.is_file(), f"missing reference/error field: {reference}")
        ratio = read_json(ratio_path)
        metric = read_json(metric_path)
        require(metric.get("tag") == tag and metric.get("scope") == "full",
                f"{tag}: full-Laplace tag/scope drift")
        require(metric.get("solve_gate", {}).get("passed") is True,
                f"{tag}: full-Laplace record is not admitted")

        try:
            with np.load(field) as z, np.load(reference) as ref:
                required = {"sigma", "xs", "ts", "shock_idx", "smooth_idx"}
                require(required.issubset(z.files),
                        f"{tag}: field lacks {sorted(required - set(z.files))}")
                reference_required = {"xs", "ts", "grad_abs", "u_pred", "u_exact"}
                require(reference_required.issubset(ref.files),
                        f"{tag}: reference field lacks "
                        f"{sorted(reference_required - set(ref.files))}")
                sigma = np.asarray(z["sigma"], dtype=float).reshape(-1)
                xs = np.asarray(z["xs"], dtype=float)
                ts = np.asarray(z["ts"], dtype=float)
                sharp = np.asarray(z["shock_idx"], dtype=np.int64).reshape(-1)
                bulk = np.asarray(z["smooth_idx"], dtype=np.int64).reshape(-1)
                ref_xs = np.asarray(ref["xs"], dtype=float)
                ref_ts = np.asarray(ref["ts"], dtype=float)
                grad_abs = np.asarray(ref["grad_abs"], dtype=float).reshape(-1)
                error = np.abs(np.asarray(ref["u_pred"], dtype=float).reshape(-1) -
                               np.asarray(ref["u_exact"], dtype=float).reshape(-1))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read {field}: {exc}") from exc
        require(sigma.size == 80_000 and np.all(np.isfinite(sigma)) and
                np.all(sigma > 0), f"{tag}: invalid posterior-scale field")
        require(xs.shape == ref_xs.shape == (400,) and ts.shape == ref_ts.shape == (200,),
                f"{tag}: unexpected evaluation-coordinate shapes")
        require(np.array_equal(xs, ref_xs) and np.array_equal(ts, ref_ts),
                f"{tag}: posterior-scale and reference grids do not match exactly")
        require(grad_abs.size == error.size == sigma.size and
                np.all(np.isfinite(grad_abs)) and np.all(grad_abs >= 0) and
                np.all(np.isfinite(error)), f"{tag}: invalid gradient/error field")
        require(sharp.size == 4_000 and bulk.size == 56_000,
                f"{tag}: unexpected regional index counts")
        require(np.all((0 <= sharp) & (sharp < sigma.size)) and
                np.all((0 <= bulk) & (bulk < sigma.size)),
                f"{tag}: regional index outside the field")
        require(np.unique(sharp).size == sharp.size and np.unique(bulk).size == bulk.size,
                f"{tag}: duplicate regional indices")
        require(np.intersect1d(sharp, bulk).size == 0,
                f"{tag}: sharp and bulk indices overlap")

        p95 = float(np.percentile(grad_abs, 95.0))
        p70 = float(np.percentile(grad_abs, 70.0))
        require(np.count_nonzero(grad_abs == p95) == 0 and
                np.count_nonzero(grad_abs == p70) == 0,
                f"{tag}: percentile boundary ties make endpoint conventions differ")
        expected_sharp = np.flatnonzero(grad_abs >= p95)
        expected_bulk = np.flatnonzero(grad_abs < p70)
        require(np.array_equal(sharp, expected_sharp) and
                np.array_equal(bulk, expected_bulk),
                f"{tag}: stored regions do not equal Definition 4 on the reference field")

        field_ratio = float(sigma[sharp].mean() / sigma[bulk].mean())
        recorded_ratio = float(metric.get("ratios", {}).get("achieved_sigma_ratio", np.nan))
        require(np.isfinite(recorded_ratio) and
                abs(field_ratio - recorded_ratio) <= 1e-10 * max(1.0, abs(field_ratio)),
                f"{tag}: field ratio {field_ratio:.12g} disagrees with full-Laplace record "
                f"{recorded_ratio:.12g}")
        raw_ratio = float(ratio.get("raw", {}).get("ratio_mean", np.nan))
        white_ratio = float(ratio.get("whitened", {}).get("ratio_mean", np.nan))
        require(np.isfinite(raw_ratio) and np.isfinite(white_ratio),
                f"{tag}: dual-metric ratio legs are absent")
        require(abs(round(field_ratio, 2) - raw_ratio) <= 5e-3,
                f"{tag}: frozen field does not match the stored raw leg")
        require(abs(field_ratio - white_ratio) > 5e-2 * max(1.0, abs(field_ratio)),
                f"{tag}: raw/whitened discrimination is too weak")

        target_ratio = float(np.sqrt(np.mean(error[sharp] ** 2)) /
                             np.sqrt(np.mean(error[bulk] ** 2)))
        recorded_target = float(metric.get("ratios", {}).get("target_rmse_ratio", np.nan))
        require(np.isfinite(target_ratio) and np.isfinite(recorded_target) and
                abs(target_ratio - recorded_target) <=
                1e-10 * max(1.0, abs(target_ratio)),
                f"{tag}: target-error ratio disagrees with the bound reference field")
        expected_absolute = {
            "shock": {
                "mean_z": float(np.mean(error[sharp] / sigma[sharp])),
                "coverage68": float(np.mean(error[sharp] <= sigma[sharp])),
            },
            "smooth": {
                "mean_z": float(np.mean(error[bulk] / sigma[bulk])),
                "coverage68": float(np.mean(error[bulk] <= sigma[bulk])),
            },
        }
        for region, values in expected_absolute.items():
            stored = metric.get("regions", {}).get(region, {})
            for name, value in values.items():
                recorded = float(stored.get(name, np.nan))
                require(np.isfinite(recorded) and
                        abs(value - recorded) <= 1e-10 * max(1.0, abs(value)),
                        f"{tag}: {region} {name} disagrees with the bound fields")

        evidence = "dual_metric_ratio_and_full_laplace_agreement"
        loading_rel = None
        loading_sha = None
        woodbury_max_rel = None
        if loading.is_file():
            d = read_json(loading)
            require(d.get("schema") == "loading_perpoint/3" and
                    d.get("sigma_metric") == "raw",
                    f"{tag}: loading record is not schema-v3 raw")
            require(d.get("tag") == tag and d.get("scope") == "full",
                    f"{tag}: loading scope drift")
            require(d.get("hashes", {}).get("sigma_raw") == digest(field, "sha1"),
                    f"{tag}: loading record does not bind the frozen field")
            sample = d.get("numerics", {}).get("woodbury_sample", {})
            require(sample and sample.get("n_sampled", sample.get("n_points", 0)) > 0,
                    f"{tag}: independent raw-Woodbury sample is absent")
            woodbury_max_rel = float(sample.get(
                "max_rel_err", sample.get("max_rel",
                                           sample.get("max_relative_error", np.inf))))
            require(woodbury_max_rel <= 1e-8,
                    f"{tag}: raw-Woodbury reproduction is too loose "
                    f"({woodbury_max_rel:.3e})")
            sandwich = d.get("sandwich", {})
            require(sandwich and all(v.get("inside") is True for v in sandwich.values()),
                    f"{tag}: stored-field regional sandwich failed")
            evidence += "+current_J_Woodbury_and_regional_sandwich"
            loading_rel = str(loading.relative_to(root)).replace("\\", "/")
            loading_sha = digest(loading, "sha256")
            n_woodbury += 1

        files.append({
            "tag": tag,
            "path": str(field.relative_to(root)).replace("\\", "/"),
            "historical_schema": "checksum_bound_npz",
            "sigma_metric": "raw",
            "scope": "full",
            "sha1": digest(field, "sha1"),
            "sha256": digest(field, "sha256"),
            "field_mean_width_ratio": field_ratio,
            "full_laplace_record": str(metric_path.relative_to(root)).replace("\\", "/"),
            "full_laplace_record_sha256": digest(metric_path, "sha256"),
            "reference_field": str(reference.relative_to(root)).replace("\\", "/"),
            "reference_field_sha256": digest(reference, "sha256"),
            "region_definition": "sharp: grad_abs >= p95; bulk: grad_abs < p70",
            "p95": p95,
            "p70": p70,
            "target_rmse_ratio": target_ratio,
            "dual_metric_record": str(ratio_path.relative_to(root)).replace("\\", "/"),
            "dual_metric_record_sha256": digest(ratio_path, "sha256"),
            "stored_raw_ratio_2dp": raw_ratio,
            "stored_whitened_ratio_2dp": white_ratio,
            "evidence": evidence,
            "loading_record": loading_rel,
            "loading_record_sha256": loading_sha,
            "woodbury_sample_max_relative_error": woodbury_max_rel,
        })

    require(n_woodbury == 11,
            f"expected 11 current-J Woodbury reproductions, found {n_woodbury}")
    return {
        "schema": SCHEMA,
        "producer": "analysis/fullgrid_metric_contract.py",
        "producer_sha256": digest(Path(__file__).resolve(), "sha256"),
        "sigma_metric": "raw",
        "field_schema_for_future_outputs": "fullgrid_sigma/1",
        "admitted_field_count": len(files),
        "current_J_Woodbury_count": n_woodbury,
        "evidence": (
            "All admitted historical fields share exact coordinates with their hash-bound "
            "reference/error fields, reproduce Definition 4's finite-grid regions, target "
            "ratio, and absolute calibration columns, and reproduce their full-Laplace "
            "mean-width value "
            "to numerical precision, match the explicitly stored raw leg of a dual-metric "
            "calculation after its documented two-decimal rounding, and are separated from "
            "the whitened leg. Eleven additionally have a current raw-J Woodbury reproduction "
            "and regional sandwich. The contract labels and hashes, but does not rewrite, the "
            "historical NPZ files."
        ),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path,
                        help="optional record path (useful for a read-only data root)")
    args = parser.parse_args()
    root = args.root.resolve()
    record = build(root)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = (args.output.resolve() if args.output else
              root / "calibration_figures" / "fullgrid_sigma_metric_contract.json")
    if args.check:
        require(target.is_file(), f"missing {target}")
        require(target.read_text(encoding="utf-8") == text, f"{target} is stale")
        print("FULL-GRID METRIC CONTRACT CHECK: PASS")
    else:
        atomic_text(target, text)
        print(f"wrote {target}")
    print(f"verified {len(record['files'])} admitted raw fields "
          f"({record['current_J_Woodbury_count']} with current-J Woodbury evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
