#!/usr/bin/env python3
"""Recompute the all-admitted-run off-manifold summary from the 17 released artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


SCHEMA = "offmanifold_summary/3"
HISTORICAL_PRODUCER_SHA256 = "4964dff75410cee27f6243bc6abbd18727c637c14967531a732489bcf9a642bb"
RELEASE_PRODUCER_SHA256 = "0b09b604ad69b3b6dc8a8a37239b1ae30637c5e406f52c55c98d4443d6119aa5"
SEMANTIC_AST_SHA256 = "85790e54f233473ebcf3bb7a1919edded3fe6b4f17b7b850123d1de69893b672"
TAGS = (
    *(f"burgers_mlp_seed{i}" for i in range(8)),
    "burgers_fourier_seed0", "burgers_fourier_seed2", "burgers_fourier_seed6",
    "allen_cahn_mlp_seed0", "allen_cahn_mlp_seed1", "allen_cahn_mlp_seed3",
    "allen_cahn_fourier_seed0", "allen_cahn_fourier_seed1", "allen_cahn_fourier_seed3",
)
CANON = (
    "burgers_mlp_seed2", "burgers_fourier_seed2",
    "allen_cahn_mlp_seed0", "allen_cahn_fourier_seed0",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class _StripNonScientificText(ast.NodeTransformer):
    @staticmethod
    def _strip(node):
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Expr(self, node):
        # Reporting-only print strings may be corrected without pretending that historical
        # numerical artifacts were rerun. All executable expressions other than print remain.
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                node.value.func.id == "print"):
            return None
        return self.generic_visit(node)


def semantic_ast_sha256(path: Path) -> str:
    tree = _StripNonScientificText().visit(ast.parse(path.read_text(encoding="utf-8")))
    # Python 3.13 added ``show_empty`` to ast.dump and changed its default output
    # by omitting empty optional fields.  The AST itself is unchanged, but hashing
    # that version-dependent display produced a false source-contract failure when
    # the reviewed Python-3.12 hash was checked under Python 3.13.  Request the
    # pre-3.13 representation explicitly where supported.
    try:
        dumped = ast.dump(tree, include_attributes=False, show_empty=True)
    except TypeError:  # Python <= 3.12 always showed these fields.
        dumped = ast.dump(tree, include_attributes=False)
    payload = dumped.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def build(root: Path) -> dict:
    producer = root / "d4_manifold.py"
    require(sha256(producer) == RELEASE_PRODUCER_SHA256,
            "d4_manifold.py differs from the reviewed documentation/reporting-only release source")
    require(semantic_ast_sha256(producer) == SEMANTIC_AST_SHA256,
            "d4_manifold.py executable AST differs from the historical artifact producer")
    rows, hashes = {}, {}
    for tag in TAGS:
        path = root / "results" / tag / f"{tag}_manifold_metrics.json"
        require(path.is_file(), f"missing admitted-run artifact: {path}")
        d = json.loads(path.read_text(encoding="utf-8"))
        require(d.get("tag") == tag, f"{tag}: declared tag mismatch")
        map_path = root / "results" / tag / f"{tag}_params.pkl"
        require(map_path.is_file(), f"{tag}: required MAP is missing: {map_path}")
        require(d.get("map_sha1") == sha1(map_path),
                f"{tag}: manifold artifact is not bound to the current MAP")
        require(d.get("scope") == "last_layer" and d.get("feature_span_dim") == 129,
                f"{tag}: wrong feature scope")
        seed = int(tag.rsplit("seed", 1)[1])
        arch = "fourier" if "fourier" in tag else "mlp"
        pde = "allen_cahn" if tag.startswith("allen_cahn") else "burgers"
        cfg = d.get("provenance", {}).get("config", {})
        expected = {
            "pde": pde, "arch": arch, "seed": seed,
            "nu": 0.01 / math.pi, "eps2": 1e-4, "reac": 5.0,
            "fourier_scale": 1.0 if pde == "allen_cahn" else 2.0,
            "fourier_dim": 16, "Mx": 20, "Mt": 20, "hidden": 64,
            "fusion": 128, "d": 128, "d_aug": 129,
            "N_f": 5000, "N_u": 1000, "N_b": 200,
            "eval_nx": 400, "eval_nt": 200, "smoke": 0,
        }
        require(all(cfg.get(key) == value for key, value in expected.items()),
                f"{tag}: artifact configuration does not match the declared experiment")
        map_record_path = root / "results" / tag / f"{tag}_map.json"
        require(map_record_path.is_file(), f"{tag}: admitted-run metadata is missing")
        map_record = json.loads(map_record_path.read_text(encoding="utf-8"))
        require(map_record.get("tag") == tag and
                map_record.get("solve_gate", {}).get("passed") is True,
                f"{tag}: off-manifold row is not bound to an admitted run")
        require(map_record.get("map_sha1") == d.get("map_sha1"),
                f"{tag}: admission record and off-manifold artifact name different MAPs")
        primary_path = root / "results" / tag / f"{tag}_fields.npz"
        field_path = root / "results" / tag / f"{tag}_manifold_fields.npz"
        require(primary_path.is_file() and field_path.is_file(),
                f"{tag}: primary/manifold field artifact is missing")
        try:
            with np.load(primary_path) as primary, np.load(field_path, allow_pickle=True) as field:
                primary_required = {"xs", "ts", "u_pred", "u_exact", "grad_abs"}
                manifold_required = {"xs", "ts", "e", "e_tangent", "e_offmanifold",
                                     "regions"}
                require(primary_required.issubset(primary.files),
                        f"{tag}: primary field lacks required arrays")
                require(manifold_required.issubset(field.files),
                        f"{tag}: manifold field lacks required arrays")
                primary_xs, primary_ts = np.asarray(primary["xs"]), np.asarray(primary["ts"])
                field_xs, field_ts = np.asarray(field["xs"]), np.asarray(field["ts"])
                error = (np.asarray(primary["u_pred"], dtype=float).reshape(-1) -
                         np.asarray(primary["u_exact"], dtype=float).reshape(-1))
                grad_abs = np.asarray(primary["grad_abs"], dtype=float).reshape(-1)
                stored_error = np.asarray(field["e"], dtype=float).reshape(-1)
                tangent = np.asarray(field["e_tangent"], dtype=float).reshape(-1)
                residual = np.asarray(field["e_offmanifold"], dtype=float).reshape(-1)
                region_labels = np.asarray(field["regions"], dtype=object).reshape(-1)
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"{tag}: cannot read primary/manifold fields: {exc}") from exc
        require(primary_xs.shape == field_xs.shape == (400,) and
                primary_ts.shape == field_ts.shape == (200,) and
                np.array_equal(primary_xs, field_xs) and
                np.array_equal(primary_ts, field_ts),
                f"{tag}: primary/manifold evaluation grids differ")
        require(all(array.size == 80000 for array in
                    (error, grad_abs, stored_error, tangent, residual, region_labels)) and
                all(np.all(np.isfinite(array)) for array in
                    (error, grad_abs, stored_error, tangent, residual)) and
                np.all(grad_abs >= 0), f"{tag}: invalid or unaligned manifold arrays")
        scale = max(float(np.linalg.norm(error)), np.finfo(float).tiny)
        require(np.linalg.norm(stored_error - error) <= 1e-12 * scale and
                np.linalg.norm(stored_error - tangent - residual) <= 1e-12 * scale,
                f"{tag}: stored error decomposition disagrees with the primary field")
        orthogonality = abs(float(tangent @ residual)) / max(
            float(np.linalg.norm(tangent) * np.linalg.norm(residual)), np.finfo(float).tiny)
        require(orthogonality <= 1e-8,
                f"{tag}: global least-squares components are not orthogonal")
        p70, p95 = np.percentile(grad_abs, (70.0, 95.0))
        require(np.count_nonzero(grad_abs == p70) == 0 and
                np.count_nonzero(grad_abs == p95) == 0,
                f"{tag}: percentile endpoint conventions differ on this grid")
        expected_labels = np.full(80000, "transition", dtype=object)
        expected_labels[grad_abs < p70] = "smooth"
        expected_labels[grad_abs >= p95] = "shock"
        require(np.array_equal(region_labels, expected_labels),
                f"{tag}: manifold regions disagree with Definition 4")
        global_fraction = float(residual @ residual / (stored_error @ stored_error + 1e-30))
        require(math.isclose(float(d.get("global_offmanifold_frac", math.nan)),
                             global_fraction, rel_tol=1e-10, abs_tol=1e-12),
                f"{tag}: global off-manifold fraction disagrees with its field")
        regions = d.get("regions", {})
        vals = {}
        total = 0
        for region in ("smooth", "transition", "shock"):
            require(region in regions, f"{tag}: missing {region}")
            value = float(regions[region]["offmanifold_frac_local"])
            count = int(regions[region]["count"])
            require(math.isfinite(value) and 0 <= value <= 1, f"{tag}: invalid {region} fraction")
            require(count > 0, f"{tag}: empty {region}")
            mask = expected_labels == region
            rmse = float(np.sqrt(np.mean(stored_error[mask] ** 2)))
            global_rmse = float(np.sqrt(np.mean(residual[mask] ** 2)))
            local_rmse = float(regions[region].get("offmanifold_rmse_local", math.nan))
            require(count == int(mask.sum()) and math.isfinite(local_rmse) and local_rmse >= 0 and
                    math.isclose(float(regions[region].get("rmse", math.nan)), rmse,
                                 rel_tol=1e-10, abs_tol=1e-15) and
                    math.isclose(float(regions[region].get("offmanifold_rmse_global", math.nan)),
                                 global_rmse, rel_tol=1e-10, abs_tol=1e-15) and
                    math.isclose(value, (local_rmse / rmse) ** 2,
                                 rel_tol=1e-10, abs_tol=1e-12),
                    f"{tag}: {region} summary disagrees with the stored decomposition")
            vals[region] = value
            total += count
        require(total == 80000, f"{tag}: expected 80000 evaluation points, got {total}")
        require(int(regions["smooth"]["count"]) == 56000 and
                int(regions["transition"]["count"]) == 20000 and
                int(regions["shock"]["count"]) == 4000,
                f"{tag}: finite-grid percentile regions do not have the declared 70/25/5% sizes")
        require(vals["shock"] > 0, f"{tag}: sharp fraction must be positive for the finite ratio claim")
        require(vals["shock"] <= vals["smooth"] and vals["shock"] <= vals["transition"],
                f"{tag}: sharp fraction is not the smallest")
        rows[tag] = {
            **vals,
            "bulk_over_sharp": vals["smooth"] / vals["shock"],
            "captured_sharp_energy_fraction": 1.0 - vals["shock"],
        }
        hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
        hashes[str(primary_path.relative_to(root)).replace("\\", "/")] = sha256(primary_path)
        hashes[str(field_path.relative_to(root)).replace("\\", "/")] = sha256(field_path)

    worst_tag = max(TAGS, key=lambda tag: rows[tag]["shock"])
    ratios = [rows[tag]["bulk_over_sharp"] for tag in TAGS]
    return {
        "schema": SCHEMA,
        "definition": "regionwise least-squares residual energy / regional error energy",
        "evaluation_points_per_run": 80000,
        "feature_scope": "last_layer",
        "feature_span_dim": 129,
        "validation": (
            "exact primary/manifold grids and regions; error decomposition; global projection "
            "orthogonality; regional RMSE and local-fraction identities"
        ),
        "admitted_tags": list(TAGS),
        "n_admitted": len(TAGS),
        "sharp_is_smallest_in_every_run": True,
        "worst_sharp": {"tag": worst_tag, "fraction": rows[worst_tag]["shock"],
                        "captured_energy_fraction": 1.0 - rows[worst_tag]["shock"]},
        "bulk_over_sharp": {"min": min(ratios), "max": max(ratios)},
        "canonical_bulk_over_sharp": {
            "min": min(rows[tag]["bulk_over_sharp"] for tag in CANON),
            "max": max(rows[tag]["bulk_over_sharp"] for tag in CANON),
        },
        "canonical": {tag: rows[tag] for tag in CANON},
        "rows": rows,
        "artifact_sha256": hashes,
        "producer_contract": {
            "historical_artifact_producer_sha256": HISTORICAL_PRODUCER_SHA256,
            "release_source_sha256": RELEASE_PRODUCER_SHA256,
            "docstring_stripped_ast_sha256": SEMANTIC_AST_SHA256,
            "change": "module docstring and reporting-only print text; numerical AST is identical",
        },
    }


def check_manuscript_headline(root: Path, summary: dict) -> None:
    """Bind the abstract's rounded headline to the all-run artifact summary."""
    abstract = (root / "submissions" / "jmlr_ceiling" / "j_abstract.tex").read_text(
        encoding="utf-8")
    captured = 100.0 * float(summary["worst_sharp"]["captured_energy_fraction"])
    rounded = f"{captured:.1f}"
    require(rounded == "98.8", f"unexpected one-decimal captured fraction {rounded}%")
    require(f"at least ${rounded}\\%$" in abstract,
            f"abstract does not contain the artifact-derived {rounded}% headline")
    require("99.8\\%" not in abstract,
            "abstract retains the obsolete four-canonical-run 99.8% headline")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    summary = build(root)
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = root / "calibration_figures" / "offmanifold_summary.json"
    if args.check:
        require(target.is_file(), f"missing {target}")
        require(target.read_text(encoding="utf-8") == text, f"{target} is stale")
        check_manuscript_headline(root, summary)
        print("OFF-MANIFOLD SUMMARY CHECK: PASS")
    else:
        atomic_text(target, text)
        print(f"wrote {target}")
    w = summary["worst_sharp"]
    q = summary["bulk_over_sharp"]
    print(f"all {summary['n_admitted']} admitted runs: worst sharp={w['fraction']:.12g} "
          f"({100*w['captured_energy_fraction']:.6f}% captured), "
          f"bulk/sharp={q['min']:.6g}--{q['max']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
