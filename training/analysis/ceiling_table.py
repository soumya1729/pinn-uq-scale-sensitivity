"""ceiling_table.py -- persist every cell of Table `tab:ceiling`, the paper's central result.

WHY THIS FILE EXISTS. `m4_full_laplace.py` produces the per-run inputs
(`{TAG}_full_laplace_metrics.json`, one per seed) and `gate_numbers.py` recomputes the row
aggregates at gate time, but nothing persisted the table itself: there was no artifact holding
"Burgers-Fourier's RMS-matched ratio gap is 16.3 +- 10.4 over three admitted seeds", only a gate that would
notice if the printed value changed. Same gap `solve_gate_table.py` closes for `tab:rejected`
and `temporal_per_seed.py` for `tab:temporal`.

THE CLOSED POPULATION IS BOTH DISCOVERED AND DECLARED. Admission is read from each run's own
literal Boolean `solve_gate.passed`; the resulting attempted and admitted seed sets must also
equal the paper's exact inventory. The independent raw-field contract names the same 17 admitted
tags. Thus neither a hard-coded answer alone nor an incomplete glob with the right count can
define a row.

THE CONVENTION, AND THE BUG IT REPLACED. Every nonlinear-PDE cell is across admitted seeds:
mean +- POPULATION standard deviation (divisor n, per `app:conventions`), with the ratio gap the
mean of the per-seed gaps R_tar/R_rms -- NOT the ratio of the two column means, which is a
different number. The control row uses the same convention across its three admitted paired
pulse-width/diffusivity settings at one seed. Until 2026-08-19 only the Burgers-MLP row was across-seed; the other three printed their
canonical seed's R_tar and gap with a dagger. Both conventions were individually correct and the
column was still wrong: Burgers-Fourier's canonical seed 2 carries R_tar 15.77 against 116.48
and 60.17 on its other two admitted seeds, so the row read 5.8 where the across-seed mean is
24.4, understating that configuration's deficit fourfold.

THE CONTROL ROW. Convection-diffusion has no `full_laplace_metrics.json` -- the full-network
battery was never run on it -- so both its primary R_rms and its secondary R_mean_width are
derived from a newly generated, self-describing raw-metric field, while R_tar is recomputed from
the reference/error field under Definition 4 (p95/p70 of |d_x u*|, RMS error ratio). The field
and terminal metadata must bind the current producer stack, exact admitted MAP, full 196009-
parameter scope, and exact stored region split. The derived mean-width contrast is compared
against the legacy `fullgrid_ratio.json` raw.ratio_mean to 2 dp as a historical-change diagnostic,
not a criterion that a newly validated field must reproduce the old answer. Paired-setting
old/new values and deltas are retained in the output; any change requires scientific review.
The --check mode still blocks stale printed PAPER_CTRL cells and a stale stored summary until
the result and the paper baseline are explicitly reviewed and updated. The legacy record is
not overwritten and is not used as the primary statistic. The exact five-run control suite is declared and independently
rediscovered, but the numerical row is architecture-homogeneous: its denominator is the four MLP
attempts, of which three are admitted. The fifth run is a separately disclosed rejected Fourier
attempt at the narrowest setting, so an incomplete glob cannot define either the row or the
surrounding control inventory.

    python analysis/ceiling_table.py           # table + calibration_figures/ceiling_table.json
    python analysis/ceiling_table.py --check   # additionally assert every printed cell

Pure numpy/json, no GPU, ~10 s. Reads only shipped artifacts:
    results/{TAG}/{TAG}_full_laplace_metrics.json -> ratios.{achieved_sigma_ratio,target_rmse_ratio}
    results/{TAG}/{TAG}_map.json                  -> solve_gate.passed
    results/cd_sweep/mlp_*/…_fields.npz + …_fullgrid_sigma_raw.npz
                              + …_fullgrid_sigma_raw.metadata.json + …_fullgrid_ratio.json
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "calibration_figures")
P_SHARP, P_SMOOTH = 95.0, 70.0

FAMILIES = [("Burgers, MLP", "burgers_mlp"),
            ("Burgers, Fourier", "burgers_fourier"),
            ("Allen--Cahn, MLP", "allen_cahn_mlp"),
            ("Allen--Cahn, Fourier", "allen_cahn_fourier")]
EXPECTED_SEEDS = {
    "burgers_mlp": {"attempted": tuple(range(8)), "admitted": tuple(range(8))},
    "burgers_fourier": {"attempted": tuple(range(8)), "admitted": (0, 2, 6)},
    "allen_cahn_mlp": {"attempted": tuple(range(5)), "admitted": (0, 1, 3)},
    "allen_cahn_fourier": {"attempted": tuple(range(4)), "admitted": (0, 1, 3)},
}
CONTROL_ATTEMPTS = {
    "results/cd_sweep/fou_d06_n01/convection_diffusion_fourier_seed0_map.json": {
        "passed": False, "arch": "fourier", "delta": 0.06, "nu": 0.0003183,
    },
    "results/cd_sweep/mlp_d09_n06/convection_diffusion_mlp_seed0_map.json": {
        "passed": True, "arch": "mlp", "delta": 0.09, "nu": 0.0019099,
    },
    "results/cd_sweep/mlp_d11_n08/convection_diffusion_mlp_seed0_map.json": {
        "passed": True, "arch": "mlp", "delta": 0.11, "nu": 0.0025465,
    },
    "results/cd_sweep/mlp_d13_n10/convection_diffusion_mlp_seed0_map.json": {
        "passed": True, "arch": "mlp", "delta": 0.13, "nu": 0.0031831,
    },
    "results/convection_diffusion_mlp_seed0/convection_diffusion_mlp_seed0_map.json": {
        "passed": False, "arch": "mlp", "delta": 0.06,
        "nu": 0.0003183098861837907,
    },
}
CONTROL_ADMITTED = (
    ("mlp_d09_n06", 0.09, 0.0019099),
    ("mlp_d11_n08", 0.11, 0.0025465),
    ("mlp_d13_n10", 0.13, 0.0031831),
)

# Table tab:ceiling as printed:
# (admitted, attempted, Rrms, Rrms_sd, Rmeanwidth, Rmeanwidth_sd, Rtar, Rtar_sd, gap, gap_sd)
PAPER = {
    "Burgers, MLP":         (8, 8, "6.37", "0.38", "3.87", "0.15", "130.6", "82.4", "20.5", "13.1"),
    "Burgers, Fourier":     (3, 8, "3.92", "0.04", "2.68", "0.09", "64.1", "41.2", "16.3", "10.4"),
    "Allen--Cahn, MLP":     (3, 5, "7.30", "0.35", "9.53", "0.57", "28.9", "5.2", "4.0", "0.8"),
    "Allen--Cahn, Fourier": (3, 4, "5.86", "0.05", "7.35", "0.13", "15.1", "1.9", "2.6", "0.3"),
}
PAPER_CTRL = (3, 4, "1.21", "0.10", "1.23", "0.10", "0.92", "0.10", "0.77", "0.10")
PAPER_ABSOLUTE = {
    "Burgers, MLP":         ("0.319", "0.275", "0.012", "0.003", "0.923", "0.083", "1.000", "0.000"),
    "Burgers, Fourier":     ("0.596", "0.141", "0.064", "0.044", "0.848", "0.031", "1.000", "0.000"),
    "Allen--Cahn, MLP":     ("0.271", "0.062", "0.030", "0.001", "0.990", "0.014", "1.000", "0.000"),
    "Allen--Cahn, Fourier": ("0.267", "0.016", "0.025", "0.002", "0.999", "0.002", "1.000", "0.000"),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path, purpose):
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception as exc:
        raise ValueError(f"cannot read {purpose} {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{purpose} {path}: expected a JSON object")
    return record


def _raw_contract():
    """Load the exact field/record bindings that establish the historical raw metric."""
    path = os.path.join(ROOT, "calibration_figures", "fullgrid_sigma_metric_contract.json")
    try:
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
    except Exception as exc:
        raise ValueError(f"cannot read raw-metric contract {path}: {exc}") from exc
    if (d.get("schema") != "fullgrid_sigma_metric_contract/3" or
            d.get("sigma_metric") != "raw" or d.get("admitted_field_count") != 17 or
            len(d.get("files", [])) != 17):
        raise ValueError(f"{path}: expected the complete v3 raw-metric contract")
    out = {}
    for row in d["files"]:
        tag = row.get("tag")
        if (not isinstance(tag, str) or tag in out or row.get("sigma_metric") != "raw" or
                not row.get("sha256") or not row.get("reference_field_sha256") or
                not row.get("full_laplace_record_sha256")):
            raise ValueError(f"{path}: malformed or duplicate field binding for {tag!r}")
        out[tag] = row
    return out


def _admitted(fam):
    """Seeds passing the solve gate, discovered from the artifacts."""
    adm, att = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", f"{fam}_seed*", "*_map.json"))):
        seed = int(os.path.basename(os.path.dirname(p)).rsplit("seed", 1)[1])
        try:
            meta = json.load(open(p))
        except Exception as exc:
            raise ValueError(f"{p}: unreadable attempted-run metadata: {exc}") from exc
        tag = f"{fam}_seed{seed}"
        sg = meta.get("solve_gate") or {}
        config = meta.get("provenance", {}).get("config", {})
        pde = "burgers" if fam.startswith("burgers_") else "allen_cahn"
        arch = "fourier" if fam.endswith("_fourier") else "mlp"
        expected = {
            "pde": pde, "arch": arch, "seed": seed,
            "Mx": 20, "Mt": 20, "hidden": 64, "fusion": 128,
            "N_f": 5000, "N_u": 1000, "N_b": 200,
            "eval_nx": 400, "eval_nt": 200, "smoke": 0,
        }
        if meta.get("tag") != tag:
            raise ValueError(f"{p}: tag {meta.get('tag')!r} does not match {tag!r}")
        if not all(config.get(key) == value for key, value in expected.items()):
            raise ValueError(f"{p}: configuration metadata does not match {tag}")
        if "bulk_rel_l2" not in sg or not isinstance(sg.get("passed"), bool):
            raise ValueError(f"{p}: solve-gate decision is absent or malformed")
        att.append(seed)
        if sg["passed"] is True:
            adm.append(seed)
    got_att, got_adm = tuple(sorted(att)), tuple(sorted(adm))
    expected = EXPECTED_SEEDS[fam]
    if (got_att != expected["attempted"] or got_adm != expected["admitted"] or
            len(set(att)) != len(att)):
        raise ValueError(
            f"{fam}: found attempted/admitted seeds {got_att}/{got_adm}; expected exactly "
            f"{expected['attempted']}/{expected['admitted']}"
        )
    return list(got_adm), list(got_att)


def _control_inventory():
    """Require all five control attempts; return the architecture-homogeneous MLP count."""
    def finite_number(value):
        if type(value) not in (int, float):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (OverflowError, ValueError):
            return False

    paths = set()
    for pattern in (
            os.path.join(ROOT, "results", "cd_sweep", "*", "convection_diffusion_*_seed0_map.json"),
            os.path.join(ROOT, "results", "convection_diffusion_*_seed0",
                         "convection_diffusion_*_seed0_map.json")):
        paths.update(os.path.normpath(path) for path in glob.glob(pattern))
    rels = {os.path.relpath(path, ROOT).replace(os.sep, "/"): path for path in paths}
    if set(rels) != set(CONTROL_ATTEMPTS):
        raise ValueError(
            "control: discovered attempted runs do not equal the declared five-run inventory; "
            f"found {sorted(rels)}, expected {sorted(CONTROL_ATTEMPTS)}")
    for rel, expected in CONTROL_ATTEMPTS.items():
        record = _read_json(rels[rel], "control admission record")
        provenance = record.get("provenance")
        config = provenance.get("config") if isinstance(provenance, dict) else None
        sg = record.get("solve_gate")
        if not isinstance(config, dict) or not isinstance(sg, dict):
            raise ValueError(f"{rel}: configuration or solve-gate metadata is malformed")
        for key in ("cd_delta", "cd_nu"):
            value = config.get(key)
            if not finite_number(value):
                raise ValueError(f"{rel}: {key} must be a finite number")
        bulk_error = sg.get("bulk_rel_l2")
        if (not finite_number(bulk_error) or bulk_error < 0 or
                sg.get("rel_l2_threshold") != 0.02 or
                sg.get("passed") is not (bulk_error < 0.02)):
            raise ValueError(f"{rel}: invalid control solve-gate value, threshold or decision")
        tag = f"convection_diffusion_{expected['arch']}_seed0"
        if (record.get("tag") != tag or
                sg.get("passed") is not expected["passed"] or
                config.get("pde") != "convection_diffusion" or
                config.get("arch") != expected["arch"] or
                type(config.get("seed")) is not int or config.get("seed") != 0 or
                not np.isclose(float(config["cd_delta"]), expected["delta"],
                               rtol=0.0, atol=1e-15) or
                not np.isclose(float(config["cd_nu"]), expected["nu"],
                               rtol=0.0, atol=1e-15)):
            raise ValueError(f"{rel}: configuration or solve-gate decision is not the declared one")
    mlp_attempts = sum(expected["arch"] == "mlp" for expected in CONTROL_ATTEMPTS.values())
    if mlp_attempts != 4:
        raise ValueError(f"control: declared MLP denominator changed to {mlp_attempts}, expected 4")
    return mlp_attempts


def _nonlinear_observables(reference_path, sigma_path):
    """Rebuild one nonlinear-PDE row from aligned arrays and Definition 4.

    The full-Laplace JSON is retained as an independent cross-check, not as the source of
    the table cells.  Exact coordinate and stored-index checks prevent two same-length fields
    on different grids, or two different percentile endpoint conventions, from being mixed.
    """
    try:
        with np.load(reference_path) as ref, np.load(sigma_path) as field:
            reference_required = {"xs", "ts", "grad_abs", "u_pred", "u_exact"}
            sigma_required = {"sigma", "xs", "ts", "shock_idx", "smooth_idx"}
            if not reference_required.issubset(ref.files):
                raise ValueError(
                    f"reference lacks {sorted(reference_required - set(ref.files))}")
            if not sigma_required.issubset(field.files):
                raise ValueError(
                    f"posterior field lacks {sorted(sigma_required - set(field.files))}")
            ref_xs, ref_ts = np.asarray(ref["xs"]), np.asarray(ref["ts"])
            sig_xs, sig_ts = np.asarray(field["xs"]), np.asarray(field["ts"])
            grad_abs = np.asarray(ref["grad_abs"], dtype=float).reshape(-1)
            error = np.abs(np.asarray(ref["u_pred"], dtype=float).reshape(-1) -
                           np.asarray(ref["u_exact"], dtype=float).reshape(-1))
            sigma = np.asarray(field["sigma"], dtype=float).reshape(-1)
            stored_sharp = np.asarray(field["shock_idx"], dtype=np.int64).reshape(-1)
            stored_bulk = np.asarray(field["smooth_idx"], dtype=np.int64).reshape(-1)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read {reference_path}/{sigma_path}: {exc}") from exc
    if (ref_xs.shape != sig_xs.shape or ref_ts.shape != sig_ts.shape or
            ref_xs.shape != (400,) or ref_ts.shape != (200,) or
            not np.array_equal(ref_xs, sig_xs) or not np.array_equal(ref_ts, sig_ts)):
        raise ValueError("posterior and reference evaluation grids do not match exactly")
    if (sigma.size != error.size or error.size != grad_abs.size or sigma.size != 80_000 or
            not np.isfinite(sigma).all() or np.any(sigma <= 0) or
            not np.isfinite(error).all() or not np.isfinite(grad_abs).all() or
            np.any(grad_abs < 0)):
        raise ValueError("invalid or unaligned sigma, error, or gradient field")
    p95, p70 = np.percentile(grad_abs, P_SHARP), np.percentile(grad_abs, P_SMOOTH)
    if np.count_nonzero(grad_abs == p95) or np.count_nonzero(grad_abs == p70):
        raise ValueError("percentile boundary ties make the endpoint convention ambiguous")
    sharp = np.flatnonzero(grad_abs >= p95)
    bulk = np.flatnonzero(grad_abs < p70)
    if (not np.array_equal(stored_sharp, sharp) or not np.array_equal(stored_bulk, bulk) or
            sharp.size != 4_000 or bulk.size != 56_000 or
            np.intersect1d(sharp, bulk).size):
        raise ValueError("stored regions do not equal Definition 4 on the reference field")
    return {
        "R_rms": float(np.sqrt(np.mean(sigma[sharp] ** 2) /
                               np.mean(sigma[bulk] ** 2))),
        "R_mean_width": float(np.mean(sigma[sharp]) / np.mean(sigma[bulk])),
        "R_tar": float(np.sqrt(np.mean(error[sharp] ** 2)) /
                       np.sqrt(np.mean(error[bulk] ** 2))),
        "meanZ_S": float(np.mean(error[sharp] / sigma[sharp])),
        "meanZ_B": float(np.mean(error[bulk] / sigma[bulk])),
        "coverage68_S": float(np.mean(error[sharp] <= sigma[sharp])),
        "coverage68_B": float(np.mean(error[bulk] <= sigma[bulk])),
    }


def _control():
    """Derive the admitted CD row only from stamped raw full-scope fields."""
    attempted = _control_inventory()
    rms, mean_width, target, crosscheck = [], [], [], []
    tag = "convection_diffusion_mlp_seed0"
    producer = os.path.join(ROOT, "step_fullgrid_ratio.py")
    common = os.path.join(ROOT, "common.py")
    laplace = os.path.join(ROOT, "laplace_core.py")
    for dirname, delta, nu in CONTROL_ADMITTED:
        directory = os.path.join(ROOT, "results", "cd_sweep", dirname)
        paths = {
            "map_record": os.path.join(directory, f"{tag}_map.json"),
            "map": os.path.join(directory, f"{tag}_params.pkl"),
            "reference": os.path.join(directory, f"{tag}_fields.npz"),
            "sigma": os.path.join(directory, f"{tag}_fullgrid_sigma_raw.npz"),
            "metadata": os.path.join(directory, f"{tag}_fullgrid_sigma_raw.metadata.json"),
            "legacy": os.path.join(directory, f"{tag}_fullgrid_ratio.json"),
        }
        missing = [name for name, path in paths.items() if not os.path.isfile(path)]
        if missing:
            raise ValueError(f"control/{dirname}: required inputs absent: {', '.join(missing)}")
        map_record = _read_json(paths["map_record"], "control admission record")
        metadata = _read_json(paths["metadata"], "raw-field metadata")
        legacy = _read_json(paths["legacy"], "legacy mean-width cross-check")
        config = map_record.get("provenance", {}).get("config", {})
        produced_config = metadata.get("configuration", {})
        expected_map_sha = _sha256(paths["map"])
        expected_code = {
            "producer": _sha256(producer), "common": _sha256(common),
            "laplace_core": _sha256(laplace),
        }
        if (map_record.get("tag") != tag or
                map_record.get("solve_gate", {}).get("passed") is not True or
                config.get("pde") != "convection_diffusion" or
                config.get("arch") != "mlp" or config.get("seed") != 0 or
                not np.isclose(config.get("cd_delta", np.nan), delta, rtol=0.0, atol=1e-15) or
                not np.isclose(config.get("cd_nu", np.nan), nu, rtol=0.0, atol=1e-15)):
            raise ValueError(f"control/{dirname}: admission record has the wrong run identity")
        if (metadata.get("schema") != "fullgrid_sigma_metadata/1" or
                metadata.get("tag") != tag or metadata.get("sigma_metric") != "raw" or
                metadata.get("scope") != "full" or metadata.get("P_bayes") != 196009 or
                metadata.get("alpha_multiplier") != 1.0 or
                metadata.get("negative_variance_relative_tolerance") != 1e-10 or
                metadata.get("inputs_sha256", {}).get("map") != expected_map_sha or
                metadata.get("code_sha256") != expected_code or
                metadata.get("field", {}).get("schema") != "fullgrid_sigma/1" or
                metadata.get("field", {}).get("sigma_metric") != "raw" or
                metadata.get("field", {}).get("sha256") != _sha256(paths["sigma"]) or
                not config or any(produced_config.get(key) != value
                                  for key, value in config.items())):
            raise ValueError(f"control/{dirname}: raw field is not bound to the admitted run "
                             "and current producer stack")
        with np.load(paths["reference"]) as ref, np.load(paths["sigma"]) as field:
            required = {"schema", "sigma_metric", "sigma", "xs", "ts", "shock_idx",
                        "smooth_idx", "scope", "P_bayes", "producer_sha256", "map_sha256",
                        "negative_variance_max_relative"}
            if not required.issubset(field.files):
                raise ValueError(f"control/{dirname}: raw field lacks its v1 provenance payload")
            sigma = np.asarray(field["sigma"], dtype=float).reshape(-1)
            xs, ts = np.asarray(field["xs"]), np.asarray(field["ts"])
            shock_idx = np.asarray(field["shock_idx"], dtype=int).reshape(-1)
            smooth_idx = np.asarray(field["smooth_idx"], dtype=int).reshape(-1)
            if (str(field["schema"].item()) != "fullgrid_sigma/1" or
                    str(field["sigma_metric"].item()) != "raw" or
                    str(field["scope"].item()) != "full" or int(field["P_bayes"]) != 196009 or
                    str(field["producer_sha256"].item()) != expected_code["producer"] or
                    str(field["map_sha256"].item()) != expected_map_sha or
                    float(field["negative_variance_max_relative"]) > 1e-10 or
                    sigma.size != 80000 or not np.isfinite(sigma).all() or
                    np.any(sigma <= 0) or xs.shape != (400,) or ts.shape != (200,) or
                    not np.array_equal(xs, np.asarray(ref["xs"])) or
                    not np.array_equal(ts, np.asarray(ref["ts"]))):
                raise ValueError(f"control/{dirname}: invalid or misbound raw sigma field")
            grad_abs = np.asarray(ref["grad_abs"], dtype=float).reshape(-1)
            error = np.abs(np.asarray(ref["u_pred"], dtype=float).reshape(-1) -
                           np.asarray(ref["u_exact"], dtype=float).reshape(-1))
            if (grad_abs.size != 80000 or error.size != 80000 or
                    not np.isfinite(grad_abs).all() or np.any(grad_abs < 0) or
                    not np.isfinite(error).all()):
                raise ValueError(f"control/{dirname}: invalid reference/error field")
            sharp = grad_abs >= np.percentile(grad_abs, P_SHARP)
            smooth = grad_abs < np.percentile(grad_abs, P_SMOOTH)
            expected_sharp = np.flatnonzero(sharp)
            expected_smooth = np.flatnonzero(smooth)
            if (not np.array_equal(shock_idx, expected_sharp) or
                    not np.array_equal(smooth_idx, expected_smooth) or
                    shock_idx.size != 4000 or smooth_idx.size != 56000 or
                    np.intersect1d(shock_idx, smooth_idx).size):
                raise ValueError(f"control/{dirname}: stored region split disagrees with Definition 4")
        mean_width.append(float(sigma[sharp].mean() / sigma[smooth].mean()))
        rms.append(float(np.sqrt(np.mean(sigma[sharp] ** 2) /
                                 np.mean(sigma[smooth] ** 2))))
        if np.sqrt(np.mean(error[smooth] ** 2)) <= 0:
            raise ValueError(f"control/{dirname}: zero bulk RMS error")
        target.append(float(np.sqrt(np.mean(error[sharp] ** 2)) /
                            np.sqrt(np.mean(error[smooth] ** 2))))
        try:
            crosscheck.append(float(legacy["raw"]["ratio_mean"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"control/{dirname}: malformed legacy mean-width cross-check") from exc
    return rms, mean_width, target, crosscheck, attempted


def _stat(a):
    return {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=0)),
            "min": float(np.min(a)), "max": float(np.max(a)),
            "per_run": [float(x) for x in a]}


def _row(rms, rs, rt):
    gp = [t / s for s, t in zip(rms, rt)]
    gp_mw = [t / s for s, t in zip(rs, rt)]
    return {"R_rms": _stat(rms), "R_mean_width": _stat(rs), "R_tar": _stat(rt),
            "ratio_gap": _stat(gp), "mean_width_gap": _stat(gp_mw)}


def _control_legacy_diagnostic(current, legacy):
    """Compare paired old/new scalars without defining validity by the historical answer."""
    if len(current) != len(CONTROL_ADMITTED) or len(legacy) != len(CONTROL_ADMITTED):
        raise ValueError("control: historical comparison requires the exact three paired settings")
    comparisons = []
    for (setting, _, _), new, old in zip(CONTROL_ADMITTED, current, legacy):
        new, old = float(new), float(old)
        if not np.isfinite(new) or not np.isfinite(old) or new <= 0 or old <= 0:
            raise ValueError(f"control/{setting}: invalid historical mean-width comparison")
        delta = new - old
        comparisons.append({
            "setting": setting,
            "legacy_record": f"results/cd_sweep/{setting}/"
                             "convection_diffusion_mlp_seed0_fullgrid_ratio.json",
            "new_mean_width_contrast": new,
            "legacy_mean_width_contrast": old,
            "signed_delta": delta,
            "absolute_delta": abs(delta),
            "relative_delta_vs_legacy": delta / old,
            "same_at_2_decimal_places": f"{new:.2f}" == f"{old:.2f}",
        })
    return {
        "role": "historical regression diagnostic, not current-field validation",
        "changed_at_print_precision": any(not row["same_at_2_decimal_places"]
                                           for row in comparisons),
        "review_approval_recorded_here": False,
        "per_setting": comparisons,
    }


def main():
    unknown = [arg for arg in sys.argv[1:] if arg != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown)}")
    check = "--check" in sys.argv
    report, bad = {}, []
    contract = _raw_contract()

    print(f"{'configuration':<24}{'adm':>5}{'R_rms':>16}{'R_mw':>16}"
          f"{'R_tar':>18}{'rho':>16}")
    for label, fam in FAMILIES:
        adm, att = _admitted(fam)
        rms, rs, rt = [], [], []
        zS, zB, c68S, c68B = [], [], [], []
        for s in adm:
            p = os.path.join(ROOT, "results", f"{fam}_seed{s}",
                             f"{fam}_seed{s}_full_laplace_metrics.json")
            if not os.path.exists(p):
                bad.append(f"{label}: seed {s} admitted but no full_laplace_metrics.json")
                continue
            tag = f"{fam}_seed{s}"
            try:
                d = _read_json(p, "full-Laplace record")
            except ValueError as exc:
                bad.append(f"{label}/seed{s}: {exc}")
                continue
            binding = contract.get(tag)
            sp = os.path.join(ROOT, "results", tag, f"{tag}_fullgrid_sigma_raw.npz")
            fp = os.path.join(ROOT, "results", tag, f"{tag}_fields.npz")
            if (binding is None or not os.path.isfile(sp) or not os.path.isfile(fp) or
                    _sha256(fp) != binding.get("reference_field_sha256") or
                    _sha256(sp) != binding.get("sha256") or
                    _sha256(p) != binding.get("full_laplace_record_sha256")):
                bad.append(f"{label}/seed{s}: raw field/full-Laplace record is not the "
                           "exact field/reference/record set bound by the v3 metric contract")
                continue
            if (d.get("tag") != f"{fam}_seed{s}" or d.get("scope") != "full" or
                    ("sigma_metric" in d and d.get("sigma_metric") != "raw") or
                    d.get("solve_gate", {}).get("passed") is not True):
                bad.append(f"{label}/seed{s}: full-Laplace record is not an admitted "
                           "full-scope record for this tag (any explicit metric must be raw)")
                continue
            try:
                observed = _nonlinear_observables(fp, sp)
                recorded = {
                    "R_mean_width": float(d["ratios"]["achieved_sigma_ratio"]),
                    "R_tar": float(d["ratios"]["target_rmse_ratio"]),
                    "meanZ_S": float(d["regions"]["shock"]["mean_z"]),
                    "meanZ_B": float(d["regions"]["smooth"]["mean_z"]),
                    "coverage68_S": float(d["regions"]["shock"]["coverage68"]),
                    "coverage68_B": float(d["regions"]["smooth"]["coverage68"]),
                }
            except (ValueError, KeyError, TypeError) as exc:
                bad.append(f"{label}/seed{s}: cannot rebuild table inputs: {exc}")
                continue
            mismatch = [name for name, value in recorded.items()
                        if (not np.isfinite(value) or
                            abs(value - observed[name]) >
                            1e-10 * max(1.0, abs(observed[name])))]
            if mismatch:
                bad.append(f"{label}/seed{s}: full-Laplace record disagrees with bound "
                           f"fields for {', '.join(mismatch)}")
                continue
            rms.append(observed["R_rms"])
            rs.append(observed["R_mean_width"])
            rt.append(observed["R_tar"])
            zS.append(observed["meanZ_S"])
            zB.append(observed["meanZ_B"])
            c68S.append(observed["coverage68_S"])
            c68B.append(observed["coverage68_B"])
        if not rs:
            print(f"{label:<24}  (no metrics)")
            continue
        row = _row(rms, rs, rt)
        row["absolute"] = {"meanZ_S": _stat(zS), "meanZ_B": _stat(zB),
                           "coverage68_S": _stat(c68S), "coverage68_B": _stat(c68B)}
        row.update(admitted=len(adm), attempted=len(att), seeds=adm)
        report[label] = row
        R, H, T, G = row["R_rms"], row["R_mean_width"], row["R_tar"], row["ratio_gap"]
        print(f"{label:<24}{len(adm)}/{len(att):<3}"
              f"{R['mean']:>9.2f} ± {R['sd']:<4.2f}"
              f"{H['mean']:>9.2f} ± {H['sd']:<4.2f}"
              f"{T['mean']:>11.1f} ± {T['sd']:<5.1f}"
              f"{G['mean']:>9.1f} ± {G['sd']:<5.1f}")

        if check and label in PAPER:
            a, t, m1, s1, h1, hs, m2, s2, m3, s3 = PAPER[label]
            got = (len(adm), len(att), f"{R['mean']:.2f}", f"{R['sd']:.2f}",
                   f"{H['mean']:.2f}", f"{H['sd']:.2f}",
                   f"{T['mean']:.1f}", f"{T['sd']:.1f}",
                   f"{G['mean']:.1f}", f"{G['sd']:.1f}")
            if got != (a, t, m1, s1, h1, hs, m2, s2, m3, s3):
                bad.append(f"{label}: got {got}, paper {PAPER[label]}")
            arow = row["absolute"]
            got_abs = tuple(
                f"{arow[k][field]:.3f}"
                for k in ("meanZ_S", "meanZ_B", "coverage68_S", "coverage68_B")
                for field in ("mean", "sd")
            )
            if got_abs != PAPER_ABSOLUTE[label]:
                bad.append(f"{label}: absolute row got {got_abs}, paper {PAPER_ABSOLUTE[label]}")

    try:
        rms, rs, rt, chk, control_attempted = _control()
        historical = _control_legacy_diagnostic(rs, chk)
    except ValueError as exc:
        rms, rs, rt, chk, control_attempted = [], [], [], [], None
        bad.append(str(exc))
    if rs:
        row = _row(rms, rs, rt)
        row.update(admitted=len(rs), attempted=control_attempted,
                   attempted_architecture="mlp", control_suite_attempted=len(CONTROL_ATTEMPTS),
                   separate_fourier_attempt={"admitted": 0, "attempted": 1},
                   seeds=["admitted pulse-width/diffusivity settings, seed 0"],
                   R_mean_width_crosscheck_fullgrid_ratio=chk,
                   historical_mean_width_comparison=historical)
        report["convection--diffusion (control)"] = row
        R, H, T, G = row["R_rms"], row["R_mean_width"], row["R_tar"], row["ratio_gap"]
        print(f"{'conv--diff, MLP':<24}{len(rs)}/{control_attempted}  "
              f"{R['mean']:>7.2f} ± {R['sd']:<4.2f}"
              f"{H['mean']:>9.2f} ± {H['sd']:<4.2f}"
              f"{T['mean']:>11.2f} ± {T['sd']:<5.2f}"
              f"{G['mean']:>9.2f} ± {G['sd']:<5.2f}")
        # A valid correction must be reportable even when it differs from the historical scalar.
        # Keep the paired deltas visible without substituting equality to an old answer for the
        # provenance/numerical contracts already enforced by _control(). No approval is inferred.
        if historical["changed_at_print_precision"]:
            print("\nREVIEW REQUIRED: control mean-width values differ from the historical "
                  "records at printed precision. This does not invalidate the new fields; "
                  "review the differences and paper before publication.")
            for comparison in historical["per_setting"]:
                if not comparison["same_at_2_decimal_places"]:
                    print(f"  {comparison['setting']}: legacy "
                          f"{comparison['legacy_mean_width_contrast']:.10g} -> new "
                          f"{comparison['new_mean_width_contrast']:.10g}; delta "
                          f"{comparison['signed_delta']:+.6g} "
                          f"({100 * comparison['relative_delta_vs_legacy']:+.4g}%)")
        if check:
            a, t, m1, s1, h1, hs, m2, s2, m3, s3 = PAPER_CTRL
            got = (len(rs), control_attempted, f"{R['mean']:.2f}", f"{R['sd']:.2f}",
                   f"{H['mean']:.2f}", f"{H['sd']:.2f}",
                   f"{T['mean']:.2f}", f"{T['sd']:.2f}",
                   f"{G['mean']:.2f}", f"{G['sd']:.2f}")
            if got != (a, t, m1, s1, h1, hs, m2, s2, m3, s3):
                bad.append(f"control: scientifically validated row {got} differs from the "
                           f"printed baseline {PAPER_CTRL}; scientific review and an explicit "
                           "paper/baseline update are required (not a reason to force old values)")

    print("\nper-seed detail (the aggregation the dagger convention used to hide)")
    for label, row in report.items():
        z = zip(row["seeds"] * len(row["R_tar"]["per_run"]) if len(row["seeds"]) == 1
                else row["seeds"], row["R_tar"]["per_run"], row["ratio_gap"]["per_run"])
        cells = "  ".join(f"{s}:{t:.1f}/{g:.1f}" for s, t, g in z)
        print(f"  {label:<24} R_tar/gap by seed -- {cells}")

    if bad:
        print("\nCHECK FAILED" if check else "\nPROBLEMS")
        for b in bad:
            print("  " + b)
        return 1
    dst = os.path.join(OUT, "ceiling_table.json")
    if check:
        if not os.path.isfile(dst):
            bad.append(f"stored table artifact is missing: {dst}")
        else:
            try:
                stored = _read_json(dst, "stored ceiling-table artifact")
                if stored != report:
                    bad.append(f"{dst} is stale: stored payload differs from the live rebuild")
            except (OSError, ValueError) as exc:
                bad.append(f"cannot read stored table artifact {dst}: {exc}")
    if bad:
        print("\nCHECK FAILED" if check else "\nPROBLEMS")
        for b in bad:
            print("  " + b)
        return 1
    if not check:
        os.makedirs(OUT, exist_ok=True)
        with open(dst + ".partial", "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(dst + ".partial", dst)
        print(f"\nwrote {os.path.relpath(dst, ROOT)}")
    if check:
        print("\nCHECK OK -- stored table and every RMS-primary cell of tab:ceiling reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
