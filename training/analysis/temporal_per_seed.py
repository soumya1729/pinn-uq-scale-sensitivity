"""temporal_per_seed.py -- persist the PER-SEED pre/post shock-formation table.

WHY THIS FILE EXISTS. `temporal_calibration.py` disaggregates ONE run and prints its report to
stdout; `_paper_scripts/check_paper_consistency.py` recomputes all 8 seeds but only compares
against the .tex and keeps nothing. So the per-seed table behind Table `tab:temporal` and behind
the two ranges quoted in section 5.4 -- post-formation cov@68 in [0.37, 1.00] and meanZ in
[0.02, 2.43] -- existed in NO artifact. Both ranges, and the "four seeds at 1.00" claim, were
hand-typed into a docstring from a shell session that was not saved.

That is precisely the defect that produced the fabricated 1-D loading overlaps earlier in this
project (a range unpacked from memory into four per-config values, three of them wrong, one
inverted). A number that lives only in prose has no producer and cannot be re-derived after the
shell scrollback is gone. This script gives those numbers a producer.

WHAT IT DOES NOT DO. It does not replace the consistency gate. The gate reads its comparison
values from the .tex and would catch an edit to the paper; this script reads nothing from the
paper and would not. They are complementary: this one makes the table reproducible, that one
makes it agree with what is printed. Running this does not discharge the gate.

CONVENTIONS. Partition is the paper's: global percentiles over all 80,000 (x,t) points, sharp =
top 5% of |d_x u*|, smooth = bottom 70%, computed once per seed over all t and then masked by
the time window (NOT re-derived per window -- the point is to disaggregate the published
partition, not to replace it). Split at the classical breaking time t = 1/pi. Dispersions are
POPULATION standard deviations, divisor n, matching `app:conventions`. Aggregate ratio-gap is
the mean of the per-seed RMS-matched gaps, matching the reference row of Table `tab:ceiling`.
The old mean-width contrast and its mismatched gap remain in the artifact as explicitly named
secondary fields.

    python analysis/temporal_per_seed.py           # table + calibration_figures/temporal_per_seed.json
    python analysis/temporal_per_seed.py --check   # additionally assert the paper's printed values

Pure numpy, no GPU, ~15 s. Reads only shipped artifacts:
    results/{TAG}/{TAG}_fields.npz
    results/{TAG}/{TAG}_fullgrid_sigma_raw.npz
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "calibration_figures")
SEEDS = list(range(8))
TAG = "burgers_mlp_seed%d"
TSH = 1.0 / np.pi          # classical breaking time for u0 = -sin(pi x), nu -> 0
P_SHARP, P_SMOOTH = 95.0, 70.0
SCHEMA = "temporal_per_seed/2"

# Table `tab:temporal` as printed, for --check. Kept here deliberately: this script is a
# PRODUCER, and a producer asserting its own output against the paper is the right direction of
# dependence. The gate that must not hard-code these is check_paper_consistency.py, which reads
# them from the .tex instead.
PAPER = {
    "pre":  {"R_rms": 2.65, "gap": 16.9, "meanZ": 0.10, "cov68": 0.99, "cov95": 1.00},
    "post": {"R_rms": 10.75, "gap": 20.6, "meanZ": 0.81, "cov68": 0.76, "cov95": 0.87},
    "all":  {"R_rms": 6.37, "gap": 20.5, "meanZ": 0.32, "cov68": 0.92, "cov95": 0.96},
}
# Population sds, ALL five columns. Until 2026-08-19 the table printed a dispersion on
# R_sigma alone and bare means elsewhere -- showing it on the one column that replicates
# (CV 6%) and hiding it on the four that do not (meanZ 100%, gap 63%, cov@68 35%, cov@95
# 19%). Section 5.4 argues that absolute calibration does not survive a seed change while
# the contrast does; the table was its evidence and implied the opposite.
PAPER_SD = {
    "pre":  {"R_rms": 0.23, "gap": 14.5, "meanZ": 0.07, "cov68": 0.01, "cov95": 0.00},
    "post": {"R_rms": 0.71, "gap": 13.1, "meanZ": 0.81, "cov68": 0.27, "cov95": 0.17},
    "all":  {"R_rms": 0.38, "gap": 13.1, "meanZ": 0.28, "cov68": 0.08, "cov95": 0.05},
}
PAPER_PRE_SHARE = 69          # % of S at t < 1/pi
PAPER_THRESHOLD = 3.34        # |d_x u*| at the 95th percentile


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_contract():
    path = os.path.join(ROOT, "calibration_figures", "fullgrid_sigma_metric_contract.json")
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read raw-field contract: {exc}") from exc
    files = record.get("files", [])
    if (record.get("schema") != "fullgrid_sigma_metric_contract/3" or
            record.get("sigma_metric") != "raw" or
            record.get("admitted_field_count") != 17 or len(files) != 17):
        raise ValueError("complete v3 raw-field contract is required")
    bindings = {row.get("tag"): row for row in files if isinstance(row, dict)}
    if len(bindings) != 17 or None in bindings:
        raise ValueError("raw-field contract contains a duplicate or malformed tag")
    return path, bindings


def _load(seed, contract):
    tag = TAG % seed
    d = os.path.join(ROOT, "results", tag)
    map_path = os.path.join(d, "%s_map.json" % tag)
    fields_path = os.path.join(d, "%s_fields.npz" % tag)
    sigma_path = os.path.join(d, "%s_fullgrid_sigma_raw.npz" % tag)
    binding = contract.get(tag)
    try:
        if (not binding or binding.get("sigma_metric") != "raw" or
                binding.get("reference_field_sha256") != _sha256(fields_path) or
                binding.get("sha256") != _sha256(sigma_path)):
            raise ValueError("field pair is not bound by the v3 raw-metric contract")
    except OSError as exc:
        raise ValueError(f"{tag}: cannot hash required field: {exc}") from exc
    try:
        meta = json.load(open(map_path))
    except Exception as exc:
        raise ValueError(f"{tag}: missing/unreadable MAP metadata: {exc}") from exc
    cfg = meta.get("provenance", {}).get("config", {})
    expected = {"pde": "burgers", "arch": "mlp", "seed": seed, "Mx": 20, "Mt": 20,
                "hidden": 64, "fusion": 128, "N_f": 5000, "N_u": 1000, "N_b": 200,
                "eval_nx": 400, "eval_nt": 200, "smoke": 0}
    if (meta.get("tag") != tag or meta.get("solve_gate", {}).get("passed") is not True or
            not all(cfg.get(k) == v for k, v in expected.items())):
        raise ValueError(f"{tag}: tag, admission, or experiment configuration drift")
    try:
        with np.load(fields_path) as z:
            required = {"xs", "ts", "grad_abs", "u_pred", "u_exact"}
            if not required.issubset(z.files):
                raise ValueError(f"fields lack {sorted(required - set(z.files))}")
            xs = np.asarray(z["xs"], dtype=float).reshape(-1)
            ts = np.asarray(z["ts"], dtype=float).reshape(-1)
            grad = np.asarray(z["grad_abs"], dtype=float).reshape(-1)
            pred = np.asarray(z["u_pred"], dtype=float).reshape(-1)
            exact = np.asarray(z["u_exact"], dtype=float).reshape(-1)
        with np.load(sigma_path) as z:
            required = {"sigma", "xs", "ts", "shock_idx", "smooth_idx"}
            if not required.issubset(z.files):
                raise ValueError(f"sigma field lacks {sorted(required - set(z.files))}")
            sigma = np.asarray(z["sigma"], dtype=float).reshape(-1)
            sx = np.asarray(z["xs"], dtype=float).reshape(-1)
            st = np.asarray(z["ts"], dtype=float).reshape(-1)
            shi = np.asarray(z["shock_idx"], dtype=np.int64).reshape(-1)
            smi = np.asarray(z["smooth_idx"], dtype=np.int64).reshape(-1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{tag}: malformed field artifact: {exc}") from exc
    if (xs.size != 400 or ts.size != 200 or sigma.size != 80_000 or
            not (grad.size == pred.size == exact.size == sigma.size) or
            not np.array_equal(xs, sx) or not np.array_equal(ts, st) or
            not all(np.all(np.isfinite(a)) for a in (xs, ts, grad, pred, exact, sigma)) or
            np.any(grad < 0) or np.any(sigma <= 0)):
        raise ValueError(f"{tag}: non-finite, non-positive, misaligned, or wrong-size fields")
    sh_expected = np.flatnonzero(grad >= np.percentile(grad, P_SHARP))
    sm_expected = np.flatnonzero(grad < np.percentile(grad, P_SMOOTH))
    if (np.count_nonzero(grad == np.percentile(grad, P_SHARP)) or
            np.count_nonzero(grad == np.percentile(grad, P_SMOOTH)) or
            not np.array_equal(shi, sh_expected) or not np.array_equal(smi, sm_expected)):
        raise ValueError(f"{tag}: stored regional indices do not implement the paper's split")
    nx = xs.size
    t = np.repeat(ts, nx)
    err = np.abs(pred - exact)
    return grad, t, err, sigma, {
        os.path.relpath(map_path, ROOT).replace(os.sep, "/"): _sha256(map_path),
        os.path.relpath(fields_path, ROOT).replace(os.sep, "/"): _sha256(fields_path),
        os.path.relpath(sigma_path, ROOT).replace(os.sep, "/"): _sha256(sigma_path),
    }


def _window(grad, t, err, sigma, mask):
    """Metrics on one time window, using the globally-defined regions."""
    sharp = (grad >= np.percentile(grad, P_SHARP)) & mask
    smooth = (grad < np.percentile(grad, P_SMOOTH)) & mask
    if not sharp.any() or not smooth.any():
        raise ValueError("empty sharp or bulk region in temporal window")
    if np.sqrt((err[smooth] ** 2).mean()) <= 0:
        raise ValueError("zero bulk RMS error makes the target contrast undefined")
    z = err[sharp] / sigma[sharp]
    r_mean_width = float(sigma[sharp].mean() / sigma[smooth].mean())
    r_rms = float(np.sqrt((sigma[sharp] ** 2).mean() / (sigma[smooth] ** 2).mean()))
    r_tar = float(np.sqrt((err[sharp] ** 2).mean()) / np.sqrt((err[smooth] ** 2).mean()))
    return {
        "R_rms": r_rms,
        "R_mean_width": r_mean_width,
        "R_tar": r_tar,
        "gap": r_tar / r_rms,
        "mean_width_gap": r_tar / r_mean_width,
        "meanZ": float(z.mean()),
        "cov68": float((z <= 1.0).mean()),
        "cov95": float((z <= 1.96).mean()),
        "n_sharp": int(sharp.sum()),
    }


def main():
    unknown = [arg for arg in sys.argv[1:] if arg != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown)}")
    check = "--check" in sys.argv
    rows, shares, thresholds, inputs = {}, [], [], {}

    try:
        contract_path, contract = _raw_contract()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for s in SEEDS:
        grad, t, err, sigma, seed_inputs = _load(s, contract)
        inputs.update(seed_inputs)
        sharp_all = grad >= np.percentile(grad, P_SHARP)
        shares.append(100.0 * (sharp_all & (t < TSH)).sum() / sharp_all.sum())
        thresholds.append(float(np.percentile(grad, P_SHARP)))
        rows[s] = {
            "pre":  _window(grad, t, err, sigma, t < TSH),
            "post": _window(grad, t, err, sigma, t >= TSH),
            "all":  _window(grad, t, err, sigma, np.ones_like(t, dtype=bool)),
        }

    agg = {}
    for w in ("pre", "post", "all"):
        agg[w] = {}
        for k in ("R_rms", "R_mean_width", "R_tar", "gap", "mean_width_gap",
                  "meanZ", "cov68", "cov95"):
            v = np.array([rows[s][w][k] for s in SEEDS])
            agg[w][k] = float(v.mean())
            agg[w][k + "_sd"] = float(v.std(ddof=0))     # population, per app:conventions
            agg[w][k + "_min"] = float(v.min())
            agg[w][k + "_max"] = float(v.max())
        agg[w]["n_cov68_at_1"] = int(sum(rows[s][w]["cov68"] >= 0.9995 for s in SEEDS))

    hdr = "%-6s %9s %9s %8s %8s %8s" % ("seed", "R_rms", "meanZ", "cov@68", "cov@95", "gap")
    for w, label in (("pre", "PRE-FORMATION  t < 1/pi"),
                     ("post", "POST-FORMATION t >= 1/pi"),
                     ("all", "ALL t  (as reported)")):
        print("\n%s" % label)
        print(hdr)
        for s in SEEDS:
            r = rows[s][w]
            print("%-6d %9.3f %9.3f %8.3f %8.3f %8.1f"
                  % (s, r["R_rms"], r["meanZ"], r["cov68"], r["cov95"], r["gap"]))
        a = agg[w]
        # Every column carries its population sd, matching the table since 2026-08-19. Printing
        # a dispersion under R_sigma alone -- as this script and the table both once did -- puts
        # the error bar on the one quantity that replicates (CV 6%) and hides it on the four
        # that do not, which is backwards from what section 5.4 argues.
        print("%-6s %9.2f %9.2f %8.2f %8.2f %8.1f   <- Table tab:temporal"
              % ("MEAN", a["R_rms"], a["meanZ"], a["cov68"], a["cov95"], a["gap"]))
        print("%-6s %9s %9s %8s %8s %8s"
              % ("sd(n)", "± %.2f" % a["R_rms_sd"], "± %.2f" % a["meanZ_sd"],
                 "± %.2f" % a["cov68_sd"], "± %.2f" % a["cov95_sd"],
                 "± %.1f" % a["gap_sd"]))
        print("%-6s %9s %9s %8s %8s %8s"
              % ("CV", "%.0f%%" % (100 * a["R_rms_sd"] / a["R_rms"]),
                 "%.0f%%" % (100 * a["meanZ_sd"] / max(a["meanZ"], 1e-12)),
                 "%.0f%%" % (100 * a["cov68_sd"] / a["cov68"]),
                 "%.0f%%" % (100 * a["cov95_sd"] / a["cov95"]),
                 "%.0f%%" % (100 * a["gap_sd"] / a["gap"])))
        print("       ranges: meanZ [%.2f, %.2f]  cov@68 [%.2f, %.2f]  (%d seeds at cov@68 = 1.00)"
              % (a["meanZ_min"], a["meanZ_max"], a["cov68_min"], a["cov68_max"],
                 a["n_cov68_at_1"]))

    share = float(np.mean(shares))
    thr = float(np.mean(thresholds))
    print("\npre-formation share of S: %.1f%%  (paper: %d%%)" % (share, PAPER_PRE_SHARE))
    print("p95 threshold |d_x u*|:   %.2f   (paper: %.2f)" % (thr, PAPER_THRESHOLD))

    dst = os.path.join(OUT, "temporal_per_seed.json")
    record = {
        "schema": SCHEMA,
        "sigma_metric": "raw",
        "seeds": SEEDS,
        "t_split": float(TSH),
        "percentiles": {"sharp": P_SHARP, "smooth": P_SMOOTH},
        "sd_convention": "population, ddof=0 (app:conventions)",
        "primary_contrast": "R_rms = sqrt(<sigma^2>_S/<sigma^2>_B)",
        "per_seed": {str(s): rows[s] for s in SEEDS},
        "aggregate": agg,
        "pre_formation_share_pct": share,
        "p95_threshold": thr,
        "inputs_sha256": inputs,
        "raw_metric_contract_sha256": _sha256(contract_path),
        "code_sha256": _sha256(__file__),
    }
    if not check:
        os.makedirs(OUT, exist_ok=True)
        with open(dst + ".partial", "w") as f:
            json.dump(record, f, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(dst + ".partial", dst)
        print("\nwrote %s" % os.path.relpath(dst, ROOT))

    if check:
        bad = []
        for w, exp in PAPER.items():
            for k, v in exp.items():
                got = agg[w][k]
                tol = 0.05 if k == "gap" else 0.005
                if abs(got - v) > tol:
                    bad.append("%s.%s: got %.4f, paper %.4f" % (w, k, got, v))
            for k, v in PAPER_SD[w].items():
                got = agg[w][k + "_sd"]
                tol = 0.05 if k == "gap" else 0.005
                if abs(got - v) > tol:
                    bad.append("%s.%s_sd: got %.4f, paper %.4f" % (w, k, got, v))
        if abs(share - PAPER_PRE_SHARE) > 0.5:
            bad.append("pre-formation share: got %.2f, paper %d" % (share, PAPER_PRE_SHARE))
        if abs(thr - PAPER_THRESHOLD) > 0.005:
            bad.append("p95 threshold: got %.4f, paper %.2f" % (thr, PAPER_THRESHOLD))
        try:
            stored = json.load(open(dst))
        except Exception as exc:
            bad.append(f"cannot read stored temporal summary: {exc}")
        else:
            if stored != record:
                bad.append("stored temporal summary differs from live artifacts")
        if bad:
            print("\nCHECK FAILED")
            for b in bad:
                print("  " + b)
            return 1
        print("\nCHECK OK -- every printed value in tab:temporal and section 5.4 reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
