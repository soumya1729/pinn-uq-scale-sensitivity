#!/usr/bin/env python3
"""Deep-ensemble calibration on the 8 admitted Burgers-MLP seeds (``sec:ensemble``).

    python3 analysis/ensemble_calibration.py            # rewrite the artifact
    python3 analysis/ensemble_calibration.py --check    # verify stored artifact and paper values

WHY THIS EXISTS. `results/ensemble_calibration_burgers_mlp.json` shipped without a writer: it
was produced by the pre-consolidation step3a pipeline and carried hand-rounded values and no
`provenance` block, alone among the artifacts in this tree. The ``sec:ensemble`` claim -- the one
multi-mode intervention that exceeds the observed single-mode contrasts -- rested on a file
nobody could regenerate.  The fixed-linearisation ceiling does not apply to that ensemble.
Every input it
needs does ship, so this reconstructs it from them and reproduces all eight printed values.

THE ESTIMATOR, stated because the paper got it wrong. sigma is the standard deviation across
the K = 8 members with divisor K, not K - 1. That is what `app:conventions` mandates for every
dispersion in the paper, and it is what reproduces the artifact: at ddof = 1 the shock meanZ is
0.598 against the printed 0.64 and cov@68 is 0.910 against 0.876. The prose said "sample
standard deviation"; it is the population one. Corrected 2026-08-24.

The ensemble prediction is the member mean, the error is taken against the same spectral
reference every member is scored on, and the partition is ``def:regions`` applied to that
reference -- all three identical across the eight files, which this asserts rather than assumes.
"""
import argparse, hashlib, json, math, os, platform, sys
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(CORE, "results", "ensemble_calibration_burgers_mlp.json")
TAGS = [f"burgers_mlp_seed{s}" for s in range(8)]
SCHEMA = "ensemble_calibration/3"

# what ``sec:ensemble`` prints, at the precision printed there. The RMS-matched ratio is
# primary; the historical mean-width ratio is retained as a secondary operational quantity.
PRINTED = {"sigma_rms_ratio_between_basin": (136.5, 1),
           "target_rms_ratio_ensemble_mean": (158.2, 1),
           "ratio_gap_ensemble_mean": (1.16, 2),
           "sigma_ratio_between_basin": (167, 0), "shock.meanZ": (0.64, 2),
           "shock.cov68": (0.876, 3), "shock.cov95": (0.993, 3),
           "smooth.meanZ": (0.45, 2), "shock.mean_sigma": (0.0532, 4)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute():
    U, ref, ga, xs0, ts0 = [], None, None, None, None
    inputs = {}
    for seed, t in enumerate(TAGS):
        p = os.path.join(CORE, "results", t, f"{t}_fields.npz")
        mp = os.path.join(CORE, "results", t, f"{t}_map.json")
        if not os.path.isfile(p):
            raise SystemExit(f"required ensemble member field is missing: {p}")
        if not os.path.isfile(mp):
            raise SystemExit(f"required ensemble member metadata is missing: {mp}")
        meta = json.load(open(mp))
        config = meta.get("provenance", {}).get("config", {})
        expected = {
            "pde": "burgers", "arch": "mlp", "seed": seed,
            "nu": 0.01 / math.pi, "Mx": 20, "Mt": 20,
            "hidden": 64, "fusion": 128, "d": 128, "d_aug": 129,
            "N_f": 5000, "N_u": 1000, "N_b": 200,
            "ic_weight": 10.0, "bc_weight": 10.0,
            "adam_lr": 1e-3, "adam_iters": 5000, "lm_iters": 50,
            "eval_nx": 400, "eval_nt": 200, "smoke": 0,
        }
        if meta.get("tag") != t or meta.get("solve_gate", {}).get("passed") is not True:
            raise SystemExit(f"{t}: metadata does not identify an admitted run")
        if not all(config.get(key) == value for key, value in expected.items()):
            raise SystemExit(f"{t}: metadata does not match the declared ensemble configuration")
        inputs[f"results/{t}/{t}_fields.npz"] = sha256(p)
        inputs[f"results/{t}/{t}_map.json"] = sha256(mp)
        try:
            with np.load(p) as z:
                required = {"u_pred", "u_exact", "grad_abs", "xs", "ts"}
                if not required.issubset(z.files):
                    raise ValueError(f"missing {sorted(required - set(z.files))}")
                pred = np.asarray(z["u_pred"]).ravel()
                ue, g = np.asarray(z["u_exact"]).ravel(), np.asarray(z["grad_abs"]).ravel()
                xs, ts = np.asarray(z["xs"]).ravel(), np.asarray(z["ts"]).ravel()
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"{t}: cannot read ensemble member field: {exc}") from exc
        if (len(xs) != 400 or len(ts) != 200 or
                len(pred) != len(ue) or len(ue) != len(g) or len(g) != 80_000 or
                not np.all(np.isfinite(pred)) or not np.all(np.isfinite(ue)) or
                not np.all(np.isfinite(g)) or np.any(g < 0)):
            raise SystemExit(f"{t}: malformed/non-finite 400x200 field artifact")
        U.append(pred)
        if ref is None:
            ref, ga, xs0, ts0 = ue, g, xs, ts
        else:
            # the members must share a reference and a partition, or the ensemble is not one
            if not np.array_equal(ref, ue):
                raise SystemExit(f"{t}: u_exact differs byte-for-byte from seed 0's")
            if not np.array_equal(ga, g):
                raise SystemExit(f"{t}: grad_abs differs byte-for-byte from seed 0's")
            if not np.array_equal(xs0, xs) or not np.array_equal(ts0, ts):
                raise SystemExit(f"{t}: evaluation grid differs byte-for-byte from seed 0's")
    rebuilt_ga = np.abs(np.gradient(ref.reshape(len(ts0), len(xs0)),
                                    xs0[1] - xs0[0], axis=1)).ravel()
    if not np.allclose(ga, rebuilt_ga, rtol=1e-12, atol=1e-14):
        raise SystemExit("stored grad_abs does not reproduce from u_exact and the stored grid")
    U = np.asarray(U)
    S = ga >= np.percentile(ga, 95)
    B = ga < np.percentile(ga, 70)
    if (np.count_nonzero(ga == np.percentile(ga, 95)) or
            np.count_nonzero(ga == np.percentile(ga, 70)) or
            int(S.sum()) != 4000 or int(B.sum()) != 56000):
        raise SystemExit("finite-grid percentile regions do not have the declared 5%/70% sizes")
    sig = U.std(axis=0, ddof=0)                 # divisor K, per app:conventions
    err = np.abs(U.mean(axis=0) - ref)
    if (not np.all(np.isfinite(sig)) or np.any(sig <= 0) or
            not np.all(np.isfinite(err)) or np.any(err < 0)):
        raise SystemExit("ensemble spread must be finite and strictly positive at every point")
    z = err / sig
    reg = lambda m: {"meanZ": float(z[m].mean()),
                     "cov68": float((z[m] <= 1.0).mean()),
                     "cov95": float((z[m] <= 1.96).mean()),
                     "mean_sigma": float(sig[m].mean())}
    r_rms = float(np.sqrt((sig[S] ** 2).mean() / (sig[B] ** 2).mean()))
    r_tar = float(np.sqrt((err[S] ** 2).mean() / (err[B] ** 2).mean()))
    return {"schema": SCHEMA, "tag": "burgers_mlp_ensemble",
            "n_members": len(TAGS), "members": TAGS,
            "sigma_rms_ratio_between_basin": r_rms,
            "target_rms_ratio_ensemble_mean": r_tar,
            "ratio_gap_ensemble_mean": r_tar / r_rms,
            "sigma_ratio_between_basin": float(sig[S].mean() / sig[B].mean()),
            "shock": reg(S), "smooth": reg(B),
            "inputs_sha256": inputs,
            "code_sha256": sha256(__file__),
            "estimator": "std across members with divisor K (ddof=0); error is |member mean - reference|",
            "note": "deep ensemble of the 8 admitted burgers_mlp seeds; the legacy "
                    "*_between_basin key names do not assert distinct basin membership; "
                    "R_rms and R_tar are "
                    "computed from the ensemble spread and ensemble-mean error field on the "
                    "same RMS scale; sigma_ratio_between_basin is secondary mean width"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = compute()
    get = lambda k: d[k] if "." not in k else d[k.split(".")[0]][k.split(".")[1]]
    bad = []
    print(f"  {'quantity':32} {'computed':>12}   printed")
    for k, (want, dp) in PRINTED.items():
        got = get(k)
        ok = round(got, dp) == want
        bad += [] if ok else [f"{k}: computed {got:.6g} -> {round(got, dp)}, paper prints {want}"]
        print(f"  {k:32} {got:12.6g}   {want}   {'OK' if ok else 'MISMATCH'}")
    if a.check:
        bad.extend(check_stored(d))
    if bad:
        print("\n  MISMATCH:")
        for b in bad:
            print("    " + b)
        return 1
    if a.check:
        print("\nCHECK OK -- stored JSON and every printed value of sec:ensemble "
              "reproduce from the 8 fields.npz")
        return 0
    d["provenance"] = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "command": "analysis/ensemble_calibration.py",
                       "python": platform.python_version(), "numpy": np.__version__,
                       "reconstructs": "the pre-consolidation step3a artifact, whose writer was "
                                       "not in the tree; values agree at the printed precision"}
    with open(OUT + ".partial", "w") as f:
        json.dump(d, f, indent=1, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(OUT + ".partial", OUT)
    print(f"\n  wrote {os.path.relpath(OUT, CORE)}")
    return 0


def check_stored(expected):
    """Require the release JSON to be the current schema and current scientific payload."""
    if not os.path.exists(OUT):
        return [f"stored artifact is missing: {OUT}"]
    try:
        stored = json.load(open(OUT))
    except (OSError, ValueError) as exc:
        return [f"cannot read stored artifact {OUT}: {exc}"]
    if stored.get("schema") != SCHEMA:
        return [f"{OUT}: schema {stored.get('schema')!r}, expected {SCHEMA!r}"]
    # Provenance contains a timestamp and environment; the remaining payload must agree exactly
    # up to floating-point roundoff with a fresh computation.
    payload = {k: v for k, v in stored.items() if k != "provenance"}
    bad = []
    if set(payload) != set(expected):
        bad.append(
            f"{OUT}: keys differ; missing={sorted(set(expected)-set(payload))}, "
            f"extra={sorted(set(payload)-set(expected))}"
        )

    def compare(path, got, want):
        if isinstance(want, dict):
            if not isinstance(got, dict):
                bad.append(f"{OUT}: {path} has type {type(got).__name__}, expected object")
                return
            if set(got) != set(want):
                bad.append(f"{OUT}: {path} keys differ")
            for key in sorted(set(got) & set(want)):
                compare(f"{path}.{key}", got[key], want[key])
        elif isinstance(want, list):
            if got != want:
                bad.append(f"{OUT}: {path} stored={got!r}, live={want!r}")
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            if not np.isclose(got, want, rtol=1e-12, atol=1e-15):
                bad.append(f"{OUT}: {path} stored={got!r}, live={want!r}")
        elif got != want:
            bad.append(f"{OUT}: {path} stored={got!r}, live={want!r}")

    compare("payload", payload, expected)
    if "provenance" not in stored:
        bad.append(f"{OUT}: provenance block is missing")
    return bad


if __name__ == "__main__":
    sys.exit(main())
