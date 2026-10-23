"""solve_gate_table.py -- persist the complete solve-acceptance inventory behind Table
`tab:rejected` for EVERY configuration attempted: admitted, rejected, and the
convection-diffusion control.

WHY THIS FILE EXISTS. The per-run verdicts have a producer (`train_map.py` writes each run's
`solve_gate` block at admission time). The TABLE did not: its min/max/median and tau-multiples
were recomputed inside `submissions/jmlr_ceiling/tools/gate_numbers.py` at gate time and
persisted nowhere, so there was no artifact a referee could be handed and no way to see the
inventory without reading the gate. This is the same gap that `temporal_per_seed.py` closes for
Table `tab:temporal`.

Building it surfaced two things the gate could not, because a checker only inspects what it is
pointed at:

  1. The Nagumo bump arm lives at results/_nagumo_round1_gauss/nagumo_mlp_seed*/, one level
     deeper than the gate's results/*/*_map.json glob, so five rejected runs sat outside every
     check while the prose named two Nagumo initial conditions. A glob is a silent filter.
  2. The convection-diffusion control varies pulse width and diffusivity together at ONE seed,
     not as a seed sweep or one-factor diffusivity sweep. Three of four MLP attempts are admitted;
     a separate Fourier attempt at the narrowest setting is also rejected and is retained as an
     auxiliary inventory row rather than pooled into the MLP denominator.

WHAT THE CONTROL SHOWS, and why it belongs beside the four sharp configurations. CD has no
nonlinear self-organised shock. Across its three admitted paired MLP operating points at one seed, the primary
field-derived RMS uncertainty contrast is 1.09--1.33 against R_tar 0.78--1.00, i.e. an RMS-matched
gap of 0.65--0.90. The secondary mean-width contrast is 1.11--1.36. Thus the large allocation
deficit studied in the nonlinear sharp-feature rows is absent in this control under the same
evaluation and partition protocol. This is a descriptive control comparison, not proof that
nonlinearity alone causes the deficit.

CONVENTIONS. The reported tau_bulk is 2% for Burgers, Fisher-KPP, Nagumo, Kuramoto-Sivashinsky
and the CD control; 3% for Allen-Cahn (`j_setting.tex`, eq:gate). The KS records used 5% at
production time; 2% is the common retrospective comparison threshold for those two rows, and
every KS run is rejected at either threshold. Ranges and medians are over the runs
stated in the `runs` column -- ALL runs attempted for a rejected family, ADMITTED runs only for
an admitted configuration, which is the convention Table `tab:rejected` prints. Dispersions
elsewhere in the paper are population (ddof=0); no dispersion is quoted here.

    python analysis/solve_gate_table.py           # table + calibration_figures/solve_gate_table.json
    python analysis/solve_gate_table.py --check   # read-only: assert every cell + stored JSON

Pure numpy/json, no GPU, ~5 s. Reads only shipped MAP records:
    results/**/{TAG}_map.json -> solve_gate.bulk_rel_l2, solve_gate.passed

No uncertainty contrast is copied into this artifact.  Table `tab:ceiling` is rebuilt from
pointwise fields by `ceiling_table.py`; keeping that separate prevents historical mean-width or
cross-metric records from being mistaken for the primary RMS statistic.
"""
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "calibration_figures")
SCHEMA = "solve_gate_inventory/3"

# label, globs of *_map.json, tau_bulk, admitted-only?
FAMILIES = [
    ("Fisher--KPP",                      ["results/fisher_kpp_mlp_seed*/*_map.json"], 0.02, False),
    ("Nagumo, both initial conditions",  ["results/nagumo_mlp_seed*/*_map.json",
                                          "results/_nagumo_round1_gauss/*/*_map.json"], 0.02, False),
    ("Kuramoto--Sivashinsky, T=100",     ["results/ks_T100_mlp_seed*/*_map.json"], 0.02, False),
    ("Kuramoto--Sivashinsky, T=10",      ["results/ks_T10_mlp_seed*/*_map.json"], 0.02, False),
    ("Burgers, MLP",                     ["results/burgers_mlp_seed*/*_map.json"], 0.02, True),
    ("Burgers, Fourier",                 ["results/burgers_fourier_seed*/*_map.json"], 0.02, True),
    ("Allen--Cahn, MLP",                 ["results/allen_cahn_mlp_seed*/*_map.json"], 0.03, True),
    ("Allen--Cahn, Fourier",             ["results/allen_cahn_fourier_seed*/*_map.json"], 0.03, True),
    ("convection--diffusion, MLP (control)",
                                         ["results/cd_sweep/mlp_*/*_map.json",
                                          "results/convection_diffusion_mlp_seed0/*_map.json"], 0.02, True),
    ("convection--diffusion, Fourier (rejected control)",
                                         ["results/cd_sweep/fou_*/*_map.json"], 0.02, False),
]


def _seed_paths(directory_prefix, tag_prefix, seeds):
    return {f"results/{directory_prefix}_seed{s}/{tag_prefix}_seed{s}_map.json" for s in seeds}


# Counts alone do not close a population: a missing declared run and an accidental extra run can
# cancel. Every family therefore has an exact, human-readable attempted-run inventory.
EXPECTED_PATHS = {
    "Fisher--KPP": _seed_paths("fisher_kpp_mlp", "fisher_kpp_mlp", range(5)),
    "Nagumo, both initial conditions": (
        _seed_paths("nagumo_mlp", "nagumo_mlp", range(5)) |
        {f"results/_nagumo_round1_gauss/nagumo_mlp_seed{s}/nagumo_mlp_seed{s}_map.json"
         for s in range(5)}),
    "Kuramoto--Sivashinsky, T=100":
        _seed_paths("ks_T100_mlp", "kuramoto_sivashinsky_mlp", range(5)),
    "Kuramoto--Sivashinsky, T=10":
        _seed_paths("ks_T10_mlp", "kuramoto_sivashinsky_mlp", range(5)),
    "Burgers, MLP": _seed_paths("burgers_mlp", "burgers_mlp", range(8)),
    "Burgers, Fourier": _seed_paths("burgers_fourier", "burgers_fourier", range(8)),
    "Allen--Cahn, MLP": _seed_paths("allen_cahn_mlp", "allen_cahn_mlp", range(5)),
    "Allen--Cahn, Fourier":
        _seed_paths("allen_cahn_fourier", "allen_cahn_fourier", range(4)),
    "convection--diffusion, MLP (control)": {
        "results/cd_sweep/mlp_d09_n06/convection_diffusion_mlp_seed0_map.json",
        "results/cd_sweep/mlp_d11_n08/convection_diffusion_mlp_seed0_map.json",
        "results/cd_sweep/mlp_d13_n10/convection_diffusion_mlp_seed0_map.json",
        "results/convection_diffusion_mlp_seed0/convection_diffusion_mlp_seed0_map.json",
    },
    "convection--diffusion, Fourier (rejected control)": {
        "results/cd_sweep/fou_d06_n01/convection_diffusion_fourier_seed0_map.json",
    },
}

# Pin the accepted identities as well as the attempted identities. In particular the
# auxiliary Fourier control must remain a rejected attempt, not an unvalidated extra row.
EXPECTED_ADMITTED_PATHS = {label: set() for label in EXPECTED_PATHS}
EXPECTED_ADMITTED_PATHS.update({
    "Burgers, MLP": _seed_paths("burgers_mlp", "burgers_mlp", range(8)),
    "Burgers, Fourier": _seed_paths("burgers_fourier", "burgers_fourier", (0, 2, 6)),
    "Allen--Cahn, MLP": _seed_paths("allen_cahn_mlp", "allen_cahn_mlp", (0, 1, 3)),
    "Allen--Cahn, Fourier": _seed_paths("allen_cahn_fourier", "allen_cahn_fourier", (0, 1, 3)),
    "convection--diffusion, MLP (control)": {
        "results/cd_sweep/mlp_d09_n06/convection_diffusion_mlp_seed0_map.json",
        "results/cd_sweep/mlp_d11_n08/convection_diffusion_mlp_seed0_map.json",
        "results/cd_sweep/mlp_d13_n10/convection_diffusion_mlp_seed0_map.json",
    },
})

# Table tab:rejected as printed, for --check: (runs, min%, max%, median%, tau_lo, tau_hi)
PAPER_REJECTED = {
    "Fisher--KPP":                     (5,  "81.7", "170.5", "100.0", "41", "85"),
    "Nagumo, both initial conditions": (10, "75.9", "90.3",  "80.5",  "38", "45"),
    "Kuramoto--Sivashinsky, T=100":    (5,  "98.2", "103.8", "100.4", "49", "52"),
    "Kuramoto--Sivashinsky, T=10":     (5,  "9.9",  "17.0",  "12.8",  "5",  "9"),
}
PAPER_ADMITTED = {  # (admitted, attempted, min%, max%, median%) -- 2dp
    "Burgers, MLP":                    (8, 8, "0.05", "0.40", "0.08"),
    "Burgers, Fourier":                (3, 8, "0.37", "1.48", "0.78"),
    "Allen--Cahn, MLP":                (3, 5, "1.39", "2.33", "2.33"),
    "Allen--Cahn, Fourier":            (3, 4, "1.64", "2.60", "2.34"),
    "convection--diffusion, MLP (control)": (3, 4, "0.19", "0.31", "0.21"),
}


def _runs(globs, expected_stored_tau=None):
    out, seen = [], set()
    for g in globs:
        for p in sorted(glob.glob(os.path.join(ROOT, g))):
            p = os.path.abspath(p)
            if p in seen:
                continue
            seen.add(p)
            try:
                with open(p, encoding="utf-8") as handle:
                    record = json.load(handle)
            except Exception as exc:
                raise ValueError(f"{p}: unreadable attempted-run metadata: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{p}: attempted-run metadata must be a JSON object")
            sg = record.get("solve_gate")
            if (not isinstance(sg, dict) or "bulk_rel_l2" not in sg or
                    not isinstance(sg.get("passed"), bool)):
                raise ValueError(f"{p}: solve-gate value or Boolean decision is absent")
            value = sg["bulk_rel_l2"]
            if type(value) not in (int, float):
                raise ValueError(f"{p}: invalid bulk relative error {value!r}")
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{p}: bulk relative error is outside the numeric range") from exc
            if not math.isfinite(value) or value < 0 or not math.isfinite(100.0 * value):
                raise ValueError(f"{p}: invalid bulk relative error {value!r}")
            stored_tau = sg.get("rel_l2_threshold")
            if type(stored_tau) not in (int, float):
                raise ValueError(f"{p}: stored solve threshold is absent or malformed")
            try:
                stored_tau = float(stored_tau)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{p}: stored solve threshold is outside the numeric range") from exc
            if (not math.isfinite(stored_tau) or stored_tau <= 0 or
                    (expected_stored_tau is not None and stored_tau != expected_stored_tau) or
                    sg["passed"] != (value < stored_tau)):
                raise ValueError(f"{p}: stored solve threshold or admission decision is inconsistent")
            out.append({"path": os.path.relpath(p, ROOT).replace(os.sep, "/"),
                        "bulk_rel_l2_pct": 100.0 * value,
                        "passed": sg["passed"]})
    return out


def main():
    unknown = [arg for arg in sys.argv[1:] if arg != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown)}")
    check = "--check" in sys.argv[1:]
    families, bad = {}, []

    print(f"{'configuration':<34}{'runs':>5}{'adm':>5}{'att':>5}"
          f"{'min%':>9}{'max%':>9}{'median%':>9}{'  vs tau':>10}")
    for label, globs, tau, admitted_only in FAMILIES:
        try:
            # Preserve the historical KS setting while reporting its rejection distance
            # against the common 2% comparison. Both admission decisions must agree.
            stored_tau = 0.05 if label.startswith("Kuramoto--Sivashinsky") else tau
            runs = _runs(globs, expected_stored_tau=stored_tau)
        except ValueError as exc:
            bad.append(str(exc))
            continue
        if not runs:
            print(f"{label:<34}  (no runs found)")
            bad.append(f"{label}: all declared attempted-run records are absent")
            continue
        got_paths = {run["path"] for run in runs}
        if got_paths != EXPECTED_PATHS[label]:
            bad.append(f"{label}: discovered attempted-run paths do not equal the declared "
                       f"inventory; missing={sorted(EXPECTED_PATHS[label] - got_paths)}, "
                       f"unexpected={sorted(got_paths - EXPECTED_PATHS[label])}")
            continue
        admitted_paths = {run["path"] for run in runs if run["passed"]}
        if admitted_paths != EXPECTED_ADMITTED_PATHS[label]:
            bad.append(f"{label}: admitted paths do not equal the declared inventory; "
                       f"found={sorted(admitted_paths)}, "
                       f"expected={sorted(EXPECTED_ADMITTED_PATHS[label])}")
            continue
        inconsistent = [run["path"] for run in runs
                        if run["passed"] != (run["bulk_rel_l2_pct"] < 100.0 * tau)]
        if inconsistent:
            bad.append(f"{label}: stored decisions disagree with the reported "
                       f"{100 * tau:g}% threshold for {inconsistent}")
            continue
        att = len(runs)
        adm = sum(r["passed"] for r in runs)
        use = [r for r in runs if r["passed"]] if admitted_only else runs
        v = sorted(r["bulk_rel_l2_pct"] for r in use)
        if not v:
            bad.append(f"{label}: no runs remain in the declared scored population")
            continue
        # ``statistics.median`` averages the two central observations for an even
        # population.  Selecting v[len(v)//2] would instead report the upper median.
        med = float(statistics.median(v))
        lo, hi = min(v), max(v)
        taus = (f"{lo / (100 * tau):.0f}-{hi / (100 * tau):.0f}x") if not admitted_only else "---"
        fmt = "{:.2f}" if admitted_only else "{:.1f}"
        print(f"{label:<34}{len(v):>5}{adm:>5}{att:>5}"
              f"{fmt.format(lo):>9}{fmt.format(hi):>9}{fmt.format(med):>9}{taus:>10}")

        families[label] = {
            "tau_bulk": tau, "runs_scored": len(v), "admitted": adm, "attempted": att,
            "admitted_only": admitted_only,
            "median_definition": "standard median; mean of central pair for even n",
            "bulk_rel_l2_pct": {"min": lo, "max": hi, "median": med},
            "tau_multiple": {"min": lo / (100 * tau), "max": hi / (100 * tau)},
            "runs": use,
            "reported_in_table": label != "convection--diffusion, Fourier (rejected control)",
        }

        # Completeness is a scientific invariant, not merely a --check presentation test.
        if label in PAPER_REJECTED and att != PAPER_REJECTED[label][0]:
            bad.append(f"{label}: complete rejected inventory requires "
                       f"{PAPER_REJECTED[label][0]} attempts, found {att}")
        if label in PAPER_ADMITTED:
            expected_adm, expected_att = PAPER_ADMITTED[label][:2]
            if (adm, att) != (expected_adm, expected_att):
                bad.append(f"{label}: complete inventory requires {expected_adm}/{expected_att} "
                           f"admitted/attempted, found {adm}/{att}")

        if check and label in PAPER_REJECTED:
            n, mn, mx, md, tl, th = PAPER_REJECTED[label]
            got = (len(v), f"{lo:.1f}", f"{hi:.1f}", f"{med:.1f}",
                   f"{lo / (100 * tau):.0f}", f"{hi / (100 * tau):.0f}")
            if got != (n, mn, mx, md, tl, th):
                bad.append(f"{label}: got {got}, paper {(n, mn, mx, md, tl, th)}")
        if check and label in PAPER_ADMITTED:
            a, t, mn, mx, md = PAPER_ADMITTED[label]
            got = (adm, att, f"{lo:.2f}", f"{hi:.2f}", f"{med:.2f}")
            if got != (a, t, mn, mx, md):
                bad.append(f"{label}: got {got}, paper {(a, t, mn, mx, md)}")

    report = {
        "schema": SCHEMA,
        "statistic": "bulk relative L2 solve error",
        "median_definition": "standard median; mean of central pair for even n",
        "families": families,
    }

    dst = os.path.join(OUT, "solve_gate_table.json")
    if bad:
        print("\nCHECK FAILED" if check else "\nGENERATION FAILED -- no artifact written")
        for b in bad:
            print("  " + b)
        return 1
    if check:
        try:
            with open(dst, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"\nCHECK FAILED -- cannot read {os.path.relpath(dst, ROOT)}: {exc}")
            return 1
        if stored != report:
            print("\nCHECK FAILED -- stored solve-gate summary differs from live artifacts; "
                  "run without --check to regenerate")
            return 1
        print("\nCHECK OK -- every printed cell reproduces and stored JSON matches (read-only)")
        return 0

    os.makedirs(OUT, exist_ok=True)
    partial = dst + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(partial, dst)
    print(f"\nwrote {os.path.relpath(dst, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
