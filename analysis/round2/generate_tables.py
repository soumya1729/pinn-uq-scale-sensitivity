"""Regenerate five all-17 manuscript tables from frozen archives and review CSVs.

Requires NumPy. This reads saved outputs only, with no checkpoint evaluation,
Jacobian, evidence search, new scale, or change to the frozen producer scripts.

From the paper root:
    python analysis/round2/generate_tables.py --production /path/to/production

--check compares the existing table text without writing any result.
"""
import argparse
import csv
from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path

import numpy as np

FAMILIES = ("burgers_mlp", "burgers_fourier", "allen_cahn_mlp", "allen_cahn_fourier")
NAMES = dict(zip(FAMILIES, ("Burgers--MLP", "Burgers--Fourier",
                          "Allen--Cahn--MLP", "Allen--Cahn--Fourier")))
PREFIXES = dict(zip(FAMILIES, ("BM", "BF", "AM", "AF")))
SEEDS = dict(zip(FAMILIES, (tuple(range(8)), (0, 2, 6), (0, 1, 3), (0, 1, 3))))
TAGS = tuple(f"{family}_seed{seed}" for family in FAMILIES for seed in SEEDS[family])
TEMPLATES = {'round2_evidence_table.tex': {'caption': 'Original and extended best tested evidence settings, '
                                          'and the six-case sensitivity relative to the extended '
                                          'setting. BM/BF denote Burgers MLP/Fourier and AM/AF '
                                          'Allen--Cahn MLP/Fourier; the suffix is the admitted '
                                          'seed. The first two columns are $\\alpha/\\alpha_0$. '
                                          'For BM, BF and AM, original noise is $10^{-4}\\widehat '
                                          'v$ and extended noise is $10^{-8}\\widehat v$; all are '
                                          'lower noise boundaries. AF settings are interior and '
                                          'unchanged by extension. $\\delta s$ is the $1\\times$ '
                                          'envelope; its squared term alone is not a uniform '
                                          'eigenvalue error bound.',
                               'label': 'tab:r2-evidence',
                               'columns': 'lrrrrrr',
                               'heading': 'run & $a_{\\rm orig}$ & $a_{\\rm ext}$ & $v_{\\rm '
                                          'ext}/\\widehat v$ & $\\gamma_{\\rm ext}$ & perturbed '
                                          '$\\gamma/\\gamma_{\\rm ext}$ & $\\delta '
                                          's^2/\\gamma_{\\rm ext}$',
                               'size': 'footnotesize'},
 'round2_centre_table.tex': {'caption': 'Full-grid RMS error at the two centres. Entries are '
                                        'unweighted means of per-run RMS errors, reconstructed '
                                        'with the archived regional point counts; the trained '
                                        'error is independent of scale. Baseline uses '
                                        '$(\\alpha_0,1)$ and conditional MLE uses '
                                        "$(\\alpha_0,\\widehat v)$. ``Worse'' counts affine errors "
                                        'exceeding the corresponding trained error, retaining the '
                                        'original admitted cohort.',
                             'label': 'tab:r2-centres',
                             'columns': 'lrrrrrr',
                             'heading': 'family & $n$ & trained & affine baseline & worse & affine '
                                        'MLE & worse',
                             'size': 'small'},
 'round2_oracle_table.tex': {'caption': "Across-run sharp-region coverage ranges for trained and zero-mean-prior affine centres at each run's "
                                        'closest saved contrast setting, chosen separately for '
                                        'each centre and subset. A closest sample need not have '
                                        '$\\rho\\approx1$, so these rows do not generally evaluate '
                                        'coverage after RMS-contrast matching. At each saved '
                                        '$\\gamma$, a higher empirical bulk-$0.68$ quantile '
                                        'supplies the width multiplier; selection minimises '
                                        '$|\\log\\rho|$ and the lowest saved $\\gamma$ breaks '
                                        'exact ties. These reference-error oracles use all $8$ '
                                        'Burgers--MLP or $3$ other admitted runs, including runs '
                                        "without a crossing. ``Post'' uses the unchanged global "
                                        'masks intersected with Burgers $t\\ge1/\\pi$. Bulk '
                                        'one-sigma coverage is about $0.68002$; neither nominal '
                                        'sharp coverage nor an exact root is enforced.',
                             'label': 'tab:r2-oracle',
                             'columns': 'llrrrr',
                             'heading': 'family & subset & trained $68\\%$ & affine $68\\%$ & '
                                        'trained $95\\%$ & affine $95\\%$',
                             'size': 'footnotesize'},
 'round2_limits_table.tex': {'caption': 'Full-time regional RMS tangent-norm ratios, QR-complement '
                                        'ratios and measured contrast ranges across every saved '
                                        '$\\gamma$ and admitted run in each family. The QR column '
                                        'equals the whole-kernel weak limit when $J$ has full row '
                                        'rank. Endpoint ratios are not global contrast bounds; the '
                                        'final column is a finite sampled range shared by all predictive '
                                        'centres.',
                             'label': 'tab:r2-limits',
                             'columns': 'lrrr',
                             'heading': 'family & strong-$\\gamma$ ratio & QR-complement ratio & '
                                        'sampled $\\Rrms$',
                             'size': 'small'},
 'round2_changed_bounds_table.tex': {'caption': 'Changed-scale two-block brackets from the saved '
                                                'direct-$J$ regional energies, at '
                                                '$\\epsilon=\\gamma$, in $\\sigma$-ratio units. '
                                                'Each displayed interval encloses the union of '
                                                "that family's run-specific returned-spectrum "
                                                'brackets; endpoints are rounded outward. These '
                                                'arithmetic evaluations enclose their returned '
                                                'modal contrasts but do not certify weak-mode '
                                                'forward accuracy. Their breadth prevents using '
                                                'them alone to establish the realised-error '
                                                'deficit.',
                                     'label': 'tab:r2-changed-bounds',
                                     'columns': 'lrr',
                                     'heading': 'family & conditional MLE & extended evidence',
                                     'size': 'small'}}


def family_of(tag):
    return tag.rsplit("_seed", 1)[0]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, digits=3):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite table value: {value}")
    if value == 0:
        return "0"
    if .001 <= abs(value) < 1000:
        rendered = format(value, f".{digits}g")
        return format(Decimal(rendered), "f") if "e" in rendered else rendered
    mantissa, exponent = format(value, f".{digits-1}e").split("e")
    return mantissa.rstrip("0").rstrip(".") + r"\!\times\!10^{" + str(int(exponent)) + "}"


def interval(lower, upper, digits=3):
    return "[" + number(lower, digits) + r",\," + number(upper, digits) + "]"


def math(value):
    return "$" + str(value) + "$"


def row(*values):
    return " & ".join(values) + r" \\"


def outward(value, upper, digits=3):
    exact = Decimal.from_float(float(value))
    quantum = Decimal("1e" + str(exact.adjusted()-digits+1))
    mode = ROUND_CEILING if upper else ROUND_FLOOR
    return float(exact.quantize(quantum, rounding=mode))


def render(name, rows):
    t = TEMPLATES[name]
    return (
        "% Derived only from frozen production/review archives; no new grid or checkpoint evaluation.\n"
        + r"\begin{table}[t]" + "\n" + r"\centering" + "\\" + t["size"]
        + r"\setlength{\tabcolsep}{4pt}" + "\n\n\n"
        + r"\begin{tabular}{@{}" + t["columns"] + "@{}}\n"
        + r"\toprule" + "\n" + t["heading"] + r" \\" + "\n" + r"\midrule" + "\n"
        + "\n".join(rows) + "\n" + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\caption{" + t["caption"] + "}\n" + r"\label{" + t["label"] + "}\n"
        + r"\end{table}" + "\n"
    )


def evidence_rows(best, perturbations, spectra):
    rows = []
    for tag in TAGS:
        o, e = best[(tag, "original")], best[(tag, "extended")]
        a0, vhat = float(spectra[tag]["alpha0"]), float(spectra[tag]["conditional_noise_mle"])
        cases = perturbations[tag]
        if len(cases) != 6:
            raise ValueError(f"{tag}: expected six archived sensitivity cases")
        gamma = float(e["effective_precision"])
        ratios = [float(r["gamma_ratio_to_extended"]) for r in cases]
        ds = float(next(r for r in cases if r["case"] == "increase_1x")["singular_value_envelope"])
        values = (number(float(o["alpha"])/a0), number(float(e["alpha"])/a0),
                  number(float(e["noise_variance"])/vhat), number(gamma),
                  interval(min(ratios), max(ratios)), number(ds**2/gamma))
        run = PREFIXES[family_of(tag)] + tag.rsplit("seed", 1)[1]
        rows.append(row(run, *(math(v) for v in values)))
    return rows


def centre_rows(comparisons):
    selected = defaultdict(list)
    for r in comparisons:
        if r["subset"] != "full":
            continue
        if r["setting"] == "conditional_noise_mle":
            setting = "mle"
        elif (r["setting"] == "prior_sweep" and float(r["alpha_multiplier"]) == 1
              and float(r["noise_variance"]) == 1):
            setting = "baseline"
        else:
            continue
        selected[(r["tag"], setting)].append(r)
    rms = {}
    for key, regions in selected.items():
        if sorted(r["region"] for r in regions) != ["bulk", "sharp", "transition"]:
            raise ValueError(f"{key}: expected three disjoint full-grid regions")
        counts = np.array([int(r["count"]) for r in regions])
        rms[key] = tuple(float(np.sqrt(np.sum(
            counts*np.array([float(r[column]) for r in regions])**2
        )/counts.sum())) for column in ("trained_error_rms", "affine_error_rms"))
    rows = []
    for family in FAMILIES:
        tags = [tag for tag in TAGS if family_of(tag) == family]
        b, m = [rms[(tag, "baseline")] for tag in tags], [rms[(tag, "mle")] for tag in tags]
        n = len(tags)
        rows.append(row(
            NAMES[family], math(n), math(number(np.mean([r[0] for r in b]), 4)),
            math(number(np.mean([r[1] for r in b]), 4)), math(f"{sum(r[1]>r[0] for r in b)}/{n}"),
            math(number(np.mean([r[1] for r in m]), 4)), math(f"{sum(r[1]>r[0] for r in m)}/{n}"),
        ))
    return rows


def oracle_rows(nearest):
    rows = []
    for family in FAMILIES:
        for subset in ("full", "post_formation"):
            group = [r for r in nearest if family_of(r["tag"]) == family and r["subset"] == subset]
            if not group:
                continue
            values = []
            for key, centre in (("sharp_oracle68", "trained"), ("sharp_oracle68", "affine"),
                                ("sharp_oracle95", "trained"), ("sharp_oracle95", "affine")):
                samples = [float(r[key]) for r in group if r["centre"] == centre]
                if len(samples) != len(SEEDS[family]):
                    raise ValueError(f"{family}/{subset}/{centre}: incomplete nearest table")
                values.append(math(interval(min(samples), max(samples), 4)))
            rows.append(row(NAMES[family], "full" if subset == "full" else "post", *values))
    return rows


def limit_rows(limits, nearest):
    rows = []
    for family in FAMILIES:
        endpoints = [r for r in limits if family_of(r["tag"]) == family and r["subset"] == "full"]
        curves = [r for r in nearest if family_of(r["tag"]) == family
                  and r["subset"] == "full" and r["centre"] == "trained"]
        values = []
        for key in ("strong_gamma_tangent_norm_contrast", "qr_complement_contrast"):
            samples = [float(r[key]) for r in endpoints]
            values.append(math(interval(min(samples), max(samples), 4)))
        values.append(math(interval(min(float(r["sampled_contrast_min"]) for r in curves),
                                    max(float(r["sampled_contrast_max"]) for r in curves), 4)))
        rows.append(row(NAMES[family], *values))
    return rows


def changed_bound_rows(spectra, best):
    """Two-block theorem at epsilon=gamma for previously saved scale choices.

    Sloppy loading includes the QR complement and returned lambda <= gamma.
    This numerical split does not assert exact rank or kernel classification.
    Common noise variance cancels in the spectral bracket and contrast.
    """
    bounds = defaultdict(list)
    for tag in TAGS:
        a = spectra[tag]
        lam = np.asarray(a["eigenvalues"], dtype=float)
        labels = list(a["region_labels"])
        b, s = labels.index("full/bulk"), labels.index("full/sharp")
        qb, qs = a["mean_mode_energy"][[b, s]]
        db, ds = a["mean_complement_energy"][[b, s]]
        choices = (("MLE", float(a["alpha0"])*float(a["conditional_noise_mle"])),
                   ("extended", float(best[(tag, "extended")]["effective_precision"])))
        for setting, gamma in choices:
            sloppy = lam <= gamma
            bsl, ssl = db+qb[sloppy].sum(), ds+qs[sloppy].sum()
            bst, sst = qb[~sloppy].sum(), qs[~sloppy].sum()
            mb = bsl/(2*gamma)+bst/(gamma+lam.max())
            ms = ssl/(2*gamma)+sst/(gamma+lam.max())
            Mb, Ms = bsl/gamma+bst/(2*gamma), ssl/gamma+sst/(2*gamma)
            lo, hi = float(np.sqrt(ms/Mb)), float(np.sqrt(Ms/mb))
            returned = float(np.sqrt((ds/gamma+np.sum(qs/(lam+gamma))) /
                                     (db/gamma+np.sum(qb/(lam+gamma)))))
            if not (0 < lo <= returned <= hi and np.isfinite(hi)):
                raise ValueError(f"{tag}/{setting}: invalid returned-spectrum bracket")
            bounds[(family_of(tag), setting)].append((lo, hi))
    rows = []
    for family in FAMILIES:
        values = []
        for setting in ("MLE", "extended"):
            group = bounds[(family, setting)]
            lo = outward(min(r[0] for r in group), upper=False)
            hi = outward(max(r[1] for r in group), upper=True)
            values.append(math(interval(lo, hi)))
        rows.append(row(NAMES[family], *values))
    return rows


def generate(production, review):
    best = {(r["tag"], r["grid"]): r for r in read_csv(review/"evidence_best.csv")}
    perturbations = defaultdict(list)
    for r in read_csv(review/"evidence_perturbations.csv"):
        perturbations[r["tag"]].append(r)
    nearest = read_csv(review/"nearest_sampled_agreement.csv")
    spectra = {}
    for tag in TAGS:
        with np.load(production/f"{tag}_round2_spectral.npz", allow_pickle=False) as data:
            spectra[tag] = {key: data[key] for key in (
                "alpha0", "conditional_noise_mle", "eigenvalues", "region_labels",
                "mean_mode_energy", "mean_complement_energy",
            )}
    bodies = {
        "round2_evidence_table.tex": evidence_rows(best, perturbations, spectra),
        "round2_centre_table.tex": centre_rows(read_csv(review/"n3_centre_comparison.csv")),
        "round2_oracle_table.tex": oracle_rows(nearest),
        "round2_limits_table.tex": limit_rows(read_csv(review/"spectral_limits.csv"), nearest),
        "round2_changed_bounds_table.tex": changed_bound_rows(spectra, best),
    }
    return {name: render(name, rows) for name, rows in bodies.items()}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True,
                        help="Folder containing the frozen all-17 spectral NPZ archives.")
    parser.add_argument("--review", type=Path, default=root/"analysis/round2/outputs/production_review")
    parser.add_argument("--out", type=Path, default=root/"generated")
    parser.add_argument("--check", action="store_true",
                        help="Compare table text without writing or changing any result.")
    args = parser.parse_args()
    outputs = generate(args.production, args.review)
    if args.check:
        bad = [name for name, text in outputs.items() if not (args.out/name).exists()
               or (args.out/name).read_text(encoding="utf-8") != text]
        if bad:
            raise SystemExit("Table mismatch: " + ", ".join(bad))
        print("All five tables reproduce the existing text; no files changed.")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, text in outputs.items():
            (args.out/name).write_text(text, encoding="utf-8")
        print("Wrote five tables from the existing saved outputs.")


if __name__ == "__main__":
    main()
