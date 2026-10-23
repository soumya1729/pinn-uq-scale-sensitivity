"""Report completed Burgers-MLP N1 outputs; consolidates the two ad hoc summaries."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def report(path):
    settings = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            settings[(row["setting"], row["effective_precision"])][row["region"]] = row
    sweep = [regions for (name, _), regions in settings.items() if name == "prior_sweep"]
    picks = {
        "operating": next(r for r in sweep if float(r["sharp"]["alpha_multiplier"]) == 1.0),
        "noise-MLE": next(r for (name, _), r in settings.items() if name == "conditional_noise_mle"),
        "evidence": next(r for (name, _), r in settings.items() if name.startswith("affine_evidence")),
        "sweep-max": max(sweep, key=lambda r: float(r["sharp"]["R_sigma_rms"])),
    }
    print(f'\n{picks["operating"]["sharp"]["tag"]} | R_tar = '
          f'{float(picks["operating"]["sharp"]["R_tar"]):.1f}')
    for name, regions in picks.items():
        sharp, bulk = regions["sharp"], regions["bulk"]
        print(f'   {name:9s} eff={float(sharp["effective_precision"]):8.1e}  '
              f'R_rms={float(sharp["R_sigma_rms"]):6.2f}  rho={float(sharp["rho"]):6.2f}  '
              f'Z_S={float(sharp["mean_Z"]):.3f}  Z_B={float(bulk["mean_Z"]):.3f}')
    print("   Evidence candidate:", picks["evidence"]["sharp"]["setting"])
    with np.load(path.with_suffix(".npz")) as archive:
        evidence = archive["log_evidence"]
        i = np.unravel_index(np.argmax(evidence), evidence.shape)[0]
        alpha = archive["evidence_alpha_grid"][i]
        noise = archive["evidence_noise_grid"][:4]
        differences = np.round(evidence[i, :4] - evidence[i, 0], 2)
    print("   Four smallest tested noise variances:", noise)
    print(f"   Evidence at four smallest tested variances, fixed alpha={alpha:.8g}; "
          "relative to the smallest:", differences)
    for name in ("operating", "noise-MLE"):
        for region in ("bulk", "sharp"):
            row = picks[name][region]
            print(f'   {name:9s} {region:5s} meanZ={float(row["mean_Z"]):6.3f}  '
                  f'cov68={float(row["coverage68"]):.3f}  cov95={float(row["coverage95"]):.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "outputs_cpu")
    args = parser.parse_args()
    paths = [p for p in sorted(args.directory.glob("burgers_mlp_seed*_n1.csv"))
             if p.with_suffix(".json").is_file()]
    print(f"Completed Burgers-MLP summaries found: {len(paths)}")
    print("Input directory:", args.directory.resolve())
    print("sweep-max is the best sampled curve point, including matched MLE/evidence points.")
    print("Coverage uses intervals centred on the saved trained prediction.")
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()
