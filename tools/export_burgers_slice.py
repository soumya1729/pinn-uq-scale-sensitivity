"""Rebuild Figure 1's frozen CSV from the saved prediction and raw sigma fields."""
import argparse
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields', type=Path, default=ROOT / 'analysis/n5/input/burgers_mlp_seed2/burgers_mlp_seed2_fields.npz')
    p.add_argument('--sigma', type=Path, default=ROOT / 'analysis/round2/baseline/burgers_mlp_seed2_fullgrid_sigma_raw.npz')
    p.add_argument('--out', type=Path, default=ROOT / 'regenerated/burgers_slice.csv')
    args = p.parse_args()
    with np.load(args.fields, allow_pickle=False) as a, np.load(args.sigma, allow_pickle=False) as b:
        np.testing.assert_array_equal(a['xs'], b['xs'])
        np.testing.assert_array_equal(a['ts'], b['ts'])
        nx, nt = len(a['xs']), len(a['ts'])
        i = int(np.argmin(np.abs(a['ts'] - .5)))
        ix = np.arange(i * nx, (i + 1) * nx)
        e = np.abs(a['u_pred'] - a['u_exact'])[ix]
        sigma = b['sigma'][ix]
        sharp = np.zeros(nx * nt, dtype=bool)
        bulk = np.zeros(nx * nt, dtype=bool)
        sharp[b['shock_idx']] = True
        bulk[b['smooth_idx']] = True
        rows = [(format(x, '.17g'), format(err, '.17g'), format(s, '.17g'), int(sh), int(bu))
                for x, err, s, sh, bu in zip(a['xs'], e, sigma, sharp[ix], bulk[ix])]
        time = float(a['ts'][i])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['x', 'absolute_error', 'sigma', 'sharp', 'bulk'])
        writer.writerows(rows)
    print(f"Saved {nx} spatial points at time index {i}, t={time:.17g}: {args.out}")

if __name__ == '__main__':
    main()
