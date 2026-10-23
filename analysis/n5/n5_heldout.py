"""N5: held-out last-layer error fits on the saved evaluation grid.

NumPy only. Frozen checkpoints; separate fits in each fixed reference region.
Compare random halves with alternating blocks of time slices, then swap folds.
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np


ADMITTED = {
    "burgers_mlp": range(8),
    "burgers_fourier": (0, 2, 6),
    "allen_cahn_mlp": (0, 1, 3),
    "allen_cahn_fourier": (0, 1, 3),
}


def mlp(layers, x):
    for layer in layers[:-1]:
        x = np.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def features(params, xs, ts, fourier=None):
    """The tanh/affine forward map in common.py, with x-fast grid ordering."""
    x, t = xs[:, None], ts[:, None]
    if fourier is not None:
        x_phase = 2 * np.pi * (x @ fourier["B_X"])
        t_phase = 2 * np.pi * (t @ fourier["B_T"])
        x = np.concatenate((np.cos(x_phase), np.sin(x_phase)), axis=1)
        t = np.concatenate((np.cos(t_phase), np.sin(t_phase)), axis=1)
    sx = np.concatenate([mlp(p, x) for p in params["spatial"]], axis=1)
    st = np.concatenate([mlp(p, t) for p in params["temporal"]], axis=1)
    h = np.concatenate((np.tile(sx, (len(ts), 1)),
                        np.repeat(st, len(xs), axis=0)), axis=1)
    for layer in params["fusion"][:-1]:
        h = np.tanh(h @ layer["w"] + layer["b"])
    readout = params["fusion"][-1]
    prediction = (h @ readout["w"] + readout["b"]).ravel()
    return np.column_stack((h, np.ones(len(h)))), prediction


def coefficients(phi, error, rconds):
    """One SVD per fitting set, reused for the rank-cutoff comparison."""
    u, s, vt = np.linalg.svd(phi, full_matrices=False)
    projected = u.T @ error
    for rcond in rconds:
        keep = s > rcond * s[0]
        beta = vt[keep].T @ (projected[keep] / s[keep])
        yield rcond, beta, int(keep.sum()), float(s[0] / s[-1])


def split_scores(phi, error, fold, rconds):
    """Pool residual energies, rather than averaging fold fractions."""
    scores = {rc: {"fit_energy": 0.0, "test_energy": 0.0} for rc in rconds}
    total = float(error @ error)
    for k in (0, 1):
        fit, test = fold == k, fold != k
        if not fit.any() or not test.any():
            raise ValueError("Each regional fold must contain points.")
        for rc, beta, rank, condition in coefficients(phi[fit], error[fit], rconds):
            train_res = error[fit] - phi[fit] @ beta
            test_res = error[test] - phi[test] @ beta
            train_energy, test_energy = float(train_res @ train_res), float(test_res @ test_res)
            rec = scores[rc]
            rec[f"fit_count_{k}"] = int(fit.sum())
            rec[f"rank_{k}"] = rank
            rec[f"condition_{k}"] = condition
            rec[f"in_sample_{k}"] = train_energy / float(error[fit] @ error[fit])
            rec[f"heldout_{k}"] = test_energy / float(error[test] @ error[test])
            rec["fit_energy"] += train_energy
            rec["test_energy"] += test_energy
    for rc, rec in scores.items():
        rec["in_sample_pooled"] = rec.pop("fit_energy") / total
        rec["heldout_pooled"] = rec["test_energy"] / total
        rec["heldout_rmse"] = np.sqrt(rec.pop("test_energy") / len(error))
        yield rc, rec


def analyse(run_dir, split_seed, block_lengths, rconds):
    start = time.perf_counter()
    tag = run_dir.name
    metadata = json.loads((run_dir / f"{tag}_map.json").read_text())
    with (run_dir / f"{tag}_params.pkl").open("rb") as f:
        params = pickle.load(f)
    fourier = None
    if metadata["provenance"]["config"]["arch"] == "fourier":
        with np.load(run_dir / f"{tag}_fourier_B.npz") as z:
            fourier = {k: z[k] for k in ("B_X", "B_T")}
    with np.load(run_dir / f"{tag}_fields.npz") as z:
        xs, ts = z["xs"], z["ts"]
        prediction, reference, gradient = (z[k].ravel() for k in ("u_pred", "u_exact", "grad_abs"))
    phi, reconstructed = features(params, xs, ts, fourier)
    np.testing.assert_allclose(reconstructed, prediction, rtol=1e-9, atol=1e-10)
    if phi.shape[1] != 129:
        raise ValueError(f"Expected the paper's 129 features; found {phi.shape[1]}.")
    error = prediction - reference
    p70, p95 = np.percentile(gradient, (70, 95))
    regions = {"bulk": gradient < p70,
               "transition": (gradient >= p70) & (gradient < p95),
               "sharp": gradient >= p95}
    time_index = np.repeat(np.arange(len(ts)), len(xs))
    rng = np.random.default_rng(split_seed)
    rows = []
    for region, mask in regions.items():
        pr, er = phi[mask], error[mask]
        same_grid = {}
        for rc, beta, rank, condition in coefficients(pr, er, rconds):
            residual = er - pr @ beta
            same_grid[rc] = (float(residual @ residual) / float(er @ er), rank, condition)
        random_fold = np.empty(len(er), dtype=int)
        random_fold[rng.permutation(len(er))] = np.arange(len(er)) % 2
        splits = [("random", 0, random_fold)]
        splits += [("time_blocks", length, (time_index[mask] // length) % 2)
                   for length in block_lengths]
        for name, length, fold in splits:
            for rc, scores in split_scores(pr, er, fold, rconds):
                rows.append({"tag": tag, "region": region, "split": name,
                             "block_slices": length, "split_seed": split_seed,
                             "rcond": rc, "count": len(er), "features": phi.shape[1],
                             "same_grid_fraction": same_grid[rc][0],
                             "same_grid_rank": same_grid[rc][1],
                             "same_grid_condition": same_grid[rc][2],
                             "error_rmse": np.sqrt(np.mean(er**2)), **scores})
        primary = [r for r in rows if r["region"] == region and r["rcond"] == rconds[0]]
        print(f"{tag} {region}: same-grid={same_grid[rconds[0]][0]:.5g}; " +
              "; ".join(f"{r['split']}{r['block_slices'] or ''}={r['heldout_pooled']:.5g}"
                        for r in primary), flush=True)
    print(f"{tag}: {time.perf_counter() - start:.1f} seconds", flush=True)
    return rows


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="One frozen run directory")
    source.add_argument("--results", type=Path, default=Path(__file__).parent / "input",
                        help="Results root for all 17 runs; defaults to the bundled input folder")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[10])
    parser.add_argument("--rconds", type=float, nargs="+", default=[1e-10, 1e-12])
    args = parser.parse_args()
    if any(n <= 0 for n in args.block_lengths) or any(not 0 < rc < 1 for rc in args.rconds):
        parser.error("Block lengths must be positive and rank cutoffs between zero and one.")
    runs = [args.run_dir] if args.run_dir else [
        args.results / f"{family}_seed{seed}" for family, seeds in ADMITTED.items() for seed in seeds]
    rows = []
    for run in runs:
        rows.extend(analyse(run, args.split_seed, args.block_lengths, args.rconds))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(runs)} runs to {args.out}; total {time.perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()
