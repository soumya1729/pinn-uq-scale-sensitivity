"""Consolidate stored round-two evidence; no grid or perturbation is added.

Inputs are loaded producer archives with tag, summary, fields, spectral and rows.
Grouped energies describe the supplied coordinates without resolving individual
singular vectors in negligible or unresolved clusters.
"""
import csv
import json
from pathlib import Path

import numpy as np


GROUPS = ("lambda_below_gamma/100", "lambda_near_gamma", "lambda_above_100gamma")
NOTES = [
    "Original and extended selections are best tested grid pairs, not continuous optima; near-best ranges are within one log unit.",
    "Six stored coherent perturbations are finite sensitivity cases, not accuracy bounds or exhaustive perturbations.",
    "Sensitivity-choice fields use the original Jacobian at alternative selected scales.",
    "Slopes are partial derivatives with alpha fixed, not slopes along an alpha-reoptimised path.",
    "Grouped projected energy and model variance do not resolve individual coordinates in negligible or unresolved eigenclusters.",
    "Small positive singular values are not classified as exact kernel directions; the old formed-Gram floor is not a direct-J error bound.",
    "Matched-scale checks share stored gamma fields and test a scale identity, not independent numerical accuracy.",
    "Numerical status is retained; PASS does not certify weak-mode accuracy or working-model adequacy.",
    "The existing formed-Gram-floor synthetic stress test remains REVIEW_REQUIRED; this archive review does not revise that result.",
]


def _write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _best(values):
    """Flatten the producer's grid summary without dropping boundary flags."""
    result = {k: v for k, v in values.items() if not k.endswith("_range")}
    for key in ("within_one_log_unit_alpha_range", "within_one_log_unit_noise_range"):
        result[key + "_min"], result[key + "_max"] = values[key]
    return result


def _slope_groups(run):
    spectral, summary = run["spectral"], run["summary"]
    eigenvalues = np.asarray(spectral["eigenvalues"], dtype=float)
    energy = np.asarray(spectral["projected_data"], dtype=float) ** 2
    stored_terms = np.asarray(spectral["mode_evidence_slope"], dtype=float)
    term_indices = {str(name): i for i, name in enumerate(spectral["slope_settings"])}
    slopes, groups = [], []
    for record in summary["evidence_slopes"]:
        alpha, variance = record["alpha"], record["noise_variance"]
        gamma = alpha * variance
        model_variance = variance + eigenvalues / alpha
        terms = -.5 * (variance / model_variance) * (1 - energy / model_variance)
        ratio = eigenvalues / gamma
        masks = (ratio < .01, (ratio >= .01) & (ratio <= 100), ratio > 100)
        common = dict(tag=run["tag"], numerical_status=summary["numerical_status"],
                      setting=record["setting"], alpha=alpha, noise_variance=variance,
                      effective_precision=gamma)
        slopes.append(dict(common, slope=record["slope"],
            slope_finite_difference=record["slope_finite_difference"],
            slope_absolute_difference=record["slope_absolute_difference"],
            reconstructed_slope=float(terms.sum()),
            reconstruction_absolute_difference=abs(float(terms.sum()) - record["slope"]),
            max_stored_mode_term_difference=float(np.max(np.abs(
                terms - stored_terms[term_indices[record["setting"]]])))))
        archived = {group["group"]: group for group in record["groups"]}
        for label, mask in zip(GROUPS, masks):
            saved, count = archived[label], int(mask.sum())
            total_energy = float(energy[mask].sum())
            total_variance = float(model_variance[mask].sum())
            groups.append(dict(common, group=label, count=count,
                archived_count=saved["count"], projected_data_energy=total_energy,
                archived_projected_data_energy=saved["projected_data_energy"],
                slope=float(terms[mask].sum()), archived_slope=saved["slope"],
                lambda_over_alpha_sum=float((eigenvalues[mask] / alpha).sum()),
                noise_variance_sum=count * variance, model_variance_sum=total_variance,
                energy_over_model_variance=total_energy / total_variance if count else None,
                mean_lambda_over_alpha=float(np.mean(eigenvalues[mask] / alpha)) if count else None,
                mean_projected_data_energy=total_energy / count if count else None,
                mean_model_variance=total_variance / count if count else None))
    return slopes, groups


def _plot(groups, tag, outdir):
    """One reference figure of group means, with all runs retained in CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    choices = sorted({r["setting"] for r in groups if r["tag"] == tag
                      and r["setting"].startswith("affine_evidence_")},
                     key=lambda name: "_original_" not in name)
    if not choices:
        return None
    fig, axes = plt.subplots(1, len(choices), figsize=(5 * len(choices), 3.8), squeeze=False)
    for axis, setting in zip(axes[0], choices):
        selected = [r for r in groups if r["tag"] == tag and r["setting"] == setting]
        x = np.arange(len(selected))
        axis.bar(x - .18, [r["mean_projected_data_energy"] or 0 for r in selected],
                 width=.36, label="Mean projected data energy")
        axis.bar(x + .18, [r["mean_model_variance"] or 0 for r in selected],
                 width=.36, label="Mean model variance")
        axis.set_xticks(x, [f"{label}\nn={r['count']}" for label, r in zip(
            ("lambda/gamma < .01", ".01 to 100", "> 100"), selected)])
        axis.set_yscale("symlog", linthresh=selected[0]["noise_variance"] * .01)
        axis.set_title(setting.replace("affine_evidence_", "").replace("_", " "))
        axis.set_ylabel("Group mean: z² and v + lambda/alpha")
        axis.legend(fontsize=8)
    fig.suptitle(f"{tag}: grouped energy/model variance (empty groups show no bars)")
    fig.tight_layout()
    path = outdir / "evidence_energy_reference.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def review_evidence(runs, outdir):
    """Write compact evidence tables for every supplied archive and return metadata.

    The caller supplies validated production archives. No original output is
    changed; outdir must be the caller's separate review directory.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_rows, perturbation_rows, selected_rows, slope_rows, group_rows = [], [], [], [], []
    records = []
    for run in runs:
        summary = run["summary"]
        common = dict(tag=run["tag"], numerical_status=summary["numerical_status"])
        original, extended = summary["original_affine_evidence"], summary["extended_affine_evidence"]
        for grid, best in (("original", original), ("extended", extended)):
            best_rows.append(dict(common, grid=grid, **_best(best)))
        stability = summary["evidence_stability"]
        for case in stability["perturbations"]:
            best = case["best"]
            perturbation_rows.append(dict(common, case=case["case"],
                singular_value_envelope=case["singular_value_envelope"],
                max_singular_change=case["max_singular_change"],
                max_eigenvalue_change=case["max_eigenvalue_change"], **_best(best),
                alpha_ratio_to_extended=best["alpha"] / extended["alpha"],
                noise_ratio_to_extended=best["noise_variance"] / extended["noise_variance"],
                gamma_ratio_to_extended=best["effective_precision"] / extended["effective_precision"],
                alpha_boundary_changed=best["alpha_boundary"] != extended["alpha_boundary"],
                noise_boundary_changed=best["noise_boundary"] != extended["noise_boundary"]))
            for selected in case["selected"]:
                selected_rows.append(dict(common, case=case["case"], **selected,
                    slope_change=selected["perturbed_slope"] - selected["original_slope"]))
        slopes, groups = _slope_groups(run)
        slope_rows.extend(slopes)
        group_rows.extend(groups)
        records.append(dict(common, numerical_review_required=summary["numerical_review_required"],
            original_best=original, extended_best=extended,
            extension_log_evidence_change=extended["log_evidence"] - original["log_evidence"],
            perturbation_count=len(stability["perturbations"]), slope_count=len(slopes),
            envelope_definition=stability["envelope_definition"],
            affine_projection_consistency_norm=stability["affine_projection_consistency_norm"],
            equal_singular_cluster_rotations=stability["equal_singular_cluster_rotations"],
            maximum_slope_fd_absolute_difference=max(r["slope_absolute_difference"] for r in slopes),
            maximum_slope_reconstruction_difference=max(r["reconstruction_absolute_difference"] for r in slopes)))
    tables = {"evidence_best.csv": best_rows, "evidence_perturbations.csv": perturbation_rows,
              "evidence_perturbation_selected.csv": selected_rows, "evidence_slopes.csv": slope_rows,
              "evidence_slope_groups.csv": group_rows}
    for name, rows in tables.items():
        _write_csv(outdir / name, rows)
    reference = next((r["tag"] for r in runs if r["tag"] == "burgers_mlp_seed2"),
                     runs[0]["tag"] if runs else None)
    plot = _plot(group_rows, reference, outdir) if reference else None
    result = dict(reviewed_run_count=len(runs), runs=records, interpretation=NOTES,
        review_required_tags=[r["tag"] for r in records if r["numerical_status"] == "REVIEW_REQUIRED"
                              or r["numerical_review_required"]],
        table_row_counts={name: len(rows) for name, rows in tables.items()},
        reference_plot=plot, reference_plot_tag=reference if plot else None,
        unplotted_run_count=len(runs) - int(plot is not None),
        plot_note=("Group means are descriptive; all supplied runs and selected scales are tabulated."
                   if plot else "Reference plot not generated (Matplotlib is optional); all group data are tabulated."))
    (outdir / "evidence_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
