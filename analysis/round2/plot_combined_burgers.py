"""Pair existing full-time and post-formation plots; no new numerical analysis."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter
from pypdf import PdfReader


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--paper-root", type=Path, required=True)
args = parser.parse_args()
csv_path = args.source_root / "analysis/round2/outputs/production_review/uncertainty_contrast_curves.csv"
rows = list(csv.DictReader(csv_path.open(newline="")))
out = args.paper_root / "figures"
out.mkdir(parents=True, exist_ok=True)
audit_path = args.paper_root / "verification/combined_burgers_figure_audit.json"
audit_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 15.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
colors = plt.get_cmap("tab10").colors
audit = {
    "source": str(csv_path),
    "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    "purpose": "Display-only re-layout of the original Burgers full/post curves",
    "families": {},
}

for arch in ("mlp", "fourier"):
    family = [r for r in rows if r["pde"] == "burgers" and r["arch"] == arch
              and r["region"] == "sharp" and r["subset"] in ("full", "post_formation")]
    tags = sorted({r["tag"] for r in family})
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 7.3))
    fig.subplots_adjust(left=.15, right=.99, top=.87, bottom=.21, wspace=.50, hspace=.54)
    displayed = []
    curves = []
    affine_tick_records = []
    for row_index, subset in enumerate(("full", "post_formation")):
        for tag in tags:
            seed = int(tag.rsplit("seed", 1)[1])
            color = colors[seed]
            for centre, column in (("trained", 1), ("affine", 2)):
                # Exactly the selection and ordering used by the original three-panel figures.
                cr = sorted((r for r in family if r["tag"] == tag and
                             r["subset"] == subset and r["centre"] == centre),
                            key=lambda r: float(r["effective_precision"]))
                x = [float(r["effective_precision"]) for r in cr]
                axes[row_index, column].loglog(x, [float(r["rho"]) for r in cr],
                                               ".-", color=color, lw=1.45, ms=3)
                if centre == "trained":
                    axes[row_index, 0].loglog(x, [float(r["R_sigma_rms"]) for r in cr],
                                              ".-", color=color, lw=1.45, ms=3)
                    axes[row_index, 0].axhline(float(cr[0]["R_tar"]), color=color,
                                              ls="--", lw=.85, alpha=.7)
                signature = [(r["tag"], r["subset"], r["centre"],
                              r["effective_precision"], r["R_sigma_rms"],
                              r["rho"], r["R_tar"]) for r in cr]
                displayed.extend(signature)
                curves.append({
                    "tag": tag, "subset": subset, "centre": centre,
                    "point_count": len(cr), "gamma_range": [x[0], x[-1]],
                    "data_signature_sha256": hashlib.sha256(
                        json.dumps(signature, separators=(",", ":")).encode()).hexdigest(),
                })
        for column in (1, 2):
            axes[row_index, column].axhline(1, color=".3", ls="--", lw=1, zorder=0)
        for axis in axes[row_index]:
            axis.set_xlabel(r"$\gamma=\alpha v$")
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=3))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="major", alpha=.18)
        axes[row_index, 0].set_ylabel(r"$R_{\sigma}^{\mathrm{rms}}$")
        for column in (1, 2):
            axes[row_index, column].set_ylabel(r"$\rho=R_{\mathrm{tar}}/R_{\sigma}^{\mathrm{rms}}$")
        affine_axis = axes[row_index, 2]
        affine_limits = affine_axis.get_ylim()
        affine_ticks = [tick for tick in (.5, 1, 2, 5)
                        if affine_limits[0] <= tick <= affine_limits[1]]
        affine_axis.yaxis.set_major_locator(FixedLocator(affine_ticks))
        affine_axis.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in affine_ticks]))
        assert affine_axis.get_ylim() == affine_limits
        affine_tick_records.append({"subset": subset, "unchanged_limits": list(affine_limits),
                                    "labelled_ticks": affine_ticks})
        row_name = "Full grid" if subset == "full" else "Post-formation"
        axes[row_index, 0].text(-.66, .5, row_name, rotation=90, ha="center", va="center",
                                transform=axes[row_index, 0].transAxes, fontsize=16)
    for axis, title in zip(axes[0], ("Covariance contrast", "Trained centre", "Affine centre")):
        axis.set_title(title, pad=14)
    fig.suptitle("Burgers / " + ("MLP" if arch == "mlp" else "Fourier"), fontsize=18, y=.97)
    handles = [Line2D([], [], color=colors[int(tag.rsplit("seed", 1)[1])], lw=1.6,
                      label=f'Seed {int(tag.rsplit("seed", 1)[1])}') for tag in tags]
    fig.legend(handles=handles, loc="lower center", ncol=4 if arch == "mlp" else 3,
               frameon=False, bbox_to_anchor=(.56, -.025), columnspacing=1.3, handlelength=1.5)

    # Compare the entire displayed data, including repeated settings, with the old selection.
    original = [(r["tag"], r["subset"], r["centre"], r["effective_precision"],
                 r["R_sigma_rms"], r["rho"], r["R_tar"]) for r in family]
    assert sorted(original) == sorted(displayed)
    whole_signature = hashlib.sha256(json.dumps(sorted(displayed), separators=(",", ":")).encode()).hexdigest()
    stem = out / f"round2_burgers_{arch}_combined"
    for suffix in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + suffix), bbox_inches="tight", dpi=180)
    pdf_width = float(PdfReader(stem.with_suffix(".pdf")).pages[0].mediabox.width)
    audit["families"][arch] = {
        "tags": tags, "six_panel_layout": True, "source_row_count": len(original),
        "displayed_row_count": len(displayed), "same_selection_as_original": True,
        "data_signature_sha256": whole_signature, "curves": curves,
        "source_font_points": {"base": 16, "legend": 15.5, "title": 18},
        "source_canvas_inches": [8.6, 7.3], "output_pdf": str(stem.with_suffix(".pdf")),
        "affine_gap_ticks": affine_tick_records,
        "pdf_width_points": pdf_width,
        "min_ordinary_font_at_0_98_textwidth_points": 15.5 * (.98 * 432) / pdf_width,
    }
    plt.close(fig)

audit_path.write_text(json.dumps(audit, indent=2) + "\n")
print("Saved two six-panel Burgers figures with unchanged curve selections.")
print(audit_path)
