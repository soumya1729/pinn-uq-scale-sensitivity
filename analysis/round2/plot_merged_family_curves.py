"""Merge the retained family curve figures; selections and values are unchanged."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from pypdf import PdfReader


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--paper-root", type=Path, required=True)
args = parser.parse_args()
csv_path = args.source_root / "analysis/round2/outputs/production_review/uncertainty_contrast_curves.csv"
with csv_path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
out = args.paper_root / "figures"
out.mkdir(parents=True, exist_ok=True)
audit_path = args.paper_root / "verification/merged_family_figure_audit.json"
audit_path.parent.mkdir(parents=True, exist_ok=True)

font_size = 10.5
plt.rcParams.update({
    "font.size": font_size, "axes.labelsize": font_size,
    "axes.titlesize": font_size, "xtick.labelsize": font_size,
    "ytick.labelsize": font_size, "legend.fontsize": font_size,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
colors = plt.get_cmap("tab10").colors
audit = {
    "source": str(csv_path),
    "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    "purpose": "Display-only merge of the four retained family figures",
    "final_print_width_points": .98 * 432,
    "figures": {},
}

def readable_x_ticks(axis):
    """Keep three labelled decades spanning the displayed precision range."""
    low, high = axis.get_xlim()
    first, last = math.ceil(math.log10(low)), math.floor(math.log10(high))
    exponents = (first, round((first + last) / 2), last)
    axis.xaxis.set_major_locator(FixedLocator([10.0**k for k in exponents]))
    axis.xaxis.set_major_formatter(FixedFormatter([rf"$10^{{{k}}}$" for k in exponents]))


def readable_y_ticks(axis):
    """Label each decade, adding 2/5 ticks only for a narrow vertical range."""
    low, high = axis.get_ylim()
    exponents = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
    narrow = math.log10(high / low) <= 1.7
    factors = (1, 2, 5) if narrow else (1,)
    ticks = [factor * 10.0**k for k in exponents for factor in factors
             if low <= factor * 10.0**k <= high]
    labels = ([f"{tick:g}" for tick in ticks] if narrow else
              [rf"$10^{{{round(math.log10(tick))}}}$" for tick in ticks])
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FixedFormatter(labels))


layouts = [
    ("burgers", [("mlp", "full"), ("mlp", "post_formation"),
                 ("fourier", "full"), ("fourier", "post_formation")], 7.0),
    ("allen_cahn", [("mlp", "full"), ("fourier", "full")], 4.0),
]

for pde, row_specs, height in layouts:
    fig, axes = plt.subplots(len(row_specs), 3, figsize=(6, height))
    fig.subplots_adjust(left=.18, right=.985, top=.91 if pde == "burgers" else .85,
                        bottom=.12 if pde == "burgers" else .20,
                        wspace=.52, hspace=.76 if pde == "burgers" else .85)
    displayed = []
    families = []
    for row_index, (arch, subset) in enumerate(row_specs):
        family = [r for r in rows if r["pde"] == pde and r["arch"] == arch
                  and r["subset"] == subset and r["region"] == "sharp"]
        tags = sorted({r["tag"] for r in family})
        signatures = []
        curves = []
        for tag in tags:
            seed = int(tag.rsplit("seed", 1)[1])
            for centre, column in (("trained", 1), ("affine", 2)):
                cr = sorted((r for r in family if r["tag"] == tag and r["centre"] == centre),
                            key=lambda r: float(r["effective_precision"]))
                x = [float(r["effective_precision"]) for r in cr]
                axes[row_index, column].loglog(x, [float(r["rho"]) for r in cr],
                                               ".-", color=colors[seed], lw=1.0, ms=2.2)
                if centre == "trained":
                    axes[row_index, 0].loglog(x, [float(r["R_sigma_rms"]) for r in cr],
                                              ".-", color=colors[seed], lw=1.0, ms=2.2)
                    axes[row_index, 0].axhline(float(cr[0]["R_tar"]), color=colors[seed],
                                              ls="--", lw=.65, alpha=.7)
                signature = [(r["tag"], r["subset"], r["centre"], r["effective_precision"],
                              r["R_sigma_rms"], r["rho"], r["R_tar"]) for r in cr]
                signatures.extend(signature)
                curves.append({"tag": tag, "centre": centre, "point_count": len(cr),
                               "gamma_range": [x[0], x[-1]]})
        original = [(r["tag"], r["subset"], r["centre"], r["effective_precision"],
                     r["R_sigma_rms"], r["rho"], r["R_tar"]) for r in family]
        assert sorted(signatures) == sorted(original)
        displayed.extend(signatures)
        for column, axis in enumerate(axes[row_index]):
            readable_x_ticks(axis)
            axis.xaxis.set_minor_formatter(NullFormatter())
            if column:
                axis.axhline(1, color=".3", ls="--", lw=.8, zorder=0)
            if pde == "allen_cahn" and arch == "mlp" and column == 0:
                low, high = axis.get_ylim()
                axis.set_ylim(low, high * 1.3)
            if column < 2:
                readable_y_ticks(axis)
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.tick_params(which="major", pad=2, length=3)
            axis.grid(True, which="major", alpha=.18)
            if pde == "allen_cahn" or subset == "post_formation":
                axis.set_xlabel(r"$\gamma=\alpha v$", labelpad=1)
        axes[row_index, 0].set_ylabel(r"$R_\sigma^{\mathrm{rms}}$", labelpad=2)
        axes[row_index, 1].set_ylabel(r"$\rho$ trained", labelpad=2)
        axes[row_index, 2].set_ylabel(r"$\rho$ zero-mean", labelpad=2)
        affine_axis = axes[row_index, 2]
        affine_limits = affine_axis.get_ylim()
        ticks = [tick for tick in (.5, 1, 2, 5) if affine_limits[0] <= tick <= affine_limits[1]]
        affine_axis.yaxis.set_major_locator(FixedLocator(ticks))
        affine_axis.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in ticks]))
        assert affine_axis.get_ylim() == affine_limits
        row_label = ("MLP" if arch == "mlp" else "Fourier")
        row_label += " / " + ("full" if subset == "full" else "post")
        position = axes[row_index, 0].get_position()
        fig.text(.025, (position.y0 + position.y1) / 2, row_label,
                 rotation=90, ha="center", va="center", weight="bold")
        families.append({
            "arch": arch, "subset": subset, "tags": tags,
            "source_row_count": len(original), "displayed_row_count": len(signatures),
            "same_selection_as_original": True,
            "data_signature_sha256": hashlib.sha256(
                json.dumps(sorted(signatures), separators=(",", ":")).encode()).hexdigest(),
            "curves": curves, "affine_limits": list(affine_limits), "affine_ticks": ticks,
            "display_axes": [{"column": column,
                              "x_limits": [float(value) for value in axis.get_xlim()],
                              "x_ticks": [float(value) for value in axis.get_xticks()],
                              "y_limits": [float(value) for value in axis.get_ylim()],
                              "y_ticks": [float(value) for value in axis.get_yticks()],
                              "y_labels": [label.get_text() for label in axis.get_yticklabels()]}
                             for column, axis in enumerate(axes[row_index])],
        })
    for axis, title in zip(axes[0], ("Covariance\ncontrast", "Trained-centre\ngap", "Zero-mean\naffine gap")):
        axis.set_title(title, pad=8)
    seeds = sorted({int(r[0].rsplit("seed", 1)[1]) for r in displayed})
    handles = [Line2D([], [], color=colors[s], lw=1.4,
                      label=f"Seed {s}" if s == seeds[0] else f"{s}") for s in seeds]
    fig.legend(handles=handles, loc="lower center", ncol=len(seeds),
               frameon=False, bbox_to_anchor=(.55, .012),
               handlelength=1.1, columnspacing=.85, handletextpad=.35, borderaxespad=0)
    stem = out / f"round2_{pde}_families_merged"
    # Fixed canvas width makes the final paper font size explicit; no shrinking of old PDFs.
    for suffix in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=200)
    page = PdfReader(stem.with_suffix(".pdf")).pages[0]
    width, pdf_height = float(page.mediabox.width), float(page.mediabox.height)
    scale = audit["final_print_width_points"] / width
    audit["figures"][pde] = {
        "layout": [len(row_specs), 3], "seeds": seeds, "families": families,
        "source_font_points": font_size, "pdf_width_points": width,
        "pdf_height_points": pdf_height, "height_at_print_width_points": pdf_height * scale,
        "min_ordinary_font_at_print_width_points": font_size * scale,
        "output_pdf": str(stem.with_suffix(".pdf")),
    }
    plt.close(fig)

audit_path.write_text(json.dumps(audit, indent=2) + "\n")
print("Saved merged Burgers and Allen–Cahn figures; all curve rows and values are unchanged.")
print(audit_path)
