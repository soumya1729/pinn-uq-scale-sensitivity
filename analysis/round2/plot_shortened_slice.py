"""Plot the saved Burgers-MLP seed-2 slice nearest t=0.5; no model evaluation."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,required=True)
parser.add_argument('--paper-root',type=Path,required=True)
args = parser.parse_args()
root = args.paper_root
data = np.genfromtxt(args.source_root / "analysis/burgers_slice.csv", delimiter=",", names=True)
x, error, sigma = data["x"], data["absolute_error"], data["sigma"]
sharp = data["sharp"].astype(bool)

plt.rcParams.update({
    "font.family": "serif", "font.size": 13.5, "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})
fig, ax = plt.subplots(figsize=(7.2, 3.35))
fig.subplots_adjust(left=.12, right=.99, bottom=.17, top=.85)
dx = x[1] - x[0]
ax.axvspan(x[sharp].min() - dx/2, x[sharp].max() + dx/2,
           color="#e7e7e7", zorder=0)
error_line, = ax.semilogy(x, error, color="#b64130", lw=1.65,
                          label=r"Absolute error $|u_\theta-u^\ast|$")
sigma_line, = ax.semilogy(x, sigma, color="#235789", lw=1.85,
                          label=r"Full-network standard deviation $\sigma$")
ax.set(xlim=(-1, 1), ylim=(2e-7, 2), xlabel=r"$x$",
       ylabel="Error and uncertainty (log scale)")
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
ax.grid(axis="y", which="major", color="#dddddd", lw=0.6)
ax.legend(handles=[error_line, sigma_line,
                   Patch(facecolor="#e7e7e7", label="Sharp region")],
          loc="upper center", bbox_to_anchor=(.5,-.30), frameon=False, fontsize=12.5)
ax.set_title(r"Burgers-MLP, seed 2: $t=0.503>1/\pi$", fontsize=13.5, pad=9)
for extension in ("pdf", "png"):
    fig.savefig(root / f"figures/burgers_error_uncertainty_slice.{extension}", dpi=300)
plt.close(fig)
