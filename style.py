"""Shared plot style. Same colour = same method, same linestyle = same model / beta."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import pendulum as p

COLOR = {"VV": "#2a78d6", "RK4": "#eb6834"}
STEPS = {"VV": p.step_vv, "RK4": p.step_rk4}
INK = "#52514e"          # exact / analytic curves
THIRD = "#1baf7a"        # a third entity that is neither method (e.g. a damped orbit)
LS_BETA = {0.0: "-", 0.05: "-.", 0.15: "--"}

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Figures are drawn at their printed size: report text width is 15.9 cm = 6.27 in
# (a4, 2.54 cm margins), so fonts here are the fonts on paper.
FULL = 6.27
HALF = 3.1
GUIDE = dict(color="#52514e", ls=":", lw=0.8)     # every slope / growth guide line

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "legend.handlelength": 2.2,
    "pdf.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 4,
})


def shade(color, k, n):
    """k-th of n shades of one colour: k = 0 lightest, k = n-1 the full colour."""
    c = np.array(to_rgb(color))
    mix = 0.55 * (n - 1 - k) / max(n - 1, 1)
    return tuple(c + (1 - c) * mix)


def save(fig, name):
    """Save as vector PDF (for the report) and PNG (for quick viewing)."""
    stem = os.path.splitext(name)[0]
    for ext in ("pdf", "png"):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=200)
    # the report includes the PDFs from report/figures; keep that copy in sync
    report_dir = os.path.join(os.path.dirname(FIGDIR), "report", "figures")
    if os.path.isdir(report_dir):
        fig.savefig(os.path.join(report_dir, f"{stem}.pdf"))
    print(f"saved {os.path.join(FIGDIR, stem)}.pdf/.png")
