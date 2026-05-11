"""Shared plotting style for all figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


PALETTE = {
    "control": "#7F7F7F",
    "cka":     "#0072B2",
    "cos":     "#D55E00",
    "oracle":  "#117733",
    "router":  "#CC79A7",
    "single":  "#000000",
}

LABEL = {
    "control": "Control (no penalty)",
    "cka":     "CKA penalty (ours)",
    "cos":     r"COS$^2$ penalty",
    "oracle":  "Oracle",
    "router":  "Learned router",
    "single":  "Best single",
}


def fig_size(width_in: float, aspect: float = 0.62):
    return (width_in, width_in * aspect)
