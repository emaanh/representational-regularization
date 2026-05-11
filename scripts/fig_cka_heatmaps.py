"""Figure: pairwise CKA / similarity heatmaps for the three training regimes."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import fig_size

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> np.ndarray:
    return pd.read_csv(DATA / f"cka_matrix_{name}.csv", index_col=0).values


def main():
    groups = [("control", "Control"),
              ("cka",     "CKA penalty (ours)"),
              ("cos",     r"COS$^2$ penalty")]
    mats = {g: load(g) for g, _ in groups}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.viridis

    for ax, (key, title) in zip(axes, groups):
        m = mats[key]
        im = ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks([0, 9, 19, 29], ["1", "10", "20", "30"])
        ax.set_yticks([0, 9, 19, 29], ["1", "10", "20", "30"])
        ax.set_title(title, pad=4)
        ax.set_xlabel("Model index")
        ax.tick_params(width=0.6, length=2.5)
        ax.grid(False)

        off = m[np.triu_indices_from(m, k=1)]
        ax.text(0.02, 0.98, fr"$\bar{{s}} = {off.mean():.3f}$",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55))

    axes[0].set_ylabel("Model index")

    cax = fig.add_axes([0.93, 0.18, 0.014, 0.65])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Linear CKA", rotation=90, labelpad=6)
    cb.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.16, top=0.90, wspace=0.18)
    out = OUT / "fig_cka_heatmaps.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
