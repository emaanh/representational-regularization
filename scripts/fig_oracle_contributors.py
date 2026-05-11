"""Figure: individual oracle contribution per ensemble member."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import PALETTE, LABEL

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figs"


def main():
    benches = [("mmlu", "MMLU"), ("arc_challenge", "ARC-Challenge")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
    for ax, (slug, title) in zip(axes, benches):
        for grp in ["control", "cka", "cos"]:
            df = pd.read_csv(DATA / f"scaling_{slug}_{grp}.csv")

            cum = df["OracleAcc"].values
            marg = np.concatenate([[cum[0]], np.diff(cum)])

            ks = pd.Series(marg).rolling(3, min_periods=1).mean().values
            ax.plot(df["Models"], ks * 100,
                    color=PALETTE[grp], marker="o", markersize=2.3,
                    label=LABEL[grp])
        ax.set_title(title, pad=3)
        ax.set_xlabel("k-th model added")
        ax.set_ylabel("Marginal oracle gain (\\%)")
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlim(1, 30)
    axes[0].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.90, wspace=0.27)
    out = OUT / "fig_oracle_contributors.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
