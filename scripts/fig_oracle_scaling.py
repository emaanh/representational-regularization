"""Figure: oracle accuracy vs ensemble size, 3-panel."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import PALETTE, LABEL

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figs"

BENCHES = [("mmlu",      "MMLU"),
           ("gsm8k",     "GSM8K"),
           ("humaneval", "HumanEval")]
GROUPS = ["control", "cka", "cos"]


def load(slug: str, group: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"scaling_{slug}_{group}.csv")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharex=True)
    for ax, (slug, title) in zip(axes, BENCHES):
        for grp in GROUPS:
            df = load(slug, grp)
            ax.plot(df["Models"], df["OracleAcc"] * 100,
                    color=PALETTE[grp], label=LABEL[grp],
                    marker="o", markersize=2.7,
                    lw=1.6 if grp == "cka" else 1.3)
        bs = load(slug, "cka")["BestSingleAcc"].max() * 100
        ax.axhline(bs, color="black", lw=0.9, ls="--",
                   label=f"Best single (\\,{bs:.0f}\\%\\,)")
        ax.set_title(title, pad=3)
        ax.set_xlabel("Ensemble size $k$")
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

    axes[0].set_ylabel("Oracle accuracy (\\%)")
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0.0, ymin - 1.0), min(101.0, ymax + 1.5))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.28, top=0.90, wspace=0.26)
    out = OUT / "fig_oracle_scaling.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
