"""Figure: router vs best-single vs oracle on the router test split."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import PALETTE

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figs"

BENCH_ORDER = ["MMLU", "GSM8K", "HumanEval", "ARC-Challenge", "ARC-Easy", "GPQA"]
BENCH_LABEL = {
    "MMLU": "MMLU",
    "GSM8K": "GSM8K",
    "HumanEval": "HumanEval",
    "ARC-Challenge": "ARC-C",
    "ARC-Easy": "ARC-E",
    "GPQA": "GPQA",
}


def main():
    df = pd.read_csv(DATA / "router_cka.csv").set_index("Benchmark").loc[BENCH_ORDER]

    methods = [
        ("Best single",   df["BestSingle_TEST"].values, "#333333"),
        ("Router (ours)", df["Router_TEST"].values, PALETTE["router"]),
        ("Oracle",        df["Oracle_TEST"].values, PALETTE["oracle"]),
    ]

    n_methods = len(methods)
    bar_w = 0.26
    x = np.arange(len(BENCH_ORDER))

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i, (name, vals, color) in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, label=name, color=color,
                      edgecolor="black", linewidth=0.5)
        if name == "Router (ours)":

            for b, bs, rt in zip(bars, df["BestSingle_TEST"].values, df["Router_TEST"].values):
                gap = (rt - bs) * 100
                ax.annotate(f"+{gap:.1f}", xy=(b.get_x()+b.get_width()/2, rt),
                            xytext=(0, 1.5), textcoords="offset points",
                            fontsize=7, ha="center",
                            color=PALETTE["router"])

    ax.set_xticks(x, [BENCH_LABEL[b] for b in BENCH_ORDER])
    ax.set_ylabel("Test accuracy (router split)")
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, 1.13))
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.08, right=0.99)
    out = OUT / "fig_router_bars.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
