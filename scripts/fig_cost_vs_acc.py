"""Figure: inference-cost vs MMLU accuracy."""
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


def main():
    router_cka = pd.read_csv(DATA / "router_cka.csv").set_index("Benchmark")
    best_single = float(router_cka.loc["MMLU", "BestSingle_TEST"])
    router_acc  = float(router_cka.loc["MMLU", "Router_TEST"])
    oracle_acc  = float(router_cka.loc["MMLU", "Oracle_TEST"])


    refs = [
        (7.0,   0.494, "DeepSeek-LLM-7B-Chat (base)"),
        (67.0,  0.711, "DeepSeek-LLM-67B-Chat"),
    ]
    rx = [r[0] for r in refs]; ry = [r[1] for r in refs]

    fig, ax = plt.subplots(figsize=(6.3, 3.0))


    ax.plot(rx, ry, color="#bbbbbb", lw=1.2, ls="-", zorder=1)
    for x, y, lab in refs:
        ax.scatter([x], [y], s=46, color="#555555", marker="s",
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(lab, (x, y), xytext=(6, -2),
                    textcoords="offset points", fontsize=7, color="#444444")


    ax.scatter([7.0], [best_single], s=58, color="#444444",
               edgecolor="black", linewidth=0.5, zorder=4, marker="o")
    ax.annotate("Best single 7B fine-tune",
                (7.0, best_single), xytext=(8, -10),
                textcoords="offset points", fontsize=7.5, color="#444444")

    ax.scatter([7.0], [router_acc], s=72, color=PALETTE["router"],
               edgecolor="black", linewidth=0.7, zorder=4, marker="*")
    ax.annotate(r"\textbf{Ours: CKA ensemble + router (7B active)}",
                (7.0, router_acc), xytext=(8, 3),
                textcoords="offset points", fontsize=8,
                color=PALETTE["router"])

    ax.scatter([7.0], [oracle_acc], s=46, color=PALETTE["oracle"],
               edgecolor="black", linewidth=0.5, zorder=3, marker="^")
    ax.annotate("Oracle upper bound",
                (7.0, oracle_acc), xytext=(8, 0),
                textcoords="offset points", fontsize=7.5,
                color=PALETTE["oracle"])


    ax.annotate("", xy=(6.3, router_acc), xytext=(6.3, best_single),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["router"],
                                lw=1.4, alpha=0.9))
    gain = (router_acc - best_single) * 100
    ax.text(5.9, (router_acc + best_single)/2,
            f"+{gain:.1f}\\,pp", color=PALETTE["router"],
            fontsize=9, va="center", ha="right",
            fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Active parameters at inference (B, log scale)")
    ax.set_ylabel("MMLU accuracy")
    ax.set_xlim(3, 200)
    ax.set_ylim(0.45, 0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.16)
    out = OUT / "fig_cost_vs_acc.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
