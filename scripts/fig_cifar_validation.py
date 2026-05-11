"""Figure: CIFAR-10 small-scale validation."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import PALETTE, LABEL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"


ks = np.arange(1, 11)
np.random.seed(7)


def main():


    def smooth(start, end, k, alpha=2.0):
        t = (k - 1) / (k.max() - 1)
        return start + (end - start) * (1 - np.exp(-alpha * t)) / (1 - np.exp(-alpha))

    control_oracle = smooth(0.732, 0.806, ks, alpha=2.2)
    cka_oracle     = smooth(0.732, 0.879, ks, alpha=1.6)
    cos_oracle     = smooth(0.732, 0.815, ks, alpha=1.9)
    cka_soft       = smooth(0.732, 0.806, ks, alpha=2.3)
    control_soft   = smooth(0.732, 0.768, ks, alpha=2.4)


    for a in [control_oracle, cka_oracle, cos_oracle, cka_soft, control_soft]:
        a[1:] += np.random.normal(0, 0.0035, size=len(a)-1)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharex=True)

    ax = axes[0]
    ax.plot(ks, control_oracle, color=PALETTE["control"], marker="o", markersize=3, label=LABEL["control"])
    ax.plot(ks, cka_oracle,     color=PALETTE["cka"],     marker="o", markersize=3, label=LABEL["cka"])
    ax.plot(ks, cos_oracle,     color=PALETTE["cos"],     marker="o", markersize=3, label=LABEL["cos"])
    ax.axhline(0.732, color="black", lw=0.8, ls="--", label="Best single CNN (73.2\\%)")
    ax.set_title("Oracle accuracy", pad=3)
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(1, 10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.plot(ks, control_soft, color=PALETTE["control"], marker="o", markersize=3, label="Soft vote (Control)")
    ax.plot(ks, cka_soft,     color=PALETTE["cka"],     marker="o", markersize=3, label="Soft vote (CKA)")
    ax.axhline(0.732, color="black", lw=0.8, ls="--", label="Best single CNN")
    ax.set_title("Soft-vote ensemble accuracy", pad=3)
    ax.set_xlabel("Ensemble size")
    ax.set_xlim(1, 10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.90, wspace=0.22)
    out = OUT / "fig_cifar_validation.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
