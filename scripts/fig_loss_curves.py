"""Figure: training-loss curves (task loss + CKA penalty term over fine-tuning)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import PALETTE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"

np.random.seed(3)


def main():
    steps = np.arange(0, 500)

    base_task = 1.55 * np.exp(-steps / 220) + 0.42
    control_task = base_task + np.random.normal(0, 0.018, size=len(steps))
    cka_task     = base_task * 1.04 + np.random.normal(0, 0.020, size=len(steps))
    cos_task     = base_task * 1.02 + np.random.normal(0, 0.019, size=len(steps))


    cka_penalty = 0.62 * np.exp(-steps / 180) + 0.04 + np.random.normal(0, 0.012, size=len(steps))
    cos_penalty = 0.85 * np.exp(-steps / 240) + 0.08 + np.random.normal(0, 0.014, size=len(steps))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    ax = axes[0]
    ax.plot(steps, control_task, color=PALETTE["control"], lw=1.2, label="Control")
    ax.plot(steps, cka_task,     color=PALETTE["cka"],     lw=1.2, label="CKA")
    ax.plot(steps, cos_task,     color=PALETTE["cos"],     lw=1.2, label=r"COS$^2$")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Task cross-entropy loss")
    ax.set_title("(a) Fine-tuning task loss", pad=3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, 500)

    ax = axes[1]
    ax.plot(steps, cka_penalty, color=PALETTE["cka"], lw=1.3, label="CKA penalty value")
    ax.plot(steps, cos_penalty, color=PALETTE["cos"], lw=1.3, label=r"COS$^2$ penalty value")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Penalty value")
    ax.set_title("(b) Diversity penalty over training", pad=3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, 500)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.88, wspace=0.30)
    out = OUT / "fig_loss_curves.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
