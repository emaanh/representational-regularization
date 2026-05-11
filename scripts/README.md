# Figure and data-extraction scripts

Every figure in `figs/` is produced by exactly one script in this
directory; `make figs` from the repo root re-generates the lot.

| Script | Produces | Notes |
|---|---|---|
| `extract_xlsx.py`            | `data/*.csv`                      | Parses `CKA_Model_FINAL.xlsx` (kept outside the repo) into normalized CSVs. The CKA matrices, oracle scaling tables, and router results all derive from this. |
| `fig_cka_heatmaps.py`        | `figs/fig_cka_heatmaps.pdf`       | Pairwise linear-CKA heatmaps for Control / CKA-penalty / COS². |
| `fig_oracle_scaling.py`      | `figs/fig_oracle_scaling.pdf`     | Oracle vs. ensemble-size on MMLU / GSM8K / HumanEval. |
| `fig_router_bars.py`         | `figs/fig_router_bars.pdf`        | Best-single / Router / Oracle bars on the router test split. |
| `fig_oracle_contributors.py` | `figs/fig_oracle_contributors.pdf`| Marginal oracle gain from adding the k-th member. |
| `fig_cost_vs_acc.py`         | `figs/fig_cost_vs_acc.pdf`        | MMLU accuracy vs. active params at inference. |
| `fig_cifar_validation.py`    | `figs/fig_cifar_validation.pdf`   | CIFAR-10 pilot scaling (representative). |
| `fig_loss_curves.py`         | `figs/fig_loss_curves.pdf`        | Illustrative training dynamics. |
| `_plot_style.py`             | (module)                          | Shared matplotlib rcParams + palette / labels. Imported by every figure script. |

## Re-running

```bash
make figs                                # regenerate every figure
.venv/bin/python scripts/fig_oracle_scaling.py   # regenerate just one
```

## Adding a new figure

1. Drop a new script `scripts/fig_X.py` that writes to `figs/fig_X.pdf`.
2. Add the corresponding entry to the `FIGS` list in `Makefile`.

The figure scripts all share `_plot_style.py` so the visual language
stays consistent.
