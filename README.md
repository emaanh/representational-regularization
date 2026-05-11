# Representational Regularization for Diverse Large Language Model Ensembles

> **Maxwell Fung** (UC Berkeley, `maxwellfung@berkeley.edu`) · **Emaan Heidari** (USC, `eheidari@usc.edu`) — equal contribution.

This repository accompanies the paper *"Representational Regularization
for Diverse Large Language Model Ensembles"*. It contains:

```
paper/        the workshop paper (LaTeX) + figures + extracted result CSVs
scripts/      figure generators (Python; matplotlib)
cka_ens/      runnable Python package for the CIFAR-10 pilot study
notebooks/    original exploratory notebooks (CIFAR pilot, router training)
tests/        unit tests for the linear-CKA implementation
```

## TL;DR

We sequentially fine-tune `N=30` members from a single
[`deepseek-ai/deepseek-llm-7b-chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
backbone, regularising each new member against the layer-wise linear CKA of
all previous members. The recipe drives mean pairwise CKA across the
ensemble from `0.846` (control) to `0.044` (CKA-penalty). With a lightweight
learned router on top of frozen BGE sentence embeddings, the deployed
pipeline beats the best single 7B model by **+4.2pp** average across MMLU,
GSM8K, HumanEval, ARC-Easy, and ARC-Challenge while keeping inference cost
at one 7B forward pass.

See the paper PDF: [`paper/main.pdf`](paper/main.pdf).

## Reproducing the figures and the paper

Everything is reproducible from the extracted CSVs in `paper/data/`.

```bash
# (1) Bootstrap the local Python environment (matplotlib, pandas, openpyxl).
uv venv .venv
uv pip install -r requirements.txt  # or: uv pip sync pyproject.toml

# (2) Re-extract CSVs from the raw Excel workbook (idempotent).
.venv/bin/python scripts/extract_xlsx.py

# (3) Re-generate all figures.
make figs

# (4) Build the PDF.
make paper       # uses tectonic; install via `brew install tectonic`
```

Or just `make all` to do everything.

## The CIFAR-10 pilot (runnable)

The CIFAR pilot study in Appendix A is shipped as a runnable package:

```bash
# Train a 5-model CKA-penalized ensemble + a control on CIFAR-10.
.venv/bin/python -m cka_ens.cifar.train --mode control --num-models 5
.venv/bin/python -m cka_ens.cifar.train --mode cka     --num-models 5 --lambda 3.0

# Evaluate (hard / soft / oracle vote).
.venv/bin/python -m cka_ens.cifar.evaluate weights_control weights_cka

# Train + evaluate the logistic-regression router.
.venv/bin/python -m cka_ens.cifar.router
```

Each entry point also doubles as a library; see `cka_ens/cifar/*.py` for
the imports.

## Results at a glance

| | MMLU | GSM8K | HumanEval | ARC-C | ARC-E | GPQA |
|--|--|--|--|--|--|--|
| Best single 7B | 49.3 | 96.8 | 50.0 | 59.7 | 79.4 | 5.0 |
| **Router (ours, top-1)** | **55.1** | **98.4** | **56.3** | **63.9** | **82.6** | **25.0** |
| Oracle upper bound | 71.0 | 100.0 | 81.3 | 75.6 | 90.4 | 85.0 |

(Numbers are on the held-out router test split, see `paper/data/router_cka.csv`.)

| | Control | COS² | **CKA (ours)** |
|--|--|--|--|
| Mean pairwise CKA | 0.846 | 0.898 | **0.044** |
| MMLU oracle @ k=30 | 64.3 | 63.7 | **72.1** |

## File-level map of the raw artifacts

- `CKA_Model_FINAL.xlsx` — raw 30-model bootstrap results (pairwise CKA
  matrices, per-benchmark scaling, router test outcomes) for the three
  training regimes (Control, CKA-penalty, COS²-penalty).
- `paper/data/*.csv` — clean, normalized extracts produced by
  `scripts/extract_xlsx.py`. These are the inputs to every figure script.
- `notebooks/perspective_2.ipynb` — original CIFAR-10 CNN pilot.
- `notebooks/Router.ipynb` — original router training pipeline (BGE +
  residual MLP + reward-aligned + temperature-annealed loss).

## Citation

If you use this work please cite

```bibtex
@misc{fung2026repreg,
  title  = {Representational Regularization for Diverse Large Language Model Ensembles},
  author = {Maxwell Fung and Emaan Heidari},
  year   = {2026},
  note   = {Workshop manuscript},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
