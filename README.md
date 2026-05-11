# Representational Regularization for Diverse Large Language Model Ensembles

> **Maxwell Fung** (UC Berkeley, `maxwellfung@berkeley.edu`) · **Emaan Heidari** (USC, `eheidari@usc.edu`) — equal contribution.

We sequentially fine-tune `N=30` LLM members from a single
[`deepseek-ai/deepseek-llm-7b-chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
backbone, regularizing each new member against the layer-wise linear CKA of
all previous members. The recipe drives mean pairwise CKA across the
ensemble from **0.846** (control) to **0.044** (CKA-penalty). A lightweight
learned router on top of frozen BGE sentence embeddings then beats the
best single 7B model by **+4.2 pp** average across MMLU, GSM8K, HumanEval,
ARC-Easy, and ARC-Challenge while keeping inference cost at one 7B
forward pass.

## Headline numbers

Router on the held-out test split (see `data/router_cka.csv`):

| | MMLU | GSM8K | HumanEval | ARC-C | ARC-E | GPQA |
|--|--|--|--|--|--|--|
| Best single 7B | 49.3 | 96.8 | 50.0 | 59.7 | 79.4 | 5.0 |
| **Router (ours, top-1)** | **55.1** | **98.4** | **56.3** | **63.9** | **82.6** | **25.0** |
| Oracle upper bound | 71.0 | 100.0 | 81.3 | 75.6 | 90.4 | 85.0 |

Pairwise CKA across the ensemble and MMLU oracle at k=30:

| | Control | COS² | **CKA (ours)** |
|--|--|--|--|
| Mean pairwise CKA | 0.846 | 0.898 | **0.044** |
| MMLU oracle @ k=30 | 64.3 | 63.7 | **72.1** |

## Repo layout

```
cka_ens/        the runnable Python package
├── similarity/  linear CKA + COS² (NumPy / TensorFlow / PyTorch)
├── llm/         sequential CKA-penalty trainer + router for DeepSeek-7B
├── cifar/       CIFAR-10 pilot study (CNN ensemble)
└── tools/       Excel → CSV extractor

data/           extracted result CSVs (CKA matrices + per-bench scaling + router)
figs/           rendered figures (PDF) — produced by scripts/
scripts/        figure generators + one-off data tools
notebooks/      walkthrough notebooks (CIFAR pilot, LLM walkthrough, router training)
tests/          unit tests for the linear-CKA math and router features
```

## Quickstart

```bash
uv venv .venv
uv pip install -r requirements.txt
make figs          # regenerate every figure from the CSVs in data/
make test          # run the unit-test suite
```

## Reproducing the LLM results

The LLM pipeline lives in `cka_ens/llm/` and runs end-to-end given a GPU
and the `[llm]` extra. All four stages are exposed as console entry
points; see `notebooks/llm_walkthrough.ipynb` for a guided trace.

```bash
uv pip install -e '.[llm]'

# 1. Sequentially fine-tune 30 members from DeepSeek-7B-Chat under the
#    CKA penalty. Member k uses members 0..k-1 as frozen anchors.
for i in $(seq 0 29); do
  python -m cka_ens.llm.train --root ./members_cka --idx $i --mode cka --lambda 10.0
done

# 2. Score every member on the six benchmarks; emits all_results.csv.
for bench in mmlu gsm8k humaneval arc_challenge arc_easy gpqa; do
  python -m cka_ens.llm.evaluate --root ./members_cka --benchmark $bench
done

# 3. BGE-encode the prompts and pivot rewards into a router checkpoint.
#    (See notebooks/Router.ipynb for the full pipeline.)
jupyter nbconvert --to notebook --execute notebooks/Router.ipynb

# 4. Train + evaluate the residual-MLP router.
python -m cka_ens.llm.router_train    --checkpoint router_data_checkpoint.pt --out router.pt
python -m cka_ens.llm.router_evaluate --router router.pt --data router_data_checkpoint.pt --topk 1
```

Hyperparameter defaults match the values used to produce the headline
table (Alpaca-2k slice, lr `2e-5`, batch=4, λ=10, 1 epoch per member;
router top-K=3, τ anneal `2.0→0.3`, 50 epochs).

> **Note.** The numbers above were produced by an earlier run of this
> pipeline. The released code is a clean port; rerunning it should
> recover these numbers up to seed noise.

## Reproducing the CIFAR-10 pilot (no GPU required)

The CIFAR-10 ensemble runs on CPU in a few minutes and is the cheapest
way to verify the recipe end-to-end on your own hardware.

```bash
uv pip install -e '.[cifar]'

python -m cka_ens.cifar.train    --mode control --num-models 5
python -m cka_ens.cifar.train    --mode cka     --num-models 5 --lambda 3.0
python -m cka_ens.cifar.evaluate weights_control weights_cka
python -m cka_ens.cifar.router   --combined-probs combined_probs.npz
```

The pilot mirrors the qualitative LLM result: control oracle saturates
near 80% by k=5; the CKA penalty pushes the oracle past 87% at k=10.

## What's in the package

`cka_ens.similarity` — three parallel implementations of linear CKA
(NumPy reference, TensorFlow for the CIFAR pilot, PyTorch for the LLM
trainer) plus the squared-cosine baseline. All three agree numerically
within float32 tolerance (`tests/test_linear_cka.py`).

`cka_ens.llm` — `train.py` (sequential CKA-penalty trainer with
forward-hook activation capture against frozen anchors), `penalty.py`
(differentiable linear-CKA penalty), `hooks.py` (HuggingFace decoder
hooks), `data.py` (Alpaca-2k loader with response-only loss masking),
`evaluate.py` (per-member benchmark scorer), `router_train.py` and
`router_evaluate.py` (BGE + residual-MLP router).

`cka_ens.cifar` — small CNN ensemble pilot. Same training-loop shape as
the LLM trainer; reads as a reference implementation that runs in
minutes.

## Tests

```bash
make test
```

23 tests covering: linear-CKA invariants (self-similarity, symmetry,
orthogonal + scale invariance), the squared-cosine baseline's failure
mode, router feature shapes, and data-extraction sanity (CKA matrices
are symmetric, the penalty actually reduces pairwise CKA, oracle
accuracy is non-decreasing in ensemble size).

## Citation

```bibtex
@misc{fung2026repreg,
  title  = {Representational Regularization for Diverse Large Language Model Ensembles},
  author = {Maxwell Fung and Emaan Heidari},
  year   = {2026},
  note   = {Independent research preprint},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
