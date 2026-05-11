# Notebooks

Three notebooks ship with the repo:

## `perspective_2.ipynb` — CIFAR-10 pilot

Sequentially trains a small CNN ensemble on CIFAR-10, with the linear-CKA
penalty optionally activated against the activations of all previously
trained members. Also implements hard/soft/oracle voting and a
logistic-regression router with confidence fallback.

**Production equivalent:**

```bash
python -m cka_ens.cifar.train    --mode cka --num-models 5
python -m cka_ens.cifar.evaluate weights_control weights_cka
python -m cka_ens.cifar.router   --combined-probs combined_probs.npz
```

The math in the notebook (`tf_linear_cka_fast`) and the package
(`cka_ens.similarity.tf_cka.tf_linear_cka_fast`) are identical; a NumPy
reference (`cka_ens.similarity.numpy_cka.linear_cka`) is locked down by
`tests/test_linear_cka.py`.

## `Router.ipynb` — LLM router (BGE + residual MLP)

Trains the BGE-embedding + residual-MLP router used for the LLM ensemble.
Highlights:

- frozen `BAAI/bge-large-en-v1.5` sentence embeddings of the input prompt,
- residual MLP `1024 → 256 → 256 (+skip) → N` head,
- top-K expected-reward loss with temperature anneal `2.0 → 0.3`,
- entropy bonus that decays during training,
- auxiliary cross-entropy toward the oracle argmax,
- per-sample hard-example reweighting.

It expects an `all_results.csv` with columns `Model,Question,Benchmark,Reward`
and emits `router_data_checkpoint.pt` containing BGE features and the
per-model reward matrix. The training stage produces the router
accuracy / oracle / best-single numbers in `data/router_cka.csv`.

**Production equivalent:**

```bash
python -m cka_ens.llm.router_train    --checkpoint router_data_checkpoint.pt --out router.pt
python -m cka_ens.llm.router_evaluate --router router.pt --data router_data_checkpoint.pt --topk 1
```

## `llm_walkthrough.ipynb` — full LLM pipeline guide

A 5-stage walkthrough of the LLM pipeline from backbone fine-tuning to
deployed routing. It is a reading guide for `cka_ens.llm` rather than a
runnable demo; running it end-to-end requires ~30 GPU-hours.
