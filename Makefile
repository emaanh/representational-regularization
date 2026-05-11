# Convenience targets.
# Requires: uv. Optional: a GPU + the [llm] extra for the LLM pipeline.
#
# Usage:
#   make venv     # bootstrap .venv via uv
#   make figs     # regenerate every figure in figs/
#   make test     # run pytest
#   make clean    # nuke build artifacts (keeps source)

PY := .venv/bin/python

FIGS := \
  figs/fig_cka_heatmaps.pdf \
  figs/fig_oracle_scaling.pdf \
  figs/fig_router_bars.pdf \
  figs/fig_oracle_contributors.pdf \
  figs/fig_cost_vs_acc.pdf \
  figs/fig_cifar_validation.pdf \
  figs/fig_loss_curves.pdf

.PHONY: all venv figs clean test

all: venv figs test

venv:
	uv venv .venv
	uv pip install -r requirements.txt --python .venv/bin/python

figs: $(FIGS)

figs/fig_cka_heatmaps.pdf: scripts/fig_cka_heatmaps.py
	$(PY) $<
figs/fig_oracle_scaling.pdf: scripts/fig_oracle_scaling.py
	$(PY) $<
figs/fig_router_bars.pdf: scripts/fig_router_bars.py
	$(PY) $<
figs/fig_oracle_contributors.pdf: scripts/fig_oracle_contributors.py
	$(PY) $<
figs/fig_cost_vs_acc.pdf: scripts/fig_cost_vs_acc.py
	$(PY) $<
figs/fig_cifar_validation.pdf: scripts/fig_cifar_validation.py
	$(PY) $<
figs/fig_loss_curves.pdf: scripts/fig_loss_curves.py
	$(PY) $<

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -f figs/*.pdf
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
