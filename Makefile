SHELL := /bin/bash

UV ?= uv
PYTHON ?= python3
UV_CACHE_DIR ?= /tmp/uv-cache-master-deg-project
RUFF_CACHE_DIR ?= /tmp/ruff-cache-master-deg-project
PY_FILES := propagation.py tests main.py model_fitting.py scripts/__init__.py scripts/download_snap_data.py scripts/run_collegemsg_fit.py scripts/run_real_dataset_tail_comparison.py scripts/run_nk_experiment.py scripts/run_nk_grid.py scripts/run_pn_experiment.py scripts/run_sk_experiment.py scripts/run_sk_grid.py scripts/run_sk_kstar_experiment.py scripts/run_sk_kstar_grid.py

.DEFAULT_GOAL := help

.PHONY: test help sync format lint check notebook snap-download run-nk run-nk-grid run-pn run-sk run-sk-grid run-sk-kstar run-sk-kstar-grid run-collegemsg-fit

help:
	@echo "Available targets:"
	@echo "  make sync        - install/update dependencies via uv"
	@echo "  make format      - format Python files with ruff"
	@echo "  make lint        - lint Python files with ruff"
	@echo "  make test        - run pytest"
	@echo "  make check       - run format check and lint"
	@echo "  make notebook    - launch JupyterLab with the visualization notebook"
	@echo "  make snap-download - download SNAP temporal datasets"
	@echo "  make run-nk      - run one N_k experiment"
	@echo "  make run-nk-grid - run the default N_k parameter grid"
	@echo "  make run-pn      - run the default p_n experiment for G0^(1)"
	@echo "  make run-sk      - run the default S_k experiment for G0^(1)"
	@echo "  make run-sk-grid - run the default S_k parameter grid"
	@echo "  make run-sk-kstar      - run the default K* experiment for S_k"
	@echo "  make run-sk-kstar-grid - run the default K* parameter grid"
	@echo "  make run-collegemsg-fit - fit the model to SNAP CollegeMsg"
	@echo "  make empirical-smoke - generate small ensembles and render intervals"
	@echo "  make render-empirical - render conditional intervals from saved inputs"
	@echo "  make verify-gossip-rate - enumerate fixed-topology completion bounds"

test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m pytest -q

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --dev

format:
	UV_CACHE_DIR=$(UV_CACHE_DIR) RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(UV) run ruff format $(PY_FILES)

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(UV) run ruff check $(PY_FILES)

check:
	UV_CACHE_DIR=$(UV_CACHE_DIR) RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(UV) run ruff format --check $(PY_FILES)
	UV_CACHE_DIR=$(UV_CACHE_DIR) RUFF_CACHE_DIR=$(RUFF_CACHE_DIR) $(UV) run ruff check $(PY_FILES)

notebook:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra notebook jupyter lab notebooks/wang_resnick_visuals.ipynb

SNAP_DOWNLOAD_ARGS ?=

snap-download:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.download_snap_data $(SNAP_DOWNLOAD_ARGS)

run-nk:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_nk_experiment --p 0.5 --c 0.1 --runs 1000 --steps 200

run-nk-grid:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_nk_grid --p-values 0.1,0.5,0.9 --c-values 0.1,0.2,0.3 --runs 1000 --steps 200

run-pn:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_pn_experiment --p-values 0.1,0.5,0.9 --c-values 0.1,0.2,0.3 --snapshot-steps 10,30,50,100 --base-runs 5 --replications 500

run-sk:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_sk_experiment --p 0.5 --c 0.2 --runs 100 --steps 200

run-sk-grid:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_sk_grid --p-values 0.1,0.5,0.9 --c-values 0.1,0.2,0.3 --runs 100 --steps 200

run-sk-kstar:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_sk_kstar_experiment --p 0.5 --c 0.2 --runs 200 --nu 1 --t-star 200

run-sk-kstar-grid:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_sk_kstar_grid --p-values 0.1,0.5,0.9 --c-values 0.1,0.2,0.3 --runs 200 --nu 1 --t-star 200

COLLEGEMSG_PATH ?= data/CollegeMsg.txt.gz

run-collegemsg-fit:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_collegemsg_fit --data-path $(COLLEGEMSG_PATH)

# Portable empirical rendering and finite-state verification.
PY_FILES += scripts/empirical_intervals.py scripts/render_supervisor_revision.py scripts/verify_gossip_rate_bound.py scripts/run_empirical_smoke.py
.PHONY: empirical-smoke render-empirical verify-gossip-rate
EMPIRICAL_ARGS ?= --output-dir results/empirical-intervals
EMPIRICAL_SMOKE_ARGS ?= --output-dir results/empirical-smoke
GOSSIP_RATE_ARGS ?= --output-file results/gossip-rate-verification.json

empirical-smoke:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.run_empirical_smoke $(EMPIRICAL_SMOKE_ARGS)

render-empirical:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.render_supervisor_revision $(EMPIRICAL_ARGS)

verify-gossip-rate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run $(PYTHON) -m scripts.verify_gossip_rate_bound $(GOSSIP_RATE_ARGS)
