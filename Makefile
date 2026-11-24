# Config
ENV_MAIN ?= dl
ENV_BENCH ?= psml
PYTHON          ?= python3
CONFIG          ?= configs/train.yaml
EVAL_CONFIG     ?= configs/eval.yaml
REFS_CONFIG     ?= configs/rebuild_psml_references.yaml
SUBMODULE_PATH  ?= baselines/benchmark
ARTIFACTS_DIR   ?= artifacts
PSML_ENV_FILE ?= environment_psml_explicit.txt
SWEEP_CONFIG ?=
TPE_CONFIG ?=
SWEEP_ARGS   ?=

# Matplotlib headless (to enable terminal only setups), setting deterministic cuda kernels
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

.PHONY: help env submodules install-baselines install-core install \
        verify-env train eval sweep  prune-sweep-dry prune-sweep refs get-refs \
		freeze export-env remove-artifacts veryclean sweep-tpe

help:
	@echo "Targets:"
	@echo "  make env               - set/update the conda environment from environment.yml"
	@echo "	 make env-bench 		- creates the benchmark env used for the submodule"
	@echo "  make submodules        - update the submodule for the benchmark"
	@echo "  make install-baselines - install the submodule package for the benchmark"
	@echo "  make install-core      - install the main package"
	@echo "  make install           - full install: set the environment, install main and submodule"
	@echo "  make verify-env        - show if the setup of python and pytorch was succesfull and cuda available"
	@echo "  make train             - start training of selected path: (CONFIG=$(CONFIG))"
	@echo "  make eval              - start evaluation of selected path: (CONFIG=$(CONFIG))"
	@echo "  make freeze            - collect the artifacts needed to rebuild the project in $(ARTIFACTS_DIR)/"
	@echo "  make export-env        - update the environment.yml to guarantee the correct env (minimal)"
	@echo "  make clean             - clean the artifacts"
	@echo "  make veryclean         - remove the build artifacts"
	@echo "  make sweep             - run sweep: SWEEP_CONFIG=configs/sweeps/mlp_sweep.yaml"
	@echo "  make prune-sweep-dry   - show preds.csv files that would be removed"
	@echo "  make prune-sweep       - remove preds.csv files from an evaluated sweep"
	@echo "	 make get-refs 			- rebuild PSML benchmark references: (REFS_CONFIG = $(REFS_CONFIG))"
	@echo "  make sweep-tpe         - run Optuna TPE sweep: TPE_CONFIG=configs/sweeps/mlp_sweep.yaml"

# environment and installation
env:
	conda env create -n $(ENV_MAIN) -f environment.yml || \
	conda env update -n $(ENV_MAIN) -f environment.yml --prune

env-bench:
	@if conda env list | grep -q "^$(ENV_BENCH)[[:space:]]"; then \
		echo "Benchmark environment $(ENV_BENCH) already exists."; \
	else \
		conda create -n $(ENV_BENCH) --file $(PSML_ENV_FILE); \
	fi

submodules:
	git submodule update --init --recursive

install-baselines: submodules env-bench
	conda run -n $(ENV_BENCH) $(PYTHON) -m pip install -e $(SUBMODULE_PATH)

install-core: env
	conda run -n $(ENV_MAIN) $(PYTHON) -m pip install -e .

install: install-baselines install-core

verify-env:
	conda run -n $(ENV_MAIN) $(PYTHON) - <<'PY'
	import sys, torch
	print("Python:", sys.version.split()[0])
	print("Torch:", getattr(torch, "__version__", None))
	print("CUDA available:", torch.cuda.is_available())
	print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
	PY

# base
train:
	conda run -n $(ENV_MAIN) --no-capture-output $(PYTHON) src/train.py --config $(CONFIG)

eval:
	conda run -n $(ENV_MAIN) --no-capture-output $(PYTHON) \
		src/evaluate_model.py \
		--config $(CONFIG) \
		--eval_config $(EVAL_CONFIG)

sweep:
	@if [ -z "$(SWEEP_CONFIG)" ]; then \
		echo "Error: Please provide SWEEP_CONFIG."; \
		echo "Usage: make sweep SWEEP_CONFIG=configs/sweeps/mlp_sweep.yaml"; \
		exit 1; \
	fi
	conda run -n $(ENV_MAIN) --no-capture-output $(PYTHON) src/sweep.py \
		--sweep_config $(SWEEP_CONFIG) \
		$(SWEEP_ARGS)


sweep-tpe:
	@test -n "$(TPE_CONFIG)" || (echo "Error: Missing config. Example: make sweep-tpe TPE_CONFIG=configs/sweeps/mlp_sweep.yaml"; exit 1)
	conda run -n dl --no-capture-output python -u src/sweep_optuna.py \
		--sweep_config $(TPE_CONFIG)

prune-sweep-dry:
	@if [ -z "$(SWEEP_CONFIG)" ]; then \
		echo "Error: Please provide SWEEP_CONFIG."; \
		echo "Usage: make prune-sweep-dry SWEEP_CONFIG=configs/sweeps/mlp_sweep.yaml"; \
		exit 1; \
	fi
	conda run -n $(ENV_MAIN) --no-capture-output $(PYTHON) src/prune_sweep.py \
		--config $(SWEEP_CONFIG)

prune-sweep:
	@if [ -z "$(SWEEP_CONFIG)" ]; then \
		echo "Error: Please provide SWEEP_CONFIG."; \
		echo "Usage: make prune-sweep SWEEP_CONFIG=configs/sweeps/mlp_sweep.yaml"; \
		exit 1; \
	fi
	conda run -n $(ENV_MAIN) --no-capture-output $(PYTHON) src/prune_sweep.py \
		--config $(SWEEP_CONFIG) \
		--delete

# reproducibility
freeze:
	mkdir -p $(ARTIFACTS_DIR)
	conda run -n $(ENV_MAIN) --no-capture-output conda list --explicit > $(ARTIFACTS_DIR)/conda-explicit.txt
	conda run -n $(ENV_MAIN) --no-capture-output $(PYTHON) -m pip freeze > $(ARTIFACTS_DIR)/pip-freeze.txt
	git rev-parse --short HEAD > $(ARTIFACTS_DIR)/git-commit.txt
	- git -C $(SUBMODULE_PATH) rev-parse --short HEAD > $(ARTIFACTS_DIR)/baselines-commit.txt || true

# updates the environment to make sure it can be reproduced later
export-env:
	conda env export --from-history -n $(ENV_MAIN) > environment.yml

# freezes the benchmark environment to make sure it can be reproduced
export-env-bench:
	conda list -n $(ENV_BENCH) --explicit > $(PSML_ENV_FILE)

# rebuilds the references from psml which are set in the meta config and saves them as new references
get-refs:
	conda run -n $(ENV_BENCH) --no-capture-output $(PYTHON) references/psml-benchmark-scripts/rebuild_psml_references.py --config $(REFS_CONFIG) --overwrite

# clean up
remove-artifacts:
	rm -rf $(ARTIFACTS_DIR)/*.tmp __pycache__ **/__pycache__

veryclean: clean
	rm -rf build dist *.egg-info **/*.egg-info
