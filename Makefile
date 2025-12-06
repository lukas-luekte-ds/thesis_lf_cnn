# Config
ENV_NAME        ?= dl
PYTHON          ?= python
CONFIG          ?= configs/mlp.yaml
SUBMODULE_PATH  ?= baselines/benchmark
ARTIFACTS_DIR   ?= artifacts

# Matplotlib headless (to enable terminal only setups), setting deterministic cuda kernels 
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

.PHONY: help env submodules install-baselines install-core install \
        verify-env train eval freeze export-env remove-artifacts veryclean

help:
	@echo "Targets:"
	@echo "  make env               - set/update the conda environment from environment.yml"
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

# environment and installation
env:
	conda env update -n $(ENV_NAME) -f environment.yml --prune

submodules:
	git submodule update --init --recursive

install-baselines: submodules
	conda run -n $(ENV_NAME) $(PYTHON) -m pip install -e $(SUBMODULE_PATH)

install-core:
	conda run -n $(ENV_NAME) $(PYTHON) -m pip install -e .

install: env install-baselines install-core

verify-env:
	conda run -n $(ENV_NAME) $(PYTHON) - <<'PY'
	import sys, torch
	print("Python:", sys.version.split()[0])
	print("Torch:", getattr(torch, "__version__", None))
	print("CUDA available:", torch.cuda.is_available())
	print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
	PY

# base 
train:
	conda run -n $(ENV_NAME) --no-capture-output $(PYTHON) src/train.py --config $(CONFIG)

eval:
	conda run -n $(ENV_NAME) --no-capture-output $(PYTHON) src/eval.py --config $(CONFIG)

# reproducibility
freeze:
	mkdir -p $(ARTIFACTS_DIR)
	conda run -n $(ENV_NAME) --no-capture-output conda list --explicit > $(ARTIFACTS_DIR)/conda-explicit.txt
	conda run -n $(ENV_NAME) --no-capture-output $(PYTHON) -m pip freeze > $(ARTIFACTS_DIR)/pip-freeze.txt
	git rev-parse --short HEAD > $(ARTIFACTS_DIR)/git-commit.txt
	- git -C $(SUBMODULE_PATH) rev-parse --short HEAD > $(ARTIFACTS_DIR)/baselines-commit.txt || true

# updates the environment to make sure it can be reproduced later
export-env:
	conda env export --from-history -n $(ENV_NAME) > environment.yml


# clean up
remove-artifacts:
	rm -rf $(ARTIFACTS_DIR)/*.tmp __pycache__ **/__pycache__

veryclean: clean
	rm -rf build dist *.egg-info **/*.egg-info
