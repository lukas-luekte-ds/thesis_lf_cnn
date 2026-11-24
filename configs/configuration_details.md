# Configuration Details
Since the framework is primarily controlled through YAML configuration files and `make` commands, this section provides additional details about the available configuration options.

## rebuild psml references
The `rebuild_psml_references.yaml` configuration is used by `references/psml-benchmark-scripts/rebuild_psml_references.py` to execute the selected PSML benchmark models and create frozen reference results for later comparison with custom models.

For each configured benchmark model, a reference directory is created containing:

- `summary.json`: metrics and result summary of the benchmark run,
- `config_used.yaml`: configuration used by the benchmark model,
- `meta.yaml`: metadata required for reproducibility, including the benchmark commit hash and the selected benchmark run.

### benchmark
- repo_root: Path to root of the benchmark
- workdir: Path to the directory with the model implementations
- logs_root: Path to the run logs
- conda_env: The name of the conda environment in which the psml submodule is installed

### references
- out_root: the output directory where the references are stored

### models
- reference_id: The name used in the output directory for the model
- entrypoint: The name of the implementation in the benchmark module
- args: The settings which are used to train the models (for more settings look into the referred model configs of the benchmark submodule)
- log_subdir: The name of the log subdirectory of the benchmark submodule

Note: Multiple benchmark models can be specified by adding multiple entries to the `models` list.

[!TODO] Make sure to list which additional time_features are implemented