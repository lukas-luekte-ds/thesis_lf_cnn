# Installation and Usage
The project is designed to be controlled mainly through "make" commands and YAML configuration files. The Makefile provides standardized entry points for installation, training, evaluation, sweeps, reproducibility and cleanup. Experiment-specific settings such as model type, dataset selection, and training parameters are defined in YAML config files.

## Installation
1. Clone the Repository
2. Download the dataset
3. Navigate into the root "THESIS_LF_CNN"
4. Execute "make install" in the terminal to set an environment
 "dl" and install the requirements and the submodule for the benchmark
5. Execute "make verify-env" to verify the installation and see if cuda is available
(6. install Cuda if needed)

## Reproducability
The project provides a freeze target to store environment and version information requiered to reproduce the environment of the work.
The `make freeze` command creates the reproducabiliy artifacts in the artifacts directory, incuding the conda package list, pip package list, the current Git commit of the main repository, and the Git commit of the benchmark submodule.
For a fully reproducible experiment, the corresponding training configuration, evaluation configuration, dataset version, random seeds, checkpoints, and evaluation results should also be stored together with the run output.

## Project-Structure
- artifacts -> artifacts needed to guarantee reproducibility
- baselines -> implementations of baseline forecasting models to have a reference
- configs -> configuration files
- data -> data folder
- diagrams -> some diagramms to show the project structure
- notebooks -> Jupyter notebooks for experimenting
- references -> the references generated from the baseline implementations
- runs -> the output of the runs
- src -> source code of the framework

## Quick Start (train a model)

### Show available commands
Use `make help` to get the info you need.

### Rebuilding PSML References
The PSML benchmark provides reference results that can be rebuilt using the benchmark submodule. Before rebuilding the references, make sure that the benchmark submodule is installed. (`make install-baselines` or `make install`)
To rebuild the references configure the `configs/rebuild_psml_references.yaml` and use `make get-refs`.
Most relevant fields of the `rebuild_psml_references.yaml`:
- benchmark.repo_root: Path to the PSML benchmark submodule.
- benchmark.workdir: Path to the directory from which the benchmark model scripts are executed.
- benchmark.logs_root: Path to the directory where the logs of the benchmark runs are stored
- references.out_root: Output directory where the rebuilt references are stored.
- models: List of the benchmark models to rebuild (multiselection possible) and their specifications (see `configuration_details.md` for more information)

Note: Rebuilding the references may take a considerable amount of time because the benchmark models are trained again. If the reference files already exist, this step usually does not need to be repeated.

### Dataloader
The framework supports two data loading modes: The original PSML benchmark dataloader and a customizable project-specific dataloader. The selected dataloader is used during training and sweeps, which are started via `make train` or the `make sweep` and can be configured in the data asection of the `configs/train.yaml`.

Relevant fields of the `train.yaml` referring to the data section:
- loader_type: The two modes are 'custom' and 'benchmark'
- root: Path to the data (in this case most likely the PSML folder)
- batch_size: Number of samples per batch
- shuffle: Boolean whether the training data should be shuffled
- val_ratio: Fraction of the training data used for validation (not from the full dataset since test data are not in this pool)

#### benchmark_loader
- include_grids: Selects the grids to load from the benchmark dataset. If left empty, all available grids are included. (Possible Options: CAISO, ERCOT, MISO, NYISO, PJM, SPP)
- include_years: Selects the years to load from the benchmark dataset. If left empty, all available years are included. (Possible Options: 2018, 2019, 2020)
- include_zones: Selects the zones number to load from the benchmark dataset. If left empty, all available zone numbers are included. (Possible Options (depends on the grid): 1-20)

#### custom_loader
- cache.enabled: Boolean to choose whether cache is used for the training to save time or not
- cache.name: The name of the folder where the cache is stored
- cache.rebuild: Set 'false' to use the existing cache file or 'true' to build a new cache file with the `cahce.name`
- source.subdir: The used subdirectory of the data
- source.timestamp_col: The name of the time column
- source.files: List of input files used by the custom dataloader. Files must be specified explicitly, as no benchmark-style filtering by grid, year, or zone is implemented. There is a customizable `name` and there `file_name` as flag
- extra_sources: Optional additional data sources that are merged with the main dataset before feature generation (e.g. PMU measurements).
- features.target_as_input: Boolean whether to use the target of the past for the training or not
- features.external: Set the features for the training
- features.time_features: Some features can be generated from raw time data. Choose which ones to use (for details which ones are implemented see `configuration_details.md`)
- task.targets: Define which targets are the task. Multi target possible. Set the `name`, `column`and `horizon_steps`
- task.input_length: The used timesteps as a training input
- task.horizon_steps: The distance of the prediction (for minute data one step is a minute) global if the horizon_steps are not set individualy
- task.prediction_stride: Sets the distance of the used data (for minute data 60 means to only use every 60th data point which means hourly)
- preprocessing.dropna_features: Boolean to remove 'NaN' data

### Training
The configuration of the training is set in the `configs/train.yaml`. The relevant fields are the `train` fields:
- train.epochs: The maximum training epochs
- train.optimizer: The chosen optimizer ('adamw' or 'adam' implemented)
- train.lr: The loss rate
- train.weight_decay: The weight_decay
- train.early_stopping_patience: The epochs until early stopping will be performed if no improvement
- train.early_stopping_min_delta: The min value to reset the early stopping counter
- train.loss_type: the loss type ('mae', 'mse' and 'huber' implemented)
- train.huber_delta: The used delta if loss_type is 'huber'

To select the model set the config (.yaml file) of the model as `model_ref` (e.g. model_ref: configs/mlp.yaml).


## Evaluate results

## Configure Experiments

## Advanced Usage