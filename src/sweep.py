import argparse
import random
import subprocess
from typing import Any

from config_utils import to_set_args, validate_sweep_config
from utils import load_config

"""
Hyperparameter sweep launcher which randomly trains different 
hyperparameter configurations (used for hyperparameter tuning).

This script does not train a model itself. Instead, it repeatedly starts
`src/train.py` with different hyperparameter overrides passed via `--set`.

Typical usage:
    python3 src/sweep.py --sweep_config configs/sweeps/mlp_sweep.yaml
    OR make sweep SWEEP_CONFIG=configs/sweeps/mlp_sweep.yaml (make sure to input a sweep configuration as .yaml)

Main responsibilities:
    - sample random hyperparameter configurations
    - convert sampled values into CLI `--set` arguments
    - launch one training process per sampled configuration
    - optionally print commands without executing them via `--dry`
    - load the sweep search space from a YAML config

Important assumptions:
    - `src/train.py` can parse all `--set key=value` overrides
    - each training run stores its outputs without overwriting previous runs
    - hyperparameter selection is based on validation metrics, not test metrics (but the eval has to be started manually)
    - uses a config as yaml file to config the search space
"""

def sample_cfg(
    sweep_cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:

    sampled = dict(sweep_cfg.get("fixed", {}))

    for name, spec in sweep_cfg["parameters"].items():
        sampled[name] = rng.choice(spec["values"])

    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=None)
    ap.add_argument("--train_config", type=str, default=None)
    ap.add_argument("--conda_env", type=str, default="dl")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
    "--sweep_config",
    type=str,
    required=True,
    help="Path to the sweep YAML config, e.g. configs/sweeps/mlp_sweep.yaml",
)
    args = ap.parse_args()

    sweep_cfg = load_config(args.sweep_config)
    validate_sweep_config(sweep_cfg)
    rng = random.Random(args.seed)
    run_cfg = sweep_cfg.get("run", {})

    n_runs = args.n_runs if args.n_runs is not None else run_cfg.get("n_runs", 30)
    if type(n_runs) is not int or n_runs < 1:
        raise ValueError("n_runs must be a positive integer.")

    train_config = args.train_config if args.train_config is not None else run_cfg.get("train_config", "configs/train.yaml")
    run_overrides = run_cfg.get("overrides", {})

    for i in range(1, n_runs + 1):
        sampled = sample_cfg(sweep_cfg, rng)

        cmd = [
            "conda", "run", "-n", args.conda_env, "--no-capture-output",
            "python3", "src/train.py",
            "--config", train_config,
        ] + to_set_args(run_overrides) + to_set_args(sampled)

        print(f"\n=== Run {i}/{n_runs} ===")
        print(" ".join(cmd))

        if not args.dry:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()