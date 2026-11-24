"""Continue hyperparameter optimization with Optuna TPE, initialized from random runs.

Run from project root. Requires optuna and pyyaml.
The random-sweep YAML is the single source of truth for the search space.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial

from config_utils import to_set_args, validate_sweep_config
from utils import load_config


def get_nested(cfg: dict, dotted: str) -> Any:
    value = cfg
    for part in dotted.split("."):
        value = value[part]
    return value


def encode(value: Any) -> Any:
    # Optuna categorical values cannot contain lists/dicts.
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def decode(value: Any, choices: list) -> Any:
    return json.loads(value) if any(isinstance(x, (list, dict)) for x in choices) else value

def normalize_param(value: Any, allowed_values: list[Any]) -> Any:
    if isinstance(value, str) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in allowed_values
    ):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def read_metrics(run_dir: Path) -> tuple[float, list[tuple[str, int]]]:
    path = run_dir / "metrics_aggregate.json"
    with path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    units = [(str(loc), int(year)) for loc, year in metrics["run_units"]]
    if not units:
        raise ValueError(f"No run units in {path}")
    if len(units) != len(set(units)):
        raise ValueError(f"Duplicate run units in {path}")
    if int(metrics["n_run_units"]) != len(units):
        raise ValueError(f"Inconsistent n_run_units in {path}")
    score = float(metrics["macro_avg_best_val_mae"])
    if not math.isfinite(score):
        raise ValueError(f"Non-finite validation MAE in {path}")
    return score, sorted(units)


def merge_settings(random_values: dict, tpe_values: dict, label: str) -> dict:
    if not isinstance(random_values, dict) or not isinstance(tpe_values, dict):
        raise ValueError(f"{label} must be YAML mappings")  # noqa: TRY004
    merged = dict(random_values)
    for key, value in tpe_values.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"Conflicting {label} for {key}: {merged[key]!r} != {value!r}")
        merged[key] = value
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_config", required=True, help="Shared Random/TPE sweep YAML")
    ap.add_argument("--conda_env", default="dl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry", action="store_true", help="Validate and preview import; do not train or write")
    args = ap.parse_args()

    # One shared YAML controls both Random Search and TPE.
    sweep = load_config(Path(args.sweep_config))
    validate_sweep_config(sweep)
    run_cfg = sweep.get("run", {})
    opt_cfg = sweep["optuna"]
    n_runs = opt_cfg["n_runs"]
    if type(n_runs) is not int or n_runs < 1:
        raise ValueError("optuna.n_runs must be a positive integer")
    exp_name = opt_cfg["exp_name"]
    study_name = opt_cfg["study_name"]
    storage_path = Path(opt_cfg["storage"])
    output_root = Path(opt_cfg.get("output_root", "runs"))
    source = Path(opt_cfg["import_dir"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Random sweep import directory does not exist: {source}")

    specs = sweep["parameters"]
    choices = {}

    for name, spec in specs.items():
        values = spec["values"]
        encoded = [encode(v) for v in values]

        if len(set(map(repr, encoded))) != len(encoded):
            raise ValueError(f"Duplicate categorical choices for {name}")

        choices[name] = values

    distributions = {
        name: CategoricalDistribution([encode(v) for v in values])
        for name, values in choices.items()
    }
    fixed = sweep.get("fixed", {})
    overrides = run_cfg.get("overrides", {})

    train_config = run_cfg.get("train_config", "configs/train.yaml")

    signature_payload = {
        "choices": {k: [encode(v) for v in vs] for k, vs in choices.items()},
        "fixed": fixed, "overrides": overrides, "train_config": train_config,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, default=str).encode()
    ).hexdigest()

    if args.dry:
        print(f"[dry] Shared search space: {args.sweep_config} ({len(choices)} parameters)")
        print(f"[dry] Import source: {source}")
        print(f"[dry] TPE experiment: {exp_name}; additional trials: {n_runs}")
        return

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path.resolve().as_posix()}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    existing_signature = study.user_attrs.get("sweep_signature")
    if existing_signature is not None and existing_signature != signature:
        raise ValueError("This Optuna study was created with a different search space/settings")
    if existing_signature is None:
        # Existing studies predating this check cannot be fully verified automatically.
        if study.trials:
            print("[warning] Existing study has no configuration fingerprint; checking existing trial distributions")
            for old in study.trials:
                for name, dist in old.distributions.items():
                    if name not in distributions or dist != distributions[name]:
                        raise ValueError(f"Existing trial {old.number} has incompatible distribution for {name}")
        study.set_user_attr("sweep_signature", signature)

    baseline = study.user_attrs.get("expected_run_units")
    if baseline is not None:
        baseline = [tuple(x) for x in baseline]

    def verify_units(units: list[tuple[str, int]]) -> None:
        nonlocal baseline
        if baseline is None:
            baseline = units
            study.set_user_attr("expected_run_units", units)
        elif units != baseline:
            raise ValueError("Run units differ from previously imported/trained trials")

    # Automatically import completed, compatible random-search runs on every start.
    imported_sources = {
        t.user_attrs.get("source_run_dir")
        for t in study.trials if t.user_attrs.get("source_run_dir")
    }
    imported = already_imported = rejected = 0
    for run_dir in sorted(source.iterdir()):
        if not run_dir.is_dir():
            continue
        source_id = str(run_dir.resolve())
        if source_id in imported_sources:
            already_imported += 1
            continue
        if not (run_dir / "config_used.yaml").is_file() or not (run_dir / "metrics_aggregate.json").is_file():
            rejected += 1
            continue
        try:
            cfg = load_config(run_dir / "config_used.yaml")
            for name, value in {**overrides, **fixed}.items():
                actual = normalize_param(get_nested(cfg, name), [value])
                if actual != value:
                    raise ValueError(f"{name} differs from random/TPE sweep settings")
            params = {
                name: encode(normalize_param(get_nested(cfg, name), choices[name]))
                for name in choices
            }
            for name, value in params.items():
                if value not in distributions[name].choices:
                    raise ValueError(f"{name}={value!r} outside random sweep search space")
            score, units = read_metrics(run_dir)
            verify_units(units)
            trial = create_trial(
                params=params,
                distributions=distributions,
                value=score,
                user_attrs={"source_run_dir": source_id, "origin": "random_search"},
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"[skip] {run_dir}: {exc}")
            rejected += 1
            continue
        study.add_trial(trial)
        imported_sources.add(source_id)
        imported += 1
    print(f"Random import: {imported} new, {already_imported} already imported, "
          f"{rejected} skipped; study total: {len(study.trials)}")
    if not imported and not already_imported:
        raise RuntimeError("No compatible random-search runs were imported. "
                           "Check import_dir, fixed settings and search space before starting TPE.")

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: decode(trial.suggest_categorical(name, dist.choices), choices[name])
            for name, dist in distributions.items()
        }
        run_id = f"trial_{trial.number:05d}"
        run_dir = output_root / exp_name / run_id
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
        cmd = [
            "conda", "run", "-n", args.conda_env, "--no-capture-output",
            "python3", "src/train.py", "--config", str(train_config),
            "--run_id", run_id,
            "--set", f"exp_name={exp_name}",
            "--set", f"out.dir={output_root}",
        ] + to_set_args(overrides) + to_set_args(fixed) + to_set_args(params)
        trial.set_user_attr("run_dir", str(run_dir.resolve()))
        print(f"\n[trial {trial.number}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        score, units = read_metrics(run_dir)
        verify_units(units)
        return score

    study.optimize(objective, n_trials=n_runs)
    print(f"Best trial: {study.best_trial.number}; MAE: {study.best_value:.8f}")
    print("Best parameters:", study.best_trial.params)


if __name__ == "__main__":
    main()