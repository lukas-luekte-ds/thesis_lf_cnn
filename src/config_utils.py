from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


# need this to auto generate a merged config, which is used to define the training + model which was used (without hard coding it)
def deep_merge(a: dict, b: dict) -> dict:
    """Return merged dict (b overrides a)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def to_set_args(values: dict[str, Any]) -> list[str]:
    """Convert configuration values into CLI --set arguments."""
    args = []

    for key, value in values.items():
        formatted = (
            yaml.safe_dump(value, default_flow_style=True)
            .removesuffix("...\n")
            .strip()
        )

        args.extend(["--set", f"{key}={formatted}"])

    return args

def save_config_used(run_cfg: dict, 
                     run_dir: str, 
                    source_train_cfg_path: str, 
                    source_model_cfg_path: str | None, 
                    overrides: list[str] | None = None, 
                    run_id: str | None = None):
    cfg_used = deepcopy(run_cfg)
    cfg_used.setdefault("meta", {})
    cfg_used["meta"]["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    cfg_used["meta"]["source_train_config"] = str(source_train_cfg_path)
    if source_model_cfg_path is not None:
        cfg_used["meta"]["source_model_config"] = str(source_model_cfg_path)
    if overrides is not None:
        cfg_used["meta"]["overrides"] = list(overrides)
    if run_id is not None:
        cfg_used["meta"]["run_id"] = run_id

    out_path = Path(run_dir) / "config_used.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False)

def _set_by_path(d: dict, path: str, value):
    """Set nested dict key by dot path, creating dicts if needed."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

# config utils
def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply overrides like ["train.lr=3e-4", "model.kwargs.dropout=0.2"].
    Values are parsed via yaml.safe_load so numbers/lists/bools work.
    Is used for the randomized sweep
    """
    for s in overrides:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}'. Use key=value.")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        parsed_v = yaml.safe_load(v)  # parses "3e-4", "[128,128]", "true", etc.
        _set_by_path(cfg, k, parsed_v)
    return cfg

def validate_sweep_config(cfg: dict) -> None:
    """Validate the structure of a sweep configuration."""

    parameters = cfg.get("parameters")

    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "'parameters' must be a non-empty dictionary."
        )

    fixed = cfg.get("fixed", {})
    run = cfg.get("run", {})

    if not isinstance(fixed, dict):
        raise ValueError("'fixed' must be a dictionary.")  # noqa: TRY004

    if not isinstance(run, dict):
        raise ValueError("'run' must be a dictionary.")  # noqa: TRY004

    overrides = run.get("overrides", {})

    if not isinstance(overrides, dict):
        raise ValueError(  # noqa: TRY004
            "'run.overrides' must be a dictionary."
        )

    for name, spec in parameters.items():

        if not isinstance(spec, dict):
            raise ValueError(  # noqa: TRY004
                f"Parameter '{name}' must be a dictionary."
            )

        values = spec.get("values")

        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Parameter '{name}' must define "
                "a non-empty 'values' list."
            )

    overlap = set(parameters) & (set(fixed) | set(overrides))

    if overlap:
        raise ValueError(
            f"Parameters cannot also be fixed or overridden: "
            f"{sorted(overlap)}"
        )
