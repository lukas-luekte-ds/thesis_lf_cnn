import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def make_run_id(exp_root: str) -> str:
    exp_path = Path(exp_root)

    if not exp_path.exists():
        return "id_001"

    existing = [
        p.name for p in exp_path.iterdir()
        if p.is_dir() and p.name.startswith("id_")
    ]

    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue

    next_id = max(nums) + 1 if nums else 1
    return f"id_{next_id:03d}"

# setting fixed seeds to guarantee reproducability (I hope I found all needed seeds)
def fix_seed(seed: int, deterministic: bool=True):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # this is only needed for multi GPU setup but better safe then sorry and it will be ignored if not needed
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

def get_device(pref: str="cuda:0"):
    return torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping: {path}")  # noqa: TRY004

    return cfg

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def save_json(
    path: str | Path,
    obj: Any,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. "
            "Pass overwrite=True to replace it."
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            _to_builtin(obj),
            f,
            indent=2,
            ensure_ascii=False,
        )

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
