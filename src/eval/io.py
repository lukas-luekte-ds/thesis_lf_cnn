from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import yaml
import json
from pathlib import Path
import numpy as np
from typing import Any, Dict

def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def load_yaml(path: Path):
    """
    wrapper to make the loading of yaml a bit more readable
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(path: Path, obj: Dict[str, Any], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Set eval.overwrite: true to overwrite.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_text(path: Path, text: str, overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Set eval.overwrite: true to overwrite.")
    path.write_text(text, encoding="utf-8")

def _to_builtin(obj: Any) -> Any:
    """Minimal converter for json.dump on evaluator outputs (numpy scalars, dicts, lists)."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj

@contextmanager
def suppress_output():
    """
    the dataloader produces annoying outputs so I suppress it and replace it with an easier one
    """
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield