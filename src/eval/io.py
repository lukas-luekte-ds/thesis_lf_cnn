import json
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import yaml


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

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_text(path: Path, text: str, overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Set eval.overwrite: true to overwrite.")
    path.write_text(text, encoding="utf-8")

@contextmanager
def suppress_output():
    """
    the PSML dataloader produces annoying outputs so I suppress it and replace it with an easier one
    """
    with open(os.devnull, "w") as devnull:  # noqa: SIM117
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield