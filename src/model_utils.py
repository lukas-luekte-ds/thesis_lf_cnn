import importlib
from pathlib import Path

import torch


def build_model_from_cfg(cfg_run):
    m = cfg_run["model"]
    module = importlib.import_module(m["module"])
    cls = getattr(module, m["class"])
    return cls(**m.get("kwargs", {}))

def load_best_checkpoint(model, model_dir: Path, device: torch.device):
    ckpt_path = model_dir / "checkpoints" / "best-val.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)

def get_output_dims_from_run(cfg_run: dict) -> list[str]:
    loader_type = cfg_run["data"].get("loader_type", "benchmark")

    if loader_type == "benchmark":
        return ["yl_t+60", "yl_t+1440", "yw_t+5", "yw_t+30", "ys_t+5", "ys_t+30"]

    targets = cfg_run["data"]["custom_loader"]["task"]["targets"]

    if isinstance(targets[0], dict):
        return [t["name"] for t in targets]

    return list(targets)