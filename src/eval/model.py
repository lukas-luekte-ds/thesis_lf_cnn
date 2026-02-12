import importlib
from pathlib import Path
import torch

def build_model_from_run_cfg(cfg_run):
    m = cfg_run["model"]
    module = importlib.import_module(m["module"])
    cls = getattr(module, m["class"])
    return cls(**m.get("kwargs", {}))

def load_best_checkpoint(model, run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "checkpoints" / "best-val.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)