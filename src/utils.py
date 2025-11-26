import os
import json
import random
import subprocess
from typing import Any, Dict
import numpy as np
import torch

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
        except Exception:
            pass

def get_device(pref: str="cuda:0"):
    return torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")

def mae(y, p): return float(np.mean(np.abs(p - y)))
def rmse(y, p): return float(np.sqrt(np.mean((p - y)**2)))
def mape(y, p, eps=1e-6):
    denom = np.clip(np.abs(y), eps, None)
    return float(np.mean(np.abs((p - y)/denom)))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def git_hash():
    try: 
        return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception: 
        return "unknown"
