import os
import json
import random
import subprocess
from typing import Any, Dict
import numpy as np
import torch
import yaml
import shutil

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

# put this to utils or somewhere else, since loading configs will be used in every model
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def copy_config_file(src_config_path: str, exp_dir: str):
    os.makedirs(exp_dir, exist_ok=True)
    dst_path = os.path.join(exp_dir, "config_used.yaml")
    shutil.copyfile(src_config_path, dst_path)

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
    
def make_loss(cfg):
    lt = cfg["train"]["loss_type"].lower()
    delta = cfg["train"]["huber_delta"]

    if lt == "mae":
        def loss_fun(pred, y):
            return (pred - y).abs().mean()
        return loss_fun

    if lt == "mse":
        def loss_fun(pred, y):
            return torch.nn.functional.mse_loss(pred, y)
        return loss_fun

    if lt == "huber":
        def loss_fun(pred, y):
            return torch.nn.functional.huber_loss(pred, y, delta=delta)
        return loss_fun

    raise ValueError(...)
