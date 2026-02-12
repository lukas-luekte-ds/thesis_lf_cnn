### this will be where to start the training loop
# imports
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml
import importlib
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from utils import load_config,copy_config_file, fix_seed, make_loss, get_device, ensure_dir, save_json, git_hash
from eval import eval_mae_per_target, eval_mae
from dataloader import TimeSeriesLoader

def parse_args():
    # making the default path robust
    here = Path(__file__).resolve().parent
    default_cfg = (here / "configs" / "mlp.yaml").as_posix()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=default_cfg)

    # use Jupyter args:
    args, _ = ap.parse_known_args()

    # making the arg parsed config path robust
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Couldn't find config: {cfg_path}")
    args.config = cfg_path.as_posix()
    return args

# need this to auto generate a merged config, which is used to define the training + model which was used (without hart coding it)
def deep_merge(a: dict, b: dict) -> dict:
    """Return merged dict (b overrides a)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def save_config_used(run_cfg: dict, run_dir: str, source_train_cfg_path: str, source_model_cfg_path: str | None):
    cfg_used = deepcopy(run_cfg)
    cfg_used.setdefault("meta", {})
    cfg_used["meta"]["created_at"] = datetime.now().isoformat(timespec="seconds")
    cfg_used["meta"]["source_train_config"] = str(source_train_cfg_path)
    if source_model_cfg_path is not None:
        cfg_used["meta"]["source_model_config"] = str(source_model_cfg_path)
    # optional: git hash, command line, hostname ...

    out_path = Path(run_dir) / "config_used.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False)

def instantiate_from_module_class(module_name: str, class_name: str, kwargs: dict):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


# define the train loop
def train_loop(model, train_loader, val_loader, device, cfg, ckpt_dir):
    opt_name = cfg["train"]["optimizer"].lower()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                weight_decay=cfg["train"]["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    patience = cfg["train"]["early_stopping_patience"]
    min_delta = cfg["train"]["early_stopping_min_delta"] 

    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_mae": []}
    os.makedirs(ckpt_dir, exist_ok=True)
    loss_fn = make_loss(cfg)

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        batch_losses = []
        for x, y, _ in train_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # optional: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_mae = eval_mae(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)

        improved = val_mae < (best_val - min_delta)
        if improved:
            best_val = val_mae
            wait = 0
            torch.save({"model": model.state_dict()}, os.path.join(ckpt_dir, "best-val.pt"))
        else:
            wait += 1

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_MAE {val_mae:.4f} | best {best_val:.4f} | wait {wait}")
        if patience and wait >= patience:
            print("Early stopping.")
            break

    return best_val, history

# this is the real training loop
def main(cfg_path):
    train_cfg = load_config(cfg_path)

    model_ref = train_cfg.get("model_ref")
    if not model_ref:
        raise KeyError("train config must define model_ref")

    model_cfg = load_config(model_ref)

    cfg_used = deep_merge(train_cfg, model_cfg)

    print("Loaded config:", cfg_used["exp_name"])
    fix_seed(cfg_used["seed"], deterministic=True) # we want to use a fixed seed and a deterministic algorithm to ensure reproducibility
    device = get_device(cfg_used["device"])

    # trying to use the dataloader as it is used in the benchmark for comparability (implement another dataloader later for performance)
    loader = TimeSeriesLoader(task='forecasting', root=cfg_used["data"]["root"])

    # train loader full will be used as it is but we need an evaluation set and therefore split the test set
    train_loader_full, test_loader = loader.load(
        batch_size=cfg_used["data"]["batch_size"],
        shuffle=cfg_used["data"]["shuffle"]  
    )  

    # split a val from training set (makeing sure it is chronological)
    full_ds = train_loader_full.dataset
    N = len(full_ds)
    val_len = int(N * cfg_used["data"]["val_ratio"])
    train_len = N - val_len
    train_idx = list(range(0, train_len))
    val_idx   = list(range(train_len, N))

    # optional : num_workers or pin_memory in config to increase training (only use if enough RAM and training is too slow)
    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=cfg_used["data"]["batch_size"],
                              shuffle=cfg_used["data"]["shuffle"],
                              drop_last=False)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=cfg_used["data"]["batch_size"],
                              shuffle=False,
                              drop_last=False)


    # the model architecture (dynamic)
    m = cfg_used["model"]
    module_name = m["module"]
    class_name  = m["class"]
    kwargs      = m.get("kwargs", {})

    model = instantiate_from_module_class(module_name, class_name, kwargs).to(device)


    # using this only for training data to improve training but NEVER on val or test, input normalizer active is prob always the best option so I remains active
    if hasattr(model, "fit_input_normalizer_from_loader"):
        model.fit_input_normalizer_from_loader(train_loader)

    # training loop and checkpoint to save the best model
    exp_dir  = os.path.join(cfg_used["out"]["dir"], cfg_used["exp_name"])
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ensure_dir(exp_dir)
    copy_config_file(cfg_path, exp_dir)
    # write self-contained config used for this run (train + model composed)
    save_config_used(cfg_used, exp_dir, source_train_cfg_path=cfg_path, source_model_cfg_path=model_ref)


    best_val, history = train_loop(model, train_loader, val_loader, device, cfg_used, ckpt_dir)

    # laod the best model and not the last one
    ckpt = torch.load(os.path.join(ckpt_dir, "best-val.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    model.eval()
    ids_all, preds_all = [], []
    with torch.no_grad():
        for ID, x in test_loader:
            x = x.to(device).float()
            p = model(x).cpu().numpy()
            ids_all.append(ID.cpu().numpy())
            preds_all.append(p)

    IDs = np.concatenate(ids_all, 0)
    P   = np.concatenate(preds_all, 0)

    cols = ["yl_t+60","yl_t+1440","yw_t+5","yw_t+30","ys_t+5","ys_t+30"]
    preds_df = pd.DataFrame(P, columns=cols)
    preds_df.insert(0, "ID", IDs)
    preds_df.to_csv(os.path.join(exp_dir, "preds.csv"), index=False)
    print("saved test predictions:", os.path.join(exp_dir, "preds.csv"))

    # collect meta data
    meta = {
        "git_hash": git_hash(),
        "config": cfg_used,
        "best_val_mae": best_val,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "history": history,
    }
    save_json(os.path.join(exp_dir, "metrics.json"), meta)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_mae"], label="val_mae")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "learning_curve.png"))

    per_target = eval_mae_per_target(model, val_loader, device)
    print("Val-MAE per target:", per_target)

    print("done. artifacts in:", exp_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args.config)