### this will be where to start the training loop
# imports
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from utils import load_config, fix_seed, make_loss, get_device, ensure_dir, save_json, git_hash
from eval import eval_mae_per_target, eval_mae
from dataloader import TimeSeriesLoader
from models.mlp import STLFMLP

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
    cfg = load_config(cfg_path)
    print("Loaded config:", cfg["exp_name"])
    fix_seed(cfg["seed"], deterministic=True) # we want to use a fixed seed and a deterministic algorithm to ensure reproducibility
    device = get_device(cfg["device"])

    # trying to use the dataloader as it is used in the benchmark for comparability (implement another dataloader later for performance)
    loader = TimeSeriesLoader(task='forecasting', root=cfg["data"]["root"])

    # train loader full will be used as it is but we need an evaluation set and therefore split the test set
    train_loader_full, test_loader = loader.load(
        batch_size=cfg["data"]["batch_size"],
        shuffle=cfg["data"]["shuffle"]  
    )  

    # split a val from training set (makeing sure it is chronological)
    full_ds = train_loader_full.dataset
    N = len(full_ds)
    val_len = int(N * cfg["data"]["val_ratio"])
    train_len = N - val_len
    train_idx = list(range(0, train_len))
    val_idx   = list(range(train_len, N))

    # optional : num_workers or pin_memory in config to increase training (only use if enough RAM and training is too slow)
    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=cfg["data"]["batch_size"],
                              shuffle=cfg["data"]["shuffle"],
                              drop_last=False)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=cfg["data"]["batch_size"],
                              shuffle=False,
                              drop_last=False)

    # the model architecture
    model = STLFMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        out_dim=cfg["model"]["out_dim"],
        dropout=cfg["model"]["dropout"],
        use_layernorm=cfg["model"]["use_layernorm"],
        use_input_norm=cfg["model"]["use_input_norm"],
        activation=cfg["model"]["activation"],
    ).to(device)

    # using this only for training data to improve training but NEVER on val or test, input normalizer active is prob always the best option so I remains active
    model.fit_input_normalizer_from_loader(train_loader)

    # training loop and checkpoint to save the best model
    exp_dir  = os.path.join(cfg["out"]["dir"], cfg["exp_name"])
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ensure_dir(exp_dir)
    best_val, history = train_loop(model, train_loader, val_loader, device, cfg, ckpt_dir)

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
        "config": cfg,
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