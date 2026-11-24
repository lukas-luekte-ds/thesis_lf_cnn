# imports
import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataloader import TimeSeriesLoader
from torch.utils.data import DataLoader, Subset

from config_utils import apply_overrides, deep_merge, save_config_used
from custom_dataloader import CustomTimeSeriesLoader
from eval_metrics import eval_mae, eval_mae_per_target
from model_utils import build_model_from_cfg, load_best_checkpoint
from training_utils import make_loss, make_optimizer
from utils import (
    ensure_dir,
    fix_seed,
    get_device,
    git_hash,
    load_config,
    make_run_id,
    save_json,
)


def parse_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str,  help="Path to training configuration YAML.")
    ap.add_argument("--set", action="append", default=[],
                    help='Override config values, e.g. --set train.lr=3e-4 --set model.kwargs.dropout=0.2')
    ap.add_argument("--run_id", type=str, default=None)

    # use Jupyter args:
    #args, _ = ap.parse_known_args()

    args = ap.parse_args()

    # making the arg parsed config path robust
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Couldn't find config: {cfg_path}")
    args.config = cfg_path.as_posix()
    return args


# define the train loop
def train_loop(model, train_loader, val_loader, device, cfg, ckpt_dir):
    patience = cfg["train"]["early_stopping_patience"]
    min_delta = cfg["train"]["early_stopping_min_delta"] 
    opt = make_optimizer(model, cfg)
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
            if pred.shape != y.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: pred.shape={tuple(pred.shape)}, y.shape={tuple(y.shape)}"
                )
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

def main(cfg_path, overrides=None, run_id=None):
    # ========= 1. load and prepare configuration ============================
    overrides = overrides or []
    train_cfg = load_config(cfg_path)

    model_ref = train_cfg.get("model_ref")
    if not model_ref:
        raise KeyError("train config must define model_ref")

    model_cfg = load_config(model_ref)

    cfg_used = deep_merge(train_cfg, model_cfg)

    cfg_used = apply_overrides(cfg_used, overrides)

    print("Loaded config:", cfg_used["exp_name"])
    fix_seed(cfg_used["seed"], deterministic=True)
    device = get_device(cfg_used["device"])

    # ============ 2. create experiment directory and save the used config =====
    exp_root = os.path.join(cfg_used["out"]["dir"], cfg_used["exp_name"])
    if run_id is None:
        run_id = make_run_id(exp_root)

    exp_dir = os.path.join(exp_root, run_id)
    if os.path.exists(exp_dir):
        raise FileExistsError(f"Run directory already exists: {exp_dir}")

    ensure_dir(exp_dir)
    save_config_used(cfg_used, exp_dir, source_train_cfg_path=cfg_path, source_model_cfg_path=model_ref, overrides=overrides, run_id=run_id)

    zones_dir = os.path.join(exp_dir, "zones")
    ensure_dir(zones_dir)

    # =========== 3. select data loader and define run units =================
    loader_type = cfg_used["data"].get("loader_type", "benchmark")

    zone_results = []

    if loader_type == "benchmark":
        processed_dir = Path(cfg_used["data"]["root"]) / "processed_dataset" / "forecasting"
        files = sorted(processed_dir.glob("*.csv"))

        benchmark_cfg = cfg_used["data"].get("benchmark_loader", {})

        include_grids = benchmark_cfg.get("include_grids", None)
        include_years = benchmark_cfg.get("include_years", None)
        include_zones = benchmark_cfg.get("include_zones", None)

        measured_zones = []
        for f in files:
            name = f.stem
            parts = name.split("_")

            if len(parts) < 4:
                continue
            grid = parts[0]
            if parts[1] != "zone":
                continue

            try:
                zone_num = int(parts[2])
                year = int(parts[3])
            except ValueError:
                continue

            loc = f"{grid}_zone_{zone_num}"

            if include_grids and grid not in include_grids:
                continue
            if include_years and year not in include_years:
                continue
            if include_zones and zone_num not in include_zones:
                continue

            measured_zones.append((loc, year))

        measured_zones = sorted(set(measured_zones), key=lambda t: (t[0], t[1]))

        if not measured_zones:
            raise RuntimeError(f"No measured_zones found in {processed_dir}")

        print(f"Found {len(measured_zones)} measured_zones")

        loader = TimeSeriesLoader(task='forecasting', root=cfg_used["data"]["root"])
        run_units = measured_zones

    elif loader_type == "custom":
        print("Using custom dataloader")

        loader = CustomTimeSeriesLoader(cfg_used["data"])
        run_units = [("custom_run", "all")]

    else:
        raise ValueError(f"Unknown loader_type: {loader_type}")

    # ======== 4. run training and evaluation for each run unit ================
    for z, y in run_units:
        # 4.1 Create output directories for this run unit
        zone_dir = os.path.join(zones_dir, str(z), str(y))
        ensure_dir(zone_dir)
        ckpt_dir = os.path.join(zone_dir, "checkpoints")
        ensure_dir(ckpt_dir)

        # 4.2 Load train/test data
        if loader_type == "benchmark":
            train_loader_full, test_loader = loader.load(
                batch_size=cfg_used["data"]["batch_size"],
                shuffle=cfg_used["data"]["shuffle"],
                sliding_window=cfg_used["data"].get("sliding_window", 120),
                loc=z,
                year=y
            )
        else:
            train_loader_full, test_loader = loader.load(
                batch_size=cfg_used["data"]["batch_size"],
                shuffle=cfg_used["data"]["shuffle"]
            )

        # 4.3 Create chronological train/validation split
        # split a val from training set (makeing sure it is chronological)
        full_ds = train_loader_full.dataset
        N = len(full_ds)
        val_len = int(N * cfg_used["data"]["val_ratio"])
        train_len = N - val_len
        train_idx = list(range(train_len))
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

        # 4.4 Infer model input/output dimensions for custom loader
        if loader_type == "custom":
            x_example, y_example, _ = next(iter(train_loader_full))

            # infer input_dim
            if x_example.ndim == 3:
                inferred_input_dim = x_example.shape[1] * x_example.shape[2]
            # just in case the input ist [B, F]
            elif x_example.ndim == 2:
                inferred_input_dim = x_example.shape[1]
            else:
                raise ValueError(f"Unexpected x shape from custom loader: {tuple(x_example.shape)}")

            # infer out_dim
            if y_example.ndim == 2:
                inferred_out_dim = y_example.shape[1]
            elif y_example.ndim == 1:
                inferred_out_dim = 1
            else:
                raise ValueError(f"Unexpected y shape from custom loader: {tuple(y_example.shape)}")

            cfg_used.setdefault("model", {})
            cfg_used["model"].setdefault("kwargs", {})
            cfg_used["model"]["kwargs"]["input_dim"] = inferred_input_dim
            cfg_used["model"]["kwargs"]["out_dim"] = inferred_out_dim

            print(f"[custom] inferred input_dim={inferred_input_dim}, out_dim={inferred_out_dim}")


        # 4.5 Instantiate (dynamic) model from config
        model = build_model_from_cfg(cfg_used).to(device)
        num_params_total = sum(p.numel() for p in model.parameters())
        num_params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
)


        # using this only for training data to improve training but NEVER on val or test, input normalizer active is prob always the best option so I remains active
        # 4.6 Fit optional input normalizer on training data only
        if hasattr(model, "fit_input_normalizer_from_loader"):
            model.fit_input_normalizer_from_loader(train_loader)

        # 4.7 Train model and reload best validation checkpoint
        if device.type == "cuda":
            torch.cuda.synchronize()

        train_start = time.perf_counter()

        best_val, history = train_loop(
            model, train_loader, val_loader, device, cfg_used, ckpt_dir
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        training_runtime_seconds = time.perf_counter() - train_start

        # load the best model and not the last one
        load_best_checkpoint(model, Path(zone_dir), device)

        model.eval()

        # 4.8 Generate and export test predictions
        preds_all = []

        with torch.no_grad():
            if loader_type == "benchmark":
                ids_all = []

                for ID, x in test_loader:
                    x = x.to(device).float()
                    p = model(x).cpu().numpy()
                    ids_all.append(ID.cpu().numpy())
                    preds_all.append(p)

                IDs = np.concatenate(ids_all, 0)
                P = np.concatenate(preds_all, 0)

                cols = ["yl_t+60","yl_t+1440","yw_t+5","yw_t+30","ys_t+5","ys_t+30"]
                preds_df = pd.DataFrame(P, columns=cols)
                preds_df.insert(0, "ID", IDs)

            else:
                targets_all = []

                for x, y_true, mask in test_loader:
                    x = x.to(device).float()
                    p = model(x).cpu().numpy()
                    preds_all.append(p)
                    targets_all.append(y_true.cpu().numpy())

                P = np.concatenate(preds_all, axis=0)
                Y = np.concatenate(targets_all, axis=0)

                P = np.asarray(P).reshape(len(P), -1)
                Y = np.asarray(Y).reshape(len(Y), -1)

                if P.shape[1] != Y.shape[1]:
                    raise ValueError(
                        f"Prediction/target export shape mismatch: P.shape={P.shape}, Y.shape={Y.shape}"
                    )

                base_ds = test_loader.dataset

                if hasattr(base_ds, "datasets"):
                    first_ds = base_ds.datasets[0]
                else:
                    first_ds = base_ds

                target_names = first_ds.target_names

                if len(target_names) != P.shape[1]:
                    raise ValueError(
                        f"Number of target names does not match output dimension: "
                        f"len(target_names)={len(target_names)}, P.shape[1]={P.shape[1]}"
                    )

                pred_cols = [f"pred_{name}" for name in target_names]
                target_cols = [f"target_{name}" for name in target_names]

                preds_df = pd.DataFrame(
                    np.concatenate([P, Y], axis=1),
                    columns=pred_cols + target_cols
                )

        preds_df.to_csv(os.path.join(zone_dir, "preds.csv"), index=False)
        print("saved test predictions:", os.path.join(zone_dir, "preds.csv"))

        # 4.9 Save validation metrics and training history
        # collect meta data
        zone_meta = {
            "loc": z,
            "year": y,
            "git_hash": git_hash(),
            "config": cfg_used,
            "best_val_mae": float(best_val),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "history": history,
            "efficiency": {
                "num_params_total": num_params_total,
                "num_params_trainable": num_params_trainable,
                "training_runtime_seconds": training_runtime_seconds,
                "runtime_scope": "training_with_validation",
            },
        }

        per_target = eval_mae_per_target(model, val_loader, device)
        print("Val-MAE per target:", per_target)
        zone_meta["val_mae_per_target"] = [float(v) for v in per_target]
        zone_meta["val_mae_mean_over_targets"] = float(np.mean(per_target))
        if loader_type == "custom":
            zone_meta["target_names"] = target_names

        save_json(os.path.join(zone_dir, "metrics.json"), zone_meta)

        # 4.10 Save learning curve plot
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_mae"], label="val_mae")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(zone_dir, "learning_curve.png"))
        plt.close()


        print("done. artifacts in:", zone_dir)

        # 4.11 Store run-unit summary for aggregate metrics
        zone_results.append({
            "loc": z,
            "year": y,
            "best_val_mae": float(best_val),
            "val_mae_mean_over_targets": float(np.mean(per_target)),
            "n_test": len(test_loader.dataset),
            "num_params_total": num_params_total,
            "num_params_trainable": num_params_trainable,
            "training_runtime_seconds": training_runtime_seconds,
        })
    # ---- aggregate over zones (macro avg) ----
    agg = {
        "loader_type": loader_type,
        "n_run_units": len(run_units),
        "run_units": run_units,
        "macro_avg_best_val_mae": float(
            np.mean([r["best_val_mae"] for r in zone_results])
        ) if zone_results else None,
        "macro_avg_val_mae_mean_over_targets": float(
            np.mean([
                r["val_mae_mean_over_targets"]
                for r in zone_results
            ])
        ) if zone_results else None,

        "efficiency": {
            "num_params_total": (
                zone_results[0]["num_params_total"]
                if zone_results else None
            ),
            "num_params_trainable": (
                zone_results[0]["num_params_trainable"]
                if zone_results else None
            ),
            "training_runtime_seconds_total": float(
                sum(r["training_runtime_seconds"] for r in zone_results)
            ),
            "training_runtime_seconds_mean_per_unit": float(
                np.mean([
                    r["training_runtime_seconds"]
                    for r in zone_results
                ])
            ) if zone_results else None,
            "runtime_scope": "training_with_validation",
        },
    }
    save_json(os.path.join(exp_dir, "metrics_aggregate.json"), agg)
    print("\nSaved aggregate metrics:", os.path.join(exp_dir, "metrics_aggregate.json"))
    print("done. artifacts in:", exp_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args.config, overrides=args.set, run_id=args.run_id)