import argparse
from pathlib import Path
from typing import Any, Dict
from eval.inference import infer_test, infer_val, subset_preds_by_ids
from eval.io import ensure_dir, load_yaml, suppress_output
from eval.model import build_model_from_run_cfg, load_best_checkpoint
from dataloader import TimeSeriesLoader
from eval.reporting import report_and_save
from eval.test_eval import discover_psml_targets, load_test_ids_for_loc_year, run_psml_evaluator_and_save, write_psml_prediction_csv
from eval.val_eval import eval_val_and_save
import numpy as np
import torch

def build_dataloaders(cfg_run: Dict[str, Any]):
    loader = TimeSeriesLoader(
        task="forecasting",
        root=cfg_run["data"]["root"],
    )
    train_loader, test_loader = loader.load(
        batch_size=int(cfg_run["data"]["batch_size"]),
        shuffle=False,
    )
    return train_loader, test_loader

def main() -> int:
    ap = argparse.ArgumentParser("evaluate_model.py (Layer 2a/2b)")
    ap.add_argument("--config", type=str, default=None, help="train config (to derive run_dir)")
    ap.add_argument("--run_dir", type=str, default=None, help="runs/<exp_name>/ (direct)")
    ap.add_argument("--eval_config", type=str, required=True)

    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 ...")
    args = ap.parse_args()

    if (args.config is None) == (args.run_dir is None):
        raise ValueError("Provide exactly one of --config or --run_dir")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        train_cfg = load_yaml(Path(args.config).expanduser().resolve())
        run_dir = Path(train_cfg["out"]["dir"]) / train_cfg["exp_name"]
        run_dir = run_dir.expanduser().resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    cfg_run = load_yaml(run_dir / "config_used.yaml")

    eval_path = Path(args.eval_config).expanduser().resolve()
    cfg_eval_file = load_yaml(eval_path)
    eval_cfg = cfg_eval_file.get("eval", cfg_eval_file)  # allow eval: {...} or root dict

    mode = eval_cfg["mode"]

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[env] device = {device}")

    # 1) build loaders (but suppress the annoying output of the dataloader)
    with suppress_output():
        train_loader_run, test_loader_run = build_dataloaders(cfg_run)
    print("[data] PSML dataloader ready")

    # 2) build model (dynamic) + load checkpoint from run_dir
    model = build_model_from_run_cfg(cfg_run).to(device)
    load_best_checkpoint(model, run_dir, device)

    # 3) infer
    if mode == "val":
        pred = infer_val(model, train_loader_run, device)
    elif mode == "test":
        pred = infer_test(model, test_loader_run, device)

    # 4) print summary checks
    print(f"[infer] mode={mode} done")
    print("  ids:", pred.ids.shape, pred.ids.dtype)
    print("  y_pred:", pred.y_pred.shape, pred.y_pred.dtype)
    print("  y_true:", None if pred.y_true is None else pred.y_true.shape)
    print("  eval_mask:", None if pred.eval_mask is None else pred.eval_mask.shape)

    out_dims = eval_cfg["output_dims"]  # required
    D = pred.y_pred.shape[1] if pred.y_pred.ndim == 2 else 0
    if D != len(out_dims):
        print(f"[warn] D={D} but len(output_dims)={len(out_dims)}")
    else:
        print(f"[ok] D matches output_dims ({D})")

    if mode == "test":
        if len(np.unique(pred.ids)) != len(pred.ids):
            raise AssertionError("Test IDs are not unique!")
        #print("  first 5 ids:", pred.ids[:5])

    # Layer 2a: Validation evaluation + write JSON
    if mode == "val":
        exp_name = cfg_run["exp_name"]
        out_path = eval_val_and_save(preds=pred, run_dir=run_dir, exp_name=exp_name, eval_cfg=eval_cfg)
        print(f"[val] wrote: {out_path}")

    # layer 2b: Test evaluation
    if mode == "test":
        exp_name = cfg_run["exp_name"]
        tag = eval_cfg["tag"]
        overwrite = bool(eval_cfg.get("overwrite", False))

        test_cfg = eval_cfg["test"]
        prediction_interval = float(test_cfg.get("prediction_interval", 0.95))

        eval_root = run_dir / f"{exp_name}_{tag}"
        test_dir = eval_root / "test"
        pred_dir = test_dir / "predictions"
        ensure_dir(pred_dir)


        # 1) discover targets from processed_dataset folder
        targets = discover_psml_targets(cfg_run, eval_cfg)
        print(f"[test] discovered {len(targets)} loc_year targets")

        pred_all = pred

        # optional sanity
        if len(np.unique(pred_all.ids)) != len(pred_all.ids):
            raise AssertionError("Global test IDs are not unique!")


        # 2) per loc_year: load IDs -> subset -> write csv
        for i, (loc, year) in enumerate(targets, start=1):
            loc_year = f"{loc}_{year}"
            print(f"[test] ({i}/{len(targets)}) {loc_year}")
            
            ids_loc_year = load_test_ids_for_loc_year(cfg_run, eval_cfg, loc, year)
            #print("[debug] global ids: min/max", pred_all.ids.min(), pred_all.ids.max(), "dtype", pred_all.ids.dtype)
            #print("[debug] loc_year ids: min/max", ids_loc_year.min(), ids_loc_year.max(), "dtype", ids_loc_year.dtype)
            pred_sub = subset_preds_by_ids(pred_all, ids_loc_year)

            out_csv = pred_dir / f"{loc_year}.csv"
            write_psml_prediction_csv(
                pred_sub,
                out_csv,
                list(eval_cfg["output_dims"]),
                overwrite=overwrite,
                interval="degenerate",
            )

        # 3) run evaluator once over all CSVs
        out_summary = test_dir / "summary.json"  # <-- hierhin!
        out_summary = run_psml_evaluator_and_save(
            psml_root=Path(cfg_run["data"]["root"]),
            prediction_dir=pred_dir,          # <-- CSVs hier
            out_summary_json=out_summary,     # <-- Summary hier
            overwrite=overwrite,
            prediction_interval=prediction_interval,
        )
        print(f"[test] wrote: {out_summary}")

    #layer 3
    exp_name = cfg_run["exp_name"]
    report_path = report_and_save(
        run_dir=run_dir,
        exp_name=exp_name,
        eval_cfg=eval_cfg,
        cfg_run=cfg_run,
        eval_path=eval_path,
    )
    print(f"[report] wrote: {report_path}")

    return 0
if __name__ == "__main__":
    raise SystemExit(main())
