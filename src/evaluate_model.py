import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from dataloader import TimeSeriesLoader

from eval.inference import infer_test
from eval.io import ensure_dir, load_yaml, suppress_output
from eval.reporting import (
    make_sweep_val_figures,
    make_test_figures,
    make_val_compare_figures,
    save_leaderboard,
    write_val_aggregate_outputs,
)
from eval.test_eval import (
    discover_trained_run_units,
    run_psml_evaluator_and_save,
    write_psml_prediction_csv,
)
from eval.val_eval import aggregate_val_over_loc_year_from_zone_metrics
from model_utils import (
    build_model_from_cfg,
    get_output_dims_from_run,
    load_best_checkpoint,
)


def list_run_dirs(exp_or_run_dir: Path) -> list[Path]:
    """If exp_or_run_dir is a run dir -> [it]. If experiment dir -> all subdirs that contain config_used.yaml."""
    p = exp_or_run_dir.expanduser().resolve()
    if (p / "config_used.yaml").exists():
        return [p]

    runs = []
    for sub in p.iterdir():
        if sub.is_dir() and (sub / "config_used.yaml").exists():
            runs.append(sub)

    runs.sort(key=lambda x: x.stat().st_mtime, reverse=False)
    return runs

def collect_val_run_dirs(sources: list[Path]) -> list[Path]:
    """Collect unique runs from experiment directories or individual run dirs."""
    seen: set[Path] = set()
    runs: list[Path] = []
    for source in sources:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Validation run directory not found: {source}")
        for run in list_run_dirs(source):
            run = run.resolve()
            if run not in seen:
                seen.add(run)
                runs.append(run)
    return runs


def read_best_val_mae(run_dir: Path) -> float | None:
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
        v = m.get("best_val_mae", None)
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None

def build_benchmark_dataloaders(
    cfg_run: dict[str, Any],
    loc: str,
    year: int,
):
    """
    Rebuild the benchmark dataloaders for one trained loc/year unit.
    """
    loader = TimeSeriesLoader(
        task="forecasting",
        root=cfg_run["data"]["root"],
    )

    train_loader, test_loader = loader.load(
        batch_size=int(cfg_run["data"]["batch_size"]),
        shuffle=False,
        sliding_window=int(
            cfg_run["data"].get("sliding_window", 120)
        ),
        loc=loc,
        year=year,
    )

    return train_loader, test_loader

def eval_one_run(
    run_dir: Path,
    eval_cfg: dict[str, Any],
    eval_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg_run = load_yaml(run_dir / "config_used.yaml")

    mode = str(eval_cfg["mode"]).lower()
    exp_name = cfg_run["exp_name"]

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    # Training already saved validation metrics separately for each loc/year.
    # Therefore, validation evaluation aggregates the existing metrics instead
    # of rebuilding models and running inference again.
    if mode == "val":
        zones_dir = run_dir / "zones"

        if not zones_dir.exists():
            raise FileNotFoundError(
                f"Zones directory required for validation aggregation: {zones_dir}"
            )

        agg = aggregate_val_over_loc_year_from_zone_metrics(run_dir)

        if not agg.get("per_loc_year"):
            raise RuntimeError(
                f"No validation metrics found below: {zones_dir}"
            )

        df_per = pd.DataFrame(agg["per_loc_year"])

        agg_for_report = dict(agg)
        agg_for_report["objective"] = "val_mae_mean_over_targets"

        paths = write_val_aggregate_outputs(
            run_dir=run_dir,
            exp_name=exp_name,
            df_per=df_per,
            agg=agg_for_report,
        )


        return {
            "run_dir": str(run_dir),
            "run_id": cfg_run.get("meta", {}).get("run_id", run_dir.name),
            "exp_name": exp_name,
            "mode": mode,
            "val_n_experiments": agg.get("n_experiments"),
            "val_mean_over_loc_year": agg.get("macro_mean"),
            "val_median_over_loc_year": agg.get("macro_median"),
            "val_std_over_loc_year": agg.get("macro_std"),
            "val_aggregate_json": paths["aggregate_json"],
            "val_per_loc_year_csv": paths["per_loc_year_csv"],
            "val_aggregate_plot": paths["aggregate_plot"]
        }

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------
    if mode != "test":
        raise ValueError(
            f"Unsupported evaluation mode: {mode}. "
            "Expected 'val' or 'test'."
        )

    loader_type = cfg_run["data"].get("loader_type", "benchmark")

    if loader_type != "benchmark":
        raise NotImplementedError(
            "The PSML test evaluator currently supports only runs "
            "trained with loader_type='benchmark'."
        )

    overwrite = bool(eval_cfg.get("overwrite", False))
    test_cfg = eval_cfg.get("test", {})
    prediction_interval = float(
        test_cfg.get("prediction_interval", 0.95)
    )

    eval_root = run_dir / "evaluation"
    test_dir = eval_root / "test"
    pred_dir = test_dir / "predictions"
    ensure_dir(pred_dir)

    # Remove stale prediction files before rebuilding the evaluation.
    # Otherwise, the PSML evaluator could include CSVs from an older run.
    if overwrite:
        for old_csv in pred_dir.glob("*.csv"):
            old_csv.unlink()

    output_dims = get_output_dims_from_run(cfg_run)

    # The actual checkpoints are the source of truth for the trained units.
    run_units = discover_trained_run_units(run_dir)

    print(
        f"[test] found {len(run_units)} trained "
        f"(loc, year) unit(s)"
    )

    for index, (loc, year) in enumerate(run_units, start=1):
        print(
            f"[test] ({index}/{len(run_units)}) "
            f"{loc}, {year}"
        )

        unit_dir = run_dir / "zones" / loc / str(year)

        # Every loc/year unit has its own independently trained checkpoint.
        model = build_model_from_cfg(cfg_run).to(device)
        load_best_checkpoint(model, unit_dir, device)

        # Recreate exactly the benchmark test loader belonging to this unit.
        with suppress_output():
            _, test_loader = build_benchmark_dataloaders(
                cfg_run,
                loc=loc,
                year=year,
            )

        pred = infer_test(
            model,
            test_loader,
            device,
        )

        out_csv = pred_dir / f"{loc}_{year}.csv"

        write_psml_prediction_csv(
            pred,
            out_csv,
            output_dims,
            overwrite=overwrite,
            interval="degenerate",
        )

        print(f"[test] wrote predictions: {out_csv}")

    out_summary = test_dir / "summary.json"

    test_out = run_psml_evaluator_and_save(
        psml_root=Path(cfg_run["data"]["root"]),
        prediction_dir=pred_dir,
        out_summary_json=out_summary,
        overwrite=overwrite,
        prediction_interval=prediction_interval,
    )


    test_figures = make_test_figures(
        eval_root=eval_root,
        eval_cfg=eval_cfg,
        eval_path=eval_path,
        overwrite=overwrite,
    )

    for figure_path in test_figures:
        print(f"[test] wrote figure: {figure_path}")

    return {
        "run_dir": str(run_dir),
        "run_id": cfg_run.get("meta", {}).get("run_id", run_dir.name),
        "exp_name": exp_name,
        "mode": mode,
        "val_metrics_path": None,
        "test_summary_path": str(test_out),
        "best_val_mae_train": read_best_val_mae(run_dir),
    }

def main() -> int:
    ap = argparse.ArgumentParser("evaluate_model.py (Layer 2a/2b)")
    ap.add_argument("--config", type=str, default=None, help="train config (to derive run_dir)")
    ap.add_argument("--run_dir", type=str, default=None, help="runs/<exp_name>/ (direct)")
    ap.add_argument("--eval_config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 ...")
    args = ap.parse_args()

    eval_path = Path(args.eval_config).expanduser().resolve()
    cfg_eval_file = load_yaml(eval_path)
    eval_cfg = cfg_eval_file.get("eval", cfg_eval_file)
    mode = str(eval_cfg["mode"]).lower()

    val_cfg = eval_cfg.get("val", {}) if mode == "val" else {}
    configured_dirs = val_cfg.get("sweep_dirs") if mode == "val" else None
    multi_val = configured_dirs is not None

    if multi_val:
        if not isinstance(configured_dirs, list) or not configured_dirs or not all(
            isinstance(p, str) and p.strip() for p in configured_dirs
        ):
            raise ValueError("eval.val.sweep_dirs must be a non-empty list of paths")
        # Paths are relative to project working directory, like train.yaml out.dir.
        sources = [Path(p).expanduser().resolve() for p in configured_dirs]
        run_dirs = collect_val_run_dirs(sources)
        if not run_dirs:
            raise FileNotFoundError(f"No runs found in eval.val.sweep_dirs: {sources}")
        # A combined validation is identified by its source experiment directories.
        # Preserve the YAML order, but do not repeat the same source twice.
        sources = list(dict.fromkeys(sources))
        names = [source.name if not (source / "config_used.yaml").exists()
                 else f"{source.parent.name}_{source.name}" for source in sources]
        if len(set(names)) != len(names):
            raise ValueError("Sweep directory names are ambiguous; use unique experiment directory names")
        exp_name = "__".join(names)
        # For regular sweeps this is <project>/runs/evaluations/<source names>/.
        parents = {source.parent for source in sources}
        runs_root = next(iter(parents)) if len(parents) == 1 else Path("runs").resolve()
        output_dir = runs_root / "evaluations" / exp_name
        if any(output_dir == rd or rd in output_dir.parents for rd in run_dirs):
            raise ValueError("Combined evaluation output must be outside the source runs")
        validation_root = output_dir / "validation"
        print(f"[eval] validation sources: {', '.join(map(str, sources))}")
    else:
        # Preserve the original selection rules for test and single-directory val.
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
        run_dirs = list_run_dirs(run_dir)
        if not run_dirs:
            raise FileNotFoundError(f"No runs found in: {run_dir}")
        exp_name = run_dir.name
        validation_root = run_dir / "validation"

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[env] device = {device}")
    print(f"[eval] found {len(run_dirs)} run(s)")
    results = []

    for i, rd in enumerate(run_dirs, start=1):
        print(f"[eval] ({i}/{len(run_dirs)}) {rd}")
        res = eval_one_run(rd, eval_cfg=eval_cfg, eval_path=eval_path, device=device)
        if mode == "val":
            res["run_key"] = f"{rd.parent.name}/{rd.name}"
            res["source_experiment"] = rd.parent.name
        results.append(res)

    if mode == "val":
        if multi_val:
            ensure_dir(output_dir)
            manifest = {
                "mode": "val",
                "name": exp_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sources": [str(source) for source in sources],
                "n_runs": len(results),
                "runs": [str(Path(result["run_dir"]).resolve()) for result in results],
            }
            with (output_dir / "sources.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
            print(f"[eval] wrote sources: {output_dir / 'sources.yaml'}")
        df_lb = pd.DataFrame(results)
        leaderboard_paths = save_leaderboard(
            exp_eval_root=validation_root,
            df_lb=df_lb,
            metric_col="val_mean_over_loc_year",
            exp_name=exp_name,
        )
        print(f"\n[leaderboard] wrote: {leaderboard_paths['leaderboard_csv']}")

        if bool(val_cfg.get("sweep_analysis_enabled", False)):
            figures = make_sweep_val_figures(
                validation_root.parent,
                include_incomplete=bool(val_cfg.get("include_incomplete", False)),
            )
            print(f"[sweep analysis] wrote {len(figures)} figure(s)")

        if bool(val_cfg.get("compare_enabled", False)):
            runs_root = validation_root.parent.parent
            compare_root = runs_root / "compare" / "validation"
            compare_figures = make_val_compare_figures(
                compare_root=compare_root,
                eval_cfg=eval_cfg,
                eval_path=eval_path,
                overwrite=bool(eval_cfg.get("overwrite", False)),
            )
            for path in compare_figures:
                print(f"[val compare] wrote: {path.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())