import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .io import ensure_dir, load_json


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve path_str. If relative, resolve relative to base_dir."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir / p)
    return p.resolve()

def make_test_figures(
    *,
    eval_root: Path,
    eval_cfg: dict[str, Any],
    eval_path: Path,
    overwrite: bool,
) -> list[Path]:
    """Generate test comparison figures."""
    fig_dir = eval_root / "test" / "figures"
    ensure_dir(fig_dir)

    base_dir = eval_path.parent
    test_cfg = eval_cfg.get("test", {})

    if not isinstance(test_cfg, dict):
        return []

    include_current_run = bool(
        test_cfg.get("include_current_run", True)
    )

    items: list[dict[str, Any]] = []

    # Automatically include the run currently being evaluated.
    if include_current_run:
        current_summary_path = (
            eval_root / "test" / "summary.json"
        )

        if current_summary_path.exists():
            run_dir = eval_root.parent
            exp_name = run_dir.parent.name
            run_id = run_dir.name

            items.append(
                {
                    "name": f"ours:{exp_name}/{run_id}",
                    "path": str(current_summary_path),
                    "data": load_json(current_summary_path),
                }
            )
        else:
            print(
                "[warn] current run summary missing: "
                f"{current_summary_path}"
            )

    # Add manually configured comparison items.
    compare = test_cfg.get("compare", [])

    if compare is None:
        compare = []

    if not isinstance(compare, list):
        raise TypeError(
            "eval.test.compare must be a list."
        )

    if compare:
        compare_items = load_compare_items(
            compare=compare,
            base_dir=base_dir,
            warn_prefix="compare test",
        )
        items.extend(compare_items)

    if not items:
        return []

    # PSML summary -> aggregated metrics.
    for item in items:
        item["data"] = aggregate_psml_summary(
            item["data"]
        )

    outs: list[Path] = []

    for metric in ["RMSE", "MAE", "MSIS", "MAPE"]:
        out_png = (
            fig_dir / f"compare_overall_{metric}.png"
        )

        if out_png.exists() and not overwrite:
            outs.append(out_png)
            continue

        plot_compare_overall(
            items,
            out_png,
            metric=metric,
            getter=lambda item, metric: item["data"]["overall"][metric],
            ylabel=metric,
            title=(
                f"Test model comparison (overall): "
                f"{metric}"
            ),
            zero_line=False,
        )

        outs.append(out_png)

    return outs

def write_val_aggregate_outputs(run_dir: Path, exp_name: str, df_per: pd.DataFrame, agg: dict) -> dict:
    """
    Aggregates the outputs of the different (loc, year)-models from a single run.

    Writes:
      <run_dir>/evaluation/per_loc_year_val.csv
      <run_dir>/evaluation/aggregate_val.json
      <run_dir>/evaluation/aggregate_val_distribution.png
    Returns paths.
    """
    eval_root = run_dir / "evaluation"
    ensure_dir(eval_root)

    per_path = eval_root / "per_loc_year_val.csv"
    df_per.to_csv(per_path, index=False)

    agg_path = eval_root / "aggregate_val.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    # One plot per run: distribution + mean line
    plot_path = eval_root / "aggregate_val_distribution.png"
    vals = df_per[agg["objective"]].to_numpy(dtype=float)

    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=30)
    plt.axvline(float(agg["macro_mean"]))
    plt.xlabel(agg["objective"])
    plt.ylabel("count")
    plt.title(f"{exp_name} – {agg['objective']} over (loc,year)\nmean={agg['macro_mean']:.4f}, median={agg['macro_median']:.4f}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "per_loc_year_csv": str(per_path),
        "aggregate_json": str(agg_path),
        "aggregate_plot": str(plot_path),
    }

def save_leaderboard(exp_eval_root: Path, df_lb: pd.DataFrame, metric_col: str, exp_name: str) -> dict:
    """
    use for validation
    Writes:
      leaderboard.csv
      leaderboard.png
    """
    ensure_dir(exp_eval_root)
    lb_path = exp_eval_root / "leaderboard.csv"
    df_lb.to_csv(lb_path, index=False)

    plot_path = exp_eval_root / "leaderboard.png"
    df_sorted = df_lb.sort_values(metric_col).reset_index(drop=True)

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(df_sorted)), df_sorted[metric_col].to_numpy(dtype=float))
    plt.xlabel("Run (sorted)")
    plt.ylabel(metric_col)
    plt.xticks(range(len(df_sorted)), df_sorted.get("run_key", df_sorted["run_id"]).astype(str), rotation=45)
    plt.title(f"{exp_name} - {metric_col} (sorted)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {"leaderboard_csv": str(lb_path), "leaderboard_plot": str(plot_path)}


##---------- val------------------

def make_val_compare_figures(
    *,
    compare_root: Path,
    eval_cfg: dict[str, Any],
    eval_path: Path,
    overwrite: bool,
) -> list[Path]:
    val_cfg = eval_cfg.get("val", {})
    compare = val_cfg.get("compare", [])
    compare_name = str(val_cfg.get("compare_name", "val_compare"))
    compare_name = compare_name.strip().replace(" ", "_")

    if not isinstance(compare, list) or not compare:
        return []

    items = load_compare_items(
        compare=compare,
        base_dir=eval_path.parent,
        warn_prefix="compare val",
        path_resolver=resolve_val_compare_path,
    )

    if not items:
        return []

    ensure_dir(compare_root)

    out_png = compare_root / f"{compare_name}.png"

    if out_png.exists() and not overwrite:
        return [out_png]

    plot_compare_overall(
        items,
        out_png,
        metric="macro_mean",
        getter=lambda item, metric: item["data"][metric],
        ylabel="Validation MAE",
        title=compare_name.replace("_", " "),
        zero_line=False,
    )

    return [out_png]

def plot_compare_overall(
    items: list[dict[str, Any]],
    out_png: Path,
    *,
    metric: str,
    getter,
    ylabel: str,
    title: str,
    zero_line: bool = False,
) -> None:
    """ the compare plots between the different models.
    Can be used for validation and for tests
    (use the different getters for the test or val metrics)"""
    rows = []
    dropped = []
    for it in items:
        try:
            v = float(getter(it, metric))
        except (TypeError, ValueError):
            dropped.append(it.get("name", "?"))
            continue
        if np.isnan(v):
            dropped.append(it.get("name", "?"))
            continue
        rows.append((it["name"], v))

    if dropped:
        print(f"[warn] compare '{metric}': dropped {len(dropped)} items: {dropped}")

    if not rows:
        return

    rows = sorted(rows, key=lambda t: t[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, vals)
    ax.tick_params(axis="x", labelrotation=30)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    if zero_line:
        ax.axhline(0.0, linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def load_compare_items(
    *,
    compare: list[dict[str, Any]],
    base_dir: Path,
    warn_prefix: str,
    path_resolver=None,
) -> list[dict[str, Any]]:
    items = []

    for entry in compare:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
        if not name:
            continue

        if path_resolver is None:
            path_str = entry.get("path")
            if not path_str:
                continue
            path = resolve_path(str(path_str), base_dir)
        else:
            path = path_resolver(entry, base_dir)

        if not path.exists():
            print(f"[warn] {warn_prefix} missing: {path}")
            continue

        items.append({
            "name": name,
            "path": path,
            "data": load_json(path),
        })

    return items

def aggregate_psml_summary(summary: dict[str, Any]) -> dict[str, Any]:

    """
    Input format (from the summary.json):
      year -> dim -> {locs, RMSE, MAE, MAPE, MSIS}
    Output:
      {
        "per_dim": {dim: {"locs_total": float, "RMSE": float, "MAE": float, "MAPE": float, "MSIS": float}},
        "overall": {"RMSE": float, "MAE": float, "MAPE": float, "MSIS": float}
      }
    Aggregation:
      - per_dim: weighted mean over years, weighted by locs
      - overall: mean over dims (equal weight) of per_dim values where locs_total>0
    """
    metrics = ["RMSE", "MAE", "MAPE", "MSIS"]

    # collect dims that appear anywhere
    dims: list[str] = []
    dim_set = set()
    for year_block in summary.values():
        if isinstance(year_block, dict):
            for d in year_block:
                if d not in dim_set:
                    dim_set.add(d)
                    dims.append(d)
    dims = sorted(dims)

    per_dim: dict[str, dict[str, Any]] = {}
    for dim in dims:
        num = {m: 0.0 for m in metrics}
        den = 0.0
        for year_block in summary.values():
            if not isinstance(year_block, dict) or dim not in year_block:
                continue
            entry = year_block[dim]
            if not isinstance(entry, dict):
                continue
            locs = entry.get("locs", entry.get("#locs", 0))
            try:
                locs_f = float(locs)
            except (TypeError, ValueError):
                locs_f = 0.0
            if locs_f <= 0:
                continue

            den += locs_f
            for m in metrics:
                if m in entry and entry[m] is not None:
                    num[m] += float(entry[m]) * locs_f

        if den > 0:
            per_dim[dim] = {"locs_total": den, **{m: (num[m] / den) for m in metrics}}
        else:
            per_dim[dim] = {"locs_total": 0.0, **{m: float("nan") for m in metrics}}

    overall: dict[str, Any] = {}
    for m in metrics:
        vals = [float(per_dim[d][m]) for d in dims if per_dim[d]["locs_total"] > 0 and float(per_dim[d][m]) == float(per_dim[d][m])]
        overall[m] = (sum(vals) / len(vals)) if vals else float("nan")

    return {"per_dim": per_dim, "overall": overall}

def resolve_best_sweep_val_path(
    sweep_dir: Path,
    metric_col: str = "val_mean_over_loc_year",
) -> Path:
    leaderboard_path = sweep_dir / "validation" / "leaderboard.csv"

    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Validation leaderboard not found: {leaderboard_path}"
        )

    df = pd.read_csv(leaderboard_path)

    if metric_col not in df.columns:
        raise KeyError(
            f"Metric '{metric_col}' not found in {leaderboard_path}"
        )

    if "run_dir" not in df.columns:
        raise KeyError(
            f"'run_dir' column missing in {leaderboard_path}"
        )

    valid = df.dropna(subset=[metric_col])
    if valid.empty:
        raise RuntimeError(
            f"No valid '{metric_col}' values in {leaderboard_path}"
        )

    best_row = valid.loc[valid[metric_col].idxmin()]
    best_run_dir = Path(best_row["run_dir"])

    return best_run_dir / "evaluation" / "aggregate_val.json"

def resolve_val_compare_path(
    entry: dict[str, Any],
    base_dir: Path,
) -> Path:
    if entry.get("use_best_run", False):
        sweep_str = entry.get("sweep")
        if not sweep_str:
            raise ValueError(
                "use_best_run=true requires 'sweep'."
            )

        sweep_dir = resolve_path(str(sweep_str), base_dir)
        return resolve_best_sweep_val_path(sweep_dir)

    path_str = entry.get("path")
    if not path_str:
        raise ValueError(
            "Validation compare item needs 'path' "
            "or 'use_best_run: true'."
        )

    return resolve_path(str(path_str), base_dir)




# ---- Hyperparameter sweep validation analysis ----
SCORE = "val_mean_over_loc_year"
PARAMS = {
    "learning_rate": ("train", "lr"),
    "weight_decay": ("train", "weight_decay"),
    "optimizer": ("train", "optimizer"),
    "loss_type": ("train", "loss_type"),
    "huber_delta": ("train", "huber_delta"),
    "dropout": ("model", "kwargs", "dropout"),
    "hidden_sizes": ("model", "kwargs", "hidden_sizes"),
    "activation": ("model", "kwargs", "activation"),
    "use_layernorm": ("model", "kwargs", "use_layernorm"),
    "use_input_norm": ("model", "kwargs", "use_input_norm"),
}


def _sweep_nested(obj, path):
    for part in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _sweep_scalar(v):
    if isinstance(v, list):
        return "-".join(map(str, v))
    if isinstance(v, (dict, tuple)):
        return str(v)
    return v


def _read_sweep_zone_metrics(run_dir):
    records = []
    for p in sorted((run_dir / "zones").glob("*/*/metrics.json")):
        try:
            with p.open(encoding="utf-8") as f:
                m = json.load(f)
            eff = m.get("efficiency", {})
            score = m.get("val_mae_mean_over_targets")
            if score is None and m.get("val_mae_per_target"):
                score = float(np.mean(m["val_mae_per_target"]))
            records.append({
                "loc": str(m.get("loc", p.parent.parent.name)),
                "year": str(m.get("year", p.parent.name)),
                "zone_mae": score,
                "runtime_seconds": eff.get("training_runtime_seconds"),
                "num_params_trainable": eff.get("num_params_trainable"),
                "num_params_total": eff.get("num_params_total"),
            })
        except (OSError, ValueError, TypeError) as exc:
            warnings.warn(f"Skipping {p}: {exc}")
    return pd.DataFrame(records)


def _load_sweep_analysis(sweep_dir, include_incomplete=False):
    leaderboard = sweep_dir / "validation" / "leaderboard.csv"
    if not leaderboard.exists():
        raise FileNotFoundError(f"Missing leaderboard: {leaderboard}")
    df = pd.read_csv(leaderboard)
    if SCORE not in df or "run_id" not in df:
        raise ValueError(f"Leaderboard requires run_id and {SCORE}")
    df[SCORE] = pd.to_numeric(df[SCORE], errors="coerce")
    df = df.loc[np.isfinite(df[SCORE])].copy()
    if "val_n_experiments" in df and not include_incomplete:
        counts = pd.to_numeric(df["val_n_experiments"], errors="coerce")
        expected = counts.mode().iloc[0] if counts.notna().any() else None
        omitted = int((counts != expected).sum())
        if omitted:
            print(f"[info] {omitted} run(s) with a different unit count excluded from comparisons (expected {expected}).")
        df = df.loc[counts == expected].copy()

    rows, zones = [], []
    for _, r in df.iterrows():
        # Multi-directory leaderboards must use the original run path.
        saved_path = r.get("run_dir")
        if isinstance(saved_path, str) and saved_path.strip() and Path(saved_path).is_dir():
            run_dir = Path(saved_path).resolve()
        else:
            run_dir = sweep_dir / str(r["run_id"])
        config_file = run_dir / "config_used.yaml"
        if not config_file.exists():
            warnings.warn(f"Missing config: {config_file}; skipping run")
            continue
        with config_file.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        record = r.to_dict()
        record["run_id"] = str(r.get("run_key", r["run_id"]))
        for name, path in PARAMS.items():
            record[name] = _sweep_scalar(_sweep_nested(cfg, path))
        hs = _sweep_nested(cfg, ("model", "kwargs", "hidden_sizes"))
        if isinstance(hs, list):
            record["num_layers"] = len(hs)
            record["max_hidden_size"] = max(hs) if hs else np.nan
        zone_df = _read_sweep_zone_metrics(run_dir)
        if not zone_df.empty:
            zone_df["run_id"] = str(r.get("run_key", r["run_id"]))
            zones.append(zone_df)
            times = pd.to_numeric(zone_df["runtime_seconds"], errors="coerce")
            record["training_time_seconds"] = times.sum() if times.notna().all() else np.nan
            params = pd.to_numeric(zone_df["num_params_trainable"], errors="coerce").dropna().unique()
            record["num_params_trainable"] = params[0] if len(params) == 1 else np.nan
            if len(params) > 1:
                warnings.warn(f"Inconsistent parameter counts in {run_dir}")
        rows.append(record)
    if not rows:
        raise ValueError("No runs with readable config_used.yaml")
    return pd.DataFrame(rows), (pd.concat(zones, ignore_index=True) if zones else pd.DataFrame())


def _save_sweep_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def _sweep_top_ranking(df, out, top_n=15):
    d = df.nsmallest(top_n, SCORE).sort_values(SCORE, ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4.5, .38 * len(d))))
    ax.barh(d["run_id"].astype(str), d[SCORE])
    ax.set_xlabel("Validation MAE (macro mean over zone-years)")
    ax.set_ylabel("Run")
    ax.set_title(f"Top {len(d)} configurations")
    _save_sweep_plot(fig, out / "top15_ranking.png")
    d.to_csv(out / "top15_ranking.csv", index=False)


def _sweep_scatter(df, x, out, *, logx=False, title=None):
    d = df[[x, SCORE]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if logx:
        d = d[d[x] > 0]
    if len(d) < 2 or d[x].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(d[x], d[SCORE], alpha=.55, s=22)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel("Validation MAE")
    ax.set_title(title or f"{x} vs. validation MAE")
    _save_sweep_plot(fig, out / f"sensitivity_{x}.png")


def _sweep_categorical(df, col, out, max_categories=16):
    d = df[[col, SCORE]].dropna().copy()
    d[col] = d[col].astype(str)
    groups = d.groupby(col)[SCORE]
    if not 2 <= len(groups) <= max_categories:
        return
    names = sorted(groups.groups, key=lambda name: groups.get_group(name).median())
    fig, ax = plt.subplots(figsize=(max(7, .65 * len(names)), 5))
    vals = [groups.get_group(name).to_numpy(dtype=float) for name in names]
    ax.boxplot(vals, labels=names, showfliers=True)
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_xlabel(col.replace("_", " "))
    ax.set_ylabel("Validation MAE")
    ax.set_title(f"{col} vs. validation MAE")
    _save_sweep_plot(fig, out / f"sensitivity_{col}.png")


def _sweep_importance_fanova(df, out):
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution, FloatDistribution
        from optuna.importance import FanovaImportanceEvaluator, get_param_importances
    except ImportError:
        warnings.warn("optuna not installed; skipping fANOVA importance")
        return
    candidates = ["learning_rate", "weight_decay", "dropout", "hidden_sizes", "activation", "optimizer", "loss_type", "huber_delta", "use_layernorm", "use_input_norm"]
    # fANOVA requires a complete, consistent parameter space across trials.
    selected = [p for p in candidates if p in df and df[p].notna().all() and df[p].astype(str).nunique() > 1]
    if not selected:
        warnings.warn("No varying hyperparameters found for importance")
        return
    numeric = {"learning_rate", "weight_decay", "dropout", "huber_delta"}
    distributions = {}
    for p in selected:
        values = df[p].tolist()
        if p in numeric and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in values):
            lo, hi = float(min(values)), float(max(values))
            distributions[p] = FloatDistribution(lo, hi, log=(p == "learning_rate" and lo > 0))
        else:
            distributions[p] = CategoricalDistribution(sorted(set(values), key=str))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))
    for _, r in df.iterrows():
        values = {p: (float(r[p]) if isinstance(distributions[p], FloatDistribution) else r[p]) for p in selected}
        study.add_trial(optuna.trial.create_trial(params=values, distributions=distributions, value=float(r[SCORE])))
    try:
        imp = get_param_importances(study, evaluator=FanovaImportanceEvaluator(n_trees=64, seed=0), params=selected)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"fANOVA could not be calculated: {exc}")
        return
    s = pd.Series(imp, name="importance").sort_values()
    s.to_csv(out / "hyperparameter_importance_fanova.csv", header=True)
    fig, ax = plt.subplots(figsize=(8, max(4, .45 * len(s))))
    ax.barh(s.index, s.values)
    ax.set_xlabel("Estimated relative importance (fANOVA)")
    ax.set_title("Hyperparameter importance (validation MAE)")
    _save_sweep_plot(fig, out / "hyperparameter_importance_fanova.png")


def _sweep_zone_boxplots(df, zone_df, out, top_n=5):
    if zone_df.empty:
        warnings.warn("No zone metrics found; skipping zone plots")
        return
    zone_df = zone_df.copy()
    zone_df["zone_mae"] = pd.to_numeric(zone_df["zone_mae"], errors="coerce")
    zone_df = zone_df.dropna(subset=["zone_mae"])
    best = df.nsmallest(top_n, SCORE)["run_id"].astype(str).tolist()
    z = zone_df[zone_df["run_id"].isin(best)]
    if z.empty:
        return
    # One point per physical zone and run, averaging available years.
    per_zone = z.groupby(["run_id", "loc"], as_index=False)["zone_mae"].mean()
    per_zone.to_csv(out / "top_runs_per_zone_mae.csv", index=False)
    groups = [per_zone.loc[per_zone["run_id"] == rid, "zone_mae"].to_numpy() for rid in best]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, labels=best, showfliers=True)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(groups, start=1):
        ax.scatter(i + rng.uniform(-.12, .12, len(vals)), vals, s=12, alpha=.4)
    ax.set_xlabel("Run")
    ax.set_ylabel("Mean validation MAE per zone (across years)")
    ax.set_title("Zone variation in top configurations")
    _save_sweep_plot(fig, out / "zone_boxplots_top5.png")
    # Explicitly identify zones with the largest MAE for the best run.
    best_zone = per_zone[per_zone["run_id"] == best[0]].nlargest(15, "zone_mae")
    best_zone.to_csv(out / "worst_zones_best_run.csv", index=False)



def make_sweep_val_figures(sweep_dir: Path, *, include_incomplete: bool = False) -> list[Path]:
    """Create post-hoc sweep plots from existing validation results; no inference."""
    sweep_dir = Path(sweep_dir)
    out = sweep_dir / "validation" / "analysis"
    ensure_dir(out)
    df, zones = _load_sweep_analysis(sweep_dir, include_incomplete=include_incomplete)
    df.to_csv(out / "sweep_analysis_data.csv", index=False)
    _sweep_top_ranking(df, out)
    _sweep_importance_fanova(df, out)
    _sweep_scatter(df, "learning_rate", out, logx=True)
    _sweep_scatter(df, "weight_decay", out, logx=False)
    for param in ("hidden_sizes", "num_layers", "max_hidden_size", "dropout", "activation", "use_layernorm", "optimizer"):
        if param in df:
            _sweep_categorical(df, param, out)
    _sweep_zone_boxplots(df, zones, out)
    _sweep_scatter(df, "training_time_seconds", out, title="Training time vs. validation MAE")
    _sweep_scatter(df, "num_params_trainable", out, title="Trainable parameters vs. validation MAE")
    return sorted(out.glob("*.png"))