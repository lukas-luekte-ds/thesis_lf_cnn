from typing import Dict, Any, List
from pathlib import Path
from .io import load_json, ensure_dir
import matplotlib.pyplot as plt
import re
from .io import write_text

def plot_model_comparison_overall(models: List[Dict[str, Any]], out_png: Path, metric: str) -> None:
    names = [m["name"] for m in models]
    vals = [float(m["agg"]["overall"].get(metric, float("nan"))) for m in models]

    plt.figure()
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"Model comparison (overall): {metric}")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=150)
    plt.close()

def auto_discover_compare_summaries(
    *,
    eval_root: Path,
    eval_cfg: Dict[str, Any],
    eval_config_dir: Path,
) -> List[Dict[str, str]]:
    """
    If eval.compare is empty/missing:
      - include current run's test/summary.json if exists
      - include references summaries: references_root/**/summary.json
    Names:
      - ours:<exp_tag_dirname>
      - ref:<foldername>
    Optional:
      - compare_filter: regex string applied to name (after prefix)
      - compare_max: int maximum number of models (after filtering), deterministic order
    """
    items: List[Dict[str, str]] = []

    # include current run
    ours = eval_root / "test" / "summary.json"
    if ours.exists():
        items.append({"name": f"ours:{eval_root.name}", "summary": str(ours)})

    # include references
    test_cfg = eval_cfg.get("test", {})
    if not isinstance(test_cfg, dict):
        test_cfg = {}
    ref_root_str = str(test_cfg.get("references_root", "references/psml_forecasting"))
    ref_root = resolve_path(ref_root_str, eval_config_dir)
    if ref_root.exists():
        for p in sorted(ref_root.glob("**/summary.json")):
            name = p.parent.name
            items.append({"name": f"ref:{name}", "summary": str(p)})

    # optional filter
    filt = test_cfg.get("compare_filter")
    if filt:
        rx = re.compile(str(filt))
        items = [it for it in items if rx.search(it["name"])]

    # optional max (deterministic)
    compare_max = test_cfg.get("compare_max", None)
    if compare_max is not None:
        try:
            k = int(compare_max)
            if k > 0:
                items = items[:k]
        except Exception:
            pass

    return items

def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve path_str. If relative, resolve relative to base_dir."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir / p)
    return p.resolve()

def get_compare_list(
    *,
    eval_root: Path,
    eval_cfg: Dict[str, Any],
    eval_path: Path,
) -> List[Dict[str, str]]:
    """
    Returns compare list.
    Priority:
      - if eval_cfg.compare is non-empty -> use it (resolve paths)
      - else -> auto-discover (current + references)
    Also applies compare_max/filter to auto-discovery.
    """
    base_dir = eval_path.parent

    test_cfg = eval_cfg.get("test", {})
    if not isinstance(test_cfg, dict):
        test_cfg = {}

    compare = test_cfg.get("compare_test", None)
    if isinstance(compare, list) and len(compare) > 0:
        out = []
        for item in compare:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            summary_p = item.get("summary")
            if not name or not summary_p:
                continue
            p = resolve_path(str(summary_p), base_dir)
            out.append({"name": name, "summary": str(p)})
        return out

    return auto_discover_compare_summaries(eval_root=eval_root, eval_cfg=eval_cfg, eval_config_dir=base_dir)

def make_test_figures(
    *,
    eval_root: Path,
    eval_cfg: Dict[str, Any],
    eval_path: Path,
    overwrite: bool,
) -> List[Path]:
    """
    Creates comparison plots under eval_root/test/figures.
    Uses eval.compare if provided; otherwise auto-discovers.
    """
    fig_dir = eval_root / "test" / "figures"
    ensure_dir(fig_dir)

    compare_list = get_compare_list(eval_root=eval_root, eval_cfg=eval_cfg, eval_path=eval_path)
    models = []

    for item in compare_list:
        name = item["name"]
        p = Path(item["summary"]).expanduser().resolve()
        if not p.exists():
            print(f"[warn] compare summary missing: {p}")
            continue
        summary = load_json(p)
        agg = aggregate_psml_summary(summary)
        models.append({"name": name, "agg": agg, "path": p})

    if len(models) < 1:
        return []

    outs: List[Path] = []
    for metric in ["RMSE", "MAE", "MSIS", "MAPE"]:
        out_png = fig_dir / f"compare_overall_{metric}.png"
        if out_png.exists() and not overwrite:
            outs.append(out_png)
            continue
        plot_model_comparison_overall(models, out_png, metric)
        outs.append(out_png)

    return outs

def render_test_section(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("## Test (PSML Evaluator)\n")

    if not summary:
        lines.append("_No test summary found or summary is empty._\n")
        return "\n".join(lines)

    # compact listing: years present, dims present
    years = sorted(summary.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    dims = set()
    for y in years:
        blk = summary.get(y, {})
        if isinstance(blk, dict):
            dims.update(blk.keys())
    lines.append(f"- Years: {', '.join(map(str, years))}")
    lines.append(f"- Dims: {', '.join(sorted(dims))}\n")

    return "\n".join(lines)

def _fmt(x: Any, ndigits: int = 6) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:  # NaN
            return "NaN"
        return f"{x:.{ndigits}f}"
    return str(x)


##---------- val------------------
def get_val_compare_list(*, eval_cfg: Dict[str, Any], eval_path: Path) -> List[Dict[str, str]]:
    base_dir = eval_path.parent
    val_cfg = eval_cfg.get("val", {})
    if not isinstance(val_cfg, dict):
        return []
    compare = val_cfg.get("compare_val", None)
    if not isinstance(compare, list) or not compare:
        return []

    out = []
    for item in compare:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        p_str = item.get("val")
        if not name or not p_str:
            continue
        p = resolve_path(str(p_str), base_dir)
        out.append({"name": name, "val": str(p)})
    return out

def plot_val_compare_overall(models: List[Dict[str, Any]], out_png: Path, metric: str) -> None:
    rows = []
    for m in models:
        v = m["val_metrics"].get("metrics", {}).get(metric)
        try:
            v = float(v)
        except Exception:
            continue
        rows.append((m["name"], v))

    if not rows:
        return

    # simple: ascending for all metrics
    rows = sorted(rows, key=lambda t: t[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    plt.figure()
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    if metric.lower() == "me":
        plt.axhline(0.0, linewidth=1)
        plt.ylabel("ME = mean(y_pred - y_true)")
    else:
        plt.ylabel(metric.upper())
    plt.title(f"Val model comparison (overall): {metric.upper()}")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=150)
    plt.close()

def make_val_compare_figures(*, eval_root: Path, eval_cfg: Dict[str, Any], eval_path: Path, overwrite: bool) -> List[Path]:
    fig_dir = eval_root / "val" / "figures"
    ensure_dir(fig_dir)

    compare_list = get_val_compare_list(eval_cfg=eval_cfg, eval_path=eval_path)
    models = []
    for item in compare_list:
        p = Path(item["val"]).expanduser().resolve()
        if not p.exists():
            print(f"[warn] compare val missing: {p}")
            continue
        vm = load_json(p)
        models.append({"name": item["name"], "val_metrics": vm, "path": p})

    if len(models) < 1:
        return []

    outs: List[Path] = []
    for metric in ["rmse", "mae", "me"]:
        out_png = fig_dir / f"compare_val_overall_{metric}.png"
        if out_png.exists() and not overwrite:
            outs.append(out_png)
            continue
        plot_val_compare_overall(models, out_png, metric)
        outs.append(out_png)

    return outs

def render_val_section(val_metrics: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("## Validation\n")

    shape = val_metrics.get("shape", {})
    mask = val_metrics.get("mask", {})
    metrics = val_metrics.get("metrics", {})

    lines.append(f"- Shape: N={shape.get('N','?')}, D={shape.get('D','?')}")
    lines.append(f"- Mask: n_obs={mask.get('n_obs','?')}, true_ratio={_fmt(mask.get('true_ratio'), 4)}\n")

    lines.append("### Overall\n")
    if isinstance(metrics, dict) and metrics:
        for k, v in metrics.items():
            lines.append(f"- {k.upper()}: {_fmt(v)}")
    else:
        lines.append("_No overall metrics found._")
    lines.append("")

    per_dim = val_metrics.get("per_dim")
    if isinstance(per_dim, dict) and per_dim:
        metric_names = list(metrics.keys()) if isinstance(metrics, dict) else []
        if not metric_names:
            first = next(iter(per_dim.values()))
            metric_names = [k for k in first.keys() if k != "n_obs"]

        lines.append("### Per dimension\n")
        header = "| dim | n_obs | " + " | ".join(m.upper() for m in metric_names) + " |"
        sep = "|---|---:|" + "|".join(["---:"] * len(metric_names)) + "|"
        lines.append(header)
        lines.append(sep)

        for dim, entry in per_dim.items():
            row = [dim, str(entry.get("n_obs", "—"))]
            for m in metric_names:
                row.append(_fmt(entry.get(m)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    else:
        lines.append("_No per-dimension metrics found._\n")

    return "\n".join(lines)

def plot_val_per_dim(val_metrics: Dict[str, Any], out_png: Path, metric: str = "rmse", top_k: int = 20) -> None:
    per_dim = val_metrics.get("per_dim")
    if not isinstance(per_dim, dict) or not per_dim:
        return

    rows = []
    for dim, entry in per_dim.items():
        if not isinstance(entry, dict):
            continue
        v = entry.get(metric)
        try:
            v = float(v)
        except Exception:
            v = float("nan")
        n_obs = entry.get("n_obs", 0)
        rows.append((dim, v, n_obs))

    # sort descending (worst first)
    rows = sorted(rows, key=lambda t: (float("-inf") if (t[1] != t[1]) else t[1]), reverse=False)[:top_k]
    dims = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    plt.figure()
    plt.bar(dims, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"Validation per-dim {metric.upper()} (top {top_k})")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_val_overall(val_metrics: Dict[str, Any], out_png: Path) -> None:
    metrics = val_metrics.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return

    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]
    # convert to float if possible
    vals2 = []
    for v in vals:
        try:
            vals2.append(float(v))
        except Exception:
            vals2.append(float("nan"))

    plt.figure()
    plt.bar([k.upper() for k in keys], vals2)
    plt.ylabel("metric")
    plt.title("Validation overall metrics")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=150)
    plt.close()

def make_val_figures(
    *,
    eval_root: Path,
    overwrite: bool,
    top_k: int = 20,
) -> List[Path]:
    """
    Creates diagnostic plots under eval_root/val/figures.
    Only uses val_metrics.json from THIS run (no cross-run comparison by default).
    """
    val_path = eval_root / "val" / "val_metrics.json"
    if not val_path.exists():
        return []

    val_metrics = load_json(val_path)
    fig_dir = eval_root / "val" / "figures"
    ensure_dir(fig_dir)

    outs: List[Path] = []

    out_overall = fig_dir / "val_overall.png"
    if (not out_overall.exists()) or overwrite:
        plot_val_overall(val_metrics, out_overall)
    outs.append(out_overall)

    # per-dim RMSE/MAE/ME if present
    out_rmse = fig_dir / f"val_per_dim_rmse_top{top_k}.png"
    if (not out_rmse.exists()) or overwrite:
        plot_val_per_dim(val_metrics, out_rmse, metric="rmse", top_k=top_k)
    outs.append(out_rmse)

    out_mae = fig_dir / f"val_per_dim_mae_top{top_k}.png"
    if (not out_mae.exists()) or overwrite:
        plot_val_per_dim(val_metrics, out_mae, metric="mae", top_k=top_k)
    outs.append(out_mae)

    out_me = fig_dir / f"val_per_dim_me_top{top_k}.png"
    if (not out_me.exists()) or overwrite:
        plot_val_per_dim(val_metrics, out_me, metric="me", top_k=top_k)
    outs.append(out_me)

    return outs

def build_report_md(
    *,
    run_dir: Path,
    exp_name: str,
    tag: str,
    eval_cfg: Dict[str, Any],
    cfg_run: Dict[str, Any],
    eval_path: Path,
) -> str:
    eval_root = run_dir / f"{exp_name}_{tag}"

    lines: List[str] = []
    lines.append(f"# Report: {exp_name} ({tag})\n")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Eval root: `{eval_root}`\n")

    m = cfg_run.get("model", {})
    lines.append("## Config snapshot\n")
    lines.append(f"- Model: `{m.get('module','?')}.{m.get('class','?')}`")
    lines.append(f"- Model kwargs: `{m.get('kwargs', {})}`")
    lines.append(f"- Output dims: `{eval_cfg.get('output_dims', [])}`\n")

    overwrite = bool(eval_cfg.get("overwrite", False))

    # Validation section + figures
    mode = str(eval_cfg.get("mode", "test")).lower()

    # Validation section
    val_path = eval_root / "val" / "val_metrics.json"
    if val_path.exists():
        val_metrics = load_json(val_path)
        lines.append(render_val_section(val_metrics))

        # Only generate val figures in val/both mode
        if mode in ("val", "both"):
            val_figs = make_val_figures(eval_root=eval_root, overwrite=overwrite, top_k=int(eval_cfg.get("val_top_k", 20)))
            if val_figs:
                lines.append("### Validation figures\n")
                for p in val_figs:
                    rel = p.relative_to(eval_root)
                    lines.append(f"![]({rel.as_posix()})")
                lines.append("")
            val_compare_figs = make_val_compare_figures(
                eval_root=eval_root,
                eval_cfg=eval_cfg,
                eval_path=eval_path,
                overwrite=overwrite,
            )
            if val_compare_figs:
                lines.append("### Validation model comparison figures\n")
                for p in val_compare_figs:
                    rel = p.relative_to(eval_root)
                    lines.append(f"![]({rel.as_posix()})")
                lines.append("")
    else:
        lines.append("## Validation\n\n_No val_metrics.json found._\n")


    # Test section + figures
    summary_path = eval_root / "test" / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        lines.append(render_test_section(summary))

        if mode in ("test", "both"):
            # compare plots
            test_figs = make_test_figures(eval_root=eval_root, eval_cfg=eval_cfg, eval_path=eval_path, overwrite=overwrite)
            if test_figs:
                lines.append("### Test model comparison figures\n")
                test_cfg = eval_cfg.get("test", {})
                if not isinstance(test_cfg, dict):
                    test_cfg = {}
                lines.append(f"- compare_max: `{test_cfg.get('compare_max', 'None')}`")
                lines.append(f"- references_root: `{test_cfg.get('references_root', 'references/psml_forecasting')}`")
                if test_cfg.get("compare_filter"):
                    lines.append(f"- compare_filter: `{test_cfg.get('compare_filter')}`")
                for p in test_figs:
                    rel = p.relative_to(eval_root)
                    lines.append(f"![]({rel.as_posix()})")
                lines.append("")
        else:
            lines.append("## Test (PSML Evaluator)\n\n_No test/summary.json found._\n")

    # Artifacts
    lines.append("## Artifacts\n")
    lines.append(f"- Validation metrics: `{val_path}`" + ("" if val_path.exists() else " _(missing)_"))
    lines.append(f"- Test summary: `{summary_path}`" + ("" if summary_path.exists() else " _(missing)_"))
    pred_dir = eval_root / "test" / "predictions"
    lines.append(f"- Test predictions dir: `{pred_dir}`" + ("" if pred_dir.exists() else " _(missing)_"))
    lines.append("")

    return "\n".join(lines)

def aggregate_psml_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input format:
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
    dims: List[str] = []
    dim_set = set()
    for year_block in summary.values():
        if isinstance(year_block, dict):
            for d in year_block.keys():
                if d not in dim_set:
                    dim_set.add(d)
                    dims.append(d)
    dims = sorted(dims)

    per_dim: Dict[str, Dict[str, Any]] = {}
    for dim in dims:
        num = {m: 0.0 for m in metrics}
        den = 0.0
        for year, year_block in summary.items():
            if not isinstance(year_block, dict) or dim not in year_block:
                continue
            entry = year_block[dim]
            if not isinstance(entry, dict):
                continue
            locs = entry.get("locs", entry.get("#locs", 0))
            try:
                locs_f = float(locs)
            except Exception:
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

    overall: Dict[str, Any] = {}
    for m in metrics:
        vals = [float(per_dim[d][m]) for d in dims if per_dim[d]["locs_total"] > 0 and not (float(per_dim[d][m]) != float(per_dim[d][m]))]
        overall[m] = (sum(vals) / len(vals)) if vals else float("nan")

    return {"per_dim": per_dim, "overall": overall}

def report_and_save(
    *,
    run_dir: Path,
    exp_name: str,
    eval_cfg: Dict[str, Any],
    cfg_run: Dict[str, Any],
    eval_path: Path,
) -> Path:
    tag = eval_cfg["tag"]
    overwrite = bool(eval_cfg.get("overwrite", False))
    eval_root = run_dir / f"{exp_name}_{tag}"
    out_path = eval_root / "report.md"

    md = build_report_md(
        run_dir=run_dir,
        exp_name=exp_name,
        tag=tag,
        eval_cfg=eval_cfg,
        cfg_run=cfg_run,
        eval_path=eval_path,
    )
    write_text(out_path, md, overwrite=overwrite)
    return out_path