import numpy as np
from .types import PredFrame
from typing import List, Dict, Any
from pathlib import Path
from .io import save_json


def _masked_values(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns 1D arrays of only valid entries (mask==True).
    Works for any shape as long as mask matches shape.
    """
    m = mask.astype(bool)
    yt = y_true[m]
    yp = y_pred[m]
    return yt, yp

def _compute_metric(name: str, yt: np.ndarray, yp: np.ndarray) -> float:
    if yt.size == 0:
        return float("nan")
    if name == "mae":
        return float(np.mean(np.abs(yt - yp)))
    if name == "rmse":
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    if name == "me":
        return float(np.mean(yp - yt))
    raise ValueError(f"Unknown metric '{name}'. Implement it or remove it from eval.val.metrics.")

def eval_val_metrics(
    preds: PredFrame,
    *,
    dim_names: List[str],
    metrics: List[str],
    per_dim: bool = True,
) -> Dict[str, Any]:
    """
    Masked metrics on validation predictions.
    Requires preds.y_true and preds.eval_mask.
    """
    if preds.y_true is None or preds.eval_mask is None:
        raise ValueError("Val evaluation requires y_true and eval_mask (flag).")

    y_true = preds.y_true
    y_pred = preds.y_pred
    mask = preds.eval_mask

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} != y_pred {y_pred.shape}")
    if mask.shape != y_pred.shape:
        raise ValueError(f"mask {mask.shape} != y_pred {y_pred.shape}")
    if len(dim_names) != y_pred.shape[1]:
        raise ValueError(f"len(dim_names)={len(dim_names)} != D={y_pred.shape[1]}")

    # overall (flattened masked)
    yt_flat, yp_flat = _masked_values(y_true, y_pred, mask)

    out: Dict[str, Any] = {
        "split": "val",
        "shape": {"N": int(y_pred.shape[0]), "D": int(y_pred.shape[1])},
        "mask": {
            "n_obs": int(yt_flat.size),                       # number of included values
            "true_ratio": float(mask.astype(bool).mean()),    # ratio of True in (N,D)
        },
        "metrics": {},
    }

    for mname in metrics:
        out["metrics"][mname] = _compute_metric(mname, yt_flat, yp_flat)

    # per-dim metrics
    if per_dim:
        per: Dict[str, Any] = {}
        for j, name in enumerate(dim_names):
            yt_j, yp_j = _masked_values(y_true[:, j], y_pred[:, j], mask[:, j])
            entry = {
                "n_obs": int(yt_j.size),
            }
            for mname in metrics:
                entry[mname] = _compute_metric(mname, yt_j, yp_j)
            per[name] = entry
        out["per_dim"] = per

    return out

def eval_val_and_save(
    preds: PredFrame,
    *,
    run_dir: Path,
    exp_name: str,
    eval_cfg: Dict[str, Any],
) -> Path:
    """
    Writes: <run_dir>/<exp_name>_<tag>/val/val_metrics.json
    """
    tag = eval_cfg["tag"]
    overwrite = bool(eval_cfg.get("overwrite", False))
    dim_names = list(eval_cfg["output_dims"])

    val_cfg = eval_cfg.get("val", {})
    metrics = list(val_cfg.get("metrics", ["mae", "rmse", "me"]))
    per_dim = bool(val_cfg.get("per_dim", True))

    val_results = eval_val_metrics(preds, dim_names=dim_names, metrics=metrics, per_dim=per_dim)

    eval_root = run_dir / f"{exp_name}_{tag}"
    out_path = eval_root / "val" / "val_metrics.json"
    save_json(out_path, val_results, overwrite=overwrite)
    return out_path
