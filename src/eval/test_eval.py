import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import save_json

from .io import ensure_dir, suppress_output
from .types import PredFrame

# matches "<loc>_<year>" filenames, e.g. CAISO_zone_1_2018
_LOC_YEAR_RE = re.compile(r"^(?P<loc>.+)_(?P<year>\d{4})$")

def discover_trained_run_units(run_dir: Path) -> list[tuple[str, int]]:
    """
    Discover all successfully trained (loc, year) units from the run directory.

    Expected structure:
        run_dir/zones/<loc>/<year>/checkpoints/best-val.pt
    """
    zones_root = run_dir / "zones"

    if not zones_root.exists():
        raise FileNotFoundError(
            f"Zones directory not found: {zones_root}"
        )

    units: list[tuple[str, int]] = []

    checkpoint_paths = sorted(
        zones_root.glob("*/*/checkpoints/best-val.pt")
    )

    for checkpoint_path in checkpoint_paths:
        year_dir = checkpoint_path.parent.parent
        loc_dir = year_dir.parent

        loc = loc_dir.name

        try:
            year = int(year_dir.name)
        except ValueError:
            print(
                f"[warn] Ignoring checkpoint with invalid year directory: "
                f"{checkpoint_path}"
            )
            continue

        units.append((loc, year))

    if not units:
        raise RuntimeError(
            f"No trained loc/year checkpoints found under: {zones_root}"
        )

    return units

#layer 2b

def _psml_csv_columns(output_dims: list[str]) -> list[str]:
    cols = ["ID"]
    for name in output_dims:
        cols += [f"{name}(mean)", f"{name}(U)", f"{name}(L)"]
    return cols

def write_psml_prediction_csv(
    preds: PredFrame,
    out_csv: Path,
    output_dims: list[str],
    *,
    overwrite: bool,
    interval: str = "degenerate",
    std_per_dim: np.ndarray | None = None,
    interval_multiplier: float = 1.96,
) -> Path:
    """
    Writes ONE PSML prediction CSV for a loc_year, e.g. CAISO_zone_1_2018.csv
    """
    ensure_dir(out_csv.parent)
    if out_csv.exists() and not overwrite:
        raise FileExistsError(f"{out_csv} exists. Set eval.overwrite: true to overwrite.")

    if preds.ids is None:
        raise ValueError("Test writing requires preds.ids (real PSML IDs).")
    if preds.y_true is not None or preds.eval_mask is not None:
        # not fatal, but should not happen in test mode
        print("[warn] test preds contains y_true/eval_mask; expected None.")

    ids = preds.ids
    y_pred = preds.y_pred

    if y_pred.ndim != 2:
        raise ValueError(f"Expected y_pred 2D (N,D), got {y_pred.shape}")
    if y_pred.shape[1] != len(output_dims):
        raise ValueError(f"D mismatch: y_pred has D={y_pred.shape[1]} but len(output_dims)={len(output_dims)}")

    mean = y_pred.astype(float)

    if interval == "degenerate":
        U = mean
        L = mean
    elif interval == "std":
        if std_per_dim is None:
            raise ValueError("interval='std' requires std_per_dim.")
        std_per_dim = np.asarray(std_per_dim).reshape(-1)
        if std_per_dim.shape[0] != mean.shape[1]:
            raise ValueError("std_per_dim must have shape (D,).")
        U = mean + interval_multiplier * std_per_dim[None, :]
        L = mean - interval_multiplier * std_per_dim[None, :]
    else:
        raise ValueError("interval must be 'degenerate' or 'std'.")

    data = {"ID": ids}
    for j, name in enumerate(output_dims):
        data[f"{name}(mean)"] = mean[:, j]
        data[f"{name}(U)"] = U[:, j]
        data[f"{name}(L)"] = L[:, j]

    df = pd.DataFrame(data, columns=_psml_csv_columns(output_dims))
    df.to_csv(out_csv, index=False)
    return out_csv

def run_psml_evaluator_and_save(
    *,
    psml_root: Path,
    prediction_dir: Path,
    out_summary_json: Path,
    overwrite: bool,
    prediction_interval: float = 0.95,
) -> Path:
    """
    Runs official PSML evaluator:
      BenchmarkModel.LoadForecasting.evaluating.run_evaluate_forecasting
    using all CSVs in prediction_dir.
    """
    ensure_dir(out_summary_json.parent)
    if out_summary_json.exists() and not overwrite:
        raise FileExistsError(f"{out_summary_json} exists. Set eval.overwrite: true to overwrite.")

    # Import evaluator (installed via baselines benchmark package)
    try:
        mod = importlib.import_module("BenchmarkModel.LoadForecasting.evaluating")
        run_evaluate_forecasting = mod.run_evaluate_forecasting
    except Exception as e:
        raise ImportError(
            "Could not import PSML evaluator. Make sure the benchmark package is installed "
            "in the SAME conda env you run evaluation in (ENV_MAIN). "
            "E.g. run 'make install-baselines' for ENV_MAIN or install it into dl env."
        ) from e

    input_dict: dict[str, dict[str, np.ndarray]] = {}
    csvs = sorted(prediction_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No prediction CSVs found in {prediction_dir}")

    for csv_path in csvs:
        loc_year = csv_path.stem
        df = pd.read_csv(csv_path)
        input_dict[loc_year] = {col: df[col].to_numpy() for col in df.columns}
    # psml evaluation prints a big bad readable summary output (so the output is suppressed here)
    with suppress_output():
        summary = run_evaluate_forecasting(
            root=str(psml_root),
            input_dict=input_dict,
            prediction_interval=prediction_interval,
        )

    save_json(out_summary_json, summary, overwrite=overwrite)
    return out_summary_json