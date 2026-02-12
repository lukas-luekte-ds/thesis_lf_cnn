import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from .types import PredFrame
from .io import ensure_dir
from .io import suppress_output, _to_builtin, save_json
import importlib

# matches "<loc>_<year>" filenames, e.g. CAISO_zone_1_2018
_LOC_YEAR_RE = re.compile(r"^(?P<loc>.+)_(?P<year>\d{4})$")

def discover_psml_targets(cfg_run: dict, eval_cfg: dict) -> List[Tuple[str, int]]:
    test_cfg = eval_cfg["test"]

    # 1) Manual targets override
    if "targets" in test_cfg and test_cfg["targets"]:
        return [(str(t["loc"]), int(t["year"])) for t in test_cfg["targets"]]

    # 2) Discover from processed_dataset folder
    root = Path(cfg_run["data"]["root"])
    rel = test_cfg.get("data_folder", "processed_dataset/forecasting")
    data_folder = (root / rel).resolve()
    pattern = test_cfg.get("pattern", "*.csv")

    if not data_folder.exists():
        raise FileNotFoundError(f"PSML data_folder not found: {data_folder}")

    include_locs = set(map(str, test_cfg.get("include_locations", []) or []))
    include_years = set(map(int, test_cfg.get("include_years", []) or []))

    targets: List[Tuple[str, int]] = []
    for p in sorted(data_folder.glob(pattern)):
        stem = p.stem  # "CAISO_zone_1_2018"
        m = _LOC_YEAR_RE.match(stem)
        if not m:
            continue
        loc = m.group("loc")
        year = int(m.group("year"))

        if include_locs and loc not in include_locs:
            continue
        if include_years and year not in include_years:
            continue

        targets.append((loc, year))

    if not targets:
        raise RuntimeError(f"No targets discovered in {data_folder} (pattern={pattern}).")

    return targets

#layer 2b
def load_test_ids_for_loc_year(cfg_run: dict, eval_cfg: dict, loc: str, year: int) -> np.ndarray:
    """
    Load the psml test data from the processed csv data and 
    returns the ids for the corresponding year and location. 
    The filter is using train_flag==0 data as it is done in 
    the psml benchmark. 
    """
    root = Path(cfg_run["data"]["root"])
    test_cfg = eval_cfg["test"]
    rel = test_cfg.get("data_folder", "processed_dataset/forecasting")
    csv_path = (root / rel / f"{loc}_{year}.csv").resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"loc_year file not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["ID", "train_flag"])
    # evaluator uses train_flag==0 as test rows like psml does
    ids = df.loc[df["train_flag"].to_numpy() == 0, "ID"].to_numpy()
    return ids

def _psml_csv_columns(output_dims: List[str]) -> List[str]:
    cols = ["ID"]
    for name in output_dims:
        cols += [f"{name}(mean)", f"{name}(U)", f"{name}(L)"]
    return cols

def write_psml_prediction_csv(
    preds: PredFrame,
    out_csv: Path,
    output_dims: List[str],
    *,
    overwrite: bool,
    interval: str = "degenerate",
    std_per_dim: Optional[np.ndarray] = None,
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
        run_evaluate_forecasting = getattr(mod, "run_evaluate_forecasting")
    except Exception as e:
        raise ImportError(
            "Could not import PSML evaluator. Make sure the benchmark package is installed "
            "in the SAME conda env you run evaluation in (ENV_MAIN). "
            "E.g. run 'make install-baselines' for ENV_MAIN or install it into dl env."
        ) from e

    input_dict: Dict[str, Dict[str, np.ndarray]] = {}
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

    # Make JSON-serializable and save
    summary2 = _to_builtin(summary)
    save_json(out_summary_json, summary2, overwrite=overwrite)
    return out_summary_json