import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def normalize_target_specs(task_cfg):
    """
    Normalizes both horizon_steps formats (global or individual setting)
    """
    targets = task_cfg["targets"]
    if all(isinstance(t, str) for t in targets):
        if "horizon_steps" not in task_cfg:
            raise ValueError(
                "data.custom_loader.task.horizon_steps is required when "
                "targets is a list of strings"
            )

        horizon_steps = int(task_cfg["horizon_steps"])

        return [
            {
                "name": t,
                "column": t,
                "horizon_steps": horizon_steps,
            }
            for t in targets
        ]

    if all(isinstance(t, dict) for t in targets):
        normalized = []

        for i, t in enumerate(targets):
            if "column" not in t:
                raise ValueError(f"Missing field targets[{i}].column")
            if "horizon_steps" not in t:
                raise ValueError(f"Missing field targets[{i}].horizon_steps")

            column = t["column"]
            horizon_steps = int(t["horizon_steps"])
            name = t.get("name", f"{column}_t+{horizon_steps}")

            if horizon_steps <= 0:
                raise ValueError(f"targets[{i}].horizon_steps must be > 0")

            normalized.append(
                {
                    "name": name,
                    "column": column,
                    "horizon_steps": horizon_steps,
                }
            )

        return normalized

    raise ValueError(
        "data.custom_loader.task.targets must be either a list of strings "
        "or a list of dicts with column/horizon_steps"
    )

class CustomForecastDataset(Dataset):
    """
    Input:
        X: [N, F]
        y: [N, n_targets]
        sample_end_indices: valid end positions t

    Output per sample:
        x_window: [T, F]
        y_target: [n_targets]
        target_mask: [n_targets]
    """

    def __init__(self, X, y, sample_end_indices, input_length, target_specs):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.sample_end_indices = sample_end_indices
        self.input_length = input_length
        self.target_specs = target_specs
        self.target_names = [t["name"] for t in target_specs]
        self.target_horizons = torch.as_tensor(
            [t["horizon_steps"] for t in target_specs],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.sample_end_indices)

    def __getitem__(self, idx):
        end_idx = int(self.sample_end_indices[idx])
        start_idx = end_idx - self.input_length + 1
        target_indices = end_idx + self.target_horizons

        assert start_idx >= 0, f"Invalid start_idx={start_idx}"
        assert int(target_indices.max()) < len(self.y), (
            f"Invalid target index max={int(target_indices.max())}, len(y)={len(self.y)}"
        )

        x = self.X[start_idx:end_idx + 1]

        y_values = []
        for target_pos, target_idx in enumerate(target_indices):
            y_values.append(self.y[int(target_idx), target_pos])

        y = torch.stack(y_values)

        # it is always true and does nothing, implement real mask logic if needed
        target_mask = torch.ones_like(y, dtype=torch.float32)

        return x, y, target_mask



class CustomTimeSeriesLoader:
    """
    Uses the global data config plus data.custom_loader for custom-loader-specific settings.

    Expected structure:
        data:
          root: ...
          batch_size: ...
          shuffle: ...
          ...
          custom_loader:
            source: ...
            features: ...
            task: ...
            split: ...
            preprocessing: ...
    """

    def _cache_dir(self):
        cache_cfg = self.cfg.get("cache", {})
        cache_name = cache_cfg.get("name")

        if not cache_name:
            raise ValueError("cache.enabled=true requires data.custom_loader.cache.name")

        root = Path(self.data_config["root"])
        return root / "custom_processed" / cache_name


    def __init__(self, data_config):
        self.data_config = data_config
        if "custom_loader" not in self.data_config:
            raise ValueError("Missing data.custom_loader in config")

        self.cfg = self.data_config["custom_loader"]
        self._validate_config()

        self.target_specs = normalize_target_specs(self.cfg["task"])
        self.target_names = [t["name"] for t in self.target_specs]
        self.target_columns = [t["column"] for t in self.target_specs]
        self.max_horizon_steps = max(t["horizon_steps"] for t in self.target_specs)

    def _validate_config(self):
        required = ["source", "features", "task", "split"]
        for key in required:
            if key not in self.cfg:
                raise ValueError(f"Missing config section: data.custom_loader.{key}")

        if "root" not in self.data_config:
            raise ValueError("Missing config section: data.root")

        source_cfg = self.cfg["source"]
        if "file_name" not in source_cfg and "files" not in source_cfg:
            raise ValueError(
                "Missing config field: either data.custom_loader.source.file_name "
                "or data.custom_loader.source.files is required"
            )

        task = self.cfg["task"]

        if "targets" not in task:
            raise ValueError("Missing config field: data.custom_loader.task.targets")

        if not isinstance(task["targets"], list) or len(task["targets"]) == 0:
            raise ValueError("data.custom_loader.task.targets must be a non-empty list")

        if (
            all(isinstance(t, dict) for t in task["targets"])
            and "horizon_steps" in task
        ):
            print(
                "Warning: data.custom_loader.task.horizon_steps is ignored "
                "when using per-target horizons."
            )

        if task["input_length"] <= 0:
            raise ValueError("data.custom_loader.task.input_length must be > 0")
        if all(isinstance(t, str) for t in task["targets"]):
            if "horizon_steps" not in task:
                raise ValueError(
                    "data.custom_loader.task.horizon_steps is required "
                    "when targets is a list of strings"
                )
            if task["horizon_steps"] <= 0:
                raise ValueError("data.custom_loader.task.horizon_steps must be > 0")
        if task["prediction_stride"] <= 0:
            raise ValueError("data.custom_loader.task.prediction_stride must be > 0")

        split_cfg = self.cfg["split"]
        required_split_keys = ["train_start", "train_end", "test_start", "test_end"]

        for key in required_split_keys:
            try:
                pd.Timestamp(split_cfg[key])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid datetime format for "
                    f"data.custom_loader.split.{key}: "
                    f"'{split_cfg[key]}'.\n"
                    f"Expected format example: "
                    f"'2018-01-01 00:00:00'"
                )

    def _save_zone_cache(self, train_arrays_by_zone, test_arrays_by_zone):
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines = []

        for i, (zone_name, arrays) in enumerate(train_arrays_by_zone):
            X, y, indices = arrays
            np.save(cache_dir / f"train_zone{i}_X.npy", X)
            np.save(cache_dir / f"train_zone{i}_y.npy", y)
            np.save(cache_dir / f"train_zone{i}_indices.npy", indices)

            manifest_lines.append(f"{i}\t{zone_name}")

        for i, (zone_name, arrays) in enumerate(test_arrays_by_zone):
            X, y, indices = arrays
            np.save(cache_dir / f"test_zone{i}_X.npy", X)
            np.save(cache_dir / f"test_zone{i}_y.npy", y)
            np.save(cache_dir / f"test_zone{i}_indices.npy", indices)

        with open(cache_dir / "manifest.txt", "w") as f:
            f.write("\n".join(manifest_lines))

        print(f"Saved zone cache to: {cache_dir}")

    def _load_zone_cache_as_datasets(self):
        cache_dir = self._cache_dir()

        with open(cache_dir / "manifest.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        train_datasets = []
        test_datasets = []

        for line in lines:
            i_str, _zone_name = line.split("\t")
            i = int(i_str)

            train_arrays = (
                np.load(cache_dir / f"train_zone{i}_X.npy"),
                np.load(cache_dir / f"train_zone{i}_y.npy"),
                np.load(cache_dir / f"train_zone{i}_indices.npy"),
            )
            test_arrays = (
                np.load(cache_dir / f"test_zone{i}_X.npy"),
                np.load(cache_dir / f"test_zone{i}_y.npy"),
                np.load(cache_dir / f"test_zone{i}_indices.npy"),
            )

            train_datasets.append(self._dataset_from_arrays(train_arrays))
            test_datasets.append(self._dataset_from_arrays(test_arrays))

        print(f"Loaded zone cache from: {cache_dir}")

        return train_datasets, test_datasets

    def load(self, batch_size, shuffle=True):
        cache_cfg = self.cfg.get("cache", {})
        use_cache = cache_cfg.get("enabled", False)
        rebuild_cache = cache_cfg.get("rebuild", False)

        if use_cache and (self._cache_dir() / "manifest.txt").exists() and not rebuild_cache:
            train_datasets, test_datasets = self._load_zone_cache_as_datasets()
        else:
            zone_dfs = self._read_zone_dataframes()

            train_datasets = []
            test_datasets = []

            cached_train_arrays = []
            cached_test_arrays = []

            for zone_name, df in zone_dfs:
                print(f"Processing zone: {zone_name}")

                df = self._merge_extra_sources(df)
                df = self._add_time_features(df)

                feature_cols, target_cols = self._build_column_lists(df)
                train_df, test_df = self._split_dataframe(df)

                train_arrays = self._build_split_arrays(train_df, feature_cols, target_cols)
                test_arrays = self._build_split_arrays(test_df, feature_cols, target_cols)

                train_datasets.append(self._dataset_from_arrays(train_arrays))
                test_datasets.append(self._dataset_from_arrays(test_arrays))

                cached_train_arrays.append((zone_name, train_arrays))
                cached_test_arrays.append((zone_name, test_arrays))

            if use_cache:
                self._save_zone_cache(cached_train_arrays, cached_test_arrays)

        train_dataset_full = ConcatDataset(train_datasets)
        test_dataset_full = ConcatDataset(test_datasets)

        train_loader_full = DataLoader(
            train_dataset_full,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset_full,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        return train_loader_full, test_loader

    def _read_zone_dataframes(self):
        source_cfg = self.cfg["source"]

        if "files" in source_cfg:
            zone_dfs = []
            for file_cfg in source_cfg["files"]:
                zone_name = file_cfg.get("name", file_cfg["file_name"])
                df = self._read_single_zone_file(source_cfg, file_cfg)
                zone_dfs.append((zone_name, df))
            return zone_dfs

        df = self._read_raw_data()
        return [("single_source", df)]

    def _read_single_zone_file(self, source_cfg, file_cfg):
        root = self.data_config["root"]

        subdir = source_cfg.get("subdir", "")
        file_name = file_cfg["file_name"]
        timestamp_col = file_cfg.get(
            "timestamp_col",
            source_cfg.get("timestamp_col", "time")
        )

        path = os.path.join(root, subdir, file_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Zone source file not found: {path}")

        df = pd.read_csv(path)

        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != ""]

        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in {path}")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).copy()
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        df = df.rename(columns={timestamp_col: "timestamp"})

        if df["timestamp"].duplicated().any():
            raise ValueError(f"Duplicate timestamps found in zone file: {path}")

        return df

    def _read_raw_data(self):
        root = self.data_config["root"]
        source_cfg = self.cfg["source"]

        subdir = source_cfg.get("subdir", "")
        file_name = source_cfg["file_name"]
        timestamp_col = source_cfg.get("timestamp_col", "time")

        path = os.path.join(root, subdir, file_name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Raw data file not found: {path}. "
                f"Check data.root and data.custom_loader.source.*"
            )

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != ""]

        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in {path}")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        df = df.rename(columns={timestamp_col: "timestamp"})

        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps found in raw data")

        return df

    def _add_time_features(self, df):
        time_features = self.cfg["features"].get("time_features", [])
        holidays = USFederalHolidayCalendar().holidays()

        if "month" in time_features:
            df["feat_month"] = df["timestamp"].dt.month.astype(np.float32)

        if "dayofweek" in time_features:
            df["feat_dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.float32)

        if "holiday" in time_features:
            date_only = pd.to_datetime(df["timestamp"].dt.date)
            df["feat_holiday"] = date_only.isin(holidays).astype(np.float32)

        return df

    def _build_column_lists(self, df):
        target_cols = list(dict.fromkeys(self.target_columns))

        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in dataframe")

        feature_cols = []

        if self.cfg["features"].get("target_as_input", True):
            feature_cols.extend(target_cols)

        for col in self.cfg["features"].get("external", []):
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found in dataframe")
            feature_cols.append(col)

        time_feature_map = {
            "month": "feat_month",
            "dayofweek": "feat_dayofweek",
            "holiday": "feat_holiday",
        }
        for feat in self.cfg["features"].get("time_features", []):
            if feat not in time_feature_map:
                raise ValueError(f"Unsupported time feature: {feat}")
            feature_cols.append(time_feature_map[feat])

        return feature_cols, target_cols

    def _split_dataframe(self, df):
        split_cfg = self.cfg["split"]

        train_df = df.loc[
            (df["timestamp"] >= pd.Timestamp(split_cfg["train_start"])) &
            (df["timestamp"] <= pd.Timestamp(split_cfg["train_end"]))
        ].copy().reset_index(drop=True)

        test_df = df.loc[
            (df["timestamp"] >= pd.Timestamp(split_cfg["test_start"])) &
            (df["timestamp"] <= pd.Timestamp(split_cfg["test_end"]))
        ].copy().reset_index(drop=True)

        if len(train_df) == 0:
            raise ValueError("Train split is empty")
        if len(test_df) == 0:
            raise ValueError("Test split is empty")

        return train_df, test_df

    def _build_split_dataset(self, df, feature_cols, target_cols):
        arrays = self._build_split_arrays(df, feature_cols, target_cols)
        return self._dataset_from_arrays(arrays)

    def _build_split_arrays(self, df, feature_cols, target_cols):
        df = df.copy()

        if self.cfg.get("preprocessing", {}).get("dropna_features", True):
            df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[self.target_columns].to_numpy(dtype=np.float32)

        sample_end_indices = self._build_valid_indices(len(df))

        if len(sample_end_indices) == 0:
            raise ValueError(
                "No valid sample indices for this split. "
                "Check split sizes, input_length, and horizon_steps."
            )

        return X, y, sample_end_indices

    def _dataset_from_arrays(self, arrays):
        X, y, sample_end_indices = arrays

        return CustomForecastDataset(
            X=X,
            y=y,
            sample_end_indices=sample_end_indices,
            input_length=self.cfg["task"]["input_length"],
            target_specs=self.target_specs,
        )

    def _build_valid_indices(self, n_rows):
        input_length = self.cfg["task"]["input_length"]
        horizon_steps = self.max_horizon_steps
        prediction_stride = self.cfg["task"]["prediction_stride"]

        min_end_idx = input_length - 1
        max_end_idx = n_rows - 1 - horizon_steps

        if max_end_idx < min_end_idx:
            return np.array([], dtype=np.int64)

        all_indices = np.arange(min_end_idx, max_end_idx + 1, dtype=np.int64)

        offset = min_end_idx
        valid_mask = ((all_indices - offset) % prediction_stride) == 0

        return all_indices[valid_mask]

    def _read_extra_source(self, source_cfg):
        root = self.data_config["root"]

        subdir = source_cfg.get("subdir", "")
        file_name = source_cfg["file_name"]
        timestamp_col = source_cfg.get("timestamp_col", "time")

        path = os.path.join(root, subdir, file_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Extra source file not found: {path}")

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != ""]

        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in {path}")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).copy()
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        df = df.rename(columns={timestamp_col: "timestamp"})

        if df["timestamp"].duplicated().any():
            df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        selected_cols = source_cfg.get("columns")
        if selected_cols:
            missing_cols = [col for col in selected_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing columns in extra source '{source_cfg.get('name', file_name)}': {missing_cols}"
                )
            df = df[["timestamp"] + selected_cols].copy()

        resample_rule = source_cfg.get("resample_rule")
        if resample_rule:
            df = df.set_index("timestamp")

            agg = source_cfg.get("agg", "mean")
            df = df.resample(resample_rule).agg(agg)

            df = df.reset_index()

        fillna_method = source_cfg.get("fillna")
        if fillna_method == "ffill":
            df = df.ffill()
        elif fillna_method == "bfill":
            df = df.bfill()
        elif fillna_method == "interpolate":
            df = df.set_index("timestamp").interpolate(method="time").reset_index()
        elif fillna_method is not None and fillna_method != "none":
            raise ValueError(f"Unsupported fillna method: {fillna_method}")

        if source_cfg.get("prefix", True):
            source_name = source_cfg.get("name")
            if not source_name:
                raise ValueError("Each extra source needs a 'name' when prefix=True")

            rename_map = {
                col: f"{source_name}_{col}"
                for col in df.columns
                if col != "timestamp"
            }
            df = df.rename(columns=rename_map)

        return df

    def _merge_extra_sources(self, main_df):
        extra_sources = self.cfg.get("extra_sources", [])
        if not extra_sources:
            return main_df

        df = main_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        for src_cfg in extra_sources:
            extra_df = self._read_extra_source(src_cfg)

            merge_method = src_cfg.get("merge_method", "left")

            if merge_method == "asof":
                tolerance = src_cfg.get("tolerance")
                tolerance = pd.Timedelta(tolerance) if tolerance else None

                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    extra_df.sort_values("timestamp"),
                    on="timestamp",
                    direction=src_cfg.get("direction", "backward"),
                    tolerance=tolerance,
                )
            else:
                df = df.merge(extra_df, on="timestamp", how=merge_method)

        return df
