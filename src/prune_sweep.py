from __future__ import annotations

import argparse
from pathlib import Path

from utils import load_config


def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable representation."""
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def load_sweep_name(config_path: Path) -> str:
    """
    Read run.prune_name from a sweep configuration.
    """

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Sweep config not found: {config_path}"
        )

    config = load_config(config_path)

    try:
        prune_name = config["run"]["prune_name"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"'run.prune_name' missing in {config_path}"
        ) from exc

    if not isinstance(prune_name, str) or not prune_name.strip():
        raise ValueError(
            "'run.prune_name' must be a non-empty string"
        )

    return prune_name


def find_prediction_files(run_dir: Path) -> list[Path]:
    """
    Find prediction files following

        id_xxx/zones/<zone>/<year>/preds.csv
    """

    zones_dir = run_dir / "zones"

    if not zones_dir.is_dir():
        return []

    return sorted(
        path
        for path in zones_dir.glob("*/*/preds.csv")
        if path.is_file()
    )


def prune_sweep(
    config_path: Path,
    runs_root: Path,
    delete: bool,
) -> None:

    prune_name = load_sweep_name(config_path)

    sweep_dir = runs_root / prune_name

    if not sweep_dir.is_dir():
        raise FileNotFoundError(
            f"Sweep directory not found: {sweep_dir}"
        )

    run_dirs = sorted(
        path
        for pattern in ("id_*", "trial_*")
        for path in sweep_dir.glob(pattern)
        if path.is_dir()
    )

    if not run_dirs:
        raise FileNotFoundError(
            f"No id_* or trial_* directories found in {sweep_dir}"
        )

    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Sweep:  {prune_name}")
    print(f"Path:   {sweep_dir}")
    print(f"Runs:   {len(run_dirs)}")
    print("=" * 70)
    print()

    total_files = 0
    total_size = 0

    for run_dir in run_dirs:

        pred_files = find_prediction_files(run_dir)

        run_size = sum(
            path.stat().st_size
            for path in pred_files
        )

        total_files += len(pred_files)
        total_size += run_size

        if delete:
            for path in pred_files:
                path.unlink()

            action = "DELETED"
        else:
            action = "WOULD DELETE"

        print(
            f"[{action}] "
            f"{run_dir.name}: "
            f"{len(pred_files)} files, "
            f"{format_size(run_size)}"
        )

    print()
    print("=" * 70)

    if delete:
        print("PRUNING COMPLETE")
        print(f"Deleted files: {total_files}")
        print(f"Freed space:   {format_size(total_size)}")
    else:
        print("DRY RUN COMPLETE")
        print(f"Files found:   {total_files}")
        print(f"Space to free: {format_size(total_size)}")
        print()
        print("No files were deleted.")

    print("=" * 70)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Prune preds.csv files from an evaluated hyperparameter sweep."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the sweep YAML configuration.",
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing sweep runs (default: runs).",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Actually delete preds.csv files. "
            "Without this flag a dry run is performed."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prune_sweep(
        config_path=args.config,
        runs_root=args.runs_root,
        delete=args.delete,
    )


if __name__ == "__main__":
    main()