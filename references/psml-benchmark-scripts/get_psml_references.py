#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
from pathlib import Path

import yaml

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_cmd_capture(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout.strip()


def git_commit(repo_root: Path) -> str:
    return run_cmd_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)

def write_meta(meta_path: Path, reference_id: str, benchmark_commit_hash: str, picked_version_dir: str) -> None:
    meta = {
        "schema_version": 1,
        "reference_id": reference_id,
        "created_at_utc": utc_now_iso(),
        "benchmark_commit": benchmark_commit_hash,
        "picked_version_dir": picked_version_dir,
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-id", required=True, help="e.g. cnn_ref_v1")
    ap.add_argument("--version-dir", required=True, help="Path to .../logs/<model>/version_N")
    ap.add_argument("--out-root", required=True, help="Output root, e.g. references/psml_forecasting")
    ap.add_argument("--benchmark-repo-root", required=True, help="Repo root of the benchmark submodule")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing reference folder")
    args = ap.parse_args()

    reference_id = args.reference_id
    version_dir = Path(args.version_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    benchmark_repo_root = Path(args.benchmark_repo_root).expanduser().resolve()

    if not version_dir.exists():
        raise FileNotFoundError(f"version_dir not found: {version_dir}")

    summary_src = version_dir / "summary.json"
    config_src = version_dir / "config.yaml"

    if not summary_src.exists():
        # fallback: nested search (just in case)
        hits = list(version_dir.glob("**/summary.json"))
        if len(hits) == 1:
            summary_src = hits[0]
        else:
            raise FileNotFoundError(f"summary.json not found in {version_dir}")

    if not config_src.exists():
        # fallback: nested config.yaml
        hits = list(version_dir.glob("**/config.yaml"))
        if len(hits) == 1:
            config_src = hits[0]
        else:
            raise FileNotFoundError(f"config.yaml not found in {version_dir}")

    commit_hash = git_commit(benchmark_repo_root)

    ref_dir = out_root / reference_id
    if ref_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Reference already exists: {ref_dir}. Use --overwrite.")
        shutil.rmtree(ref_dir)

    ref_dir.mkdir(parents=True, exist_ok=True)

    # Copy frozen artifacts
    shutil.copy2(summary_src, ref_dir / "summary.json")
    shutil.copy2(config_src, ref_dir / "config_used.yaml")

    # Minimal meta
    write_meta(
        meta_path=ref_dir / "meta.yaml",
        reference_id=reference_id,
        benchmark_commit_hash=commit_hash,
        picked_version_dir=version_dir.name,
    )

    print(f"OK: Frozen reference written to: {ref_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
