#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

VERSION_RE = re.compile(r"^version_(\d+)$")

@dataclass
class ModelSpec:
    reference_id: str
    entrypoint: str
    args: List[str]
    log_subdir: str


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_cmd(cmd: List[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def list_versions(model_logs_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not model_logs_dir.exists():
        return out
    for p in model_logs_dir.iterdir():
        if not p.is_dir():
            continue
        m = VERSION_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def pick_version_with_summary(new_dirs: Dict[int, Path]) -> Tuple[int, Path]:
    """
    Pick the highest version_N directory (among newly created ones) that contains summary.json. (since the submodule creates 2 folders with different version number)
    """
    candidates = [(n, d) for n, d in new_dirs.items() if (d / "summary.json").exists()]
    if not candidates:
        candidates = []
        for n, d in new_dirs.items():
            if list(d.glob("**/summary.json")):
                candidates.append((n, d))
        if not candidates:
            raise FileNotFoundError(
                "No summary.json found in any newly created version dir. "
                "Check benchmark output layout or update the locator."
            )
    return max(candidates)


def call_snapshotter(
    snapshotter_path: Path,
    reference_id: str,
    version_dir: Path,
    out_root: Path,
    benchmark_repo_root: Path,
    overwrite: bool,
) -> None:
    cmd = [
        "python3",
        str(snapshotter_path),
        "--reference-id",
        reference_id,
        "--version-dir",
        str(version_dir.resolve()),
        "--out-root",
        str(out_root.resolve()),
        "--benchmark-repo-root",
        str(benchmark_repo_root.resolve()),
    ]

    if overwrite:
        cmd.append("--overwrite")

    run_cmd(cmd, cwd=Path.cwd())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rebuild_psml_references.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-runs", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    benchmark_repo_root = Path(cfg["benchmark"]["repo_root"]).resolve()
    workdir = Path(cfg["benchmark"]["workdir"]).resolve()
    logs_root = Path(cfg["benchmark"]["logs_root"]).resolve()
    out_root = Path(cfg["references"]["out_root"]).resolve()

    snapshotter_path = (Path(__file__).parent / "get_psml_references.py").resolve()
    if not snapshotter_path.exists():
        raise FileNotFoundError(f"Snapshotter not found: {snapshotter_path}")

    models = [
        ModelSpec(
            reference_id=m["reference_id"],
            entrypoint=m["entrypoint"],
            args=m.get("args", []),
            log_subdir=m["log_subdir"],
        )
        for m in cfg["models"]
    ]

    for m in models:
        model_logs_dir = logs_root / m.log_subdir
        before = list_versions(model_logs_dir)

        cmd = ["python3", m.entrypoint] + m.args
        print(f"\n RUN: {m.reference_id}: {' '.join(cmd)} (cwd={workdir})")
        run_cmd(cmd, cwd=workdir)

        after = list_versions(model_logs_dir)
        new_nums = sorted(set(after.keys()) - set(before.keys()))
        if not new_nums:
            raise RuntimeError(f"No new version_* dirs detected in {model_logs_dir}")

        new_dirs = {n: after[n] for n in new_nums}
        pick_num, pick_dir = pick_version_with_summary(new_dirs)
        print(f"PICK: {m.reference_id}: picked {pick_dir.name}")

        # Snapshot (copy files + generate meta)
        call_snapshotter(
            snapshotter_path=snapshotter_path,
            reference_id=m.reference_id,
            version_dir=pick_dir,
            out_root=out_root,
            benchmark_repo_root=benchmark_repo_root,
            overwrite=args.overwrite,
        )

        # Cleanup: remove other newly created runs
        if not args.keep_runs:
            for n, d in new_dirs.items():
                if d.resolve() == pick_dir.resolve():
                    continue
                print(f"CLEANUP: removing extra run dir: {d}")
                shutil.rmtree(d)

    print("\n DONE: References rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
