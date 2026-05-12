#!/usr/bin/env python3
"""
Aggregate multi-run search results into mean / std / CI95 per SearchEF.

Usage:
    # Single mode directory:
    python python/aggregate_results.py --mode_dir executable_data/gist1m/250k/results/cpu_serial

    # All datasets and modes at once:
    python python/aggregate_results.py --all

Output: {mode_dir}/aggregate/*.csv
Columns: SearchEF, Recall_mean, Recall_std, Recall_ci95,
         QPS_mean, QPS_std, QPS_ci95,
         DCO_mean, DCO_std, DCO_ci95,
         HOP_mean, HOP_std, HOP_ci95,
         n_runs
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "executable_data"

MODES = ["cpu_serial", "cpu_parallel", "gpu_normal", "gpu_pq"]

METRICS = ["Recall", "QPS", "DCO", "HOP"]


def aggregate_mode_dir(mode_dir: Path) -> bool:
    run_dirs = sorted(
        [d for d in mode_dir.iterdir() if d.is_dir() and re.fullmatch(r"run\d+", d.name)],
        key=lambda d: int(d.name[3:]),
    )

    if not run_dirs:
        print(f"  no run dirs in {mode_dir} — skipping")
        return False

    n = len(run_dirs)
    print(f"  {n} runs: {[d.name for d in run_dirs]}")
    t_crit = stats.t.ppf(0.975, df=n - 1)

    csv_names = sorted({f.name for d in run_dirs for f in d.glob("*.csv")})
    if not csv_names:
        print(f"  no CSVs found — skipping")
        return False

    out_dir = mode_dir / "aggregate"
    out_dir.mkdir(exist_ok=True)

    for csv_name in csv_names:
        frames = []
        for run_dir in run_dirs:
            p = run_dir / csv_name
            if not p.exists():
                print(f"  WARNING: missing {p}")
                continue
            frames.append(pd.read_csv(p))

        if not frames:
            continue

        available = [m for m in METRICS if m in frames[0].columns]
        combined = pd.concat(frames)
        grouped = combined.groupby("SearchEF")[available]

        agg = grouped.mean().rename(columns={m: f"{m}_mean" for m in available})
        std = grouped.std(ddof=1).rename(columns={m: f"{m}_std" for m in available})
        agg = agg.join(std).reset_index()

        for m in available:
            agg[f"{m}_ci95"] = t_crit * agg[f"{m}_std"] / np.sqrt(n)

        agg["n_runs"] = n

        cols = ["SearchEF"]
        for m in available:
            cols += [f"{m}_mean", f"{m}_std", f"{m}_ci95"]
        cols += ["n_runs"]
        agg = agg[cols].sort_values("SearchEF", ascending=False)

        out_path = out_dir / csv_name
        agg.to_csv(out_path, index=False, float_format="%.6f")
        print(f"    → {out_path.relative_to(PROJECT_ROOT)}")

    return True


def aggregate_all() -> None:
    found = 0
    for results_dir in sorted(DATA_ROOT.glob("*/*/results")):
        for mode in MODES:
            mode_dir = results_dir / mode
            if not mode_dir.is_dir():
                continue
            label = f"{results_dir.parent.parent.name}/{results_dir.parent.name}/{mode}"
            print(f"\n{label}")
            if aggregate_mode_dir(mode_dir):
                found += 1

    print(f"\ndone — aggregated {found} mode dir(s)")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode_dir", type=Path, help="path to a single mode dir (e.g. …/cpu_serial)")
    group.add_argument("--all", action="store_true", help="process all datasets and modes")
    args = parser.parse_args()

    if args.all:
        aggregate_all()
    else:
        mode_dir = args.mode_dir.resolve()
        if not mode_dir.is_dir():
            print(f"ERROR: not a directory: {mode_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"{mode_dir.relative_to(PROJECT_ROOT)}")
        aggregate_mode_dir(mode_dir)


if __name__ == "__main__":
    main()
