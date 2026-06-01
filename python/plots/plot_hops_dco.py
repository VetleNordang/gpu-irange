#!/usr/bin/env python3
"""
Plot HOP (graph hops) and DCO (distance computations) vs SearchEF for each method.

Produces two figures per dataset:
  hops_vs_ef[_env].png   — mean hops per query vs SearchEF, one line per method per range
  dco_vs_ef[_env].png    — mean distance computations per query vs SearchEF

Output: {base_dir}/{dataset}/results/analysis/

Usage:
  python python/plots/plot_hops_dco.py --idun --dataset audi1m
  python python/plots/plot_hops_dco.py --idun --dataset audi1m --methods cpu_parallel,gpu_normal
  python python/plots/plot_hops_dco.py --idun --dataset audi1m --methods cpu_parallel,gpu_normal --ranges 2,8
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR_P100 = Path(__file__).resolve().parent.parent.parent / "executable_data"
BASE_DIR_IDUN = Path(__file__).resolve().parent.parent.parent / "temp_idun_results"
THESIS_FIGS   = Path(__file__).resolve().parent.parent.parent / "thesis/Master/figs/results"

DATASETS = [
    {"key": "gist250k",  "name": "GIST1M 250k",      "path": "gist1m/250k"},
    {"key": "gist500k",  "name": "GIST1M 500k",       "path": "gist1m/500k"},
    {"key": "gist750k",  "name": "GIST1M 750k",       "path": "gist1m/750k"},
    {"key": "gist1000k", "name": "GIST1M 1M",         "path": "gist1m/1000k"},
    {"key": "video1m",   "name": "YouTube Video 1M",  "path": "video/1m"},
    {"key": "video2m",   "name": "YouTube Video 2M",  "path": "video/2m"},
    {"key": "video4m",   "name": "YouTube Video 4M",  "path": "video/4m"},
    {"key": "video6m",   "name": "YouTube Video 6M",  "path": "video/6m"},
    {"key": "audi1m",    "name": "YouTube Audio 1M",  "path": "audi/1m"},
    {"key": "audi2m",    "name": "YouTube Audio 2M",  "path": "audi/2m"},
    {"key": "audi4m",    "name": "YouTube Audio 4M",  "path": "audi/4m"},
    {"key": "audi6m",    "name": "YouTube Audio 6M",  "path": "audi/6m"},
]

ALL_METHODS = [
    {"key": "cpu_parallel", "label": "CPU-P",      "color": "tab:blue"},
    {"key": "gpu_normal",   "label": "GPU Normal", "color": "tab:orange"},
    {"key": "gpu_pq",       "label": "GPU PQ",     "color": "tab:green"},
    {"key": "gpu_root",     "label": "GPU Root",   "color": "tab:purple"},
    {"key": "cpu_serial",   "label": "CPU-S",      "color": "tab:gray"},
]

ALL_RANGES  = [2, 5, 8]
LINESTYLES  = {2: "-", 5: "--", 8: ":"}
RANGE_LABEL = {2: "Range 2 (wide)", 5: "Range 5 (medium)", 8: "Range 8 (narrow)"}


def parse_range(filename):
    for m in re.finditer(r'(\d+)', Path(filename).stem):
        val = int(m.group(1))
        if val in ALL_RANGES:
            return val
    return None


def _run_aggregate_script(method_dir):
    script = Path(__file__).resolve().parent.parent / "aggregate_results.py"
    if not script.exists():
        return False
    print(f"  [aggregate] running aggregate_results.py for {method_dir} ...")
    result = subprocess.run(
        [sys.executable, str(script), "--mode_dir", str(method_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [aggregate] FAILED:\n{result.stderr.strip()}")
        return False
    return True


def load_method(method_dir):
    if not method_dir.is_dir():
        return None

    _run_aggregate_script(method_dir)

    agg_dir = method_dir / "aggregate"
    if agg_dir.is_dir():
        frames = []
        for csv in sorted(agg_dir.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in ALL_RANGES:
                continue
            df = pd.read_csv(csv)
            rename = {}
            if "HOP_mean" in df.columns:
                rename["HOP_mean"] = "HOP"
            if "DCO_mean" in df.columns:
                rename["DCO_mean"] = "DCO"
            df = df.rename(columns=rename)
            if "HOP" not in df.columns and "DCO" not in df.columns:
                continue
            df["range"] = rng
            keep = ["range", "SearchEF"] + [c for c in ("HOP", "DCO") if c in df.columns]
            frames.append(df[keep])
        if frames:
            return pd.concat(frames, ignore_index=True)

    run_dirs = sorted(d for d in method_dir.glob("run*") if d.is_dir() and any(d.glob("*.csv")))
    sources = run_dirs if run_dirs else [method_dir]
    frames = []
    for src in sources:
        for csv in sorted(src.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in ALL_RANGES:
                continue
            df = pd.read_csv(csv)
            if "HOP" not in df.columns and "DCO" not in df.columns:
                continue
            df["range"] = rng
            keep = ["range", "SearchEF"] + [c for c in ("HOP", "DCO") if c in df.columns]
            frames.append(df[keep])
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    cols = [c for c in ("HOP", "DCO") if c in combined.columns]
    return combined.groupby(["range", "SearchEF"])[cols].mean().reset_index()


def _copy_to_thesis(src, dataset_path):
    import shutil
    dest = THESIS_FIGS / dataset_path / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied to thesis → {dest.relative_to(THESIS_FIGS.parent.parent.parent)}")


def plot_metric(dataset_name, data, methods, active_ranges, metric, ylabel, outpath, hardware=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("SearchEF")
    ax.set_ylabel(ylabel)

    plotted = False
    for method in methods:
        df = data.get(method["key"])
        if df is None or metric not in df.columns:
            continue
        for rng in active_ranges:
            subset = df[df["range"] == rng].sort_values("SearchEF")
            if subset.empty:
                continue
            ax.plot(
                subset["SearchEF"], subset[metric],
                color=method["color"],
                linestyle=LINESTYLES[rng],
                label=f"{method['label']} — {RANGE_LABEL[rng]}",
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default=None,
                   help="Dataset key prefix (e.g. 'audi1m'). Omit for all.")
    p.add_argument("--methods",  default=None,
                   help="Comma-separated method keys, e.g. 'cpu_parallel,gpu_normal'. "
                        "Valid: cpu_parallel, gpu_normal, gpu_pq, gpu_root, cpu_serial.")
    p.add_argument("--ranges",   default="2,5,8",
                   help="Comma-separated range indices to plot. Default: 2,5,8.")
    p.add_argument("--env",      default=None,
                   help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware", default="",
                   help="Hardware label for figure titles.")
    p.add_argument("--copy-to-thesis", action="store_true",
                   help="Copy output figures into thesis/Master/figs/results/{dataset}/.")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--idun", action="store_true",
                     help="Use IDUN results folder (temp_idun_results/).")
    src.add_argument("--p100", action="store_true",
                     help="Use P100 results folder (executable_data/). This is the default.")
    src.add_argument("--base-dir", default=None, type=Path,
                     help="Explicit path to results root (overrides --idun/--p100).")

    args = p.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    elif args.idun:
        base_dir = BASE_DIR_IDUN
    else:
        base_dir = BASE_DIR_P100

    if args.methods:
        keys = [k.strip() for k in args.methods.split(",")]
        valid = {m["key"] for m in ALL_METHODS}
        unknown = [k for k in keys if k not in valid]
        if unknown:
            p.error(f"Unknown method key(s): {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}")
        methods = [m for m in ALL_METHODS if m["key"] in keys]
    else:
        methods = [m for m in ALL_METHODS if m["key"] != "cpu_serial"]

    try:
        active_ranges = [int(r.strip()) for r in args.ranges.split(",")]
    except ValueError:
        p.error("--ranges must be comma-separated integers, e.g. '2,8'")
    unknown_ranges = [r for r in active_ranges if r not in ALL_RANGES]
    if unknown_ranges:
        p.error(f"Unknown range(s): {unknown_ranges}. Valid: {ALL_RANGES}")

    suffix = f"_{args.env}" if args.env else ""

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"].startswith(args.dataset)]
        if not datasets:
            print(f"No dataset matched '{args.dataset}'.")
            return

    for d in datasets:
        print(f"\n{d['name']}")
        results_dir = base_dir / d["path"] / "results"
        data = {m["key"]: load_method(results_dir / m["key"]) for m in methods}
        if all(v is None for v in data.values()):
            print("  no data found — skipping")
            continue
        out = results_dir / "analysis"
        hops_out = out / f"hops_vs_ef{suffix}.png"
        dco_out  = out / f"dco_vs_ef{suffix}.png"
        plot_metric(d["name"], data, methods, active_ranges, "HOP",
                    "Mean hops per query", hops_out, args.hardware)
        plot_metric(d["name"], data, methods, active_ranges, "DCO",
                    "Mean distance computations per query", dco_out, args.hardware)
        if args.copy_to_thesis:
            if hops_out.exists():
                _copy_to_thesis(hops_out, d["path"])
            if dco_out.exists():
                _copy_to_thesis(dco_out, d["path"])


if __name__ == "__main__":
    main()
