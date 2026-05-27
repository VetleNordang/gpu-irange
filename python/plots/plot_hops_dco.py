#!/usr/bin/env python3
"""
Plot HOP (graph hops) and DCO (distance computations) vs SearchEF for each method.

Produces two figures per dataset:
  hops_vs_ef[_env].png   — mean hops per query vs SearchEF, one line per method per range
  dco_vs_ef[_env].png    — mean distance computations per query vs SearchEF

Output: executable_data/{dataset}/results/analysis/

Usage:
  python python/plots/plot_hops_dco.py
  python python/plots/plot_hops_dco.py --dataset audi1m
  python python/plots/plot_hops_dco.py --dataset audi1m --env idun --aggregate
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "executable_data"

DATASETS = [
    {"key": "gist250k",  "name": "GIST1M 250k",      "path": "gist1m/250k"},
    {"key": "gist500k",  "name": "GIST1M 500k",       "path": "gist1m/500k"},
    {"key": "gist750k",  "name": "GIST1M 750k",       "path": "gist1m/750k"},
    {"key": "gist1000k", "name": "GIST1M 1M",         "path": "gist1m/1000k"},
    {"key": "video1m",   "name": "YouTube Video 1M",  "path": "video/1m"},
    {"key": "video2m",   "name": "YouTube Video 2M",  "path": "video/2m"},
    {"key": "video4m",   "name": "YouTube Video 4M",  "path": "video/4m"},
    {"key": "video8m",   "name": "YouTube Video 8M",  "path": "video/8m"},
    {"key": "audi1m",    "name": "YouTube Audio 1M",  "path": "audi/1m"},
    {"key": "audi2m",    "name": "YouTube Audio 2M",  "path": "audi/2m"},
    {"key": "audi4m",    "name": "YouTube Audio 4M",  "path": "audi/4m"},
    {"key": "audi8m",    "name": "YouTube Audio 8M",  "path": "audi/8m"},
]

METHODS = [
    {"key": "cpu_parallel", "label": "CPU-P",      "color": "tab:blue"},
    {"key": "gpu_normal",   "label": "GPU Normal", "color": "tab:orange"},
    {"key": "gpu_pq",       "label": "GPU PQ",     "color": "tab:green"},
    {"key": "gpu_root",     "label": "GPU Root",   "color": "tab:purple"},
]

RANGES      = [2, 5, 8]
LINESTYLES  = {2: "-", 5: "--", 8: ":"}
RANGE_LABEL = {2: "Range 2 (wide)", 5: "Range 5 (medium)", 8: "Range 8 (narrow)"}


def parse_range(filename):
    import re
    # Match any number in the filename that corresponds to a known range
    for m in re.finditer(r'(\d+)', Path(filename).stem):
        val = int(m.group(1))
        if val in RANGES:
            return val
    return None


def load_method(method_dir):
    if not method_dir.is_dir():
        return None

    agg_dir = method_dir / "aggregate"
    if agg_dir.is_dir():
        frames = []
        for csv in sorted(agg_dir.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = pd.read_csv(csv)
            rename = {}
            if "HOP_mean" in df.columns:
                rename["HOP_mean"] = "HOP"
            if "DCO_mean" in df.columns:
                rename["DCO_mean"] = "DCO"
            df = df.rename(columns=rename)
            if "HOP" not in df.columns or "DCO" not in df.columns:
                continue
            df["range"] = rng
            frames.append(df[["range", "SearchEF", "HOP", "DCO"]])
        if frames:
            return pd.concat(frames, ignore_index=True)

    run_dirs = sorted(d for d in method_dir.glob("run*") if d.is_dir() and any(d.glob("*.csv")))
    sources = run_dirs if run_dirs else [method_dir]
    frames = []
    for src in sources:
        for csv in sorted(src.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = pd.read_csv(csv)
            if "HOP" not in df.columns or "DCO" not in df.columns:
                continue
            df["range"] = rng
            frames.append(df[["range", "SearchEF", "HOP", "DCO"]])
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["range", "SearchEF"])[["HOP", "DCO"]].mean().reset_index()


def plot_metric(dataset_name, data, metric, ylabel, outpath, hardware=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    title = f"{metric} vs SearchEF — {dataset_name}"
    if hardware:
        title += f" ({hardware})"
    ax.set_title(title)
    ax.set_xlabel("SearchEF")
    ax.set_ylabel(ylabel)

    plotted = False
    for method in METHODS:
        df = data.get(method["key"])
        if df is None or metric not in df.columns:
            continue
        for rng in RANGES:
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
    p.add_argument("--dataset",   default=None, help="Dataset key prefix (e.g. 'audi1m'). Omit for all.")
    p.add_argument("--env",       default=None, help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware",  default="",   help="Hardware label for figure titles.")
    p.add_argument("--aggregate", action="store_true", help="Re-run aggregate_results.py before plotting.")
    args = p.parse_args()

    if args.aggregate:
        agg_script = Path(__file__).resolve().parent.parent / "aggregate_results.py"
        print("Running aggregate_results.py --all ...")
        result = subprocess.run([sys.executable, str(agg_script), "--all"], check=False)
        if result.returncode != 0:
            print("Warning: aggregate_results.py exited with errors — plotting with existing data.")

    suffix = f"_{args.env}" if args.env else ""

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"].startswith(args.dataset)]
        if not datasets:
            print(f"No dataset matched '{args.dataset}'.")
            return

    for d in datasets:
        print(f"\n{d['name']}")
        results_dir = BASE_DIR / d["path"] / "results"
        data = {m["key"]: load_method(results_dir / m["key"]) for m in METHODS}
        if all(v is None for v in data.values()):
            print("  no data found — skipping")
            continue
        out = results_dir / "analysis"
        plot_metric(d["name"], data, "HOP", "Mean hops per query",
                    out / f"hops_vs_ef{suffix}.png", args.hardware)
        plot_metric(d["name"], data, "DCO", "Mean distance computations per query",
                    out / f"dco_vs_ef{suffix}.png", args.hardware)


if __name__ == "__main__":
    main()
