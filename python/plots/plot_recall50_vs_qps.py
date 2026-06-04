#!/usr/bin/env python3
"""
Plot Recall@50 vs QPS for video/1m and audi/1m.

Usage:
  python python/plots/plot_recall50_vs_qps.py
  python python/plots/plot_recall50_vs_qps.py --env idun
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR_P100 = Path(__file__).resolve().parent.parent.parent / "executable_data"
BASE_DIR_IDUN = Path(__file__).resolve().parent.parent.parent / "temp_idun_results"

DATASETS = [
    {"key": "video1m", "name": "YouTube Video 1M", "path": "video/1m"},
    {"key": "video6m", "name": "YouTube Video 6M", "path": "video/6m"},
    {"key": "audi1m",  "name": "YouTube Audio 1M", "path": "audi/1m"},
    {"key": "audi6m",  "name": "YouTube Audio 6M", "path": "audi/6m"},
]

METHODS = [
    {"key": "cpu_parallel", "label": "CPU-P",     "color": "tab:blue"},
    {"key": "gpu_normal",   "label": "GPU-Full",  "color": "tab:orange"},
    {"key": "gpu_pq",       "label": "GPU-PQ",    "color": "tab:green"},
    {"key": "gpu_root",     "label": "GPU Root",  "color": "tab:purple"},
]

RANGES      = [2, 5, 8]
RANGE_LABEL = {2: "wide", 5: "medium", 8: "narrow"}


def _parse_range(filename):
    m = re.search(r"(\d+)(?:_gpu)?\.csv$", filename)
    return int(m.group(1)) if m else None


def _read_csv(path):
    with open(path) as fh:
        first = fh.readline()
    has_header = first.startswith("SearchEF")
    df = pd.read_csv(path) if has_header else pd.read_csv(
        path, names=["SearchEF", "Recall@10", "QPS", "DCO", "HOP"])
    if "Recall" in df.columns and "Recall@10" not in df.columns:
        df = df.rename(columns={"Recall": "Recall@10"})
    if "Recall@50" not in df.columns or "QPS" not in df.columns:
        return None
    return df[["SearchEF", "Recall@50", "QPS"]].apply(
        pd.to_numeric, errors="coerce").dropna()


def _load_method(method_dir):
    if not method_dir.is_dir():
        return None

    agg_dir = method_dir / "aggregate"
    if agg_dir.is_dir():
        frames = []
        for csv in sorted(agg_dir.glob("*.csv")):
            rng = _parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = pd.read_csv(csv)
            rename = {}
            for col in ("Recall@50_mean",):
                if col in df.columns:
                    rename[col] = "Recall@50"
            if "QPS_mean" in df.columns:
                rename["QPS_mean"] = "QPS"
            df = df.rename(columns=rename)
            if "Recall@50" not in df.columns or "QPS" not in df.columns:
                continue
            df["range"] = rng
            frames.append(df[["range", "SearchEF", "Recall@50", "QPS"]])
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            return combined

    run_dirs = sorted(d for d in method_dir.glob("run*")
                      if d.is_dir() and any(d.glob("*.csv")))
    sources = run_dirs if run_dirs else [method_dir]
    frames = []
    for src in sources:
        for csv in sorted(src.glob("*.csv")):
            rng = _parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = _read_csv(csv)
            if df is None or df.empty:
                continue
            df["range"] = rng
            frames.append(df)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    return raw.groupby(["range", "SearchEF"], as_index=False).agg(
        Recall50=("Recall@50", "mean"),
        QPS=("QPS", "mean"),
    ).rename(columns={"Recall50": "Recall@50"})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_recall50_vs_qps(name, data, out_path, methods=METHODS):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    handles, labels = [], []
    drew = False
    for ax, rng in zip(axes, RANGES):
        for m in methods:
            df = data.get(m["key"])
            if df is None:
                continue
            sub = df[df["range"] == rng].sort_values("Recall@50")
            if sub.empty:
                continue
            line, = ax.plot(sub["Recall@50"], sub["QPS"],
                            color=m["color"], lw=1.8, marker="o", ms=4,
                            label=m["label"])
            drew = True
            if m["label"] not in labels:
                handles.append(line)
                labels.append(m["label"])
        ax.set_yscale("log")
        ax.set_xlim(0, 1.05)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Range {rng} ({RANGE_LABEL[rng]})", fontsize=11)
        ax.set_xlabel("Recall@50")
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel("QPS (queries per second)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(f"{name}: Recall@50 vs QPS", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None, help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--dataset", default=None, help="Dataset key prefix to filter, e.g. 'video6m' or 'audi'.")
    p.add_argument("--method", nargs="+", default=None,
                   help="Restrict to these method keys, e.g. --method cpu_parallel gpu_normal")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--idun", action="store_true")
    src.add_argument("--p100", action="store_true")
    args = p.parse_args()

    base_dir = BASE_DIR_IDUN if args.idun else BASE_DIR_P100
    suffix = f"_{args.env}" if args.env else ""

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"].startswith(args.dataset)]
        if not datasets:
            print(f"No dataset matched '{args.dataset}'. Valid keys: {[d['key'] for d in DATASETS]}")
            return

    methods = METHODS
    if args.method:
        methods = [m for m in METHODS if m["key"] in args.method]
        if not methods:
            print(f"No methods matched {args.method}. Valid keys: {[m['key'] for m in METHODS]}")
            return

    for d in datasets:
        print(f"\n{d['name']}")
        results_dir = base_dir / d["path"] / "results"
        data = {m["key"]: _load_method(results_dir / m["key"]) for m in methods}
        if all(v is None for v in data.values()):
            print("  no data found — skipping")
            continue
        out = results_dir / "analysis" / f"recall50_vs_qps{suffix}.png"
        plot_recall50_vs_qps(d["name"], data, out, methods=methods)

    print("\ndone")


if __name__ == "__main__":
    main()
