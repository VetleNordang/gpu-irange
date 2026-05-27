#!/usr/bin/env python3
"""
Plot Recall@10, Recall@50, Recall@100 vs SearchEF for each method.

Produces two figure types per dataset:

  recall_vs_ef[_env].png
      Three lines per method (one per threshold), x=SearchEF, y=Recall.
      Shows how recall degrades as you ask for more neighbours.

  recall_vs_qps[_env].png
      Recall@10 vs QPS for all methods on one figure (the primary thesis
      tradeoff figure that was missing due to the old single-recall schema).

Output: executable_data/{dataset}/results/analysis/

Usage:
  python python/plots/plot_recall_thresholds.py
  python python/plots/plot_recall_thresholds.py --dataset audi1m
  python python/plots/plot_recall_thresholds.py --env idun
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

RANGES     = [2, 5, 8]
RANGE_LABEL = {2: "wide", 5: "medium", 8: "narrow"}
COLUMNS    = ["SearchEF", "Recall", "QPS", "DCO", "HOP"]

THRESHOLD_STYLE = {
    "Recall@10":  {"ls": "-",  "label": "Recall@10"},
    "Recall@50":  {"ls": "--", "label": "Recall@50"},
    "Recall@100": {"ls": ":",  "label": "Recall@100"},
}


def parse_range(filename):
    m = re.search(r"(\d+)(?:_gpu)?\.csv$", filename)
    return int(m.group(1)) if m else None


def read_csv(path):
    with open(path) as fh:
        first = fh.readline()
    has_header = first.startswith("SearchEF")
    df = pd.read_csv(path) if has_header else pd.read_csv(path, names=COLUMNS)
    if "Recall" in df.columns and "Recall@10" not in df.columns:
        df = df.rename(columns={"Recall": "Recall@10"})
    needed = {"SearchEF", "Recall@10", "QPS"}
    if not needed.issubset(df.columns):
        return None
    keep = [c for c in ["SearchEF", "Recall@10", "Recall@50", "Recall@100", "QPS"] if c in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["SearchEF", "Recall@10", "QPS"])


def load_method(method_dir):
    """Return mean over run1..runN, or None. Falls back to top-level CSVs."""
    if not method_dir.is_dir():
        return None

    # Try aggregate first
    agg_dir = method_dir / "aggregate"
    if agg_dir.is_dir():
        frames = []
        for csv in sorted(agg_dir.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = pd.read_csv(csv)
            # Map aggregate column names back to plain names
            rename = {}
            for col in ("Recall@10_mean", "Recall_mean"):
                if col in df.columns:
                    rename[col] = "Recall@10"
                    break
            for col in ("Recall@50_mean",):
                if col in df.columns:
                    rename[col] = "Recall@50"
            for col in ("Recall@100_mean",):
                if col in df.columns:
                    rename[col] = "Recall@100"
            if "QPS_mean" in df.columns:
                rename["QPS_mean"] = "QPS"
            df = df.rename(columns=rename)
            if "Recall@10" not in df.columns or "QPS" not in df.columns:
                continue
            df["range"] = rng
            keep = [c for c in ["range", "SearchEF", "Recall@10", "Recall@50", "Recall@100", "QPS"] if c in df.columns]
            frames.append(df[keep])
        if frames:
            return pd.concat(frames, ignore_index=True)

    # Fall back to raw run dirs or top-level CSVs
    run_dirs = sorted(d for d in method_dir.glob("run*") if d.is_dir())
    sources = run_dirs if run_dirs else [method_dir]
    frames = []
    for src in sources:
        for csv in sorted(src.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = read_csv(csv)
            if df is None or df.empty:
                continue
            df["range"] = rng
            frames.append(df)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    agg_cols = {"Recall@10": ("Recall@10", "mean"), "QPS": ("QPS", "mean")}
    if "Recall@50" in raw.columns:
        agg_cols["Recall@50"] = ("Recall@50", "mean")
    if "Recall@100" in raw.columns:
        agg_cols["Recall@100"] = ("Recall@100", "mean")
    return raw.groupby(["range", "SearchEF"], as_index=False).agg(**agg_cols)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(BASE_DIR)}")


def plot_recall_vs_ef(name, data, out_path, hw_label=""):
    """One panel per range. Lines: one per method×threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    drew = False
    for ax, rng in zip(axes, RANGES):
        for m in METHODS:
            df = data.get(m["key"])
            if df is None:
                continue
            sub = df[df["range"] == rng].sort_values("SearchEF")
            if sub.empty:
                continue
            for thresh, style in THRESHOLD_STYLE.items():
                if thresh not in sub.columns:
                    continue
                ax.plot(sub["SearchEF"], sub[thresh],
                        color=m["color"], ls=style["ls"], lw=1.6, marker="o", ms=3,
                        label=f"{m['label']} {style['label']}")
                drew = True
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Range {rng} ({RANGE_LABEL[rng]})", fontsize=11)
        ax.set_xlabel("SearchEF")
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel("Recall")
    # Compact legend: deduplicate by label
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=min(len(seen), 6), bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)
    t = f"{name} — {hw_label}: Recall vs SearchEF" if hw_label else f"{name}: Recall vs SearchEF"
    fig.suptitle(t, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_recall_vs_qps(name, data, out_path, hw_label=""):
    """Three panels (one per range). Recall@10 vs QPS for all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    handles, labels = [], []
    drew = False
    for ax, rng in zip(axes, RANGES):
        for m in METHODS:
            df = data.get(m["key"])
            if df is None:
                continue
            sub = df[df["range"] == rng].sort_values("Recall@10")
            if sub.empty or "Recall@10" not in sub.columns:
                continue
            line, = ax.plot(sub["Recall@10"], sub["QPS"],
                            color=m["color"], lw=1.8, marker="o", ms=4,
                            label=m["label"])
            drew = True
            if m["label"] not in labels:
                handles.append(line)
                labels.append(m["label"])
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Range {rng} ({RANGE_LABEL[rng]})", fontsize=11)
        ax.set_xlabel("Recall@10")
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel("QPS (queries per second)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    t = f"{name} — {hw_label}: Recall@10 vs QPS" if hw_label else f"{name}: Recall@10 vs QPS"
    fig.suptitle(t, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="Dataset key prefix (e.g. 'audi1m'). Omit for all.")
    p.add_argument("--env", default=None, help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware", default="", help="Hardware label for figure titles.")
    args = p.parse_args()

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
        plot_recall_vs_ef(d["name"], data,
                          out / f"recall_vs_ef{suffix}.png", args.hardware)
        plot_recall_vs_qps(d["name"], data,
                           out / f"recall_vs_qps{suffix}.png", args.hardware)

    print("\ndone")


if __name__ == "__main__":
    main()
