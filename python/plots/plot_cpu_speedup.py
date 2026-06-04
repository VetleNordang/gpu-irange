#!/usr/bin/env python3
"""
Plot CPU-P speedup over CPU-S for iRangeGraph range-filtered search.

Reads results directly from two flat directories:
  results_cpu_p/{dataset}/results{range}.csv
  results_cpu_s/{dataset}/results{range}.csv

Each CSV has a header row: SearchEF,Recall,QPS,DCO,HOP[,RAM_MB]

Produces one figure per dataset written to:
  {base_dir}/analysis/speedup_cpu_p_vs_cpu_s[_env].png

Usage:
  python python/plots/plot_cpu_speedup.py
  python python/plots/plot_cpu_speedup.py --datasets audi,video
  python python/plots/plot_cpu_speedup.py --ranges 2,5,8 --env idun
  python python/plots/plot_cpu_speedup.py --copy-to-thesis
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent.parent.parent / "executable_data" / "cpu_p_vs_cpu_s"
THESIS_FIGS  = Path(__file__).resolve().parent.parent.parent / "thesis/Master/figs/results"

DATASETS = [
    {"key": "audi",  "name": "YT-Audio",  "dir": "audi"},
    {"key": "video", "name": "YT-Video",  "dir": "video"},
    {"key": "gist",  "name": "GIST-1M",   "dir": "gist"},
]

ALL_RANGES  = [2, 5, 8]
RANGE_LABEL = {2: "wide", 5: "medium", 8: "narrow"}
RANGE_MARKER = {2: "o", 5: "s", 8: "^"}


def _parse_range(filename):
    m = re.search(r"results(\d+)\.csv$", filename)
    if not m:
        return None
    val = int(m.group(1))
    return val if val in ALL_RANGES else None


HEADERLESS_COLS = ["SearchEF", "Recall", "QPS", "DCO", "HOP"]


def load_method(results_dir, active_ranges):
    """Return {range: DataFrame(SearchEF, QPS)} from a flat results directory."""
    if not results_dir.is_dir():
        return {}
    data = {}
    for csv in sorted(results_dir.glob("results*.csv")):
        rng = _parse_range(csv.name)
        if rng not in active_ranges:
            continue
        raw = pd.read_csv(csv, header=None, nrows=1)
        try:
            float(raw.iloc[0, 0])
            df = pd.read_csv(csv, header=None, names=HEADERLESS_COLS[:raw.shape[1]])
        except (ValueError, TypeError):
            df = pd.read_csv(csv)
        if "SearchEF" not in df.columns or "QPS" not in df.columns:
            continue
        data[rng] = df[["SearchEF", "QPS"]].sort_values("SearchEF")
    return data


def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _copy_to_thesis(src, dataset_key):
    import shutil
    dest = THESIS_FIGS / dataset_key / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied to thesis → {dest.relative_to(THESIS_FIGS.parent.parent.parent)}")


def plot_speedup(name, cpu_p, cpu_s, active_ranges, out_path, title, hw_label=""):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drew = False
    for rng in active_ranges:
        p = cpu_p.get(rng)
        s = cpu_s.get(rng)
        if p is None or s is None:
            continue
        merged = p.set_index("SearchEF")["QPS"].rename("p").to_frame().join(
            s.set_index("SearchEF")["QPS"].rename("s"), how="inner"
        )
        if merged.empty:
            continue
        speedup = (merged["p"] / merged["s"]).sort_index()
        ax.plot(speedup.index, speedup.values,
                marker=RANGE_MARKER[rng], ms=5, lw=1.8,
                color="tab:red", alpha=0.85,
                label=f"Range {rng} ({RANGE_LABEL[rng]})")
        drew = True
    if not drew:
        plt.close(fig)
        print(f"  skip {out_path.name}: no overlapping data")
        return
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("SearchEF")
    ax.set_ylabel("Speedup  (CPU-P QPS / CPU-S QPS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        t = f"{name} — {hw_label}: CPU-P speedup over CPU-S" if hw_label else f"{name}: CPU-P speedup over CPU-S"
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def main():
    p = argparse.ArgumentParser(description="Plot CPU-P speedup over CPU-S.")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated dataset keys to plot (audi, video, gist). Omit for all.")
    p.add_argument("--ranges", default="2,5,8",
                   help="Comma-separated range indices to plot. Default: 2,5,8.")
    p.add_argument("--env", default=None,
                   help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware", default="",
                   help="Hardware label for figure titles.")
    p.add_argument("--title", action="store_true",
                   help="Draw a title on each figure.")
    p.add_argument("--base-dir", default=None, type=Path,
                   help="Explicit path to the cpu_p_vs_cpu_s directory.")
    p.add_argument("--copy-to-thesis", action="store_true",
                   help="Copy output figures into thesis/Master/figs/results/{dataset}/.")
    args = p.parse_args()

    base_dir = args.base_dir or BASE_DIR

    try:
        active_ranges = [int(r.strip()) for r in args.ranges.split(",")]
    except ValueError:
        p.error("--ranges must be comma-separated integers, e.g. '2,8'")
    unknown = [r for r in active_ranges if r not in ALL_RANGES]
    if unknown:
        p.error(f"Unknown range(s): {unknown}. Valid: {ALL_RANGES}")

    datasets = DATASETS
    if args.datasets:
        keys = {k.strip() for k in args.datasets.split(",")}
        datasets = [d for d in DATASETS if d["key"] in keys]
        if not datasets:
            print(f"No datasets matched '{args.datasets}'.")
            return

    suffix = f"_{args.env}" if args.env else ""

    for d in datasets:
        print(f"\n{d['name']}")
        cpu_p = load_method(base_dir / "results_cpu_p" / d["dir"], active_ranges)
        cpu_s = load_method(base_dir / "results_cpu_s" / d["dir"], active_ranges)
        if not cpu_p and not cpu_s:
            print("  no data found — skipping")
            continue
        out_dir = base_dir / "analysis"
        out_path = out_dir / f"speedup_cpu_p_vs_cpu_s_{d['key']}{suffix}.png"
        plot_speedup(d["name"], cpu_p, cpu_s, active_ranges, out_path, args.title, args.hardware)
        if args.copy_to_thesis and out_path.exists():
            _copy_to_thesis(out_path, d["key"])


if __name__ == "__main__":
    main()
