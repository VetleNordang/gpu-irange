#!/usr/bin/env python3
"""
Plot QPS and speedup for the explicit-device-memory GPU version (GPU-v2)
versus CPU-P on YT-Audio 1M.

Produces two figures written to executable_data/explicit_device_mem/analysis/:
  qps_explicit_vs_cpu[_env].png
  speedup_explicit_vs_cpu[_env].png

Usage:
  python python/plots/plot_explicit_mem_vs_cpu.py
  python python/plots/plot_explicit_mem_vs_cpu.py --dataset audi
  python python/plots/plot_explicit_mem_vs_cpu.py --ranges 2,8 --env idun
  python python/plots/plot_explicit_mem_vs_cpu.py --copy-to-thesis
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent.parent.parent / "executable_data"
GPU_BASE    = BASE_DIR / "explicit_device_mem"
THESIS_FIGS = Path(__file__).resolve().parent.parent.parent / "thesis/Master/figs/design/gpu_v2"

DATASETS = [
    {"key": "audi",  "name": "YT-Audio 1M",  "cpu_path": "audi/1m/results/cpu_parallel"},
    {"key": "video", "name": "YT-Video 1M",  "cpu_path": "video/1m/results/cpu_parallel"},
    {"key": "gist",  "name": "GIST-1M",       "cpu_path": "gist1m/1000k/results/cpu_parallel"},
]

ALL_RANGES      = [2, 5, 8]
RANGE_LABEL     = {2: "wide", 5: "medium", 8: "narrow"}
RANGE_MARKER    = {2: "o", 5: "s", 8: "^"}
RANGE_LINESTYLE = {2: "-", 5: "--", 8: ":"}

METHODS = [
    {"key": "cpu_p",  "label": "CPU-P",   "color": "tab:blue"},
    {"key": "gpu_v2", "label": "GPU-v2",  "color": "tab:orange"},
]


def _parse_range(filename):
    m = re.search(r"results(\d+)", filename)
    if not m:
        return None
    val = int(m.group(1))
    return val if val in ALL_RANGES else None


def load_gpu(gpu_dir, active_ranges):
    """Return {range: DataFrame(SearchEF, QPS)} from a flat GPU results directory."""
    data = {}
    for csv in sorted(gpu_dir.glob("results*_gpu.csv")):
        rng = _parse_range(csv.name)
        if rng not in active_ranges:
            continue
        df = pd.read_csv(csv)
        if "SearchEF" not in df.columns or "QPS" not in df.columns:
            continue
        data[rng] = df[["SearchEF", "QPS"]].sort_values("SearchEF")
    return data


def load_cpu(cpu_dir, active_ranges):
    """Return {range: DataFrame(SearchEF, QPS)} from aggregate/ or raw run dirs."""
    agg_dir = cpu_dir / "aggregate"
    data = {}
    for rng in active_ranges:
        csv = agg_dir / f"results{rng}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            col = next((c for c in ("QPS_mean", "QPS") if c in df.columns), None)
            if col is None:
                continue
            data[rng] = df[["SearchEF", col]].rename(columns={col: "QPS"}).sort_values("SearchEF")
            continue
        frames = []
        for run in sorted(cpu_dir.glob("run*")):
            csv_r = run / f"results{rng}.csv"
            if not csv_r.exists():
                continue
            df = pd.read_csv(csv_r)
            if "QPS" not in df.columns:
                continue
            frames.append(df[["SearchEF", "QPS"]])
        if not frames:
            continue
        data[rng] = pd.concat(frames).groupby("SearchEF")["QPS"].mean().reset_index().sort_values("SearchEF")
    return data


def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _copy_to_thesis(src, dataset_key):
    import shutil
    dest = THESIS_FIGS / f"{dataset_key}_{src.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied to thesis → {dest.relative_to(dest.parent.parent.parent.parent)}")


def plot_qps(cpu, gpu, active_ranges, out_path, title, hw_label, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    handles, labels = [], []
    drew = False
    for m, series in [("cpu_p", cpu), ("gpu_v2", gpu)]:
        meta = next(x for x in METHODS if x["key"] == m)
        for rng in active_ranges:
            df = series.get(rng)
            if df is None:
                continue
            lbl = f"{meta['label']} — Range {rng} ({RANGE_LABEL[rng]})"
            line, = ax.plot(df["SearchEF"], df["QPS"],
                            linestyle=RANGE_LINESTYLE[rng],
                            marker=RANGE_MARKER[rng], ms=4, lw=1.8,
                            color=meta["color"], label=lbl)
            handles.append(line)
            labels.append(lbl)
            drew = True
    if not drew:
        plt.close(fig)
        print(f"  skip {out_path.name}: no data")
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SearchEF")
    ax.set_ylabel("QPS (queries per second)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    if title:
        t = f"{dataset_name} — {hw_label}" if hw_label else dataset_name
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_speedup(cpu, gpu, active_ranges, out_path, title, hw_label, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drew = False
    for rng in active_ranges:
        c = cpu.get(rng)
        g = gpu.get(rng)
        if c is None or g is None:
            continue
        merged = c.set_index("SearchEF")["QPS"].rename("c").to_frame().join(
            g.set_index("SearchEF")["QPS"].rename("g"), how="inner"
        )
        if merged.empty:
            continue
        speedup = (merged["g"] / merged["c"]).sort_index()
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
    ax.set_ylabel("Speedup  (GPU-v2 QPS / CPU-P QPS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        t = f"{dataset_name} — {hw_label}: GPU-v2 speedup over CPU-P" if hw_label \
            else f"{dataset_name}: GPU-v2 speedup over CPU-P"
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def main():
    p = argparse.ArgumentParser(
        description="Plot GPU-v2 (explicit device memory) vs CPU-P.")
    p.add_argument("--dataset", default=None,
                   help="Dataset key to plot (audi, video, gist). Omit for all.")
    p.add_argument("--ranges", default="2,5,8",
                   help="Comma-separated range indices. Default: 2,5,8.")
    p.add_argument("--env", default=None,
                   help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware", default="",
                   help="Hardware label for figure titles.")
    p.add_argument("--title", action="store_true",
                   help="Draw a title on each figure.")
    p.add_argument("--copy-to-thesis", action="store_true",
                   help="Copy output figures into thesis/Master/figs/design/gpu_v2/.")
    args = p.parse_args()

    try:
        active_ranges = [int(r.strip()) for r in args.ranges.split(",")]
    except ValueError:
        p.error("--ranges must be comma-separated integers, e.g. '2,8'")
    unknown = [r for r in active_ranges if r not in ALL_RANGES]
    if unknown:
        p.error(f"Unknown range(s): {unknown}. Valid: {ALL_RANGES}")

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"] == args.dataset]
        if not datasets:
            p.error(f"Unknown dataset '{args.dataset}'. Valid: {[d['key'] for d in DATASETS]}")

    suffix = f"_{args.env}" if args.env else ""
    out_dir = GPU_BASE / "analysis"

    for d in datasets:
        print(f"\n{d['name']}")
        gpu = load_gpu(GPU_BASE / d["key"], active_ranges)
        cpu = load_cpu(BASE_DIR / d["cpu_path"], active_ranges)
        if not gpu and not cpu:
            print("  no data found — skipping")
            continue

        qps_out     = out_dir / f"qps_explicit_vs_cpu_{d['key']}{suffix}.png"
        speedup_out = out_dir / f"speedup_explicit_vs_cpu_{d['key']}{suffix}.png"

        plot_qps(cpu, gpu, active_ranges, qps_out, args.title, args.hardware, d["name"])
        plot_speedup(cpu, gpu, active_ranges, speedup_out, args.title, args.hardware, d["name"])

        if args.copy_to_thesis:
            for path in (qps_out, speedup_out):
                if path.exists():
                    _copy_to_thesis(path, d["key"])


if __name__ == "__main__":
    main()
