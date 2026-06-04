#!/usr/bin/env python3
"""
Plot QPS and speedup for the one-thread-per-query managed-memory GPU version
versus CPU-P on YT-Audio 1M.

Produces two figures written to {out_dir}:
  qps_otpq_vs_cpu[_env].png       QPS vs SearchEF, one panel per range
  speedup_otpq_vs_cpu[_env].png   GPU-v1 QPS / CPU-P QPS vs SearchEF

Usage:
  python python/plots/plot_otpq_vs_cpu.py
  python python/plots/plot_otpq_vs_cpu.py --env idun
  python python/plots/plot_otpq_vs_cpu.py --ranges 2,8
  python python/plots/plot_otpq_vs_cpu.py --copy-to-thesis
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GPU_DIR  = Path(__file__).resolve().parent.parent.parent / \
           "executable_data/one_thread_per_query_managed_mem/audi"
CPU_DIR  = Path(__file__).resolve().parent.parent.parent / \
           "executable_data/audi/1m/results/cpu_parallel"
OUT_DIR  = Path(__file__).resolve().parent.parent.parent / \
           "executable_data/one_thread_per_query_managed_mem/analysis"
THESIS_FIGS = Path(__file__).resolve().parent.parent.parent / \
              "thesis/Master/figs/results"

ALL_RANGES  = [2, 5, 8]
RANGE_LABEL = {2: "wide", 5: "medium", 8: "narrow"}
RANGE_MARKER= {2: "o", 5: "s", 8: "^"}

METHODS = [
    {"key": "cpu_p", "label": "CPU-P",   "color": "tab:blue"},
    {"key": "gpu_v1","label": "GPU-v1",  "color": "tab:orange"},
]


def _parse_range(filename):
    m = re.search(r"results(\d+)", filename)
    if not m:
        return None
    val = int(m.group(1))
    return val if val in ALL_RANGES else None


def load_gpu(gpu_dir, active_ranges):
    """Return {range: DataFrame(SearchEF, QPS)} from the GPU flat directory."""
    data = {}
    for csv in sorted(gpu_dir.glob("results*_GPU.csv")):
        rng = _parse_range(csv.name)
        if rng not in active_ranges:
            continue
        df = pd.read_csv(csv)
        if "SearchEF" not in df.columns or "QPS" not in df.columns:
            continue
        data[rng] = df[["SearchEF", "QPS"]].sort_values("SearchEF")
    return data


def load_cpu(cpu_dir, active_ranges):
    """Return {range: DataFrame(SearchEF, QPS)} from aggregate/ or raw CSVs."""
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
        # fallback: mean over run dirs
        frames = []
        for run in sorted(cpu_dir.glob("run*")):
            csv_r = run / f"results{rng}.csv"
            if not csv_r.exists():
                continue
            df = pd.read_csv(csv_r)
            col = next((c for c in ("QPS", "Recall@10") if c in df.columns), None)
            if "QPS" not in df.columns:
                continue
            frames.append(df[["SearchEF", "QPS"]])
        if not frames:
            continue
        combined = pd.concat(frames).groupby("SearchEF")["QPS"].mean().reset_index()
        data[rng] = combined.sort_values("SearchEF")
    return data


def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _copy_to_thesis(src):
    import shutil
    dest = THESIS_FIGS / "audi" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied to thesis → {dest.relative_to(THESIS_FIGS.parent.parent.parent)}")


RANGE_LINESTYLE = {2: "-", 5: "--", 8: ":"}


def plot_qps(cpu, gpu, active_ranges, out_path, title, hw_label=""):
    """QPS vs SearchEF, all ranges on one grid. Color = method, linestyle = range."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    handles, labels = [], []
    drew = False
    for m, series in [("cpu_p", cpu), ("gpu_v1", gpu)]:
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
        t = f"YT-Audio 1M — {hw_label}" if hw_label else "YT-Audio 1M"
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_speedup(cpu, gpu, active_ranges, out_path, title, hw_label=""):
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
    ax.set_ylabel("Speedup  (GPU-v1 QPS / CPU-P QPS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        t = f"YT-Audio 1M — {hw_label}: GPU-v1 speedup over CPU-P" if hw_label \
            else "YT-Audio 1M: GPU-v1 speedup over CPU-P"
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def main():
    p = argparse.ArgumentParser(
        description="Plot GPU-v1 (one-thread-per-query) vs CPU-P on YT-Audio 1M.")
    p.add_argument("--ranges", default="2,5,8",
                   help="Comma-separated range indices. Default: 2,5,8.")
    p.add_argument("--env", default=None,
                   help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware", default="",
                   help="Hardware label for figure titles.")
    p.add_argument("--title", action="store_true",
                   help="Draw a title on each figure.")
    p.add_argument("--copy-to-thesis", action="store_true",
                   help="Copy output figures into thesis/Master/figs/results/audi/.")
    args = p.parse_args()

    try:
        active_ranges = [int(r.strip()) for r in args.ranges.split(",")]
    except ValueError:
        p.error("--ranges must be comma-separated integers, e.g. '2,8'")
    unknown = [r for r in active_ranges if r not in ALL_RANGES]
    if unknown:
        p.error(f"Unknown range(s): {unknown}. Valid: {ALL_RANGES}")

    suffix = f"_{args.env}" if args.env else ""

    print("Loading GPU-v1 ...")
    gpu = load_gpu(GPU_DIR, active_ranges)
    print("Loading CPU-P ...")
    cpu = load_cpu(CPU_DIR, active_ranges)

    qps_out     = OUT_DIR / f"qps_otpq_vs_cpu{suffix}.png"
    speedup_out = OUT_DIR / f"speedup_otpq_vs_cpu{suffix}.png"

    plot_qps(cpu, gpu, active_ranges, qps_out, args.title, args.hardware)
    plot_speedup(cpu, gpu, active_ranges, speedup_out, args.title, args.hardware)

    if args.copy_to_thesis:
        for path in (qps_out, speedup_out):
            if path.exists():
                _copy_to_thesis(path)


if __name__ == "__main__":
    main()
