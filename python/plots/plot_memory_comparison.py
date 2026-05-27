#!/usr/bin/env python3
"""
Plot VRAM usage comparison across GPU methods and dataset sizes.

Produces one figure per dataset family (audi, video, gist):

  memory_comparison[_env].png
      Grouped bar chart — base VRAM and peak VRAM per method per dataset size.
      Shows how PQ compression reduces VRAM footprint vs full-precision GPU Normal.

Output: executable_data/{family}/results/analysis/

Usage:
  python python/plots/plot_memory_comparison.py
  python python/plots/plot_memory_comparison.py --env idun
  python python/plots/plot_memory_comparison.py --family audi
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "executable_data"

FAMILIES = [
    {
        "key": "audi",
        "name": "YouTube Audio",
        "datasets": [
            {"label": "1M",  "path": "audi/1m"},
            {"label": "2M",  "path": "audi/2m"},
            {"label": "4M",  "path": "audi/4m"},
            {"label": "8M",  "path": "audi/8m"},
        ],
    },
    {
        "key": "video",
        "name": "YouTube Video",
        "datasets": [
            {"label": "1M",  "path": "video/1m"},
            {"label": "2M",  "path": "video/2m"},
            {"label": "4M",  "path": "video/4m"},
            {"label": "8M",  "path": "video/8m"},
        ],
    },
    {
        "key": "gist",
        "name": "GIST1M",
        "datasets": [
            {"label": "250k",  "path": "gist1m/250k"},
            {"label": "500k",  "path": "gist1m/500k"},
            {"label": "750k",  "path": "gist1m/750k"},
            {"label": "1M",    "path": "gist1m/1000k"},
        ],
    },
]

METHODS = [
    {"key": "gpu_normal", "label": "GPU Normal", "color": "tab:orange"},
    {"key": "gpu_pq",     "label": "GPU PQ",     "color": "tab:green"},
    {"key": "gpu_root",   "label": "GPU Root",   "color": "tab:purple"},
]


def _read_vram(method_dir: Path):
    """Return (vram_mb, peak_vram_mb) as the mean across all run CSVs, or None."""
    vrам_vals, peak_vals = [], []

    run_dirs = sorted(d for d in method_dir.glob("run*") if d.is_dir())
    sources = run_dirs if run_dirs else [method_dir]

    for src in sources:
        for csv in src.glob("*.csv"):
            try:
                df = pd.read_csv(csv, nrows=1)
            except Exception:
                continue
            if df.empty:
                continue
            if "VRAM_MB" in df.columns:
                v = pd.to_numeric(df["VRAM_MB"].iloc[0], errors="coerce")
                if not pd.isna(v):
                    vrам_vals.append(float(v))
            if "PeakVRAM_MB" in df.columns:
                p = pd.to_numeric(df["PeakVRAM_MB"].iloc[0], errors="coerce")
                if not pd.isna(p):
                    peak_vals.append(float(p))

    if not vrам_vals:
        return None
    return np.mean(vrам_vals), np.mean(peak_vals) if peak_vals else np.mean(vrам_vals)


def plot_family(family, out_path, hw_label=""):
    datasets = family["datasets"]
    n_datasets = len(datasets)
    n_methods = len(METHODS)

    # Collect data: rows = datasets, cols = methods
    vram_grid = np.full((n_datasets, n_methods), np.nan)
    peak_grid = np.full((n_datasets, n_methods), np.nan)
    any_data = False

    for i, ds in enumerate(datasets):
        results_dir = BASE_DIR / ds["path"] / "results"
        for j, m in enumerate(METHODS):
            method_dir = results_dir / m["key"]
            if not method_dir.is_dir():
                continue
            result = _read_vram(method_dir)
            if result is None:
                continue
            vram_grid[i, j] = result[0] / 1024  # convert MB → GB
            peak_grid[i, j] = result[1] / 1024
            any_data = True

    if not any_data:
        print(f"  no VRAM data found for {family['name']} — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(n_datasets)
    bar_width = 0.22
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for panel_idx, (ax, grid, ylabel, title_suffix) in enumerate(zip(
        axes,
        [vram_grid, peak_grid],
        ["VRAM (GB)", "VRAM (GB)"],
        ["Base VRAM", "Peak VRAM"],
    )):
        for j, m in enumerate(METHODS):
            vals = grid[:, j]
            mask = ~np.isnan(vals)
            if not mask.any():
                continue
            bars = ax.bar(
                x[mask] + offsets[j], vals[mask],
                width=bar_width, color=m["color"], label=m["label"],
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )
            for bar, v in zip(bars, vals[mask]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([ds["label"] for ds in datasets])
        ax.set_xlabel("Dataset size")
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    t = f"{family['name']}"
    if hw_label:
        t += f" — {hw_label}"
    t += ": GPU VRAM Usage"
    fig.suptitle(t, fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(BASE_DIR)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None, help="Suffix for output filenames (e.g. 'idun').")
    p.add_argument("--hardware", default="", help="Hardware label for figure titles.")
    p.add_argument("--family", default=None, help="Family key to plot (audi, video, gist). Omit for all.")
    args = p.parse_args()

    suffix = f"_{args.env}" if args.env else ""

    families = FAMILIES
    if args.family:
        families = [f for f in FAMILIES if f["key"] == args.family]
        if not families:
            print(f"No family matched '{args.family}'. Choose from: {[f['key'] for f in FAMILIES]}")
            return

    for family in families:
        print(f"\n{family['name']}")
        out_dir = BASE_DIR / family["key"] / "results" / "analysis"
        plot_family(family, out_dir / f"memory_comparison{suffix}.png", args.hardware)

    print("\ndone")


if __name__ == "__main__":
    main()
