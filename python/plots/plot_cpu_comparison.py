#!/usr/bin/env python3
"""
Plot CPU-Serial vs CPU-Parallel performance comparison.

Reads from:
  executable_data/cpu_p_vs_cpu_s/results_cpu_s/{dataset}/
  executable_data/cpu_p_vs_cpu_s/results_cpu_p/{dataset}/

Saves plots to:
  executable_data/cpu_p_vs_cpu_s/plots/{dataset}/

Usage:
  python python/plots/plot_cpu_comparison.py
"""

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = ROOT / "executable_data" / "cpu_p_vs_cpu_s"
PLOT_DIR   = RESULT_DIR / "plots"

DATASETS = ["audi", "video", "gist"]
RANGES   = ["2", "5", "8"]
MARKERS  = {"2": "o", "5": "s", "8": "^"}


COLS = ["SearchEF", "Recall", "QPS", "DCO", "HOP"]

def load_dataset(folder: Path) -> pd.DataFrame | None:
    csvs = sorted(glob.glob(str(folder / "*.csv")))
    frames = []
    for path in csvs:
        match = re.search(r"(\d+)\.csv$", path)
        if not match or match.group(1) not in RANGES:
            continue
        raw = pd.read_csv(path, header=None, nrows=1)
        has_header = isinstance(raw.iloc[0, 0], str)
        df = pd.read_csv(path) if has_header else pd.read_csv(path, names=COLS)
        df = df[COLS]
        df["Range"] = match.group(1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def plot_dataset(dataset: str) -> None:
    serial_df   = load_dataset(RESULT_DIR / "results_cpu_s" / dataset)
    parallel_df = load_dataset(RESULT_DIR / "results_cpu_p" / dataset)

    if serial_df is None and parallel_df is None:
        print(f"  {dataset}: no data found — skipping")
        return

    out_dir = PLOT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "CPU Serial":   {"df": serial_df,   "color": "tab:gray"},
        "CPU Parallel": {"df": parallel_df, "color": "tab:blue"},
    }

    # ── QPS vs SearchEF ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for method, meta in methods.items():
        if meta["df"] is None:
            continue
        for r in RANGES:
            sub = meta["df"][meta["df"]["Range"] == r].sort_values("SearchEF")
            if sub.empty:
                continue
            ax.plot(sub["SearchEF"], sub["QPS"],
                    marker=MARKERS[r], linestyle="--", color=meta["color"],
                    label=f"{method} – Range {r}", linewidth=2, markersize=7, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SearchEF", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title(f"{dataset.upper()} 1M — CPU Serial vs CPU Parallel: QPS", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "qps_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out.relative_to(ROOT)}")

    # ── Recall vs QPS tradeoff ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for method, meta in methods.items():
        if meta["df"] is None:
            continue
        for r in RANGES:
            sub = meta["df"][meta["df"]["Range"] == r].sort_values("Recall")
            if sub.empty:
                continue
            ax.plot(sub["Recall"], sub["QPS"],
                    marker=MARKERS[r], linestyle="-", color=meta["color"],
                    label=f"{method} – Range {r}", linewidth=2, markersize=7, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title(f"{dataset.upper()} 1M — CPU Serial vs CPU Parallel: Recall–QPS", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "recall_qps_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out.relative_to(ROOT)}")

    # ── Speedup (CPU-P / CPU-S) ──────────────────────────────────────────────
    if serial_df is not None and parallel_df is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        plotted = False
        for r in RANGES:
            s = serial_df[serial_df["Range"] == r].set_index("SearchEF")["QPS"]
            p = parallel_df[parallel_df["Range"] == r].set_index("SearchEF")["QPS"]
            common = s.index.intersection(p.index)
            if common.empty:
                continue
            speedup = p.loc[common] / s.loc[common]
            ax.plot(common, speedup, marker=MARKERS[r], linestyle="-",
                    color="tab:red", label=f"Range {r}", linewidth=2, markersize=7, alpha=0.85)
            plotted = True

        if plotted:
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="1× (no gain)")
            ax.set_xscale("log")
            ax.set_xlabel("SearchEF", fontsize=12)
            ax.set_ylabel("Speedup (CPU-P / CPU-S)", fontsize=12)
            ax.set_title(f"{dataset.upper()} 1M — CPU Parallel Speedup over CPU Serial", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = out_dir / "speedup_comparison.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  saved: {out.relative_to(ROOT)}")
        else:
            plt.close()


def main():
    print("CPU Serial vs CPU Parallel — plotting\n")
    for ds in DATASETS:
        print(f"{ds}:")
        plot_dataset(ds)
        print()
    print("done")


if __name__ == "__main__":
    main()
