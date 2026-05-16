#!/usr/bin/env python3
"""
Generate per-dataset QPS, recall, and speedup plots for each GPU version vs CPU-P baseline.

For each GPU version × dataset produces:
  qps_vs_cpu_p.png     — QPS vs SearchEF (log-log), GPU vs CPU-P, ranges 2/5/8
  recall_vs_qps.png    — Recall@10 vs QPS (log), GPU vs CPU-P, ranges 2/5/8
  speedup_vs_cpu_p.png — Speedup (GPU/CPU-P) vs SearchEF, ranges 2/5/8

Output: executable_data/{version}/plots/{dataset}/

Usage:
  python python/plots/plot_gpu_versions.py
"""

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "executable_data"
CPU_P    = DATA_DIR / "cpu_p_vs_cpu_s" / "results_cpu_p"

GPU_VERSIONS = {
    "one_thread_per_query_managed_mem":  "GPU v1 (one thread per query)",
    "explicit_device_mem":               "GPU v2 (explicit device memory)",
    "parallel_neighbour_serial_dist":    "GPU v3 (parallel neighbour expansion)",
    "dist_split_4_threads":              "GPU v4 (distance split across threads)",
}

DATASETS = ["audi", "video", "gist"]
RANGES   = ["2", "5", "8"]
MARKERS  = {"2": "o", "5": "s", "8": "^"}


def load_folder(folder: Path) -> pd.DataFrame | None:
    csvs = glob.glob(str(folder / "*.csv"))
    frames = []
    for path in csvs:
        match = re.search(r"(\d+)(?:_[Gg][Pp][Uu])?\.csv$", path)
        if not match or match.group(1) not in RANGES:
            continue
        df = pd.read_csv(path)
        if "SearchEF" not in df.columns:
            df = pd.read_csv(path, names=["SearchEF", "Recall", "QPS", "DCO", "HOP"])
        df = df[["SearchEF", "Recall", "QPS"]].copy()
        df["Range"] = match.group(1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def plot_qps(gpu_df: pd.DataFrame, cpu_df: pd.DataFrame,
             version_label: str, dataset: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in RANGES:
        sub = cpu_df[cpu_df["Range"] == r].sort_values("SearchEF")
        if not sub.empty:
            ax.plot(sub["SearchEF"], sub["QPS"],
                    marker=MARKERS[r], linestyle="--", color="tab:gray",
                    label=f"CPU-P – Range {r}", linewidth=2, markersize=7, alpha=0.85)

    for r in RANGES:
        sub = gpu_df[gpu_df["Range"] == r].sort_values("SearchEF")
        if not sub.empty:
            ax.plot(sub["SearchEF"], sub["QPS"],
                    marker=MARKERS[r], linestyle="-", color="tab:orange",
                    label=f"GPU – Range {r}", linewidth=2, markersize=7, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SearchEF", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title(f"{dataset.upper()} 1M — {version_label}: QPS vs CPU-P",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "qps_vs_cpu_p.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out.relative_to(ROOT)}")


def plot_recall(gpu_df: pd.DataFrame, cpu_df: pd.DataFrame,
                version_label: str, dataset: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in RANGES:
        sub = cpu_df[cpu_df["Range"] == r].sort_values("Recall")
        if not sub.empty:
            ax.plot(sub["Recall"], sub["QPS"],
                    marker=MARKERS[r], linestyle="--", color="tab:gray",
                    label=f"CPU-P – Range {r}", linewidth=2, markersize=7, alpha=0.85)

    for r in RANGES:
        sub = gpu_df[gpu_df["Range"] == r].sort_values("Recall")
        if not sub.empty:
            ax.plot(sub["Recall"], sub["QPS"],
                    marker=MARKERS[r], linestyle="-", color="tab:orange",
                    label=f"GPU – Range {r}", linewidth=2, markersize=7, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title(f"{dataset.upper()} 1M — {version_label}: Recall vs QPS",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "recall_vs_qps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out.relative_to(ROOT)}")


def plot_speedup(gpu_df: pd.DataFrame, cpu_df: pd.DataFrame,
                 version_label: str, dataset: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    plotted = False

    for r in RANGES:
        g = gpu_df[gpu_df["Range"] == r].set_index("SearchEF")["QPS"]
        c = cpu_df[cpu_df["Range"] == r].set_index("SearchEF")["QPS"]
        common = g.index.intersection(c.index)
        if common.empty:
            continue
        speedup = g.loc[common] / c.loc[common]
        ax.plot(common, speedup, marker=MARKERS[r], linestyle="-",
                color="tab:red", label=f"Range {r}", linewidth=2, markersize=7, alpha=0.85)
        plotted = True

    if not plotted:
        plt.close()
        return

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="1× (break-even)")
    ax.set_xscale("log")
    ax.set_xlabel("SearchEF", fontsize=12)
    ax.set_ylabel("Speedup (GPU QPS / CPU-P QPS)", fontsize=12)
    ax.set_title(f"{dataset.upper()} 1M — {version_label}: Speedup over CPU-P",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "speedup_vs_cpu_p.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out.relative_to(ROOT)}")


def main():
    for folder, label in GPU_VERSIONS.items():
        print(f"\n{label}")
        for ds in DATASETS:
            gpu_df = load_folder(DATA_DIR / folder / ds)
            cpu_df = load_folder(CPU_P / ds)
            if gpu_df is None:
                print(f"  {ds}: no GPU data — skipping")
                continue
            if cpu_df is None:
                print(f"  {ds}: no CPU-P baseline — skipping")
                continue

            out_dir = DATA_DIR / folder / "plots" / ds
            out_dir.mkdir(parents=True, exist_ok=True)

            plot_qps(gpu_df, cpu_df, label, ds, out_dir)
            plot_recall(gpu_df, cpu_df, label, ds, out_dir)
            plot_speedup(gpu_df, cpu_df, label, ds, out_dir)

    print("\ndone")


if __name__ == "__main__":
    main()
