#!/usr/bin/env python3
"""
Plot GPU PQ vs CPU Parallel performance comparison.
Reads GPU PQ results from PQ_based_memory/ and CPU-P results from standard results/.
Generates Recall-QPS tradeoff and SearchEF-QPS plots for gist1000k and audi1m.
"""

import os
import re
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "executable_data"
PQ_MEM_DIR = BASE_DIR / "PQ_based_memory"

DATASETS = [
    {
        "key":      "gist1000k",
        "name":     "GIST 1000k",
        "pq_dir":   PQ_MEM_DIR / "gist",
        "cpu_dir":  BASE_DIR / "gist1m/1000k/results/cpu_parallel",
        "pq_pat":   r"results(\d+)_gpu\.csv$",
        "cpu_pat":  r"results_1000k(\d+)\.csv$",
        "out_dir":  PQ_MEM_DIR / "gist",
    },
    {
        "key":      "audi1m",
        "name":     "Audi 1M",
        "pq_dir":   PQ_MEM_DIR / "audi",
        "cpu_dir":  BASE_DIR / "audi/1m/results/cpu_parallel",
        "pq_pat":   r"results(\d+)_gpu\.csv$",
        "cpu_pat":  r"results(\d+)\.csv$",
        "out_dir":  PQ_MEM_DIR / "audi",
    },
]

TARGET_SUFFIXES = ["2", "5", "8"]


def read_csvs(directory, pattern):
    if not directory.exists():
        return None

    aggregate_dir = directory / "aggregate"
    read_dir = aggregate_dir if aggregate_dir.is_dir() else directory
    is_aggregate = aggregate_dir.is_dir()

    dfs = []
    for csv_file in sorted(glob.glob(str(read_dir / "*.csv"))):
        match = re.search(pattern, os.path.basename(csv_file))
        if not match:
            continue
        suffix = match.group(1)
        if suffix not in TARGET_SUFFIXES:
            continue
        try:
            df = pd.read_csv(csv_file)
            if 'SearchEF' not in df.columns:
                df = pd.read_csv(csv_file, names=['SearchEF', 'Recall', 'QPS', 'DCO', 'HOP'])
            if is_aggregate:
                rename = {c: c.replace('_mean', '') for c in df.columns if c.endswith('_mean')}
                df = df.rename(columns=rename)
            df['Suffix'] = suffix
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {csv_file}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_series(df, suffix):
    if df is None:
        return None
    sub = df[df['Suffix'] == suffix]
    if sub.empty:
        return None
    return sub.groupby('SearchEF').agg({'Recall': 'mean', 'QPS': 'mean'}).sort_index()


def plot_dataset(dataset, pq_df, cpu_df):
    name = dataset['name']
    out_dir = dataset['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "CPU Parallel": {"df": cpu_df, "color": "tab:blue"},
        "GPU (PQ)":     {"df": pq_df,  "color": "tab:green"},
    }
    suffix_markers = {"2": "o", "5": "s", "8": "^"}

    # ── Recall vs QPS ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted_any = False
    for method_name, method_data in methods.items():
        for suffix in TARGET_SUFFIXES:
            series = get_series(method_data['df'], suffix)
            if series is None:
                continue
            ax.plot(series['Recall'], series['QPS'],
                    marker=suffix_markers[suffix], linestyle='-',
                    color=method_data['color'], label=f'{method_name} - Range {suffix}',
                    linewidth=2, markersize=8, alpha=0.8)
            plotted_any = True

    if not plotted_any:
        print(f"  Warning: no data for recall-QPS plot ({name})")
        plt.close()
    else:
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
        ax.set_title(f'{name}: Recall vs QPS Tradeoff (Ranges 2, 5, 8)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        out = out_dir / "tradeoff_gpu_pq_vs_cpu_par.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out}")

    # ── SearchEF vs QPS ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted_any = False
    for method_name, method_data in methods.items():
        for suffix in TARGET_SUFFIXES:
            series = get_series(method_data['df'], suffix)
            if series is None:
                continue
            ax.plot(series.index, series['QPS'],
                    marker=suffix_markers[suffix], linestyle='--',
                    color=method_data['color'], label=f'{method_name} - Range {suffix}',
                    linewidth=2, markersize=8, alpha=0.8)
            plotted_any = True

    if not plotted_any:
        print(f"  Warning: no data for SearchEF-QPS plot ({name})")
        plt.close()
    else:
        ax.set_xlabel('SearchEF', fontsize=12)
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
        ax.set_title(f'{name}: SearchEF vs QPS (Ranges 2, 5, 8)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        out = out_dir / "qps_gpu_pq_vs_cpu_par.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot GPU PQ vs CPU Parallel comparison")
    parser.add_argument("--dataset", default=None,
                        help="Dataset key to plot (gist1000k, audi1m). Omit for all.")
    args = parser.parse_args()

    datasets_to_plot = DATASETS
    if args.dataset:
        datasets_to_plot = [d for d in DATASETS if d['key'] == args.dataset]
        if not datasets_to_plot:
            print(f"Unknown dataset '{args.dataset}'. Valid: {[d['key'] for d in DATASETS]}")
            return

    print("=" * 60)
    print("GPU PQ vs CPU Parallel Performance Comparison")
    print("=" * 60)

    for dataset in datasets_to_plot:
        print(f"\nProcessing: {dataset['name']}")
        pq_df  = read_csvs(dataset['pq_dir'],  dataset['pq_pat'])
        cpu_df = read_csvs(dataset['cpu_dir'], dataset['cpu_pat'])

        print(f"  GPU PQ ranges found:  {sorted(pq_df['Suffix'].unique()) if pq_df is not None else 'none'}")
        print(f"  CPU-P ranges found:   {sorted(cpu_df['Suffix'].unique()) if cpu_df is not None else 'none'}")

        if pq_df is None and cpu_df is None:
            print("  ✗ No data found.")
            continue

        plot_dataset(dataset, pq_df, cpu_df)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
