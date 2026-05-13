#!/usr/bin/env python3
"""
Plot CPU vs GPU Normal vs GPU PQ performance comparison for iRange search.
Generates QPS comparison plots and Recall-QPS tradeoff plots for each dataset.
Only includes data ranges for mixed 2, 5, and 8.
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

DATASETS = [
    {"key": "gist250k",  "name": "GIST 250k",              "path": "gist1m/250k"},
    {"key": "gist500k",  "name": "GIST 500k",              "path": "gist1m/500k"},
    {"key": "gist750k",  "name": "GIST 750k",              "path": "gist1m/750k"},
    {"key": "gist1000k", "name": "GIST 1000k",             "path": "gist1m/1000k"},
    {"key": "video1m",  "name": "Video 1M (YouTube RGB)", "path": "video/1m"},
    {"key": "video2m",  "name": "Video 2M (YouTube RGB)", "path": "video/2m"},
    {"key": "video4m",  "name": "Video 4M (YouTube RGB)", "path": "video/4m"},
    {"key": "video8m",  "name": "Video 8M (YouTube RGB)", "path": "video/8m"},
    {"key": "audi1m",   "name": "Audi 1M",                "path": "audi/1m"},
    {"key": "audi2m",   "name": "Audi 2M",                "path": "audi/2m"},
    {"key": "audi4m",   "name": "Audi 4M",                "path": "audi/4m"},
    {"key": "audi8m",   "name": "Audi 8M",                "path": "audi/8m"},
]

TARGET_SUFFIXES = ["2", "5", "8"]

def read_csv_files(result_dir):
    if not result_dir.exists():
        return None

    csv_files = sorted(glob.glob(os.path.join(result_dir, "*.csv")))
    if not csv_files:
        return None

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'SearchEF' not in df.columns:
                df = pd.read_csv(csv_file, names=['SearchEF', 'Recall', 'QPS', 'DCO', 'HOP'])

            match = re.search(r'(\d+)(?:_gpu)?\.csv$', os.path.basename(csv_file))
            if match:
                suffix = match.group(1)
                if suffix in TARGET_SUFFIXES:
                    df['Suffix'] = suffix
                    dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to read {csv_file}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)

def get_series_by_suffix(df, suffix):
    if df is None:
        return None
    suffix_df = df[df['Suffix'] == suffix]
    if suffix_df.empty:
        return None
    return suffix_df.groupby('SearchEF').agg({'Recall': 'mean', 'QPS': 'mean'}).sort_index()

def plot_methods_comparison(cpu_df, gpu_normal_df, gpu_pq_df, dataset_name, output_dir, env_tag=None):
    tag_label = f" [{env_tag.upper()}]" if env_tag else ""
    tag_suffix = f"_{env_tag}" if env_tag else ""

    methods = {
        "CPU":         {"df": cpu_df,        "color": "tab:blue"},
        "GPU (Normal)":{"df": gpu_normal_df, "color": "tab:orange"},
        "GPU (PQ)":    {"df": gpu_pq_df,     "color": "tab:green"},
    }
    suffix_markers = {"2": "o", "5": "s", "8": "^"}

    # ── Plot 1: Recall vs QPS ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted_any = False
    for method_name, method_data in methods.items():
        df = method_data["df"]
        if df is None:
            continue
        for suffix in TARGET_SUFFIXES:
            series = get_series_by_suffix(df, suffix)
            if series is None:
                continue
            ax.plot(series['Recall'], series['QPS'],
                    marker=suffix_markers[suffix], linestyle='-',
                    color=method_data["color"], label=f'{method_name} - Range {suffix}',
                    linewidth=2, markersize=8, alpha=0.8)
            plotted_any = True

    if not plotted_any:
        print(f"  Warning: No data to plot tradeoff for {dataset_name}")
        plt.close()
    else:
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_title(f'{dataset_name}: Recall vs QPS Tradeoff (Ranges 2, 5, 8){tag_label}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        out = output_dir / f"tradeoff_methods_comparison{tag_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out}")

    # ── Plot 2: SearchEF vs QPS ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted_any = False
    for method_name, method_data in methods.items():
        df = method_data["df"]
        if df is None:
            continue
        for suffix in TARGET_SUFFIXES:
            series = get_series_by_suffix(df, suffix)
            if series is None:
                continue
            ax.plot(series.index, series['QPS'],
                    marker=suffix_markers[suffix], linestyle='--',
                    color=method_data["color"], label=f'{method_name} - Range {suffix}',
                    linewidth=2, markersize=8, alpha=0.8)
            plotted_any = True

    if not plotted_any:
        print(f"  Warning: No data to plot QPS for {dataset_name}")
        plt.close()
    else:
        ax.set_xlabel('SearchEF', fontsize=12)
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
        ax.set_title(f'{dataset_name}: SearchEF vs QPS (Ranges 2, 5, 8){tag_label}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        out = output_dir / f"qps_methods_comparison{tag_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out}")

    # ── Plot 3: GPU Normal speedup over CPU ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted_any = False
    if cpu_df is not None and gpu_normal_df is not None:
        for suffix in TARGET_SUFFIXES:
            cpu_series = get_series_by_suffix(cpu_df, suffix)
            gpu_series = get_series_by_suffix(gpu_normal_df, suffix)
            if cpu_series is not None and gpu_series is not None:
                common_efs = cpu_series.index.intersection(gpu_series.index)
                if not common_efs.empty:
                    speedup = gpu_series.loc[common_efs, 'QPS'] / cpu_series.loc[common_efs, 'QPS']
                    ax.plot(common_efs, speedup,
                            marker=suffix_markers[suffix], linestyle='-',
                            color="tab:red", label=f'Speedup - Range {suffix}',
                            linewidth=2, markersize=8, alpha=0.8)
                    plotted_any = True

    if not plotted_any:
        print(f"  Warning: No data to plot Speedup for {dataset_name}")
        plt.close()
    else:
        ax.set_xlabel('SearchEF', fontsize=12)
        ax.set_ylabel('Speedup (GPU Normal QPS / CPU QPS)', fontsize=12)
        ax.set_title(f'{dataset_name}: GPU Normal Speedup over CPU (Ranges 2, 5, 8){tag_label}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        plt.tight_layout()
        out = output_dir / f"speedup_comparison{tag_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out}")

def main():
    parser = argparse.ArgumentParser(description="Plot CPU vs GPU performance comparison")
    parser.add_argument("--dataset", default=None,
                        help="Dataset key prefix to plot (e.g. 'gist250k', 'video', 'audi_1m'). "
                             "Omit to plot all datasets.")
    parser.add_argument("--env", default=None,
                        help="Environment tag added to filenames and plot titles (e.g. 'idun').")
    args = parser.parse_args()

    print("=" * 60)
    print("CPU vs GPU Normal vs GPU PQ Performance Comparison")
    if args.env:
        print(f"Environment: {args.env.upper()}")
    print("=" * 60)
    print()

    if args.dataset:
        datasets_to_plot = [d for d in DATASETS if d['key'].startswith(args.dataset)]
        if not datasets_to_plot:
            print(f"Warning: no dataset matched prefix '{args.dataset}', plotting all.")
            datasets_to_plot = DATASETS
    else:
        datasets_to_plot = DATASETS

    success_count = 0
    failure_count = 0

    for dataset in datasets_to_plot:
        print(f"Processing: {dataset['name']}")

        dataset_dir = BASE_DIR / dataset['path']
        output_dir = dataset_dir / "results" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        cpu_df        = read_csv_files(dataset_dir / "results" / "cpu_serial")
        gpu_normal_df = read_csv_files(dataset_dir / "results" / "gpu_normal")
        gpu_pq_df     = read_csv_files(dataset_dir / "results" / "gpu_pq")

        def count_unique(df):
            return len(df['Suffix'].unique()) if df is not None else 0

        print(f"  Found ranges: CPU={count_unique(cpu_df)}, "
              f"GPU_Normal={count_unique(gpu_normal_df)}, GPU_PQ={count_unique(gpu_pq_df)}")

        if cpu_df is None and gpu_normal_df is None and gpu_pq_df is None:
            print("  ✗ No data found for any method.")
            failure_count += 1
            print()
            continue

        try:
            plot_methods_comparison(cpu_df, gpu_normal_df, gpu_pq_df,
                                    dataset['name'], output_dir, env_tag=args.env)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed to create plots: {e}")
            failure_count += 1

        print()

    print("=" * 60)
    print(f"Successfully plotted: {success_count} datasets")
    print(f"Failed / no data:     {failure_count} datasets")
    print("=" * 60)

if __name__ == "__main__":
    main()
