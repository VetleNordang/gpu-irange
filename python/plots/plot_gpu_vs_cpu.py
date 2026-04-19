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

# Base directory (relative to script location)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "exectable_data"

DATASETS = [
    {"key": "gist250k", "name": "GIST 250k", "path": "gist1m/250k"},
    {"key": "gist500k", "name": "GIST 500k", "path": "gist1m/500k"},
    {"key": "gist750k", "name": "GIST 750k", "path": "gist1m/750k"},
    {"key": "gist1000k", "name": "GIST 1000k", "path": "gist1m/1000k"},
    {"key": "video", "name": "Video (YouTube RGB)", "path": "video"},
    {"key": "audi", "name": "Audi", "path": "audi"}
]

TARGET_SUFFIXES = ["2", "5", "8"]

def read_csv_files(result_dir):
    """Read all CSV files from a directory and filter for target suffixes."""
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
            
            # Extract suffix (the numbers right before .csv or _gpu.csv)
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

def get_series_by_suffix(df, suffix, col='QPS'):
    """Return series (grouped by SearchEF) for a specific suffix."""
    if df is None:
        return None
    suffix_df = df[df['Suffix'] == suffix]
    if suffix_df.empty:
        return None
    return suffix_df.groupby('SearchEF').agg({'Recall': 'mean', 'QPS': 'mean'}).sort_index()

def plot_methods_comparison(cpu_df, gpu_normal_df, gpu_pq_df, dataset_name, output_dir):
    """Create comprehensive plots comparing the 3 methods across suffixes 2, 5, 8."""
    
    # Setup Colors & Markers
    methods = {
        "CPU": {"df": cpu_df, "color": "tab:blue"},
        "GPU (Normal)": {"df": gpu_normal_df, "color": "tab:orange"},
        "GPU (PQ)": {"df": gpu_pq_df, "color": "tab:green"}
    }
    
    suffix_markers = {
        "2": "o",  # Circle
        "5": "s",  # Square
        "8": "^"   # Triangle
    }

    # Plot 1: Recall vs QPS Tradeoff
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
                
            ax.plot(
                series['Recall'], 
                series['QPS'], 
                marker=suffix_markers[suffix],
                linestyle='-',
                color=method_data["color"],
                label=f'{method_name} - Range {suffix}',
                linewidth=2,
                markersize=8,
                alpha=0.8
            )
            plotted_any = True
            
    if not plotted_any:
        print(f"  Warning: No data available to plot tradeoff for {dataset_name}")
        plt.close()
    else:
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_title(f'{dataset_name}: Recall vs QPS Tradeoff (Ranges 2, 5, 8)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_methods_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {output_dir / 'tradeoff_methods_comparison.png'}")

    
    # Plot 2: SearchEF vs QPS
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
                
            ax.plot(
                series.index, 
                series['QPS'], 
                marker=suffix_markers[suffix],
                linestyle='--',
                color=method_data["color"],
                label=f'{method_name} - Range {suffix}',
                linewidth=2,
                markersize=8,
                alpha=0.8
            )
            plotted_any = True
            
    if not plotted_any:
        print(f"  Warning: No data available to plot QPS for {dataset_name}")
        plt.close()
    else:
        ax.set_xlabel('SearchEF', fontsize=12)
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=12)
        ax.set_title(f'{dataset_name}: SearchEF vs QPS (Ranges 2, 5, 8)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / "qps_methods_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {output_dir / 'qps_methods_comparison.png'}")

def main():
    print("=" * 60)
    print("CPU vs GPU Normal vs GPU PQ Performance Comparison")
    print("=" * 60)
    print()
    
    success_count = 0
    failure_count = 0
    
    for dataset in DATASETS:
        print(f"Processing: {dataset['name']}")
        
        dataset_dir = BASE_DIR / dataset['path']
        output_dir = dataset_dir / "results" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read results
        cpu_df = read_csv_files(dataset_dir / "results" / "cpu")
        gpu_normal_df = read_csv_files(dataset_dir / "results" / "gpu_normal")
        gpu_pq_df = read_csv_files(dataset_dir / "results" / "gpu_pq")
        
        # Count available files
        def count_unique(df): return len(df['Suffix'].unique()) if df is not None else 0
        print(f"  Found data for queries: CPU={count_unique(cpu_df)}, GPU_Normal={count_unique(gpu_normal_df)}, GPU_PQ={count_unique(gpu_pq_df)}")
        
        if cpu_df is None and gpu_normal_df is None and gpu_pq_df is None:
            print("  ✗ No data found for any method for target ranges.")
            failure_count += 1
            print()
            continue

        try:
            plot_methods_comparison(cpu_df, gpu_normal_df, gpu_pq_df, dataset['name'], output_dir)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed to create plots: {e}")
            failure_count += 1
        
        print()
    
    print("=" * 60)
    print("Summary")
    print(f"Successfully plotted: {success_count} datasets")
    print(f"Failed: {failure_count} datasets")
    print("=" * 60)

if __name__ == "__main__":
    main()
