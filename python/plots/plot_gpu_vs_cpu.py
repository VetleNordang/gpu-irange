#!/usr/bin/env python3
"""
Plot GPU vs CPU performance comparison for iRange search.
Generates QPS comparison plots for each dataset and saves them in the respective results folders.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Base directory (relative to script location)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "exectable_data"

# Dataset configurations
DATASETS = [
    {
        "name": "GIST 250k",
        "path": "gist1m/250k",
        "cpu_pattern": "results_250k",
        "gpu_pattern": "results_250k_gpu"
    },
    {
        "name": "GIST 500k",
        "path": "gist1m/500k",
        "cpu_pattern": "results_500k",
        "gpu_pattern": "results_500k_gpu"
    },
    {
        "name": "GIST 750k",
        "path": "gist1m/750k",
        "cpu_pattern": "results_750k",
        "gpu_pattern": "results_750k_gpu"
    },
    {
        "name": "Video (YouTube RGB)",
        "path": "video",
        "cpu_pattern": "results",
        "gpu_pattern": "results_gpu"
    },
    {
        "name": "Audi",
        "path": "audi",
        "cpu_pattern": "results",
        "gpu_pattern": "results_gpu"
    }
]


def read_csv_files(result_dir, pattern, is_gpu=False):
    """Read all CSV files matching pattern and combine them."""
    csv_files = sorted(glob.glob(os.path.join(result_dir, f"{pattern}*.csv")))
    
    # Filter based on whether we want GPU or CPU files
    if is_gpu:
        # GPU files must contain "_gpu" in the filename
        csv_files = [f for f in csv_files if "_gpu" in os.path.basename(f)]
    else:
        # CPU files must NOT contain "_gpu" in the filename
        csv_files = [f for f in csv_files if "_gpu" not in os.path.basename(f)]
    
    if not csv_files:
        return None
    
    # Debug: show which files are being read
    file_type = "GPU" if is_gpu else "CPU"
    # print(f"  Reading {len(csv_files)} {file_type} files: {[os.path.basename(f) for f in csv_files[:3]]}...")
    
    # Read all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            # Try reading the CSV
            df = pd.read_csv(csv_file)
            
            # Check if the first column is 'SearchEF' (header exists)
            # If not, the file has no header, so read again with column names
            if 'SearchEF' not in df.columns:
                # File has no header, read with explicit column names
                df = pd.read_csv(csv_file, names=['SearchEF', 'Recall', 'QPS', 'DCO', 'HOP'])
            
            # Extract suffix from filename (e.g., results0.csv -> 0)
            suffix = os.path.basename(csv_file).replace(pattern, "").replace(".csv", "").replace("_gpu", "").replace("_", "")
            df['Suffix'] = suffix
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to read {csv_file}: {e}")
    
    if not dfs:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def plot_qps_comparison(cpu_df, gpu_df, dataset_name, output_path):
    """Create QPS comparison plot for GPU vs CPU."""
    
    # Group by SearchEF and calculate mean QPS across all suffixes
    cpu_grouped = cpu_df.groupby('SearchEF')['QPS'].mean().sort_index()
    gpu_grouped = gpu_df.groupby('SearchEF')['QPS'].mean().sort_index()
    
    # Also calculate recall for reference
    cpu_recall = cpu_df.groupby('SearchEF')['Recall'].mean().sort_index()
    gpu_recall = gpu_df.groupby('SearchEF')['Recall'].mean().sort_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: QPS comparison
    ax1.plot(cpu_grouped.index, cpu_grouped.values, 'o-', label='CPU', linewidth=2, markersize=6)
    ax1.plot(gpu_grouped.index, gpu_grouped.values, 's-', label='GPU', linewidth=2, markersize=6)
    ax1.set_xlabel('SearchEF', fontsize=12)
    ax1.set_ylabel('QPS (Queries Per Second)', fontsize=12)
    ax1.set_title(f'{dataset_name}: QPS Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Speedup (GPU/CPU)
    common_ef = sorted(set(cpu_grouped.index) & set(gpu_grouped.index))
    speedup = [gpu_grouped[ef] / cpu_grouped[ef] for ef in common_ef]
    
    ax2.plot(common_ef, speedup, 'o-', color='green', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='No speedup')
    ax2.set_xlabel('SearchEF', fontsize=12)
    ax2.set_ylabel('Speedup (GPU QPS / CPU QPS)', fontsize=12)
    ax2.set_title(f'{dataset_name}: GPU Speedup', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {output_path}")
    
    # Print summary statistics
    avg_speedup = sum(speedup) / len(speedup)
    max_speedup = max(speedup)
    min_speedup = min(speedup)
    print(f"    Average speedup: {avg_speedup:.2f}x")
    print(f"    Max speedup: {max_speedup:.2f}x (at EF={common_ef[speedup.index(max_speedup)]})")
    print(f"    Min speedup: {min_speedup:.2f}x (at EF={common_ef[speedup.index(min_speedup)]})")


def plot_recall_qps_tradeoff(cpu_df, gpu_df, dataset_name, output_path):
    """Create Recall vs QPS tradeoff plot."""
    
    # Group by SearchEF
    cpu_grouped = cpu_df.groupby('SearchEF').agg({'Recall': 'mean', 'QPS': 'mean'}).sort_index()
    gpu_grouped = gpu_df.groupby('SearchEF').agg({'Recall': 'mean', 'QPS': 'mean'}).sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Recall vs QPS
    ax.plot(cpu_grouped['QPS'], cpu_grouped['Recall'], 'o-', label='CPU', linewidth=2, markersize=8)
    ax.plot(gpu_grouped['QPS'], gpu_grouped['Recall'], 's-', label='GPU', linewidth=2, markersize=8)
    
    # Add SearchEF labels for some points
    for ef in [10, 50, 100, 500, 1000]:
        if ef in cpu_grouped.index:
            ax.annotate(f'EF={ef}', 
                       xy=(cpu_grouped.loc[ef, 'QPS'], cpu_grouped.loc[ef, 'Recall']),
                       xytext=(10, 10), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(f'{dataset_name}: Recall vs QPS Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {output_path}")


def main():
    print("=" * 60)
    print("GPU vs CPU Performance Comparison")
    print("=" * 60)
    print()
    
    success_count = 0
    failure_count = 0
    
    for dataset in DATASETS:
        print(f"Processing: {dataset['name']}")
        
        # Paths
        dataset_dir = BASE_DIR / dataset['path']
        results_dir = dataset_dir / "results"
        
        if not results_dir.exists():
            print(f"  ✗ Results directory not found: {results_dir}")
            failure_count += 1
            print()
            continue
        
        # Read CPU results
        cpu_df = read_csv_files(results_dir, dataset['cpu_pattern'], is_gpu=False)
        if cpu_df is None:
            print(f"  ✗ No CPU results found with pattern: {dataset['cpu_pattern']}")
            failure_count += 1
            print()
            continue
        
        # Read GPU results
        gpu_df = read_csv_files(results_dir, dataset['gpu_pattern'], is_gpu=True)
        if gpu_df is None:
            print(f"  ✗ No GPU results found with pattern: {dataset['gpu_pattern']}")
            failure_count += 1
            print()
            continue
        
        print(f"  Found {len(cpu_df['Suffix'].unique())} CPU result files")
        print(f"  Found {len(gpu_df['Suffix'].unique())} GPU result files")
        
        # Create plots
        try:
            # QPS comparison plot
            qps_plot_path = results_dir / "qps_gpu_vs_cpu.png"
            plot_qps_comparison(cpu_df, gpu_df, dataset['name'], qps_plot_path)
            
            # Recall vs QPS tradeoff plot
            tradeoff_plot_path = results_dir / "recall_vs_qps.png"
            plot_recall_qps_tradeoff(cpu_df, gpu_df, dataset['name'], tradeoff_plot_path)
            
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed to create plots: {e}")
            import traceback
            traceback.print_exc()
            failure_count += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully plotted: {success_count} datasets")
    print(f"Failed: {failure_count} datasets")
    print()
    print("Plots saved in each dataset's results folder:")
    print("  - qps_gpu_vs_cpu.png")
    print("  - recall_vs_qps.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
