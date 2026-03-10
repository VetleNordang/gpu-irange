# GPU vs CPU Performance Plotting

This directory contains scripts for generating performance comparison plots between GPU and CPU implementations of iRange search.

## Scripts

### plot_gpu_vs_cpu.py

Generates comparison plots for all datasets, saving results in each dataset's results folder.

**Usage:**
```bash
# From workspace root
python3 python/plots/plot_gpu_vs_cpu.py

# Or directly
cd python/plots
./plot_gpu_vs_cpu.py
```

**What it does:**
- Reads CPU and GPU result CSV files from each dataset
- Generates two plots per dataset:
  1. **qps_gpu_vs_cpu.png** - QPS comparison and speedup analysis
  2. **recall_vs_qps.png** - Recall vs QPS tradeoff curves
- Saves plots in the dataset's results folder

**Output locations:**
```
exectable_data/gist1m/250k/results/
├── qps_gpu_vs_cpu.png
└── recall_vs_qps.png

exectable_data/gist1m/500k/results/
├── qps_gpu_vs_cpu.png
└── recall_vs_qps.png

exectable_data/gist1m/750k/results/
├── qps_gpu_vs_cpu.png
└── recall_vs_qps.png

exectable_data/video/results/
├── qps_gpu_vs_cpu.png
└── recall_vs_qps.png

exectable_data/audi/results/
├── qps_gpu_vs_cpu.png
└── recall_vs_qps.png
```

**Requirements:**
- pandas
- matplotlib

Install with:
```bash
pip install pandas matplotlib
```

## Expected CSV Format

The script expects CSV files with the following columns:
- SearchEF
- Recall
- QPS
- DCO (Distance Computations)
- HOP

CPU files: `results*.csv` (e.g., `results0.csv`, `results1.csv`, ...)
GPU files: `results*_gpu*.csv` (e.g., `results0_gpu.csv`, `results_gpu0.csv`, ...)
