# GPU Results Organization - Fix Summary

## Problem
GPU search results were being saved to `results/results_gpu*.csv` (same folder as CPU results), mixing with CPU results and making comparison difficult.

## Solution
Organized results into separate subdirectories:
- **CPU results** → `results/cpu/`
- **GPU results** → `results/gpu/`
- **GPU PQ results** → `results/pq/`

## Files Modified

### 1. **scripts/run_all_searches_gpu.sh** ✓
Updated all 6 GPU search dataset configurations (8 total lines) to save in `results/gpu/` subdirectory:
- Video: `results/results_gpu` → `results/gpu/results_gpu`
- GIST 250k: `results/results_250k_gpu` → `results/gpu/results_250k`
- GIST 500k: `results/results_500k_gpu` → `results/gpu/results_500k`
- GIST 750k: `results/results_750k_gpu` → `results/gpu/results_750k`
- GIST 1000k: `results/results_1000k_gpu` → `results/gpu/results_1000k`
- Audi: `results/results_gpu` → `results/gpu/results_gpu`

### 2. **scripts/run_all_searches.sh** (CPU) ✓
Updated all 6 CPU search dataset configurations to save in `results/cpu/` subdirectory:
- GIST 250k: `results/results_250k` → `results/cpu/results_250k`
- GIST 500k: `results/results_500k` → `results/cpu/results_500k`
- GIST 750k: `results/results_750k` → `results/cpu/results_750k`
- GIST 1000k: `results/results_1000k` → `results/cpu/results_1000k`
- Video: `results/results` → `results/cpu/results`
- Audi: `results/results` → `results/cpu/results`

### 3. **cude_version/Makefile** (GPU PQ) ✓
Updated GPU PQ result path to save in `results/pq/` subdirectory:
- `results/results_pq` → `results/pq/results_pq`

### 4. **scripts/run_all_experiments.sh** (Orchestrator) ✓
- Added call to `setup_result_directories.sh` at start
- Updated results summary section to display correct `cpu/`, `gpu/`, `pq/` paths

## New Scripts Created

### 1. **scripts/setup_result_directories.sh** ✓
Creates organized directory structure for all datasets:
```bash
executable_data/[dataset]/results/
├── cpu/    (CPU search results)
├── gpu/    (GPU normal search results)
└── pq/     (GPU PQ search results)
```

### 2. **scripts/check_results_organization.sh** ✓
Verifies the result directory organization and shows which folders have results.

## How to Use

### Setup directories (run once):
```bash
cd /workspaces/irange
bash scripts/setup_result_directories.sh
```

### Run all experiments (CPU → GPU → GPU-PQ):
```bash
bash scripts/run_all_experiments.sh
```

### Run specific experiment mode:
```bash
bash scripts/run_all_experiments.sh cpu    # CPU only
bash scripts/run_all_experiments.sh gpu    # GPU only
bash scripts/run_all_experiments.sh pq     # GPU PQ only
bash scripts/run_all_experiments.sh all    # All modes (default)
```

### Check current result organization:
```bash
bash scripts/check_results_organization.sh
```

## Result Directory Structure
```
executable_data/
├── audi/results/
│   ├── cpu/      (CPU results)
│   ├── gpu/      (GPU results)
│   └── pq/       (GPU PQ results)
├── video/results/
│   ├── cpu/
│   ├── gpu/
│   └── pq/
└── gist1m/
    ├── 250k/results/ {cpu, gpu, pq}
    ├── 500k/results/ {cpu, gpu, pq}
    ├── 750k/results/ {cpu, gpu, pq}
    └── 1000k/results/ {cpu, gpu, pq}
```

## Benefits
✓ Clear separation between CPU, GPU, and GPU-PQ results
✓ Easy to compare results across modes
✓ Cleaner results directory structure
✓ Prevents result file overwriting
✓ Scalable for future experiment modes

## Next Steps
1. Run `bash setup_result_directories.sh` to ensure all directories exist
2. Run `bash run_all_experiments.sh` to execute all experiments
3. Results will be organized in their respective subdirectories
4. Compare CPU vs GPU vs GPU-PQ performance across all datasets
