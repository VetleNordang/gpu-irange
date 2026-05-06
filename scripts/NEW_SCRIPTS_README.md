# New Experiment Scripts Guide

Created 4 simplified scripts to replace the old complex ones. Each script writes logs to `logs/` folder.

---

## Quick Start

```bash
cd /workspaces/irange/scripts

# 1. Setup folder structure
./setup_folders.sh

# 2. Create PQ compressed datasets
./make_pq.sh all

# 3. Build and run tests
./run_tests.sh

# 4. Execute experiments
./execute_experiments.sh all
```

---

## Individual Scripts

### 1. **setup_folders.sh** - Create Folder Structure
Creates organized result directories for all datasets.

**Usage:**
```bash
./setup_folders.sh
```

**What it does:**
- Creates `results/` folder structure under each dataset
- Creates subfolders: `cpu/`, `gpu_normal/`, `gpu_pq/`, `analysis/`
- For gist1m: `executable_data/gist1m/{250k,500k,750k,1000k}/results/`
- For audi and video: `executable_data/{audi,video}/results/`
- Adds `.gitkeep` files to preserve directory structure

**Log file:** `logs/setup_folders.log`

**Example output:**
```
✓ Created: gist1m/250k/results
  ├─ Created: cpu/
  ├─ Created: gpu_normal/
  ├─ Created: gpu_pq/
  ├─ Created: analysis/
```

---

### 2. **make_pq.sh** - Create PQ Compressed Datasets
Compresses datasets using Product Quantization (Faiss library).

**Usage:**
```bash
# Compress all datasets
./make_pq.sh all

# Compress only gist1m datasets
./make_pq.sh gist1m

# Compress specific gist1m size
./make_pq.sh gist1m 250k

# Compress only audi
./make_pq.sh audi

# Compress only video
./make_pq.sh video
```

**What it does:**
- Runs `pq_compress_only` executable from `build/tests/`
- Trains PQ model and encodes all vectors
- Outputs `.faiss` model files and `.bin` code files
- Saves to `dataset/pq/` folder

**PQ Parameters by Dataset:**
| Dataset | Dim | M | nbits | Centroids |
|---------|-----|---|-------|-----------|
| gist1m  | 960 | 320 | 9 | 163,840 |
| audi    | 128 | 32  | 9 | 16,384  |
| video   | 1024| 256 | 9 | 131,072 |

**Log file:** `logs/make_pq.log`

**Example output:**
```
✓ Processing: gist1m/250k
  Vector dimension: 960
  M (subspaces): 320
  nbits: 9 (2^9 = 512 centroids per subspace)
  Total centroids: 163,840
✓ Compression completed for gist1m/250k
  ✓ Model file: 124M
  ✓ Codes file: 89M
```

---

### 3. **run_tests.sh** - Build and Run Tests
Compiles the project and runs available tests.

**Usage:**
```bash
./run_tests.sh
```

**What it does:**
- Runs CMake configuration
- Compiles project with Release configuration
- Runs available test executables:
  - `buildindex` - Creates an index (always runs)
  - `pq_compress_only` - PQ compression (requires data files)
  - `pq_encode` - Encodes vectors with PQ
  - `search` - Search tests (requires index files)
  - `search_multi` - Multi-threaded search (requires index files)

**Log file:** `logs/run_tests.log`

**Example output:**
```
✓ CMake configuration completed
✓ Build completed successfully
✓ Running: buildindex
✓ buildindex test passed
```

---

### 4. **execute_experiments.sh** - Run Experiments
Executes complete experiment workflows.

**Usage:**
```bash
# Run all phases (CPU → GPU → GPU PQ)
./execute_experiments.sh all

# Run only CPU experiments
./execute_experiments.sh cpu

# Run only GPU normal experiments
./execute_experiments.sh gpu

# Run only GPU PQ experiments
./execute_experiments.sh pq

# Run specific datasets
./execute_experiments.sh all gist250k gist500k video

# Run CPU on specific datasets
./execute_experiments.sh cpu gist1000k audi
```

**Dataset names:**
- `gist250k`, `gist500k`, `gist750k`, `gist1000k` - gist1m datasets
- `audi` - Audio dataset
- `video` - Video dataset

**What it does:**

**Phase 1 (CPU):**
- Runs CPU-based search on each dataset
- Uses `build/tests/search` executable
- Saves results to `[dataset]/results/cpu/`

**Phase 2 (GPU):**
- Builds GPU version from `cude_version/`
- Runs GPU normal experiments with full embeddings
- Saves results to `[dataset]/results/gpu_normal/`

**Phase 3 (GPU PQ):**
- Builds GPU PQ version
- Runs GPU experiments with PQ compression
- Saves results to `[dataset]/results/gpu_pq/`

**Log file:** `logs/execute_experiments.log`

**Example output:**
```
Phase 1: CPU Experiments
  Processing dataset: gist250k
✓ CPU experiment completed for gist250k
  Processing dataset: gist500k
✓ CPU experiment completed for gist500k

Phase 2: GPU Normal Experiments
✓ GPU build completed
✓ GPU normal experiments completed

Results Summary:
CPU Results:
  gist250k: executable_data/gist1m/250k/results/cpu (5 files)
```

---

## Log Files

All logs are saved to `/workspaces/irange/logs/`:

- `setup_folders.log` - Folder creation logs
- `make_pq.log` - PQ compression logs
- `run_tests.log` - Build and test logs
- `execute_experiments.log` - Experiment execution logs

**View recent logs:**
```bash
tail -f /workspaces/irange/logs/execute_experiments.log
```

---

## Example Workflow

### Complete Workflow (CPU → GPU → GPU PQ)

```bash
cd /workspaces/irange/scripts

# Step 1: Setup folders
./setup_folders.sh

# Step 2: Create PQ datasets (required for PQ experiments)
./make_pq.sh all

# Step 3: Build and test
./run_tests.sh

# Step 4: Run full experiment suite
./execute_experiments.sh all

# Monitor progress
tail -f ../logs/execute_experiments.log
```

### PQ-Only Workflow

```bash
cd /workspaces/irange/scripts

# Setup
./setup_folders.sh

# Prepare PQ
./make_pq.sh gist1m 500k

# Run GPU PQ experiments only
./execute_experiments.sh pq gist500k
```

### CPU Testing Only

```bash
cd /workspaces/irange/scripts

# Setup
./setup_folders.sh

# Run tests
./run_tests.sh

# Run CPU experiments
./execute_experiments.sh cpu
```

---

## Features

✓ **Logging** - All output written to `logs/` folder
✓ **Error Handling** - Scripts exit on errors with clear messages
✓ **Colored Output** - Easy to read console output
✓ **Progress Tracking** - Detailed info messages for each step
✓ **Flexible** - Run all datasets or specific ones
✓ **Modular** - Each script is independent and can run separately

---

## Troubleshooting

**"pq_compress_only executable not found"**
- Run: `cd /workspaces/irange && cmake -B build && cmake --build build`

**"Dataset not found"**
- Verify dataset exists in `executable_data/`

**"Index file not found"**
- Need to build indexes first (run `buildindex` test)

**Build failed**
- Check logs: `tail /workspaces/irange/logs/run_tests.log`

---

## File Locations

```
/workspaces/irange/
├── scripts/
│   ├── setup_folders.sh          ← Setup folder structure
│   ├── make_pq.sh                ← Create PQ datasets
│   ├── run_tests.sh              ← Build and test
│   ├── execute_experiments.sh    ← Run experiments
│   └── README.md                 ← This file
├── logs/                         ← All log files
│   ├── setup_folders.log
│   ├── make_pq.log
│   ├── run_tests.log
│   └── execute_experiments.log
├── build/                        ← CMake build output
│   └── tests/
│       ├── search               ← CPU search executable
│       ├── pq_compress_only     ← PQ compression executable
│       └── buildindex           ← Index builder executable
├── cude_version/                ← GPU code
│   └── Makefile                ← GPU targets: make, make pq_target, make run, make run_pq
└── executable_data/
    ├── gist1m/
    │   ├── 250k/
    │   │   ├── gist_base_250k.bin
    │   │   ├── gist_query_250k.bin
    │   │   ├── gist_250k.index
    │   │   ├── pq/               ← PQ files
    │   │   └── results/{cpu,gpu_normal,gpu_pq,analysis}/
    │   └── ...
    ├── audi/
    │   ├── yt_aud_sorted_vec_by_attr.bin
    │   ├── pq/
    │   └── results/{cpu,gpu_normal,gpu_pq,analysis}/
    └── video/
        ├── youtube_rgb_sorted.bin
        ├── pq/
        └── results/{cpu,gpu_normal,gpu_pq,analysis}/
```

---
