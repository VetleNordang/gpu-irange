# FAISS Product Quantization (PQ) Integration Guide

This document explains what was added to this repository for FAISS/PQ, how to use it, and what GPU integration paths you can choose next.

## What has been added

### 1) FAISS dependency support
- Installed in the dev container:
  - `libfaiss-dev`
  - `python3-faiss`

### 2) Optional FAISS build flag
- Added CMake option:
  - `ENABLE_FAISS` (default: `ON`)
- Location: root `CMakeLists.txt`

### 3) New PQ encoding tool
- New target: `pq_encode`
- Source: `tests/pq_encode.cpp`
- Build wiring: `tests/CMakeLists.txt`

This tool trains a FAISS `ProductQuantizer` on your existing iRange `.bin` vectors and outputs:
- a FAISS PQ model file
- PQ codes for base vectors
- optional PQ codes for query vectors

---

## Build

From project root:

```bash
mkdir -p build
cd build
cmake ..
make -j2 pq_encode
```

If FAISS is missing, CMake prints a warning and skips `pq_encode`.

---

## Input format expected

`pq_encode` reads the same vector binary format used in your project:
- first 4 bytes: `int32 n`
- next 4 bytes: `int32 d`
- then `n * d` float32 values

This is the format of files like:
- `executable_data/gist1m/250k/gist_base_250k.bin`
- `executable_data/gist1m/250k/gist_query_250k.bin`

---

## Run `pq_encode`

Example for GIST 250k:

```bash
/workspaces/irange/build/tests/pq_encode \
  --data_path /workspaces/irange/executable_data/gist1m/250k/gist_base_250k.bin \
  --query_path /workspaces/irange/executable_data/gist1m/250k/gist_query_250k.bin \
  --pq_model_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_pq.faiss \
  --pq_codes_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_codes.bin \
  --query_codes_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_query_250k_codes.bin \
  --M 64 \
  --nbits 8 \
  --train_size 20000
```

### Arguments
- `--data_path` (required): base vectors `.bin`
- `--pq_model_out` (required): FAISS PQ model output (`.faiss`)
- `--pq_codes_out` (required): encoded base PQ codes output (`.bin`)
- `--query_path` (optional): query vectors `.bin`
- `--query_codes_out` (required if `--query_path` is used)
- `--M` (optional, default `64`): number of sub-quantizers
- `--nbits` (optional, default `8`): bits per sub-quantizer code
- `--train_size` (optional, default `100000`): vectors sampled for training

### Constraints
- `d % M == 0` must hold.
- For GIST (`d=960`), valid examples include `M=32, 48, 60, 64`.

---

## Output format of PQ codes (`*_codes.bin`)

Each code file is written as:
1. `int32 n`
2. `int32 d`
3. `int32 M`
4. `int32 nbits`
5. `int32 code_size` (bytes per vector)
6. raw `uint8` codes of size `n * code_size`

---

## Current status vs GPU

### Already done
- PQ training + encoding pipeline is ready and validated.
- You can generate compact vector codes for each dataset.
- CPU ADC search over PQ codes is available via `pq_search_adc`.

### Not done yet
- Current GPU search kernels still read original float vectors.
- So memory reduction in GPU search is **not active yet** until kernels are updated to consume PQ codes.

---

## GPU options from here

Choose one of these integration paths:

### Option A — Keep current graph + add PQ decode/rerank on GPU (lowest code risk)
- Keep graph traversal exactly as today.
- Use PQ codes as compressed storage.
- Decode/rerank only top candidates in GPU.
- Pros: smallest architecture change.
- Cons: less memory/perf gain than full ADC distance over codes.

### Option B — Keep current graph + full PQ distance in GPU kernel (best memory reduction with your graph)
- Store PQ codebooks + node PQ codes on GPU.
- Replace float L2 with ADC distance from PQ codes inside search kernel.
- Pros: major vector-memory reduction while preserving your iRange graph logic.
- Cons: medium kernel refactor complexity.

### Option C — Switch to FAISS GPU index stack (fastest time-to-production)
- Use FAISS `GpuIndexIVFPQ` / PQ-native pipeline directly.
- Pros: minimal custom kernel maintenance.
- Cons: larger behavior change from your current graph/search implementation.

### Option D — Hybrid two-stage
- Stage 1: graph candidate generation (current pipeline).
- Stage 2: PQ compressed scoring/rerank for candidates.
- Pros: balanced migration path.
- Cons: still requires integration work between two representations.

---

## Recommended next step

If your priority is reducing memory footprint while keeping your current iRange graph behavior, start with **Option B**.

Practical sequence:
1. Load PQ model (`.faiss`) at startup.
2. Load PQ code matrix (`*_codes.bin`) to GPU.
3. Add a GPU ADC distance function (codebook lookup based).
4. Replace/guard current float distance path with PQ path via compile flag.
5. Compare recall/QPS/memory against current baseline.

---

## Quick sanity checklist

- `pq_encode` builds: `build/tests/pq_encode` exists.
- `pq_encode` run creates files under dataset `/pq/` folder.
- `M` divides vector dim.
- Training size is sufficient (typically `>= 10k`, preferably `20k-100k`).
- Keep a float baseline to measure recall tradeoff.

---

## CPU-first: make it work now

You asked to make PQ work on CPU first. This repo now has that path.

### Build

```bash
cd /workspaces/irange/build
cmake ..
make -j2 pq_encode pq_search_adc
```

### Step 1: Train PQ + encode vectors

```bash
/workspaces/irange/build/tests/pq_encode \
  --data_path /workspaces/irange/executable_data/gist1m/250k/gist_base_250k.bin \
  --query_path /workspaces/irange/executable_data/gist1m/250k/gist_query_250k.bin \
  --pq_model_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_pq.faiss \
  --pq_codes_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_codes.bin \
  --query_codes_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_query_250k_codes.bin \
  --M 64 --nbits 8 --train_size 20000
```

### Step 2: Run CPU ADC search on PQ codes

```bash
/workspaces/irange/build/tests/pq_search_adc \
  --pq_model /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_pq.faiss \
  --pq_codes /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_codes.bin \
  --query_path /workspaces/irange/executable_data/gist1m/250k/gist_query_250k.bin \
  --out_csv /workspaces/irange/executable_data/gist1m/250k/pq/adc_top10_q100.csv \
  --topk 10 --max_queries 100
```

This writes CSV rows:
- `query_id`
- `rank`
- `id`
- `adc_dist`

---

## What is stored as PQ right now?

There are two things:

1) **PQ model** (`.faiss`)
- Contains codebooks (centroids) learned by FAISS for each sub-quantizer.

2) **PQ codes** (`*_codes.bin`)
- Header:
  - `int32 n`
  - `int32 d`
  - `int32 M`
  - `int32 nbits`
  - `int32 code_size`
- Payload:
  - `n * code_size` bytes of compressed codes (`uint8` stream)

No full float vectors are needed for ADC search itself once these files are created.

---

## How distance is computed (ADC) in CPU tool

For each query vector `q`:

1. Split `q` into `M` subvectors.
2. For each subvector `m`, compute distances to all `ksub = 2^nbits` centroids.
   - This creates a distance table of size `M x ksub`.
3. For each database code:
   - Read centroid id for each subvector.
   - Sum table lookups across `m=0..M-1`.

Formula:

$$
\hat{d}(q, c) = \sum_{m=1}^{M} \| q_m - c_m[\text{code}_m] \|^2
$$

where `code_m` is the centroid index stored in the PQ code for subvector `m`.

This is exactly why memory drops and distance remains fast: distance becomes table lookups + adds.
