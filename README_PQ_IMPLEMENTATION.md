# Product Quantization (PQ) Implementation Guide

## Overview

This document explains the Product Quantization (PQ) compression implementation integrated into the iRange search system. PQ dramatically reduces memory usage while maintaining fast approximate nearest neighbor search accuracy.

### What is Product Quantization?

Product Quantization is a vector compression technique that:
1. **Divides** each vector into M subspaces
2. **Learns** k-means centroids for each subspace using FAISS
3. **Assigns** each vector segment to its nearest centroid
4. **Stores** only centroid indices (8 bits per subspace = compact representation)

**Example with GIST 250k:**
- Original vector: 960 dimensions × 4 bytes = 3,840 bytes per vector
- Compressed vector: 64 subspaces × 1 byte = 64 bytes per vector
- **60x compression ratio** with high recall

---

## Changes Made to Codebase

### 1. **[include/iRG_search.h]** - Core Search Engine

#### Constructor Signature (Line 63)
```cpp
iRangeGraph_Search(std::string vectorfilename, std::string edgefilename, 
                    DataLoader *store, int M)
```
- **vectorfilename**: Path to raw vector data (binary format: int n, int d, then n×d floats)
- **edgefilename**: Path to HNSW graph structure (pre-built index)
- **store**: DataLoader object holding queries, ranges, and ground truth
- **M**: Number of subspaces for PQ decomposition

#### Key Member Variables (Lines 60-61)
```cp
uint8_t* data_code{nullptr};        // Compressed codes: M bytes per vector
faiss::ProductQuantizer* pq_model_{nullptr};  // FAISS codebook with centroids
```
- `data_code`: Array storing centroid indices for all DB vectors
- `pq_model_`: FAISS ProductQuantizer object containing learned centroids for each subspace

#### Distance Computation (Lines 124-150)
```cpp
float compute_pq_distance(const float* query_vector, int data_id) const
```
**How it works:**
1. For each subspace m (0 to M-1):
   - Extract centroid index from `data_code[data_id * M + m]` (1 byte per subspace)
   - Get query subvector: `query_m = query_vector + m * dsub`
   - Get centroid subvector from FAISS: `centroid_m = pq_model_->get_centroids(m, 0) + idx * dsub`
   - Compute L2 distance: sum of squared differences
   - Add to total distance
2. Return aggregated distance across all M subspaces

**Why this is fast:**
- No full vector comparisons
- Per-subspace L2 distance computation
- Centroid lookups use pre-trained FAISS model
- Memory efficient: only 1 byte lookup per subspace

#### Search Integration (Lines 155-167, 262-264)
The TopDown_nodeentries_search function:
1. Casts raw query data to float*
2. Passes query_vector to compute_pq_distance
3. Ranks neighbors by PQ distance instead of full L2 distance

---

### 2. **[tests/pq_encode.cpp]** - Compression & Search Tool

#### CompressedIndexData Structure (Lines 17-37)
```cpp
struct CompressedIndexData {
    std::shared_ptr<faiss::ProductQuantizer> pq_model;  // Trained codebook
    std::vector<uint8_t> data_codes_blob;                // Compressed vectors
    int M{0};                                             // Subspace count
};
```

#### Workflow

**Step 1: Train PQ Model** (Lines 201-215)
```cpp
// Load raw vectors from disk
std::vector<float> raw_vectors = LoadRawVectors(data_path, n, d);

// Create and train FAISS ProductQuantizer
auto pq = std::make_shared<faiss::ProductQuantizer>(d, M, 8);
pq->train(n, raw_vectors.data());
```
- **Input**: n vectors of dimension d
- **Process**: FAISS learns k-means centroids for each of M subspaces
- **Output**: pq_model with M×256 centroid vectors (8 bits = 256 possible centroids per subspace)

**Step 2: Encode Vectors** (Lines 216-230)
```cpp
std::vector<uint8_t> codes(n * M);
pq->compute_codes(raw_vectors.data(), codes.data(), n);
```
- **Input**: n raw vectors (n × d floats)
- **Process**: Assigns each vector segment to nearest centroid, stores centroid index
- **Output**: Compact codes (n × M bytes)

**Step 3: Save Compressed Data** (Lines 232-240)
```cpp
// Save FAISS model to file
WriteIndexToFile(pq_model_path, pq);

// Save compressed codes to file
SaveVectorBinary(codes_path, codes);
```
- Codebook and codes persisted for efficient re-runs

**Step 4: Search** (Lines 155-180)
```cpp
// Load compressed codes and codebook
InitializeIndexWithCompressedData(index, compressed_data);

// Search with raw queries against compressed DB
index.TopDown_nodeentries_search(query_vector, k, result);
```
- Distance computed on-the-fly using PQ model
- No precomputed distance tables needed

---

## Parameter Explanations

### Command Line Parameters

```bash
./pq_encode \
    --data_path_comp <PATH>                    # Raw vector file (GIST binary)
    --query_path <PATH>                        # Query vectors (GIST binary)
    --range_saveprefix <PREFIX>                # Query range file prefix
    --groundtruth_saveprefix <PREFIX>          # Ground truth file prefix
    --index_file <PATH>                        # HNSW graph structure
    --result_saveprefix <PREFIX>               # Output results file prefix
    --pq_model_out <PATH>                      # FAISS codebook output file
    --pq_codes_out <PATH>                      # Compressed codes output file
    --M <INT>                                  # Number of subspaces
```

#### Detailed Parameter Descriptions

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--data_path_comp` | Path | Input vector file in GIST binary format (3,840 bytes per 960-dim vector) | `gist_base_250k.bin` |
| `--query_path` | Path | Query vector file same format as data | `gist_query_250k.bin` |
| `--range_saveprefix` | Prefix | Prefix for query range files (generated if not exist) | `query_ranges_250k` → generates `query_ranges_250k_0`, `query_ranges_250k_1`, etc. |
| `--groundtruth_saveprefix` | Prefix | Prefix for ground truth files (k-nearest neighbors pre-computed) | `groundtruth_250k` → generates `groundtruth_250k.bin` |
| `--index_file` | Path | HNSW graph structure (pre-built with hnsw or buildindex) | `gist_250k.index` |
| `--result_saveprefix` | Prefix | Where to save search results (recall, latency stats) | `results_250k_pq` → generates `results_250k_pq0.csv`, `results_250k_pq1.csv`, etc. |
| `--pq_model_out` | Path | Output file for trained FAISS codebook (persisted for re-use) | `gist_250k_pq.faiss` |
| `--pq_codes_out` | Path | Output file for compressed vector codes | `gist_250k_codes.bin` |
| `--M` | Integer | **Number of subspaces** - controls compression tradeoff | `32` or `64` |

#### Understanding M (Subspaces)

`M` is the **most critical parameter** controlling the compression/accuracy tradeoff:

**When M = 32:**
- Vector divided into 32 subspaces
- Each subspace = 960 / 32 = 30 dimensions
- Compressed size = 32 bytes per vector (120x compression)
- **Faster computation, lower accuracy**

**When M = 64:**
- Vector divided into 64 subspaces
- Each subspace = 960 / 64 = 15 dimensions
- Compressed size = 64 bytes per vector (60x compression)
- **Slower computation, higher accuracy (better recall)**

**Recommendation:** Start with M=32. If recall is too low, increase to M=64.

---

## Example Execution

### Full Command (Absolute Paths)

```bash
cd /workspaces/irange/build/tests

./pq_encode \
    --data_path_comp /workspaces/irange/executable_data/gist1m/250k/gist_base_250k.bin \
    --query_path /workspaces/irange/executable_data/gist1m/250k/gist_query_250k.bin \
    --range_saveprefix /workspaces/irange/executable_data/gist1m/250k/query_ranges/query_ranges_250k \
    --groundtruth_saveprefix /workspaces/irange/executable_data/gist1m/250k/groundtruth/groundtruth_250k \
    --index_file /workspaces/irange/executable_data/gist1m/250k/gist_250k.index \
    --result_saveprefix /workspaces/irange/executable_data/gist1m/250k/results/results_250k_pq \
    --pq_model_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_pq.faiss \
    --pq_codes_out /workspaces/irange/executable_data/gist1m/250k/pq/gist_250k_codes.bin \
    --M 32
```

### Same Command (Relative Paths)

```bash
cd /workspaces/irange/build/tests

./pq_encode \
    --data_path_comp ../../executable_data/gist1m/250k/gist_base_250k.bin \
    --query_path ../../executable_data/gist1m/250k/gist_query_250k.bin \
    --range_saveprefix ../../executable_data/gist1m/250k/query_ranges/query_ranges_250k \
    --groundtruth_saveprefix ../../executable_data/gist1m/250k/groundtruth/groundtruth_250k \
    --index_file ../../executable_data/gist1m/250k/gist_250k.index \
    --result_saveprefix ../../executable_data/gist1m/250k/results/results_250k_pq \
    --pq_model_out ../../executable_data/gist1m/250k/pq/gist_250k_pq.faiss \
    --pq_codes_out ../../executable_data/gist1m/250k/pq/gist_250k_codes.bin \
    --M 32
```

---

## How PQ Search Works (Step-by-Step)

### Phase 1: Initialization
```
1. Load HNSW graph structure from index_file
2. Load raw query vectors into memory (all of them)
3. Load compressed DB vectors (1 byte per subspace instead of 4 bytes per dimension)
4. Load FAISS codebook with trained centroids
```

### Phase 2: Search for Each Query
```
For each query vector q:
  1. Start at HNSW entry point
  2. For each candidate DB vector c:
     - Compute PQ distance using compute_pq_distance(q, c)
     - This sums L2 distances per subspace between:
       * Query subvector q[m*dsub : (m+1)*dsub]
       * DB centroid[m][code[c*M + m]] (1-byte lookup)
  3. Greedy walk to best neighbor in HNSW graph
  4. Return top-k results
```

### Phase 3: Output
```
Results saved to CSV:
- Query ID | Ground Truth Positives | Retrieved IDs | Recall@k | Latency(ms)
```

---

## Memory Usage Comparison

### Without PQ (Raw Vectors)
- GIST 250k: 250,000 vectors × 3,840 bytes = **960 MB**
- Plus HNSW graph structure: ~100 MB
- **Total: ~1.1 GB**

### With PQ (M=64)
- GIST 250k: 250,000 vectors × 64 bytes = **15 MB**
- Plus HNSW graph structure: ~100 MB
- Plus FAISS codebook: ~5 MB
- **Total: ~120 MB (89% reduction)**

---

## Key Design Decisions

1. **All queries loaded to RAM**: Query vectors are full-precision floats for best accuracy
2. **DB vectors compressed**: Only centroid indices stored (1 byte per subspace)
3. **Distance computed on-the-fly**: No precomputed distance tables (saves space, trades CPU for memory)
4. **FAISS ProductQuantizer API**: Industry-standard implementation, proven accuracy

---

## Troubleshooting

### "Cannot open" errors
- Verify file paths are correct
- Ensure input files (gist_base_250k.bin, gist_query_250k.bin) exist
- Check index_file exists (must be pre-built)

### Low recall
- Increase M parameter (e.g., from 32 to 64)
- Larger M = less compression = better accuracy

### High memory usage
- Decrease number of query vectors loaded simultaneously
- Or pre-build index with `buildindex` tool before running pq_encode

### Slow search
- Reduce M parameter for faster per-vector computation
- Or increase HNSW ef parameter if available in search code

---

## Summary

The PQ implementation provides:
- ✅ **60-120x compression** of vector database
- ✅ **Fast search** via hierarchical HNSW + PQ distance
- ✅ **High accuracy** with configurable compression/accuracy tradeoff (M parameter)
- ✅ **Scalability** to millions of vectors on modest hardware

Use this system to test approximate nearest neighbor search at scale with memory-efficient representations.
