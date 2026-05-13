# PQ Distance Computation: What You Built vs What You Should Have Built

## 1. Background — what Product Quantization does

Product Quantization (PQ) compresses high-dimensional vectors into a small sequence of
integer codes. The idea is to split each vector into M subvectors, then replace each
subvector with the index of the nearest centroid from a pre-trained codebook.

**Example — GIST dataset (960-dim, M=320, nbits=9):**

```
Original vector:    960 floats  =  3840 bytes
Compressed vector:  320 × 9-bit codes  ≈  360 bytes   (10.7× compression)
Codebook:           320 subspaces × 512 centroids × 3 floats/centroid
                  = 491,520 floats  =  1.88 MB  (stored once, shared by all vectors)
```

**Example — Audio dataset (128-dim, M=32, nbits=9):**

```
Original vector:    128 floats  =  512 bytes
Compressed vector:  32 × 9-bit codes  =  36 bytes   (14.2× compression)
Codebook:           32 subspaces × 512 centroids × 4 floats/centroid
                  = 65,536 floats  =  256 KB
```

The compression is real and substantial. The question is: **how do you compute the
distance from a query to a compressed database vector?**

There are two approaches. You built one of them. FAISS uses the other.

---

## 2. What you built — on-the-fly centroid fetch

Your distance function is in `cude_version/gpu_pq_distance.cuh`, function
`gpu_compute_pq_distance` (line 40).

**What it does for each distance computation:**

```
distance = 0
for m in 0..M:
    centroid_index = decode code[m]              // extract integer from bit-packed codes
    centroid = d_centroids[m * ksub * dsub       // fetch float vector from global memory
                           + centroid_index * dsub]
    for d in 0..dsub:
        diff = query[m*dsub + d] - centroid[d]   // compute L2 per dimension
        distance += diff * diff
return distance
```

**Memory touched per distance computation (GIST, M=320, dsub=3):**

| What | Size |
|---|---|
| Compressed code to decode | 360 bytes |
| Centroid vectors fetched (320 subspaces × 3 floats × 4 bytes) | 3,840 bytes |
| Query subvectors read | 3,840 bytes |
| **Total** | **~8 KB per distance** |

The centroid table is `M × ksub × dsub` floats = `320 × 512 × 3 × 4` = **1.88 MB**.
This does **not fit** in L1 cache (32 KB per SM on P100). So almost every centroid
fetch is a cache miss going all the way to global memory (600+ cycle latency on P100).

**What this means in practice:**

Every single distance computation is as memory-heavy as computing the full uncompressed
L2 distance — or heavier, because you also pay the bit-decode overhead and the
unpredictable centroid address lookup on top. The compression saved storage space on
the GPU, but did not reduce the memory bandwidth consumed during search.

---

## 3. What FAISS does — ADC (Asymmetric Distance Computation)

ADC front-loads the expensive work to a one-time setup step per query, then makes each
per-vector distance computation extremely cheap.

### Step 1 — Build the distance table (once per query)

```
dist_table[M][ksub]   // M × ksub floats

for m in 0..M:
    for k in 0..ksub:
        dist_table[m][k] = L2(query_subvector[m], centroid[m][k])
```

This visits every centroid once. Cost: `M × ksub × dsub` multiply-adds.
For GIST: `320 × 512 × 3 = 491,520` operations — done **once per query**.

The resulting table size: `M × ksub` floats = `320 × 512 × 4` = **655 KB** for GIST.

> Note: for audio (M=32, ksub=512) the table is `32 × 512 × 4 = 65 KB` — still too big
> for L1 (32 KB), but fits in L2 (4 MB on P100), so it stays warm across distance calls.

### Step 2 — Per-vector distance lookup (once per vector evaluated)

```
distance = 0
for m in 0..M:
    k = compressed_code[m]        // decode one integer
    distance += dist_table[m][k]  // one float addition — no multiplication
return distance
```

**Memory touched per distance computation (GIST, M=320):**

| What | Size |
|---|---|
| Compressed code to decode | 360 bytes |
| Distance table reads (320 lookups × 4 bytes) | 1,280 bytes — from L2 cache |
| **Total** | **~1.6 KB per distance, mostly cached** |

**Operations per distance:** M additions. No multiplications. No centroid fetches.

### Side by side

| | Your implementation | ADC (FAISS) |
|---|---|---|
| Centroid table size (GIST) | 1.88 MB | not touched during search |
| Distance table size (GIST) | none | 655 KB (L2 cached after setup) |
| Distance table size (audio) | none | 65 KB (fits L2) |
| Memory per distance (GIST) | ~8 KB from global memory | ~1.6 KB from L2 cache |
| Operations per distance | M×dsub multiply-adds | M additions |
| Setup cost per query | none | M×ksub×dsub multiply-adds |
| Cache behaviour | cold misses on centroid table | table warm in L2 after setup |

---

## 4. Why your implementation is slower than it should be

### Problem 1 — You fetch raw centroids, not pre-computed distances

Every call to `gpu_compute_pq_distance` fetches full centroid float vectors from global
memory. For GIST at M=320, dsub=3, that is 320 separate global memory loads of 3
floats each per distance computation. The centroid table (1.88 MB) is far too large to
cache across calls, so nearly every load is a cache miss.

ADC would replace this with 320 reads from a 655 KB distance table that, once computed
for the current query, can be served from L2 cache.

### Problem 2 — Bit decoding inside the hot loop

`gpu_extract_centroid_index` (lines 7–21 of `gpu_pq_distance.cuh`) runs a bit
extraction loop for every subspace. This is extra work on top of the centroid fetch.
In ADC the decode still happens, but the subsequent memory access hits cache instead of
global memory, so the decode cost is relatively less painful.

### Problem 3 — No distance table in shared memory

FAISS GPU kernels pre-load the distance table into shared memory before the search
loop begins. With the table in shared memory (48 KB on P100), each lookup is a
~1-cycle shared memory read rather than a 600-cycle global memory miss. Your kernel
has no shared memory table at all.

For audio (M=32, ksub=512): table = `32 × 512 × 4 = 65 KB` — slightly too large for
shared memory (48 KB limit on P100). Would need quantisation to 16-bit or splitting.

For GIST (M=320, ksub=512): table = 655 KB — too large for shared memory regardless.
Would need a different strategy (smaller M, or streaming blocks of the table).

### Problem 4 — The compression ratio worked against you

Because you use M=320 for GIST (very high), the codebook is huge and the centroid
table is large. A smaller M (e.g. M=64) with larger dsub would reduce codebook size at
the cost of approximation quality. For GPU ADC, smaller M is better because the
distance table fits in faster memory.

---

## 5. What this means for the thesis

Your PQ result underperformed because you implemented the correct algorithm (PQ
compression + graph traversal) but with the naive distance computation method. The
standard GPU PQ approach (ADC with distance table in L2 or shared memory) was not
implemented. The result is that your PQ version pays both the overhead of decompression
and the overhead of graph traversal's irregular memory access, without gaining the
cache-friendly distance lookups that ADC provides.

**The honest thesis framing:**

> GPU PQ underperformed because the implementation used on-the-fly centroid fetching
> rather than ADC. Each distance computation fetched full centroid vectors from global
> memory (~8 KB at GIST scale), whereas ADC would reduce this to ~1.6 KB served from
> L2 cache by pre-computing a distance table once per query. The structural problem
> (irregular graph traversal causing cache thrashing) is real, but the implementation
> choice amplified the penalty significantly. An ADC-based implementation would be
> expected to perform better, though it remains unclear whether it would overcome the
> irregular access pattern enough to match CPU-P.

---

## 6. How you could improve it

These are ordered from easiest to hardest.

### Option A — Implement ADC (medium effort, high impact)

Before the search loop for each query, compute the distance table on the GPU:

```cuda
// In shared memory or global memory, once per query:
__shared__ float dist_table[MAX_M][MAX_KSUB];   // 32 * 512 * 4 = 65KB (audio only)

// Build table — assign one centroid per thread:
for (int m = threadIdx.x / ksub; m < M; m += blockDim.x / ksub) {
    int k = threadIdx.x % ksub;
    float d = 0.0f;
    for (int i = 0; i < dsub; i++) {
        float diff = query[m*dsub + i] - centroids[m*ksub*dsub + k*dsub + i];
        d += diff * diff;
    }
    dist_table[m][k] = d;
}
__syncthreads();

// Then each distance becomes:
float dist = 0.0f;
for (int m = 0; m < M; m++)
    dist += dist_table[m][code[m]];
```

Works well for audio (table = 65 KB, fits in L2). For GIST (655 KB) you need a
smaller M or 16-bit quantisation of the table entries.

### Option B — Reduce M

Use a smaller number of subspaces (e.g. M=16 or M=32 for GIST instead of M=320).
This reduces compression ratio but makes the distance table small enough to cache.
FAISS typically uses M=8–64 in practice for this reason.

### Option C — Store distance table in global memory, one per query block

Each query block allocates its own region of global memory for the distance table,
computes it at the start, then uses it throughout the search. Slower than shared
memory but works for large M.

### Option D — Use FAISS GPU directly for PQ search

FAISS already implements GPU ADC with IVF+PQ (IndexIVFPQ on GPU). The reason it works
is IVF: each cluster is a flat, predictable memory region that maps directly to matrix
multiplication. This is architecturally different from graph traversal and would require
replacing the HNSW index with an IVF structure, which changes the algorithm entirely.

---

## 7. Summary

| Concept | Your code | FAISS ADC |
|---|---|---|
| Distance computation | Fetch centroid → compute L2 | Look up pre-computed table entry |
| Memory per distance | ~8 KB global memory | ~1.6 KB L2 cache |
| Setup per query | none | Build distance table once |
| Why it is slow | Cold centroid misses + bit decode | N/A — designed for this |
| Fix | Implement ADC table before search loop | Already done |
| Thesis implication | Implementation choice amplified the bottleneck | Would improve results but not fix graph traversal irregularity |

---

## 8. Implementation plan for adding ADC to your code

This section is a concrete blueprint for converting `gpu_search_pq.cuh` to ADC. It is
tailored to your existing kernel structure (1 query per block, 128 threads cooperating
per query).

### 8.1 Memory analysis — which table size fits where

| Dataset | M | ksub | Table size (FP32) | Table size (FP16) | Fits in shared mem (48 KB)? |
|---|---|---|---|---|---|
| audi    | 32  | 512 | 64 KB  | 32 KB  | FP16 yes, FP32 no |
| video   | 256 | 512 | 512 KB | 256 KB | no |
| gist    | 320 | 512 | 640 KB | 320 KB | no |

**Conclusion:** Use shared memory + FP16 for audi. Use global memory for video and gist.
A unified code path with a compile-time switch is easiest.

### 8.2 Files you will touch

```
cude_version/
├── gpu_pq_distance.cuh   ← add ADC table builder + lookup functions
├── gpu_search_pq.cuh     ← replace distance calls with table lookups
├── gpu_index.cuh         ← add d_dist_tables buffer pointer
└── hello_pq.cu           ← allocate d_dist_tables, free after search
```

### 8.3 Step 1 — Allocate a per-query distance table buffer

In `hello_pq.cu`, before the kernel launch loop (around line 638):

```cpp
// Allocate one distance table per query, in global memory
size_t table_size_floats = (size_t)pq_model->M * pq_model->ksub;
size_t total_table_bytes = (size_t)query_nb * table_size_floats * sizeof(float);

float* d_dist_tables = nullptr;
cudaMalloc(&d_dist_tables, total_table_bytes);
gpu_index.d_dist_tables = d_dist_tables;

printf("Allocated %.1f MB for ADC distance tables (%d queries × %zu floats)\n",
       total_table_bytes / (1024.0 * 1024.0), query_nb, table_size_floats);
```

Add to `GPUIndex` struct in `gpu_index.cuh`:
```cpp
float* d_dist_tables;   // [query_nb][M][ksub] — built per query at kernel start
```

Free it after the search loop completes.

**Memory cost check before you do this:**

```
audi  query_nb = 10,000 → 10,000 × 64 KB = 640 MB
gist  query_nb =  1,000 →  1,000 × 640 KB = 640 MB
video query_nb =  1,000 →  1,000 × 512 KB = 512 MB
```

All fit comfortably in the 12 GB P100. If query counts grow, switch to a recycled
buffer indexed by `blockIdx.x % MAX_CONCURRENT_BLOCKS`.

### 8.4 Step 2 — Add the table builder to gpu_pq_distance.cuh

Append two new functions to `gpu_pq_distance.cuh`:

```cuda
// Build the ADC distance table for one query.
// All threads in the block cooperate. Output: dist_table[M * ksub] floats.
__device__ void build_adc_table(
    const float* query_vector,      // dim floats
    const float* centroids,         // [M][ksub][dsub] in global memory
    float* dist_table,              // [M * ksub] — output
    int M, int ksub, int dsub,
    int thread_id, int num_threads)
{
    const int total_entries = M * ksub;

    for (int idx = thread_id; idx < total_entries; idx += num_threads) {
        int m = idx / ksub;
        int k = idx % ksub;

        const float* centroid = &centroids[(m * ksub + k) * dsub];
        const float* q_sub    = &query_vector[m * dsub];

        float d = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < dsub; i++) {
            float diff = q_sub[i] - centroid[i];
            d += diff * diff;
        }
        dist_table[idx] = d;
    }
}

// Lookup ADC distance for one compressed vector — single thread version
__device__ float adc_distance(
    const uint8_t* db_code,        // pointer to this vector's code
    const float* dist_table,       // pre-built table for this query
    int M, int nbits, int ksub)
{
    float distance = 0.0f;
    for (int m = 0; m < M; m++) {
        uint32_t k = gpu_extract_centroid_index_direct(db_code, m, nbits);
        distance += dist_table[m * ksub + k];
    }
    return distance;
}

// Cooperative partial-sum version — for multi-thread distance compute
__device__ float adc_distance_partial(
    const uint8_t* db_code,
    const float* dist_table,
    int M, int nbits, int ksub,
    int lane_in_group, int threads_in_group)
{
    float partial = 0.0f;
    for (int m = lane_in_group; m < M; m += threads_in_group) {
        uint32_t k = gpu_extract_centroid_index_direct(db_code, m, nbits);
        partial += dist_table[m * ksub + k];
    }
    return partial;
}
```

### 8.5 Step 3 — Modify the kernel in gpu_search_pq.cuh

Three changes:

**(a) At kernel start, build the table cooperatively. Insert before Phase 1
(line ~107 in current `gpu_search_pq.cuh`):**

```cuda
// === ADC table build (all 128 threads cooperate) ===
float* my_dist_table = gpu_index.d_dist_tables
                     + (long long)query_id * gpu_index.pq_M * gpu_index.pq_ksub;

build_adc_table(
    query_vector,
    gpu_index.d_centroids,
    my_dist_table,
    gpu_index.pq_M, gpu_index.pq_ksub, gpu_index.pq_dsub,
    threadIdx.x, blockDim.x);

__syncthreads();   // critical: all threads must wait until table is complete
```

**(b) Replace `PQDistance(...)` (entry-point seeding, line ~130) with:**

```cuda
const uint8_t* entry_code = gpu_index.d_compressed_codes
                          + (long long)entry_point * gpu_index.pq_code_size;
float entry_dist = adc_distance(
    entry_code, my_dist_table,
    gpu_index.pq_M, gpu_index.pq_nbits, gpu_index.pq_ksub);
```

**(c) Replace `PQDistancePartial(...)` (neighbour distance compute, line ~194) with:**

```cuda
const uint8_t* nb_code = gpu_index.d_compressed_codes
                       + (long long)neighbor_id * gpu_index.pq_code_size;
float partial = adc_distance_partial(
    nb_code, my_dist_table,
    gpu_index.pq_M, gpu_index.pq_nbits, gpu_index.pq_ksub,
    lane_in_group, DIST_THREADS_PER_NEIGHBOR);
```

Everything else in the kernel stays the same. The warp-shuffle reduction that combines
the partial sums into a final distance continues to work because `partial` is computed
the same way (sum over a strided subset of M).

### 8.6 Step 4 — Sanity-check with a small dataset

Before running the full benchmark:

1. Build with: `cd cude_version && make optimized_test`
2. Run on gist250k only first — fastest feedback loop
3. Check the recall@10 column in the output CSV — it should match the old
   non-ADC version within 1–2% (numerical differences from FP32 reordering are
   normal). If recall drops dramatically, the table build has a bug.
4. Compare QPS to the old PQ version. Expect a 3–10× improvement.

### 8.7 Optional optimisations (in order of impact)

**(i) Use FP16 for the distance table.** Halves memory and improves L2 hit rate.
Requires `__half` arithmetic in the lookup loop — slight precision cost, usually
unnoticeable for recall@10.

**(ii) Move the audi table into shared memory.** With FP16 it is exactly 32 KB,
fits in shared memory, and lookup latency drops from ~30 cycles (L2 hit) to ~1 cycle.
Conditional compile based on the dataset.

**(iii) Reduce M.** A smaller M (say 64 for GIST) drops table size to 128 KB FP32,
fits well in L2, and reduces lookup work from 320 to 64 additions per distance. The
tradeoff is approximation quality — recall@10 may drop 1–2 percentage points.
Worth testing both.

**(iv) Use `__ldg()` for distance table reads.** Forces the load through the
read-only data cache, which has dedicated bandwidth on Pascal/Volta/Ampere.

### 8.8 Expected outcome for the thesis

After implementing ADC you can rerun your PQ benchmarks and report:

- New PQ-GPU QPS curves (very likely competitive with or beating CPU-P for audi)
- A direct comparison: naive PQ vs ADC PQ on the same hardware — this is a
  legitimate ablation study and a strong addition to the results chapter
- An honest revised conclusion in the discussion: PQ on GPU is viable when paired
  with ADC; the original underperformance was caused by the implementation choice,
  not a fundamental architectural barrier

If after ADC your PQ is still slower than non-PQ GPU normal, that finding is also
valuable: it isolates the irregular graph access pattern as the real bottleneck,
independent of compression strategy.
