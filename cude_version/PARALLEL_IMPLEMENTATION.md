# GPU Warp-Cooperative Search Kernel

## Overview

The search kernel uses **32 threads (one full warp) per query** to compute
neighbour distances in parallel, while keeping all heap and control-flow logic
on lane 0 to avoid race conditions.

There is no single-threaded fallback — the warp-cooperative approach is the
only kernel.

---

## How It Works

Each warp owns one query for its entire lifetime. Within every greedy search
step the warp follows a strict six-step protocol:

```
Step (a)  Lane 0 only
          │  Check termination (empty heap, distance bound exceeded).
          │  If continuing: pop candidate, call SelectEdge,
          │  write up to 32 neighbour IDs into s_edges[warp][0..31],
          │  write edge count into s_num_edges[warp].
          │  If stopping: write s_num_edges[warp] = -1.
          │
          ▼ __syncwarp()

Step (b)  All 32 lanes read s_num_edges[warp].
          If -1 → break.

Step (c)  Lane i (where i < num_edges)
          │  Reads s_edges[warp][i].
          │  Computes L2Distance between query vector and neighbour i.
          │  Writes result to s_dists[warp][i].
          │  (All 32 lanes run simultaneously — this is the parallel step.)
          │
          ▼ __syncwarp()

Step (d)  Lane 0 only
          │  Loops i = 0..num_edges-1, reads s_dists[warp][i],
          │  marks nodes as visited, inserts into candidate_set / top_candidates.
```

Because every write to a shared-memory cell comes from exactly one lane
(`lane_id == i`), and every heap write comes from exactly lane 0, there are
no race conditions.

---

## Key Constants

```cuda
#define THREADS_PER_QUERY     32   // one warp per query (hardcoded)
#define MAX_QUERIES_PER_BLOCK  8   // blockDim.x (256) / THREADS_PER_QUERY (32)
#define MAX_SEARCH_EF       2000   // maximum supported SearchEF
```

`THREADS_PER_QUERY = 32` also acts as the **edge limit** passed to
`SelectEdge_gpu`. This means the kernel assumes at most 32 neighbours per
node — matching a graph built with `M <= 32`.

---

## Kernel Launch Configuration

```cpp
int threads_per_block = 256;           // 8 warps per block
int queries_per_block = 256 / 32;      // = 8
int num_blocks = (query_nb + queries_per_block - 1) / queries_per_block;

irange_search_kernel<<<num_blocks, threads_per_block>>>(
    gpu_index, visited, query_nb, ef, query_K, dim,
    suffix_idx, d_hops, d_dist_comps,
    index.size_links_per_layer_, kernel_seed
);
```

Each block handles 8 queries concurrently. `query_id` inside the kernel is:

```cuda
const int lane_id     = threadIdx.x % 32;
const int warp_in_blk = threadIdx.x / 32;
const int query_id    = blockIdx.x * (blockDim.x / 32) + warp_in_blk;
```

---

## Shared Memory Layout

Per block (8 warps):

| Array         | Type    | Shape    | Size      |
|---------------|---------|----------|-----------|
| `s_edges`     | `int`   | `[8][32]`| 1024 B    |
| `s_dists`     | `float` | `[8][32]`| 1024 B    |
| `s_num_edges` | `int`   | `[8]`    |   32 B    |
| **Total**     |         |          | **~2 KB** |

Heap buffers (`candidate_buffer`, `top_candidate_buffer`) live in per-thread
**local (register/spill) memory** and are only used by lane 0.

---

## Thread Responsibilities

| Who              | What                                                                                         |
|------------------|----------------------------------------------------------------------------------------------|
| **Lane 0**       | Heap init, entry-point seeding, `SelectEdge`, loop control, heap insertion, result writing, metric counting |
| **All 32 lanes** | `L2Distance` computation for their assigned neighbour (step c above)                        |
| **Lanes 1–31**   | Write distance result into shared memory, then idle until next iteration                    |

---

## Synchronization Points

```
Phase 1 (seeding)   → __syncwarp() after lane-0 init
Each search step    → __syncwarp() after lane-0 SelectEdge   (step a → b)
                    → __syncwarp() after parallel distances   (step c → d)
```

---

## Safety Guarantees

- Only lane 0 pushes to / pops from the heaps
- Each `s_dists[warp][i]` is written by exactly lane `i`
- `__syncwarp()` ensures writes are visible before reads
- `s_num_edges = -1` provides a clean early-exit signal visible to all lanes

---

## Performance Notes

- Parallel distance gain scales with `num_edges`: if SelectEdge returns
  N edges, the wall-clock time for the distance phase is 1/N of the
  sequential cost (ignoring memory latency).
- `L2Distance` iterates over all `dim` floats — at 128-D this is the
  dominant cost per hop, making parallelisation worthwhile.
- Warp divergence is minimal: all lanes execute `L2Distance` together;
  the only divergence is the `if (lane_id < num_edges)` guard when
  `num_edges < 32`.

---

## Future Optimisations

1. **Vectorised loads** — use `float4` to load 128-D vectors in 32 loads
   instead of 128.
2. **Warp-level L2 reduction** — use `__shfl_down_sync` to reduce partial
   dot products across lanes instead of writing to shared memory.
3. **Async prefetch** — prefetch the next candidate's vector while the
   current distances are being processed.
4. **Larger M** — if `M > 32`, increase `THREADS_PER_QUERY` and recompute
   `MAX_QUERIES_PER_BLOCK` so that `threads_per_block` stays <= 1024.

---

## Compile Optimisation Status

Current status in this repo:

- **CMake CPU build** (`CMakeLists.txt`) is already using strong flags:
  `-O3 -march=native -fopenmp -ftree-vectorize`.
- **CUDA Makefile build** (`cude_version/Makefile`) currently compiles with:
  `nvcc -arch=sm_60 ...` but does **not** explicitly set `-O3`.

Recommended compile flags for best practical performance:

```bash
# C++ (CMake)
-O3 -march=native -mtune=native -fopenmp -DNDEBUG

# CUDA (nvcc)
-O3 --use_fast_math -lineinfo -Xcompiler "-O3 -march=native -fopenmp -DNDEBUG"
```

Notes:

- Keep `-lineinfo` for profiling and remove it only for final max-throughput runs.
- `--use_fast_math` can change numerical behavior slightly; validate recall before adopting.
- `-march=native` improves local machine performance but reduces binary portability.

---

## Potential CPU Optimisations

Below are the highest-impact CPU-side improvements (separate from GPU kernel work):

1. **Remove atomic counters in hot loops**  
   Replace `#pragma omp atomic` metric updates with per-thread local counters,
   then reduce once per query batch.

2. **Reduce allocation churn per query**  
   Reuse containers (`selected_edges`, heaps, temporary maps/sets) across queries
   instead of recreating them in the inner loop.

3. **Replace recall check linear search**  
   `std::find(gt[i].begin(), gt[i].end(), x)` is O(K). Convert GT to a hash/set
   structure for O(1) average lookup during evaluation.

4. **Cache range-filter results**  
   `tree->range_filter(...)` is recomputed for every query and EF. Cache by
   `(suffix, query_id)` and reuse across all EF values.

5. **Distance micro-optimisation**  
   Ensure L2 path is vectorized (AVX2/AVX-512 where available), aligned loads,
   and unrolled loops for common dimensions (e.g., 128D).

6. **NUMA and thread placement tuning**  
   Pin OpenMP threads and use compact affinity for memory-local traversal:
   `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.

7. **Prefetch policy tuning**  
   Current prefetches only first few edges; tune count/distance dynamically
   based on observed miss rates and average node degree.

8. **Build-profile-guided optimisation (PGO/LTO)**  
   Add `-flto` and PGO (`-fprofile-generate`/`-fprofile-use`) for stable query
   distributions to improve branch/layout efficiency.
