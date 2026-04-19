# GPU Threading & Work Distribution Strategy

## Overview
This document explains how the GPU workload is distributed across threads, blocks, and warps in the PQ search implementation.

---

## Thread & Block Configuration

### Constants (from hello_pq.cu line 25)
```cuda
#define THREADS_PER_QUERY_VAL 128
```

### Kernel Launch Configuration (lines 620-624)
```cuda
int threads_per_block = THREADS_PER_QUERY_VAL;    // 128
int threads_per_query = THREADS_PER_QUERY_VAL;    // 128
int queries_per_block = threads_per_block / threads_per_query;  // = 1
int num_blocks = (query_nb + queries_per_block - 1) / queries_per_block;  // = query_nb
```

### Launch Configuration
```cuda
irange_search_kernel_pq<<<num_blocks, threads_per_block>>>(...);
irange_search_kernel_pq<<<query_nb, 128>>>(...);
```

---

## Work Distribution Model

### Example: 500K Dataset with 100 Queries

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU GRID                             │
│  (query_nb Blocks) × (128 Threads per Block)                 │
└─────────────────────────────────────────────────────────────┘

┌──────────┬──────────┬──────────┬─────────┬──────────────────┐
│  Block 0 │  Block 1 │  Block 2 │ Block 3 │    ...Block 99   │
│  Query 0 │  Query 1 │  Query 2 │ Query 3 │    Query 99      │
└──────────┴──────────┴──────────┴─────────┴──────────────────┘
     ▲          ▲          ▲          ▲             ▲
     │          │          │          │             │
   ┌─┴──┐    ┌─┴──┐    ┌─┴──┐    ┌─┴──┐        ┌──┴─┐
   │128 │    │128 │    │128 │    │128 │        │128 │
   │Thr │    │Thr │    │Thr │    │Thr │        │Thr │
   │    │    │    │    │    │    │    │        │    │
   │eads│    │eads│    │eads│    │eads│        │eads│
   └────┘    └────┘    └────┘    └────┘        └────┘

100 Blocks × 128 Threads/Block = 12,800 Total Threads Running in Parallel
```

---

## Block-Level Processing

### What Each Block Does
Each block is assigned **one query**. All 128 threads in that block collaborate to process that single query.

```
┌────────────────────────────────────────────────┐
│           BLOCK 0 (Query 0)                     │
│         128 Threads Collaborate                 │
├────────────────────────────────────────────────┤
│                                                 │
│  Thread 0:    Search tasks 0, 128, 256, ...   │
│  Thread 1:    Search tasks 1, 129, 257, ...   │
│  Thread 2:    Search tasks 2, 130, 258, ...   │
│  ...                                            │
│  Thread 127:  Search tasks 127, 255, 383, ... │
│                                                 │
│  [Synchronize at barrier points]              │
│  [All threads work together on:                │
│   - Vertex IPS (K-nearest candidates)         │
│   - Distance computations                      │
│   - Graph traversal decisions]                │
│                                                 │
└────────────────────────────────────────────────┘
```

---

## Warp-Level Processing

### GPU Warp Structure
- **Warp Size**: 32 threads (standard for NVIDIA GPUs)
- **Warps per Block**: 128 threads ÷ 32 = 4 warps

```
Block (128 Threads / 1 Query)
├─ Warp 0 (Threads 0-31):   32 threads
├─ Warp 1 (Threads 32-63):  32 threads
├─ Warp 2 (Threads 64-95):  32 threads
└─ Warp 3 (Threads 96-127): 32 threads
```

### Warp-Level Execution
Within each warp, all 32 threads execute **in lockstep** (SIMD):
- Same instruction every cycle
- Can diverge (if-then-else), but with performance penalty
- Shared L1 cache (96 KB per block)
- Cooperative work on graph traversal

---

## Per-Query Search Breakdown

### Search Flow (For Each of 100 Queries)

```
Query Vector Q
     │
     ├─ [Host CPU] Load Q from GPU memory
     │              (all 128 threads access d_query_vectors)
     │
     ├─ [GPU Block] Thread 0-127 collaborate on K-NN search:
     │   ├─ Step 1: GR Initialization (parallel across threads)
     │   ├─ Step 2: Greedy Search:
     │   │   ├─ Get candidates (from segment tree)
     │   │   ├─ Compute PQ distances (128 threads parallel)
     │   │   │  - Load compressed codes: d_compressed_codes
     │   │   │  - Load centroids: d_centroids
     │   │   │  - PQDistance() kernel work shared
     │   │   ├─ Find K best (heap operations)
     │   │   ├─ Get neighbors (from adjacency lists: d_data_memory)
     │   │   │  - Each thread: get_linklist_gpu(pid, layer, ...)
     │   │   │            = d_data_memory + pid * d_size_data_per_element
     │   │   │                            + layer * size_links_per_layer
     │   │   │  - size_links_per_element = 2528 bytes (compact, no padding)
     │   │   └─ Mark visited: d_mass[pid] (atomic operations)
     │   │
     │   └─ Output: K results stored in d_results[query_id * K + 0..9]
     │
     └─ [Host CPU] Copy results back & compute metrics
```

---

## Memory Access Patterns

### Per-Query Memory Footprint (128 threads active)

```
Per Query Search Iteration:

1. LOAD QUERY VECTOR:
   - Each thread reads d_query_vectors[query_id * dim + ...]
   - Bandwidth: 128 threads × 96 dims × 4 bytes = 49 KB/query
   
2. LOAD COMPRESSED CODES (for distance):
   - Access d_compressed_codes[neighbor_id * code_size]
   - code_size ≈ 40 bytes (M=320, 1 bit per subspace)
   - Per iteration: varied access pattern
   
3. LOAD PQ CENTROIDS:
   - Access d_centroids for PQDistance computation
   - Cached: 20-50 MB (shared among all queries)
   
4. LOAD ADJACENCY LISTS:
   - Access d_data_memory[pid * 2528 + layer * size_per_layer]
   - Compact: only 2,528 bytes per point (NO embeddings)
   - Per iteration: multiple neighbor list accesses
   
5. WRITE VISITED MARKS:
   - Atomic writes to d_mass[query_id * max_elements + pid]
   - Coalesced writes for efficiency
   
6. WRITE RESULTS:
   - Store K results in d_results[query_id * K + 0..9]
   - Coalesced write once per query
```

---

## Scheduling & Occupancy

### GPU Scheduler Perspective

```
GPU Streaming Multiprocessor (SM) - Multiple per GPU

SM can run multiple blocks concurrently:

Time T0:  [Block 0][Block 1][Block 2]....[BlockN]
          Active blocks depends on:
          - Register usage per thread
          - Shared memory per block
          - Hardware SM count

Register Usage per Thread: ~64 bytes (estimated)
Shared Memory per Block: ~4 KB (segment tree locals)
→ Can fit multiple blocks per SM

Theoretical Max Blocks per SM: 8-16
GPU with 60 SMs: Can run 480-960 blocks simultaneously
```

---

## Work Distribution Timeline (For One SearchEF Value)

```
Timeline:
├─ T0: Launch all 100 blocks (THREADS_PER_QUERY_VAL=128)
│       100 blocks × 128 threads = 12,800 threads total
│
├─ T1-T10000: Parallel Search on All Queries
│       - GR initialization (all blocks in parallel)
│       - Greedy search iterations (synchronized across queries)
│       - Each iteration: variable # of neighbor checks
│
├─ T10001: All blocks complete
│       - Results written to d_results (1000 ints per query)
│
├─ T10002: Copy results back to CPU
│       - 100 queries × 10 results = 1000 ints = 4 KB
│
└─ T10003: Compute metrics (CPU side)
       - Recall calculation
       - Throughput: QPS = 100 / search_time
```

---

## Memory Layout for Compact Mode

### CPU Memory (5 GB for 500K dataset)
```
data_memory_ Layout:
┌─────────────┬─────────────┬─────────────┐
│ Adjacency 0 │ Adjacency 1 │ Adjacency 2 │ ... (500K points)
│  2528 bytes │  2528 bytes │  2528 bytes │
└─────────────┴─────────────┴─────────────┘
       ↑            ↑            ↑
    No padding   No embeddings   No padding
```

### GPU Memory (Compact Layout)
```
d_data_memory (GPU):
┌─────────────┬─────────────┬─────────────┐
│ Adjacency 0 │ Adjacency 1 │ Adjacency 2 │ ... (500K points)
│  2528 bytes │  2528 bytes │  2528 bytes │
└─────────────┴─────────────┴─────────────┘
  Total: 1.27 GB

d_compressed_codes (GPU):
┌─────────┬─────────┬─────────┐
│ Code 0  │ Code 1  │ Code 2  │ ... (500K vectors)
│  40 B   │  40 B   │  40 B   │
└─────────┴─────────┴─────────┘
  Total: 20 MB

d_centroids (GPU):
┌─────────────────────────────────┐
│ M × ksub × dsub float32 values   │
│ 320 × 256 × 96 floats = 7.86 MB │
└─────────────────────────────────┘

d_query_vectors (GPU):
├─ 100 queries × 96 dims × 4 bytes = 38.4 KB

────────────────────────────────────────────
Total GPU Memory: ~1.23 GB (fits easily in 12GB P100)
```

---

## Key Properties

| Property | Value |
|----------|-------|
| **Threads per Block** | 128 |
| **Threads per Query** | 128 |
| **Queries per Block** | 1 |
| **Num Blocks** | query_nb (e.g., 100) |
| **Total Threads** | query_nb × 128 |
| **Warps per Block** | 4 (128 ÷ 32) |
| **Max Queries Parallel** | min(query_nb, GPU_capacity) |
| **Memory per Query** | PQ codes + centroids (shared) + results |
| **Synchronization** | Inter-block (implicit in kernel completion) |

---

## Performance Characteristics

### Strengths ✓
- **Massive parallelism**: All queries processed simultaneously
- **Good occupancy**: Each block well-fed with work (128 threads)
- **Memory efficiency**: Compact layout, no padding
- **Reduced memory**: 1.27 GB vs 5 GB (CPU padded)

### Considerations ⚠
- **Per-query overhead**: Each query gets dedicated block
- **Load imbalance**: Different queries may finish at different times
- **Synchronization cost**: k-NN heap operations may have thread divergence

---

## Example Execution

For 500K dataset with 100 queries and SearchEF=1000:

```
GPU Utilization:
┌─────────────────────────────────────────┐
│ Iteration 0:  12,800 threads active     │
│            (100 blocks × 128 threads)   │
│                                          │
│ Iteration 1-1000: Variable Activity     │
│ (depends on graph traversal patterns)    │
│                                          │
│ Expected Duration: 10-100ms per query    │
│ Total SearchEF=1000 time: ~50-500ms     │
│                                          │
│ QPS = 100 queries / 0.05-0.5 seconds    │
│     = 200-2000 queries/second           │
└─────────────────────────────────────────┘
```

---

## GPU Memory Optimization Strategies

### GPU Memory Hierarchy (Fast to Slow)

```
GPU Memory Hierarchy (Latency & Bandwidth):

┌─────────────────────────────────────────────────────────────────┐
│ REGISTERS                                                        │
│ ├─ Per-thread: 64 KB (NVIDIA P100)                              │
│ ├─ Latency: ~0 cycles (immediate access)                        │
│ ├─ Bandwidth: Unlimited (one value per cycle per warp)          │
│ └─ Usage: Local variables, loop counters                        │
├─────────────────────────────────────────────────────────────────┤
│ SHARED MEMORY (L1 Cache Equivalent)                              │
│ ├─ Per-block: 96 KB shared + 96 KB L1 per SM                    │
│ ├─ Latency: 20-30 cycles (after bank conflicts resolved)        │
│ ├─ Bandwidth: ~1 TB/s (all 32 threads in warp access)           │
│ └─ Usage: Thread collaboration, small working sets              │
├─────────────────────────────────────────────────────────────────┤
│ L2 CACHE                                                         │
│ ├─ GPU-wide: 1.5-4 MB (shared by all SMs)                       │
│ ├─ Latency: 200-400 cycles                                      │
│ ├─ Bandwidth: Progressive (hit dependent)                       │
│ └─ Usage: Auto cache for global memory                          │
├─────────────────────────────────────────────────────────────────┤
│ GLOBAL MEMORY (HBM - Main GPU Memory)                            │
│ ├─ Total: 12-16 GB (P100/V100 typical)                          │
│ ├─ Latency: 400-800 cycles (uncached)                           │
│ ├─ Bandwidth: 720 GB/s (with good coalescing)                   │
│ └─ Usage: Index, vectors, results                               │
└─────────────────────────────────────────────────────────────────┘

Relative Speed: Registers >> L1/Shared >> L2 >> Global Memory
                ≈ 1,000x faster registers vs global memory
```

---

### Current Memory Placement Strategy

```
Current Implementation:

GLOBAL MEMORY (GPU HBM):
├─ d_data_memory:        1.27 GB   (Adjacency lists - COMPACT)
├─ d_compressed_codes:   20 MB     (PQ codes - uint8)
├─ d_centroids:          7.86 MB   (PQ centroids - float32)
├─ d_query_vectors:      38.4 KB   (Queries - float32)
├─ d_segment_tree.d_nodes: ~2 MB   (Tree nodes)
└─ d_results:            ~4 KB     (Results)

Total Used: ~1.26 GB (out of 12 GB available = 10.5% utilized)
          = Excellent! Plenty of room for optimization
```

---

### Memory Optimization Techniques

#### 1. **Shared Memory Prefetching** (Cache Warmup)

**Problem**: Each thread repeatedly loads:
- Adjacency lists (d_data_memory)
- PQ codes (d_compressed_codes) 
- Centroids (d_centroids)

**Solution**: Prefetch to shared memory at block level

```cuda
// In kernel code:
__shared__ float shared_centroids[512];  // Cache subset of centroids

// Thread 0 loads centroid subset once at block start
if (threadIdx.x < 512) {
    shared_centroids[threadIdx.x] = 
        d_centroids[blockIdx.x * 512 + threadIdx.x];
}
__syncthreads();  // Wait for all threads

// Now all 128 threads access fast shared memory instead of global
float dist = PQDistance(code, shared_centroids);  // 20-30 cycle latency vs 400-800
```

**Benefits:**
- ✓ 10-20x latency reduction for centroids
- ✓ Reduced global memory bandwidth pressure
- ✓ Cost: 512 floats × 4 bytes = 2 KB shared memory (96 KB available, plenty)

---

#### 2. **Adjacency List Caching via Shared Memory**

**Problem**: Threads repeatedly access same neighbors:
```cuda
// Current: Every thread loads from global memory independently
int* neighbors = get_linklist_gpu(pid, layer, d_data_memory, ...);  // 400+ cycles
```

**Solution**: Cache frequently accessed neighbor lists

```cuda
// Optimized: Share neighbor lists within block
__shared__ int shared_neighbors[256];  // 1 KB (can fit ~64 neighbor entries)

// One thread per warp loads neighbors once
#pragma unroll 4
for (int entry = threadIdx.x; entry < num_neighbors; entry += blockDim.x) {
    if (entry < 256) {
        shared_neighbors[entry] = neighbors[entry];  // Load from global
    }
}
__syncthreads();

// All threads access fast shared memory
int neighbor_id = shared_neighbors[entry_idx];  // 20-30 cycles
```

**Benefits:**
- ✓ 15-20x speedup for neighbor list access
- ✓ Cost: 256 ints × 4 bytes = 1 KB (96 KB available)

---

#### 3. **Coalesced Global Memory Access**

**Current Access Pattern (Inefficient):**
```cuda
// Thread i reads from scattered addresses in d_data_memory
int pid = candidates[threadIdx.x];
int* neighbors = get_linklist_gpu(pid, layer, d_data_memory, stride);
// stride calculation: pid * 2528 (non-contiguous per thread)
```

**Optimized Access Pattern:**
```cuda
// All threads in warp access contiguous memory segment
// Addresses: [base, base+128, base+256, base+384, ...]

// Ensure d_data_memory layout matches warp access pattern
// Current layout: Point0[2528] | Point1[2528] | Point2[2528] | ...
// Access pattern: Thread0→Point0, Thread1→Point1, ...
// ✓ ALREADY WELL COALESCED! (One point = 2528 bytes ≈ 88 cache lines)

// Further optimization: Process multiple neighbors per thread
#pragma unroll 4
for (int n = 0; n < neighbors_per_thread; n++) {
    int neighbor_offset = n * 32;  // Stride by 32 neighbors
    if (threadIdx.x + neighbor_offset < total_neighbors) {
        int neighbor_id = neighbors[threadIdx.x + neighbor_offset];
        // Coalesced access across warp
    }
}
```

**Benefits:**
- ✓ Memory bandwidth: 720 GB/s (vs ~40 GB/s if uncoalesced)
- ✓ 18x improvement in throughput
- ✓ Currently good - can be better with neighbor batching

---

#### 4. **Texture Cache for Random Access** (Advanced)

**Use Case**: PQ codes have random access patterns per query

```cuda
// Bind PQ codes to texture for automatic caching
// Texture cache: 48 KB per SM, optimized for 2D locality

// In host code:
cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint8_t>();
cudaBindTexture(nullptr, tex_pq_codes, d_compressed_codes, 
                desc, pq_codes_size);

// In kernel:
uint8_t code_byte = tex1Dfetch(tex_pq_codes, code_index);
// Faster than d_compressed_codes[code_index] for non-sequential access
```

**Benefits:**
- ✓ Better cache hit rate for random access patterns
- ✓ 2-3x speedup for scattered PQ code access
- ✓ Tradeoff: Slightly less intuitive code

---

#### 5. **Register Tiling for Distance Computation**

**Problem**: Computing distances requires many loads from global memory

**Solution**: Keep multiple query vectors in registers

```cuda
// Cache multiple dimensions in registers per thread
#pragma unroll 8
for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    query_registers[d % 32] = query_vector[d];  // registers faster than shared
}

// Now compute distances:
float dist = 0;
#pragma unroll 8
for (int d = 0; d < dim; d += 1) {
    // Access query from registers (0 cycle latency)
    float q_val = query_registers[d % 32];
    // Access code from centroids (shared memory, 20-30 cycles)
    dist += fabs(q_val - centroids[code[d]]);
}
```

**Benefits:**
- ✓ Eliminate repeated global memory loads
- ✓ 10-15% improvement for distance-heavy kernels

---

### Memory Layout Optimization

#### Current Layout (Good)
```
d_data_memory (Compact):
Byte 0    |------ Point 0 (2528 bytes) ------|  Byte 2528
Byte 2528 |------ Point 1 (2528 bytes) ------|  Byte 5056
Byte 5056 |------ Point 2 (2528 bytes) ------|  Byte 7584
...
```

**Why it's good:**
- ✓ Sequential access = coalesced memory reads
- ✓ No padding = 71.6% memory savings vs CPU version
- ✓ One query accesses sequential blocks

#### Potential Improvement: Struct-of-Arrays (SoA) Layout

**Current (Array-of-Structs - AoS):**
```
Point0 = [Edge0_count, Edge0_ids..., Edge1_ids..., ...]
Point1 = [Edge1_count, Edge1_ids..., Edge1_ids..., ...]
Point2 = [Edge2_count, Edge2_ids..., Edge2_ids..., ...]
```

**Proposed (Struct-of-Arrays - SoA):**
```
d_edge_counts[500K]      // Layer 0 edge counts
d_edge_lists_layer0[...]  // Layer 0 edges
d_edge_lists_layer1[...]  // Layer 1 edges
```

**Trade-off:**
- ✗ More complex indexing
- ✗ May fragment cache
- ✓ Better for batch operations across all points
- ✓ Only needed if accessing same layer across many points

---

### Practical Optimization Roadmap

#### Phase 1: Easy Wins (Implement First)
1. **Prefetch Centroids to Shared Memory** (2 KB overhead)
   - Expected gain: 15-20% faster distance computation
   - Risk: Low
   - Lines to modify: gpu_search_pq.cuh (PQDistance kernel)

2. **Cache Neighbor Lists in Shared Memory** (1 KB overhead)
   - Expected gain: 10-15% faster graph traversal
   - Risk: Low
   - Lines to modify: gpu_search_pq.cuh (get_linklist calls)

#### Phase 2: Medium Complexity
3. **Use Texture Cache for PQ Codes**
   - Expected gain: 5-10% for random access patterns
   - Risk: Medium (binding overhead)
   - Lines to modify: gpu_index.cuh, gpu_search_pq.cuh

4. **Register Tiling for Query Vectors**
   - Expected gain: 5-10% for distance computation
   - Risk: Low (register spilling would hurt)
   - Lines to modify: gpu_search_pq.cuh (distance loops)

#### Phase 3: Advanced
5. **Persistent Kernel Approach** (Run searches continuously)
   - Current: Launch kernel per SearchEF value
   - Advanced: Keep kernel running, feed queries continuously
   - Expected gain: 20-30% (eliminate kernel launch overhead)
   - Risk: High (major restructuring)

---

### Memory Bandwidth Analysis

#### Current Bottleneck

```
Typical SearchEF=1000 Iteration:

Per thread per iteration:
├─ Load query vector:        96 dims × 4 bytes =  384 bytes
├─ Load neighbor IDs:        32 neighbors × 4 bytes = 128 bytes  
├─ Load PQ codes:            32 codes × 40 bytes = 1,280 bytes
├─ Load centroids:           Variable (from distance comp) ≈ 512 bytes
└─ Write distances + updates: ≈ 50 bytes

Total per iteration per thread: ~2,400 bytes

128 threads × 2,400 bytes = 307 KB per iteration
100 blocks × 307 KB = 30.7 MB per iteration
1000 iterations × 30.7 MB = 30.7 GB of traffic

GPU Bandwidth: 720 GB/s
Time for memory: 30.7 GB / 720 GB/s = 42.6 ms

Actual runtime: 50-500 ms (memory is NOT the bottleneck!)
└─ Bottleneck is actually: Graph traversal logic, not memory
```

**Implication**: Memory optimizations will help, but diminishing returns after Phase 1.

---

### Memory Optimization Checklist

- [ ] Prefetch centroids to shared memory (2 KB alloc)
- [ ] Cache neighbor lists in shared memory (1 KB alloc)
- [ ] Verify coalescing with `nvprof --print-gpu-memory-trace`
- [ ] Check cache hit rates: `nvprof --events sm_efficiency`
- [ ] Consider texture cache for PQ codes if profiling shows low L2 hit rate
- [ ] Profile with `-Xptxas -v` to check register pressure
- [ ] Measure improvement: baseline vs optimized versions
- [ ] Test on P100/V100 for SM-specific optimizations

---

## Summary

The system uses a **one-query-per-block** parallelization strategy where:

1. Each query gets its own CUDA block with 128 threads
2. All 128 threads collaborate on searching graph neighbors for that query
3. PQ distance computations are parallelized across threads
4. Results are written concurrently for all queries
5. Memory is compact (1.27 GB) with no embeddings or padding

This approach maximizes GPU utilization by keeping all cores busy across multiple queries simultaneously.
