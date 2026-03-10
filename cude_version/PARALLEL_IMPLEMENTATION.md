# GPU Parallel Distance Computation

## Problem
You correctly identified that the neighbor exploration loop has a **race condition** when trying to parallelize it. Multiple threads accessing the same `candidates` heap would cause data corruption without proper synchronization.

## Solution: Warp-Level Parallelism

I've implemented a **warp-parallel version** that uses 32 threads per query to compute distances in parallel while avoiding race conditions.

### Key Features

1. **32 threads per query**: One full warp cooperates on each query
2. **Parallel distance computation**: Each thread computes distances for a subset of neighbors
3. **NO race conditions**: Only lane 0 (the leader thread) updates the heap
4. **Shared memory**: Used for communication between threads

### How It Works

```
Query Processing (32 threads collaborate):
├─ Lane 0: Controls search flow, manages heaps, calls SelectEdge
├─ All lanes: Read query vector (broadcast from lane 0)
└─ Parallel phase: Each thread processes subset of neighbors
   ├─ Thread 0: neighbors[0, 32, 64, ...]
   ├─ Thread 1: neighbors[1, 33, 65, ...]
   ├─ ...
   └─ Thread 31: neighbors[31, 63, 95, ...]
   
After parallel computation:
└─ Lane 0: Collects all results from shared memory, updates heap serially
```

### Memory Layout

**Shared Memory per Warp:**
- `shared_edges[64]`: Edge IDs to process
- `shared_distances[64]`: Computed distances  
- `shared_neighbors[64]`: Valid neighbor IDs

**Synchronization:**
- `warp.sync()`: Ensures all threads finish before lane 0 reads results
- `warp.shfl()`: Broadcasts data from lane 0 to all threads

### Avoiding Race Conditions

✅ **Safe**: Only lane 0 modifies heaps
✅ **Safe**: Shared memory writes are to unique indices (lane_id)
✅ **Safe**: Warp sync barriers ensure proper ordering

❌ **Would be unsafe**: Multiple threads calling `candidates.push()` simultaneously

## Usage

### Option 1: Single-threaded (Current - Safe Default)
```cpp
#define USE_PARALLEL_KERNEL 0  // in hello.cu line 273
```
- 1 thread per query
- Simpler, lower memory usage
- Good for debugging

### Option 2: Parallel (32 threads per query)
```cpp
#define USE_PARALLEL_KERNEL 1  // in hello.cu line 273
```
- 32 threads per query (1 warp)
- Parallel distance computation
- Potentially faster for large SearchEF

### Kernel Launch Configuration

**Single-threaded:**
```cpp
threads_per_block = 256
num_blocks = (query_nb + 256 - 1) / 256
```

**Parallel:**
```cpp
threads_per_block = 256  // 8 warps per block
num_blocks = (query_nb * 32 + 256 - 1) / 256
shared_memory = 8 warps × (64 floats + 128 ints) per block
```

## Performance Considerations

### When Parallel is Faster:
- Large `num_edges` (many neighbors to compute distances for)
- High-dimensional vectors (128D in your case)
- Memory bandwidth is bottleneck

### When Single-threaded is Faster:
- Small `num_edges` (< 32 neighbors)
- Overhead of synchronization dominates
- Limited shared memory

## Implementation Details

### Critical Code Sections

1. **Entry point initialization (Lane 0 only)**:
```cuda
if (lane_id == 0) {
    // Initialize heaps, add entry points
    // No race conditions - only one thread
}
```

2. **Parallel distance computation**:
```cuda
// Each thread computes distance for its assigned neighbor
if (local_idx < num_edges) {
    neighbor_id = shared_edges[local_idx];
    dist = L2Distance(query_vector, getVectorByID(...), dim);
}
warp.sync();  // Wait for all threads
```

3. **Serial heap update (Lane 0 only)**:
```cuda
if (lane_id == 0) {
    for (int i = 0; i < batch_size; i++) {
        // Read from shared memory, update heap
        candidates.push(shared_distances[i], shared_neighbors[i]);
    }
}
```

## Testing

Current results with **single-threaded version** (USE_PARALLEL_KERNEL=0):
- ✅ Recall: 92-95% across all suffixes
- ✅ QPS: 19K-32K
- ✅ No race conditions

To test **parallel version**:
1. Set `USE_PARALLEL_KERNEL 1` in hello.cu
2. Recompile: `make clean && make`
3. Run: `make run`
4. Compare recall and performance

## Notes

- The parallel version is more complex but eliminates the race condition you identified
- Shared memory usage: ~1.5 KB per warp (very reasonable)
- Warp divergence is minimal (all threads compute distances together)
- The single-threaded version already achieves 93%+ recall, so parallel version is for performance optimization

## Future Optimizations

1. **Vectorized loads**: Use `float4` for loading 128D vectors
2. **Prefetching**: Prefetch next batch while processing current
3. **Async copies**: Use `__pipeline` for overlapping computation and memory access
4. **Multi-query batching**: Process multiple queries with block-level cooperation
