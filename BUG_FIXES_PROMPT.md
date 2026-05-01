Please apply the following bug fixes to my GPU iRangeGraph codebase. There are three main issues to fix: a warp divergence deadlock in the CUDA kernel, a file naming issue in the Makefile paths, and the removal of noisy debug prints.

### 1. Fix Warp Divergence Deadlock in `gpu_search_updated.cuh`
**Why:** In the `irange_search_kernel` (around Phase 2, inside the edge batch processing loop), the `__shfl_down_sync(0xffffffff, ...)` instruction was placed inside an `if (neighbor_slot < edges_in_batch)` condition. Because `0xffffffff` expects all 32 threads in the warp to participate, if `edges_in_batch` wasn't perfectly divisible, the inactive threads would skip the shuffle instructions and hit the following `__syncthreads()`, causing a permanent deadlock.
**What to change:** Bring the warp shuffle reduction outside of the `if` block, ensuring all threads participate, but initialize `partial` to `0.0f` for inactive threads so they don't corrupt the reduction.

**Before (Deadlocked Code):**
```cpp
            if (neighbor_slot < edges_in_batch) {
                int neighbor_idx = edge_base + neighbor_slot;
                int neighbor_id  = s_edges[warp_in_blk][neighbor_idx];

                float partial = L2DistancePartial(
                    query_vector,
                    getVectorByID(neighbor_id, gpu_index.d_data_memory,
                                  gpu_index.d_size_data_per_element, gpu_index.d_offsetData),
                    dim,
                    lane_in_group,
                    DIST_THREADS_PER_NEIGHBOR);

                for (int offset = DIST_THREADS_PER_NEIGHBOR / 2; offset > 0; offset >>= 1) {
                    partial += __shfl_down_sync(0xffffffff, partial, offset, DIST_THREADS_PER_NEIGHBOR);
                }

                if (lane_in_group == 0) {
                    s_dists[warp_in_blk][neighbor_slot] = partial;
                }
            }

            // Sync so lane 0 sees all distance results for this batch
            __syncthreads();
```

**After (Fixed Code):**
```cpp
            float partial = 0.0f;
            int neighbor_id = -1;
            
            if (neighbor_slot < edges_in_batch) {
                int neighbor_idx = edge_base + neighbor_slot;
                neighbor_id  = s_edges[warp_in_blk][neighbor_idx];

                partial = L2DistancePartial(
                    query_vector,
                    getVectorByID(neighbor_id, gpu_index.d_data_memory,
                                  gpu_index.d_size_data_per_element, gpu_index.d_offsetData),
                    dim,
                    lane_in_group,
                    DIST_THREADS_PER_NEIGHBOR);
            }

            for (int offset = DIST_THREADS_PER_NEIGHBOR / 2; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset, DIST_THREADS_PER_NEIGHBOR);
            }

            if (neighbor_slot < edges_in_batch && lane_in_group == 0) {
                s_dists[warp_in_blk][neighbor_slot] = partial;
            }

            // Sync so lane 0 sees all distance results for this batch
            __syncthreads();
```

### 2. Fix the outputs paths and FAISS library paths in `Makefile`
**Why:** The `range_saveprefix`, `groundtruth_saveprefix`, and `result_saveprefix` arguments were missing a trailing underscore, causing files to be saved as `query_ranges_500k0.bin` instead of `query_ranges_500k_0.bin`. In addition, `FAISS_INCLUDE` and `FAISS_LIB_PATH` need default fallback values so `make run` works out of the box in the terminal.
**What to change:** 
1. Set the top variables as follows:
   `FAISS_INCLUDE ?= -I/cluster/home/vetlean/.conda/envs/irange/include`
   `FAISS_LIB_PATH ?= -L/cluster/home/vetlean/.conda/envs/irange/lib`
2. Under `ARGS = ...` and `PQ_ARGS = ...`, append an underscore (`_`) to the ends of the paths for `--range_saveprefix`, `--groundtruth_saveprefix`, and `--result_saveprefix`.

### 3. Remove stray debug logs
**Why:** A test script likely automatically inserted print statements that flood the terminal and slow down execution. 
**What to change:**
1. In `cude_version/gpu_search_updated.cuh`, remove the `printf` statement related to `hop_count++` around line 394 (e.g. `printf("Query0 hop %d pop id %d dist %f\n", ...)`).
2. In `cude_version/hello.cu`, remove all instances of `std::cout << "Init visited array..." << std::endl;` and `std::cout << "Calling CheckPath..." << std::endl;` and `std::cout << "CheckPath done." << std::endl;`. 
