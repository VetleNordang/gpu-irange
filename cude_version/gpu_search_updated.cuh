#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include "gpu_index.cuh"
#include "gpu_visited.cuh"
#include "gpu_heap.cuh"
#include "gpu_pq_distance.cuh"

// Maximum SearchEF value supported (arrays allocated to this size)
#define MAX_SEARCH_EF 2000

// ============ GPU Range Filter ============
// Filter segment tree nodes that overlap with query range [ql, qr]
__device__ int range_filter_gpu(GPUNode* d_nodes, int root_idx, int ql, int qr, 
                                int* output_indices, int max_output) {
    // Manual stack for iterative traversal (avoid recursion on GPU)
    int stack[50];  // Max depth ~20 for most trees
    int stack_top = 0;
    int output_count = 0;
    
    // Start with root
    stack[stack_top++] = root_idx;
    
    while (stack_top > 0 && output_count < max_output) {
        // Pop node from stack
        int current_idx = stack[--stack_top];
        GPUNode current = d_nodes[current_idx];
        
        // If this node is completely within range, add it and skip children
        if (current.lbound >= ql && current.rbound <= qr) {
            output_indices[output_count++] = current_idx;
            continue;
        }
        
        // If no overlap, skip this node
        if (current.lbound > qr || current.rbound < ql) {
            continue;
        }
        
        // Partial overlap - push children onto stack
        if (!current.is_leaf) {
            if (current.right_child_index != -1 && stack_top < 50) {
                stack[stack_top++] = current.right_child_index;
            }
            if (current.left_child_index != -1 && stack_top < 50) {
                stack[stack_top++] = current.left_child_index;
            }
        }
    }
    
    return output_count;
}

// ============ Distance Calculation ============
__device__ float L2Distance(const float *a, const float *b, int dim) {
    const bool aligned16 = ((((uintptr_t)a | (uintptr_t)b) & 0xF) == 0);

    if (aligned16 && (dim % 4 == 0)) {
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);

        float sum = 0.0f;
        const int dim4 = dim >> 2;
        for (int i = 0; i < dim4; i++) {
            float4 av = a4[i];
            float4 bv = b4[i];

            float dx = av.x - bv.x;
            float dy = av.y - bv.y;
            float dz = av.z - bv.z;
            float dw = av.w - bv.w;

            sum += dx * dx + dy * dy + dz * dz + dw * dw;
        }
        return sum;
    }

    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;  // Squared distance
}

// Partial L2 distance for cooperative groups of threads.
// lane_in_group computes a strided subset and caller reduces across group.
__device__ float L2DistancePartial(const float *a, const float *b, int dim,
                                   int lane_in_group, int threads_in_group) {
    const bool aligned16 = ((((uintptr_t)a | (uintptr_t)b) & 0xF) == 0);

    if (aligned16 && (dim % 4 == 0)) {
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);

        float sum = 0.0f;
        const int dim4 = dim >> 2;
        for (int i = lane_in_group; i < dim4; i += threads_in_group) {
            float4 av = a4[i];
            float4 bv = b4[i];

            float dx = av.x - bv.x;
            float dy = av.y - bv.y;
            float dz = av.z - bv.z;
            float dw = av.w - bv.w;

            sum += dx * dx + dy * dy + dz * dz + dw * dw;
        }
        return sum;
    }

    float sum = 0.0f;
    for (int i = lane_in_group; i < dim; i += threads_in_group) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Helper to get vector data for a given node ID
__device__ float* getVectorByID(int node_id, char* data_memory, 
                                size_t size_data_per_element, size_t offsetData) {
    char* data_ptr = data_memory + node_id * size_data_per_element + offsetData;
    return (float*)data_ptr;
}

// ============ Graph Navigation ============
// Get the neighbor list for a given node at a specific layer/depth
__device__ int* get_linklist_gpu(int node_id, int layer, char* data_memory,
                                  size_t size_data_per_element, size_t size_links_per_layer,
                                  int* list_size_out) {
    // Calculate offset: node_id * size_per_element + layer * size_per_layer
    char* linklist_ptr = data_memory + node_id * size_data_per_element + layer * size_links_per_layer;
    
    // First 4 bytes contain the list size
    int* list_size_ptr = (int*)linklist_ptr;
    *list_size_out = *list_size_ptr;
    
    // Neighbor IDs start after the size
    int* neighbor_list = list_size_ptr + 1;
    
    return neighbor_list;
}

// Get overlap between two ranges
__device__ int GetOverLap(int l, int r, int ql, int qr) {
    int L = (l > ql) ? l : ql;
    int R = (r < qr) ? r : qr;
    return R - L + 1;
}

// SelectEdge: Navigate segment tree to find edges at appropriate depths (matching CPU logic)
__device__ void SelectEdge_gpu(int pid, int ql, int qr, int edge_limit,
                                GPUNode* d_nodes, int root_idx,
                                char* data_memory, size_t size_data_per_element,
                                size_t size_links_per_layer,
                                GPUVisitedArray visited, int query_id,
                                int* output_edges, int* output_count) {
    *output_count = 0;
    
    // Start from root and navigate to find node containing pid
    int cur_idx = root_idx;
    int nxt_idx = root_idx;
    GPUNode cur_node;  // Declare here so it's visible for the outer while condition
    
    do {
        cur_idx = nxt_idx;
        cur_node = d_nodes[cur_idx];
        
        // Inner loop: keep descending while overlap doesn't change
        bool contain = false;
        do {
            contain = false;
            
            // Find child containing pid
            nxt_idx = -1;
            if (!cur_node.is_leaf) {
                // Check left child
                if (cur_node.left_child_index != -1) {
                    GPUNode left_child = d_nodes[cur_node.left_child_index];
                    if (left_child.lbound <= pid && left_child.rbound >= pid) {
                        nxt_idx = cur_node.left_child_index;
                    }
                }
                // Check right child if left didn't match
                if (nxt_idx == -1 && cur_node.right_child_index != -1) {
                    GPUNode right_child = d_nodes[cur_node.right_child_index];
                    if (right_child.lbound <= pid && right_child.rbound >= pid) {
                        nxt_idx = cur_node.right_child_index;
                    }
                }
                
                // If found child and overlap is same, continue descending
                if (nxt_idx != -1) {
                    GPUNode nxt_node = d_nodes[nxt_idx];
                    int cur_overlap = GetOverLap(cur_node.lbound, cur_node.rbound, ql, qr);
                    int nxt_overlap = GetOverLap(nxt_node.lbound, nxt_node.rbound, ql, qr);
                    
                    if (cur_overlap == nxt_overlap) {
                        cur_idx = nxt_idx;
                        cur_node = nxt_node;
                        contain = true;
                    }
                }
            }
        } while (contain);
        
        // Get edges for pid at this depth (cur_node.depth)
        int neighbor_count = 0;
        int* neighbors = get_linklist_gpu(pid, cur_node.depth, data_memory, 
                                         size_data_per_element, size_links_per_layer,
                                         &neighbor_count);
        
        
        // Filter neighbors by range and visited status (0-indexed neighbor list)
        for (int i = 0; i < neighbor_count && *output_count < edge_limit; i++) {
            int neighbor_id = neighbors[i];
            
            // Check if in range [ql, qr]
            if (neighbor_id < ql || neighbor_id > qr) 
                continue;
            
            // Check if already visited
            if (isVisited(visited, query_id, neighbor_id)) 
                continue;
            
            // Add this neighbor to results
            output_edges[(*output_count)++] = neighbor_id;
    
            
            // Stop if collected enough edges
            if (*output_count >= edge_limit) 
                return;
        }
        
        // Continue while current node doesn't fully contain query range [ql, qr]
    } while (cur_node.lbound < ql || cur_node.rbound > qr);
}

// Simple random number generator for GPU
__device__ unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ============ 128-thread Search Kernel (Option B) ============
// 128 threads (4 warps) collaborate on every query:
//   - Lane 0 owns the heap, SelectEdge, loop control, and result writing.
//   - All 128 threads split each distance across DIST_THREADS_PER_NEIGHBOR=4 threads.
//   - NEIGHBORS_PER_BATCH = 128/4 = 32, matching M=32 exactly (no idle lanes).
//   - 1 query per block so __syncthreads() is always safe.

#define THREADS_PER_QUERY     128
#define MAX_QUERIES_PER_BLOCK   1  // 1 query per block of 128 threads

// Distance parallelization mode (compile-time switch):
// 1 = one thread computes one full distance (max neighbour parallelism)
// 4 = four threads cooperate per distance (more per-distance parallelism)
#ifndef DIST_THREADS_PER_NEIGHBOR
#define DIST_THREADS_PER_NEIGHBOR 4
#endif

#if (THREADS_PER_QUERY % DIST_THREADS_PER_NEIGHBOR) != 0
#error "DIST_THREADS_PER_NEIGHBOR must divide THREADS_PER_QUERY"
#endif

#define NEIGHBORS_PER_BATCH (THREADS_PER_QUERY / DIST_THREADS_PER_NEIGHBOR)

// ============================================================================
// PQ-Optimized Kernel: On-the-fly PQ distance computation (no branching)
// ============================================================================
__global__ void irange_search_kernel_pq(
    GPUIndex gpu_index,
    GPUVisitedArray visited,
    int query_nb,
    int SearchEF,
    int query_K,
    int dim,
    int suffix_id,           // Which range to use
    int* d_hops,             // Output: hops per query
    int* d_dist_comps,       // Output: distance computations per query
    size_t size_links_per_layer,
    unsigned long long seed
) {
    // --- Thread identity ---
    const int lane_id           = threadIdx.x % THREADS_PER_QUERY;
    const int warp_in_blk       = threadIdx.x / THREADS_PER_QUERY;  // always 0 (1 query/block)
    const int queries_per_block = blockDim.x  / THREADS_PER_QUERY;
    const int query_id          = blockIdx.x  * queries_per_block + warp_in_blk;

    if (query_id >= query_nb) return;

    // --- Shared memory: one slot per query in the block (MAX_QUERIES_PER_BLOCK=1) ---
    // s_edges    : up to NEIGHBORS_PER_BATCH (32) neighbour IDs from SelectEdge
    // s_dists    : final distances written by lane-in-group==0 after reduction
    // s_num_edges: set by lane 0; -1 signals loop termination
    __shared__ int   s_edges    [MAX_QUERIES_PER_BLOCK][NEIGHBORS_PER_BATCH];
    __shared__ float s_dists    [MAX_QUERIES_PER_BLOCK][NEIGHBORS_PER_BATCH];
    __shared__ int   s_num_edges[MAX_QUERIES_PER_BLOCK];

    // --- Per-query values readable by all lanes ---
    float* query_vector = gpu_index.d_query_vectors + (long long)query_id * dim;
    int    range_idx    = query_id * 2 * gpu_index.num_suffixes + suffix_id * 2;  // Dynamic suffix count
    int    ql           = gpu_index.d_query_range[range_idx];
    int    qr           = gpu_index.d_query_range[range_idx + 1];

    // --- Heap storage in per-thread local memory (only lane 0 uses it) ---
    HeapNode candidate_buffer    [MAX_SEARCH_EF];
    HeapNode top_candidate_buffer[MAX_SEARCH_EF];
    MinHeap  candidate_set;
    MaxHeap  top_candidates;
    float    lowerBound     = FLT_MAX;
    int      hop_count      = 0;
    int      dist_comp_count = 0;

    // =========================================================
    // Phase 1 - Entry-point seeding (lane 0 only)
    // =========================================================
    if (lane_id == 0) {
        candidate_set.init(candidate_buffer,     MAX_SEARCH_EF);
        top_candidates.init(top_candidate_buffer, SearchEF);

        unsigned int rng_state = (unsigned int)(seed + query_id);

        int filtered_indices[100];
        int num_filtered = range_filter_gpu(gpu_index.d_segment_tree.d_nodes, 0, ql, qr,
                                            filtered_indices, 100);


        for (int f = 0; f < num_filtered; f++) {
            GPUNode seg_node  = gpu_index.d_segment_tree.d_nodes[filtered_indices[f]];
            int range_size    = seg_node.rbound - seg_node.lbound + 1;
            int entry_point   = seg_node.lbound + (xorshift32(&rng_state) % range_size);

            if (isVisited(visited, query_id, entry_point)) continue;
            markVisited(visited, query_id, entry_point);

            // PQ kernel: always use PQ distance
            float entry_dist = gpu_compute_pq_distance(
                query_vector, gpu_index.d_compressed_codes, gpu_index.d_centroids,
                entry_point, gpu_index.pq_M, gpu_index.pq_nbits,
                gpu_index.pq_dsub, gpu_index.pq_code_size, gpu_index.pq_ksub);
            dist_comp_count++;

            candidate_set.push(entry_dist, entry_point);
            top_candidates.push(entry_dist, entry_point);
            if (top_candidates.size > SearchEF) top_candidates.pop();
        }

        lowerBound = (top_candidates.size > 0) ? top_candidates.top().dist : FLT_MAX;

        // Pre-clear so the first __syncthreads is well-defined
        s_num_edges[warp_in_blk] = 0;
    }
    __syncthreads();

    // =========================================================
    // Phase 2 - Greedy search with parallel distance computation
    // =========================================================
    // Each iteration:
    //  a) Lane 0 checks termination; calls SelectEdge; writes s_edges / s_num_edges.
    //     s_num_edges == -1 signals "stop".
    //  b) __syncthreads() — safe because only 1 query per block
    //  c) All lanes break if s_num_edges == -1.
    //  d) Groups of DIST_THREADS_PER_NEIGHBOR compute distances cooperatively.
    //  e) __syncthreads()
    //  f) Lane 0 reads batched s_dists[], marks visited, inserts into heaps.

    while (true) {
        // --- (a) Lane 0: advance one greedy step ---
        if (lane_id == 0) {
            if (candidate_set.empty()) {
                s_num_edges[warp_in_blk] = -1;            // stop: exhausted
            } else {
                HeapNode current = candidate_set.top();
                hop_count++;
                if (current.dist > lowerBound) {
                    s_num_edges[warp_in_blk] = -1;        // stop: below bound
                } else {
                    candidate_set.pop();
                    // SelectEdge writes directly into shared memory (at most NEIGHBORS_PER_BATCH=32)
                    SelectEdge_gpu(current.id, ql, qr, NEIGHBORS_PER_BATCH,
                                   gpu_index.d_segment_tree.d_nodes, 0,
                                   gpu_index.d_data_memory,
                                   gpu_index.d_size_data_per_element,
                                   size_links_per_layer,
                                   visited, query_id,
                                   s_edges[warp_in_blk], &s_num_edges[warp_in_blk]);
                }
            }
        }

        // --- (b) Sync: all 128 threads can now read s_num_edges / s_edges ---
        __syncthreads();

        // --- (c) Termination check (all lanes) ---
        if (s_num_edges[warp_in_blk] == -1) break;

        const int num_edges = s_num_edges[warp_in_blk];

        // --- (d/e/f) Process edges in batches so mode=1 and mode=4 both work ---
        for (int edge_base = 0; edge_base < num_edges; edge_base += NEIGHBORS_PER_BATCH) {
            int edges_in_batch = num_edges - edge_base;
            if (edges_in_batch > NEIGHBORS_PER_BATCH) edges_in_batch = NEIGHBORS_PER_BATCH;

            const int neighbor_slot   = lane_id / DIST_THREADS_PER_NEIGHBOR;
            const int lane_in_group   = lane_id % DIST_THREADS_PER_NEIGHBOR;

            if (neighbor_slot < edges_in_batch) {
                int neighbor_idx = edge_base + neighbor_slot;
                int neighbor_id  = s_edges[warp_in_blk][neighbor_idx];

                // PQ kernel: always compute PQ distance (no L2 fallback)
                float partial = gpu_compute_pq_distance(
                    query_vector, gpu_index.d_compressed_codes, gpu_index.d_centroids,
                    neighbor_id, gpu_index.pq_M, gpu_index.pq_nbits,
                    gpu_index.pq_dsub, gpu_index.pq_code_size, gpu_index.pq_ksub);
                
                // PQ distance is the same for all lanes in group; keep lane 0's result, zero others
                if (lane_in_group != 0) {
                    partial = 0.0f;
                }

                for (int offset = DIST_THREADS_PER_NEIGHBOR / 2; offset > 0; offset >>= 1) {
                    partial += __shfl_down_sync(0xffffffff, partial, offset, DIST_THREADS_PER_NEIGHBOR);
                }

                if (lane_in_group == 0) {
                    s_dists[warp_in_blk][neighbor_slot] = partial;
                }
            }

            // Sync so lane 0 sees all distance results for this batch
            __syncthreads();

            if (lane_id == 0) {
                for (int i = 0; i < edges_in_batch; i++) {
                    int   neighbor_id   = s_edges[warp_in_blk][edge_base + i];
                    float neighbor_dist = s_dists[warp_in_blk][i];

                    markVisited(visited, query_id, neighbor_id);
                    dist_comp_count++;

                    if (top_candidates.size < SearchEF) {
                        // Result set not full, always add
                        candidate_set.push(neighbor_dist, neighbor_id);
                        top_candidates.push(neighbor_dist, neighbor_id);
                        lowerBound = top_candidates.top().dist;
                    } else if (neighbor_dist < lowerBound) {
                        // Result set full but neighbor is better than worst result
                        candidate_set.push(neighbor_dist, neighbor_id);
                        top_candidates.push(neighbor_dist, neighbor_id);  // Add first
                        top_candidates.pop();                            // Then remove worst
                        lowerBound = top_candidates.top().dist;
                    }
                }
            }

            // Ensure lane 0 finished reading batch distances before overwrite next batch
            __syncthreads();
        }
    }

    // =========================================================
    // Phase 3 - Write results (lane 0 only)
    // =========================================================
    if (lane_id == 0) {
        while (top_candidates.size > query_K) top_candidates.pop();

        int* result_ptr = gpu_index.d_results + (long long)query_id * query_K;
        for (int i = 0; i < top_candidates.size; i++)
            result_ptr[i] = top_candidates.data[i].id;
        for (int i = top_candidates.size; i < query_K; i++)
            result_ptr[i] = -1;

        d_hops[query_id]       = hop_count;
        d_dist_comps[query_id] = dist_comp_count;
    }
}

// ============================================================================
// Normal Vector Kernel: Full-precision L2 distance (no branching)
// ============================================================================
__global__ void irange_search_kernel_normal(
    GPUIndex gpu_index,
    GPUVisitedArray visited,
    int query_nb,
    int SearchEF,
    int query_K,
    int dim,
    int suffix_id,           // Which range to use
    int* d_hops,             // Output: hops per query
    int* d_dist_comps,       // Output: distance computations per query
    size_t size_links_per_layer,
    unsigned long long seed
) {
    // --- Thread identity ---
    const int lane_id           = threadIdx.x % THREADS_PER_QUERY;
    const int warp_in_blk       = threadIdx.x / THREADS_PER_QUERY;  // always 0 (1 query/block)
    const int queries_per_block = blockDim.x  / THREADS_PER_QUERY;
    const int query_id          = blockIdx.x  * queries_per_block + warp_in_blk;

    if (query_id >= query_nb) return;

    // --- Shared memory: one slot per query in the block (MAX_QUERIES_PER_BLOCK=1) ---
    __shared__ int   s_edges    [MAX_QUERIES_PER_BLOCK][NEIGHBORS_PER_BATCH];
    __shared__ float s_dists    [MAX_QUERIES_PER_BLOCK][NEIGHBORS_PER_BATCH];
    __shared__ int   s_num_edges[MAX_QUERIES_PER_BLOCK];

    // --- Per-query values readable by all lanes ---
    float* query_vector = gpu_index.d_query_vectors + (long long)query_id * dim;
    int    range_idx    = query_id * 2 * gpu_index.num_suffixes + suffix_id * 2;
    int    ql           = gpu_index.d_query_range[range_idx];
    int    qr           = gpu_index.d_query_range[range_idx + 1];

    // --- Heap storage in per-thread local memory (only lane 0 uses it) ---
    HeapNode candidate_buffer    [MAX_SEARCH_EF];
    HeapNode top_candidate_buffer[MAX_SEARCH_EF];
    MinHeap  candidate_set;
    MaxHeap  top_candidates;
    float    lowerBound     = FLT_MAX;
    int      hop_count      = 0;
    int      dist_comp_count = 0;

    // =========================================================
    // Phase 1 - Entry-point seeding (lane 0 only)
    // =========================================================
    if (lane_id == 0) {
        candidate_set.init(candidate_buffer,     MAX_SEARCH_EF);
        top_candidates.init(top_candidate_buffer, SearchEF);

        unsigned int rng_state = (unsigned int)(seed + query_id);

        int filtered_indices[100];
        int num_filtered = range_filter_gpu(gpu_index.d_segment_tree.d_nodes, 0, ql, qr,
                                            filtered_indices, 100);

        for (int f = 0; f < num_filtered; f++) {
            GPUNode seg_node  = gpu_index.d_segment_tree.d_nodes[filtered_indices[f]];
            int range_size    = seg_node.rbound - seg_node.lbound + 1;
            int entry_point   = seg_node.lbound + (xorshift32(&rng_state) % range_size);

            if (isVisited(visited, query_id, entry_point)) continue;
            markVisited(visited, query_id, entry_point);

            // Normal kernel: always use L2 distance
            float entry_dist = L2Distance(
                query_vector,
                getVectorByID(entry_point, gpu_index.d_data_memory,
                              gpu_index.d_size_data_per_element, gpu_index.d_offsetData),
                dim);
            dist_comp_count++;

            candidate_set.push(entry_dist, entry_point);
            top_candidates.push(entry_dist, entry_point);
            if (top_candidates.size > SearchEF) top_candidates.pop();
        }

        lowerBound = (top_candidates.size > 0) ? top_candidates.top().dist : FLT_MAX;

        // Pre-clear so the first __syncthreads is well-defined
        s_num_edges[warp_in_blk] = 0;
    }
    __syncthreads();

    // =========================================================
    // Phase 2 - Greedy search with parallel distance computation
    // =========================================================
    while (true) {
        // --- (a) Lane 0: advance one greedy step ---
        if (lane_id == 0) {
            if (candidate_set.empty()) {
                s_num_edges[warp_in_blk] = -1;            // stop: exhausted
            } else {
                HeapNode current = candidate_set.top();
                hop_count++;
                if (current.dist > lowerBound) {
                    s_num_edges[warp_in_blk] = -1;        // stop: below bound
                } else {
                    candidate_set.pop();
                    SelectEdge_gpu(current.id, ql, qr, NEIGHBORS_PER_BATCH,
                                   gpu_index.d_segment_tree.d_nodes, 0,
                                   gpu_index.d_data_memory,
                                   gpu_index.d_size_data_per_element,
                                   size_links_per_layer,
                                   visited, query_id,
                                   s_edges[warp_in_blk], &s_num_edges[warp_in_blk]);
                }
            }
        }

        // --- (b) Sync: all 128 threads can now read s_num_edges / s_edges ---
        __syncthreads();

        // --- (c) Termination check (all lanes) ---
        if (s_num_edges[warp_in_blk] == -1) break;

        const int num_edges = s_num_edges[warp_in_blk];

        // --- (d/e/f) Process edges in batches ---
        for (int edge_base = 0; edge_base < num_edges; edge_base += NEIGHBORS_PER_BATCH) {
            int edges_in_batch = num_edges - edge_base;
            if (edges_in_batch > NEIGHBORS_PER_BATCH) edges_in_batch = NEIGHBORS_PER_BATCH;

            const int neighbor_slot   = lane_id / DIST_THREADS_PER_NEIGHBOR;
            const int lane_in_group   = lane_id % DIST_THREADS_PER_NEIGHBOR;

            if (neighbor_slot < edges_in_batch) {
                int neighbor_idx = edge_base + neighbor_slot;
                int neighbor_id  = s_edges[warp_in_blk][neighbor_idx];

                // Normal kernel: always compute L2 distance
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

            if (lane_id == 0) {
                for (int i = 0; i < edges_in_batch; i++) {
                    int   neighbor_id   = s_edges[warp_in_blk][edge_base + i];
                    float neighbor_dist = s_dists[warp_in_blk][i];

                    markVisited(visited, query_id, neighbor_id);
                    dist_comp_count++;

                    if (top_candidates.size < SearchEF) {
                        // Result set not full, always add
                        candidate_set.push(neighbor_dist, neighbor_id);
                        top_candidates.push(neighbor_dist, neighbor_id);
                        lowerBound = top_candidates.top().dist;
                    } else if (neighbor_dist < lowerBound) {
                        // Result set full but neighbor is better than worst result
                        candidate_set.push(neighbor_dist, neighbor_id);
                        top_candidates.push(neighbor_dist, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().dist;
                    }
                }
            }

            // Ensure lane 0 finished reading batch distances before overwrite next batch
            __syncthreads();
        }
    }

    // =========================================================
    // Phase 3 - Write results (lane 0 only)
    // =========================================================
    if (lane_id == 0) {
        while (top_candidates.size > query_K) top_candidates.pop();

        int* result_ptr = gpu_index.d_results + (long long)query_id * query_K;
        for (int i = 0; i < top_candidates.size; i++)
            result_ptr[i] = top_candidates.data[i].id;
        for (int i = top_candidates.size; i < query_K; i++)
            result_ptr[i] = -1;

        d_hops[query_id]       = hop_count;
        d_dist_comps[query_id] = dist_comp_count;
    }
}
