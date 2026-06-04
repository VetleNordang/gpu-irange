#pragma once

#include "gpu_search_common.cuh"
#include "gpu_pq_index.cuh"
#include "gpu_pq_distance.cuh"


// ============ 128-thread PQ Search Kernel ============
// Same architecture as normal kernel but with PQ distances
#undef  THREADS_PER_QUERY
#define THREADS_PER_QUERY     128
#define MAX_QUERIES_PER_BLOCK   1

#ifndef DIST_THREADS_PER_NEIGHBOR
#define DIST_THREADS_PER_NEIGHBOR 4
#endif

#define NEIGHBORS_PER_BATCH (THREADS_PER_QUERY / DIST_THREADS_PER_NEIGHBOR)

__global__ void irange_search_kernel_pq(
    GPUIndex gpu_index,
    GPUPQParams pq,
    GPUVisitedArray visited,
    int query_nb,
    int SearchEF,
    int query_K,
    int dim,
    int suffix_id,
    int num_suffixes,
    int* d_hops,
    int* d_dist_comps,
    size_t size_links_per_layer,
    unsigned long long seed,
    HeapNode* d_heap_buf
) {
    // --- Thread identity ---
    const int lane_id           = threadIdx.x % THREADS_PER_QUERY;
    const int warp_in_blk       = threadIdx.x / THREADS_PER_QUERY;
    const int queries_per_block = blockDim.x  / THREADS_PER_QUERY;
    const int query_id          = blockIdx.x  * queries_per_block + warp_in_blk;

    if (query_id >= query_nb) return;

    // --- Shared memory ---
    __shared__ int   s_edges      [MAX_QUERIES_PER_BLOCK][NEIGHBORS_PER_BATCH];
    __shared__ float s_dists      [MAX_QUERIES_PER_BLOCK][NEIGHBORS_PER_BATCH];
    __shared__ int   s_num_edges  [MAX_QUERIES_PER_BLOCK];
    __shared__ int   s_current_id [MAX_QUERIES_PER_BLOCK];

    // --- Per-query values ---
    float* query_vector = gpu_index.d_query_vectors + (long long)query_id * dim;
    int    range_idx    = query_id * num_suffixes * 2 + suffix_id * 2;
    int    ql           = gpu_index.d_query_range[range_idx];
    int    qr           = gpu_index.d_query_range[range_idx + 1];

    // --- ADC distance table build (all 128 threads cooperate) ---
    float* my_dist_table = pq.d_dist_tables
                         + (long long)query_id * pq.pq_M * pq.pq_ksub;

    build_adc_table(
        query_vector,
        pq.d_centroids,
        my_dist_table,
        pq.pq_M, pq.pq_ksub, pq.pq_dsub,
        threadIdx.x, blockDim.x);

    __syncthreads();  // all threads must finish building the table before any thread reads it

    // --- Heap storage (lane 0 only) — lives in global memory to avoid 32 KB thread-stack usage ---
    HeapNode* candidate_buffer     = d_heap_buf + (long long)query_id * 2 * MAX_SEARCH_EF;
    HeapNode* top_candidate_buffer = candidate_buffer + MAX_SEARCH_EF;
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
        top_candidates.init(top_candidate_buffer, SearchEF + 1);

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

            // ADC table lookup for entry point
            const uint8_t* entry_code = pq.d_compressed_codes
                                      + (long long)entry_point * pq.pq_code_size;
            float entry_dist = adc_distance(
                entry_code, my_dist_table,
                pq.pq_M, pq.pq_nbits, pq.pq_ksub);
            
            dist_comp_count++;

            candidate_set.push(entry_dist, entry_point);
            top_candidates.push(entry_dist, entry_point);
            if (top_candidates.size > SearchEF) top_candidates.pop();
        }

        lowerBound = (top_candidates.size > 0) ? top_candidates.top().dist : FLT_MAX;
        s_num_edges[warp_in_blk] = 0;
    }
    __syncthreads();

    // =========================================================
    // Phase 2 - Greedy search with parallel distance computation
    // =========================================================
    while (true) {
        // --- Lane 0: pop candidate and decide whether to continue ---
        if (lane_id == 0) {
            if (candidate_set.empty()) {
                s_num_edges[warp_in_blk] = -1;
            } else {
                HeapNode current = candidate_set.top();
                hop_count++;
                if (current.dist > lowerBound || hop_count > 3 * SearchEF + 500) {
                    s_num_edges[warp_in_blk] = -1;
                } else {
                    candidate_set.pop();
                    s_current_id[warp_in_blk] = current.id;
                    s_num_edges[warp_in_blk]  = 0;
                }
            }
        }
        __syncthreads();

        // --- All threads cooperate on SelectEdge_gpu (it uses __syncthreads internally) ---
        if (s_num_edges[warp_in_blk] != -1) {
            SelectEdge_gpu(s_current_id[warp_in_blk], ql, qr, NEIGHBORS_PER_BATCH,
                           gpu_index.d_segment_tree.d_nodes, 0,
                           gpu_index.d_data_memory,
                           gpu_index.d_size_data_per_element,
                           size_links_per_layer,
                           visited, query_id,
                           s_edges[warp_in_blk], &s_num_edges[warp_in_blk],
                           lane_id, THREADS_PER_QUERY);
        }
        __syncthreads();

        if (s_num_edges[warp_in_blk] == -1) break;

        const int num_edges = s_num_edges[warp_in_blk];

        // --- Process edges in batches ---
        for (int edge_base = 0; edge_base < num_edges; edge_base += NEIGHBORS_PER_BATCH) {
            int edges_in_batch = num_edges - edge_base;
            if (edges_in_batch > NEIGHBORS_PER_BATCH) edges_in_batch = NEIGHBORS_PER_BATCH;

            const int neighbor_slot   = lane_id / DIST_THREADS_PER_NEIGHBOR;
            const int lane_in_group   = lane_id % DIST_THREADS_PER_NEIGHBOR;

            float partial = 0.0f;
            if (neighbor_slot < edges_in_batch) {
                int neighbor_idx = edge_base + neighbor_slot;
                int neighbor_id  = s_edges[warp_in_blk][neighbor_idx];

                const uint8_t* nb_code = pq.d_compressed_codes
                                       + (long long)neighbor_id * pq.pq_code_size;
                partial = adc_distance_partial(
                    nb_code, my_dist_table,
                    pq.pq_M, pq.pq_nbits, pq.pq_ksub,
                    lane_in_group, DIST_THREADS_PER_NEIGHBOR);
            }

            // All threads participate — inactive threads contribute 0.0f.
            // Volta (V100) ITS requires all masked threads reach __shfl_sync;
            // placing this inside the if-block deadlocks when the batch is partial.
            for (int offset = DIST_THREADS_PER_NEIGHBOR / 2; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset, DIST_THREADS_PER_NEIGHBOR);
            }

            if (neighbor_slot < edges_in_batch && lane_in_group == 0) {
                s_dists[warp_in_blk][neighbor_slot] = partial;
            }

            __syncthreads();

            if (lane_id == 0) {
                for (int i = 0; i < edges_in_batch; i++) {
                    int   neighbor_id   = s_edges[warp_in_blk][edge_base + i];
                    float neighbor_dist = s_dists[warp_in_blk][i];

                    markVisited(visited, query_id, neighbor_id);
                    dist_comp_count++;

                    if (top_candidates.size < SearchEF) {
                        candidate_set.push(neighbor_dist, neighbor_id);
                        top_candidates.push(neighbor_dist, neighbor_id);
                        lowerBound = top_candidates.top().dist;
                    } else if (neighbor_dist < lowerBound) {
                        candidate_set.push(neighbor_dist, neighbor_id);
                        top_candidates.push(neighbor_dist, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().dist;
                    }
                }
            }

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
