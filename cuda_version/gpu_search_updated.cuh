#pragma once

#include "gpu_search_common.cuh"

#ifndef DIST_THREADS_PER_NEIGHBOR
#define DIST_THREADS_PER_NEIGHBOR 16
#endif

#define THREADS_PER_QUERY     512
#define MAX_QUERIES_PER_BLOCK   1

#if (THREADS_PER_QUERY % DIST_THREADS_PER_NEIGHBOR) != 0
#error "DIST_THREADS_PER_NEIGHBOR must divide THREADS_PER_QUERY"
#endif

#define NEIGHBORS_PER_BATCH (THREADS_PER_QUERY / DIST_THREADS_PER_NEIGHBOR)

__global__ void irange_search_kernel(
    GPUIndex gpu_index,
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
    int*   d_entry_ids,
    float* d_entry_dists,
    int*   d_entry_counts
) {
    const int lane_id  = threadIdx.x;
    const int query_id = blockIdx.x;

    if (query_id >= query_nb) return;

    __shared__ int   s_edges [32];
    __shared__ float s_dists [32];
    __shared__ int   s_num_edges;

    float* query_vector = gpu_index.d_query_vectors + (long long)query_id * dim;
    int    range_idx    = query_id * num_suffixes * 2 + suffix_id * 2;
    int    ql           = gpu_index.d_query_range[range_idx];
    int    qr           = gpu_index.d_query_range[range_idx + 1];

    __shared__ HeapNode s_candidate_buffer[MAX_QUERIES_PER_BLOCK * MAX_SEARCH_EF];
    __shared__ HeapNode s_top_candidate_buffer[MAX_QUERIES_PER_BLOCK * MAX_SEARCH_EF];

    MinHeap candidate_set;
    MaxHeap top_candidates;
    float   lowerBound     = FLT_MAX;
    int     hop_count      = 0;
    int     dist_comp_count = 0;

    // Phase 1 — entry-point seeding from CPU pre-computed list (lane 0 only)
    if (lane_id == 0) {
        candidate_set.init(&s_candidate_buffer[0], MAX_SEARCH_EF);
        top_candidates.init(&s_top_candidate_buffer[0], SearchEF + 1);

        int num_entries = d_entry_counts[query_id];
        int base        = query_id * 100;
        for (int f = 0; f < num_entries; f++) {
            float entry_dist  = d_entry_dists[base + f];
            int   entry_point = d_entry_ids  [base + f];
            markVisited(visited, query_id, entry_point);
            candidate_set.push(entry_dist, entry_point);
            top_candidates.push(entry_dist, entry_point);
            if (top_candidates.size > SearchEF) top_candidates.pop();
        }

        lowerBound  = (top_candidates.size > 0) ? top_candidates.top().dist : FLT_MAX;
        s_num_edges = 0;
    }
    __syncthreads();

    // Phase 2 — greedy search with parallel distance computation
    while (true) {
        __shared__ int s_current_id;

        if (lane_id == 0) {
            if (candidate_set.empty()) {
                s_num_edges = -1;
            } else {
                HeapNode current = candidate_set.top();
                hop_count++;
                if (current.dist > lowerBound || hop_count > 3 * SearchEF + 500) {
                    s_num_edges = -1;
                } else {
                    candidate_set.pop();
                    s_current_id = current.id;
                    s_num_edges  = 0;
                }
            }
        }
        __syncthreads();

        if (s_num_edges != -1) {
            SelectEdge_gpu(s_current_id, ql, qr, 32,
                           gpu_index.d_segment_tree.d_nodes, 0,
                           gpu_index.d_data_memory,
                           gpu_index.d_size_data_per_element,
                           size_links_per_layer,
                           visited, query_id,
                           s_edges, &s_num_edges, lane_id,
                           THREADS_PER_QUERY);
        }
        __syncthreads();

        if (s_num_edges == -1) break;

        const int num_edges = s_num_edges;

        for (int edge_base = 0; edge_base < num_edges; edge_base += NEIGHBORS_PER_BATCH) {
            int edges_in_batch = num_edges - edge_base;
            if (edges_in_batch > NEIGHBORS_PER_BATCH) edges_in_batch = NEIGHBORS_PER_BATCH;

            const int neighbor_slot  = lane_id / DIST_THREADS_PER_NEIGHBOR;
            const int lane_in_group  = lane_id % DIST_THREADS_PER_NEIGHBOR;

            const int neighbors_per_warp = 32 / DIST_THREADS_PER_NEIGHBOR;
            const int warp_id_in_query   = lane_id / 32;
            const int remaining          = edges_in_batch - warp_id_in_query * neighbors_per_warp;
            const unsigned int active_mask =
                (remaining <= 0)                  ? 0u :
                (remaining >= neighbors_per_warp) ? 0xffffffffu :
                                                    (1u << (remaining * DIST_THREADS_PER_NEIGHBOR)) - 1u;

            if (neighbor_slot < edges_in_batch) {
                int neighbor_id = s_edges[edge_base + neighbor_slot];
                float partial = L2DistancePartial(
                    query_vector,
                    getVectorByID(neighbor_id, gpu_index.d_data_memory,
                                  gpu_index.d_size_data_per_element, gpu_index.d_offsetData),
                    dim, lane_in_group, DIST_THREADS_PER_NEIGHBOR);

                for (int offset = DIST_THREADS_PER_NEIGHBOR / 2; offset > 0; offset >>= 1)
                    partial += __shfl_down_sync(active_mask, partial, offset, DIST_THREADS_PER_NEIGHBOR);

                if (lane_in_group == 0)
                    s_dists[neighbor_slot] = partial;
            }
            __syncthreads();

            if (lane_id == 0) {
                for (int i = 0; i < edges_in_batch; i++) {
                    int   neighbor_id   = s_edges[edge_base + i];
                    float neighbor_dist = s_dists[i];
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

    // Phase 3 — write results (lane 0 only)
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
