#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include "gpu_index.cuh"
#include "gpu_visited.cuh"
#include "gpu_heap.cuh"

// Shared constants
#define MAX_SEARCH_EF 2000

// ============ GPU Range Filter ============
// Filter segment tree nodes that overlap with query range [ql, qr]
__device__ int range_filter_gpu(GPUNode* d_nodes, int root_idx, int ql, int qr,
                                int* output_indices, int max_output) {
    int stack[50];  // max tree depth ~20
    int stack_top = 0;
    int output_count = 0;

    stack[stack_top++] = root_idx;

    while (stack_top > 0 && output_count < max_output) {
        int current_idx = stack[--stack_top];
        GPUNode current = d_nodes[current_idx];

        if (current.lbound >= ql && current.rbound <= qr) {
            output_indices[output_count++] = current_idx;
            continue;
        }

        if (current.lbound > qr || current.rbound < ql)
            continue;

        if (!current.is_leaf) {
            if (current.right_child_index != -1 && stack_top < 50)
                stack[stack_top++] = current.right_child_index;
            if (current.left_child_index != -1 && stack_top < 50)
                stack[stack_top++] = current.left_child_index;
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
            float dx = av.x - bv.x, dy = av.y - bv.y;
            float dz = av.z - bv.z, dw = av.w - bv.w;
            sum += dx*dx + dy*dy + dz*dz + dw*dw;
        }
        return sum;
    }

    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Partial L2 distance for cooperative groups — lane_in_group computes a strided
// subset and caller reduces across the group with warp shuffles.
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
            float dx = av.x - bv.x, dy = av.y - bv.y;
            float dz = av.z - bv.z, dw = av.w - bv.w;
            sum += dx*dx + dy*dy + dz*dz + dw*dw;
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

__device__ float* getVectorByID(int node_id, char* data_memory,
                                size_t size_data_per_element, size_t offsetData) {
    return (float*)(data_memory + node_id * size_data_per_element + offsetData);
}

// ============ Graph Navigation ============
__device__ int* get_linklist_gpu(int node_id, int layer, char* data_memory,
                                  size_t size_data_per_element, size_t size_links_per_layer,
                                  int* list_size_out) {
    char* linklist_ptr = data_memory + node_id * size_data_per_element + layer * size_links_per_layer;
    int* list_size_ptr = (int*)linklist_ptr;
    *list_size_out = *list_size_ptr;
    return list_size_ptr + 1;
}

__device__ int GetOverLap(int l, int r, int ql, int qr) {
    int L = (l > ql) ? l : ql;
    int R = (r < qr) ? r : qr;
    return R - L + 1;
}

// SelectEdge: Navigate segment tree to find edges collaboratively.
// Mirrors CPU SelectEdge: walk down from root, skipping levels where the child
// has the same overlap as the parent, then collect edges from the chosen level.
// Continues descending until the node is fully within [ql, qr].
__device__ void SelectEdge_gpu(int pid, int ql, int qr, int edge_limit,
                                GPUNode* d_nodes, int root_idx,
                                char* data_memory, size_t size_data_per_element,
                                size_t size_links_per_layer,
                                GPUVisitedArray visited, int query_id,
                                int* output_edges, int* output_count, int lane_id,
                                int threads_per_query) {
    __shared__ int s_cur_idx;
    __shared__ int s_neighbor_count;
    __shared__ int* s_neighbors;
    __shared__ GPUNode s_cur_node;
    __shared__ bool s_done;

    if (lane_id == 0) {
        *output_count = 0;
        s_cur_idx = root_idx;
        s_done = false;
    }
    __syncthreads();

    while (!s_done) {
        if (lane_id == 0) {
            s_cur_node = d_nodes[s_cur_idx];

            bool contain = true;
            while (contain) {
                contain = false;
                if (!s_cur_node.is_leaf) {
                    int nxt_idx = -1;
                    if (s_cur_node.left_child_index != -1) {
                        GPUNode left_child = d_nodes[s_cur_node.left_child_index];
                        if (left_child.lbound <= pid && left_child.rbound >= pid)
                            nxt_idx = s_cur_node.left_child_index;
                    }
                    if (nxt_idx == -1 && s_cur_node.right_child_index != -1) {
                        GPUNode right_child = d_nodes[s_cur_node.right_child_index];
                        if (right_child.lbound <= pid && right_child.rbound >= pid)
                            nxt_idx = s_cur_node.right_child_index;
                    }
                    if (nxt_idx != -1) {
                        GPUNode nxt_node = d_nodes[nxt_idx];
                        int cur_overlap = GetOverLap(s_cur_node.lbound, s_cur_node.rbound, ql, qr);
                        int nxt_overlap = GetOverLap(nxt_node.lbound, nxt_node.rbound, ql, qr);
                        if (cur_overlap == nxt_overlap) {
                            s_cur_idx = nxt_idx;
                            s_cur_node = nxt_node;
                            contain = true;
                        }
                    }
                }
            }

            s_neighbors = get_linklist_gpu(pid, s_cur_node.depth, data_memory,
                                           size_data_per_element, size_links_per_layer,
                                           &s_neighbor_count);
        }
        __syncthreads();

        for (int i = lane_id; i < s_neighbor_count; i += threads_per_query) {
            if (*output_count >= edge_limit) continue;
            int neighbor_id = s_neighbors[i];
            if (neighbor_id >= ql && neighbor_id <= qr) {
                if (!isVisited(visited, query_id, neighbor_id)) {
                    int pos = atomicAdd(output_count, 1);
                    if (pos < edge_limit)
                        output_edges[pos] = neighbor_id;
                    else
                        atomicSub(output_count, 1);
                }
            }
        }
        __syncthreads();

        if (lane_id == 0) {
            bool fully_contained = (s_cur_node.lbound >= ql && s_cur_node.rbound <= qr);
            bool no_child = s_cur_node.is_leaf;

            if (fully_contained || no_child || *output_count >= edge_limit) {
                s_done = true;
            } else {
                int nxt_idx = -1;
                if (s_cur_node.left_child_index != -1) {
                    GPUNode left_child = d_nodes[s_cur_node.left_child_index];
                    if (left_child.lbound <= pid && left_child.rbound >= pid)
                        nxt_idx = s_cur_node.left_child_index;
                }
                if (nxt_idx == -1 && s_cur_node.right_child_index != -1) {
                    GPUNode right_child = d_nodes[s_cur_node.right_child_index];
                    if (right_child.lbound <= pid && right_child.rbound >= pid)
                        nxt_idx = s_cur_node.right_child_index;
                }
                s_cur_idx = (nxt_idx == -1) ? (s_done = true, s_cur_idx) : nxt_idx;
            }
        }
        __syncthreads();
    }
}

// Simple xorshift RNG for GPU
__device__ unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}
