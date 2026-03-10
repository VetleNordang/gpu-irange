#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include "gpu_index.cuh"
#include "gpu_visited.cuh"
#include "gpu_heap.cuh"

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
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;  // Squared distance
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
        
        // Filter neighbors by range and visited status
        for (int i = 0; i < neighbor_count && *output_count < edge_limit; i++) {
            int neighbor_id = neighbors[i];
            
            // Check if in range
            if (neighbor_id < ql || neighbor_id > qr) continue;
            
            // Check if visited
            if (isVisited(visited, query_id, neighbor_id)) continue;
            
            output_edges[(*output_count)++] = neighbor_id;
        }
        
        // Return if we've collected enough edges
        if (*output_count >= edge_limit) return;
        
        // Continue while cur_node's bounds are not completely within [ql, qr]
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

// ============ Main Search Kernel ============
__global__ void irange_search_kernel(
    GPUIndex gpu_index,
    GPUVisitedArray visited,
    int query_nb,
    int SearchEF,
    int query_K,
    int dim,
    int suffix_id,  // Which range to use (0-9, or 10 for suffix 17)
    int* d_hops,    // Output: hops per query
    int* d_dist_comps,  // Output: distance computations per query
    size_t size_links_per_layer,  // Size of one layer's link list
    unsigned long long seed  // Random seed
) {
    int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_id >= query_nb) return;
    
    // Initialize metrics for this query
    int hop_count = 0;
    int dist_comp_count = 0;
    
    // Get query vector
    float* query_vector = gpu_index.d_query_vectors + (long long)query_id * dim;
    
    // Get query range [ql, qr]
    int range_idx = query_id * 22 + suffix_id * 2;  // 11 suffixes × 2 values
    int ql = gpu_index.d_query_range[range_idx];
    int qr = gpu_index.d_query_range[range_idx + 1];
    
    // Filter segment tree to get nodes overlapping with [ql, qr]
    int filtered_indices[100];
    int num_filtered = range_filter_gpu(gpu_index.d_segment_tree.d_nodes, 0, ql, qr, 
                                       filtered_indices, 100);
    
    if (query_id == 0) {
        printf("Query %d Suffix %d: Range [%d, %d] -> %d filtered nodes\n", 
               query_id, suffix_id, ql, qr, num_filtered);
    }
    
    // Allocate heap buffers in local memory
    // Use MAX_SEARCH_EF as compile-time constant, but heap will only use SearchEF elements
    HeapNode candidate_buffer[MAX_SEARCH_EF];  // Buffer for exploration queue - NO LIMIT!
    HeapNode top_candidate_buffer[MAX_SEARCH_EF];  // Buffer for EF best candidates
    
    // Create candidate_set (min-heap) - exploration queue, can grow up to MAX_SEARCH_EF
    // This should be UNLIMITED (up to MAX) - like std::priority_queue in CPU version
    MinHeap candidate_set;
    candidate_set.init(candidate_buffer, MAX_SEARCH_EF);  // ✅ Use MAX, not SearchEF!
    
    // Create top_candidates (max-heap) - maintains only SearchEF best candidates
    // This is the one that should be limited to SearchEF
    MaxHeap top_candidates;
    top_candidates.init(top_candidate_buffer, SearchEF);  // ✅ Limited to SearchEF
    
    // Initialize random state for this query
    unsigned int rng_state = (unsigned int)(seed + query_id);
    
    // Add entry points from ALL filtered segments (like CPU version)
    for (int f = 0; f < num_filtered; f++) {
        int seg_idx = filtered_indices[f];
        GPUNode seg_node = gpu_index.d_segment_tree.d_nodes[seg_idx];
        
        // Pick random point within segment bounds [lbound, rbound]
        int range_size = seg_node.rbound - seg_node.lbound + 1;
        int random_offset = xorshift32(&rng_state) % range_size;
        int entry_point = seg_node.lbound + random_offset;
        
        // Skip if already visited
        if (isVisited(visited, query_id, entry_point)) continue;
        
        // Mark as visited
        markVisited(visited, query_id, entry_point);
        
        // Compute distance
        float entry_dist = L2Distance(query_vector, 
                                      getVectorByID(entry_point, gpu_index.d_data_memory,
                                                   gpu_index.d_size_data_per_element,
                                                   gpu_index.d_offsetData),
                                      dim);
        dist_comp_count++;
        
        // Add to both heaps (CPU does this unconditionally for entry points)
        candidate_set.push(entry_dist, entry_point);
        top_candidates.push(entry_dist, entry_point);
        
        // If top_candidates exceeds SearchEF, remove worst
        if (top_candidates.size > SearchEF) {
            top_candidates.pop();
        }
    }
    
    // Track lower bound from top_candidates (worst of EF best)
    float lowerBound = (top_candidates.size > 0) ? top_candidates.top().dist : FLT_MAX;
    
    // Greedy search - matches CPU version exactly
    while (!candidate_set.empty()) {
        // Get closest candidate from exploration queue (MinHeap)
        HeapNode current = candidate_set.top();
        
        hop_count++;  // Count BEFORE termination check (like CPU)
        
        // Early termination: if current candidate is farther than worst in top_candidates, stop
        if (current.dist > lowerBound) {
            break;
        }
        
        candidate_set.pop();  // Pop AFTER check passes
        
        // Explore neighbors using SelectEdge (depth-aware edge selection)
        int selected_edges[50];  // Buffer for edges (M=32, but allow more)
        int num_edges = 0;
        
        SelectEdge_gpu(current.id, ql, qr, 50,  // edge_limit
                      gpu_index.d_segment_tree.d_nodes, 0,  // root at index 0
                      gpu_index.d_data_memory, gpu_index.d_size_data_per_element,
                      size_links_per_layer,
                      visited, query_id,
                      selected_edges, &num_edges);
        
        for (int i = 0; i < num_edges; i++) {
            int neighbor_id = selected_edges[i];
            
            // Mark as visited (already checked in SelectEdge, but mark explicitly)
            markVisited(visited, query_id, neighbor_id);
            
            // Compute distance
            float neighbor_dist = L2Distance(query_vector,
                                            getVectorByID(neighbor_id, gpu_index.d_data_memory,
                                                        gpu_index.d_size_data_per_element,
                                                        gpu_index.d_offsetData),
                                            dim);
            dist_comp_count++;  // Count distance computation
            
            // Add to candidate_set and top_candidates (exactly like CPU version)
            if (top_candidates.size < SearchEF) {
                // Not at EF limit yet - add unconditionally
                candidate_set.push(neighbor_dist, neighbor_id);
                top_candidates.push(neighbor_dist, neighbor_id);
                lowerBound = top_candidates.top().dist;  // Update bound (worst of EF best)
            } else if (neighbor_dist < lowerBound) {
                // Better than worst - add then pop (like CPU: emplace + pop)
                candidate_set.push(neighbor_dist, neighbor_id);
                top_candidates.push(neighbor_dist, neighbor_id);  // Adds (size becomes SearchEF+1)
                top_candidates.pop();                              // Removes worst (size back to SearchEF)
                lowerBound = top_candidates.top().dist;           // Update bound
            }
        }
    }
    
    // Trim top_candidates down to query_K (like CPU version)
    while (top_candidates.size > query_K) {
        top_candidates.pop();  // Remove worst until we have K results
    }
    
    // Write results to global memory from top_candidates
    int* result_ptr = gpu_index.d_results + (long long)query_id * query_K;
    for (int i = 0; i < query_K && i < top_candidates.size; i++) {
        result_ptr[i] = top_candidates.data[i].id;
    }
    
    // Fill remaining with -1 if fewer than K results found
    for (int i = top_candidates.size; i < query_K; i++) {
        result_ptr[i] = -1;
    }
    
    // Write metrics
    d_hops[query_id] = hop_count;
    d_dist_comps[query_id] = dist_comp_count;
}
