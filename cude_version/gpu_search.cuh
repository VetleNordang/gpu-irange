#pragma once

#include <cuda_runtime.h>
#include "utils.h"

// ============ Random Number Generation ============
__device__ unsigned int hash_random(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__device__ int random_in_range(unsigned int seed, int lbound, int rbound) {
    if (lbound >= rbound) return lbound;
    unsigned int h = hash_random(seed);
    return lbound + (h % (rbound - lbound + 1));
}

// ============ Distance Calculation ============
__device__ float L2Distance(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;  // Return squared distance (no sqrt for efficiency)
}

__device__ float L2DistanceSqrt(const float *a, const float *b, size_t dim) {
    return sqrtf(L2Distance(a, b, dim));
}

// Helper function to get vector data for a given PID
__device__ float* getVectorByPID(int pid, char* data_memory, size_t size_data_per_element, size_t offsetData) {
    char* data_ptr = data_memory + pid * size_data_per_element + offsetData;
    return (float*)data_ptr;
}

// Helper function to compute distance from query to a PID
__device__ float computeDistanceToPID(const float* query_vector, int pid, int dim, 
                                      char* data_memory, size_t size_data_per_element, size_t offsetData) {
    float* pid_vector = getVectorByPID(pid, data_memory, size_data_per_element, offsetData);
    return L2Distance(query_vector, pid_vector, dim);
}

// ============ Edge Selection ============

// Helper to get edge list size
__device__ int getListCount_gpu(int *ptr) {
    return *ptr;
}

// Helper to get overlap between two ranges
__device__ int getOverlap_gpu(int l, int r, int ql, int qr) {
    int L = (l > ql) ? l : ql;  // max(l, ql)
    int R = (r < qr) ? r : qr;  // min(r, qr)
    return R - L + 1;
}

// Get edge list for a PID at a specific layer
__device__ int* get_linklist_gpu(int pid, int layer, char* data_memory, size_t size_data_per_element, size_t size_links_per_layer) {
    return (int*)(data_memory + pid * size_data_per_element + layer * size_links_per_layer);
}

// Select edges for a given PID within query range [ql, qr]
__device__ int selectEdge_gpu(int pid, int ql, int qr, int edge_limit, 
                              iRangeGraph::SegmentTree *tree, char* visited,
                              char* data_memory, size_t size_data_per_element, size_t size_links_per_layer,
                              int* selected_edges, bool debug_nav = false) {
    iRangeGraph::TreeNode *cur_node = nullptr;
    iRangeGraph::TreeNode *nxt_node = tree->root;
    int num_selected = 0;
    int iteration = 0;
    
    // Navigate to the node containing pid
    do {
        cur_node = nxt_node;
        bool contain = false;
        
        if (debug_nav) {
            printf("[NAV] Iter %d: cur_node depth=%d, range=[%d,%d], childs=%d\n", 
                   iteration, cur_node->depth, cur_node->lbound, cur_node->rbound, cur_node->childs_size);
        }
        
        do {
            contain = false;
            if (cur_node->childs_size == 0) {
                nxt_node = nullptr;
                if (debug_nav) printf("  [NAV] Leaf node (no children)\n");
            } else {
                // Find child containing pid (use GPU-compatible childs_gpu array)
                nxt_node = nullptr;  // Reset before search
                for (int i = 0; i < cur_node->childs_size; ++i) {
                    if (cur_node->childs_gpu[i]->lbound <= pid && cur_node->childs_gpu[i]->rbound >= pid) {
                        nxt_node = cur_node->childs_gpu[i];
                        if (debug_nav) {
                            printf("  [NAV] Found child[%d]: depth=%d, range=[%d,%d]\n", 
                                   i, nxt_node->depth, nxt_node->lbound, nxt_node->rbound);
                        }
                        break;
                    }
                }
                
                if (nxt_node == nullptr) {
                    if (debug_nav) printf("  [NAV] ERROR: No child contains PID=%d!\n", pid);
                }
                
                // Check if we can skip to child (same overlap)
                if (nxt_node) {
                    int cur_overlap = getOverlap_gpu(cur_node->lbound, cur_node->rbound, ql, qr);
                    int nxt_overlap = getOverlap_gpu(nxt_node->lbound, nxt_node->rbound, ql, qr);
                    
                    if (debug_nav) {
                        printf("  [NAV] Overlap check: cur=%d, nxt=%d\n", cur_overlap, nxt_overlap);
                    }
                    
                    if (cur_overlap == nxt_overlap) {
                        cur_node = nxt_node;
                        contain = true;
                        if (debug_nav) printf("  [NAV] Same overlap, skipping to child\n");
                    }
                }
            }
        } while (contain);
        
        // Get edges from this layer
        int *data = get_linklist_gpu(pid, cur_node->depth, data_memory, size_data_per_element, size_links_per_layer);
        int size = getListCount_gpu(data);
        
        // Filter edges within range
        for (int j = 1; j <= size; ++j) {
            int neighborId = data[j];
            
            // Skip if out of range
            if (neighborId < ql || neighborId > qr)
                continue;
            
            // Skip if already visited
            if (visited[neighborId])
                continue;
            
            selected_edges[num_selected++] = neighborId;
            
            if (num_selected == edge_limit)
                return num_selected;
        }
        
        iteration++;
        
        if (debug_nav) {
            printf("  [NAV] After edges: num_selected=%d, nxt_node=%p\n", num_selected, nxt_node);
        }
        
        // Continue if we haven't reached a node that fully contains [ql, qr] AND we have a next node
    } while (nxt_node != nullptr && (cur_node->lbound < ql || cur_node->rbound > qr));
    
    if (debug_nav) {
        printf("[NAV] Done: total_selected=%d\n", num_selected);
    }
    
    return num_selected;
}

// ============ Top-Down Search Algorithm ============

__device__ void topDownSearch(
    int tid, int ql, int qr,
    iRangeGraph::TreeNode **filtered_nodes, int num_filtered,
    float* query_vector, int dim,
    char* visited, int max_elements,
    int SearchEF, int edge_limit,
    iRangeGraph::SegmentTree *tree,
    char* data_memory, size_t size_data_per_element, 
    size_t offsetData, size_t size_links_per_layer,
    int* results, int query_K,
    int* hops_out, int* dist_comps_out,
    bool debug) {
    
    // Allocate heap buffers (local arrays)
    HeapNode candidate_buffer[2000];
    HeapNode top_buffer[2000];
    
    MinHeap candidate_set;
    MaxHeap top_candidates;
    
    candidate_set.init(candidate_buffer, SearchEF);
    top_candidates.init(top_buffer, SearchEF);
    
    // Select random entry points from filtered nodes
    for (int i = 0; i < num_filtered; i++) {
        iRangeGraph::TreeNode *u = filtered_nodes[i];
        
        // Generate deterministic entry point (matching CPU groundtruth generation with seed=0)
        // Using seed=0 to match how groundtruth was generated
        unsigned int seed = 0 + i;  // Small variation per entry point
        int pid = random_in_range(seed, u->lbound, u->rbound);
        
        visited[pid] = 1;
        
        // Fetch vector and compute L2 distance
        float dist = computeDistanceToPID(query_vector, pid, dim, data_memory, size_data_per_element, offsetData);
        
        candidate_set.push(dist, pid);
        top_candidates.push(dist, pid);
    }
    
    float lowerBound = top_candidates.empty() ? 1e10f : top_candidates.top().dist;
    int hops = 0;
    int distance_computations = num_filtered;
    
    // Allocate buffer for selected edges
    int selected_edges[100];

    
    // Main search loop
    while (!candidate_set.empty()) {
        HeapNode current = candidate_set.top();
        hops++;
        
        if (current.dist > lowerBound) {
            break;
        }
        
        candidate_set.pop();
        int current_pid = current.id;
        
        // Select edges within query range
        bool debug_nav = false;  // Disable navigation debug output
        int num_edges = selectEdge_gpu(current_pid, ql, qr, edge_limit, tree, visited,
                                      data_memory, size_data_per_element, size_links_per_layer,
                                      selected_edges, debug_nav);
        
        if (debug && tid == 0 && hops <= 3) {
            printf("[HOP %d] PID=%d (dist=%.4f), found %d edges\n", hops, current_pid, current.dist, num_edges);
        }
        
        // Process each selected edge
        for (int i = 0; i < num_edges; ++i) {
            int neighbor_id = selected_edges[i];
            
            if (visited[neighbor_id])
                continue;
            
            visited[neighbor_id] = 1;
            
            // Compute distance to neighbor
            float dist = computeDistanceToPID(query_vector, neighbor_id, dim, 
                                             data_memory, size_data_per_element, offsetData);
            ++distance_computations;
            
            // Match CPU logic: only add if heap not full OR distance is better
            if (top_candidates.size < SearchEF) {
                candidate_set.push(dist, neighbor_id);
                top_candidates.push(dist, neighbor_id);
                lowerBound = top_candidates.top().dist;
            } else if (dist < lowerBound) {
                candidate_set.push(dist, neighbor_id);
                top_candidates.push(dist, neighbor_id);
                top_candidates.pop();
                lowerBound = top_candidates.top().dist;
            }
        }
        
        if (hops > SearchEF * 10) {  // Safety limit
            break;
        }
    }
    
    // Store metrics
    hops_out[tid] = hops;
    dist_comps_out[tid] = distance_computations;
    
    if (debug && tid < 3) {
        printf("[GPU DEBUG] Thread %d: Search complete - hops=%d, dist_comps=%d, top_candidates.size=%d\n", 
               tid, hops, distance_computations, top_candidates.size);
    }
    
    // Extract top-K results from MaxHeap
    // The MaxHeap has worst element on top. To get best K elements:
    // Pop until only K elements remain (those are the best K)
    while (top_candidates.size > query_K) {
        top_candidates.pop();
    }
    
    int result_count = top_candidates.size;
    
    // Now extract remaining elements (best to worst)
    // Since MaxHeap has worst on top, we extract in reverse to store best-to-worst
    for (int i = result_count - 1; i >= 0; --i) {
        if (!top_candidates.empty()) {
            results[tid * query_K + i] = top_candidates.top().id;
            top_candidates.pop();
        }
    }
    
    // Fill remaining slots with -1
    for (int i = result_count; i < query_K; ++i) {
        results[tid * query_K + i] = -1;
    }
    
}

// ============ GPU Search Kernel ============

__global__ void search_gpu(int *range_data, int num_ranges, int query_nb, int SearchEF, int edge_limit, 
                          iRangeGraph::SegmentTree *tree, char* visited_arrays, int max_elements,
                          float* query_vectors, int dim, char* data_memory, size_t size_data_per_element, 
                          size_t offsetData, size_t size_links_per_layer, 
                          int* results, int query_K, int* hops_out, int* dist_comps_out, bool debug = false) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Debug: Verify kernel is running
    if (tid == 0 && debug) {
        printf("[GPU DEBUG] Kernel started: query_nb=%d, SearchEF=%d, K=%d\n", query_nb, SearchEF, query_K);
    }
    
    // Each thread handles one query
    if (tid < query_nb) {
        // range_data is flattened: [ql0, qr0, ql1, qr1, ...]
        int ql = range_data[tid * 2];
        int qr = range_data[tid * 2 + 1];
        
        // Allocate temporary array for filtered nodes (max 100 nodes per query)
        iRangeGraph::TreeNode *filtered_nodes[100];
        int num_filtered = tree->range_filter_gpu(tree->root, ql, qr, filtered_nodes, 100);
    
        
        // Get this thread's visited array
        char* visited = visited_arrays + (long long)tid * max_elements;
        
        // Initialize visited array
        for (int i = 0; i < max_elements; i++) {
            visited[i] = 0;
        }
        
        // Get query vector
        float* query_vector = query_vectors + (long long)tid * dim;
        
        // Perform top-down search
        topDownSearch(tid, ql, qr, filtered_nodes, num_filtered,
                     query_vector, dim, visited, max_elements,
                     SearchEF, edge_limit, tree,
                     data_memory, size_data_per_element, offsetData, size_links_per_layer,
                     results, query_K, hops_out, dist_comps_out, debug);
    }
}  // End of search_gpu function


