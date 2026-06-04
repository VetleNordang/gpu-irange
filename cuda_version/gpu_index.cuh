#pragma once

#include <vector>

struct GPUNode {
    int lbound;
    int rbound;
    int depth;
    int node_id;
    int left_child_index;
    int right_child_index;
    bool is_leaf;

    void free();
};

struct GPUSegmentTree {
    GPUNode* d_nodes;
    GPUNode root;
    int max_depth;
    size_t num_nodes;

    void free();
};

struct GPUIndex {
    // Metadata (copied from CPU)
    size_t d_dim;

    // GPU device pointers
    char* d_data_memory;

    // Compact adjacency lists (for PQ mode - no padding, only graph links)
    char* d_adjacency_lists;
    size_t d_adjacency_lists_size;

    // Sizes for memory management
    size_t d_size_data_per_element;
    size_t d_size_links_per_element;
    size_t d_offsetData;

    GPUSegmentTree d_segment_tree;

    float* d_query_vectors;
    int* d_query_range;

    int* d_results;
    size_t d_num_results;

    void free();
};

void GPUIndex::free() {
    if (d_data_memory) {
        cudaFree(d_data_memory);
        d_data_memory = nullptr;
    }
    if (d_adjacency_lists) {
        cudaFree(d_adjacency_lists);
        d_adjacency_lists = nullptr;
    }
    if (d_segment_tree.d_nodes) {
        cudaFree(d_segment_tree.d_nodes);
        d_segment_tree.d_nodes = nullptr;
    }
    if (d_query_vectors) {
        cudaFree(d_query_vectors);
        d_query_vectors = nullptr;
    }
    if (d_query_range) {
        cudaFree(d_query_range);
        d_query_range = nullptr;
    }
    if (d_results) {
        cudaFree(d_results);
        d_results = nullptr;
    }
}
