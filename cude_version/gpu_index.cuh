#pragma once

struct GPUNode {

    // GPU-compatible fields (populated when copied to GPU)
    int lbound;
    int rbound;
    int depth;
    int node_id;
    int left_child_index;  // Index in the GPU array for left child
    int right_child_index; // Index in the GPU array for right child
    bool is_leaf;         // Whether this node is a leaf node

   
    // Method to free GPU memory
    void free();
};

struct GPUSegmentTree {
    // GPU device pointer to the array of GPUNodes
    GPUNode* d_nodes;
    GPUNode root;
    int max_depth;
    
    // Size of the tree (number of nodes)
    size_t num_nodes;
    
    // Method to free GPU memory
    void free();
};

struct GPUIndex {
    // Metadata (copied from CPU)
    size_t d_dim;
    
    // GPU device pointers
    char* d_data_memory;
    
    // Sizes for memory management
    size_t d_size_data_per_element;
    size_t d_size_links_per_element;
    size_t d_offsetData;

    GPUSegmentTree d_segment_tree;

    float* d_query_vectors;
    int* d_query_range;

    int* d_results;
    size_t d_num_results;
    
    // Methods
    void free();
};

void GPUIndex::free() {
    if (d_data_memory) {
        cudaFree(d_data_memory);
        d_data_memory = nullptr;
    }
    if (d_segment_tree.d_nodes) {
        cudaFree(d_segment_tree.d_nodes);
        d_segment_tree.d_nodes = nullptr;
    }
}
