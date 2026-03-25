#pragma once

#include <vector>
#include <faiss/impl/ProductQuantizer.h>

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

    // PQ-specific GPU pointers and metadata
    uint8_t* d_compressed_codes;     // GPU memory for compressed codes
    float* d_centroids;              // GPU memory for FAISS centroids
    int pq_code_size;                // Bytes per compressed vector
    int pq_M;                         // Number of subspaces
    int pq_nbits;                     // Bits per centroid
    int pq_dsub;                      // Dimensions per subspace
    int pq_ksub;                      // Number of centroids per subspace
    bool use_pq;                      // Flag: use PQ distance or raw L2?
    
    int num_suffixes;                 // Total number of suffixes for correct range indexing
    
    // CPU-side PQ data (owned by GPUIndex)
    void* pq_model;                   // Pointer to faiss::ProductQuantizer (CPU)
    std::vector<uint8_t> pq_codes_cpu; // PQ codes on CPU
    
    // Track GPU memory sizes for proper cleanup
    size_t gpu_codes_size;
    size_t gpu_centroids_size;
    
    // Methods
    void load_pq_to_gpu(void* pq_model, const std::vector<uint8_t>& codes);
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
    if (d_compressed_codes) {
        cudaFree(d_compressed_codes);
        d_compressed_codes = nullptr;
    }
    if (d_centroids) {
        cudaFree(d_centroids);
        d_centroids = nullptr;
    }
    pq_codes_cpu.clear();
    pq_codes_cpu.shrink_to_fit();
}

// Load PQ data to GPU - GPUIndex holds complete control
void GPUIndex::load_pq_to_gpu(void* pq_model_ptr, const std::vector<uint8_t>& codes) {
    // Cast to ProductQuantizer to access members
    faiss::ProductQuantizer* pq = static_cast<faiss::ProductQuantizer*>(pq_model_ptr);
    
    pq_model = pq_model_ptr;
    pq_codes_cpu = codes;  // Store copy on CPU
    
    // Set PQ metadata
    pq_M = pq->M;
    pq_dsub = pq->dsub;
    pq_ksub = pq->ksub;
    pq_nbits = pq->nbits;
    pq_code_size = pq->code_size;
    
    // Allocate GPU memory for codes
    gpu_codes_size = codes.size();
    cudaError_t err = cudaMalloc(&d_compressed_codes, gpu_codes_size);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate GPU memory for PQ codes: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy codes to GPU
    err = cudaMemcpy(d_compressed_codes, codes.data(), gpu_codes_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to copy PQ codes to GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_compressed_codes);
        d_compressed_codes = nullptr;
        return;
    }
    printf("✓ PQ codes transferred to GPU: %.2f MB\n", gpu_codes_size / (1024.0*1024.0));
    
    // Allocate GPU memory for centroids (M x ksub x dsub floats)
    gpu_centroids_size = (size_t)pq->M * pq->ksub * pq->dsub * sizeof(float);
    err = cudaMalloc(&d_centroids, gpu_centroids_size);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate GPU memory for PQ centroids: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy centroids per subspace
    for (int m = 0; m < pq->M; m++) {
        float* centroid_data = pq->centroids.data() + m * pq->ksub * pq->dsub;
        size_t centroid_size = (size_t)pq->ksub * pq->dsub * sizeof(float);
        err = cudaMemcpy((float*)d_centroids + m * pq->ksub * pq->dsub, 
                        centroid_data, centroid_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: Failed to copy centroid %d to GPU: %s\n", m, cudaGetErrorString(err));
            cudaFree(d_centroids);
            d_centroids = nullptr;
            return;
        }
    }
    printf("✓ PQ centroids transferred to GPU: %.2f MB\n", gpu_centroids_size / (1024.0*1024.0));
    
    use_pq = true;  // Enable PQ mode
    printf("✓ GPUIndex now owns and controls PQ data on GPU (M=%d, dsub=%d, ksub=%d)\n", pq_M, pq_dsub, pq_ksub);
}
