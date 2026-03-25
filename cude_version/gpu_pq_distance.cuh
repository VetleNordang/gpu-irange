#pragma once

#include <cuda_runtime.h>

// ============ Bit Extraction and Centroid Index ============

__device__ uint32_t gpu_extract_centroid_index(const uint8_t* compressed_code, int subspace_index, int nbits, int code_size, int vector_id) {
    // Calculate the bit position for the desired subspace
    int bit_pos = subspace_index * nbits;
    uint32_t result = 0;

    for (int i = 0; i < nbits; ++i) {
        int byte_index = (bit_pos + i) / 8;
        int bit_index = (bit_pos + i) % 8;
        uint8_t byte_value = compressed_code[vector_id * code_size + byte_index];
        uint8_t bit_value = (byte_value >> bit_index) & 1;
        result |= (bit_value << i);
    }
    
    return result;
}

// Direct version when code array is already positioned
__device__ uint32_t gpu_extract_centroid_index_direct(const uint8_t* code, int subspace_index, int nbits) {
    uint32_t result = 0;
    int bit_pos = subspace_index * nbits;

    for (int i = 0; i < nbits; ++i) {
        int byte_index = (bit_pos + i) / 8;
        int bit_index = (bit_pos + i) % 8;
        uint8_t bit_value = (code[byte_index] >> bit_index) & 1;
        result |= (bit_value << i);
    }
    
    return result;
}

// ============ Distance Computation ============

__device__ float gpu_compute_pq_distance(const float* query_vector, const uint8_t* db_codes, const float* centroids,
                                         int data_id, int M,              // number of subspaces
                                         int nbits, int dsub,           // dimensions per subspace
                                         int code_size, int ksub)           // centroids per subspace (2^nbits)
{

    float distance = 0.0f;

    for (int m = 0; m < M; ++m) {
        // Extract the centroid index for this subspace
        uint32_t centroid_index = gpu_extract_centroid_index(db_codes, m, nbits, code_size, data_id);
        
        // Load the corresponding centroid from global memory
        const float* centroid = &centroids[m * ksub * dsub + centroid_index * dsub];
        
        // Compute the squared L2 distance for this subspace
        for (int d = 0; d < dsub; ++d) {
            float diff = query_vector[m * dsub + d] - centroid[d];
            distance += diff * diff;
        }
    }
    
    return distance;
}

// Version with code pointer already at the right position
__device__ float gpu_compute_pq_distance_direct(
    const float* query_vector,
    const uint8_t* db_code,                      // Code for single vector (not array)
    const float* centroids,
    int M, int nbits, int dsub, int ksub) {
    
    float distance_sq = 0.0f;

    for (int m = 0; m < M; ++m) {
        uint32_t centroid_index = gpu_extract_centroid_index_direct(db_code, m, nbits);
        
        const float* centroid = &centroids[m * ksub * dsub + centroid_index * dsub];
        
        for (int d = 0; d < dsub; ++d) {
            float diff = query_vector[m * dsub + d] - centroid[d];
            distance_sq += diff * diff;
        }
    }
    
    return distance_sq;
}

// Batch compute: optimized for multiple queries
__device__ float gpu_compute_pq_distance_batch(
    const float* query_vector,              // Single query (n*d floats)
    const uint8_t* all_db_codes,           // All DB codes
    const float* centroids,                 // All centroids
    int db_id,                              // Which database vector
    int M, int nbits, int dsub, int code_size, int ksub) {
    
    float distance_sq = 0.0f;
    const uint8_t* db_code = all_db_codes + db_id * code_size;

    for (int m = 0; m < M; ++m) {
        uint32_t centroid_index = gpu_extract_centroid_index_direct(db_code, m, nbits);
        const float* centroid = &centroids[m * ksub * dsub + centroid_index * dsub];
        const float* query_m = query_vector + m * dsub;
        
        for (int d = 0; d < dsub; ++d) {
            float diff = query_m[d] - centroid[d];
            distance_sq += diff * diff;
        }
    }
    
    return distance_sq;
}
