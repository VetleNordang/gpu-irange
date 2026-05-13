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

// ============ ADC (Asymmetric Distance Computation) ============
// All threads in the block cooperate to build a distance table once per query.
// Table layout: dist_table[m * ksub + k] = L2(query_subvector[m], centroid[m][k])
// Must be followed by __syncthreads() before the search loop begins.
__device__ void build_adc_table(
    const float* query_vector,
    const float* centroids,
    float* dist_table,
    int M, int ksub, int dsub,
    int thread_id, int num_threads)
{
    const int total_entries = M * ksub;
    for (int idx = thread_id; idx < total_entries; idx += num_threads) {
        int m = idx / ksub;
        int k = idx % ksub;
        const float* centroid = &centroids[(m * ksub + k) * dsub];
        const float* q_sub    = &query_vector[m * dsub];
        float d = 0.0f;
        for (int i = 0; i < dsub; i++) {
            float diff = q_sub[i] - centroid[i];
            d += diff * diff;
        }
        dist_table[idx] = d;
    }
}

// Single-thread ADC distance lookup — replaces PQDistance()
__device__ float adc_distance(
    const uint8_t* db_code,
    const float* dist_table,
    int M, int nbits, int ksub)
{
    float distance = 0.0f;
    for (int m = 0; m < M; m++) {
        uint32_t k = gpu_extract_centroid_index_direct(db_code, m, nbits);
        distance += dist_table[m * ksub + k];
    }
    return distance;
}

// Cooperative partial-sum ADC lookup — replaces PQDistancePartial()
__device__ float adc_distance_partial(
    const uint8_t* db_code,
    const float* dist_table,
    int M, int nbits, int ksub,
    int lane_in_group, int threads_in_group)
{
    float partial = 0.0f;
    for (int m = lane_in_group; m < M; m += threads_in_group) {
        uint32_t k = gpu_extract_centroid_index_direct(db_code, m, nbits);
        partial += dist_table[m * ksub + k];
    }
    return partial;
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
