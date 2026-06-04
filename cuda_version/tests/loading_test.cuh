#include <stdio.h>
#include <cmath>
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "gpu_index.cuh"
#include "iRG_search.h"


// GPU kernel to copy last 10 vectors to output array for CPU verification
// Each thread copies one complete vector
__global__ void extract_last_vectors(char* d_data_memory, int dim, size_t size_data_per_element, 
                                      size_t offsetData, int max_elements, float* d_output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 10) {
        // Get vector_id for last 10 vectors
        int vector_id = max_elements - 10 + tid;
        
        // Calculate offset: same as getDataByInternalId
        char* vector_data = d_data_memory + vector_id * size_data_per_element + offsetData;
        float* vec = (float*)vector_data;
        
        // Copy entire vector to output array
        for (int i = 0; i < dim; i++) {
            d_output[tid * dim + i] = vec[i];
        }
    }
}  

void test_gpu_index_loading(const iRangeGraph::iRangeGraph_Search<float>& index) {
    
    GPUIndex gpu_index;
    load_gpu_index(gpu_index, index);

// Allocate device memory for 10 vectors output
    int num_test_vectors = 10;
    int dimension = index.storage->Dim;
    size_t output_size = num_test_vectors * dimension * sizeof(float);
    float* d_output;
    cudaError_t err = cudaMalloc((void**)&d_output, output_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc for output failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_data_memory);
        return;
    }
    
    // Launch kernel to extract LAST 10 vectors from GPU index
    printf("\n=== Extracting LAST 10 vectors from GPU ===\n");
    extract_last_vectors<<<1, 10>>>(gpu_index.d_data_memory, dimension, index.size_data_per_element_, 
                                     index.offsetData_, index.max_elements_, d_output);
    cudaDeviceSynchronize();
    
    // Allocate host memory and copy from GPU to CPU
    float* h_output = new float[num_test_vectors * dimension];
    err = cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("❌ cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
        delete[] h_output;
        cudaFree(d_output);
        cudaFree(gpu_index.d_data_memory);
        return;
    }
    
    // Verify on CPU by comparing with original index data
    printf("=== CPU Verification: Comparing GPU vs CPU ===\n");
    int mismatches = 0;
    for (int i = 0; i < num_test_vectors; i++) {
        int vector_id = index.max_elements_ - 10 + i;
        
        // Get vector from CPU index structure using same offset logic
        char* cpu_vector_data = index.data_memory_ + vector_id * index.size_data_per_element_ + index.offsetData_;
        float* cpu_vec = (float*)cpu_vector_data;
        
        // Compare with GPU output
        bool mismatch_found = false;
        for (int j = 0; j < dimension; j++) {
            float gpu_val = h_output[i * dimension + j];
            float cpu_val = cpu_vec[j];
            
            // Check for mismatch (use small epsilon for float comparison)
            if (std::abs(gpu_val - cpu_val) > 1e-6) {
                if (!mismatch_found) {
                    printf("MISMATCH in Vector %d:\n", vector_id);
                    mismatch_found = true;
                }
                printf("  Dim %d: GPU=%.6f, CPU=%.6f (diff=%.6e)\n", 
                       j, gpu_val, cpu_val, gpu_val - cpu_val);
                mismatches++;
                
                // Only print first 10 mismatches per vector
                if (mismatches %  10== 0 && mismatch_found) {
                    printf("  ... (showing first 10 mismatches per vector)\n");
                    break;
                }
            }
        }
    }
    
    if (mismatches == 0) {
        printf("✓ All vectors match! GPU index is correct.\n");
    } else {
        printf("Found %d total mismatches\n", mismatches);
    }
    printf("=== Done ===\n\n");
    
    // Clean up
    delete[] h_output;
    cudaFree(d_output);
    cudaFree(gpu_index.d_data_memory);
}