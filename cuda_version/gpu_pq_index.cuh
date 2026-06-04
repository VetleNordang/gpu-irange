#pragma once

#include <vector>
#include <cstdint>
#include "gpu_index.cuh"

// Plain POD struct — safe to pass by value into a CUDA kernel.
// Holds only device pointers and scalars; no std::vector, no FAISS types.
struct GPUPQParams {
    uint8_t* d_compressed_codes = nullptr;
    float*   d_centroids        = nullptr;
    float*   d_dist_tables      = nullptr;
    int  pq_code_size = 0;
    int  pq_M        = 0;
    int  pq_nbits    = 0;
    int  pq_dsub     = 0;
    int  pq_ksub     = 0;
};

// Host-side PQ state. Includes GPUIndex for the base index fields plus the
// PQ-specific allocations. Only gpu_pq.cu includes this header, so gpu_full.cu
// and gpu_root.cu never see FAISS.
struct GPUPQIndex {
    GPUIndex   base;
    GPUPQParams pq;

    std::vector<uint8_t> pq_codes_cpu;
    size_t gpu_codes_size = 0;

    GPUPQParams pq_params() const { return pq; }

    void free_pq() {
        if (pq.d_compressed_codes) { cudaFree(pq.d_compressed_codes); pq.d_compressed_codes = nullptr; }
        if (pq.d_centroids)        { cudaFree(pq.d_centroids);        pq.d_centroids        = nullptr; }
        if (pq.d_dist_tables)      { cudaFree(pq.d_dist_tables);      pq.d_dist_tables      = nullptr; }
        pq_codes_cpu.clear();
        pq_codes_cpu.shrink_to_fit();
    }
};
