#pragma once

#include <cuda_runtime.h>

// Visited array using a per-query counter instead of clearing between searches.
// d_mass stores the counter value written when a node was last visited;
// a node is "visited" iff d_mass[query_id * max_elements + node_id] == d_curV[query_id].
//
// MEMORY: num_queries * max_elements bytes (e.g. 1024 queries * 1M nodes = 1 GB).

struct GPUVisitedArray {
    uint8_t* d_mass;      // [num_queries x max_elements]
    uint8_t* d_curV;      // [num_queries] current counter per query
    int max_elements;
    int num_queries;

    size_t query_array_size() const { return (size_t)max_elements * sizeof(uint8_t); }
    size_t total_size()       const { return query_array_size() * num_queries + num_queries * sizeof(uint8_t); }
};

__device__ inline bool isVisited(const GPUVisitedArray& visited, int query_id, int node_id) {
    long long offset = (long long)query_id * visited.max_elements + node_id;
    return visited.d_mass[offset] == visited.d_curV[query_id];
}

__device__ inline void markVisited(GPUVisitedArray& visited, int query_id, int node_id) {
    long long offset = (long long)query_id * visited.max_elements + node_id;
    visited.d_mass[offset] = visited.d_curV[query_id];
}

inline cudaError_t initGPUVisitedArray(GPUVisitedArray& visited, int num_queries, int max_elements) {
    visited.num_queries = num_queries;
    visited.max_elements = max_elements;

    size_t mass_size = (size_t)num_queries * max_elements * sizeof(uint8_t);
    cudaError_t err = cudaMalloc(&visited.d_mass, mass_size);
    if (err != cudaSuccess) return err;

    err = cudaMemset(visited.d_mass, 0, mass_size);
    if (err != cudaSuccess) { cudaFree(visited.d_mass); return err; }

    size_t curV_size = num_queries * sizeof(uint8_t);
    err = cudaMalloc(&visited.d_curV, curV_size);
    if (err != cudaSuccess) { cudaFree(visited.d_mass); return err; }

    // Start counters at 1 so 0 means "unvisited"
    uint8_t* temp = new uint8_t[num_queries];
    for (int i = 0; i < num_queries; i++) temp[i] = 1;
    err = cudaMemcpy(visited.d_curV, temp, curV_size, cudaMemcpyHostToDevice);
    delete[] temp;

    if (err != cudaSuccess) { cudaFree(visited.d_mass); cudaFree(visited.d_curV); return err; }

    return cudaSuccess;
}

inline void freeGPUVisitedArray(GPUVisitedArray& visited) {
    if (visited.d_mass) cudaFree(visited.d_mass);
    if (visited.d_curV) cudaFree(visited.d_curV);
    visited.d_mass = nullptr;
    visited.d_curV = nullptr;
}
