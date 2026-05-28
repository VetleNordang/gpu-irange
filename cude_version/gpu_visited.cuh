#pragma once

#include <cuda_runtime.h>

/*
 * GPU Visited Array Structure
 * 
 * HOW IT WORKS (Simple Explanation):
 * 
 * Imagine you're playing a game where you visit different rooms (nodes).
 * Instead of writing "VISITED" on each room's door (which you'd have to erase 
 * between games), you give each game a unique number.
 * 
 * When you visit a room, you write the game number on that room's door.
 * To check if you visited a room in THIS game, you just check if the number 
 * on the door matches your current game number.
 * 
 * For the next game, you just increment your game number. All old numbers 
 * become invalid automatically - no erasing needed!
 * 
 * Example:
 *   Game 1: Visit rooms [5, 10, 15]
 *     - Write "1" on doors 5, 10, 15
 *     - Your game number is "1"
 *   
 *   Game 2: Visit rooms [10, 20]
 *     - Your game number is now "2"
 *     - Check room 10: it says "1", but you need "2", so NOT visited yet
 *     - Write "2" on doors 10, 20
 *   
 *   No need to erase the "1"s - they're just ignored!
 * 
 * TECHNICAL DETAILS:
 * 
 * - d_mass: Array storing the "game number" for each node
 *   Size: [num_queries × max_elements]
 *   Example: For query 3 checking node 100:
 *            index = 3 × max_elements + 100
 * 
 * - d_curV: Array storing the current "game number" for each query
 *   Size: [num_queries]
 *   Each query has its own independent counter
 * 
 * - We use unsigned short (0-65535), so every 65,535 searches we need
 *   to actually clear the array (memset to 0)
 * 
 * MEMORY USAGE:
 *   For 1M nodes and 1024 concurrent queries:
 *   = 1,024 × 1,000,000 × 2 bytes 
 *   = 2 GB
 * 
 * WHY THIS IS EFFICIENT ON GPU:
 *   - No atomic operations needed for clearing
 *   - No synchronization between threads
 *   - Each query operates in its own memory space
 *   - Coalesced memory access patterns
 */

struct GPUVisitedArray {
    uint8_t* d_mass;             // [num_queries × max_elements] - stores visit counter for each node
    uint8_t* d_curV;             // [num_queries] - current counter value for each query
    int max_elements;            // Total number of nodes in the graph
    int num_queries;             // Number of concurrent queries

    // Size of one query's visited array in bytes
    size_t query_array_size() const {
        return (size_t)max_elements * sizeof(uint8_t);
    }

    // Total size needed on GPU
    size_t total_size() const {
        return query_array_size() * num_queries + num_queries * sizeof(uint8_t);
    }
};

// Check if a node has been visited in this search
// query_id: which query is asking (0 to num_queries-1)
// node_id: which node to check (0 to max_elements-1)
__device__ inline bool isVisited(const GPUVisitedArray& visited, int query_id, int node_id) {
    long long offset = (long long)query_id * visited.max_elements + node_id;
    return visited.d_mass[offset] == visited.d_curV[query_id];
}

__device__ inline void markVisited(GPUVisitedArray& visited, int query_id, int node_id) {
    long long offset = (long long)query_id * visited.max_elements + node_id;
    visited.d_mass[offset] = visited.d_curV[query_id];
}

// Initialize visited array on CPU side
inline cudaError_t initGPUVisitedArray(GPUVisitedArray& visited, int num_queries, int max_elements) {
    visited.num_queries = num_queries;
    visited.max_elements = max_elements;

    size_t mass_size = (size_t)num_queries * max_elements * sizeof(uint8_t);
    cudaError_t err = cudaMalloc(&visited.d_mass, mass_size);
    if (err != cudaSuccess) return err;

    err = cudaMemset(visited.d_mass, 0, mass_size);
    if (err != cudaSuccess) {
        cudaFree(visited.d_mass);
        return err;
    }

    size_t curV_size = num_queries * sizeof(uint8_t);
    err = cudaMalloc(&visited.d_curV, curV_size);
    if (err != cudaSuccess) {
        cudaFree(visited.d_mass);
        return err;
    }

    // Start counters at 1 so 0 means "unvisited"
    uint8_t* temp = new uint8_t[num_queries];
    for (int i = 0; i < num_queries; i++) temp[i] = 1;
    err = cudaMemcpy(visited.d_curV, temp, curV_size, cudaMemcpyHostToDevice);
    delete[] temp;

    if (err != cudaSuccess) {
        cudaFree(visited.d_mass);
        cudaFree(visited.d_curV);
        return err;
    }

    return cudaSuccess;
}

// Free GPU memory
inline void freeGPUVisitedArray(GPUVisitedArray& visited) {
    if (visited.d_mass) cudaFree(visited.d_mass);
    if (visited.d_curV) cudaFree(visited.d_curV);
    visited.d_mass = nullptr;
    visited.d_curV = nullptr;
}

// Kernel to increment counters for next batch of searches
// This is how you "reset" without actually clearing memory!
__global__ void incrementVisitedCounters(uint8_t* d_curV, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries) {
        d_curV[idx]++;
        // Wraps at 255 back to 0; caller must memset d_mass when this happens.
        // With uint8 this occurs every 254 searches per query (rare in practice).
        if (d_curV[idx] == 0) d_curV[idx] = 1;
    }
}

/*
 * USAGE EXAMPLE IN YOUR SEARCH KERNEL:
 * 
 * __global__ void search_kernel(GPUVisitedArray visited, ...) {
 *     int query_id = blockIdx.x * blockDim.x + threadIdx.x;
 *     
 *     // Start search at some entry point
 *     int current_node = entry_point;
 *     
 *     // Mark entry point as visited
 *     markVisited(visited, query_id, current_node);
 *     
 *     // During search...
 *     for (int neighbor : neighbors) {
 *         // Check if we've already visited this neighbor
 *         if (!isVisited(visited, query_id, neighbor)) {
 *             // Not visited yet - mark it and explore
 *             markVisited(visited, query_id, neighbor);
 *             // ... do search operations ...
 *         }
 *     }
 * }
 * 
 * SETUP ON CPU:
 * 
 * GPUVisitedArray visited;
 * cudaError_t err = initGPUVisitedArray(visited, num_queries, max_elements);
 * printf("Allocated %.2f GB for visited arrays\n", 
 *        visited.total_size() / (1024.0*1024.0*1024.0));
 * 
 * // Launch search kernel
 * search_kernel<<<blocks, threads>>>(visited, ...);
 * 
 * // Clean up
 * freeGPUVisitedArray(visited);
 */
