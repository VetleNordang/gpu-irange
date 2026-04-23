// This program computer the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <map>
#include <tuple>
#include <cuda_profiler_api.h>
#include "iRG_search.h"
#include <curand_kernel.h>
#include "gpu_index.cuh"
#include "gpu_heap.cuh"
#include "gpu_visited.cuh"
#include "gpu_search_updated.cuh"


const int query_K = 10;
int M;

using std::cout;
std::unordered_map<std::string, std::string> paths;

// Test kernel to demonstrate visited array
__global__ void test_visited_array(GPUVisitedArray visited) {
    int query_id = threadIdx.x;  // Each thread is a different query
    
    if (query_id < visited.num_queries) {
        printf("\n=== Query %d Testing Visited Array ===\n", query_id);
        
        // Simulate visiting some nodes
        int nodes_to_visit[] = {10, 25, 50, 100, 10};  // Note: 10 appears twice!
        
        for (int i = 0; i < 5; i++) {
            int node = nodes_to_visit[i];
            
            // Check if already visited
            if (isVisited(visited, query_id, node)) {
                printf("  Node %d: Already visited! (skipping)\n", node);
            } else {
                printf("  Node %d: First time visiting (marking as visited)\n", node);
                markVisited(visited, query_id, node);
            }
        }
        
        printf("=== Query %d Complete ===\n", query_id);
    }
}

// Random number generation
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


void init()
{
    // data vectors should be sorted by the attribute values in ascending order
    paths["data_vector"] = "";

    paths["query_vector"] = "";
    // the path of document where range files are saved
    paths["range_saveprefix"] = "";
    // the path of document where groundtruth files are saved
    paths["groundtruth_saveprefix"] = "";
    // the path where index file is saved
    paths["index"] = "";
    // the path of document where search result files are saved
    paths["result_saveprefix"] = "";
    // M is the maximum out-degree same as index build
}

void Generate(iRangeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    iRangeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb);
    generator.GenerateRange(paths["range_saveprefix"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
}

 

void load_index_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int dimension = index.storage->Dim;
    int data_points = index.max_elements_;
    size_t total_index_memory = (size_t)data_points * index.size_data_per_element_;
printf("Attempting to allocate %.2f MB for index\n", total_index_memory / (1024.0 * 1024.0));
    
    // Set metadata
    gpu_index.d_dim = dimension;
    gpu_index.d_size_data_per_element = index.size_data_per_element_;
    gpu_index.d_size_links_per_element = index.size_links_per_element_;
    gpu_index.d_offsetData = index.offsetData_;
    
    // Allocate GPU memory for entire index structure
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_data_memory, total_index_memory);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Copy entire index structure from CPU to GPU
    err = cudaMemcpy(gpu_index.d_data_memory, index.data_memory_, total_index_memory, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_data_memory);
        exit(1);
    }
    printf("✓ Copied %.2f GB from CPU to GPU\n", 
            total_index_memory / (1024.0*1024.0*1024.0));
}

void load_segment_tree_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    iRangeGraph::SegmentTree *tree = index.tree;
    std::vector<GPUNode> gpu_nodes = tree->FlattenGPUTree();
    
    int mem_to_allocate_to_gpu = gpu_nodes.size() * sizeof(GPUNode);

    // Initialize segment tree structure
    gpu_index.d_segment_tree.num_nodes = gpu_nodes.size();
    gpu_index.d_segment_tree.max_depth = tree->max_depth;
    gpu_index.d_segment_tree.root = gpu_nodes[0];
    
    // Allocate GPU memory for nodes
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_segment_tree.d_nodes, mem_to_allocate_to_gpu);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for segment tree: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy nodes to GPU
    err = cudaMemcpy(gpu_index.d_segment_tree.d_nodes, gpu_nodes.data(), mem_to_allocate_to_gpu, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for segment tree: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_segment_tree.d_nodes);
        exit(1);
    }
    
    printf("✓ Copied %zu nodes (%.2f KB) to GPU\n", 
           gpu_index.d_segment_tree.num_nodes, mem_to_allocate_to_gpu / 1024.0);
    
    // Run test kernel to verify tree structure
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Test kernel error: %s\n", cudaGetErrorString(err));
    }
}


void load_queries_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int query_nb = index.storage->query_nb;
    int dim = index.storage->Dim;
    
    // Allocate GPU memory for query vectors
    size_t query_vectors_size = (size_t)query_nb * dim * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_query_vectors, query_vectors_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for query vectors: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    float *flatten_queries = new float[query_nb * dim];
    for (int i = 0; i < query_nb; i++) {
        for (int d = 0; d < dim; d++) {
            flatten_queries[i * dim + d] = index.storage->query_points[i][d];
        }
    }

    // Copy query vectors to GPU
    err = cudaMemcpy(gpu_index.d_query_vectors, flatten_queries, query_vectors_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for query vectors: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_query_vectors);
        delete[] flatten_queries;
        exit(1);
    }
    
    // Allocate GPU memory for query ranges
    // query_range is a map with suffix keys
    std::vector<int> suffix_keys;
    for (auto range : index.storage->query_range) {
        suffix_keys.push_back(range.first);
    }
    size_t query_ranges_size = (size_t)query_nb * 2 * sizeof(int) * suffix_keys.size();
    err = cudaMalloc((void**)&gpu_index.d_query_range, query_ranges_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for query ranges: %s\n", cudaGetErrorString(err));
        delete[] flatten_queries;
        exit(1);
    }
    
    int *flatten_ranges = new int[query_nb * 2 * suffix_keys.size()];
    for (int i = 0; i < query_nb; i++) {
        for (size_t s = 0; s < suffix_keys.size(); s++) {
            int suffix = suffix_keys[s];
            flatten_ranges[i * 2 * suffix_keys.size() + s * 2] = index.storage->query_range[suffix][i].first;
            flatten_ranges[i * 2 * suffix_keys.size() + s * 2 + 1] = index.storage->query_range[suffix][i].second;
        }
    }

    // Copy query ranges to GPU
    err = cudaMemcpy(gpu_index.d_query_range, flatten_ranges, query_ranges_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for query ranges: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_query_range);
        delete[] flatten_queries;
        delete[] flatten_ranges;
        exit(1);
    }
    
    delete[] flatten_queries;
    delete[] flatten_ranges;
    
    printf("✓ Copied %d queries (%d suffixes) and ranges to GPU\n", query_nb, (int)suffix_keys.size());
}

int* make_result_buffer_on_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int query_nb = index.storage->query_nb;
    size_t result_buffer_size = (size_t)query_nb * query_K * sizeof(int);

    cudaError_t err = cudaMalloc((void**)&gpu_index.d_results, result_buffer_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for result buffer: %s\n", cudaGetErrorString(err));        
        return NULL;
    }

    return gpu_index.d_results;

}

// CPU function to prepare data and launch GPU kernel
void search_on_gpu(iRangeGraph::iRangeGraph_Search<float> &index, std::vector<int> SearchEF, std::string saveprefix) {
    // Validate all SearchEF values are within supported range
    for (int ef : SearchEF) {
        if (ef > MAX_SEARCH_EF) {
            printf("ERROR: SearchEF=%d exceeds MAX_SEARCH_EF=%d\n", ef, MAX_SEARCH_EF);
            printf("Please increase MAX_SEARCH_EF in gpu_search_updated.cuh or reduce SearchEF\n");
            exit(1);
        }
    }

    GPUIndex gpu_index;

    // Load main index data to GPU
    load_index_to_gpu(index, gpu_index);
    
    // Load segment tree to GPU
    load_segment_tree_to_gpu(index, gpu_index);
    load_queries_to_gpu(index, gpu_index);
    make_result_buffer_on_gpu(index, gpu_index);

    iRangeGraph::DataLoader *storage = index.storage;

    
    // Structure to store all results in memory before writing to disk
    // Map: suffix -> vector of (SearchEF, Recall, QPS, DCO, HOP)
    std::map<int, std::vector<std::tuple<int, float, float, float, float>>> all_results;
    
    int limit_tests = 0; // Added for profiler auto-shutdown

    // Iterate over all suffixes in storage->query_range (same as CPU version)
    size_t suffix_idx = 0;
    for (auto range : storage->query_range) {
        int suffix = range.first;
        printf("\n========================================\n");
        printf("Processing suffix %d (%zu/%zu)\n", suffix, suffix_idx + 1, storage->query_range.size());
        printf("========================================\n");
        std::cout << "suffix = " << suffix << std::endl;
        

        for (int ef : SearchEF) {
            limit_tests++;
            if (limit_tests > 11) {
                cudaProfilerStop();
            }

            printf("\n--- Testing EF=%d for suffix %d ---\n", ef, suffix);
            printf("\n=== Initializing Visited Array for Search ===\n");
            int query_nb = index.storage->query_nb;
            int max_elements = index.max_elements_;
            int dim = index.storage->Dim;
            
            GPUVisitedArray visited;
            cudaError_t err = initGPUVisitedArray(visited, query_nb, max_elements);
            if (err != cudaSuccess) {
                printf("Failed to initialize visited array: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            printf("✓ Allocated %.2f MB for visited arrays (%d queries)\n", 
                   visited.total_size() / (1024.0*1024.0), query_nb);
            
            // Allocate metrics arrays on GPU
            int* d_hops;
            int* d_dist_comps;
            cudaMalloc(&d_hops, query_nb * sizeof(int));
            cudaMalloc(&d_dist_comps, query_nb * sizeof(int));
            
            HeapNode* d_candidate_buffer;
            HeapNode* d_top_candidate_buffer;
            cudaMalloc(&d_candidate_buffer, query_nb * MAX_SEARCH_EF * sizeof(HeapNode));
            cudaMalloc(&d_top_candidate_buffer, query_nb * MAX_SEARCH_EF * sizeof(HeapNode));

            // Initialize to zero (CRITICAL - otherwise contains garbage values!)
            cudaMemset(d_hops, 0, query_nb * sizeof(int));
            cudaMemset(d_dist_comps, 0, query_nb * sizeof(int));
            
            int queries_per_block = MAX_QUERIES_PER_BLOCK;
            int threads_per_query = THREADS_PER_QUERY;
            int threads_per_block = threads_per_query * queries_per_block;
            int num_blocks = (query_nb + queries_per_block - 1) / queries_per_block;
            
            printf("Configuration: %d blocks × %d threads, Queries: %d, K: %d\n", 
                   num_blocks, threads_per_block, query_nb, query_K);
            
            // Reset visited counters for new search
            unsigned short* temp_curV = new unsigned short[query_nb];
            for (int i = 0; i < query_nb; i++) temp_curV[i] = 1;
            cudaMemcpy(visited.d_curV, temp_curV, query_nb * sizeof(unsigned short), cudaMemcpyHostToDevice);
            cudaMemset(visited.d_mass, 0, (size_t)query_nb * max_elements * sizeof(unsigned short));
            delete[] temp_curV;
            
            // Time the kernel
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            // Launch kernel
            unsigned long long kernel_seed = std::chrono::system_clock::now().time_since_epoch().count();
            irange_search_kernel<<<num_blocks, threads_per_block>>>(
                gpu_index, visited, query_nb, ef, query_K, dim, suffix_idx, d_hops, d_dist_comps,
                index.size_links_per_layer_, kernel_seed
            );
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            float searchtime = milliseconds / 1000.0f;  // Convert to seconds
            
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Search kernel error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            
            // Copy results back to CPU
            int* cpu_results = new int[query_nb * query_K];
            int* cpu_hops = new int[query_nb];
            int* cpu_dist_comps = new int[query_nb];
            
            cudaMemcpy(cpu_results, gpu_index.d_results, query_nb * query_K * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_hops, d_hops, query_nb * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_dist_comps, d_dist_comps, query_nb * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Compute recall if ground truth is available
            int tp = 0;
            if (index.storage->groundtruth.count(suffix)) {
                auto &gt = index.storage->groundtruth[suffix];
                for (int i = 0; i < query_nb; i++) {
                    for (int k = 0; k < query_K; k++) {
                        int result_id = cpu_results[i * query_K + k];
                        if (result_id != -1) {
                            if (std::find(gt[i].begin(), gt[i].end(), result_id) != gt[i].end()) {
                                tp++;
                            }
                        }
                    }
                }
            }
            
            // Calculate metrics
            float recall = (index.storage->groundtruth.count(suffix)) ? 
                            (1.0f * tp / query_nb / query_K) : 0.0f;
            float qps = query_nb / searchtime;
            
            long long total_hops = 0;
            long long total_dist_comps = 0;
            for (int i = 0; i < query_nb; i++) {
                total_hops += cpu_hops[i];
                total_dist_comps += cpu_dist_comps[i];
            }
            float avg_hops = (float)total_hops / query_nb;
            float avg_dco = (float)total_dist_comps / query_nb;
            
            printf("  Recall: %.4f\n", recall);
            printf("  QPS: %.2f\n", qps);
            printf("  Avg Distance Computations: %.2f\n", avg_dco);
            printf("  Avg Hops: %.2f\n", avg_hops);
            printf("  Search time: %.3f seconds\n", searchtime);
            
            // Store results in memory (will write to file later)
            all_results[suffix].push_back(std::make_tuple(ef, recall, qps, avg_dco, avg_hops));
            
            // Show first 3 query results for this suffix (only for first EF value)
            if (suffix == 0 && ef == SearchEF[0]) {
                printf("\n  First 3 Query Results:\n");
                for (int i = 0; i < 3 && i < query_nb; i++) {
                    printf("    Query %d: [", i);
                    for (int k = 0; k < query_K; k++) {
                        printf("%d", cpu_results[i * query_K + k]);
                        if (k < query_K - 1) printf(", ");
                    }
                    printf("] (hops=%d, dco=%d)\n", cpu_hops[i], cpu_dist_comps[i]);
                }
            }
            
            delete[] cpu_results;
            delete[] cpu_hops;
            delete[] cpu_dist_comps;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            
            // Clean up
            cudaFree(d_hops);
            cudaFree(d_dist_comps);
            cudaFree(d_candidate_buffer);
            cudaFree(d_top_candidate_buffer);
            freeGPUVisitedArray(visited);
            
        }

        suffix_idx++;  // Increment for next suffix

    }

    // Write all results to CSV files (outside of loops)
    printf("\n========================================\n");
    printf("Writing all results to disk...\n");
    printf("========================================\n");
    
    for (const auto& result_entry : all_results) {
        int suffix = result_entry.first;
        const auto& results = result_entry.second;
        
        std::string savepath = saveprefix + std::to_string(suffix) + "_gpu.csv";
        CheckPath(savepath);
        std::ofstream outfile(savepath);
        
        if (outfile.is_open()) {
            // Write header
            outfile << "SearchEF,Recall,QPS,DCO,HOP\n";
            
            // Write all results for this suffix
            for (const auto& result : results) {
                int ef = std::get<0>(result);
                float recall = std::get<1>(result);
                float qps = std::get<2>(result);
                float dco = std::get<3>(result);
                float hop = std::get<4>(result);
                
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << "\n";
            }
            
            outfile.close();
            printf("✓ Saved %zu results to %s\n", results.size(), savepath.c_str());
        } else {
            printf("✗ Failed to open %s\n", savepath.c_str());
        }
    }
    
    printf("✓ All results written to disk\n");


    // Initialize visited array for all queries
    int size_of_node = sizeof(iRangeGraph::TreeNode);
    printf("\nSize of TreeNode: %d bytes\n", size_of_node);
    int size_of_tree = sizeof(iRangeGraph::SegmentTree);
    printf("Size of SegmentTree: %d bytes\n", size_of_tree);
    
    // Clean up GPU memory
    if (gpu_index.d_data_memory) cudaFree(gpu_index.d_data_memory);
    if (gpu_index.d_segment_tree.d_nodes) cudaFree(gpu_index.d_segment_tree.d_nodes);
    if (gpu_index.d_query_vectors) cudaFree(gpu_index.d_query_vectors);
    if (gpu_index.d_query_range) cudaFree(gpu_index.d_query_range);
    if (gpu_index.d_results) cudaFree(gpu_index.d_results);
    
    printf("✓ GPU memory cleaned up\n");
}

// CUDA kernel for vector addition
// No change when using CUDA unified memory
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    // Boundary check
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv) {


    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
    }

    if (argc != 15)
        throw Exception("please check input parameters");

    
    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    std::cout << "Loading queries..." << std::endl;
    storage.LoadQuery(paths["query_vector"]);
    // If it is the first run, Generate shall be called; otherwise, Generate can be skipped
    // Generate(storage);
    std::cout << "Loading query ranges..." << std::endl;
    storage.LoadQueryRange(paths["range_saveprefix"]);
    std::cout << "Loading ground truth..." << std::endl;
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);

    std::cout << "Loading index..." << std::endl;
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);
    
    // SearchEF values to test (can be adjusted)
    std::vector<int> SearchEF = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "Running GPU search with " << SearchEF.size() << " different SearchEF values" << std::endl;
    std::cout << "Testing " << storage.query_range.size() << " suffixes with all EF values" << std::endl;
    std::cout << "Total tests: " << SearchEF.size() * storage.query_range.size() << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // Test all suffixes with all EF values
    search_on_gpu(index, SearchEF, paths["result_saveprefix"]);

    std::cout << "\n================================================" << std::endl;
    std::cout << "All GPU searches complete!" << std::endl;
    std::cout << "================================================" << std::endl;
    
    
    return 0;
}