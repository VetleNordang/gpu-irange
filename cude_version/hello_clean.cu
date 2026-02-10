// GPU-accelerated iRangeGraph Search
// Using CUDA unified memory

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <map>
#include "iRG_search.h"
#include "gpu_search.cuh"

const int query_K = 10;
int M;

using std::cout;
std::unordered_map<std::string, std::string> paths;

void init()
{
    paths["data_vector"] = "";
    paths["query_vector"] = "";
    paths["range_saveprefix"] = "";
    paths["groundtruth_saveprefix"] = "";
    paths["index"] = "";
    paths["result_saveprefix"] = "";
}

void Generate(iRangeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    iRangeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb);
    generator.GenerateRange(paths["range_saveprefix"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
}

// CPU function to prepare data and launch GPU kernel
void search_on_gpu(iRangeGraph::DataLoader &storage, iRangeGraph::SegmentTree *tree, int SearchEF, int edge_limit, 
                   int max_elements, int dim, char* data_memory, size_t size_data_per_element, size_t offsetData, size_t size_links_per_layer, int K,
                   iRangeGraph::iRangeGraph_Search<float>* index) {
    char* d_visited_arrays;
    
    for (auto &range_pair : storage.query_range) {
        int suffix = range_pair.first;
        auto &ranges = range_pair.second;
        
        printf("Processing suffix %d with %zu queries\n", suffix, ranges.size());
        
        int number_of_ranges = ranges.size();
        int *h_range_data = new int[number_of_ranges * 2];
        for (size_t i = 0; i < number_of_ranges; i++) {
            h_range_data[i * 2] = ranges[i].first;
            h_range_data[i * 2 + 1] = ranges[i].second;
        }
        
        // Prepare query vectors for GPUFwar
        float* h_query_vectors = new float[number_of_ranges * dim];
        for (size_t i = 0; i < number_of_ranges; i++) {
            memcpy(h_query_vectors + i * dim, storage.query_points[i].data(), dim * sizeof(float));
        }
        
        // Allocate GPU memory
        int *d_range_data;
        float *d_query_vectors;
        int *d_results;
        int *d_hops;
        int *d_dist_comps;
        
        cudaMallocManaged(&d_range_data, number_of_ranges * 2 * sizeof(int));
        cudaMallocManaged(&d_query_vectors, number_of_ranges * dim * sizeof(float));
        cudaMallocManaged(&d_results, number_of_ranges * K * sizeof(int));
        cudaMallocManaged(&d_hops, number_of_ranges * sizeof(int));
        cudaMallocManaged(&d_dist_comps, number_of_ranges * sizeof(int));
        
        memcpy(d_range_data, h_range_data, number_of_ranges * 2 * sizeof(int));
        memcpy(d_query_vectors, h_query_vectors, number_of_ranges * dim * sizeof(float));
        
        cudaMalloc(&d_visited_arrays, (long long)number_of_ranges * max_elements * sizeof(char));
        
        // Start timing
        timeval t1, t2;
        gettimeofday(&t1, NULL);
    
        int blocks = (number_of_ranges + 255) / 256;
        int threads = 256;

        search_gpu<<<blocks, threads>>>(d_range_data, number_of_ranges, number_of_ranges, SearchEF, edge_limit, tree, 
                                        d_visited_arrays, max_elements, d_query_vectors, dim, 
                                        data_memory, size_data_per_element, offsetData, size_links_per_layer,
                                        d_results, K, d_hops, d_dist_comps, true);  // DEBUG=TRUE
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        } else {
            printf("✓ Kernel launched successfully\n");
        }
        
        printf("Synchronizing device...\n");
        cudaDeviceSynchronize();
        
        gettimeofday(&t2, NULL);
        float searchtime = GetTime(t1, t2);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        } else {
            printf("✓ Kernel executed successfully\n");
        }
        
        // ========== POST-EXECUTION DEBUGGING ==========
        printf("\n=== Checking Results ===\n");
        
        // Check if any results were written
        int non_negative_count = 0;
        int all_negative_queries = 0;
        for (int i = 0; i < number_of_ranges; i++) {
            bool has_result = false;
            for (int k = 0; k < K; k++) {
                if (d_results[i * K + k] >= 0) {
                    non_negative_count++;
                    has_result = true;
                }
            }
            if (!has_result) all_negative_queries++;
        }

        // Calculate metrics
        long long total_hops = 0;
        long long total_dist_comps = 0;
        int total_tp = 0;
        
        auto &gt = storage.groundtruth[suffix];
    
        
        for (int i = 0; i < number_of_ranges; i++) {
            total_hops += d_hops[i];
            total_dist_comps += d_dist_comps[i];
            
            // Check recall
            std::map<int, int> record;
            for (int k = 0; k < K; k++) {
                int result_id = d_results[i * K + k];
                if (result_id >= 0) {
                    if (record.count(result_id)) {
                        printf("Warning: repetitive result at query %d\n", i);
                    }
                    record[result_id] = 1;
                    if (std::find(gt[i].begin(), gt[i].end(), result_id) != gt[i].end()) {
                        total_tp++;
                    }
                }
            }
        }
        
        float recall = 1.0 * total_tp / number_of_ranges / K;
        float qps = number_of_ranges / searchtime;
        float avg_dist_comps = total_dist_comps * 1.0 / number_of_ranges;
        float avg_hops = total_hops * 1.0 / number_of_ranges;
        
        printf("GPU Results - Suffix %d:\n", suffix);
        printf("  SearchEF: %d\n", SearchEF);
        printf("  Recall: %.4f\n", recall);
        printf("  QPS: %.2f\n", qps);
        printf("  Avg Distance Computations: %.2f\n", avg_dist_comps);
        printf("  Avg Hops: %.2f\n", avg_hops);
        
        // Write results to file
        std::string savepath = paths["result_saveprefix"] + std::to_string(suffix) + "_GPU.csv";
        CheckPath(savepath);
        std::ofstream outfile(savepath);
        if (!outfile.is_open()) {
            printf("Warning: cannot open %s\n", savepath.c_str());
        } else {
            outfile << "SearchEF,Recall,QPS,DistComps,Hops\n";
            outfile << SearchEF << "," << recall << "," << qps << "," << avg_dist_comps << "," << avg_hops << std::endl;
            outfile.close();
            printf("Results written to %s\n", savepath.c_str());
        }
        
        cudaFree(d_visited_arrays);
        cudaFree(d_range_data);
        cudaFree(d_query_vectors);
        cudaFree(d_results);
        cudaFree(d_hops);
        cudaFree(d_dist_comps);
        delete[] h_range_data;
        delete[] h_query_vectors;
    }
}

int main(int argc, char **argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA Device Info:\n");
    printf("  Number of CUDA devices: %d\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("\n");
    
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
    //Generate(storage);
    std::cout << "Loading query ranges..." << std::endl;
    storage.LoadQueryRange(paths["range_saveprefix"]);
    std::cout << "Loading groundtruth..." << std::endl;
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);

    std::cout << "Loading index..." << std::endl;
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);
    int SearchEF = 500;
    
    search_on_gpu(storage, index.tree, SearchEF, M, index.max_elements_, index.dim_, 
                  index.data_memory_, index.size_data_per_element_, index.offsetData_, index.size_links_per_layer_, query_K, &index);
    
    cudaDeviceSynchronize();
    printf("GPU search completed!\n");
    
    return 0;
}