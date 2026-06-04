// GPU iRangeGraph search using only the root segment tree node as entry point.
// Entry point selection skips tree traversal — one random point from the full
// attribute range is picked per query on the CPU, then passed to the kernel.
// Everything else (HNSW traversal, distance computation, result collection)
// is identical to hello.cu.

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <map>
#include <tuple>
#include <unordered_set>
#include <cuda_profiler_api.h>
#include "iRG_search.h"
#include <curand_kernel.h>
#include "gpu_index.cuh"
#include "gpu_heap.cuh"
#include "gpu_visited.cuh"
#include "gpu_search_updated.cuh"


const int query_K = 100;
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

void load_index_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int dimension = index.storage->Dim;
    int data_points = index.max_elements_;
    size_t total_index_memory = (size_t)data_points * index.size_data_per_element_;

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("GPU memory: %.2f GB free / %.2f GB total\n",
           freeMem / (1024.0*1024*1024), totalMem / (1024.0*1024*1024));
    printf("Attempting to allocate %.2f MB for index\n", total_index_memory / (1024.0 * 1024.0));

    gpu_index.d_dim = dimension;
    gpu_index.d_size_data_per_element = index.size_data_per_element_;
    gpu_index.d_size_links_per_element = index.size_links_per_element_;
    gpu_index.d_offsetData = index.offsetData_;

    cudaError_t err = cudaMalloc((void**)&gpu_index.d_data_memory, total_index_memory);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

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

    size_t mem_to_allocate_to_gpu = gpu_nodes.size() * sizeof(GPUNode);

    gpu_index.d_segment_tree.num_nodes = gpu_nodes.size();
    gpu_index.d_segment_tree.max_depth = tree->max_depth;
    gpu_index.d_segment_tree.root = gpu_nodes[0];

    cudaError_t err = cudaMalloc((void**)&gpu_index.d_segment_tree.d_nodes, mem_to_allocate_to_gpu);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for segment tree: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMemcpy(gpu_index.d_segment_tree.d_nodes, gpu_nodes.data(), mem_to_allocate_to_gpu, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for segment tree: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_segment_tree.d_nodes);
        exit(1);
    }

    printf("✓ Copied %zu nodes (%.2f KB) to GPU\n",
           gpu_index.d_segment_tree.num_nodes, mem_to_allocate_to_gpu / 1024.0);
    cudaDeviceSynchronize();
}


void load_queries_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int query_nb = index.storage->query_nb;
    int dim = index.storage->Dim;

    size_t query_vectors_size = (size_t)query_nb * dim * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_query_vectors, query_vectors_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for query vectors: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    float *flatten_queries = new float[query_nb * dim];
    for (int i = 0; i < query_nb; i++)
        for (int d = 0; d < dim; d++)
            flatten_queries[i * dim + d] = index.storage->query_points[i][d];

    err = cudaMemcpy(gpu_index.d_query_vectors, flatten_queries, query_vectors_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for query vectors: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_query_vectors);
        delete[] flatten_queries;
        exit(1);
    }

    std::vector<int> suffix_keys;
    for (auto range : index.storage->query_range)
        suffix_keys.push_back(range.first);

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
            flatten_ranges[i * 2 * suffix_keys.size() + s * 2]     = index.storage->query_range[suffix][i].first;
            flatten_ranges[i * 2 * suffix_keys.size() + s * 2 + 1] = index.storage->query_range[suffix][i].second;
        }
    }

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

void search_on_gpu(iRangeGraph::iRangeGraph_Search<float> &index, std::vector<int> SearchEF, std::string saveprefix) {
    for (int ef : SearchEF) {
        if (ef > MAX_SEARCH_EF) {
            printf("ERROR: SearchEF=%d exceeds MAX_SEARCH_EF=%d\n", ef, MAX_SEARCH_EF);
            exit(1);
        }
    }

    GPUIndex gpu_index;

    load_index_to_gpu(index, gpu_index);
    load_segment_tree_to_gpu(index, gpu_index);
    load_queries_to_gpu(index, gpu_index);
    make_result_buffer_on_gpu(index, gpu_index);

    iRangeGraph::DataLoader *storage = index.storage;
    iRangeGraph::SegmentTree *tree = index.tree;

    // Root node covers the full attribute range
    int root_lbound = tree->root->lbound;
    int root_rbound = tree->root->rbound;
    printf("Root node: lbound=%d rbound=%d (full range, no tree traversal)\n", root_lbound, root_rbound);

    int*   d_entry_ids    = nullptr;
    float* d_entry_dists  = nullptr;
    int*   d_entry_counts = nullptr;

    size_t suffix_idx = 0;
    for (auto range : storage->query_range) {
        int suffix = range.first;

        std::string savepath = saveprefix + std::to_string(suffix) + "_gpu.csv";
        CheckPath(savepath);
        std::ofstream outfile(savepath);
        if (!outfile.is_open()) {
            printf("✗ Failed to open %s\n", savepath.c_str());
        } else {
            outfile << std::fixed << std::setprecision(6);
            outfile << "SearchEF,Recall@10,Recall@50,Recall@100,QPS,DCO,HOP,VRAM_MB,PeakVRAM_MB\n";
            outfile.flush();
        }

        // Entry points from root node only — one random point per query, no tree traversal.
        // MAX_EP must match the 100-slot stride the kernel uses when indexing d_entry_ids.
        {
            int query_nb = index.storage->query_nb;
            const int MAX_EP = 100;
            int*   h_entry_ids    = new int  [query_nb * MAX_EP]();
            float* h_entry_dists  = new float[query_nb * MAX_EP]();
            int*   h_entry_counts = new int  [query_nb];

            unsigned long long ep_seed = std::chrono::system_clock::now().time_since_epoch().count();
            int root_range_size = root_rbound - root_lbound + 1;

            for (int i = 0; i < query_nb; i++) {
                unsigned int rng = (unsigned int)(ep_seed + i);
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                int pid = root_lbound + (rng % (unsigned int)root_range_size);

                char* ep_data = index.getDataByInternalId(pid);
                float dist = index.fstdistfunc_(
                    storage->query_points[i].data(), ep_data, index.dist_func_param_);

                h_entry_ids  [i * MAX_EP] = pid;
                h_entry_dists[i * MAX_EP] = dist;
                h_entry_counts[i] = 1;  // only 1 entry point per query
            }

            cudaMalloc(&d_entry_ids,    query_nb * MAX_EP * sizeof(int));
            cudaMalloc(&d_entry_dists,  query_nb * MAX_EP * sizeof(float));
            cudaMalloc(&d_entry_counts, query_nb * sizeof(int));
            cudaMemcpy(d_entry_ids,    h_entry_ids,    query_nb * MAX_EP * sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemcpy(d_entry_dists,  h_entry_dists,  query_nb * MAX_EP * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_entry_counts, h_entry_counts, query_nb * sizeof(int),            cudaMemcpyHostToDevice);
            delete[] h_entry_ids;
            delete[] h_entry_dists;
            delete[] h_entry_counts;
        }

        for (int ef : SearchEF) {
            int query_nb = index.storage->query_nb;
            int max_elements = index.max_elements_;
            int dim = index.storage->Dim;

            GPUVisitedArray visited;
            cudaError_t err = initGPUVisitedArray(visited, query_nb, max_elements);
            if (err != cudaSuccess) {
                printf("Failed to initialize visited array: %s\n", cudaGetErrorString(err));
                exit(1);
            }

            int* d_hops;
            int* d_dist_comps;
            cudaMalloc(&d_hops, query_nb * sizeof(int));
            cudaMalloc(&d_dist_comps, query_nb * sizeof(int));

            HeapNode* d_candidate_buffer;
            HeapNode* d_top_candidate_buffer;
            cudaMalloc(&d_candidate_buffer, query_nb * MAX_SEARCH_EF * sizeof(HeapNode));
            cudaMalloc(&d_top_candidate_buffer, query_nb * MAX_SEARCH_EF * sizeof(HeapNode));

            cudaMemset(d_hops, 0, query_nb * sizeof(int));
            cudaMemset(d_dist_comps, 0, query_nb * sizeof(int));

            int queries_per_block = MAX_QUERIES_PER_BLOCK;
            int threads_per_query = THREADS_PER_QUERY;
            int threads_per_block = threads_per_query * queries_per_block;
            int num_blocks = (query_nb + queries_per_block - 1) / queries_per_block;

            size_t vram_free = 0, vram_total = 0;
            cudaMemGetInfo(&vram_free, &vram_total);
            size_t vram_used_mb = (vram_total - vram_free) / (1024 * 1024);

            cudaFuncAttributes kernel_attrs;
            cudaFuncGetAttributes(&kernel_attrs, irange_search_kernel);
            size_t lmem_total = (size_t)kernel_attrs.localSizeBytes * threads_per_block * num_blocks;
            size_t peak_vram_mb = vram_used_mb + lmem_total / (1024 * 1024);

            uint8_t* temp_curV = new uint8_t[query_nb];
            for (int i = 0; i < query_nb; i++) temp_curV[i] = 1;
            cudaMemcpy(visited.d_curV, temp_curV, query_nb * sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMemset(visited.d_mass, 0, (size_t)query_nb * max_elements * sizeof(uint8_t));
            delete[] temp_curV;

            cudaMemset(gpu_index.d_results, -1, (size_t)query_nb * query_K * sizeof(int));

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            unsigned long long kernel_seed = std::chrono::system_clock::now().time_since_epoch().count();
            irange_search_kernel<<<num_blocks, threads_per_block>>>(
                gpu_index, visited, query_nb, ef, query_K, dim, suffix_idx, d_hops, d_dist_comps,
                index.size_links_per_layer_, kernel_seed,
                d_entry_ids, d_entry_dists, d_entry_counts
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            float searchtime = milliseconds / 1000.0f;

            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Search kernel error: %s\n", cudaGetErrorString(err));
                exit(1);
            }

            int* cpu_results    = new int[query_nb * query_K];
            int* cpu_hops       = new int[query_nb];
            int* cpu_dist_comps = new int[query_nb];

            cudaMemcpy(cpu_results,    gpu_index.d_results, query_nb * query_K * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_hops,       d_hops,              query_nb * sizeof(int),           cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_dist_comps, d_dist_comps,        query_nb * sizeof(int),           cudaMemcpyDeviceToHost);

            float recall10 = 0.0f, recall50 = 0.0f, recall100 = 0.0f;
            if (index.storage->groundtruth.count(suffix)) {
                auto &gt = index.storage->groundtruth[suffix];
                int tp10 = 0, tp50 = 0, tp100 = 0;
                for (int i = 0; i < query_nb; i++) {
                    int gt10  = std::min((int)gt[i].size(), 10);
                    int gt50  = std::min((int)gt[i].size(), 50);
                    int gt100 = std::min((int)gt[i].size(), 100);
                    auto gt_end = gt[i].end();
                    std::unordered_set<int> seen;
                    for (int k = 0; k < query_K; k++) {
                        int result_id = cpu_results[i * query_K + k];
                        if (result_id == -1 || !seen.insert(result_id).second) continue;
                        if (std::find(gt_end - gt10,  gt_end, result_id) != gt_end) tp10++;
                        if (std::find(gt_end - gt50,  gt_end, result_id) != gt_end) tp50++;
                        if (std::find(gt_end - gt100, gt_end, result_id) != gt_end) tp100++;
                    }
                }
                recall10  = (float)tp10  / query_nb / 10;
                recall50  = (float)tp50  / query_nb / 50;
                recall100 = (float)tp100 / query_nb / 100;
            }

            float qps = query_nb / searchtime;
            long long total_hops = 0, total_dist_comps = 0;
            for (int i = 0; i < query_nb; i++) {
                total_hops       += cpu_hops[i];
                total_dist_comps += cpu_dist_comps[i];
            }
            float avg_hops = (float)total_hops / query_nb;
            float avg_dco  = (float)total_dist_comps / query_nb;

            if (outfile.is_open()) {
                outfile << ef << "," << recall10 << "," << recall50 << "," << recall100 << ","
                        << qps << "," << avg_dco << "," << avg_hops
                        << "," << vram_used_mb << "," << peak_vram_mb << "\n";
                outfile.flush();
            }

            delete[] cpu_results;
            delete[] cpu_hops;
            delete[] cpu_dist_comps;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_hops);
            cudaFree(d_dist_comps);
            cudaFree(d_candidate_buffer);
            cudaFree(d_top_candidate_buffer);
            freeGPUVisitedArray(visited);
        }

        if (outfile.is_open()) {
            outfile.close();
            printf("✓ Saved results for suffix %d to %s\n", suffix, savepath.c_str());
        }

        cudaFree(d_entry_ids);
        cudaFree(d_entry_dists);
        cudaFree(d_entry_counts);
        d_entry_ids    = nullptr;
        d_entry_dists  = nullptr;
        d_entry_counts = nullptr;

        suffix_idx++;
    }

    if (gpu_index.d_data_memory)          cudaFree(gpu_index.d_data_memory);
    if (gpu_index.d_segment_tree.d_nodes) cudaFree(gpu_index.d_segment_tree.d_nodes);
    if (gpu_index.d_query_vectors)        cudaFree(gpu_index.d_query_vectors);
    if (gpu_index.d_query_range)          cudaFree(gpu_index.d_query_range);
    if (gpu_index.d_results)              cudaFree(gpu_index.d_results);

    printf("✓ GPU memory cleaned up\n");
}

int main(int argc, char **argv) {
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data_path")            paths["data_vector"]           = argv[i + 1];
        if (arg == "--query_path")           paths["query_vector"]          = argv[i + 1];
        if (arg == "--range_saveprefix")     paths["range_saveprefix"]      = argv[i + 1];
        if (arg == "--groundtruth_saveprefix") paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file")           paths["index"]                 = argv[i + 1];
        if (arg == "--result_saveprefix")    paths["result_saveprefix"]     = argv[i + 1];
        if (arg == "--M")                    M = std::stoi(argv[i + 1]);
    }

    if (argc != 15)
        throw Exception("please check input parameters");

    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    cout << "Loading queries..." << std::endl;
    storage.LoadQuery(paths["query_vector"]);
    cout << "Loading query ranges..." << std::endl;
    storage.LoadQueryRange(paths["range_saveprefix"]);
    cout << "Loading ground truth..." << std::endl;
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);
    cout << "Loading index..." << std::endl;
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);

    std::vector<int> SearchEF = {2000, 1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};

    cout << "\n================================================" << std::endl;
    cout << "Running root-only GPU search (no segment tree traversal)" << std::endl;
    cout << "Testing " << storage.query_range.size() << " suffixes" << std::endl;
    cout << "================================================\n" << std::endl;

    search_on_gpu(index, SearchEF, paths["result_saveprefix"]);

    cout << "\n================================================" << std::endl;
    cout << "All GPU searches complete!" << std::endl;
    cout << "================================================" << std::endl;

    return 0;
}
