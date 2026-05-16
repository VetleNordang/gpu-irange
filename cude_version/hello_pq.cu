// GPU-accelerated iRangeGraph Search with PQ compression
// Using CUDA for greedy graph traversal with all SearchEF values
// PQ distances replace L2 for distance computation

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <map>
#include <tuple>
#include <memory>
#include "iRG_search.h"
#include <curand_kernel.h>
#include "gpu_index.cuh"
#include "gpu_heap.cuh"
#include "gpu_visited.cuh"
#include "gpu_search_updated.cuh"  // Contains irange_search_kernel (normal)
#include "gpu_search_pq.cuh"       // Contains irange_search_kernel_pq (PQ version)
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>


const int query_K = 10;
int M_graph = 32;  // Example value, adjust as needed
int M_compression = 320;  // Example value, adjust as needed

#define THREADS_PER_QUERY_VAL 128
#define MAX_THREADS_PER_BLOCK 2000

using std::cout;
std::unordered_map<std::string, std::string> paths;


void init()
{
    paths["data_vector"] = "";
    paths["query_vector"] = "";
    paths["index"] = "";
    paths["result_saveprefix"] = "";
    paths["range_saveprefix"] = "";
    paths["groundtruth_saveprefix"] = "";
    paths["pq_model"] = "";
    paths["pq_codes"] = "";
}

// Load PQ model from Faiss
std::unique_ptr<faiss::ProductQuantizer> load_pq_model(const std::string& model_path) {
    printf("Loading PQ model from: %s\n", model_path.c_str());
    std::unique_ptr<faiss::ProductQuantizer> pq(faiss::read_ProductQuantizer(model_path.c_str()));
    if (!pq) {
        throw std::runtime_error("Failed to load PQ model from " + model_path);
    }
    printf("✓ PQ model loaded: d=%d, M=%d, nbits=%d, ksub=%d, dsub=%d, code_size=%d\n",
           pq->d, pq->M, pq->nbits, pq->ksub, pq->dsub, pq->code_size);
    return pq;
}

// Load PQ codes blob (using the definition from iRG_search.h)
PQCodesBlob load_pq_codes(const std::string& codes_path) {
    printf("Loading PQ codes from: %s\n", codes_path.c_str());
    std::ifstream in(codes_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open " + codes_path);
    }
    
    PQCodesBlob blob;
    in.read(reinterpret_cast<char*>(&blob.n), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.d), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.M), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.nbits), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.code_size), sizeof(int));
    
    if (!in || blob.n <= 0 || blob.d <= 0 || blob.M <= 0) {
        throw std::runtime_error("Invalid PQ code header in " + codes_path);
    }
    
    const size_t payload_size = (size_t)blob.n * blob.code_size;
    blob.codes.resize(payload_size);
    in.read(reinterpret_cast<char*>(blob.codes.data()), payload_size);
    
    if (!in) {
        throw std::runtime_error("Failed to read PQ codes from " + codes_path);
    }
    
    printf("✓ PQ codes loaded: n=%d, d=%d, M=%d, code_size=%d\n", 
           blob.n, blob.d, blob.M, blob.code_size);
    return blob;
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

    // Set metadata
    gpu_index.d_dim = dimension;
    gpu_index.d_size_data_per_element = index.size_data_per_element_;
    gpu_index.d_size_links_per_element = index.size_links_per_element_;
    gpu_index.d_offsetData = index.offsetData_;
    
    // Allocate GPU memory for entire index structure
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_data_memory, total_index_memory);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy entire index structure from CPU to GPU
    err = cudaMemcpy(gpu_index.d_data_memory, index.data_memory_, total_index_memory, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_data_memory);
        return;
    }
}

void load_index_compact_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index,
                               const std::string& index_path) {
    int data_points = index.max_elements_;
    size_t size_links_per_element = index.size_links_per_element_;
    size_t size_data_per_element = index.size_data_per_element_;
    size_t total_compact_memory = (size_t)data_points * size_links_per_element;
    
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_data_memory, total_compact_memory);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc adjacency lists: %s\n", cudaGetErrorString(err));
        return;
    }

    gpu_index.d_adjacency_lists = gpu_index.d_data_memory;
    gpu_index.d_adjacency_lists_size = total_compact_memory;

    char* host_buffer = new char[total_compact_memory];
    if (!host_buffer) {
        printf("ERROR: failed to allocate host buffer\n");
        cudaFree(gpu_index.d_data_memory);
        return;
    }

    for (int i = 0; i < data_points; i++) {
        char* src = (char*)index.data_memory_ + i * size_data_per_element;
        char* dst = host_buffer + i * size_links_per_element;
        memcpy(dst, src, size_links_per_element);
    }

    err = cudaMemcpy(gpu_index.d_data_memory, host_buffer, total_compact_memory, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMemcpy adjacency lists: %s\n", cudaGetErrorString(err));
        delete[] host_buffer;
        cudaFree(gpu_index.d_data_memory);
        return;
    }

    delete[] host_buffer;

    gpu_index.d_dim = index.storage->Dim;
    gpu_index.d_size_data_per_element = size_links_per_element;
    gpu_index.d_size_links_per_element = index.size_links_per_element_;
    gpu_index.d_offsetData = 0;
}

void load_segment_tree_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    iRangeGraph::SegmentTree *tree = index.tree;
    std::vector<GPUNode> gpu_nodes = tree->FlattenGPUTree();
    
    size_t mem_to_allocate_to_gpu = gpu_nodes.size() * sizeof(GPUNode);

    // Initialize segment tree structure
    gpu_index.d_segment_tree.num_nodes = gpu_nodes.size();
    gpu_index.d_segment_tree.max_depth = tree->max_depth;
    gpu_index.d_segment_tree.root = gpu_nodes[0];
    
    // Allocate GPU memory for nodes
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_segment_tree.d_nodes, mem_to_allocate_to_gpu);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for segment tree: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy nodes to GPU
    err = cudaMemcpy(gpu_index.d_segment_tree.d_nodes, gpu_nodes.data(), mem_to_allocate_to_gpu, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for segment tree: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_segment_tree.d_nodes);
        return;
    }
    
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
        return;
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
        return;
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
        return;
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
        return;
    }
    
    delete[] flatten_queries;
    delete[] flatten_ranges;
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

void load_pq_model_to_gpu(GPUIndex &gpu_index, faiss::ProductQuantizer* pq_model) {
    gpu_index.pq_dsub = pq_model->dsub;
    gpu_index.pq_ksub = pq_model->ksub;
    gpu_index.pq_M = pq_model->M;
    gpu_index.pq_nbits = pq_model->nbits;
    gpu_index.pq_code_size = pq_model->code_size;
    gpu_index.use_pq = true;

    size_t centroids_size = (size_t)pq_model->M * pq_model->ksub * pq_model->dsub * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_centroids, centroids_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for PQ centroids: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(gpu_index.d_centroids, pq_model->centroids.data(), centroids_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for PQ centroids: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_centroids);
        return;
    }
    
}

void load_pq_codes_to_gpu(GPUIndex &gpu_index, const std::vector<uint8_t>& pq_codes,
                          int n_vectors, int M, int nbits, int code_size, int ksub, int dsub) {
    gpu_index.pq_codes_cpu = pq_codes;
    gpu_index.gpu_codes_size = (size_t)pq_codes.size() * sizeof(uint8_t);

    cudaError_t err = cudaMalloc((void**)&gpu_index.d_compressed_codes, gpu_index.gpu_codes_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc PQ codes: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(gpu_index.d_compressed_codes, pq_codes.data(), gpu_index.gpu_codes_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMemcpy PQ codes: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_compressed_codes);
        return;
    }
}

void write_results_to_csv(const std::string& saveprefix, int suffix, 
                          const std::vector<std::tuple<int, float, float, float, float>>& results) {
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
    } else {
        printf("ERROR: failed to open %s\n", savepath.c_str());
    }
}

void search_on_gpu(iRangeGraph::iRangeGraph_Search<float> &index, std::vector<int> SearchEF, std::string saveprefix, 
                   faiss::ProductQuantizer* pq_model, PQCodesBlob& pq_blob) {
    // Validate all SearchEF values are within supported range
    for (int ef : SearchEF) {
        if (ef > MAX_SEARCH_EF) {
            printf("ERROR: SearchEF=%d exceeds MAX_SEARCH_EF=%d\n", ef, MAX_SEARCH_EF);
            printf("Please increase MAX_SEARCH_EF in gpu_search_updated.cuh or reduce SearchEF\n");
            return;
        }
    }

    GPUIndex gpu_index;

    

    load_index_compact_to_gpu(index, gpu_index, paths["index"]);
   
    // Load segment tree to GPU
    load_segment_tree_to_gpu(index, gpu_index);
    load_queries_to_gpu(index, gpu_index);
    make_result_buffer_on_gpu(index, gpu_index);

    load_pq_model_to_gpu(gpu_index, pq_model);
    load_pq_codes_to_gpu(gpu_index, pq_blob.codes, pq_blob.n, pq_blob.M, pq_blob.nbits, pq_blob.code_size, pq_model->ksub, pq_model->dsub);
    
    std::vector<std::tuple<int, float, float, float, float>> current_suffix_results;
    
    // Allocate ADC distance table buffer: one flat table per query.
    // Each table is pq_M * pq_ksub floats. Allocated once, reused across all ef values and suffixes.
    {
        int qnb = index.storage->query_nb;
        size_t table_floats = (size_t)gpu_index.pq_M * gpu_index.pq_ksub;
        size_t table_bytes  = (size_t)qnb * table_floats * sizeof(float);
        cudaError_t err = cudaMalloc(&gpu_index.d_dist_tables, table_bytes);
        if (err != cudaSuccess) {
            printf("ERROR: Failed to allocate ADC distance tables: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    // Iterate over all suffixes in storage->query_range (same as CPU version)
    size_t suffix_idx = 0;
    for (auto range : index.storage->query_range) {
        int suffix = range.first;
        current_suffix_results.clear();

        for (int ef : SearchEF) {
            int query_nb = index.storage->query_nb;
            int max_elements = index.max_elements_;
            int dim = index.storage->Dim;
            
            GPUVisitedArray visited;
            cudaError_t err = initGPUVisitedArray(visited, query_nb, max_elements);
            if (err != cudaSuccess) {
                printf("Failed to initialize visited array: %s\n", cudaGetErrorString(err));
                return;
            }
            
            // Allocate metrics arrays on GPU
            int* d_hops;
            int* d_dist_comps;
            cudaMalloc(&d_hops, query_nb * sizeof(int));
            cudaMalloc(&d_dist_comps, query_nb * sizeof(int));
            
            // Initialize to zero (CRITICAL - otherwise contains garbage values!)
            cudaMemset(d_hops, 0, query_nb * sizeof(int));
            cudaMemset(d_dist_comps, 0, query_nb * sizeof(int));
            
            int threads_per_block = THREADS_PER_QUERY_VAL;   // 128: 1 query per block
            int threads_per_query = THREADS_PER_QUERY_VAL;   // 128 threads collaborate on each query
            int queries_per_block = threads_per_block / threads_per_query;  // = 1
            int num_blocks = (query_nb + queries_per_block - 1) / queries_per_block;
            

            
            // Reset visited counters for new search
            unsigned short* temp_curV = new unsigned short[query_nb];
            for (int i = 0; i < query_nb; i++) temp_curV[i] = 1;
            cudaMemcpy(visited.d_curV, temp_curV, query_nb * sizeof(unsigned short), cudaMemcpyHostToDevice);
            cudaMemset(visited.d_mass, 0, (size_t)query_nb * max_elements * sizeof(unsigned short));
            delete[] temp_curV;
            
            // CRITICAL: Clear output buffers before each SearchEF iteration
            cudaMemset(d_hops, 0, query_nb * sizeof(int));
            cudaMemset(d_dist_comps, 0, query_nb * sizeof(int));
            
            // Launch appropriate kernel (PQ or normal) - no branching inside kernel
            unsigned long long kernel_seed = std::chrono::system_clock::now().time_since_epoch().count();
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            irange_search_kernel_pq<<<num_blocks, threads_per_block>>>(
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
                continue;
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

            printf("    suffix %d  ef=%d  recall=%.4f  QPS=%.1f  DCO=%.1f  HOP=%.1f  time=%.3fs\n",
                   suffix, ef, recall, qps, avg_dco, avg_hops, searchtime);
            
            // Store results for this suffix
            current_suffix_results.push_back(std::make_tuple(ef, recall, qps, avg_dco, avg_hops));
            
            delete[] cpu_results;
            delete[] cpu_hops;
            delete[] cpu_dist_comps;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            
            // Clean up
            cudaFree(d_hops);
            cudaFree(d_dist_comps);
            freeGPUVisitedArray(visited);
            
        }

        // Write results to CSV file for this suffix
        write_results_to_csv(saveprefix, suffix, current_suffix_results);

        suffix_idx++;  // Increment for next suffix

    }
    
    
    // Free ADC distance table buffer
    if (gpu_index.d_dist_tables) {
        cudaFree(gpu_index.d_dist_tables);
        gpu_index.d_dist_tables = nullptr;
    }

    if (gpu_index.d_data_memory) cudaFree(gpu_index.d_data_memory);
    if (gpu_index.d_adjacency_lists) cudaFree(gpu_index.d_adjacency_lists);
    if (gpu_index.d_segment_tree.d_nodes) cudaFree(gpu_index.d_segment_tree.d_nodes);
    if (gpu_index.d_query_vectors) cudaFree(gpu_index.d_query_vectors);
    if (gpu_index.d_query_range) cudaFree(gpu_index.d_query_range);
    if (gpu_index.d_results) cudaFree(gpu_index.d_results);
    if (gpu_index.d_compressed_codes) cudaFree(gpu_index.d_compressed_codes);
    if (gpu_index.d_centroids) cudaFree(gpu_index.d_centroids);
}



int main(int argc, char **argv) {

    setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout: live progress in SLURM logs

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path_comp")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--pq_model_out")
            paths["pq_model"] = argv[i + 1];
        if (arg == "--pq_codes_out")
            paths["pq_codes"] = argv[i + 1];
        if (arg == "--M_compression_spaces")
            M_compression = std::stoi(argv[i + 1]);
        if (arg == "--graph_M")
            M_graph = std::stoi(argv[i + 1]);  // Can also override with graph_M
    }

    if (argc < 15) {
        std::cerr << "Usage: " << argv[0] << std::endl
                  << "  --data_path_comp <path>" << std::endl
                  << "  --query_path <path>" << std::endl
                  << "  --index_path <path>" << std::endl
                  << "  --result_saveprefix <path>" << std::endl
                  << "  --range_saveprefix <path>" << std::endl
                  << "  --groundtruth_saveprefix <path>" << std::endl
                  << "  --pq_model_out <path>" << std::endl
                  << "  --pq_codes_out <path>" << std::endl
                  << "  --M_compression_spaces <int>" << std::endl
                  << "  [--graph_M <int>] (default: 32 - edges per node in graph)" << std::endl;
        return 1;
    }
    
    if (paths["data_vector"].empty() || paths["query_vector"].empty() || paths["index"].empty() ||
        paths["result_saveprefix"].empty() || paths["range_saveprefix"].empty() || 
        paths["groundtruth_saveprefix"].empty() || paths["pq_model"].empty() || paths["pq_codes"].empty()) {
        std::cerr << "Error: All path arguments are required" << std::endl;
        return 1;
    }
    

    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M_graph);

    std::vector<int> SearchEF_values = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};

    std::unique_ptr<faiss::ProductQuantizer> pq_model_for_flag = load_pq_model(paths["pq_model"]);
    PQCodesBlob pq_blob_for_flag = load_pq_codes(paths["pq_codes"]);
    index.load_pq_codes(pq_blob_for_flag.codes, pq_blob_for_flag.code_size,
                        pq_blob_for_flag.M, pq_blob_for_flag.nbits, pq_model_for_flag.get());

    search_on_gpu(index, SearchEF_values, paths["result_saveprefix"], pq_model_for_flag.get(), pq_blob_for_flag);

    return 0;
}
